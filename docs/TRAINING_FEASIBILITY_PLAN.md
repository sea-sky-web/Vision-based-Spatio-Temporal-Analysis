# 训练可行性评估与改造路线图

在《训练困难原因与 MVDet 训练实现的差异》中已经总结了当前流水线与 MVDet 之间的主要缺口。基于这些发现，下面从**整体结构**与**落地难度**角度，提出一套分阶段的改造方案，确保训练可以可靠收敛。

## 1. 现状速览

| 模块 | 当前状态 | 训练阻碍 |
| --- | --- | --- |
| 数据读取/预处理 | 仅执行 `Resize + ToTensor`，标注仅填充 `centers_world`【F:project/data/transforms.py†L4-L8】【F:project/data/wildtrack_loader.py†L348-L355】 | 缺少颜色归一化、增强；`boxes_world` 恒为空导致评估失真【F:project/train.py†L40-L66】 |
| 模型架构 | `BEVNet` 只有热力图分支，MSE 监督且固定 σ | 梯度稀疏，无法学习亚像素偏移与尺寸【F:project/models/model_wrapper.py†L82-L119】 |
| 检测解码 | 固定 0.6m×0.6m 框，中心落在栅格中心【F:project/models/heads/detector.py†L36-L85】 | 无法拟合真实人体 footprint，指标受限 |
| 训练循环 | 指标依赖空的 `boxes_world`，无 hard-negative mining | 缺少有效反馈，难以判断训练质量【F:project/train.py†L200-L236】 |

## 2. 改造优先级

按照“**先打通监督—再提升表达—最后优化细节**”的原则，将工作拆为三个里程碑。每个阶段尽量保持改动可验证，避免一次性大改。

### 阶段 A：补全监督与评估（工作量：中）
1. **目标热力图与损失**
   - 将 `BEVDetector.forward` 返回 logits，并在 `BEVNet.loss` 内改用带 alpha/gamma 的 focal loss（已有实现可复用）或 `BCEWithLogitsLoss`，避免 Sigmoid 后再做 MSE。【F:project/models/model_wrapper.py†L12-L18】【F:project/models/heads/detector.py†L24-L27】
   - 允许根据目标尺寸动态调节 σ：可参考 MVDet 根据边界框大小计算半径，并把 `sigma_px` 写入配置以便调参。【F:project/models/model_wrapper.py†L91-L119】

2. **多分支检测头**
   - 新增 offset、尺寸分支：在 `BEVDetector` 中追加并行卷积分支输出 2-channel offset 与 2-channel wh，训练时采用 L1/L2 损失仿照 MVDet。
   - `decode` 逻辑利用学习到的 offset/size，替换固定 `box_size_m`。【F:project/models/heads/detector.py†L36-L85】

3. **评估指标修复**
   - 在 `WildtrackDataset._prepare_targets` 中同步填充 `boxes_world`（可根据 centers 添加固定尺寸或读取原始标注），确保验证时 `compute_metrics` 拿到真实 GT。【F:project/data/wildtrack_loader.py†L348-L355】【F:project/train.py†L40-L66】
   - 或者调整 `compute_metrics` 改为读取 `centers_world`，保持与新的 offset/size 分支一致。

> 交付标准：热力图 loss 不再梯度消失，验证指标能随 epoch 波动，至少能在小 batch 上看到 P/R 改善。

### 阶段 B：增强特征与几何一致性（工作量：中-高）
1. **数据增强管线**
   - 在 `build_transforms` 中加入 `ColorJitter`、`RandomAffine`、`Normalize(mean,std)`，并使用与 backbone 预训练相匹配的均值方差，降低 domain gap。【F:project/data/transforms.py†L4-L8】
   - 可引入 mixup/cutmix 的多视角变体以增强鲁棒性，注意保持几何关系。

2. **多尺度特征**
   - 若采用 `ConcatFusion`，随着视角数增加通道数迅速膨胀。现在已提供 `AttentionFusion`（跨视角多头注意力池化）可直接通过 `MODEL.FUSION.TYPE=attention` 启用，以控制通道尺寸并自适应分配视角权重；仍可结合 1×1 conv 做进一步压缩。【F:project/models/model_wrapper.py†L35-L75】

3. **几何一致性正则**
   - 激活 `_geom_consistency_loss`，在 `loss` 中按权重加入，可缓解标定噪声造成的 BEV 偏移。【F:project/models/model_wrapper.py†L121-L151】

> 交付标准：模型在遮挡/光照变化下稳定，验证指标相比阶段 A 有持续提升。

### 阶段 C：训练流程与调参工具（工作量：低-中）
1. **学习率与批量策略**
   - 参考 MVDet 使用 warmup + cosine schedule（`CosineAnnealingWarmRestarts`），配合梯度裁剪，缓解初始不稳定。【F:project/train.py†L32-L37】【F:project/train.py†L171-L197】
   - 若显存允许，启用 `gradient accumulation` 或 `DistributedDataParallel` 提升有效 batch size。

2. **Hard Negative Mining & Label Smoothing**
   - 为 BEV 网格中未标注区域引入 online hard negative mining，或在 focal loss 中调整 α/γ，减少假阳性影响。

3. **可视化与调试**
   - 扩展 `save_bev_heatmap` 输出 offset/size 可视化，结合 TensorBoard 监控 loss、学习率、梯度范数，保证调参效率。【F:project/utils/visualization.py†L1-L29】

> 交付标准：训练脚本支持自动日志/可视化，关键超参可观测。

## 3. 难度评估
- **阶段 A** 涉及 loss 与输出分支重构，代码量适中，但需确保数据与评估同步更新。预估 2~3 人日。
- **阶段 B** 需要深入理解几何与多视角特征融合，可能需要调试 kornia/grid_sample 的梯度稳定性。预估 3~5 人日。
- **阶段 C** 多为工程化提升，可在前两阶段之后穿插进行。预估 1~2 人日。

## 4. 验证路径
1. 单独在极小样本（如 10 帧）上过拟合，验证新 loss 能快速拟合 GT 热力图。
2. 使用与 MVDet 相同的指标脚本，比较阶段 A/B/C 的增益。
3. 最终在 Wildtrack 全量数据上复现或接近 MVDet 公开的精度基线。

---

通过以上分阶段改造，现有 BEV-PedTrack 结构可以逐步对齐 MVDet 的训练策略，同时控制开发风险，最终实现稳定可复现的训练流程。
