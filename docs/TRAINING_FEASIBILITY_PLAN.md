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

## 5. 在 BEV 特征中注入矢量交通语义层的难度/收益评估

### 5.1 背景与潜在收益
- 近期 BEV 感知研究（如 VectorMapNet、MapTR、BEVFusion 等）都会把 HD Map 中的中心线、车道边界、静态障碍等矢量化语义投影到统一的 BEV 栅格，再与视觉特征拼接或通过注意力交互，以增强定位与运动推理能力。其主要收益在于：
  1. **几何参照** —— 车辆/行人的检测结果可以对齐到真实路网，减少 BEV 中的漂移误差。
  2. **语义先验** —— 交通参与者更可能出现在车道或人行道附近，语义层可在热力图损失中提供 soft prior，缓解负样本稀疏问题。
  3. **时空一致性** —— 含有行驶方向的向量场有助于后续的速度估计或轨迹预测，与 README 中“行人感知+轨迹跟踪”的长期目标一致。

### 5.2 与当前项目的耦合点
- **BEV 特征注入位置清晰**：`BEVNet` 目前在融合多视角后会拼接 `pos_enc` 再送入 `BEVDetector`【F:project/models/model_wrapper.py†L78-L95】。语义层可以作为额外通道与 `pos_enc` 一样被 `torch.cat`，无需更改检测头 API。
- **数据加载仍缺少语义源**：`WildtrackDataset` 只读取行人世界坐标与固定尺寸，并未包含道路/车道信息【F:project/data/wildtrack_loader.py†L303-L364】。要注入矢量语义，需要额外的 HD Map 或手工标注，并在数据管线中实现矢量→栅格或矢量→多边形的 rasterization。
- **几何转换模块可复用**：现有 `GeometryTransformer` 已处理多视角到 BEV 的投影，若语义数据本身就在世界坐标系（米），可直接 rasterize 到 `MODEL.BEV_SIZE` 对应的 `bev_bounds`，与现有 `self.res_x/self.res_y` 完全一致【F:project/models/model_wrapper.py†L37-L55】。

### 5.3 实现难度分解
| 子任务 | 主要工作 | 难度 | 备注 |
| --- | --- | --- | --- |
| 获取矢量交通语义 | 收集 Wildtrack 场景的 CAD/HD Map 或人工标记人行道、车道中心线 | 高 | 若无公开地图，需要逆向工程或手标，成本最高 |
| 矢量→BEV 栅格化 | 将多段折线、多边形 rasterize 至 `bev_h×bev_w`，生成 multi-channel 语义张量（如车道 mask、方向向量、可通行区域） | 中 | 可基于 `torchvision.ops.roi_align`、`shapely` 或自定义扫描线实现；需要确保与 `bev_bounds` 对齐 |
| 模型改动 | 扩展 `BEVNet.forward` 在 `torch.cat` 语句前插入语义通道，或通过 1×1 conv 投影与视觉特征对齐 | 低 | 不影响 `BEVDetector`，参数量有限 |
| 训练与损失设计 | 设计联合训练策略（如在热力图损失里引入语义权重，或为语义层添加辅助重建 loss） | 中 | 需实验证明语义先验不会掩盖真实观测 |

### 5.4 收益与性价比结论
- **收益**：在 BEV 中加入交通语义层主要提升定位稳定性与下游轨迹推理的可行性，尤其在 Wildtrack 多视角遮挡严重的区域，语义先验可以减少伪检测；未来若扩展到车辆/非机动车类任务，语义层也是共享资产。
- **难度**：技术实现本身（栅格化+通道拼接）较轻量，但“数据来源”是最大瓶颈。Wildtrack 官方未提供 HD Map，缺乏矢量语义会让该特性只能依赖额外采集或人工标注，显著提升准备成本。
- **性价比**：若团队能够获得或构建可靠的矢量地图，则增加 1~2 周的开发即可完成注入和训练，预计能带来更稳定的 BEV 热力图；若无法获得地图，则需要投入更长的标注周期，收益被拉低。建议先在小范围（如部分车道/人行道）手工制作语义层并进行 ablation，验证增益后再扩展到全量场景。

---

## 6. 面向预测阶段的语义叠加策略评估

> 目标：**不额外训练语义模块、零人工标注**，仅在推理阶段将交通语义层与 `BEVNet` 输出对齐，以便为轨迹预测提供物理可行域约束。

### 6.1 候选思路概述

| 方案 | 数据来源 & 处理 | 与现有流水线的耦合 | 优势 | 局限 |
| --- | --- | --- | --- | --- |
| **多视角语义分割 → BEV warp** | 采用公开预训练的城市级分割模型（Mask2Former / Segment Anything + 分类器等）对每个摄像头帧输出道路/人行道 mask，再用 `GeometryTransformer` 将二值 mask warp 到 BEV 并在视角维度做 `mean/max` 聚合 | 直接放置在 `BEVNet.forward` 中 `fusion` 输出与 `pos_enc` 拼接的步骤之前即可。复用现有 `Ks/Rts`、网格分辨率与 `torch.cat` 接口，无需改变损失或检测头 | - 仅依赖视觉输入即可实时生成语义先验<br>- 可随帧更新，支持动态场景（例如临时施工遮挡）<br>- Warp 逻辑沿用主干网络，空间对齐最可靠 | - 推理时需要额外的 2D 分割算力；多视角推理成本成倍增加<br>- 预训练模型类别与 Wildtrack 场景的贴合度未知，可能输出噪声 |
| **静态航拍/卫星地图 → 单次对齐** | 寻找该场景的俯视图或 GIS 矢量地图，利用少量地面控制点估计从地图到 `bev_bounds` 的仿射/透视变换，提前 rasterize 出车道/人行道 mask 并以缓存形式加载 | 可在推理启动时读取缓存张量，并通过 `cfg.RUNTIME.semantic_path` 注入到模型 `forward`；由于静态，和 `pos_enc` 一样只需 `torch.cat` 一次 | - 仅在初始化阶段做一次对齐，无运行时开销<br>- 地图通常噪声小、类别清晰，适合做硬约束（mask gating） | - Wildtrack 未公开航拍图，需要额外数据源<br>- 一旦现场布局变化（移除护栏等）需重新对齐，缺乏实时性 |
| **矢量地图生成模型（VectorMapNet/MapTR 等）迁移** | 利用开源“图像→矢量地图”模型对多视角或拼接图像推理，得到 lane centerline/边界折线；再在 BEV 网格中 rasterize 成语义通道 | 需在推理流程中插入一次性矢量推理 + rasterize 步骤，可离线运行并缓存结果，与静态地图注入方式相同 | - 输出可直接作为拓扑图（便于轨迹约束）<br>- 不依赖场景航拍图，仅用现有相机帧 | - 模型大多在车辆数据集上训练，与行人场景 domain gap 大<br>- 推理一次性成本高，需要额外依赖栅格化库 |

### 6.2 哪种方案更适合当前项目？

| 考量 | 多视角分割 warp | 静态地图对齐 | 矢量生成模型 |
| --- | --- | --- | --- |
| **与 Wildtrack 数据的契合度** | ✅ 复用现有标定，完全依赖已有图像数据 | ⚠️ 需要额外航拍 / GIS 数据，当前仓库未提供 | ⚠️ 预训练模型多基于车辆场景，domain gap 较大 |
| **实现复杂度** | 中：需要为每个视角增加分割推理 + warp；代码可重用 `GeometryTransformer` | 低：一次性计算仿射矩阵并缓存语义栅格，即可在推理阶段直接 `torch.cat` | 高：需下载/部署大型矢量模型，并处理折线→栅格转换 |
| **运行时开销** | 高：推理阶段每帧都要执行多次分割，延迟增加 | 极低：语义层是静态通道，无额外推理成本 | 低-中：若离线生成语义层，则运行时和静态地图类似 |
| **对物理约束的帮助** | 中：语义先验来自实时视觉，可能包含噪声，但能反映临时障碍 | 高：地图提供清晰的可通行区域，可直接在轨迹预测中做 mask gating 或 cost map | 高：矢量拓扑可直接驱动图搜索/路径规划 |

**综合建议**：

1. **短期（验证阶段）** —— 首选「多视角语义分割 → BEV warp」，因为它完全依赖现有摄像头数据和标定，即可立刻在推理阶段生成语义通道并 `torch.cat` 到 `bev_feat`。虽然推理成本上升，但能快速验证语义先验是否提升行人检测 / 轨迹稳定性。
2. **中期（产品化阶段）** —— 若能获取稳定的场地航拍或 CAD 图，建议追加「静态地图对齐」：它几乎零运行成本，并能为轨迹预测提供更强的硬约束（如直接将不可通行区域设为 `-inf` 代价）。可将其与实时分割结果融合（例如 `max` 或 `weighted sum`），同时获得长期结构与短期动态信息。
3. **长期（拓展到复杂交通场景）** —— 当需要丰富的拓扑结构（交叉口连接关系、可行驶方向）时，再考虑引入矢量地图生成模型，将其输出转为图结构供轨迹规划使用。该方案实现与维护成本高，应建立在前两种策略已验证收益的基础上。

### 6.3 对轨迹预测的直接影响
- **代价地图约束**：任意方案生成的语义层都可以在后续轨迹预测器（如卡尔曼滤波或多假设追踪）中充当 cost map，直接屏蔽不可行区域，从而提升轨迹的物理可行性。
- **方向先验**：若语义层包含方向信息（如道路中心线的切向量），可将其注入速度预测模块，将状态转移矩阵沿切线方向约束，降低无物理意义的急转弯。
- **候选路径裁剪**：静态地图或矢量拓扑能够提前生成有限数量的路径簇，推理阶段只需在这些簇内匹配观测，显著减小数据关联复杂度。

综上，针对“仅在推理阶段叠加语义层”这一目标，建议先以多视角语义分割 warp 作为快速验证路径，再辅以静态地图对齐以获得稳定、低成本的物理约束；矢量生成模型可作为长远规划选项。
