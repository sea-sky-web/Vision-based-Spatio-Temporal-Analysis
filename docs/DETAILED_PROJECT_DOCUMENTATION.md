# Vision-based Spatio-Temporal Analysis — 全量方法级文档

本档案对仓库中 **每一个 Python 文件** 进行逐项说明，覆盖所有函数、类、内部辅助方法以及它们之间的依赖关系。所有描述均基于当前代码实现，包含：

- 输入 / 输出张量的形状与含义
- 使用到的外部库、配置字段、文件路径
- 异常分支、默认值、潜在副作用
- 与其他函数/类的协同方式

---

## `project/train.py`
### 模块概览
训练脚本的主入口，负责解析配置、构建数据与模型、训练/验证循环、指标评估以及检查点保存。依赖 `torch`、`torchvision` 数据工具、`WildtrackDataset`、`BEVNet` 等组件。

### `load_cfg(path: str) -> dict`
- **职责**：以 UTF-8 编码读取 YAML 配置文件，调用 `yaml.safe_load` 生成 Python 字典。
- **输入**：`path` —— 配置文件绝对或相对路径。
- **输出**：配置字典；若文件缺失会抛出 `FileNotFoundError`。
- **副作用**：无。

### `build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer`
- **职责**：依据配置中 `TRAIN.OPT`, `TRAIN.LR`, `TRAIN.WEIGHT_DECAY` 构造优化器。
- **逻辑**：当 `OPT` 忽略大小写后等于 `'adamw'` 时实例化 `optim.AdamW`，否则回退到 `optim.Adam`。
- **输出**：绑定 `model.parameters()` 的优化器实例。
- **注意**：需要保证配置中字段存在且可转换为浮点；否则触发 `KeyError` 或 `ValueError`。

### `build_scheduler(optimizer, cfg) -> torch.optim.lr_scheduler._LRScheduler`
- **职责**：根据 `TRAIN.LR_SCHEDULER` 选择学习率调度器。
- **分支**：
  - `'step'` —— 生成 `StepLR(step_size=10, gamma=0.5)`。
  - 其他任何值 —— 默认 `CosineAnnealingLR`，`T_max` 取训练总轮数 `TRAIN.EPOCHS`。
- **依赖**：需在循环外部每轮调用 `scheduler.step()`。

### `compute_metrics(pred_boxes: List[Tensor], gt_targets: List[Dict], match_dist: float = 0.5)`
- **职责**：离线评估预测框与 GT 的对齐情况，输出精确率/召回率/F1 与定位误差。
- **输入约定**：
  - `pred_boxes[b]` —— `Tensor[K,4]`，前两维为世界坐标中心 (米)。若无预测则允许 `None`。
  - `gt_targets[b]` —— 字典，需包含 `boxes_world: Tensor[N,4]` 或为空张量。
- **流程**：
  1. 逐 batch 遍历 GT，创建 `used` 掩码防止重复匹配。
  2. 对每个预测框计算与所有 GT 中心的欧氏距离，贪心选择最近且未匹配的 GT。
  3. 距离 ≤ `match_dist` 记为 TP，否则 FP；未匹配 GT 计入 FN。
  4. 累积距离作为定位误差并求平均。
- **输出**：`precision, recall, f1, mle` 均为 `float`。
- **边界**：GT 为空时避免除零（使用 `max(1, count)`）；否则指标将退化为 0。

### `_summarize_batch_gt(targets: List[Dict]) -> str`
- **定义位置**：`main()` 内部闭包。
- **职责**：统计 batch 中每个样本的目标数量（`boxes_world` 或 `centers_world`）并返回格式化字符串。
- **用途**：首个 batch 打印数据量摘要，辅助调试。

### `_summarize_calib(calib: Dict) -> str`
- **定义位置**：`main()` 内部闭包。
- **职责**：读取首个 batch 的标定矩阵，估算每个外参的旋转角与平移模长；异常情况输出 `nan`。
- **用途**：帮助确认标定是否合理。

### `main()`
- **职责**：完整训练流程。
- **关键步骤**：
  1. 解析 `--config`（必需）和 `--save_vis`（控制验证阶段是否保存热力图）。
  2. 读取配置，选择设备。当使用 CUDA 时开启 `cudnn.benchmark` 并尝试 `torch.set_float32_matmul_precision('medium')`（旧 CUDA 版本可能失败）。
  3. 构建 `WildtrackDataset`，若帧数 ≥ 500 则固定 400/100 划分；否则随机 `80/20` 分割。
  4. 为训练/验证分别创建 `DataLoader`：
     - `collate_fn` 统一打包张量与列表标定。
     - 若使用 GPU，启用 `pin_memory`、`persistent_workers` 和 `prefetch_factor=2`。
  5. 初始化 `BEVNet` 并迁移至目标设备。
  6. 通过 `build_optimizer`、`build_scheduler` 创建优化组件；设置 `GradScaler` 以支持混合精度（默认启用，除非 `USE_AMP=False` 或无 CUDA）。
  7. 训练循环：
     - 每个 batch 将图像及内外参移动到设备。
     - 首个 batch 调用 `_summarize_batch_gt` / `_summarize_calib` 打印信息。
     - 根据 `use_amp` 决定是否在 `autocast(dtype=torch.float16)` 环境下前向和回传。
     - 使用 `model.loss` 计算 MSE 热力图损失，执行优化器 / scaler 更新。
     - 累积损失以便 epoch 结束时求均值。
  8. 验证循环（`torch.no_grad()` 模式）：
     - 同样迁移张量到设备并可选保存热力图（调用 `save_bev_heatmap`）。
     - 从 `model(batch)` 返回的 `boxes` 中提取预测，与 GT 通过 `compute_metrics` 计算指标。
  9. 输出每轮的平均指标，保存 `last.pth` 和当 F1 提升时的 `best.pth`（记录 epoch、模型参数和当前 F1）。
- **副作用**：
  - 在配置所在目录的 `../{SAVE_DIR}` 下写入检查点。
  - 控制台输出大量训练日志。
- **终止**：`if __name__ == '__main__': main()` 作为脚本入口。

---

## `project/inference.py`
### 模块概览
推理脚本入口，负责加载训练后的模型并对整套 Wildtrack 数据执行前向推理与结果导出。

> **提示**：代码中使用了 `argparse`、`yaml`，需确保运行脚本前已导入（当前文件缺少显式 `import argparse, yaml`，使用时应补全）。

### `load_cfg(path: str) -> dict`
- 与训练脚本的同名函数类似，但未指定编码（使用系统默认）。
- 返回值为配置字典。

### `main()`
- **职责**：批量推理与保存预测。
- **步骤**：
  1. 解析命令行参数：`--config`（必需）和 `--checkpoint`（默认 `checkpoints/best.pth`）。
  2. 读取配置，选择设备（优先配置指定 GPU，若无 CUDA 则回退 CPU）。
  3. 构造完整的 `WildtrackDataset` 与不打乱的 `DataLoader`；batch 大小取自 `DATA.BATCH_SIZE`。
  4. 初始化 `BEVNet`，加载权重：`torch.load(..., map_location=device)`，`strict=False` 允许缺失/额外键。
  5. 将模型移动到设备并设置为评估模式。
  6. 推理循环：
     - 将 batch 图像及标定张量移至设备。
     - 在 `torch.no_grad()` 环境调用 `model(batch)`。
     - 收集 `batch['meta']` 内的 `frame_idx`，调用 `save_predictions_json` 输出到 `RUNTIME.OUTPUT_DIR`。
  7. 循环结束后打印保存目录。
- **副作用**：创建/写入 JSON 文件，每个 batch 的每一帧对应一个 `frame_XXXXXX.json`。

---

## `project/data/wildtrack_loader.py`
### 模块概览
Wildtrack 数据集读取与标定解析。包含大量 XML 解析与几何投影辅助函数。依赖 `torch`, `torchvision`, `PIL.Image`, `xml.etree.ElementTree`, `json`, `math`, `re` 等库。

### 几何辅助函数（模块级）
- `_compute_homography(K, Rt)`
  - **输入**：`K` (`3×3` 内参)、`Rt` (`4×4` 或兼容的外参矩阵)。
  - **实现**：提取旋转的前两列与平移列构成 `3×3` 矩阵 `H`，再左乘 `K` 得到世界平面 → 图像平面的单应。
  - **用途**：供 `GeometryTransformer` / `_pixel_to_world` 计算投影。

- `_compute_img_to_world_homography(K, Rt)`
  - **功能**：对上面结果求逆；优先 `torch.linalg.inv`，若奇异则回退 `pinv`。
  - **输出**：像素 → 地面世界坐标的单应矩阵。

- `_pixel_to_world(u, v, K, Rt)`
  - **输入**：像素坐标 `(u, v)`、相机内外参。
  - **步骤**：调用 `_compute_img_to_world_homography`，乘以 `[u,v,1]^T`，再除以齐次分量获取 `(x,y)`。
  - **容错**：若齐次分量接近 0 或为 NaN，返回 `None`。
  - **用途**：当标注仅提供多视角 2D 框时估算行人世界坐标。

### XML 解析辅助函数
- `_parse_float_list(text: str) -> List[float]`
  - 清理逗号、分号、制表、换行，将可解析的数字转换为浮点列表，忽略非法 token。

- `_reshape(vals, rows, cols) -> torch.Tensor`
  - 验证元素数量足够后 reshape 为 `rows×cols`；不足抛出 `ValueError`。

- `_try_get_matrix(root, tag_names, shape)`
  - 遍历多个候选标签，支持 `<data>` 节点、内联文本或嵌套 OpenCV 格式。
  - 找到满足形状需求时返回 `torch.Tensor`，否则 `None`。
  - 在 `_load_camera_xml`、`_load_wildtrack_calibrations` 中大量复用。

### 标定文件发现与解析
- `_load_camera_xml(xml_path)`
  - **职责**：给定单个 XML，提取 `K` (3×3) 与 `Rt` (4×4)。
  - **流程**：
    1. 在多个别名标签中寻找矩阵。
    2. 若缺失 `Rt`，尝试组合 `R`+`T` 或 Rodrigues 表示。
    3. 若仍失败，则默认 `K` 为对角 1000，`Rt` 为单位矩阵。
  - **输出**：`(K, Rt44)`。

- `_discover_camera_xmls(calib_dir, views)`
  - **作用**：根据相机数目寻找匹配的 XML 路径。优先匹配文件名包含 `C{i}`，否则使用数字 `i`。
  - **输出**：长度为 `views` 的路径列表，不存在的位置为 `None`。

- `_load_wildtrack_calibrations(calib_root, views)`
  - **逻辑**：
    - 确定内参目录（优先 `intrinsic_original`，其后 `intrinsic_zero`），外参目录（优先 `extrinsic`）。
    - 为 7 视角使用 Wildtrack 官方命名，其余情况下根据文件名提取 `CVLab*`、`IDIAP*`，不足时自动补齐 `Cam{i}`。
    - 对每个相机：
      - 搜索对应内参/外参 XML；缺失时打印警告并使用默认矩阵。
      - 解析 Rodrigues 表示、`rvec/tvec` 等多种格式。
      - 若平移模长 > 100，则视作毫米并除以 1000。
      - 尝试打印旋转角（由 `trace` 反算）和位移模长。
  - **输出**：`Ks: List[Tensor(3,3)]`, `Rts: List[Tensor(4,4)]`。
  - **副作用**：打印解析日志及警告。

- `_rodrigues(rvec)`
  - **输入**：Rodrigues 向量，允许 `(3,)`、`(3,1)` 或 `(1,3)`。
  - **输出**：对应旋转矩阵。角度小于阈值时返回单位阵。
  - **用途**：在 `_load_wildtrack_calibrations` 中处理 `rvec` 表达。

### `WildtrackDataset`
- **构造函数 `__init__(self, cfg)`**
  - 解析配置字段：数据根目录、相机数量、图像尺寸 (`[C,H,W]`)。
  - 构建图像变换（`build_transforms`）。
  - 校验各相机子文件夹 `Image_subsets/C{i}` 是否存在，并将第一相机的文件名作为帧列表；若为空抛错。
  - 查找标定目录（`Calibration`/`Calibrations`/`calibration`），利用 `_load_wildtrack_calibrations` 获得全局内外参，并在整个数据集内复用（每帧视为静态标定）。
  - 推断标注目录（优先 `annotations_positions`）。
  - 初始化 `targets_per_frame`，调用 `_prepare_targets()` 缓存每帧 GT。

- `__len__(self) -> int`
  - 返回帧数量，等同于 `len(self.frame_files)`。

- `_prepare_targets(self)`
  - **职责**：预处理每帧标注为世界坐标中心。
  - **流程**：
    1. 遍历所有帧，若存在 JSON 标注则读取。
    2. 支持两种格式：
       - 字典形式：`annotations[*].world_pos`。
       - 列表形式：`views` 内含 2D 框，需通过 `_pixel_to_world` 投影估计。
    3. 对每个行人取多视角世界坐标的平均作为中心；失败时忽略该目标。
    4. 保存为 `{'boxes_world': zeros, 'centers_world': Tensor[N,2], 'keypoints': None, 'calib': {...}}`。
  - **异常处理**：捕获 JSON 解析异常并打印警告，仍继续处理后续帧。

- `__getitem__(self, idx) -> Dict`
  - **流程**：
    1. 对所有视角读取对应帧图像，使用 PIL 转 RGB 后应用 `transform`（`Resize`+`ToTensor`）。
    2. 堆叠为 `[V,3,H,W]`；标定直接引用预解析列表。
    3. 读取预缓存的 `targets`；若越界则创建空目标。
    4. 生成 `meta`：帧索引与各视角图像路径。
    5. 返回字典：`{'images', 'calib', 'targets', 'meta'}`。

- `collate_fn(batch)`（模块级函数）
  - 过滤 `None` 样本后，将所有 `images` 堆叠成 `[B,V,3,H,W]`。
  - 标定保持为列表结构（`List[List[Tensor]]`），便于后续转为张量。
  - 直接返回 `targets` 与 `meta` 列表，不额外复制。

### 其他
- 模块末尾保留 `_rodrigues` 的实现供外部直接调用。

---

## `project/data/transforms.py`
### 模块概览
定义统一的图像预处理流程。

### `build_transforms(img_size: Tuple[int, int] = (256, 256))`
- 返回 `torchvision.transforms.Compose`，顺序为 `Resize(img_size)` → `ToTensor()`。
- `img_size` 应与配置的 `[H,W]` 保持一致；若传入 `(H,W)` 则将图像缩放到对应分辨率。

---

## `project/models/encoders/base.py`
### 模块概览
多视角编码器的抽象基类。

### `class ViewEncoder(nn.Module, ABC)`
- **属性**：`out_channels` —— 期望的输出通道数。
- **`__init__(self, out_channels)`**：保存目标通道数。
- **`forward(self, images)`（抽象方法）**：约定输入支持 `[B,V,3,H,W]` 或 `[B*V,3,H,W]`，输出必须为 `[B,V,C,Hf,Wf]`。
- **`load_pretrained(self, weights_path)`**：尝试载入外部权重，`strict=False` 以兼容不同结构；失败时打印错误信息但不中断程序。
- **`freeze(self)`**：遍历所有参数，将 `requires_grad=False`，用于冻结特征提取器。

---

## `project/models/encoders/cnn_encoder.py`
### 模块概览
`ViewEncoder` 的具体实现，可选使用 `timm` 特征提取器或退化为轻量卷积堆栈。

### `class CNNEncoder(ViewEncoder)`
- **`__init__(self, out_channels=32, backbone='resnet18', pretrained=True, out_index=2)`**
  - 若安装了 `timm`：创建 `features_only=True` 的模型，`out_index` 决定取哪一层特征；首次前向时会根据真实通道数创建 `1×1` 卷积投影到 `out_channels`。
  - 若 `timm` 不可用或创建失败：构造两层卷积的简单编码器（`Conv2d → ReLU → Conv2d → ReLU`），第二层输出通道固定为 `out_channels`。
  - 记录 `_use_timm`、`_feature_channels` 用于延迟初始化。

- **`_encode_single(self, x)`**
  - 输入 `[N,3,H,W]`。
  - `timm` 路径：执行骨干，取 `out_index` 对应特征，若尚未创建 `proj` 则根据 `feat.shape[1]` 初始化并缓存；返回投影后的特征。
  - 非 `timm` 路径：直接调用自建卷积网络。

- **`forward(self, images)`**
  - 允许两种输入形状：
    - `[B*V,3,H,W]`：假设 `B=1`，输出 reshape 为 `[1,V,C,Hf,Wf]`。
    - `[B,V,3,H,W]`：先展平为 `[B*V,3,H,W]`，编码后 reshape 回 `[B,V,C,Hf,Wf]`。
  - 不符合的维度抛出 `ValueError`。

- **`load_pretrained` / `freeze`**
  - 直接调用基类实现，未扩展额外逻辑。

---

## `project/models/fusion/geometry.py`
### 模块概览
将多视角特征映射到统一的鸟瞰网格。可选择 `grid_sample` 或 `kornia.warp_perspective` 实现。

### `class GeometryTransformer(nn.Module)`
- **`__init__(self, bev_h, bev_w, bev_bounds, warp_impl='grid_sample')`**
  - 存储 BEV 网格尺寸、边界 `(x_min, x_max, y_min, y_max)`。
  - 计算每个网格单元的空间分辨率 `res_x`, `res_y`。
  - 通过 `register_buffer` 预生成地面网格 (`_create_ground_grid`) 并禁止持久化。
  - 允许 `warp_impl` 为 `'grid_sample'` 或 `'kornia'`，其他值自动回退到前者。

- **`_create_ground_grid(self) -> Tensor[H,W,3]`**
  - 在世界坐标系内构建 BEV 网格：`x` 坐标均匀分布于 `[x_min+0.5res_x, x_max-0.5res_x]`，`y` 同理。
  - 返回 `[H,W,3]` 齐次坐标张量（最后一维恒为 1）。

- **`_compute_homography(K, Rt)` / `_compute_img_to_world_homography(K, Rt)`（静态方法）**
  - 前者提取 `Rt` 的旋转和平移形成 3×3 单应矩阵；对输入形状做容错处理。
  - 后者在前者基础上求逆，遇到奇异/NaN 时回退到伪逆。

- **`forward(self, feats, intrinsics, extrinsics, img_size=(1080,1920)) -> Tensor[B,V,C,H_bev,W_bev]`**
  - **输入**：
    - `feats`：编码器输出 `[B,V,C,Hf,Wf]`。
    - `intrinsics` / `extrinsics`：可为 `Tensor[B,V,...]` 或 `List[List[Tensor]]`；内部通过 `get_K` / `get_Rt` 正规化。
    - `img_size`：原始图像高宽，用于从特征空间还原到像素坐标。
  - **流程**：
    1. 创建输出张量 `[B,V,C,bev_h,bev_w]`。
    2. 双层循环遍历 batch 和视角：
       - 若选择 `kornia` 且包可用：
         - 组合 `S_feat2img`（特征→像素缩放）、`H_i2w`（像素→世界）、`A_w2bev`（世界→BEV 网格）构建透视矩阵 `M`。
         - 检查 `det(M)` 是否有效；若合法则调用 `KGT.warp_perspective` 直接采样到目标分辨率。
       - 否则 fallback：
         - 利用 `_compute_homography` 将地面网格投影到像素，再按特征图尺寸缩放至 `[−1,1]` 归一化坐标。
         - 调用 `F.grid_sample` 双线性插值，缺失区域使用 0 填充。
  - **注意**：未缓存 `M`，若频繁调用可能造成性能压力；`_grid_cache` 留作扩展。

---

## `project/models/fusion/fusion.py`
### 模块概览
定义 BEV 特征融合策略。

- `class FusionModule(nn.Module)`
  - 抽象基类，仅定义 `forward` 接口签名（`[B,V,C,H,W] -> [B,C,H,W]`）。

- `class SimpleFusion(FusionModule)`
  - **`__init__(self, mode='sum')`**：验证 mode 属于 `'sum'|'mean'|'max'`。
  - **`forward(self, bev_maps)`**：根据模式在视角维度做求和、平均或逐元素最大。

- `class AttentionFusion(FusionModule)`
  - 将每个 BEV 位置的多视角特征视作长度为 V 的序列，借助可学习的 query token 与 `nn.MultiheadAttention` 完成跨视角注意力池化。
  - `__init__(channel_dim, num_heads=4, dropout=0.0)`：校验通道数与头数的整除关系，实例化多头注意力、LayerNorm 与 MLP，用以提炼聚合后的特征。
  - `forward(self, bev_maps)`：
    1. 把 `[B,V,C,H,W]` 排列为 `[V,B*H*W,C]`，使每个 BEV 栅格对应一个 attention batch。
    2. 通过 query token 汇聚各视角特征，得到 `[B*H*W,C]` 的融合表示。
    3. 使用 `LayerNorm + MLP` 做残差平滑，最后 reshape 回 `[B,C,H,W]`。

- `class ConcatFusion(FusionModule)`
  - **`forward(self, bev_maps)`**：重新 reshape 为 `[B, V*C, H, W]`，即沿通道维拼接视角特征。`BEVNet` 默认使用该策略。

---

## `project/models/heads/detector.py`
### 模块概览
包含 BEV 热力图检测头与解码逻辑。

- `class BEVDetector(nn.Module)`
  - **`__init__(self, in_channels=32, heatmap_sigma=2.0, bev_bounds=(-6,6,-2,2))`**：
    - 构建三层扩张卷积 + GroupNorm + ReLU 的序列，输出单通道 logits。
    - 保存高斯 σ（供潜在损失使用）与 BEV 边界（解码时用）。
  - **`forward(self, bev_feat)`**：输入 `[B,C,H,W]`，输出 `{'heatmap': sigmoid(self.head(bev_feat))}`。
  - **`_nms2d(x, kernel=3)`（静态）**：通过 `max_pool2d` 找局部极大值并保留对应位置，其余置零。
  - **`decode(self, heatmap, conf_thresh=0.4, box_size_m=(0.6,0.6), nms_dist_m=0.5)`**：
    1. 对输入 `[B,1,H,W]` 先做 `_nms2d`，再根据阈值筛选峰值。
    2. 将像素索引换算到米制坐标（中心位移 0.5 个像素）。
    3. 生成固定大小的盒子 `[cx, cy, w, h]`。
    4. 执行基于欧氏距离的简单 NMS：按分数降序遍历，若与已保留中心距离 < `nms_dist_m` 则丢弃。
    5. 返回 `boxes_list` 与 `scores_list`，均为 batch 列表。

- `class AnchorDetector(nn.Module)`
  - 仅在初始化时打印占位提示，无进一步方法；表明未来可能扩展锚框检测器。

---

## `project/models/model_wrapper.py`
### 模块概览
将编码器、几何变换、融合与检测头封装为端到端模型，并包含损失与辅助函数。

- `focal_loss(pred, gt, alpha=0.25, gamma=2.0)`（模块级）
  - 对输入概率做裁剪避免 log(0)，根据 GT 选择正负样本权重，返回均值损失。当前未在训练流程中调用，预留备用。

- `class BEVNet(nn.Module)`
  - **`__init__(self, cfg)`**：
    - 解析模型相关配置：特征通道、BEV 尺寸/边界、骨干、预训练开关、特征层索引，以及推理/损失阈值。
    - 初始化组件：`CNNEncoder`、`GeometryTransformer`（默认优先 `kornia`），并依据 `MODEL.FUSION.TYPE` 选择 `ConcatFusion`、`SimpleFusion` 或 `AttentionFusion`。
    - 检测头延迟构建，以便在第一次前向时确定输入通道数（Concat 产生 `V*C + 2` 通道，注意力/简单融合保持 `C + 2`）。
    - 通过 `register_buffer` 缓存 `pos_enc`（`_create_pos_enc` 生成）。
  - **`forward(self, batch)`**：
    1. 读取 `batch['images']`（`[B,V,3,H,W]`）。
    2. 经过编码器得到每视角特征 `[B,V,C,Hf,Wf]`。
    3. 将 `calib` 中的列表标定堆叠成张量，传入 `GeometryTransformer` 获得 BEV 对齐特征。
    4. 使用 `ConcatFusion` 拼接视角特征，得到 `[B,V*C,H_bev,W_bev]`。
    5. 与位置编码拼接，若 `detector` 尚未实例化则根据当前通道构建并移动至设备。
    6. 前向检测头获得 `heatmap`，随后调用 `decode` 计算 `boxes`、`scores`。
    7. 返回字典：`{'heatmap', 'boxes', 'scores', 'bev_feat'}`。
  - **`loss(self, preds, targets, loss_cfg)`**：
    - 目前仅计算均方误差：
      1. 调用 `_build_gt_heatmap_gaussian` 生成 `[B,1,H,W]` GT 热力图（Gaussian 模板半径 `≈3σ`）。
      2. 与 `preds['heatmap']` 做 `F.mse_loss`。
      3. 返回 `{'mse_loss': mse, 'total_loss': mse}`。
  - **`_build_gt_heatmap_gaussian(self, targets, sigma_px=1.0)`**：
    - 预生成 `[-rad, rad]` 高斯核（`rad = max(1, int(3*sigma))`）。
    - 遍历目标世界坐标，将其映射到网格索引（包含裁剪），再在局部窗口内取 `max` 叠加高斯。
    - 返回位于模型设备上的热力图张量。
  - **`_geom_consistency_loss(self, targets, num_samples=128)`**（当前未被 `loss` 调用）
    - 随机采样 BEV 网格点，经 `_compute_homography` / `_compute_img_to_world_homography` 投影到像素再反投影回世界坐标。
    - 对比往返误差（L1），对所有相机平均，返回一个标量张量。
  - **`_create_pos_enc(H, W, bounds)`（静态）**
    - 构建 2 通道正弦/余弦位置编码：`x` 轴取正弦，`y` 轴取余弦，均基于归一化坐标。

---

## `project/models/__init__.py`
- 仅包含注释 `# models package for BEVNet`，用于标记 Python 包，无执行逻辑。

## `project/models/encoders/__init__.py`
- 注释 `# encoders package`，声明子包。

## `project/models/fusion/__init__.py`
- 注释 `# fusion package`。

## `project/models/heads/__init__.py`
- 注释 `# heads package`。

---

## `project/data/__init__.py`
- 注释 `# data package`，用于包初始化。

---

## `project/utils/__init__.py`
- 注释 `# utils package`。

---

## `project/utils/visualization.py`
### 模块概览
提供训练/推理结果的可视化与导出功能。

- `save_bev_heatmap(heatmap: torch.Tensor, save_path: str)`
  - 创建输出目录，接受 `[B,1,H,W]` 或 `[1,1,H,W]` 张量；若四维则默认使用第一批/通道。
  - 使用 `matplotlib` 生成 `hot` 颜色图，带 colorbar，保存并关闭图像对象。
  - 依赖：`matplotlib.pyplot`（需在无显示的环境中使用合适的后端）。

- `save_predictions_json(boxes_list, scores_list, save_dir, frame_indices)`
  - 对 batch 内每帧：
    - 将 `Tensor` 转为 Python list（若为 `None` 则输出空列表）。
    - 写入 `frame_{frame_idx:06d}.json`，包含 `frame_idx`、`boxes`、`scores`。
  - 自动创建输出目录；会覆盖同名文件。

---

## `project/utils/geometry.py`
### 模块概览
提供世界坐标与 BEV 网格之间的互相转换。

- `meters_to_bev_indices(xy, bev_bounds, bev_size)`
  - 将世界坐标 `(x,y)` 转为网格索引 `(col,row)`：按照边界计算分辨率，再做裁剪以避免越界。
  - 返回 `Tensor[N,2]`，小数部分保留（未取整）。

- `bev_indices_to_meters(idx, bev_bounds, bev_size)`
  - 将网格索引转回米制坐标，以网格中心为准 (`+0.5`)，返回 `Tensor[N,2]`。

---

## `project/scripts/check_wildtrack_dataset.py`
### 模块概览
用于快速验证本地 Wildtrack 数据路径是否有效的脚本。

- 顶部将仓库根目录加入 `sys.path`，以便直接导入 `WildtrackDataset`。
- `CFG` 提供最小化配置（包含数据根路径、视角数、图像尺寸）。
- 在 `if __name__ == '__main__'` 块中：
  1. 实例化 `WildtrackDataset`。
  2. 打印帧数、视角数量、第一组标定矩阵形状与首帧文件名。
- **用途**：运行脚本即可确认数据可读、标定解析正常。

---

## `README.md` / 其他非 Python 文件
- 本次需求聚焦于 Python 方法级说明；非代码文件未在此逐项展开。

---

> 若未来新增/修改模块，请同步更新本文档，确保“每个文件的每个方法”均有详尽说明。
