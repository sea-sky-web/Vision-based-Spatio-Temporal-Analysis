# BEV-PedTrack: 面向固定场景的多视角行人感知系统

**BEV-PedTrack: A Multi-View Pedestrian Perception System for Static Scenes using Geometric BEV Fusion**

## 摘要 (Abstract)

本项目旨在开发一个高效、轻量化的多视角行人感知系统，专门针对交通枢纽、广场、园区等拥有固定摄像头的监控场景。传统方法通常在各个2D视角独立进行检测，再进行复杂的跨镜头目标匹配（Re-ID），这种“后期融合”的策略在人群拥挤和频繁遮挡的场景下表现不佳。

为解决此问题，本项目采用**“先融合、后检测” (Fuse-before-Detect)** 的核心思想。我们利用已知的相机几何标定参数，通过逆透视变换（IPM）将来自多个视角的**深度特征图**直接投影并融合到统一的鸟瞰图（BEV）空间。所有后续的感知任务，如行人检测、跟踪和轨迹预测，都在这个信息丰富的、全局统一的BEV表征上进行。这种方法从根本上规避了跨镜头Re-ID的难题，并能有效利用多视角信息来应对遮挡。

本项目将以模块化的方式，使用现代化的深度学习框架（PyTorch 2.x, `timm`, `OpenCV`）进行构建，确保代码的健壮性、可读性和可扩展性。

## 核心特性 (Key Features)

* **BEV中心化设计**: 所有感知任务均在统一的BEV“上帝视角”下完成，天然解决了多视角数据不一致的问题。
* **几何感知，而非3D预测**: 充分利用静态场景中已知的相机几何关系，通过高效的IPM进行视图变换，而非采用复杂的、为自动驾驶设计的深度预测或3D空间推断网络，更加轻量高效。
* **天生免除Re-ID**: 通过在检测前融合多视角特征，从架构上消除了在2D图像空间进行跨镜头目标匹配的需要。
* **抗遮挡性**: 融合后的BEV特征可以综合利用不同视角的信息，一个视角中的遮挡可以被其他视角的信息所弥补。
* **现代化与模块化**: 基于最新的、稳定的深度学习库构建，代码结构清晰，易于迭代和扩展。

## 技术架构 (Technical Architecture)

系统遵循一个清晰的、端到端的流水线设计：

![Pipeline Diagram](https://i.imgur.com/your-pipeline-diagram-placeholder.png)  1.  **输入 (Input)**:
    * 来自N个同步摄像头的图像帧 (N x H x W x C)。
    * 每个摄像头的内外参标定文件。

2.  **2D特征提取 (Backbone)**:
    * 每个图像帧独立通过一个轻量级的骨干网络 (如 ResNet, EfficientNet) 提取多尺度2D特征图。

3.  **视图投影 (View Projection via IPM)**:
    * 利用相机参数计算单应性矩阵 (Homography Matrix)。
    * 使用OpenCV对2D特征图进行逆透视变换，将其“拍扁”并投影到预定义的BEV世界坐标系下的虚拟“地面”上。

4.  **BEV特征融合 (BEV Fusion)**:
    * 将来自N个摄像头的BEV特征图进行融合，生成一张单一的、信息增强的BEV特征图。仓库提供了逐元素 `SimpleFusion`、沿通道拼接的 `ConcatFusion` 以及跨视角注意力池化的 `AttentionFusion`，可通过配置 `MODEL.FUSION.TYPE` 在三者之间切换。

5.  **下游任务头 (Downstream Heads)**:
    * 在融合后的BEV特征图上，接入不同的任务头来完成特定任务：
        * **检测头 (Detection Head)**: 输出行人在BEV空间中的位置（例如，通过热力图）。
        * **跟踪模块 (Tracking Module)**: 将连续帧的检测结果关联起来，形成运动轨迹。

## 开发路线图 (Development Roadmap)

本项目将分阶段进行，确保每一步都稳固可靠。

### Phase 1: v1.0 - 核心架构验证

* **目标**: 搭建并跑通从多视角图像输入到融合BEV特征图输出的核心流程，验证技术路线的可行性。
* **核心任务**:
    1.  使用 `timm` 库实现2D特征提取。
    2.  使用 `OpenCV` 实现特征图的IPM投影。
    3.  实现最简单的BEV特征融合方法：**逐像素相加/平均**。
* **成功标准**: 能够稳定输入多路Wildtrack图像，并输出一张视觉上清晰、合理的融合BEV特征图。**此阶段不要求实现检测头和注意力机制。**

### Phase 2: v2.0 - 智能融合与检测

* **目标**: 提升BEV特征质量，并实现端到端的行人检测。
* **核心任务**:
    1.  将v1.0的简单融合模块，升级为基于**可变形注意力机制 (Deformable Attention)** 的智能融合模块，借鉴MVDeTr的思想。
    2.  在融合后的BEV特征图上，实现一个简单的检测头（如 CenterNet-style 的热力图检测头）。
* **成功标准**: 模型能够端到端训练，并在BEV图上输出准确的行人检测热点。

### Phase 3: v3.0 - 完整跟踪系统

* **目标**: 构建完整的、端到端的BEV多目标跟踪系统。
* **核心任务**:
    1.  在v2.0的检测结果基础上，集成一个轻量级的在线跟踪算法（如 SORT 或 DeepSORT 的变体）。
    2.  端到端优化整个模型，提升MODA/MODP等MCMT评价指标。
* **成功标准**: 系统能够稳定输出带有时序和ID信息的行人轨迹。

## 数据集 (Dataset)

本项目主要面向**Wildtrack**及其同类数据集。这类数据集必须提供：
* 同步的多视角视频帧。
* 每个摄像头的内外参数（相机矩阵、畸变系数、旋转和平移矩阵）。

## 环境设置 (Setup & Installation)

```bash
# 1. 创建并激活Conda环境 (推荐)
conda create -n bev_pedtrack python=3.10 -y
conda activate bev_pedtrack

# 2. 安装 PyTorch (请根据你的CUDA版本访问PyTorch官网获取对应命令)
# 例如 CUDA 12.1:
pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 3. 安装核心依赖
pip install timm         # 用于加载骨干网络
pip install opencv-python  # 用于图像处理和IPM
pip install numpy
pip install matplotlib   # 用于可视化

# 4. (可选) 安装ultralytics用于性能对比
pip install ultralytics
```

## 使用方法 (Usage)

*(待代码实现后填充)*

### 训练 (Training)

```bash
python train.py --config configs/wildtrack_v1_resnet50.yaml
```

### 推理 (Inference)

```bash
python inference.py --config configs/wildtrack_v1_resnet50.yaml --input_path /path/to/your/image_folders --output_path /path/to/save/bev_map
```

## 未来工作 (Future Work)

* 集成轨迹预测模块，实现行人未来位置的预测。
* 对模型进行实时性优化，探索知识蒸馏、模型剪枝等技术。
* 将系统扩展到更复杂的场景，如存在高度变化的非平面场景。

## 参考 (References)

* [MVDeT: Multi-view Detection with Transformer](https://arxiv.org/abs/2104.14592)
* [MVDeTr: Multi-view Detection Transformer for 3D Object Detection](https://arxiv.org/abs/2110.05214)
* [Wildtrack: A Multi-camera HD Dataset for Dense Unscripted Pedestrian Detection](https://arxiv.org/abs/1710.10103)