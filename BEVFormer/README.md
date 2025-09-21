# 端到端多视角BEV融合模型 (v1.0 框架)

本项目旨在构建一个先进的、端到端可训练的多视角行人感知模型。我们采用业界领先的 **“先融合、后检测” (Fuse-before-Detect)** 范式，在统一的鸟瞰图（BEV）空间中对行人进行感知，从根本上解决了传统方法中棘手的跨镜头重识别（Re-ID）难题。

**架构升级**: 本项目 v1.0 版本已从初期的流水线式验证脚本，**重构为一个面向对象的、模块化的、支持端到端训练的深度学习模型框架**。当前版本的核心目标是搭建并验证整个模型的骨架，为后续的正式训练和部署奠定坚实基础。

## 核心设计原则

1.  **面向对象 (Object-Oriented)**: 整个模型 (`BEVFusionNet`)、数据加载器 (`WildtrackDataset`) 和各个子模块（骨干网络、投影、融合、检测头）都被封装成独立的类，实现了高内聚、低耦合。
2.  **端到端可训练 (End-to-End Trainable)**: 整体架构被设计为一个单一的 PyTorch `nn.Module`。一旦损失函数和BEV空间的真值（Ground Truth）标签被定义，模型就可以进行完整的端到端训练。
3.  **配置驱动 (Config-Driven)**: 所有的实验参数（如数据集路径、模型参数、BEV空间定义等）都由独立的配置文件管理，实现了代码与配置的分离，便于快速迭代和实验。
4.  **模块化与可扩展性 (Modular & Scalable)**: 模型的每个组件（如骨干网络、融合策略、检测头）都是可插拔的。您可以轻松更换`ResNet50`为`EfficientNet`，或将简单的平均融合升级为注意力融合，而无需改动主体框架。

## 端到端模型架构

模型的数据流严格遵循以下步骤，构成一个完整的前向传播路径：

```
[多视角图像] --> [Backbone (特征提取)] --> [Projection (投影变换)] --> [Fusion (特征融合)] --> [BEV Head (检测头)] --> [输出预测]
```

-   **Backbone**: 并行地从N个视角的输入图像中提取2D特征图。
-   **Projection**: 利用相机内外参计算单应性矩阵，将每个视角的2D特征图精确投影到统一的BEV空间。
-   **Fusion**: 将N个投影后的BEV特征图融合成一个包含全局信息的单一BEV特征图。
-   **BEV Head**: 在融合后的BEV特征图上进行卷积操作，直接预测行人的位置（例如，通过生成热力图）。

## 项目结构

为了支持上述设计原则，项目采用了以下模块化的目录结构：

```
bev_fusion_v1/
├── train.py                # ✅ 主入口：用于启动模型训练与验证
├── configs/
│   └── wildtrack_v1.py     # 配置文件，管理所有实验参数
├── datasets/
│   └── wildtrack_dataset.py# 实现了PyTorch Dataset的Wildtrack数据加载器
├── models/
│   ├── bev_fusion_net.py   # 核心模型文件，定义了端到端的BEVFusionNet
│   └── modules/            # 存放模型子模块的目录
│       ├── backbone.py     # 2D特征提取器
│       ├── projection.py   # 几何投影模块 (计算H矩阵和执行warp)
│       ├── fusion.py       # BEV特征融合模块
│       └── bev_head.py     # BEV检测头模块 (v1.0为简化版)
└── utils/
    ├── visualization.py    # 可视化工具
    └── ...                 # 其他辅助函数
```

## 环境与依赖

环境设置保持不变，所有依赖项均已在 `requirements.txt` 中精确锁定版本。

1.  **创建并激活Python虚拟环境** (推荐使用 Conda)
    ```bash
    conda create -n bev_fusion python=3.10 -y
    conda activate bev_fusion
    ```

2.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

## 数据集准备 (Wildtrack)

数据准备方式与之前一致，但现在由一个专门的`Dataset`类 (`datasets/wildtrack_dataset.py`) 来负责加载和处理，这使得与PyTorch的`DataLoader`集成变得无缝。

- **目录结构**: 请确保Wildtrack数据集存放于 `data/wildtrack/` 目录下。
- **相机参数**: `datasets/wildtrack_dataset.py` 内部将包含解析 `calibrations.json` 的逻辑，自动完成从Rodrigues向量到旋转矩阵、从相机位置到平移向量`t`的转换。

## 如何运行 v1.0 框架

在v1.0框架下，我们不再运行一个简单的演示脚本，而是启动一个**训练脚本**来进行**模型正向传播的“空跑” (Dry Run)**。这一步的目的是验证整个端到端架构是否能够无误地运行，包括数据加载、特征提取、投影、融合，直到检测头输出最终预测。

1.  **启动训练脚本**:
    通过指定配置文件来启动主训练程序。
    ```bash
    python train.py --config configs/wildtrack_v1.py
    ```

## 快速开始 (Windows PowerShell)

下面是在 Windows PowerShell 上为本项目创建虚拟环境并运行 dry-run 的最小步骤：

1. 在仓库根目录下创建并激活 venv（使用系统 Python）:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 安装依赖:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. 运行 dry-run:

```powershell
python train.py --config configs/wildtrack_v1.py
```

注意:
- `requirements.txt` 中列出的 `torch`/`torchvision` 版本默认假设 CUDA 11.8。如果你的机器没有 GPU 或者使用不同 CUDA 版本，请根据 https://pytorch.org/get-started/locally/ 选择合适的安装命令（或通过 `pip` 指定 cpu-only 版本）。
- 当前 scaffold 在没有真实数据集时会使用随机张量进行前向传播，因此你可以先验证框架是否能运行，再准备真实数据。


2.  **预期输出**:
    v1.0的“空跑”不会进行真正的模型训练（因为损失函数和标签生成尚未实现），但它会：
    * 在控制台打印日志，显示模型所有组件已成功初始化。
    * 显示数据加载器成功读取了一批（batch）数据。
    * 打印出模型在每个关键步骤后（如特征提取后、融合后）的张量形状。
    * 最重要的是，**打印出BEV检测头最终输出的预测张量（如热力图）的形状**，例如 `(BatchSize, NumClasses, BEV_Height, BEV_Width)`。

    成功看到最后的输出张量形状，即标志着我们v1.0的**端到端模型框架已搭建成功**。

## 后续工作 (v1.x -> v2.0)

基于当前稳固的v1.0框架，后续的开发将聚焦于：

1.  **真值标签生成**: 开发脚本，根据Wildtrack的标注数据在BEV空间生成用于监督训练的真值（如高斯热力图）。
2.  **损失函数实现**: 在 `train.py` 中定义损失函数（如用于热力图的Focal Loss）。
3.  **完整训练循环**: 完善 `train.py` 中的优化器、学习率调度器和训练/验证循环，开始真正的模型训练。
4.  **性能评估**: 编写 `eval.py` 脚本，使用标准的评估指标（如MODA, MODP）来衡量模型性能。
5.  **模块升级**: 迭代 `models/modules/` 中的模块，例如将平均融合替换为更先进的注意力融合机制。