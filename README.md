# Vision-based Spatio-Temporal Analysis (Multi-View to BEV)

本工程基于 [BEVFormer](https://github.com/fundamentalvision/BEVFormer) 开源框架，旨在快速验证多视角图像到鸟瞰视图（BEV）表征的融合效果，特别针对复杂场景（如交通枢纽）下的行人检测、跟踪和分析。

## 项目简介

BEVFormer 是一个强大的、基于 Transformer 的模型，能够将来自多个摄像头的图像输入，通过空间跨注意力和时间自注意力机制，将其融合成一个统一的时空鸟瞰视图（BEV）表征。

本项目利用 BEVFormer 的强大能力，在 [Wildtrack 数据集](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) 上进行算法验证，以解决多视角场景下的目标遮挡和跨视角一致性问题。

## 项目结构

```
.
├── BEVFormer/               # BEVFormer 核心代码、配置和自定义脚本
│   ├── custom/              # 本项目的自定义模块
│   │   ├── wildtrack_dataset.py  # Wildtrack 数据集加载器
│   │   └── visualize_bev.py      # BEV 特征图可视化工具
│   ├── projects/configs/    # 模型和数据集配置文件
│   ├── validate_wildtrack.py # 用于验证和可视化的主脚本
│   └── requirements.txt     # Python 依赖项列表
├── README.md                # 本文档
...
```

## 1. 环境搭建

为了确保兼容性并避免潜在的依赖冲突，强烈建议使用 `conda` 创建一个独立的 **Python 3.8** 虚拟环境。

### 系统要求
- **Python 3.8 (强制)**
- PyTorch 1.9.0+
- CUDA 11.1+

### 安装步骤

1.  **克隆本项目:**
    ```bash
    git clone <repository-url>
    cd Vision-based-Spatio-Temporal-Analysis
    ```

2.  **创建并激活 Conda 环境:**
    ```bash
    conda create -n bevformer-wildtrack python=3.8
    conda activate bevformer-wildtrack
    ```

3.  **安装核心依赖 (PyTorch, MMCV, MMDetection):**
    *为了避免因本地编译耗时过长或失败，我们提供了一系列使用预编译包的命令。请严格按照以下步骤执行。*

    a. **安装 PyTorch:**
    ```bash
    # 适用于 CUDA 11.1 的版本
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    ```

    b. **安装 MMCV:**
    ```bash
    pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
    ```

    c. **安装 MMDetection:**
    ```bash
    pip install mmdet==2.14.0
    ```

    d. **安装 MMDetection3D:**
    ```bash
    pip install mmdet3d==0.14.0
    ```

4.  **安装其他依赖项:**
    ```bash
    cd BEVFormer
    pip install -r requirements.txt
    cd ..
    ```

## 2. 数据准备

1.  **下载 Wildtrack 数据集**:
    从 [Wildtrack 官网](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)下载数据集，并将其解压。

2.  **放置数据集**:
    将解压后的数据集内容放置在 `BEVFormer/data/Wildtrack_dataset/` 目录下。

3.  **创建数据符号链接**:
    为了让代码能正确找到数据，需要在 `BEVFormer/` 目录下创建一个指向数据集的符号链接。
    -   **Windows (以管理员身份运行命令提示符):**
        ```cmd
        cd BEVFormer
        mklink /D data\Wildtrack ..\data\Wildtrack_dataset
        cd ..
        ```
    -   **Linux / macOS:**
        ```bash
        cd BEVFormer
        ln -s ../data/Wildtrack_dataset data/Wildtrack
        cd ..
        ```

## 3. 预训练模型

1.  **下载 BEVFormer 预训练权重**:
    验证脚本需要 `bevformer_base_epoch_24.pth` 模型权重。
    从 [官方发布页面](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_base_epoch_24.pth) 下载。

2.  **放置模型文件**:
    将下载的 `.pth` 文件移动到 `BEVFormer/` 目录下。

## 4. 运行与验证

本项目提供了一个主脚本 `validate_wildtrack.py` 用于运行验证和生成可视化结果。

1.  **进入 `BEVFormer` 目录:**
    ```bash
    cd BEVFormer
    ```

2.  **运行验证脚本:**
    ```bash
    python validate_wildtrack.py
    ```

脚本将自动执行以下操作:
1.  加载预训练的 BEVFormer 模型和自定义配置。
2.  加载 Wildtrack 数据集的第一帧图像及相机参数。
3.  通过模型推理，生成 BEV 特征图。
4.  将 BEV 特征图可视化，并保存为 `test_bev.png`。

### 预期结果
成功运行后，将在 `BEVFormer/` 目录下生成 `test_bev.png` 文件。该图像是一个热力图，展示了场景的 BEV 表征。从图中可以直观地观察到行人的分布情况，从而验证多视角融合的效果。

## 注意事项

- 本项目当前仅用于**快速验证**，不包含模型训练部分。
- 使用的预训练模型基于 nuScenes 数据集，在 Wildtrack 数据集上可能不是最优的，需要进一步微调才能达到更好的性能。
- 相机参数的解析逻辑位于 `custom/wildtrack_dataset.py` 中，可能需要根据实际的数据格式进行调整。

## 常见问题

1.  **安装 `mmcv-full` 或 `mmdet3d` 失败/超时:**
    - **原因**: 这通常是由于本地环境缺少编译工具链或网络问题导致无法直接从源码编译。
    - **解决方案**: 请严格遵循 `1. 环境搭建` 中提供的命令，通过 `-f` 参数指定预编译包的下载地址，可以避免本地编译。

2.  **`ModuleNotFoundError: No module named 'numpy.distutils'`:**
    - **原因**: 这个问题出现在较新的 `numpy` 版本（>=1.22）中，因为 `distutils` 已被移除。项目依赖的 `numba==0.48.0` 需要此模块。
    - **解决方案**: 确保您使用的是 **Python 3.8** 环境。在此环境下，`pip` 会自动安装与 `numba==0.48.0` 兼容的 `numpy` 版本。

3.  **`AttributeError: module 'configparser' has no attribute 'SafeConfigParser'`:**
    - **原因**: `SafeConfigParser` 在 Python 3.12 中已被移除。
    - **解决方案**: 这是环境不兼容的典型表现。请务必使用 **Python 3.8**。
