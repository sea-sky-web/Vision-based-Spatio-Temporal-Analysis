# 多视角到BEV表征融合（基于BEVFormer）

本项目基于BEVFormer开源框架，用于快速验证多视角图像到鸟瞰视图（BEV）的融合效果，特别针对交通枢纽场景下的行人遮挡和跨视角一致性问题。

## 项目背景

BEVFormer是一个基于Transformer的框架，使用时空Transformer将多摄像头图像转换为统一的BEV表征。通过空间跨注意力聚合多视角特征，并可选地加入时间自注意力处理历史帧。本项目利用BEVFormer框架，针对WildTrack数据集进行快速验证。

## 安装步骤

1. 克隆BEVFormer仓库（如果尚未完成）：
   ```bash
   git clone https://github.com/fundamentalvision/BEVFormer
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 安装mmcv-full（可能需要特定版本）：
   ```bash
   pip install -v -U git+https://github.com/open-mmlab/mmcv.git@master
   ```

4. 下载预训练模型：
   ```bash
   wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_base_epoch_24.pth
   ```

5. 准备WildTrack数据集：
   - 将WildTrack数据集放置在`data/Wildtrack/`目录下
   - 确保数据集包含以下目录结构：
     - `Image_subsets/`：包含C1-C7七个摄像头的图像
     - `calibrations/`：包含相机标定参数
     - `annotations_positions/`：包含标注信息

## 使用方法

运行验证脚本：
```bash
python validate_wildtrack.py
```

脚本将执行以下操作：
1. 加载预训练的BEVFormer模型
2. 加载WildTrack数据集的第一帧图像和相机参数
3. 使用BEVFormer模型生成BEV特征图
4. 可视化BEV特征图并保存为`test_bev.png`

## 项目结构

- `custom/`：自定义代码目录
  - `wildtrack_dataset.py`：WildTrack数据集加载类
  - `visualize_bev.py`：BEV特征可视化工具
- `projects/configs/bevformer/`：配置文件目录
  - `custom_wildtrack.py`：适配WildTrack数据集的BEVFormer配置
- `data/`：数据集目录
  - `Wildtrack/`：WildTrack数据集
- `validate_wildtrack.py`：验证脚本
- `requirements.txt`：依赖列表
- `README_custom.md`：本文档

## 预期结果

成功运行后，将生成`test_bev.png`文件，显示BEV特征图。在该特征图中，应该能够观察到行人的分布情况，验证多视角融合的效果。

## 注意事项

- 本项目仅用于快速验证，不包含训练过程
- 使用的预训练模型基于nuScenes数据集，可能需要进一步微调以适应WildTrack数据集
- 相机参数的解析可能需要根据WildTrack数据集的实际格式进行调整