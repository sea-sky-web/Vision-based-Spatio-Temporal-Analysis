"""Configuration for Wildtrack v1 dry-run"""

ROOT = '.'
DATA_ROOT = 'Data/Wildtrack'
BATCH_SIZE = 2
NUM_CAMERAS = 7
IMAGE_SIZE = (3, 256, 256)
BEV_SIZE = (1, 128, 128)
DEVICE = 'cpu'

# BEV bounds will be updated dynamically based on annotation analysis
# Default bounds (will be overridden by bev_utils.analyze_annotation_bounds)
BEV_BOUNDS = (-6, 6, -2, 2)  # (min_x, max_x, min_y, max_y) in meters

# Projection/Fusion settings
# 'homography' 使用地平面单应性；'mvdet' 使用MVDet风格竖直采样线
PROJECTION_MODE = 'homography'
# 多视角融合策略：'sum'、'mean' 或 'max'
FUSION_MODE = 'sum'
# 透视变换实现：'grid_sample' 或 'kornia'
WARP_IMPL = 'kornia'

# Backbone settings
BACKBONE_MODEL = 'resnet18'
BACKBONE_OUT_INDEX = 2
BACKBONE_OUT_CHANNELS = 32
BACKBONE_PRETRAINED = True

# Loss and training settings
LOSS_TYPE = 'focal'  # 'focal' 或 'mse'
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
LR = 1e-3
WEIGHT_DECAY = 0.0
EPOCHS = 1  # 最小训练循环，默认1个epoch
SAVE_VIS = True  # 是否保存可视化
HEATMAP_SIGMA = 2.0  # GT高斯半径（像素尺度sigma）
NMS_KERNEL = 3       # 推理NMS核大小
CONF_THRESH = 0.4    # 推理置信度阈值


def __repr__():
    return f"WildtrackConfig(DATA_ROOT={DATA_ROOT}, BATCH_SIZE={BATCH_SIZE}, BEV_BOUNDS={BEV_BOUNDS})"
