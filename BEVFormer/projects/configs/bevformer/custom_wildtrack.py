# 自定义WildTrack数据集的BEVFormer配置
# 基于bevformer_base.py配置，并进行必要的修改

# 假设从bevformer_base.py导入基础配置
# from .bevformer_base import *

# 数据集配置
dataset_type = 'CustomWildTrackDataset'  # 自定义数据集类型
data_root = 'data/Wildtrack/'  # WildTrack数据集根目录

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # 图像均值
    std=[58.395, 57.12, 57.375],     # 图像标准差
    to_rgb=True                       # 是否转换为RGB格式
)

# BEV配置
bev_h = 64  # BEV特征图高度
bev_w = 64  # BEV特征图宽度
queue_length = 1  # 队列长度，单帧设置为1

# 模型配置
model = dict(
    type='BEVFormer',  # 模型类型
    use_grid_mask=True,  # 是否使用网格掩码
    video_test_mode=False,  # 视频测试模式
    img_backbone=dict(
        type='ResNet',  # 图像骨干网络类型
        depth=101,      # ResNet深度
        num_stages=4,   # ResNet阶段数
        out_indices=(1, 2, 3),  # 输出特征图的索引
        frozen_stages=1,  # 冻结的阶段数
        norm_cfg=dict(type='BN', requires_grad=False),  # 归一化配置
        norm_eval=True,  # 是否在评估模式下使用归一化
        style='caffe',  # ResNet风格
    ),
    img_neck=dict(
        type='FPN',  # 特征金字塔网络
        in_channels=[512, 1024, 2048],  # 输入通道数
        out_channels=256,  # 输出通道数
        start_level=0,  # 起始级别
        add_extra_convs='on_output',  # 额外卷积的位置
        num_outs=3,  # 输出特征图的数量
        relu_before_extra_convs=True,  # 额外卷积前是否使用ReLU
    ),
    pts_bbox_head=dict(
        type='BEVFormerHead',  # BEVFormer头部类型
        bev_h=bev_h,  # BEV特征图高度
        bev_w=bev_w,  # BEV特征图宽度
        num_query=900,  # 查询数量
        num_classes=1,  # 类别数量，WildTrack只有行人类别
        in_channels=256,  # 输入通道数
        sync_cls_avg_factor=True,  # 是否同步类别平均因子
        with_box_refine=True,  # 是否使用框精炼
        as_two_stage=False,  # 是否作为两阶段模型
        transformer=dict(
            type='PerceptionTransformer',  # 感知Transformer类型
            rotate_prev_bev=True,  # 是否旋转先前的BEV
            use_shift=True,  # 是否使用位移
            use_can_bus=False,  # 是否使用CAN总线信息
            embed_dims=256,  # 嵌入维度
            encoder=dict(  # 编码器配置
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[dict(type='TemporalSelfAttention', embed_dims=256),
                               dict(type='SpatialCrossAttention', pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                                    deformable_attention=dict(type='MSDeformableAttention3D', embed_dims=256, num_points=8, num_levels=3),
                                    embed_dims=256)],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))
            ),
            decoder=dict(  # 解码器配置
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[dict(type='MultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                               dict(type='CustomMSDeformableAttention', embed_dims=256, num_levels=3)],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))
            )
        ),
        bbox_coder=dict(  # 边界框编码器配置
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=1
        ),
        positional_encoding=dict(  # 位置编码配置
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=bev_h,
            col_num_embed=bev_w,
        ),
        loss_cls=dict(  # 分类损失配置
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0
        ),
        loss_bbox=dict(  # 边界框损失配置
            type='L1Loss',
            loss_weight=0.25
        ),
        loss_iou=dict(  # IoU损失配置
            type='GIoULoss',
            loss_weight=0.0
        )
    ),
    train_cfg=dict(  # 训练配置
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            )
        )
    ),
    test_cfg=dict(  # 测试配置
        pts=dict(
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=300,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=[0.2, 0.2, 8],
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2
        )
    )
)

# 数据管道配置
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),  # 加载多视图图像
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),  # 归一化多视图图像
    dict(type='PadMultiViewImage', size_divisor=32),  # 填充多视图图像
    dict(type='MultiScaleFlipAug',
         img_scale=(1600, 900),  # 图像尺寸
         pts_scale_ratio=1,  # 点云尺度比例
         flip=False,  # 是否翻转
         transforms=[
             dict(type='DefaultFormatBundle'),  # 默认格式包
             dict(type='Collect', keys=['img'],  # 收集键
                  meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                             'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                             'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                             'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                             'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                             'transformation_3d_flow', 'cam_intrinsic', 'cam_extrinsic'))
         ])
]

# 数据配置
data = dict(
    samples_per_gpu=1,  # 每个GPU的样本数
    workers_per_gpu=4,  # 每个GPU的工作进程数
    test=dict(
        type=dataset_type,  # 数据集类型
        data_root=data_root,  # 数据集根目录
        ann_file=None,  # 标注文件，WildTrack不需要
        pipeline=test_pipeline,  # 测试管道
        classes=['Pedestrian'],  # 类别列表
        modality=dict(use_lidar=False, use_camera=True),  # 模态配置
        test_mode=True,  # 测试模式
    ),
)

# 优化器配置
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)  # 优化器类型

# 运行配置
runner = dict(type='EpochBasedRunner', max_epochs=24)  # 运行器类型

# 日志配置
log_config = dict(interval=50)  # 日志间隔