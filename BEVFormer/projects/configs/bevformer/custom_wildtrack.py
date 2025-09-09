# 基础配置
_base_ = [
    '../../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../../mmdetection3d/configs/_base_/default_runtime.py'
]

# 全局变量
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
data_root = 'data/Wildtrack/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# BEV配置
bev_h = 100
bev_w = 100
point_cloud_range = [-20, -20, -1, 20, 20, 3] # 根据Wildtrack场景调整

# 模型配置
model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=False,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1, # 微调时解冻所有层
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h,
        bev_w=bev_w,
        num_query=300, # 查询数量减少
        num_classes=1, # 仅行人类别
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=False, # Wildtrack无CAN总线
            embed_dims=256,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3, # 减少层数以加速收敛
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(type='TemporalSelfAttention', embed_dims=256, num_levels=1),
                        dict(type='SpatialCrossAttention', pc_range=point_cloud_range, deformable_attention=dict(type='MSDeformableAttention3D', embed_dims=256, num_points=8, num_levels=3), embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=3, # 减少层数
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                        dict(type='CustomMSDeformableAttention', embed_dims=256, num_levels=3)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-22, -22, -1, 22, 22, 3],
            pc_range=point_cloud_range,
            max_num=100,
            voxel_size=[0.2, 0.2, 4],
            num_classes=1),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=bev_h,
            col_num_embed=bev_w),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=[0.2, 0.2, 4],
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range)))
)

# 数据集配置
dataset_type = 'WildTrackDataset'
class_names = ['Pedestrian']

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1080),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'wildtrack_infos_train.pkl', # 假设我们生成info文件
        pipeline=train_pipeline,
        classes=class_names,
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True),
        test_mode=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'wildtrack_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True),
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'wildtrack_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True),
        test_mode=True,
        box_type_3d='LiDAR')
)

# 优化器与学习率调度
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

# 运行配置
runner = dict(type='EpochBasedRunner', max_epochs=10) # 微调10个epoch
evaluation = dict(interval=1, pipeline=test_pipeline)
load_from = 'bevformer_base_epoch_24.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])