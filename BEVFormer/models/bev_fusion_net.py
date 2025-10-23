"""Simplified BEVFusionNet that wires backbone, projection, fusion, and head.

Adds a configurable projection path:
- 'homography': ground-plane homography warping using H = K [r1, r2, t]
- 'mvdet': vertical sampling lines per MVDet
"""
import torch
import torch.nn as nn
import numpy as np
from models.modules.backbone import SimpleBackbone, TimmBackbone
from models.modules.mvdet_projection import MVDetProjection
from models.modules.homography_warper import HomographyWarper
from models.modules.fusion import SimpleFusion
from models.modules.bev_head import SimpleBEVHead


class BEVFusionNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 特征提取骨干网络（支持timm预训练模型，配置可选）
        backbone_name = getattr(cfg, 'BACKBONE_MODEL', 'resnet18')
        backbone_out_idx = getattr(cfg, 'BACKBONE_OUT_INDEX', 2)
        backbone_out_channels = getattr(cfg, 'BACKBONE_OUT_CHANNELS', 32)
        backbone_pretrained = getattr(cfg, 'BACKBONE_PRETRAINED', True)
        try:
            self.backbone = TimmBackbone(
                model_name=backbone_name,
                out_indices=(backbone_out_idx,),
                out_channels=backbone_out_channels,
                pretrained=backbone_pretrained,
            )
            print(f"[BEVFusionNet] 使用TimmBackbone: {backbone_name}, out_idx={backbone_out_idx}, out_channels={backbone_out_channels}")
        except Exception as e:
            print(f"[BEVFusionNet] 初始化TimmBackbone失败({e})，回退到SimpleBackbone")
            self.backbone = SimpleBackbone(out_channels=backbone_out_channels)
        
        # 初始化投影模块，设置BEV空间范围（支持'homeography'或'mvdet'）
        bev_bounds = getattr(cfg, 'BEV_BOUNDS', (-10, 10, -10, 10))
        bev_h = cfg.BEV_SIZE[1] if isinstance(cfg.BEV_SIZE, (tuple, list)) and len(cfg.BEV_SIZE) == 3 else cfg.BEV_SIZE[0]
        bev_w = cfg.BEV_SIZE[2] if isinstance(cfg.BEV_SIZE, (tuple, list)) and len(cfg.BEV_SIZE) == 3 else cfg.BEV_SIZE[1]
        proj_mode = getattr(cfg, 'PROJECTION_MODE', 'homography')
        if proj_mode == 'mvdet':
            self.proj = MVDetProjection(bev_size=(bev_h, bev_w), bev_bounds=bev_bounds)
            print(f"[BEVFusionNet] 使用MVDetProjection, BEV=({bev_h},{bev_w}), bounds={bev_bounds}")
        else:
            warp_impl = getattr(cfg, 'WARP_IMPL', 'grid_sample')
            self.proj = HomographyWarper(bev_size=(1, bev_h, bev_w), bev_bounds=bev_bounds, warp_impl=warp_impl)
            print(f"[BEVFusionNet] 使用HomographyWarper({warp_impl}), BEV=({bev_h},{bev_w}), bounds={bev_bounds}")
        
        # 特征融合模块（根据配置设定模式：sum/mean/max）
        fusion_mode = getattr(self.cfg, 'FUSION_MODE', 'sum')
        self.fusion = SimpleFusion(mode=fusion_mode)
        
        # BEV头部网络
        self.head = SimpleBEVHead(cfg)

    def forward(self, images, intrinsics=None, extrinsics=None, meta=None, return_bev=False):
        # images: (B, NumCams, C, H, W) or (NumCams, C, H, W) if batchless
        if images.dim() == 4:
            # (NumCams, C, H, W) -> add batch
            images = images.unsqueeze(0)

        B = images.shape[0]
        num_cams = images.shape[1]
        device = images.device

        # 从meta中提取相机参数（如果提供）
        if meta is not None and intrinsics is None and 'intrinsics' in meta:
            intrinsics = meta['intrinsics']
            print(f"从meta中提取内参: {type(intrinsics)}")
            
        if meta is not None and extrinsics is None and 'extrinsics' in meta:
            extrinsics = meta['extrinsics']
            print(f"从meta中提取外参: {type(extrinsics)}")
        
        # 处理内参
        if intrinsics is not None:
            # 将内参转换为适当的格式
            if isinstance(intrinsics, list):
                # 检查是否是批次列表
                if len(intrinsics) == B:
                    # 批次列表，检查每个批次是否是相机列表
                    if isinstance(intrinsics[0], list):
                        # 将每个相机的内参转换为张量
                        tensor_intrinsics = []
                        for batch_intr in intrinsics:
                            batch_tensor_intr = []
                            for cam_intr in batch_intr:
                                if cam_intr is not None:
                                    if isinstance(cam_intr, np.ndarray):
                                        cam_intr = torch.from_numpy(cam_intr).float().to(device)
                                    elif not isinstance(cam_intr, torch.Tensor):
                                        cam_intr = torch.eye(3, dtype=torch.float32, device=device)
                                    else:
                                        cam_intr = cam_intr.to(device)
                                else:
                                    cam_intr = torch.eye(3, dtype=torch.float32, device=device)
                                batch_tensor_intr.append(cam_intr)
                            tensor_intrinsics.append(batch_tensor_intr)
                        intrinsics = tensor_intrinsics
                    else:
                        # 单个相机列表，为每个批次复制
                        tensor_intrinsics = []
                        for intr in intrinsics:
                            if intr is not None:
                                if isinstance(intr, np.ndarray):
                                    intr = torch.from_numpy(intr).float().to(device)
                                elif not isinstance(intr, torch.Tensor):
                                    intr = torch.eye(3, dtype=torch.float32, device=device)
                                else:
                                    intr = intr.to(device)
                            else:
                                intr = torch.eye(3, dtype=torch.float32, device=device)
                            tensor_intrinsics.append(intr)
                        intrinsics = tensor_intrinsics
                else:
                    # 相机列表，为每个批次复制
                    tensor_intrinsics = []
                    for i in range(B):
                        batch_tensor_intr = []
                        for j in range(min(len(intrinsics), num_cams)):
                            intr = intrinsics[j]
                            if intr is not None:
                                if isinstance(intr, np.ndarray):
                                    intr = torch.from_numpy(intr).float().to(device)
                                elif not isinstance(intr, torch.Tensor):
                                    intr = torch.eye(3, dtype=torch.float32, device=device)
                                else:
                                    intr = intr.to(device)
                            else:
                                intr = torch.eye(3, dtype=torch.float32, device=device)
                            batch_tensor_intr.append(intr)
                        # 如果相机数量不足，用默认值补充
                        while len(batch_tensor_intr) < num_cams:
                            batch_tensor_intr.append(torch.eye(3, dtype=torch.float32, device=device))
                        tensor_intrinsics.append(batch_tensor_intr)
                    intrinsics = tensor_intrinsics
            elif isinstance(intrinsics, torch.Tensor):
                # 张量格式，检查形状
                if intrinsics.dim() == 3 and intrinsics.shape[0] == B and intrinsics.shape[1] == 3 and intrinsics.shape[2] == 3:
                    # 单个相机的内参 (B, 3, 3)，为每个相机复制
                    intrinsics = [[intr.to(device)] * num_cams for intr in intrinsics]
                elif intrinsics.dim() == 4 and intrinsics.shape[0] == B and intrinsics.shape[1] == num_cams:
                    # 已经是正确的格式 (B, num_cams, 3, 3)
                    intrinsics = [[intrinsics[b, c].to(device) for c in range(num_cams)] for b in range(B)]
                else:
                    print(f"警告：内参形状不正确 {intrinsics.shape}，使用默认内参")
                    intrinsics = [[torch.eye(3, dtype=torch.float32, device=device) for _ in range(num_cams)] for _ in range(B)]
        else:
            print("警告：未提供内参，使用默认内参")
            intrinsics = [[torch.eye(3, dtype=torch.float32, device=device) for _ in range(num_cams)] for _ in range(B)]
        
        # 处理外参
        if extrinsics is not None:
            # 将外参转换为适当的格式
            if isinstance(extrinsics, list):
                # 检查是否是批次列表
                if len(extrinsics) == B:
                    # 批次列表，检查每个批次是否是相机列表
                    if isinstance(extrinsics[0], list):
                        # 将每个相机的外参转换为张量
                        tensor_extrinsics = []
                        for batch_extr in extrinsics:
                            batch_tensor_extr = []
                            for cam_extr in batch_extr:
                                if cam_extr is not None:
                                    if isinstance(cam_extr, np.ndarray):
                                        cam_extr = torch.from_numpy(cam_extr).float().to(device)
                                    elif not isinstance(cam_extr, torch.Tensor):
                                        cam_extr = torch.eye(4, dtype=torch.float32, device=device)
                                    else:
                                        cam_extr = cam_extr.to(device)
                                else:
                                    cam_extr = torch.eye(4, dtype=torch.float32, device=device)
                                batch_tensor_extr.append(cam_extr)
                            tensor_extrinsics.append(batch_tensor_extr)
                        extrinsics = tensor_extrinsics
                    else:
                        # 单个相机列表，为每个批次复制
                        tensor_extrinsics = []
                        for extr in extrinsics:
                            if extr is not None:
                                if isinstance(extr, np.ndarray):
                                    extr = torch.from_numpy(extr).float().to(device)
                                elif not isinstance(extr, torch.Tensor):
                                    extr = torch.eye(4, dtype=torch.float32, device=device)
                                else:
                                    extr = extr.to(device)
                            else:
                                extr = torch.eye(4, dtype=torch.float32, device=device)
                            tensor_extrinsics.append([extr] * num_cams)
                        extrinsics = tensor_extrinsics
                else:
                    # 相机列表，为每个批次复制
                    tensor_extrinsics = []
                    for i in range(B):
                        batch_tensor_extr = []
                        for j in range(min(len(extrinsics), num_cams)):
                            extr = extrinsics[j]
                            if extr is not None:
                                if isinstance(extr, np.ndarray):
                                    extr = torch.from_numpy(extr).float().to(device)
                                elif not isinstance(extr, torch.Tensor):
                                    extr = torch.eye(4, dtype=torch.float32, device=device)
                                else:
                                    extr = extr.to(device)
                            else:
                                extr = torch.eye(4, dtype=torch.float32, device=device)
                            batch_tensor_extr.append(extr)
                        # 如果相机数量不足，用默认值补充
                        while len(batch_tensor_extr) < num_cams:
                            batch_tensor_extr.append(torch.eye(4, dtype=torch.float32, device=device))
                        tensor_extrinsics.append(batch_tensor_extr)
                    extrinsics = tensor_extrinsics
            elif isinstance(extrinsics, torch.Tensor):
                # 张量格式，检查形状
                if extrinsics.dim() == 3 and extrinsics.shape[0] == B and extrinsics.shape[1] == 4 and extrinsics.shape[2] == 4:
                    # 单个相机的外参 (B, 4, 4)，为每个相机复制
                    extrinsics = [[extr.to(device)] * num_cams for extr in extrinsics]
                elif extrinsics.dim() == 4 and extrinsics.shape[0] == B and extrinsics.shape[1] == num_cams:
                    # 已经是正确的格式 (B, num_cams, 4, 4)
                    extrinsics = [[extrinsics[b, c].to(device) for c in range(num_cams)] for b in range(B)]
                else:
                    print(f"警告：外参形状不正确 {extrinsics.shape}，使用默认外参")
                    extrinsics = [[torch.eye(4, dtype=torch.float32, device=device) for _ in range(num_cams)] for _ in range(B)]
        else:
            print("警告：未提供外参，使用默认外参")
            extrinsics = [[torch.eye(4, dtype=torch.float32, device=device) for _ in range(num_cams)] for _ in range(B)]

        # reshape to process each view through backbone
        imgs = images.view(B * num_cams, *images.shape[2:])
        feats = self.backbone(imgs)  # (B*num_cams, Cfeat, Hf, Wf)

        # reshape back to (B, num_cams, ...)
        Cfeat = feats.shape[1]
        feats = feats.view(B, num_cams, Cfeat, feats.shape[-2], feats.shape[-1])

        # 将相机参数组织为张量 [B, N, ...] 以适配MVDetProjection
        try:
            # If intrinsics is a list of lists, stack inner lists, then outer list.
            if isinstance(intrinsics[0], list):
                intrinsics_tensor = torch.stack([torch.stack(batch_intr) for batch_intr in intrinsics]).to(device)
            # If intrinsics is a list of tensors, just stack the outer list.
            else:
                intrinsics_tensor = torch.stack(intrinsics).to(device)
        except Exception:
            # Fallback for other formats
            intrinsics_tensor = torch.as_tensor(intrinsics, device=device)
        try:
            extrinsics_tensor = torch.stack([torch.stack(batch_extr) for batch_extr in extrinsics]).to(device)
        except Exception:
            extrinsics_tensor = torch.as_tensor(extrinsics, device=device)

        # 将相机参数传递给投影模块，并传入原始图像尺寸
        img_h, img_w = images.shape[-2], images.shape[-1]
        bev_maps = self.proj(feats, intrinsics_tensor, extrinsics_tensor, img_size=(img_h, img_w))  # (B, num_cams, Cb, Hb, Wb)
        
        # 保存BEV特征图用于可视化
        self.bev_features = bev_maps.detach().clone()
        
        # 融合多视角特征（可配置：sum/mean/max）
        fused = self.fusion(bev_maps)  # (B, Cb, Hb, Wb)
        
        # 通过头部网络生成输出
        out = self.head(fused)  # (B, NumClasses, Hb, Wb)
        
        # 保存输出热图用于可视化
        self.output_heatmap = out.detach().clone()
        
        if return_bev:
            # also return the fused BEV feature map for inspection
            return out, fused
        return out
