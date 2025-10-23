"""
正确的MVDet风格透视变换实现
基于论文: Multiview Detection with Feature Perspective Transformation (ECCV 2020)
"""

import torch
import torch.nn.functional as F
import numpy as np


class MVDetProjection:
    """
    MVDet风格的透视变换投影模块
    
    核心思路：
    1. 在BEV网格的每个位置，定义一个垂直的采样线（从地面到一定高度）
    2. 将这些3D采样点投影到各个相机图像中
    3. 从图像特征图中采样对应的特征向量
    4. 在BEV空间聚合多视角特征
    """
    
    def __init__(self, bev_size=(200, 200), bev_bounds=(-7.7, 7.8, -2.0, 1.7), 
                 height_range=(0.0, 2.0), num_height_samples=8, height_aggregation='max'):
        """
        Args:
            bev_size: BEV网格大小 (H, W)
            bev_bounds: BEV空间范围 (x_min, x_max, y_min, y_max) 单位：米
            height_range: 采样高度范围 (z_min, z_max) 单位：米
            num_height_samples: 垂直方向采样点数量
        """
        self.bev_h, self.bev_w = bev_size
        self.bev_bounds = bev_bounds
        self.height_range = height_range
        self.num_height_samples = num_height_samples
        self.height_aggregation = height_aggregation
        
        # 计算BEV分辨率
        self.x_res = (bev_bounds[1] - bev_bounds[0]) / self.bev_w
        self.y_res = (bev_bounds[3] - bev_bounds[2]) / self.bev_h
        
        print(f"MVDet投影模块初始化:")
        print(f"  BEV大小: {self.bev_h}x{self.bev_w}")
        print(f"  BEV范围: x=[{bev_bounds[0]}, {bev_bounds[1]}], y=[{bev_bounds[2]}, {bev_bounds[3]}]")
        print(f"  BEV分辨率: {self.x_res:.3f}m x {self.y_res:.3f}m")
        print(f"  高度范围: [{height_range[0]}, {height_range[1]}]m, 采样点数: {num_height_samples}")
        
        # 创建BEV采样网格
        self.create_sampling_grid()
    
    def create_sampling_grid(self):
        """创建BEV空间的3D采样网格"""
        # 创建BEV平面网格
        x_coords = torch.linspace(self.bev_bounds[0], self.bev_bounds[1], self.bev_w)
        y_coords = torch.linspace(self.bev_bounds[2], self.bev_bounds[3], self.bev_h)
        
        # 创建高度采样点
        z_coords = torch.linspace(self.height_range[0], self.height_range[1], self.num_height_samples)
        
        # 创建3D网格 [H, W, num_height_samples, 3]
        y_grid, x_grid, z_grid = torch.meshgrid(y_coords, x_coords, z_coords, indexing='ij')
        
        # 组合成3D坐标 [H, W, num_height_samples, 3] (x, y, z)
        self.sampling_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)
        
        print(f"3D采样网格创建完成，形状: {self.sampling_grid.shape}")
        print(f"总采样点数: {self.sampling_grid.numel() // 3}")
    
    def project_to_camera(self, points_3d, intrinsic, extrinsic):
        """
        将3D点投影到相机图像平面
        
        Args:
            points_3d: 3D点坐标 [..., 3] (x, y, z)
            intrinsic: 相机内参矩阵 [3, 3]
            extrinsic: 相机外参矩阵 [4, 4] (世界坐标到相机坐标的变换)
            
        Returns:
            points_2d: 图像坐标 [..., 2] (u, v)
            depths: 深度值 [..., 1]
            valid_mask: 有效点掩码 [...]
        """
        original_shape = points_3d.shape[:-1]
        points_flat = points_3d.reshape(-1, 3)

        ones = torch.ones(points_flat.shape[0], 1, device=points_flat.device)
        points_homo = torch.cat([points_flat, ones], dim=1)  # [N, 4]

        # 兼容 3x4 外参（自动扩展到 4x4）
        if extrinsic.shape[-2:] == (3, 4):
            pad_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device, dtype=extrinsic.dtype)
            extrinsic = torch.cat([extrinsic, pad_row.view(1, 4)], dim=0)

        # 世界 -> 相机
        points_cam_homo = torch.matmul(extrinsic, points_homo.transpose(-1, -2)).transpose(-1, -2)  # [N, 4]
        points_cam = points_cam_homo[:, :3]  # [N, 3]

        depths = points_cam[:, 2:3]
        valid_depth = depths[:, 0] > 1e-6  # 更稳妥的阈值

        # 使用完整 K 进行透视投影（支持斜切以及非零副对角）
        # p_img = K @ p_cam ；u = p_img[0]/p_img[2], v = p_img[1]/p_img[2]
        points_2d = torch.zeros(points_flat.shape[0], 2, device=points_flat.device, dtype=points_cam.dtype)
        if valid_depth.any():
            proj = torch.matmul(intrinsic, points_cam.transpose(0, 1)).transpose(0, 1)  # [N, 3]
            u = proj[valid_depth, 0] / proj[valid_depth, 2]
            v = proj[valid_depth, 1] / proj[valid_depth, 2]
            points_2d[valid_depth] = torch.stack([u, v], dim=1)

        points_2d = points_2d.reshape(*original_shape, 2)
        depths = depths.reshape(*original_shape, 1)
        valid_mask = valid_depth.reshape(original_shape)
        return points_2d, depths, valid_mask
    
    def sample_features_from_image(self, feature_map, points_2d, valid_mask, img_h, img_w):
        """
        从图像特征图中采样特征
        
        Args:
            feature_map: 图像特征图 [C, H_feat, W_feat]
            points_2d: 图像坐标 [..., 2]
            valid_mask: 有效点掩码 [...]
            img_h, img_w: 原始图像尺寸
            
        Returns:
            sampled_features: 采样的特征 [..., C]
        """
        C, H_feat, W_feat = feature_map.shape
        
        # 计算特征图的缩放比例
        scale_h = H_feat / img_h
        scale_w = W_feat / img_w
        
        # 将图像坐标转换为特征图坐标
        feat_coords = points_2d.clone()
        feat_coords[..., 0] *= scale_w  # u坐标
        feat_coords[..., 1] *= scale_h  # v坐标
        
        # 归一化到[-1, 1]范围（grid_sample要求, align_corners=False像素中心对齐）
        norm_coords = feat_coords.clone()
        norm_coords[..., 0] = (feat_coords[..., 0] + 0.5) / W_feat * 2.0 - 1.0
        norm_coords[..., 1] = (feat_coords[..., 1] + 0.5) / H_feat * 2.0 - 1.0
        
        # 检查坐标是否在有效范围内
        in_bounds = (
            (norm_coords[..., 0] >= -1.0) & (norm_coords[..., 0] <= 1.0) &
            (norm_coords[..., 1] >= -1.0) & (norm_coords[..., 1] <= 1.0)
        )
        
        # 组合有效性掩码
        final_valid = valid_mask & in_bounds
        
        # 准备采样坐标
        original_shape = norm_coords.shape[:-1]
        flat_coords = norm_coords.reshape(-1, 2)
        
        # 使用grid_sample进行双线性插值采样
        # grid_sample需要 [1, C, H, W] 和 [1, H_out, W_out, 2]
        grid = flat_coords.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        feat_input = feature_map.unsqueeze(0)  # [1, C, H_feat, W_feat]
        
        sampled = F.grid_sample(feat_input, grid, mode='bilinear', 
                               padding_mode='zeros', align_corners=False)
        
        # 重塑输出 [1, C, 1, N] -> [N, C] -> [..., C]
        sampled = sampled.squeeze(0).squeeze(1).T  # [N, C]
        sampled = sampled.reshape(*original_shape, C)
        
        # 将无效位置的特征设为0
        sampled[~final_valid] = 0.0
        
        return sampled, final_valid
    
    def aggregate_height_features(self, height_features, valid_masks):
        """
        沿高度维度聚合特征
        
        Args:
            height_features: 高度采样特征 [H, W, num_height_samples, C]
            valid_masks: 有效性掩码 [H, W, num_height_samples]
            
        Returns:
            aggregated: 聚合后的特征 [H, W, C]
        """
        if self.height_aggregation == 'max':
            aggregated = height_features.max(dim=2).values
            return aggregated
        else:
            # 加权平均聚合（基于有效点数量）
            weights = valid_masks.float()
            total_weights = weights.sum(dim=2, keepdim=True)
            weights = torch.where(total_weights > 0, weights / total_weights,
                                  torch.ones_like(weights) / self.num_height_samples)
            aggregated = (height_features * weights.unsqueeze(-1)).sum(dim=2)
            return aggregated
    
    def __call__(self, features, intrinsics, extrinsics, img_size=(1080, 1920)):
        """
        执行MVDet风格的透视变换投影
        
        Args:
            features: 多视角特征图 [B, N, C, H_feat, W_feat]
            intrinsics: 内参矩阵 [B, N, 3, 3] 或 [N, 3, 3]
            extrinsics: 外参矩阵 [B, N, 4, 4] 或 [N, 4, 4]
            img_size: 原始图像尺寸 (H, W)
            
        Returns:
            bev_features: BEV特征图 [B, N, C, H_bev, W_bev]
        """
        B, N, C, H_feat, W_feat = features.shape
        device = features.device
        img_h, img_w = img_size
        
        print(f"[MVDet投影] 输入特征: {features.shape}")
        
        # 将采样网格移到正确设备
        sampling_grid = self.sampling_grid.to(device)  # [H, W, num_height_samples, 3]
        
        # 初始化输出
        bev_features = torch.zeros(B, N, C, self.bev_h, self.bev_w, device=device)
        
        for b in range(B):
            for n in range(N):
                # 获取当前相机的参数
                if intrinsics.dim() == 4:  # [B, N, 3, 3]
                    K = intrinsics[b, n]
                    Rt = extrinsics[b, n]
                else:  # [N, 3, 3]
                    K = intrinsics[n]
                    Rt = extrinsics[n]
                
                # 投影3D采样点到当前相机
                points_2d, depths, valid_mask = self.project_to_camera(
                    sampling_grid, K, Rt
                )
                
                # 从特征图中采样
                feat_map = features[b, n]  # [C, H_feat, W_feat]
                height_features, sample_valid = self.sample_features_from_image(
                    feat_map, points_2d, valid_mask, img_h, img_w
                )
                
                # 沿高度维度聚合特征
                aggregated_features = self.aggregate_height_features(
                    height_features, sample_valid
                )
                
                # 存储结果
                bev_features[b, n] = aggregated_features.permute(2, 0, 1)  # [C, H, W]
                
                # 统计信息
                valid_ratio = sample_valid.float().mean().item()
                feature_mean = aggregated_features.mean().item()
                print(f"  相机{n}: 有效采样比例={valid_ratio:.3f}, 特征均值={feature_mean:.4f}")
        
        print(f"[MVDet投影] 输出BEV特征: {bev_features.shape}")
        return bev_features