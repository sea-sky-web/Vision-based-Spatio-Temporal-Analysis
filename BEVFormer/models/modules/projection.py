"""Projection module that transforms feature maps to BEV using camera parameters."""
import torch
import torch.nn.functional as F
import numpy as np


class SimpleProjection:
    def __init__(self, bev_size=(1, 128, 128), bev_bounds=(-15, 15, -15, 15)):
        # bev_size: (C, H, W) - C will be inferred
        self.bev_h = bev_size[1]
        self.bev_w = bev_size[2]
        # BEV空间范围 (x_min, x_max, y_min, y_max) 单位：米 - 扩大到±15米
        self.bev_bounds = bev_bounds
        # 计算BEV网格分辨率
        self.x_res = (bev_bounds[1] - bev_bounds[0]) / self.bev_w
        self.y_res = (bev_bounds[3] - bev_bounds[2]) / self.bev_h
        # 创建BEV网格坐标
        self.create_bev_grid()
        
    def create_bev_grid(self):
        """创建BEV空间的网格坐标"""
        print(f"创建BEV网格，大小: {self.bev_h}x{self.bev_w}")
        print(f"BEV网格范围: x=[{self.bev_bounds[0]}, {self.bev_bounds[1]}], y=[{self.bev_bounds[2]}, {self.bev_bounds[3]}]")
        print(f"BEV网格分辨率: x_res={self.x_res:.4f}m, y_res={self.y_res:.4f}m")
        
        # 使用完整的BEV范围，不再缩小
        x_coords = torch.linspace(self.bev_bounds[0], self.bev_bounds[1], self.bev_w)
        y_coords = torch.linspace(self.bev_bounds[2], self.bev_bounds[3], self.bev_h)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 打印网格点的统计信息
        print(f"X网格统计: min={x_grid.min().item():.4f}, max={x_grid.max().item():.4f}")
        print(f"Y网格统计: min={y_grid.min().item():.4f}, max={y_grid.max().item():.4f}")
        
        # 使用固定的地面高度 z=0（MVDet在地面平面进行单应/投影）
        z_grid = torch.zeros_like(x_grid)
        # 堆叠为齐次坐标 [H, W, 4]
        self.bev_grid = torch.stack([x_grid, y_grid, z_grid, torch.ones_like(x_grid)], dim=-1)
        
        # 打印BEV网格的形状和示例点
        print(f"BEV网格创建完成，形状: {self.bev_grid.shape}，总元素数: {self.bev_grid.numel()}")
        print(f"BEV网格示例点(左上角):\n{self.bev_grid[0, 0]}")
        print(f"BEV网格示例点(右下角):\n{self.bev_grid[-1, -1]}")
        print(f"BEV网格示例点(中心):\n{self.bev_grid[self.bev_h//2, self.bev_w//2]}")
        
        # 验证网格大小是否符合预期
        assert self.bev_grid.shape == (self.bev_h, self.bev_w, 4), f"BEV网格形状错误: {self.bev_grid.shape} != {(self.bev_h, self.bev_w, 4)}"

    def project_points_to_image(self, points_3d, intrinsic, extrinsic):
        """将3D点投影到图像平面
        
        Args:
            points_3d (torch.Tensor): 形状为 [..., 4] 的3D点，每个点是齐次坐标 (x, y, z, 1)
            intrinsic (torch.Tensor): 形状为 [3, 3] 的相机内参矩阵
            extrinsic (torch.Tensor): 形状为 [4, 4] 的相机外参矩阵（从世界坐标系到相机坐标系）
            
        Returns:
            tuple: (points_2d, depth)
                points_2d (torch.Tensor): 形状为 [..., 2] 的2D点，每个点是像素坐标 (u, v)
                depth (torch.Tensor): 形状为 [..., 1] 的深度值
        """
        # 打印相机参数，用于调试
        print(f"内参矩阵:\n{intrinsic}")
        print(f"外参矩阵:\n{extrinsic}")
        print(f"3D点形状: {points_3d.shape}, 示例点:\n{points_3d[:5]}")
        
        # 确保内参矩阵是3x3
        if intrinsic.shape != (3, 3):
            print(f"警告: 内参矩阵形状 {intrinsic.shape} 不是 (3, 3)，尝试调整")
            if intrinsic.shape[0] >= 3 and intrinsic.shape[1] >= 3:
                intrinsic = intrinsic[:3, :3]
            else:
                # 创建默认内参
                print("创建默认内参矩阵")
                intrinsic = torch.eye(3, device=intrinsic.device)
                intrinsic[0, 0] = intrinsic[1, 1] = 1000.0  # 默认焦距
                intrinsic[0, 2] = 640.0  # 主点x (假设图像宽度1280)
                intrinsic[1, 2] = 360.0  # 主点y (假设图像高度720)
        
        # 确保外参矩阵是4x4
        if extrinsic.shape != (4, 4):
            print(f"警告: 外参矩阵形状 {extrinsic.shape} 不是 (4, 4)，创建默认外参")
            extrinsic = torch.eye(4, device=points_3d.device)
            # 设置相机位置在(0,0,2)，朝向z轴负方向
            extrinsic[2, 3] = 2.0
            extrinsic[2, 2] = -1.0
            extrinsic[0, 0] = -1.0
        
        # 检查输入点是否有无效值
        if torch.isnan(points_3d).any() or torch.isinf(points_3d).any():
            print("警告: 3D点包含NaN或Inf值，已替换为0")
            points_3d = torch.nan_to_num(points_3d, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 将世界坐标系中的点转换到相机坐标系
        # 使用更明确的变换方式，确保正确性
        R = extrinsic[:3, :3]  # 旋转矩阵
        t = extrinsic[:3, 3]   # 平移向量
        
        # 提取3D点的非齐次部分
        points_3d_xyz = points_3d[..., :3]
        
        # 应用旋转和平移
        points_cam = torch.matmul(points_3d_xyz, R.transpose(-1, -2)) + t
        
        print(f"相机坐标系中的点形状: {points_cam.shape}, 示例点:\n{points_cam[:5]}")
        
        # 检查相机坐标系中的点是否有无效值
        if torch.isnan(points_cam).any() or torch.isinf(points_cam).any():
            print("警告: 相机坐标系中的点包含NaN或Inf值，已替换为0")
            points_cam = torch.nan_to_num(points_cam, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 提取相机坐标系中的点的深度值（z坐标）
        depth = points_cam[..., 2:3]
        
        # 检查深度值的有效性 - 使用更合理的深度范围
        valid_depth = (depth > 0.1) & (depth < 50)  # 深度在0.1到50米之间
        valid_depth_count = valid_depth.sum().item()
        total_depth_count = depth.numel()
        print(f"相机坐标系中有效深度点数量: {valid_depth_count}/{total_depth_count} ({valid_depth_count/total_depth_count*100:.2f}%)")
        print(f"相机坐标系中深度值范围: min={depth.min().item():.4f}, max={depth.max().item():.4f}, mean={depth.mean().item():.4f}")
        
        # 记录有效深度点比例，但不添加调试点
        if valid_depth_count / total_depth_count < 0.1:
            print(f"警告: 有效深度点比例过低 ({valid_depth_count/total_depth_count*100:.2f}%)")
        
        # 初始化输出张量
        points_2d = torch.zeros_like(points_cam[..., :2])
        
        # 只对相机前方且在合理深度范围内的点进行投影
        valid_mask = (depth > 0.1) & (depth < 50)
        valid_indices = valid_mask.squeeze(-1)
        
        if valid_indices.sum() > 0:
            # 提取有效点
            valid_cam_points = points_cam[valid_indices]
            
            # 直接从内参矩阵获取参数，确保转换为标量
            try:
                fx = float(intrinsic[0, 0])
                fy = float(intrinsic[1, 1])
                cx = float(intrinsic[0, 2])
                cy = float(intrinsic[1, 2])
            except Exception as e:
                print(f"警告: 从内参矩阵提取参数失败: {e}")
                print(f"内参矩阵类型: {type(intrinsic)}, 形状: {intrinsic.shape if hasattr(intrinsic, 'shape') else 'unknown'}")
                # 使用默认值
                fx = fy = 1000.0
                cx = cy = 500.0
            
            # 计算像素坐标
            x = valid_cam_points[:, 0]
            y = valid_cam_points[:, 1]
            z = valid_cam_points[:, 2]
            
            # 防止除零
            z = torch.clamp(z, min=1e-6)
            
            # 应用透视投影公式 - 使用标量进行计算，避免广播问题
            u = fx * x / z + cx
            v = fy * y / z + cy
            
            # 组合成像素坐标
            valid_points_2d = torch.stack([u, v], dim=-1)
            
            # 将结果放回原始张量
            points_2d[valid_indices] = valid_points_2d
        
        print(f"投影后的2D点形状: {points_2d.shape}, 示例点:\n{points_2d[:5]}")
        print(f"投影点范围: x=[{points_2d[..., 0].min().item():.1f}, {points_2d[..., 0].max().item():.1f}], y=[{points_2d[..., 1].min().item():.1f}, {points_2d[..., 1].max().item():.1f}]")
        
        return points_2d, depth

    def __call__(self, feats, intrinsics=None, extrinsics=None):
        # feats: (B, NumCams, Cfeat, Hf, Wf)
        B, num_cams, C, H, W = feats.shape
        device = feats.device
        
        # 打印输入特征统计信息
        print(f"输入特征形状: {feats.shape}")
        print(f"输入特征统计: min={feats.min().item():.4f}, max={feats.max().item():.4f}, mean={feats.mean().item():.4f}")
        print(f"输入特征非零元素比例: {(feats != 0).float().mean().item():.4f}")
        
        # 调试相机参数的结构
        if intrinsics is not None:
            print(f"内参类型: {type(intrinsics)}")
            if isinstance(intrinsics, list):
                print(f"内参列表长度: {len(intrinsics)}")
                if len(intrinsics) > 0:
                    print(f"内参第一个元素类型: {type(intrinsics[0])}")
                    if isinstance(intrinsics[0], list):
                        print(f"内参第一个批次长度: {len(intrinsics[0])}")
                    elif hasattr(intrinsics[0], 'shape'):
                        print(f"内参第一个元素形状: {intrinsics[0].shape}")
            else:
                print(f"内参张量形状: {intrinsics.shape}")
        
        if extrinsics is not None:
            print(f"外参类型: {type(extrinsics)}")
            if isinstance(extrinsics, list):
                print(f"外参列表长度: {len(extrinsics)}")
                if len(extrinsics) > 0:
                    print(f"外参第一个元素类型: {type(extrinsics[0])}")
                    if isinstance(extrinsics[0], list):
                        print(f"外参第一个批次长度: {len(extrinsics[0])}")
                    elif hasattr(extrinsics[0], 'shape'):
                        print(f"外参第一个元素形状: {extrinsics[0].shape}")
            else:
                print(f"外参张量形状: {extrinsics.shape}")
        
        print(f"特征张量相机数量: {num_cams}")
        print(f"批次大小: {B}")
        
        # 如果没有提供相机参数，则回退到简单的插值方法
        if intrinsics is None or extrinsics is None:
            print("警告：未提供相机参数，使用简单插值进行投影")
            out = torch.zeros((B, num_cams, C, self.bev_h, self.bev_w), dtype=feats.dtype, device=device)
            for b in range(B):
                for c in range(num_cams):
                    out[b, c] = F.interpolate(feats[b, c].unsqueeze(0), size=(self.bev_h, self.bev_w), 
                                              mode='bilinear', align_corners=False).squeeze(0)
            return out
        
        # 将BEV网格移到正确的设备上
        bev_grid = self.bev_grid.to(device)
        
        # 初始化输出特征图
        out = torch.zeros((B, num_cams, C, self.bev_h, self.bev_w), dtype=feats.dtype, device=device)
        
        # 确定实际可处理的相机数量 - 始终使用特征张量的相机数量
        actual_num_cams = num_cams  # 使用特征张量的相机数量
            
        print(f"实际处理的相机数量: {actual_num_cams}")
        
        # 对每个批次和每个相机进行处理
        for b in range(B):
            for c in range(actual_num_cams):
                print(f"\n处理 batch {b}, camera {c}:")
                # 获取当前相机的内外参
                try:
                    if isinstance(intrinsics, list):
                        # 嵌套列表结构: intrinsics[batch][camera]
                        if b < len(intrinsics) and c < len(intrinsics[b]):
                            K = intrinsics[b][c]
                            if hasattr(K, 'to'):
                                K = K.to(device)
                            else:
                                K = torch.tensor(K, dtype=torch.float32, device=device)
                        else:
                            print(f"警告: 相机索引 [{b}][{c}] 超出内参列表范围")
                            continue
                    else:
                        # 张量形式，可能是堆叠的相机参数
                        if len(intrinsics.shape) == 3:  # [num_cams, 3, 3]
                            # 每个相机使用自身索引的参数
                            K = intrinsics[c].to(device)
                        else:  # [batch, num_cams, 3, 3]
                            K = intrinsics[b, c].to(device)
                        
                    if isinstance(extrinsics, list):
                        # 嵌套列表结构: extrinsics[batch][camera]
                        if b < len(extrinsics) and c < len(extrinsics[b]):
                            Rt = extrinsics[b][c]
                            if hasattr(Rt, 'to'):
                                Rt = Rt.to(device)
                            else:
                                Rt = torch.tensor(Rt, dtype=torch.float32, device=device)
                        else:
                            print(f"警告: 相机索引 [{b}][{c}] 超出外参列表范围")
                            continue
                    else:
                        # 张量形式，可能是堆叠的相机参数
                        if len(extrinsics.shape) == 3:  # [num_cams, 4, 4]
                            # 每个相机使用自身索引的参数
                            Rt = extrinsics[c].to(device)
                        else:  # [batch, num_cams, 4, 4]
                            Rt = extrinsics[b, c].to(device)
                except (IndexError, TypeError) as e:
                    print(f"警告: 无法获取相机 {c} 的参数: {e}")
                    # 跳过这个相机或使用默认参数
                    continue
                
                # 打印当前处理的相机参数
                print(f"相机 {c} 内参:\n{K}")
                print(f"相机 {c} 外参:\n{Rt}")
                
                # 计算单应矩阵（世界坐标系到图像坐标系的变换）
                # 确保K是3x3矩阵
                if K.shape[0] != 3 or K.shape[1] != 3:
                    print(f"警告: 内参矩阵K形状不正确: {K.shape}，应为(3,3)")
                    if K.shape[0] >= 3 and K.shape[1] >= 3:
                        K = K[:3, :3]
                        print(f"使用K的3x3部分:\n{K}")
                    else:
                        print("创建默认内参矩阵")
                        K = torch.eye(3, device=K.device)
                        K[0, 0] = K[1, 1] = 1000.0
                        K[0, 2] = W / 2.0
                        K[1, 2] = H / 2.0
                
                # 确保Rt形状正确
                if Rt.shape[0] != 4 or Rt.shape[1] != 4:
                    print(f"警告: 外参矩阵Rt形状不正确: {Rt.shape}，应为(4,4)")
                    Rt = torch.eye(4, device=Rt.device)
                    Rt[2, 3] = 2.0
                    Rt[2, 2] = -1.0
                    Rt[0, 0] = -1.0
                
                # Ground-plane homography: H = K [r1, r2, t]
                R = Rt[:3, :3]
                t = Rt[:3, 3:4]
                G = torch.cat([R[:, 0:1], R[:, 1:2], t], dim=1)  # [3,3]
                H_gp = K @ G
                print(f"单应矩阵(H=K@[r1,r2,t]):\n{H_gp}")
                
                # 将BEV网格点投影到图像平面（z=0 → 使用地面平面单应）
                flattened_grid = bev_grid.reshape(-1, 4)  # [H*W, 4]
                ground_xy1 = flattened_grid[:, [0, 1, 3]].T  # [3, H*W]
                uvw = H_gp @ ground_xy1  # [3, H*W]
                w = uvw[2:3, :]
                w_safe = torch.where(w.abs() < 1e-6, torch.ones_like(w), w)
                u = (uvw[0:1, :] / w_safe).squeeze(0)
                v = (uvw[1:2, :] / w_safe).squeeze(0)
                img_points = torch.stack([u, v], dim=1)  # [H*W, 2]
                
                # 深度(z_cam)用于有效性判定：z_cam = r3x*X + r3y*Y + t_z
                r3 = R[:, 2]
                depths = (flattened_grid[:, 0] * r3[0] + flattened_grid[:, 1] * r3[1] + t[2, 0]).unsqueeze(1)
                print(f"投影后点的形状: {img_points.shape}, 深度形状: {depths.shape}")
                
                # 重塑为原始网格形状
                img_points = img_points.reshape(self.bev_h, self.bev_w, 2)
                depths = depths.reshape(self.bev_h, self.bev_w, 1)
                
                # 有效性检查（在图像范围内且深度合理）
                in_image = ((img_points[..., 0] >= 0) & (img_points[..., 0] < W) & 
                            (img_points[..., 1] >= 0) & (img_points[..., 1] < H))
                valid_depth = (depths > 0.1) & (depths < 50)
                valid_points = in_image & valid_depth.squeeze(-1)
                valid_points_ratio = valid_points.float().mean().item()
                print(f"综合有效点比例: {valid_points_ratio:.4f} ({valid_points.sum().item()}/{valid_points.numel()})")
                
                # 归一化到[-1, 1]（align_corners=False：像素中心对齐）
                norm_points = torch.zeros_like(img_points)
                norm_points[..., 0] = (img_points[..., 0] + 0.5) / W * 2.0 - 1.0
                norm_points[..., 1] = (img_points[..., 1] + 0.5) / H * 2.0 - 1.0
                invalid_coords = torch.isnan(norm_points).sum().item() + torch.isinf(norm_points).sum().item()
                if invalid_coords > 0:
                    print(f"警告: 检测到 {invalid_coords} 个无效坐标点 (NaN或Inf)")
                    norm_points = torch.nan_to_num(norm_points, nan=0.0, posinf=1.0, neginf=-1.0)
                
                grid = norm_points.unsqueeze(0)  # [1, H, W, 2]
                
                # 采样特征（统一为 align_corners=False）
                sampled_feat = F.grid_sample(
                    feats[b, c].unsqueeze(0),  # [1, C, H, W]
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False
                ).squeeze(0)  # [C, H, W]
                
                # 保持几何一致性（移除无效注释与占位符）
        # 投影处理结束
