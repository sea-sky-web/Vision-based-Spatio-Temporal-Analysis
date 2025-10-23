import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
try:
    import kornia.geometry.transform as KGT
    _HAS_KORNIA = True
except Exception:
    _HAS_KORNIA = False


class GeometryTransformer(nn.Module):
    def __init__(self, bev_h: int, bev_w: int, bev_bounds: tuple, warp_impl: str = 'grid_sample'):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bounds = bev_bounds  # (x_min,x_max,y_min,y_max)
        self.res_x = (bev_bounds[1] - bev_bounds[0]) / bev_w
        self.res_y = (bev_bounds[3] - bev_bounds[2]) / bev_h
        self.warp_impl = warp_impl if warp_impl in ('grid_sample', 'kornia') else 'grid_sample'
        self.register_buffer('ground_grid', self._create_ground_grid(), persistent=False)
        self._grid_cache = {}

    def _create_ground_grid(self):
        min_x, max_x, min_y, max_y = self.bounds
        xs = torch.linspace(min_x + 0.5 * self.res_x, max_x - 0.5 * self.res_x, self.bev_w)
        ys = torch.linspace(min_y + 0.5 * self.res_y, max_y - 0.5 * self.res_y, self.bev_h)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        ones = torch.ones_like(xx)
        grid = torch.stack([xx, yy, ones], dim=-1)  # [H,W,3]
        return grid

    @staticmethod
    def _compute_homography(K: torch.Tensor, Rt: torch.Tensor) -> torch.Tensor:
        if K.dim() != 2 or K.shape[0] < 3 or K.shape[1] < 3:
            device = K.device if hasattr(K, 'device') else torch.device('cpu')
            K = torch.eye(3, device=device)
            K[0, 0] = K[1, 1] = 1000.0
        else:
            K = K[:3, :3]
        if Rt.dim() == 2:
            if Rt.shape == (4, 4):
                R = Rt[:3, :3]
                t = Rt[:3, 3:4]
            elif Rt.shape == (3, 4):
                R = Rt[:, :3]
                t = Rt[:, 3:4]
            elif Rt.shape == (3, 3):
                R = Rt
                device = Rt.device
                t = torch.zeros(3, 1, device=device)
            else:
                device = Rt.device
                R = torch.eye(3, device=device)
                t = torch.zeros(3, 1, device=device)
        else:
            device = Rt.device if hasattr(Rt, 'device') else torch.device('cpu')
            R = torch.eye(3, device=device)
            t = torch.zeros(3, 1, device=device)
        r1 = R[:, 0:1]
        r2 = R[:, 1:2]
        G = torch.cat([r1, r2, t], dim=1)
        H = K @ G
        return H

    @staticmethod
    def _compute_img_to_world_homography(K: torch.Tensor, Rt: torch.Tensor) -> torch.Tensor:
        H_w2i = GeometryTransformer._compute_homography(K, Rt)
        try:
            det = torch.det(H_w2i)
        except Exception:
            det = torch.tensor(float('nan'), device=H_w2i.device)
        if torch.isnan(det) or torch.isinf(det) or det.abs().item() < 1e-8:
            return torch.linalg.pinv(H_w2i)
        try:
            return torch.linalg.inv(H_w2i)
        except Exception:
            return torch.linalg.pinv(H_w2i)

    def forward(self, feats: torch.Tensor,
                intrinsics,
                extrinsics,
                img_size: Tuple[int, int] = (1080, 1920)) -> torch.Tensor:
        """
        feats: Tensor[B, V, C, Hf, Wf]
        intrinsics: List[List[Tensor(3,3)]] or Tensor[B,V,3,3]
        extrinsics: List[List[Tensor(4,4)]] or Tensor[B,V,4,4]
        Returns: Tensor[B, V, C, H_bev, W_bev]
        """
        B, V, C, Hf, Wf = feats.shape
        device = feats.device
        H_img, W_img = img_size
        ground = self.ground_grid.to(device)
        bev_out = torch.zeros(B, V, C, self.bev_h, self.bev_w, device=device)

        def get_K(b, v):
            if isinstance(intrinsics, torch.Tensor):
                if intrinsics.dim() == 4:
                    return intrinsics[b, v]
                elif intrinsics.dim() == 3:
                    return intrinsics[v] if intrinsics.shape[0] == V else intrinsics[b]
                elif intrinsics.dim() == 2:
                    return intrinsics[:3, :3]
            else:
                return intrinsics[b][v]
            return torch.eye(3, device=device)

        def get_Rt(b, v):
            if isinstance(extrinsics, torch.Tensor):
                if extrinsics.dim() == 4:
                    return extrinsics[b, v]
                elif extrinsics.dim() == 3:
                    return extrinsics[v] if extrinsics.shape[0] == V else extrinsics[b]
                elif extrinsics.dim() == 2:
                    return extrinsics[:4, :4]
            else:
                return extrinsics[b][v]
            return torch.eye(4, device=device)

        for b in range(B):
            for v in range(V):
                K = get_K(b, v)
                Rt = get_Rt(b, v)
                if self.warp_impl == 'kornia' and _HAS_KORNIA:
                    H_i2w = self._compute_img_to_world_homography(K, Rt)
                    S_feat2img = torch.tensor([[W_img / float(Wf), 0.0, 0.0],
                                               [0.0, H_img / float(Hf), 0.0],
                                               [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
                    min_x, max_x, min_y, max_y = self.bounds
                    A_w2bev = torch.tensor([[1.0 / self.res_x, 0.0, -min_x / self.res_x],
                                            [0.0, 1.0 / self.res_y, -min_y / self.res_y],
                                            [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
                    M = A_w2bev @ H_i2w @ S_feat2img
                    detM = torch.det(M)
                    singular = torch.isnan(detM) or torch.isinf(detM) or detM.abs().item() < 1e-8
                    if not singular:
                        feat_in = feats[b, v].unsqueeze(0)
                        warped = KGT.warp_perspective(feat_in, M.unsqueeze(0), dsize=(self.bev_h, self.bev_w),
                                                       mode='bilinear', align_corners=False, padding_mode='zeros')
                        bev_out[b, v] = warped.squeeze(0)
                        continue
                # fallback to grid_sample
                H = self._compute_homography(K, Rt)
                g_flat = ground.reshape(-1, 3).T
                uvw = H @ g_flat
                w = uvw[2:3, :]
                w_safe = torch.where(w.abs() < 1e-6, torch.ones_like(w), w)
                u = uvw[0:1, :] / w_safe
                v_ = uvw[1:2, :] / w_safe
                img_pts = torch.stack([u.squeeze(0), v_.squeeze(0)], dim=1).reshape(self.bev_h, self.bev_w, 2)
                scale_w = Wf / float(W_img)
                scale_h = Hf / float(H_img)
                feat_pts = img_pts.clone()
                feat_pts[..., 0] = feat_pts[..., 0] * scale_w
                feat_pts[..., 1] = feat_pts[..., 1] * scale_h
                norm = feat_pts.clone()
                norm[..., 0] = (norm[..., 0] + 0.5) / Wf * 2.0 - 1.0
                norm[..., 1] = (norm[..., 1] + 0.5) / Hf * 2.0 - 1.0
                grid = norm.unsqueeze(0)
                feat_in = feats[b, v].unsqueeze(0)
                sampled = F.grid_sample(feat_in, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                bev_out[b, v] = sampled.squeeze(0)
        return bev_out