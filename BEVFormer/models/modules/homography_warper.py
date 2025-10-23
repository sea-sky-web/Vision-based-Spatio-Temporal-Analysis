"""Homography-based warping from ground plane (Z=0) to image using camera parameters.

Implements H = K [r1, r2, t] to map ground-plane world coordinates (X, Y, 1)
to image pixels (u, v, w). Supports two warping implementations:
- 'grid_sample': sample per-cell via computed pixel coordinates
- 'kornia': build a single perspective transform and warp feature maps
"""
import torch
import torch.nn.functional as F
import kornia.geometry.transform as KGT


class HomographyWarper:
    def __init__(self, bev_size=(1, 128, 128), bev_bounds=(-10, 10, -10, 10), warp_impl: str = 'grid_sample'):
        # bev_size: (C, H, W)
        self.bev_h = bev_size[1]
        self.bev_w = bev_size[2]
        self.bounds = bev_bounds  # (min_x, max_x, min_y, max_y) in meters
        # resolution (meters per pixel)
        self.res_x = (bev_bounds[1] - bev_bounds[0]) / self.bev_w
        self.res_y = (bev_bounds[3] - bev_bounds[2]) / self.bev_h
        # warp implementation: 'grid_sample' or 'kornia'
        self.warp_impl = warp_impl
        # precompute ground plane grid [H, W, 3] with homogeneous last dim 1
        self._create_ground_grid()

    def _create_ground_grid(self):
        min_x, max_x, min_y, max_y = self.bounds
        xs = torch.linspace(min_x + 0.5 * self.res_x, max_x - 0.5 * self.res_x, self.bev_w)
        ys = torch.linspace(min_y + 0.5 * self.res_y, max_y - 0.5 * self.res_y, self.bev_h)
        # meshgrid produces (H, W)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        ones = torch.ones_like(xx)
        grid = torch.stack([xx, yy, ones], dim=-1)  # [H, W, 3]
        self.ground_grid = grid  # CPU tensor by default

    @staticmethod
    def _compute_homography(K: torch.Tensor, Rt: torch.Tensor) -> torch.Tensor:
        """Compute H = K [r1, r2, t] for ground plane Z=0.

        Args:
            K: [3,3] intrinsic
            Rt: [4,4] world->camera extrinsic
        Returns:
            H: [3,3] homography mapping ground plane (X,Y,1) to image homogeneous coords
        """
        # Normalize K shape to (3,3)
        if K.dim() != 2 or K.shape[0] < 3 or K.shape[1] < 3:
            # Create a reasonable default intrinsic if malformed
            device = K.device if hasattr(K, 'device') else torch.device('cpu')
            K = torch.eye(3, device=device)
            K[0, 0] = K[1, 1] = 1000.0
        else:
            K = K[:3, :3]

        # Normalize Rt to contain rotation and translation
        if Rt.dim() == 2:
            if Rt.shape == (4, 4):
                R = Rt[:3, :3]
                t = Rt[:3, 3:4]
            elif Rt.shape == (3, 4):
                # Provided as [R|t]
                R = Rt[:, :3]
                t = Rt[:, 3:4]
            elif Rt.shape == (3, 3):
                # No translation provided; assume zero translation
                R = Rt
                device = Rt.device
                t = torch.zeros(3, 1, device=device)
            else:
                # Fallback to identity pose
                device = Rt.device
                R = torch.eye(3, device=device)
                t = torch.zeros(3, 1, device=device)
        else:
            # Unexpected dims, fallback
            device = Rt.device if hasattr(Rt, 'device') else torch.device('cpu')
            R = torch.eye(3, device=device)
            t = torch.zeros(3, 1, device=device)

        r1 = R[:, 0:1]  # [3,1]
        r2 = R[:, 1:2]  # [3,1]
        G = torch.cat([r1, r2, t], dim=1)  # [3,3]
        H = K @ G  # [3,3]
        return H

    @staticmethod
    def _compute_img_to_world_homography(K: torch.Tensor, Rt: torch.Tensor) -> torch.Tensor:
        """Compute H_img2world as inverse of H_world2img for ground plane.

        Args:
            K: [3,3]
            Rt: [4,4] (world->camera)
        Returns:
            H_i2w: [3,3]
        """
        H_w2i = HomographyWarper._compute_homography(K, Rt)
        # Robust inversion: try inverse, otherwise fallback to pseudo-inverse
        try:
            det = torch.det(H_w2i)
        except Exception:
            det = torch.tensor(float('nan'), device=H_w2i.device)
        if torch.isnan(det) or torch.isinf(det) or det.abs().item() < 1e-8:
            # Singular or ill-conditioned, use pseudo-inverse
            H_i2w = torch.linalg.pinv(H_w2i)
        else:
            try:
                H_i2w = torch.linalg.inv(H_w2i)
            except Exception:
                H_i2w = torch.linalg.pinv(H_w2i)
        return H_i2w

    def __call__(self, features: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor, img_size=(1080, 1920)):
        """Warp multi-view features into BEV using ground-plane homography.

        Args:
            features: [B, N, C, Hf, Wf]
            intrinsics: [B, N, 3, 3] or [N, 3, 3]
            extrinsics: [B, N, 4, 4] or [N, 4, 4]
            img_size: original image size (H_img, W_img)

        Returns:
            bev_features: [B, N, C, H_bev, W_bev]
        """
        B, N, C, Hf, Wf = features.shape
        device = features.device
        H_img, W_img = img_size

        ground = self.ground_grid.to(device)  # [H, W, 3]
        bev_out = torch.zeros(B, N, C, self.bev_h, self.bev_w, device=device)

        for b in range(B):
            for n in range(N):
                # 选择内参K
                if intrinsics.dim() == 4:
                    K = intrinsics[b, n]
                elif intrinsics.dim() == 3:
                    if intrinsics.shape[0] == N:  # [N, 3, 3]
                        K = intrinsics[n]
                    elif intrinsics.shape[0] == B:  # [B, 3, 3]
                        K = intrinsics[b]
                    else:
                        # 退化为使用前3x3
                        K = intrinsics[:3, :3]
                elif intrinsics.dim() == 2:
                    K = intrinsics[:3, :3]
                else:
                    K = torch.eye(3, device=device)

                # 选择外参Rt
                if extrinsics.dim() == 4:
                    Rt = extrinsics[b, n]
                elif extrinsics.dim() == 3:
                    if extrinsics.shape[0] == N:  # [N, 4, 4]
                        Rt = extrinsics[n]
                    elif extrinsics.shape[0] == B:  # [B, 4, 4]
                        Rt = extrinsics[b]
                    else:
                        # 若为 [4,4] 展开错误，取前两个维度
                        Rt = extrinsics[:4, :4]
                elif extrinsics.dim() == 2:
                    Rt = extrinsics[:4, :4]
                else:
                    Rt = torch.eye(4, device=device)

                if self.warp_impl == 'kornia':
                    # Build transform from input feature map pixel coords -> output BEV pixel coords
                    # Step 1: img->world homography (ground plane)
                    H_i2w = self._compute_img_to_world_homography(K, Rt)  # [3,3]

                    # Step 2: feature-map pixels -> image pixels scaling
                    S_feat2img = torch.tensor([
                        [W_img / float(Wf), 0.0, 0.0],
                        [0.0, H_img / float(Hf), 0.0],
                        [0.0, 0.0, 1.0]
                    ], dtype=torch.float32, device=device)  # [3,3]

                    # Step 3: world meters -> BEV pixel indices affine
                    min_x, max_x, min_y, max_y = self.bounds
                    A_w2bev = torch.tensor([
                        [1.0 / self.res_x, 0.0, -min_x / self.res_x],
                        [0.0, 1.0 / self.res_y, -min_y / self.res_y],
                        [0.0, 0.0, 1.0]
                    ], dtype=torch.float32, device=device)  # [3,3]

                    # Compose transform: src(feature) -> dst(BEV)
                    M = A_w2bev @ H_i2w @ S_feat2img  # [3,3]

                    # If M is singular/ill-conditioned, fallback to grid_sample path
                    try:
                        detM = torch.det(M)
                        singular = torch.isnan(detM) or torch.isinf(detM) or detM.abs().item() < 1e-8
                    except Exception:
                        singular = True

                    if not singular:
                        feat_in = features[b, n].unsqueeze(0)  # [1, C, Hf, Wf]
                        warped = KGT.warp_perspective(
                            feat_in, M.unsqueeze(0), dsize=(self.bev_h, self.bev_w),
                            mode='bilinear', align_corners=False, padding_mode='zeros'
                        )  # [1, C, H_bev, W_bev]
                        bev_out[b, n] = warped.squeeze(0)
                    else:
                        # Fallback: use direct projection and grid_sample
                        H = self._compute_homography(K, Rt)  # [3,3]
                        g_flat = ground.reshape(-1, 3).T  # [3, H*W]
                        uvw = H @ g_flat  # [3, H*W]
                        w = uvw[2:3, :]  # [1, H*W]
                        w_safe = torch.where(w.abs() < 1e-6, torch.ones_like(w), w)
                        u = uvw[0:1, :] / w_safe
                        v = uvw[1:2, :] / w_safe
                        img_pts = torch.stack([u.squeeze(0), v.squeeze(0)], dim=1).reshape(self.bev_h, self.bev_w, 2)

                        scale_w = Wf / float(W_img)
                        scale_h = Hf / float(H_img)
                        feat_pts = img_pts.clone()
                        feat_pts[..., 0] = feat_pts[..., 0] * scale_w
                        feat_pts[..., 1] = feat_pts[..., 1] * scale_h

                        norm = feat_pts.clone()
                        norm[..., 0] = (norm[..., 0] + 0.5) / Wf * 2.0 - 1.0
                        norm[..., 1] = (norm[..., 1] + 0.5) / Hf * 2.0 - 1.0

                        grid = norm.unsqueeze(0)  # [1, H, W, 2]
                        feat_in = features[b, n].unsqueeze(0)  # [1, C, Hf, Wf]
                        sampled = F.grid_sample(feat_in, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                        bev_out[b, n] = sampled.squeeze(0)
                else:
                    H = self._compute_homography(K, Rt)  # [3,3]

                    # project ground plane points to image pixels
                    g_flat = ground.reshape(-1, 3).T  # [3, H*W]
                    uvw = H @ g_flat  # [3, H*W]
                    w = uvw[2:3, :]  # [1, H*W]
                    # avoid division by zero
                    w_safe = torch.where(w.abs() < 1e-6, torch.ones_like(w), w)
                    u = uvw[0:1, :] / w_safe
                    v = uvw[1:2, :] / w_safe
                    # stack and reshape to [H, W, 2]
                    img_pts = torch.stack([u.squeeze(0), v.squeeze(0)], dim=1).reshape(self.bev_h, self.bev_w, 2)

                    # convert to feature-map coordinates via downscale factors
                    scale_w = Wf / float(W_img)
                    scale_h = Hf / float(H_img)
                    feat_pts = img_pts.clone()
                    feat_pts[..., 0] = feat_pts[..., 0] * scale_w
                    feat_pts[..., 1] = feat_pts[..., 1] * scale_h

                    # normalize to [-1,1] for grid_sample (align_corners=False)
                    norm = feat_pts.clone()
                    norm[..., 0] = (norm[..., 0] + 0.5) / Wf * 2.0 - 1.0
                    norm[..., 1] = (norm[..., 1] + 0.5) / Hf * 2.0 - 1.0

                    # sample features
                    grid = norm.unsqueeze(0)  # [1, H, W, 2]
                    feat_in = features[b, n].unsqueeze(0)  # [1, C, Hf, Wf]
                    sampled = F.grid_sample(feat_in, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                    bev_out[b, n] = sampled.squeeze(0)  # [C, H, W]

        return bev_out