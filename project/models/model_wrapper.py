import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List

from .encoders.cnn_encoder import CNNEncoder
from .fusion.geometry import GeometryTransformer
from .fusion.fusion import SimpleFusion, ConcatFusion
from .heads.detector import BEVDetector


def focal_loss(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0):
    eps = 1e-6
    pred = pred.clamp(eps, 1.0 - eps)
    pt = torch.where(gt > 0.5, pred, 1 - pred)
    w = torch.where(gt > 0.5, alpha, 1 - alpha)
    loss = -w * (1 - pt) ** gamma * torch.log(pt)
    return loss.mean()


class BEVNet(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        # parse cfg
        feat_dim = int(cfg['MODEL']['FEAT_DIM'])
        bev_h, bev_w = cfg['MODEL']['BEV_SIZE'][1], cfg['MODEL']['BEV_SIZE'][2]
        bev_bounds = tuple(cfg['MODEL']['BEV_BOUNDS'])
        backbone = cfg['MODEL']['BACKBONE']
        use_pretrained = bool(cfg['MODEL'].get('PRETRAINED', False))
        out_index = int(cfg['MODEL'].get('OUT_INDEX', 2))
        self.conf_thresh = float(cfg.get('EVAL', {}).get('CONF_THRESH', 0.4))
        self.nms_dist_m = float(cfg.get('EVAL', {}).get('NMS_DIST_M', 0.5))
        self.gt_sigma_px = float(cfg.get('LOSS', {}).get('GT_SIGMA_PX', 1.0))

        # modules
        self.encoder = CNNEncoder(out_channels=feat_dim, backbone=backbone, pretrained=use_pretrained, out_index=out_index)
        self.geom = GeometryTransformer(bev_h=bev_h, bev_w=bev_w, bev_bounds=bev_bounds, warp_impl='kornia')
        self.fusion = ConcatFusion()
        # detector in_channels = V*C + 2 (pos enc)
        # Note: actual V known at runtime; we build head lazily on first forward
        self.detector = None
        self.bev_h, self.bev_w = bev_h, bev_w
        self.bounds = bev_bounds
        # Precompute BEV position encoding (sin/cos over normalized XY)
        self.register_buffer('pos_enc', self._create_pos_enc(bev_h, bev_w, bev_bounds), persistent=False)

    def forward(self, batch: Dict) -> Dict:
        images = batch['images']  # [B,V,3,H,W]
        B, V, _, H, W = images.shape
        feats = self.encoder(images)  # [B,V,C,Hf,Wf]
        intrinsics = batch['calib']['intrinsic']
        extrinsics = batch['calib']['extrinsic']
        # convert list to tensor if needed
        if isinstance(intrinsics, list):
            K = torch.stack([torch.stack(v, dim=0) for v in intrinsics], dim=0)  # [B,V,3,3]
        else:
            K = intrinsics
        if isinstance(extrinsics, list):
            Rt = torch.stack([torch.stack(v, dim=0) for v in extrinsics], dim=0)  # [B,V,4,4]
        else:
            Rt = extrinsics
        bev_per_view = self.geom(feats, K, Rt, img_size=(H, W))  # [B,V,C,H_bev,W_bev]
        bev_concat = self.fusion(bev_per_view)  # [B,V*C,H_bev,W_bev]
        # concat position encoding
        pos = self.pos_enc.unsqueeze(0).expand(B, -1, -1, -1)  # [B,2,H,W]
        bev_feat = torch.cat([bev_concat, pos], dim=1)
        # lazy build detector to match in_channels
        if self.detector is None:
            in_ch = bev_feat.shape[1]
            self.detector = BEVDetector(in_channels=in_ch, bev_bounds=self.bounds)
            self.detector = self.detector.to(bev_feat.device)
        det_out = self.detector(bev_feat)
        heatmap = det_out['heatmap']
        boxes_list, scores_list = self.detector.decode(heatmap, conf_thresh=self.conf_thresh, nms_dist_m=self.nms_dist_m)
        return {
            'heatmap': heatmap,
            'boxes': boxes_list,
            'scores': scores_list,
            'bev_feat': bev_feat,
        }

    def loss(self, preds: Dict, targets: List[Dict], loss_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # MSE on Gaussian-smoothed GT heatmap
        hm_gt = self._build_gt_heatmap_gaussian(targets, sigma_px=self.gt_sigma_px)  # [B,1,H,W]
        mse = F.mse_loss(preds['heatmap'], hm_gt)
        return {
            'mse_loss': mse,
            'total_loss': mse,
        }

    def _build_gt_heatmap_gaussian(self, targets: List[Dict], sigma_px: float = 1.0) -> torch.Tensor:
        B = len(targets)
        H, W = self.bev_h, self.bev_w
        x_min, x_max, y_min, y_max = self.bounds
        res_x = (x_max - x_min) / float(W)
        res_y = (y_max - y_min) / float(H)
        device = next(self.parameters()).device
        hm = torch.zeros(B, 1, H, W, device=device)
        rad = max(1, int(3 * sigma_px))
        # precompute Gaussian kernel window
        yy, xx = torch.meshgrid(torch.arange(-rad, rad + 1, device=device), torch.arange(-rad, rad + 1, device=device), indexing='ij')
        dist2 = (xx.float() ** 2 + yy.float() ** 2)
        gauss = torch.exp(-0.5 * dist2 / (sigma_px ** 2))
        for b, t in enumerate(targets):
            centers = t.get('centers_world', None)
            boxes = t.get('boxes_world', None)
            pts = centers if centers is not None else (boxes[:, :2] if boxes is not None else None)
            if pts is None or pts.shape[0] == 0:
                continue
            xs = ((pts[:, 0] - x_min) / res_x).long().clamp(0, W - 1)
            ys = ((pts[:, 1] - y_min) / res_y).long().clamp(0, H - 1)
            for i in range(xs.shape[0]):
                x = int(xs[i].item()); y = int(ys[i].item())
                x0 = max(0, x - rad); x1 = min(W - 1, x + rad)
                y0 = max(0, y - rad); y1 = min(H - 1, y + rad)
                gx0 = rad - (x - x0); gx1 = rad + (x1 - x)
                gy0 = rad - (y - y0); gy1 = rad + (y1 - y)
                hm[b, 0, y0:y1 + 1, x0:x1 + 1] = torch.maximum(hm[b, 0, y0:y1 + 1, x0:x1 + 1], gauss[gy0:gy1 + 1, gx0:gx1 + 1])
        return hm

    def _geom_consistency_loss(self, targets: List[Dict], num_samples: int = 128) -> torch.Tensor:
        # Sample points in BEV and enforce img->world->bev round-trip consistency
        device = next(self.parameters()).device
        x_min, x_max, y_min, y_max = self.bounds
        xs = torch.linspace(x_min, x_max, self.bev_w, device=device)
        ys = torch.linspace(y_min, y_max, self.bev_h, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        pts = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.ones(self.bev_w * self.bev_h, device=device)], dim=1)  # [N,3]
        idx = torch.randperm(pts.shape[0], device=device)[:num_samples]
        pts = pts[idx]  # [K,3]
        loss = torch.zeros((), device=device)
        # use first sample's calib if available
        if len(targets) == 0:
            return loss
        calib = targets[0].get('calib', None)
        if calib is None:
            return loss
        K_list = calib.get('intrinsic', [])
        Rt_list = calib.get('extrinsic', [])
        for K, Rt in zip(K_list, Rt_list):
            H_w2i = GeometryTransformer._compute_homography(K.to(device), Rt.to(device))
            uvw = (H_w2i @ pts.T)  # [3,K]
            w = uvw[2:3, :]
            w_safe = torch.where(w.abs() < 1e-6, torch.ones_like(w), w)
            u = uvw[0:1, :] / w_safe
            v = uvw[1:2, :] / w_safe
            H_i2w = GeometryTransformer._compute_img_to_world_homography(K.to(device), Rt.to(device))
            back = H_i2w @ torch.stack([u.squeeze(0), v.squeeze(0), torch.ones_like(u.squeeze(0))], dim=0)
            back_xy = back[:2, :].T  # [K,2]
            loss = loss + F.l1_loss(back_xy, pts[:, :2])
        return loss / max(1, len(K_list))

    @staticmethod
    def _create_pos_enc(H: int, W: int, bounds: Tuple[float, float, float, float]) -> torch.Tensor:
        # 2通道位置编码：sin/cos基于归一化XY
        x_min, x_max, y_min, y_max = bounds
        xs = torch.linspace(x_min, x_max, W)
        ys = torch.linspace(y_min, y_max, H)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        x_norm = (xx - x_min) / (x_max - x_min)
        y_norm = (yy - y_min) / (y_max - y_min)
        pos_sin = torch.sin(2.0 * torch.pi * x_norm)
        pos_cos = torch.cos(2.0 * torch.pi * y_norm)
        return torch.stack([pos_sin, pos_cos], dim=0)  # [2,H,W]