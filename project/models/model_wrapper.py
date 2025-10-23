import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List

from .encoders.cnn_encoder import CNNEncoder
from .fusion.geometry import GeometryTransformer
from .fusion.fusion import SimpleFusion
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
        bev_c, bev_h, bev_w = cfg['MODEL']['BEV_SIZE']
        bev_bounds = tuple(cfg['MODEL']['BEV_BOUNDS'])
        backbone = cfg['MODEL']['BACKBONE']

        # modules
        self.encoder = CNNEncoder(out_channels=feat_dim, backbone=backbone, pretrained=True)
        self.geom = GeometryTransformer(bev_h=bev_h, bev_w=bev_w, bev_bounds=bev_bounds, warp_impl='kornia')
        self.fusion = SimpleFusion(mode='sum')
        self.detector = BEVDetector(in_channels=bev_c if bev_c == feat_dim else feat_dim, bev_bounds=bev_bounds)
        self.bev_h, self.bev_w = bev_h, bev_w
        self.bounds = bev_bounds

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
        bev_feat = self.fusion(bev_per_view)  # [B,C,H_bev,W_bev]
        det_out = self.detector(bev_feat)
        heatmap = det_out['heatmap']
        boxes_list, scores_list = self.detector.decode(heatmap)
        return {
            'heatmap': heatmap,
            'boxes': boxes_list,
            'scores': scores_list,
            'bev_feat': bev_feat,
        }

    def loss(self, preds: Dict, targets: List[Dict], loss_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Build GT heatmap
        hm_gt = self._build_gt_heatmap(targets)  # [B,1,H,W]
        cls_w = float(loss_cfg['CLS_WEIGHT'])
        reg_w = float(loss_cfg['REG_WEIGHT'])
        geom_w = float(loss_cfg['GEOM_WEIGHT'])
        alpha = float(loss_cfg.get('FOCAL_ALPHA', 0.25))
        gamma = float(loss_cfg.get('FOCAL_GAMMA', 2.0))

        cls = focal_loss(preds['heatmap'], hm_gt, alpha=alpha, gamma=gamma)
        # L1 on boxes: concatenate per-batch
        l1_total = torch.zeros((), device=preds['heatmap'].device)
        for b, t in enumerate(targets):
            gt_boxes = t.get('boxes_world', None)
            if gt_boxes is not None and preds['boxes'][b] is not None:
                pred_boxes = preds['boxes'][b]
                # naive matching by nearest center (same order if possible)
                n = min(pred_boxes.shape[0], gt_boxes.shape[0])
                if n > 0:
                    l1_total = l1_total + F.l1_loss(pred_boxes[:n], gt_boxes[:n])
        reg = l1_total / max(1, len(targets))

        geom = self._geom_consistency_loss(targets)
        return {
            'cls_loss': cls_w * cls,
            'reg_loss': reg_w * reg,
            'geom_loss': geom_w * geom,
            'total_loss': cls_w * cls + reg_w * reg + geom_w * geom,
        }

    def _build_gt_heatmap(self, targets: List[Dict]) -> torch.Tensor:
        B = len(targets)
        H, W = self.bev_h, self.bev_w
        x_min, x_max, y_min, y_max = self.bounds
        res_x = (x_max - x_min) / float(W)
        res_y = (y_max - y_min) / float(H)
        hm = torch.zeros(B, 1, H, W)
        for b, t in enumerate(targets):
            centers = t.get('centers_world', None)
            boxes = t.get('boxes_world', None)
            pts = centers if centers is not None else (boxes[:, :2] if boxes is not None else None)
            if pts is None:
                continue
            xs = ((pts[:, 0] - x_min) / res_x).long().clamp(0, W - 1)
            ys = ((pts[:, 1] - y_min) / res_y).long().clamp(0, H - 1)
            hm[b, 0, ys, xs] = 1.0
        return hm.to(next(self.parameters()).device)

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