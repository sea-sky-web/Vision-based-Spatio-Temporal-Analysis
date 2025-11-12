import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List

from .encoders.cnn_encoder import CNNEncoder
from .fusion.geometry import GeometryTransformer
from .fusion.fusion import SimpleFusion, ConcatFusion
from .heads.detector import BEVDetector


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
        loss_cfg = cfg.get('LOSS', {})
        self.default_box_wh = tuple(loss_cfg.get('DEFAULT_BOX_WH', [0.6, 0.6]))
        self.max_objects = int(loss_cfg.get('MAX_OBJECTS', 64))
        self.hm_alpha = float(loss_cfg.get('HM_ALPHA', 2.0))
        self.hm_beta = float(loss_cfg.get('HM_BETA', 4.0))
        self.hm_weight = float(loss_cfg.get('HM_WEIGHT', 1.0))
        self.offset_weight = float(loss_cfg.get('OFFSET_WEIGHT', 1.0))
        self.size_weight = float(loss_cfg.get('SIZE_WEIGHT', 0.1))
        self.gaussian_min_radius = int(loss_cfg.get('GAUSSIAN_MIN_RADIUS', 2))
        self.gaussian_iou = float(loss_cfg.get('GAUSSIAN_IOU', 0.7))

        self.res_x = (bev_bounds[1] - bev_bounds[0]) / float(bev_w)
        self.res_y = (bev_bounds[3] - bev_bounds[2]) / float(bev_h)

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
            self.detector = BEVDetector(
                in_channels=in_ch,
                bev_bounds=self.bounds,
                bev_size=(self.bev_h, self.bev_w),
                default_box_wh=self.default_box_wh,
            ).to(bev_feat.device)
        det_out = self.detector(bev_feat)
        boxes_list, scores_list = self.detector.decode(
            det_out['heatmap'],
            det_out['offset'],
            det_out['size'],
            conf_thresh=self.conf_thresh,
            nms_dist_m=self.nms_dist_m,
        )
        return {
            'heatmap': det_out['heatmap'],
            'heatmap_logits': det_out['heatmap_logits'],
            'boxes': boxes_list,
            'scores': scores_list,
            'offset': det_out['offset'],
            'offset_raw': det_out['offset_raw'],
            'size': det_out['size'],
            'size_raw': det_out['size_raw'],
            'bev_feat': bev_feat,
        }

    def loss(self, preds: Dict, targets: List[Dict], loss_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        target_dict = self._build_training_targets(targets)
        hm_loss = self._heatmap_focal_loss(preds['heatmap_logits'], target_dict['heatmap'])

        mask = target_dict['mask'].unsqueeze(-1)
        denom = mask.sum() + 1e-4

        offset_pred = self._gather_feat(preds['offset'], target_dict['indices'])
        offset_loss = F.l1_loss(offset_pred * mask, target_dict['offset'] * mask, reduction='sum') / denom

        size_pred = self._gather_feat(preds['size_raw'], target_dict['indices'])
        size_loss = F.l1_loss(size_pred * mask, target_dict['size_log'] * mask, reduction='sum') / denom

        total = self.hm_weight * hm_loss + self.offset_weight * offset_loss + self.size_weight * size_loss
        return {
            'heatmap_loss': hm_loss,
            'offset_loss': offset_loss,
            'size_loss': size_loss,
            'total_loss': total,
        }

    def _build_training_targets(self, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        B = len(targets)
        hm = torch.zeros(B, 1, self.bev_h, self.bev_w, device=device)
        indices = torch.zeros(B, self.max_objects, dtype=torch.long, device=device)
        mask = torch.zeros(B, self.max_objects, dtype=torch.float32, device=device)
        offset = torch.zeros(B, self.max_objects, 2, device=device)
        size_log = torch.zeros(B, self.max_objects, 2, device=device)

        x_min, x_max, y_min, y_max = self.bounds

        for b, tgt in enumerate(targets):
            boxes = tgt.get('boxes_world', None)
            if boxes is None or boxes.numel() == 0:
                centers = tgt.get('centers_world', None)
                if centers is not None and centers.numel() > 0:
                    default_wh = torch.tensor(self.default_box_wh, device=centers.device, dtype=torch.float32)
                    boxes = torch.cat([centers, default_wh.repeat(centers.shape[0], 1)], dim=1)
            if boxes is None or boxes.numel() == 0:
                continue
            boxes = boxes.to(device)
            count = 0
            for i in range(boxes.shape[0]):
                if count >= self.max_objects:
                    break
                cx_m, cy_m, w_m, h_m = boxes[i]
                x_rel = (cx_m.item() - x_min) / self.res_x
                y_rel = (cy_m.item() - y_min) / self.res_y
                if x_rel < 0 or x_rel >= self.bev_w or y_rel < 0 or y_rel >= self.bev_h:
                    continue
                gx = int(x_rel)
                gy = int(y_rel)
                radius = self._gaussian_radius(w_m.item() / self.res_x, h_m.item() / self.res_y)
                hm[b, 0] = self._draw_gaussian(hm[b, 0], (gx, gy), radius)
                indices[b, count] = gy * self.bev_w + gx
                mask[b, count] = 1.0
                offset[b, count, 0] = x_rel - gx
                offset[b, count, 1] = y_rel - gy
                size_w_cells = max(w_m.item() / self.res_x, 1e-3)
                size_h_cells = max(h_m.item() / self.res_y, 1e-3)
                size_log[b, count, 0] = math.log(size_w_cells)
                size_log[b, count, 1] = math.log(size_h_cells)
                count += 1

        return {
            'heatmap': hm,
            'indices': indices,
            'mask': mask,
            'offset': offset,
            'size_log': size_log,
        }

    def _heatmap_focal_loss(self, pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred_logits)
        pos_mask = gt.eq(1.0)
        neg_mask = gt.lt(1.0)
        neg_weights = torch.pow(1 - gt, self.hm_beta)

        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.hm_alpha) * pos_mask
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.hm_alpha) * neg_weights * neg_mask

        num_pos = pos_mask.float().sum()
        loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos.clamp(min=1.0)
        return loss

    def _gaussian_radius(self, width_cells: float, height_cells: float) -> int:
        width_cells = max(width_cells, 1.0)
        height_cells = max(height_cells, 1.0)
        min_overlap = self.gaussian_iou
        a1 = 1
        b1 = height_cells + width_cells
        c1 = width_cells * height_cells * (1 - min_overlap) / (1 + min_overlap)
        sq1 = max(0.0, b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + math.sqrt(sq1)) / 2

        a2 = 4
        b2 = 2 * (height_cells + width_cells)
        c2 = (1 - min_overlap) * width_cells * height_cells
        sq2 = max(0.0, b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + math.sqrt(sq2)) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height_cells + width_cells)
        c3 = (min_overlap - 1) * width_cells * height_cells
        sq3 = max(0.0, b3 ** 2 - 4 * a3 * c3)
        if a3 == 0:
            r3 = float('inf')
        else:
            r3 = (b3 + math.sqrt(sq3)) / (2 * a3)

        radius = min(r1, r2, r3)
        radius = max(self.gaussian_min_radius, int(radius))
        return radius

    def _draw_gaussian(self, heatmap: torch.Tensor, center: Tuple[int, int], radius: int) -> torch.Tensor:
        radius = int(radius)
        if radius <= 0:
            return heatmap
        diameter = 2 * radius + 1
        sigma = diameter / 6.0
        x, y = center
        height, width = heatmap.shape
        if x < 0 or y < 0 or x >= width or y >= height:
            return heatmap
        left = min(x, radius)
        right = min(width - x - 1, radius)
        top = min(y, radius)
        bottom = min(height - y - 1, radius)
        if left < 0 or right < 0 or top < 0 or bottom < 0:
            return heatmap
        y_range = torch.arange(-top, bottom + 1, device=heatmap.device, dtype=heatmap.dtype)
        x_range = torch.arange(-left, right + 1, device=heatmap.device, dtype=heatmap.dtype)
        yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
        gaussian = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma * sigma))
        patch = heatmap[y - top:y + bottom + 1, x - left:x + right + 1]
        torch.maximum(patch, gaussian, out=patch)
        return heatmap

    @staticmethod
    def _gather_feat(feat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        feat = feat.view(B, C, -1).permute(0, 2, 1)
        idx_exp = indices.unsqueeze(-1).expand(-1, -1, C)
        gathered = torch.gather(feat, 1, idx_exp)
        return gathered

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