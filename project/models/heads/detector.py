import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class BEVDetector(nn.Module):
    def __init__(self, in_channels: int = 32, heatmap_sigma: float = 2.0, bev_bounds: Tuple[float, float, float, float] = (-6.0, 6.0, -2.0, 2.0)):
        super().__init__()
        # 3层扩张卷积 + GroupNorm，较大感受野
        mid1, mid2 = 512, 128
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid1, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=mid1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid1, mid2, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=mid2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid2, 1, kernel_size=3, padding=4, dilation=4)
        )
        self.heatmap_sigma = heatmap_sigma
        self.bounds = bev_bounds

    def forward(self, bev_feat: torch.Tensor) -> Dict:
        # bev_feat: [B, C, H, W]
        logits = self.head(bev_feat)  # [B,1,H,W]
        return {"heatmap": logits.sigmoid()}

    @staticmethod
    def _nms2d(x: torch.Tensor, kernel: int = 3) -> torch.Tensor:
        pad = kernel // 2
        maxpool = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=pad)
        keep = (x == maxpool).float()
        return x * keep

    def decode(self, heatmap: torch.Tensor, conf_thresh: float = 0.4, box_size_m: Tuple[float, float] = (0.6, 0.6), nms_dist_m: float = 0.5):
        """Decode peaks to BEV box centers, returns boxes in meters.
        heatmap: [B,1,H,W]
        """
        B, _, H, W = heatmap.shape
        peaks = self._nms2d(heatmap)
        xs_list, ys_list, scores_list = [], [], []
        for b in range(B):
            hm = peaks[b, 0]
            mask = hm > conf_thresh
            ys, xs = torch.where(mask)
            scores = hm[mask]
            xs_list.append(xs)
            ys_list.append(ys)
            scores_list.append(scores)
        # convert pixel to meters
        x_min, x_max, y_min, y_max = self.bounds
        res_x = (x_max - x_min) / float(W)
        res_y = (y_max - y_min) / float(H)
        boxes_list, scores_out = [], []
        for b in range(B):
            xs = xs_list[b].float()
            ys = ys_list[b].float()
            scores = scores_list[b]
            cx = x_min + (xs + 0.5) * res_x
            cy = y_min + (ys + 0.5) * res_y
            w_m, h_m = box_size_m
            if cx.numel() == 0:
                boxes = torch.zeros(0, 4)
            else:
                boxes = torch.stack([cx, cy, torch.full_like(cx, w_m), torch.full_like(cx, h_m)], dim=1)
            # 简单BEV NMS：按中心距离>nms_dist_m保留最高分
            if boxes.shape[0] > 1:
                order = torch.argsort(scores, descending=True)
                keep = []
                centers = boxes[:, :2]
                for idx in order:
                    c = centers[idx]
                    too_close = False
                    for k in keep:
                        if torch.norm(centers[k] - c).item() < nms_dist_m:
                            too_close = True
                            break
                    if not too_close:
                        keep.append(int(idx))
                boxes = boxes[keep]
                scores = scores[keep]
            boxes_list.append(boxes)
            scores_out.append(scores)
        return boxes_list, scores_out


class AnchorDetector(nn.Module):
    def __init__(self):
        super().__init__()
        print("[AnchorDetector] Placeholder only.")