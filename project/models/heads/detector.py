import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class BEVDetector(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        bev_bounds: Tuple[float, float, float, float] = (-6.0, 6.0, -2.0, 2.0),
        bev_size: Tuple[int, int] = (64, 64),
        default_box_wh: Tuple[float, float] = (0.6, 0.6),
    ):
        super().__init__()
        mid1, mid2 = 512, 128
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=mid1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid1, mid2, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=mid2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid2, mid2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=mid2),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(mid2, 1, kernel_size=3, padding=1)
        self.offset_head = nn.Conv2d(mid2, 2, kernel_size=3, padding=1)
        self.size_head = nn.Conv2d(mid2, 2, kernel_size=3, padding=1)

        # Init heads following CenterNet practice
        nn.init.constant_(self.heatmap_head.bias, -2.19)
        nn.init.constant_(self.offset_head.weight, 0.0)
        nn.init.constant_(self.offset_head.bias, 0.0)

        self.bounds = bev_bounds
        self.bev_h, self.bev_w = bev_size
        self.res_x = (bev_bounds[1] - bev_bounds[0]) / float(self.bev_w)
        self.res_y = (bev_bounds[3] - bev_bounds[2]) / float(self.bev_h)
        default_w_cells = max(default_box_wh[0] / max(self.res_x, 1e-6), 1e-3)
        default_h_cells = max(default_box_wh[1] / max(self.res_y, 1e-6), 1e-3)
        size_bias = torch.log(torch.tensor([default_w_cells, default_h_cells], dtype=torch.float32))
        with torch.no_grad():
            self.size_head.bias.copy_(size_bias)

    def forward(self, bev_feat: torch.Tensor) -> Dict:
        # bev_feat: [B, C, H, W]
        shared = self.stem(bev_feat)
        heatmap_logits = self.heatmap_head(shared)
        offset_raw = self.offset_head(shared)
        size_raw = self.size_head(shared)
        offset = torch.sigmoid(offset_raw)
        size_cells = torch.exp(size_raw)
        return {
            "heatmap_logits": heatmap_logits,
            "heatmap": torch.sigmoid(heatmap_logits),
            "offset": offset,
            "offset_raw": offset_raw,
            "size": size_cells,
            "size_raw": size_raw,
        }

    @staticmethod
    def _nms2d(x: torch.Tensor, kernel: int = 3) -> torch.Tensor:
        pad = kernel // 2
        maxpool = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=pad)
        keep = (x == maxpool).float()
        return x * keep

    def decode(
        self,
        heatmap: torch.Tensor,
        offset: torch.Tensor,
        size_cells: torch.Tensor,
        conf_thresh: float = 0.4,
        nms_dist_m: float = 0.5,
    ):
        """Decode peaks with learned offsets and BEV footprint sizes."""

        B, _, H, W = heatmap.shape
        peaks = self._nms2d(heatmap)
        offset = offset.permute(0, 2, 3, 1)
        size_cells = size_cells.permute(0, 2, 3, 1)

        x_min, x_max, y_min, y_max = self.bounds
        res_x = (x_max - x_min) / float(W)
        res_y = (y_max - y_min) / float(H)

        boxes_list, scores_out = [], []
        for b in range(B):
            hm = peaks[b, 0]
            mask = hm > conf_thresh
            ys, xs = torch.where(mask)
            scores = hm[mask]
            if xs.numel() == 0:
                boxes_list.append(torch.zeros(0, 4, device=heatmap.device))
                scores_out.append(torch.zeros(0, device=heatmap.device))
                continue
            offsets = offset[b, ys, xs]
            sizes = size_cells[b, ys, xs]
            cx = x_min + (xs.float() + offsets[:, 0]) * res_x
            cy = y_min + (ys.float() + offsets[:, 1]) * res_y
            widths = sizes[:, 0] * res_x
            heights = sizes[:, 1] * res_y
            boxes = torch.stack([cx, cy, widths, heights], dim=1)

            if boxes.shape[0] > 1:
                order = torch.argsort(scores, descending=True)
                keep = []
                centers = boxes[:, :2]
                for idx in order:
                    c = centers[idx]
                    too_close = False
                    for kept in keep:
                        if torch.norm(centers[kept] - c).item() < nms_dist_m:
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