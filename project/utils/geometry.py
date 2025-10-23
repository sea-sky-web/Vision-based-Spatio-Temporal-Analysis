import torch
from typing import Tuple


def meters_to_bev_indices(xy: torch.Tensor, bev_bounds: Tuple[float, float, float, float], bev_size: Tuple[int, int]) -> torch.Tensor:
    x_min, x_max, y_min, y_max = bev_bounds
    H, W = bev_size
    res_x = (x_max - x_min) / float(W)
    res_y = (y_max - y_min) / float(H)
    xs = ((xy[:, 0] - x_min) / res_x).clamp(0, W - 1)
    ys = ((xy[:, 1] - y_min) / res_y).clamp(0, H - 1)
    return torch.stack([xs, ys], dim=1)


def bev_indices_to_meters(idx: torch.Tensor, bev_bounds: Tuple[float, float, float, float], bev_size: Tuple[int, int]) -> torch.Tensor:
    x_min, x_max, y_min, y_max = bev_bounds
    H, W = bev_size
    res_x = (x_max - x_min) / float(W)
    res_y = (y_max - y_min) / float(H)
    x = x_min + (idx[:, 0] + 0.5) * res_x
    y = y_min + (idx[:, 1] + 0.5) * res_y
    return torch.stack([x, y], dim=1)