import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    def forward(self, bev_maps: torch.Tensor) -> torch.Tensor:
        """bev_maps: Tensor[B, V, C, H, W] -> Tensor[B, C, H, W] or [B, V*C, H, W]"""
        raise NotImplementedError


class AttentionFusion(FusionModule):
    def __init__(self, in_channels: int):
        super().__init__()
        # Learnable attention mechanism: computes a spatial weight map for each view
        # Input: [B*V, C, H, W] -> Output: [B*V, 1, H, W]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )

    def forward(self, bev_maps: torch.Tensor) -> torch.Tensor:
        # bev_maps: [B, V, C, H, W]
        B, V, C, H, W = bev_maps.shape

        # Flatten views into batch dimension to process them efficiently with Conv2d
        x = bev_maps.view(B * V, C, H, W)

        # Compute raw attention scores
        scores = self.conv(x)  # [B*V, 1, H, W]

        # Reshape back to separate views
        scores = scores.view(B, V, 1, H, W)

        # Apply Softmax across the view dimension (dim=1) to get normalized weights
        weights = F.softmax(scores, dim=1)  # [B, V, 1, H, W]

        # Weighted sum fusion
        # [B, V, C, H, W] * [B, V, 1, H, W] -> [B, V, C, H, W] --sum(dim=1)--> [B, C, H, W]
        out = (bev_maps * weights).sum(dim=1)

        return out
