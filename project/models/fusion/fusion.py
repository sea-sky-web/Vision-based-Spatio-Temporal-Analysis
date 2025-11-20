import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    def forward(self, bev_maps: torch.Tensor) -> torch.Tensor:
        """bev_maps: Tensor[B, V, C, H, W] -> Tensor[B, C, H, W]"""
        raise NotImplementedError


class SimpleFusion(FusionModule):
    def __init__(self, mode: str = 'sum'):
        super().__init__()
        assert mode in ('sum', 'mean', 'max')
        self.mode = mode

    def forward(self, bev_maps: torch.Tensor) -> torch.Tensor:
        if self.mode == 'sum':
            return bev_maps.sum(dim=1)
        if self.mode == 'mean':
            return bev_maps.mean(dim=1)
        return bev_maps.max(dim=1).values


class AttentionFusion(FusionModule):
    def __init__(self, in_channels: int):
        super().__init__()
        # Learnable attention mechanism
        # Weights per view based on feature content
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

    def forward(self, bev_maps: torch.Tensor) -> torch.Tensor:
        # bev_maps: [B, V, C, H, W]
        B, V, C, H, W = bev_maps.shape

        # Flatten views into batch for processing
        # [B*V, C, H, W]
        bev_flat = bev_maps.reshape(B * V, C, H, W)

        # Calculate global context or compare views?
        # Simple self-attention: view features -> attention score
        # But to know which view is "better", we usually need to know the occlusion state.
        # Here we implement a simple "Confidence Map" prediction per view.
        # But we need the input to be view-dependent.

        # Let's implement a simplified attention:
        # Pass each view through a small CNN to get a weight map [B, V, 1, H, W]

        # However, standard AttentionFusion often compares views against a global mean or each other.
        # Let's try a lightweight approach:
        # 1. Compute mean feature: [B, C, H, W]
        # 2. Concat mean with each view: [B*V, 2C, H, W]
        # 3. Predict score: [B*V, 1, H, W]

        mean_feat = bev_maps.mean(dim=1, keepdim=True) # [B, 1, C, H, W]
        mean_expanded = mean_feat.expand(-1, V, -1, -1, -1).reshape(B * V, C, H, W)

        concat = torch.cat([bev_flat, mean_expanded], dim=1) # [B*V, 2C, H, W]

        scores = self.conv(concat) # [B*V, 1, H, W]
        scores = scores.reshape(B, V, 1, H, W)
        weights = F.softmax(scores, dim=1) # Normalize across views

        # Weighted sum
        weighted_bev = (bev_maps * weights).sum(dim=1) # [B, C, H, W]

        return weighted_bev


class ConcatFusion(FusionModule):
    def __init__(self):
        super().__init__()

    def forward(self, bev_maps: torch.Tensor) -> torch.Tensor:
        # [B, V, C, H, W] -> [B, V*C, H, W]
        B, V, C, H, W = bev_maps.shape
        return bev_maps.reshape(B, V * C, H, W)
