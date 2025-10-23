"""Simple feature fusion strategies for multi-view BEV features."""
import torch


class SimpleFusion:
    def __init__(self, mode: str = 'sum'):
        assert mode in ('sum', 'mean', 'max'), "Fusion mode must be one of: sum, mean, max"
        self.mode = mode

    def __call__(self, bev_maps: torch.Tensor) -> torch.Tensor:
        """Fuses along view dimension.

        Args:
            bev_maps: [B, N, C, H, W]
        Returns:
            fused: [B, C, H, W]
        """
        if self.mode == 'sum':
            return bev_maps.sum(dim=1)
        if self.mode == 'mean':
            return bev_maps.mean(dim=1)
        # max
        return bev_maps.max(dim=1).values
