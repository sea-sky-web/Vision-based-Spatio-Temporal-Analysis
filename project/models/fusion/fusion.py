import torch
import torch.nn as nn


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
    def __init__(self):
        super().__init__()
        # placeholder: not implemented
        self._warned = False

    def forward(self, bev_maps: torch.Tensor) -> torch.Tensor:
        if not self._warned:
            print("[AttentionFusion] Placeholder only. Not implemented.")
            self._warned = True
        # simple fallback
        return bev_maps.mean(dim=1)


class ConcatFusion(FusionModule):
    def __init__(self):
        super().__init__()

    def forward(self, bev_maps: torch.Tensor) -> torch.Tensor:
        # [B, V, C, H, W] -> [B, V*C, H, W]
        B, V, C, H, W = bev_maps.shape
        return bev_maps.reshape(B, V * C, H, W)