"""Simple fusion by averaging BEV features across cameras."""
import torch


class SimpleFusion:
    def __call__(self, bev_maps):
        # bev_maps: (B, NumCams, C, H, W)
        return bev_maps.mean(dim=1)
