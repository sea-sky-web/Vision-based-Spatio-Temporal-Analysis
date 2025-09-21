"""Simple projection that resizes feature maps to BEV spatial resolution."""
import torch
import torch.nn.functional as F


class SimpleProjection:
    def __init__(self, bev_size=(1, 128, 128)):
        # bev_size: (C, H, W) - C will be inferred
        self.bev_h = bev_size[1]
        self.bev_w = bev_size[2]

    def __call__(self, feats):
        # feats: (B, NumCams, Cfeat, Hf, Wf)
        B, NC, C, H, W = feats.shape
        out = torch.zeros((B, NC, C, self.bev_h, self.bev_w), dtype=feats.dtype)
        for b in range(B):
            for c in range(NC):
                out[b, c] = F.interpolate(feats[b, c].unsqueeze(0), size=(self.bev_h, self.bev_w), mode='bilinear', align_corners=False).squeeze(0)
        return out
