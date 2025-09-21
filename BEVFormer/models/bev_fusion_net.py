"""Simplified BEVFusionNet that wires backbone, projection, fusion, and head."""
import torch
import torch.nn as nn
from models.modules.backbone import SimpleBackbone
from models.modules.projection import SimpleProjection
from models.modules.fusion import SimpleFusion
from models.modules.bev_head import SimpleBEVHead


class BEVFusionNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = SimpleBackbone()
        self.proj = SimpleProjection()
        self.fusion = SimpleFusion()
        self.head = SimpleBEVHead(cfg)

    def forward(self, images):
        # images: (B, NumCams, C, H, W) or (NumCams, C, H, W) if batchless
        if images.dim() == 4:
            # (NumCams, C, H, W) -> add batch
            images = images.unsqueeze(0)

        B = images.shape[0]
        num_cams = images.shape[1]

        # reshape to process each view through backbone
        imgs = images.view(B * num_cams, *images.shape[2:])
        feats = self.backbone(imgs)  # (B*num_cams, Cfeat, Hf, Wf)

        # reshape back to (B, num_cams, ...)
        Cfeat = feats.shape[1]
        feats = feats.view(B, num_cams, Cfeat, feats.shape[-2], feats.shape[-1])

        bev_maps = self.proj(feats)  # (B, num_cams, Cb, Hb, Wb)
        fused = self.fusion(bev_maps)  # (B, Cb, Hb, Wb)
        out = self.head(fused)  # (B, NumClasses, Hb, Wb)
        return out
