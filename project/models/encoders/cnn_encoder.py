import torch
import torch.nn as nn
from typing import Tuple

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

from .base import ViewEncoder


class CNNEncoder(ViewEncoder):
    def __init__(self, out_channels: int = 32, backbone: str = "resnet18", pretrained: bool = True, out_index: int = 2):
        super().__init__(out_channels)
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.out_index = out_index
        self._use_timm = False
        self._feature_channels = None

        if _HAS_TIMM:
            try:
                self._use_timm = True
                self.backbone = timm.create_model(self.backbone_name, pretrained=self.pretrained, features_only=True)
                self.proj = None
            except Exception as e:
                print(f"[CNNEncoder] timm unavailable ({e}), fallback to simple conv")
                self._use_timm = False
        if not self._use_timm:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )

    def _encode_single(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_timm:
            feats_list = self.backbone(x)
            feat = feats_list[self.out_index]
            if self._feature_channels is None:
                self._feature_channels = feat.shape[1]
                self.proj = nn.Conv2d(self._feature_channels, self.out_channels, kernel_size=1)
            return self.proj(feat)
        else:
            return self.backbone(x)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: Tensor[B*V, 3, H, W] or Tensor[B, V, 3, H, W]
        Returns: Tensor[B, V, C, Hf, Wf]
        """
        if images.dim() == 4:
            # treat as [B*V, 3, H, W], need reshape to [B, V]
            # assume V known via context; infer V by dividing batch?
            # We cannot infer V reliably here; prefer caller to pass [B,V,...]
            x = images
            B = 1
            V = x.shape[0]
            feats = self._encode_single(x)
            C, Hf, Wf = feats.shape[1], feats.shape[2], feats.shape[3]
            return feats.view(B, V, C, Hf, Wf)
        elif images.dim() == 5:
            B, V, C_in, H, W = images.shape
            x = images.view(B * V, C_in, H, W)
            feats = self._encode_single(x)
            C, Hf, Wf = feats.shape[1], feats.shape[2], feats.shape[3]
            return feats.view(B, V, C, Hf, Wf)
        else:
            raise ValueError(f"[CNNEncoder] unexpected input shape: {tuple(images.shape)}")

    def load_pretrained(self, weights_path: str):
        super().load_pretrained(weights_path)

    def freeze(self):
        super().freeze()