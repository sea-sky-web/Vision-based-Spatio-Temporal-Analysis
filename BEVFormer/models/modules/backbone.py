"""Backbone implementations for feature extraction."""
import torch
import torch.nn as nn


class SimpleBackbone(nn.Module):
    """Minimal backbone (2 convs) used as fallback."""
    def __init__(self, out_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.net(x)
        try:
            print(f"[SimpleBackbone] in shape: {tuple(x.shape)}, out shape: {tuple(out.shape)}")
            print(f"[SimpleBackbone] out stats: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}")
        except Exception:
            pass
        return out


class TimmBackbone(nn.Module):
    """Pretrained backbone via timm, projecting features to 32 channels."""
    def __init__(self, model_name='resnet18', out_indices=(2,), out_channels=32, pretrained=True):
        super().__init__()
        self.out_channels = out_channels
        self.model_name = model_name
        self._use_timm = False
        self.proj = None

        try:
            import timm
            self._use_timm = True
            # features_only returns list of feature maps; pick one index
            self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
            self.out_idx = out_indices[0]
            # create a projection layer after knowing channels lazily
            self._feature_channels = None
        except Exception as e:
            print(f"[TimmBackbone] timm not available ({e}), falling back to SimpleBackbone")
            self.backup = SimpleBackbone(out_channels=out_channels)

    def forward(self, x):
        if self._use_timm:
            feats_list = self.backbone(x)
            feat = feats_list[self.out_idx]  # (B, Cb, Hf, Wf)
            # lazy init projection if not set
            if self._feature_channels is None:
                self._feature_channels = feat.shape[1]
                self.proj = nn.Conv2d(self._feature_channels, self.out_channels, kernel_size=1)
            out = self.proj(feat)
            try:
                print(f"[TimmBackbone] in {tuple(x.shape)} -> feat {tuple(feat.shape)} -> out {tuple(out.shape)}")
            except Exception:
                pass
            return out
        else:
            return self.backup(x)
