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
    """Cross-view attention pooling for BEV features.

    The module treats the V per-view BEV tensors at each spatial location as a short
    sequence and learns a lightweight attention pooling layer that decides how to
    weight each view before aggregation. The design mirrors the cross-view attention
    blocks that appear in recent multi-camera perception work (e.g., query token +
    multi-head attention), but intentionally keeps the code simple so that it can be
    trained end-to-end alongside the existing modules without additional plumbing.
    """

    def __init__(
        self,
        channel_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channel_dim % num_heads != 0:
            raise ValueError(
                f"AttentionFusion requires channel_dim % num_heads == 0, got {channel_dim} and {num_heads}."
            )
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=channel_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.context_token = nn.Parameter(torch.randn(1, 1, channel_dim))
        nn.init.trunc_normal_(self.context_token, std=0.02)
        self.norm = nn.LayerNorm(channel_dim)
        self.mlp = nn.Sequential(
            nn.Linear(channel_dim, channel_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channel_dim, channel_dim),
        )

    def forward(self, bev_maps: torch.Tensor) -> torch.Tensor:
        """Fuse V-view BEV features with learned attention pooling."""

        B, V, C, H, W = bev_maps.shape
        if C != self.channel_dim:
            raise ValueError(
                f"AttentionFusion received channel_dim={C}, but was initialized with {self.channel_dim}."
            )
        # Rearrange to [V, B*H*W, C] so that each BEV cell becomes an attention batch.
        views = bev_maps.permute(1, 0, 3, 4, 2).contiguous().view(V, B * H * W, C)
        # The learned context token queries all views and produces a single fused feature.
        query = self.context_token.expand(1, B * H * W, C)
        attn_out, _ = self.attn(query, views, views)
        fused = attn_out.squeeze(0)  # [B*H*W, C]
        fused = fused + self.mlp(self.norm(fused))
        fused = fused.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return fused


class ConcatFusion(FusionModule):
    def __init__(self):
        super().__init__()

    def forward(self, bev_maps: torch.Tensor) -> torch.Tensor:
        # [B, V, C, H, W] -> [B, V*C, H, W]
        B, V, C, H, W = bev_maps.shape
        return bev_maps.reshape(B, V * C, H, W)