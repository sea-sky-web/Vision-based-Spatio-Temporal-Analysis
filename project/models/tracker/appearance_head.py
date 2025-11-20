import torch
import torch.nn as nn
import torch.nn.functional as F

class AppearanceHead(nn.Module):
    """
    Extracts L2-normalized appearance embeddings from ROI features.
    """
    def __init__(self, in_channels: int, emb_dim: int = 128):
        """
        Initializes the AppearanceHead module.

        Args:
            in_channels (int): The number of input channels from the feature map.
            emb_dim (int): The dimension of the output embedding.
        """
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim

        # A simple projection head: 1x1 Conv -> BN -> ReLU -> FC -> L2Norm
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.fc = nn.Linear(in_channels, emb_dim)

    def forward(self, roi_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate appearance embeddings.

        Args:
            roi_feats (torch.Tensor): A tensor of shape (B, C, H, W) containing
                                      features for each region of interest.

        Returns:
            torch.Tensor: A tensor of shape (B, D) containing the L2-normalized
                          appearance embeddings.
        """
        # Global average pooling
        x = F.adaptive_avg_pool2d(roi_feats, (1, 1))

        # 1x1 Convolution, BN, ReLU
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        # Flatten and project to embedding space
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)

        # L2 Normalization
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

# Example Usage (for verification)
if __name__ == '__main__':
    # Configuration
    batch_size = 16
    input_channels = 256
    embedding_dim = 128
    feature_map_height = 7
    feature_map_width = 7

    # Create a dummy input tensor
    dummy_roi_features = torch.randn(
        batch_size,
        input_channels,
        feature_map_height,
        feature_map_width
    )

    # Initialize the model
    appearance_head = AppearanceHead(in_channels=input_channels, emb_dim=embedding_dim)
    appearance_head.eval() # Set to evaluation mode for consistent output

    # Get the output embedding
    with torch.no_grad():
        output_embedding = appearance_head(dummy_roi_features)

    # Verification checks
    print(f"Input shape: {dummy_roi_features.shape}")
    print(f"Output shape: {output_embedding.shape}")

    # Check 1: Output shape must be (batch_size, emb_dim)
    assert output_embedding.shape == (batch_size, embedding_dim), \
        f"Shape mismatch: expected {(batch_size, embedding_dim)}, got {output_embedding.shape}"
    print("✓ Output shape is correct.")

    # Check 2: L2 norm of each embedding vector must be 1.0
    norms = torch.linalg.norm(output_embedding, ord=2, dim=1)
    assert torch.allclose(norms, torch.ones(batch_size)), \
        "L2 normalization failed. Norms are not all close to 1."
    print("✓ L2 normalization is correct.")
    print(f"Norms of first 5 embeddings: {norms[:5].numpy()}")

    print("\nAppearanceHead module implementation is verified.")
