"""Placeholder visualization utilities."""
def show_bev(bev_tensor):
    # bev_tensor: torch.Tensor
    print("BEV tensor shape:", getattr(bev_tensor, 'shape', None))
