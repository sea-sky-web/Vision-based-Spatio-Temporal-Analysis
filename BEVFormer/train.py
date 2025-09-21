"""Minimal training entrypoint for BEV Fusion v1.0 - Dry run

This script performs a single forward pass (dry run) to verify the end-to-end
pipeline can be constructed and run. It does not perform any training.
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/wildtrack_v1.py")
    args = parser.parse_args()

    try:
        import torch
    except Exception as e:
        print("PyTorch is required to run this dry run. Please install dependencies first.")
        print("Error:", e)
        sys.exit(1)

    # Import project modules
    from configs import wildtrack_v1 as cfg
    from datasets.wildtrack_dataset import WildtrackDataset
    from models.bev_fusion_net import BEVFusionNet

    print("Config:", cfg)

    # Create dataset and dataloader
    ds = WildtrackDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # Instantiate model
    model = BEVFusionNet(cfg)
    model.eval()

    # Run one batch
    batch = next(iter(dl))
    print("Loaded batch keys:", list(batch.keys()))

    with torch.no_grad():
        out = model(batch['images'])

    print("Model output shape:", out.shape)


if __name__ == "__main__":
    main()
