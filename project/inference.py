import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from data.wildtrack_loader import WildtrackDataset, collate_fn
from models.model_wrapper import BEVNet
from plugins import build_plugins
from utils.visualization import save_predictions_json


def load_cfg(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth')
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(cfg['RUNTIME']['DEVICE'] if torch.cuda.is_available() else 'cpu')

    ds = WildtrackDataset(cfg)
    dl = DataLoader(ds, batch_size=cfg['DATA']['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn)

    model = BEVNet(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    out_dir = cfg['RUNTIME']['OUTPUT_DIR']
    run_context = {'output_dir': out_dir, 'device': device}
    plugins = build_plugins(cfg, device=device, output_dir=out_dir)
    for plugin in plugins:
        plugin.on_start(run_context)

    with torch.no_grad():
        for batch in dl:
            batch['images'] = batch['images'].to(device)
            for b in range(len(batch['calib']['intrinsic'])):
                batch['calib']['intrinsic'][b] = [k.to(device) for k in batch['calib']['intrinsic'][b]]
                batch['calib']['extrinsic'][b] = [e.to(device) for e in batch['calib']['extrinsic'][b]]
            preds = model(batch)
            frame_indices = [int(m['frame_idx']) for m in batch['meta']]

            for plugin in plugins:
                plugin.on_batch(preds, batch, run_context)

            save_predictions_json(preds['boxes'], preds['scores'], out_dir, frame_indices)

    for plugin in plugins:
        plugin.on_finish(run_context)

    print(f"Saved predictions JSON to {out_dir}")


if __name__ == '__main__':
    main()
