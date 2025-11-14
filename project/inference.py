import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from data.wildtrack_loader import WildtrackDataset, collate_fn
from models.model_wrapper import BEVNet
from models.tracking import SimpleTrajectoryTracker
from utils.visualization import save_predictions_json, save_trajectories_json


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

    tracker_cfg = cfg.get('TRACKER', {})
    tracker_enabled = tracker_cfg.get('ENABLED', True)
    tracker = None
    if tracker_enabled:
        tracker = SimpleTrajectoryTracker(
            max_age=int(tracker_cfg.get('MAX_AGE', 15)),
            min_hits=int(tracker_cfg.get('MIN_HITS', 3)),
            size_alpha=float(tracker_cfg.get('SIZE_ALPHA', 0.25)),
            high_conf_thresh=float(tracker_cfg.get('HIGH_CONF_THRESH', 0.6)),
            low_conf_thresh=float(tracker_cfg.get('LOW_CONF_THRESH', 0.1)),
            gating_threshold=float(tracker_cfg.get('GATING_THRESHOLD', 9.0)),
            process_var=float(tracker_cfg.get('PROCESS_VAR', 1.0)),
            measurement_var=float(tracker_cfg.get('MEASUREMENT_VAR', 0.2)),
            reid_max_age=int(tracker_cfg.get('REID_MAX_AGE', 45)),
            use_mahalanobis=bool(tracker_cfg.get('USE_MAHALANOBIS', True)),
            dist_threshold=float(tracker_cfg.get('DIST_THRESH', 1.5)),
            device=device,
        )
        print(
            "[Tracker] SimpleTrajectoryTracker 已启用：其所有参数均来自配置，"
            "无需任何额外训练，将直接在推理阶段消费 BEV 检测结果。"
        )

    out_dir = cfg['RUNTIME']['OUTPUT_DIR']
    frame_indices = []

    with torch.no_grad():
        for batch in dl:
            batch['images'] = batch['images'].to(device)
            for b in range(len(batch['calib']['intrinsic'])):
                batch['calib']['intrinsic'][b] = [k.to(device) for k in batch['calib']['intrinsic'][b]]
                batch['calib']['extrinsic'][b] = [e.to(device) for e in batch['calib']['extrinsic'][b]]
            preds = model(batch)
            frame_indices = [int(m['frame_idx']) for m in batch['meta']]

            if tracker is not None:
                per_frame = [
                    (frame_indices[i], preds['boxes'][i], preds['scores'][i])
                    for i in range(len(frame_indices))
                ]
                per_frame.sort(key=lambda x: x[0])
                for frame_idx, boxes, scores in per_frame:
                    tracker.update(frame_idx, boxes, scores)

            save_predictions_json(preds['boxes'], preds['scores'], out_dir, frame_indices)

    if tracker is not None:
        traj_path = os.path.join(out_dir, 'trajectories.json')
        save_trajectories_json(tracker.get_trajectories(include_active=True), traj_path)
        print(f"Saved predictions JSON to {out_dir} and trajectories to {traj_path}")
    else:
        print(f"Saved predictions JSON to {out_dir}")


if __name__ == '__main__':
    main()