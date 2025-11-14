import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def save_bev_heatmap(heatmap: torch.Tensor, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    hm = heatmap.detach().cpu().numpy()
    if hm.ndim == 4:
        hm = hm[0, 0]
    plt.figure(figsize=(4, 4))
    plt.imshow(hm, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_predictions_json(boxes_list: List[torch.Tensor], scores_list: List[torch.Tensor], save_dir: str, frame_indices: List[int]):
    os.makedirs(save_dir, exist_ok=True)
    for b, frame_idx in enumerate(frame_indices):
        boxes = boxes_list[b].detach().cpu().numpy().tolist() if boxes_list[b] is not None else []
        scores = scores_list[b].detach().cpu().numpy().tolist() if scores_list[b] is not None else []
        out = {"frame_idx": int(frame_idx), "boxes": boxes, "scores": scores}
        with open(os.path.join(save_dir, f"frame_{int(frame_idx):06d}.json"), 'w') as f:
            json.dump(out, f)


def save_trajectories_json(tracks: List[dict], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    serializable = []
    for track in tracks:
        history = [
            {
                'frame_idx': int(item['frame_idx']),
                'cx': float(item['cx']),
                'cy': float(item['cy']),
                'w': float(item['w']),
                'h': float(item['h']),
                'score': float(item['score']),
            }
            for item in track.get('history', [])
        ]
        serializable.append({
            'track_id': int(track['track_id']),
            'start_frame': int(track['start_frame']),
            'end_frame': int(track['end_frame']),
            'age': int(track['age']),
            'hits': int(track['hits']),
            'history': history,
        })
    with open(save_path, 'w') as f:
        json.dump(serializable, f)
