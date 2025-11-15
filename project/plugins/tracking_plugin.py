from __future__ import annotations

import os
from typing import Any, Dict, List

from models.tracking import SimpleTrajectoryTracker
from utils.visualization import save_trajectories_json

from .base import InferencePlugin


class TrackingPlugin(InferencePlugin):
    """Wraps SimpleTrajectoryTracker as an inference-time plugin."""

    def __init__(self, cfg: Dict[str, Any], device, output_dir: str):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.tracker: SimpleTrajectoryTracker | None = None
        self.enabled = bool(cfg.get('ENABLED', True))
        self.traj_path = os.path.join(output_dir, 'trajectories.json')

    def on_start(self, context: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.tracker = SimpleTrajectoryTracker(
            max_age=int(self.cfg.get('MAX_AGE', 15)),
            min_hits=int(self.cfg.get('MIN_HITS', 3)),
            size_alpha=float(self.cfg.get('SIZE_ALPHA', 0.25)),
            high_conf_thresh=float(self.cfg.get('HIGH_CONF_THRESH', 0.6)),
            low_conf_thresh=float(self.cfg.get('LOW_CONF_THRESH', 0.1)),
            gating_threshold=float(self.cfg.get('GATING_THRESHOLD', 9.0)),
            process_var=float(self.cfg.get('PROCESS_VAR', 1.0)),
            measurement_var=float(self.cfg.get('MEASUREMENT_VAR', 0.2)),
            reid_max_age=int(self.cfg.get('REID_MAX_AGE', 45)),
            use_mahalanobis=bool(self.cfg.get('USE_MAHALANOBIS', True)),
            dist_threshold=float(self.cfg.get('DIST_THRESH', 1.5)),
            device=self.device,
        )
        print(
            "[Plugin:Tracking] SimpleTrajectoryTracker 已启用 (纯算法插件，"
            "在推理阶段消费 BEV 检测并生成 trajectories.json)。"
        )

    def on_batch(self, predictions: Dict[str, Any], batch: Dict[str, Any], context: Dict[str, Any]) -> None:
        if self.tracker is None:
            return
        frame_indices: List[int] = [int(m['frame_idx']) for m in batch['meta']]
        per_frame = [
            (frame_indices[i], predictions['boxes'][i], predictions['scores'][i])
            for i in range(len(frame_indices))
        ]
        per_frame.sort(key=lambda x: x[0])
        for frame_idx, boxes, scores in per_frame:
            self.tracker.update(frame_idx, boxes, scores)

    def on_finish(self, context: Dict[str, Any]) -> None:
        if self.tracker is None:
            return
        trajectories = self.tracker.get_trajectories(include_active=True)
        save_trajectories_json(trajectories, self.traj_path)
        print(f"[Plugin:Tracking] Trajectories saved to {self.traj_path}")
