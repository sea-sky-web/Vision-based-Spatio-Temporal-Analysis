import numpy as np
import torch
from typing import List, Dict, Any, Optional

from ..utils.association import build_cost_matrix, match
from .geometry_matcher import GeometryMatcher
from .reid_gallery import ReIDGallery

class Tracker:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg.get('tracker', {})

        # Initialize the core components
        self.matcher = GeometryMatcher(motion_model=self.cfg.get('motion_model', 'linear'))
        self.reid = ReIDGallery(emb_dim=self.cfg.get('emb_dim', 128), alpha=self.cfg.get('reid_alpha', 0.2))

        # Track management
        self.tracks: List[Dict[str, Any]] = []
        self.next_track_id = 0

    def _create_new_track(self, detection: Dict[str, Any], frame_idx: int) -> Dict[str, Any]:
        """Initializes a new track from a detection."""
        new_track = {
            "track_id": self.next_track_id,
            "center": detection['center'],
            "box": detection['box'],
            "score": detection['score'],
            "embedding": detection['feature'],
            "history": [detection['center']],
            "age": 1,
            "misses": 0,
            "frame_id": frame_idx
        }
        self.reid.update(self.next_track_id, detection['feature'])
        self.next_track_id += 1
        return new_track

    def update(self, detections: List[Dict[str, Any]], frame_idx: int) -> List[Dict[str, Any]]:
        """
        Main update function for the tracker.
        :param detections: A list of detections for the current frame.
        :param frame_idx: The current frame index.
        :return: A list of active tracks.
        """
        if not self.tracks:
            # If no tracks exist, initialize with all detections
            self.tracks = [self._create_new_track(det, frame_idx) for det in detections]
            return self.tracks

        # 1. Predict new locations for existing tracks
        active_tracks = self.tracks

        # 2. Calculate cost matrices
        space_cost = self.matcher.spatial_cost(active_tracks, detections)

        track_ids = [t['track_id'] for t in active_tracks]
        det_embs = torch.stack([d['feature'] for d in detections])
        app_cost = self.reid.batch_similarity(track_ids, det_embs).cpu().numpy()

        cost_matrix = build_cost_matrix(space_cost, app_cost,
                                        w_space=self.cfg.get('space_weight', 0.6),
                                        w_app=self.cfg.get('app_weight', 0.4))

        # 3. Match tracks and detections
        matched_indices, unmatched_track_indices, unmatched_det_indices = \
            match(cost_matrix,
                  space_thresh=self.cfg.get('space_thresh', 2.0),
                  app_thresh=self.cfg.get('app_thresh', 0.6),
                  space_cost=space_cost, app_cost=app_cost)

        # 4. Update track states
        # Update matched tracks
        for track_idx, det_idx in matched_indices:
            track = active_tracks[track_idx]
            detection = detections[det_idx]

            track['center'] = detection['center']
            track['box'] = detection['box']
            track['score'] = detection['score']
            track['history'].append(detection['center'])
            track['age'] += 1
            track['misses'] = 0
            track['frame_id'] = frame_idx

            self.reid.update(track['track_id'], detection['feature'])

        # Handle unmatched tracks
        for track_idx in unmatched_track_indices:
            self.tracks[track_idx]['misses'] += 1

        # Create new tracks for unmatched detections
        for det_idx in unmatched_det_indices:
            self.tracks.append(self._create_new_track(detections[det_idx], frame_idx))

        # 5. Prune dead tracks
        dead_tracks = [t for t in self.tracks if t['misses'] > self.cfg.get('max_age', 5)]
        for track in dead_tracks:
            self.reid.remove(track['track_id'])

        self.tracks = [t for t in self.tracks if t['misses'] <= self.cfg.get('max_age', 5)]

        return self.tracks
