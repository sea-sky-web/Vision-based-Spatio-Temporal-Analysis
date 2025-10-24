from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from project.models.tracker.geometry_matcher import GeometryMatcher
from project.models.tracker.reid_gallery import ReIDGallery
from project.models.utils.association import build_cost_matrix, match
# A placeholder for AppearanceHead, as it's a nn.Module and not needed for this logic test
from typing import Any
AppearanceHead = Any
from .track import Track

class Tracker:
    """
    Main logic controller for multi-object tracking.
    """
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.matcher = GeometryMatcher(motion_model=cfg.get('motion_model', 'linear'))
        self.reid = ReIDGallery(emb_dim=cfg['emb_dim'], alpha=cfg['reid_alpha'])

        # In a real scenario, this would be a torch model.
        # Here we mock it as it's not needed for the tracking logic itself.
        self.appearance_head = None

        self.tracks: List[Track] = []
        self.next_track_id: int = 1

    def _get_appearance_cost(self, detections: List[Dict]) -> np.ndarray:
        """Computes the appearance cost matrix (cosine similarity)."""
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        if n_tracks == 0 or n_dets == 0:
            return np.empty((n_tracks, n_dets))

        app_cost = np.zeros((n_tracks, n_dets))
        det_embeddings = [d['feature'] for d in detections]

        for i, track in enumerate(self.tracks):
            for j, det_emb in enumerate(det_embeddings):
                app_cost[i, j] = self.reid.similarity(track.track_id, det_emb)

        return app_cost

    def update(self, detections: List[Dict]):
        """
        Processes a new frame of detections and updates the state of all tracks.
        """
        # 1. Prediction & Cost Matrix Calculation
        space_cost = self.matcher.spatial_cost(self.tracks, detections)
        app_cost = self._get_appearance_cost(detections)

        # 2. Association
        final_cost = build_cost_matrix(
            space_cost, app_cost, self.cfg['space_weight'], self.cfg['app_weight']
        )

        matches, u_tracks, u_dets = match(
            final_cost, space_cost, app_cost,
            self.cfg['space_thresh'], self.cfg['app_thresh']
        )

        # 3. Update Matched Tracks
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            detection = detections[d_idx]
            embedding = detection['feature']

            track.update(detection, embedding)
            self.reid.update(track.track_id, embedding)

        # 4. Handle Unmatched Tracks
        for t_idx in u_tracks:
            self.tracks[t_idx].mark_missed()

        # 5. Create New Tracks for Unmatched Detections
        for d_idx in u_dets:
            detection = detections[d_idx]
            embedding = detection['feature']

            new_track = Track(self.next_track_id, detection, embedding)
            self.tracks.append(new_track)
            self.reid.update(self.next_track_id, embedding)
            self.next_track_id += 1

        # 6. Prune Dead Tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.cfg['max_age']]

        # 7. Return active tracks
        return self.tracks

# Example Usage (for verification)
if __name__ == '__main__':
    # --- Configuration ---
    config = {
        'emb_dim': 2, 'max_age': 1, 'space_weight': 0.6, 'app_weight': 0.4,
        'space_thresh': 2.0, 'app_thresh': 0.8, 'reid_alpha': 0.2,
        'motion_model': 'linear'
    }

    # --- Initialize Tracker ---
    tracker = Tracker(config)
    print("Tracker Initialized.\n")

    # --- Frame 1 ---
    print("--- Processing Frame 1 ---")
    f1_dets = [
        {'center': (10, 10), 'score': 0.9, 'feature': torch.tensor([1.0, 0.0])}, # Becomes T1
        {'center': (30, 30), 'score': 0.9, 'feature': torch.tensor([0.0, 1.0])}  # Becomes T2
    ]
    active_tracks = tracker.update(f1_dets)
    assert len(active_tracks) == 2
    assert tracker.next_track_id == 3
    print(f"Active tracks: {[t.track_id for t in active_tracks]}")
    print("✓ Frame 1: Correctly created 2 new tracks.\n")

    # --- Frame 2 ---
    print("--- Processing Frame 2 ---")
    # T1 moves slightly, T2 disappears, a new detection appears
    f2_dets = [
        {'center': (11, 11), 'score': 0.9, 'feature': F.normalize(torch.tensor([0.9, 0.1]), p=2, dim=0)}, # Matches T1
        {'center': (50, 50), 'score': 0.9, 'feature': torch.tensor([-1.0, 0.0])} # Becomes T3
    ]
    active_tracks = tracker.update(f2_dets)

    # Expected: T1 is updated, T2 is marked as missed, T3 is created
    track_map = {t.track_id: t for t in active_tracks}
    assert len(active_tracks) == 3
    assert 1 in track_map and track_map[1].misses == 0 and track_map[1].center == (11,11)
    assert 2 in track_map and track_map[2].misses == 1
    assert 3 in track_map
    print(f"Active tracks: {[t.track_id for t in active_tracks]}")
    print("✓ Frame 2: Correctly updated T1, missed T2, and created T3.\n")

    # --- Frame 3 ---
    print("--- Processing Frame 3 ---")
    # T1 disappears, T2 reappears nearby its original spot, T3 moves
    f3_dets = [
        {'center': (31, 31), 'score': 0.9, 'feature': F.normalize(torch.tensor([0.1, 0.9]), p=2, dim=0)}, # Matches T2
        {'center': (51.5, 50.5), 'score': 0.9, 'feature': F.normalize(torch.tensor([-0.9, 0.1]), p=2, dim=0)} # Matches T3
    ]
    active_tracks = tracker.update(f3_dets)

    # Expected: T1 is missed, T2 is re-identified, T3 is updated. T1 should still be alive (misses=1)
    track_map = {t.track_id: t for t in active_tracks}
    assert len(active_tracks) == 3
    assert 1 in track_map and track_map[1].misses == 1
    assert 2 in track_map and track_map[2].misses == 0 and track_map[2].center == (31,31)
    assert 3 in track_map and track_map[3].misses == 0
    print(f"Active tracks: {[t.track_id for t in active_tracks]}")
    print("✓ Frame 3: Correctly missed T1, re-identified T2, and updated T3.\n")

    # --- Frame 4 ---
    print("--- Processing Frame 4 ---")
    # T1 and T2 disappear. T3 moves.
    f4_dets = [
        {'center': (54, 52), 'score': 0.9, 'feature': F.normalize(torch.tensor([-0.8, 0.2]), p=2, dim=0)} # Matches T3
    ]
    active_tracks = tracker.update(f4_dets)

    # Expected: T1 is deleted (misses > max_age), T2 is missed, T3 is updated
    track_map = {t.track_id: t for t in active_tracks}
    assert len(active_tracks) == 2
    assert 1 not in track_map
    assert 2 in track_map and track_map[2].misses == 1
    assert 3 in track_map and track_map[3].misses == 0
    print(f"Active tracks: {[t.track_id for t in active_tracks]}")
    print("✓ Frame 4: Correctly pruned T1, missed T2, and updated T3.\n")

    print("Tracker module implementation is verified.")
