from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover - optional dependency
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover - fallback to greedy if SciPy missing
    linear_sum_assignment = None


class TrackState(Enum):
    TRACKED = 'tracked'
    LOST = 'lost'


class KalmanFilter2D:
    """Constant-velocity Kalman filter working on (cx, cy, vx, vy)."""

    def __init__(self, process_var: float = 1.0, measurement_var: float = 1.0):
        self.process_var = float(process_var)
        self.measurement_var = float(measurement_var)

    @staticmethod
    def _motion_mat(dt: int, device: torch.device) -> torch.Tensor:
        F = torch.eye(4, device=device, dtype=torch.float32)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _process_cov(self, dt: int, device: torch.device) -> torch.Tensor:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = self.process_var
        cov = torch.tensor(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            device=device,
            dtype=torch.float32,
        )
        return cov * q

    @staticmethod
    def _measurement_mat(device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            device=device,
            dtype=torch.float32,
        )

    def _measurement_cov(self, device: torch.device) -> torch.Tensor:
        return torch.eye(2, device=device, dtype=torch.float32) * self.measurement_var

    def predict(self, mean: torch.Tensor, cov: torch.Tensor, dt: int) -> Tuple[torch.Tensor, torch.Tensor]:
        F = self._motion_mat(dt, mean.device)
        mean = F @ mean
        cov = F @ cov @ F.t() + self._process_cov(dt, mean.device)
        return mean, cov

    def update(self, mean: torch.Tensor, cov: torch.Tensor, measurement: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = mean.device
        H = self._measurement_mat(device)
        R = self._measurement_cov(device)
        S = H @ cov @ H.t() + R
        K = cov @ H.t() @ torch.linalg.inv(S)
        innovation = measurement - (H @ mean)
        mean = mean + K @ innovation
        I = torch.eye(mean.shape[0], device=device, dtype=torch.float32)
        cov = (I - K @ H) @ cov
        return mean, cov

    def mahalanobis_distance(self, mean: torch.Tensor, cov: torch.Tensor, measurement: torch.Tensor) -> float:
        device = mean.device
        H = self._measurement_mat(device)
        R = self._measurement_cov(device)
        projected_mean = H @ mean
        S = H @ cov @ H.t() + R
        diff = measurement - projected_mean
        solved = torch.linalg.solve(S, diff.unsqueeze(-1))
        dist = diff.unsqueeze(0) @ solved
        return float(dist.squeeze())


class SimpleTrajectoryTracker:
    """Multi-stage BEV tracker with Kalman prediction and re-identification buffer."""

    def __init__(
        self,
        max_age: int = 10,
        min_hits: int = 3,
        size_alpha: float = 0.25,
        high_conf_thresh: float = 0.6,
        low_conf_thresh: float = 0.1,
        gating_threshold: float = 9.0,
        process_var: float = 1.0,
        measurement_var: float = 0.2,
        reid_max_age: Optional[int] = None,
        use_mahalanobis: bool = True,
        dist_threshold: float = 1.5,
        device: Optional[torch.device] = None,
    ):
        self.max_age = int(max(1, max_age))
        self.min_hits = int(max(1, min_hits))
        self.size_alpha = float(size_alpha)
        self.high_conf_thresh = float(high_conf_thresh)
        self.low_conf_thresh = float(low_conf_thresh)
        self.gating_threshold = float(gating_threshold)
        self.use_mahalanobis = bool(use_mahalanobis)
        self.dist_threshold = float(dist_threshold)
        self.device = device if device is not None else torch.device('cpu')
        self.reid_max_age = int(max_age if reid_max_age is None else max(reid_max_age, max_age))
        self.kf = KalmanFilter2D(process_var=process_var, measurement_var=measurement_var)
        self.reset()

    def reset(self):
        self.active_tracks: Dict[int, Dict] = {}
        self.finished_tracks: List[Dict] = []
        self._next_id = 1

    def _history_entry(self, frame_idx: int, box: torch.Tensor, score: float) -> Dict:
        return {
            'frame_idx': int(frame_idx),
            'cx': float(box[0]),
            'cy': float(box[1]),
            'w': float(box[2]),
            'h': float(box[3]),
            'score': float(score),
        }

    def _new_track(self, frame_idx: int, box: torch.Tensor, score: float) -> Dict:
        mean = torch.zeros(4, device=self.device, dtype=torch.float32)
        mean[:2] = box[:2]
        cov = torch.eye(4, device=self.device, dtype=torch.float32)
        track = {
            'track_id': self._next_id,
            'mean': mean,
            'cov': cov,
            'size': box[2:].clone(),
            'score': float(score),
            'hits': 1,
            'age': 1,
            'time_since_update': 0,
            'state': TrackState.TRACKED,
            'start_frame': int(frame_idx),
            'last_frame': int(frame_idx),
            'last_obs': int(frame_idx),
            'history': [self._history_entry(frame_idx, box, score)],
        }
        self.active_tracks[self._next_id] = track
        self._next_id += 1
        return track

    def _predict_track(self, track: Dict, frame_idx: int):
        dt = max(1, frame_idx - track['last_frame'])
        track['mean'], track['cov'] = self.kf.predict(track['mean'], track['cov'], dt)
        track['age'] += dt
        track['time_since_update'] += dt
        track['last_frame'] = int(frame_idx)

    def _update_track(self, track: Dict, frame_idx: int, det_box: torch.Tensor, det_score: torch.Tensor):
        measurement = det_box[:2]
        track['mean'], track['cov'] = self.kf.update(track['mean'], track['cov'], measurement)
        track['size'] = (1 - self.size_alpha) * track['size'] + self.size_alpha * det_box[2:]
        track['score'] = float(det_score)
        track['hits'] += 1
        track['time_since_update'] = 0
        track['state'] = TrackState.TRACKED
        track['last_obs'] = int(frame_idx)
        track['history'].append(self._history_entry(frame_idx, det_box, det_score))

    def _associate(
        self, tracks: Sequence[Dict], detections: Sequence[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = torch.zeros((len(tracks), len(detections)), device=self.device, dtype=torch.float32)
        for t_idx, track in enumerate(tracks):
            for d_idx, det in enumerate(detections):
                det_center = det['box'][:2]
                if self.use_mahalanobis:
                    cost = self.kf.mahalanobis_distance(track['mean'], track['cov'], det_center)
                else:
                    cost = torch.sum((track['mean'][:2] - det_center) ** 2).item()
                cost_matrix[t_idx, d_idx] = float(cost)

        if linear_sum_assignment is not None:
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        else:
            row_ind, col_ind = self._greedy_assignment(cost_matrix)

        matches: List[Tuple[int, int]] = []
        used_tracks = set()
        used_dets = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c].item() > (self.gating_threshold if self.use_mahalanobis else self.dist_threshold):
                continue
            matches.append((int(r), int(c)))
            used_tracks.add(int(r))
            used_dets.add(int(c))

        unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    @staticmethod
    def _greedy_assignment(cost_matrix: torch.Tensor) -> Tuple[List[int], List[int]]:
        order = torch.argsort(cost_matrix.flatten())
        num_rows, num_cols = cost_matrix.shape
        used_rows, used_cols = set(), set()
        rows, cols = [], []
        for idx in order:
            r = int(idx // num_cols)
            c = int(idx % num_cols)
            if r in used_rows or c in used_cols:
                continue
            rows.append(r)
            cols.append(c)
            used_rows.add(r)
            used_cols.add(c)
        return rows, cols

    def update(self, frame_idx: int, boxes, scores) -> List[Dict]:
        if boxes is None or (isinstance(boxes, torch.Tensor) and boxes.numel() == 0):
            boxes_t = torch.zeros((0, 4), device=self.device, dtype=torch.float32)
            scores_t = torch.zeros((0,), device=self.device, dtype=torch.float32)
        else:
            boxes_t = torch.as_tensor(boxes, device=self.device, dtype=torch.float32)
            scores_t = torch.as_tensor(scores, device=self.device, dtype=torch.float32)

        for track in self.active_tracks.values():
            self._predict_track(track, frame_idx)

        detections = [
            {'box': boxes_t[i], 'score': scores_t[i]}
            for i in range(boxes_t.shape[0])
        ]
        high_det_indices = [i for i, det in enumerate(detections) if det['score'] >= self.high_conf_thresh]
        low_det_indices = [
            i for i, det in enumerate(detections)
            if self.low_conf_thresh <= det['score'] < self.high_conf_thresh
        ]

        tracked_ids = [tid for tid, t in self.active_tracks.items() if t['state'] == TrackState.TRACKED]
        tracked_tracks = [self.active_tracks[tid] for tid in tracked_ids]
        high_dets = [detections[i] for i in high_det_indices]
        matches_high, unmatched_tracked_idx, unmatched_high_idx = self._associate(tracked_tracks, high_dets)

        for local_track_idx, local_det_idx in matches_high:
            track = tracked_tracks[local_track_idx]
            det = high_dets[local_det_idx]
            self._update_track(track, frame_idx, det['box'], det['score'])

        unmatched_high_global = [high_det_indices[i] for i in unmatched_high_idx]
        for det_idx in unmatched_high_global:
            if detections:
                self._new_track(frame_idx, detections[det_idx]['box'], detections[det_idx]['score'])

        unmatched_tracked_ids = {tracked_ids[i] for i in unmatched_tracked_idx}

        reid_track_ids: List[int] = list(unmatched_tracked_ids)
        reid_track_ids.extend(
            [
                tid
                for tid, t in self.active_tracks.items()
                if t['state'] == TrackState.LOST and tid not in reid_track_ids
            ]
        )
        low_dets = [detections[i] for i in low_det_indices]
        reid_tracks = [self.active_tracks[tid] for tid in reid_track_ids]
        matches_low, _, _ = self._associate(reid_tracks, low_dets)

        for local_track_idx, local_det_idx in matches_low:
            track_id = reid_track_ids[local_track_idx]
            track = self.active_tracks[track_id]
            det = low_dets[local_det_idx]
            self._update_track(track, frame_idx, det['box'], det['score'])
            if track_id in unmatched_tracked_ids:
                unmatched_tracked_ids.remove(track_id)

        for tid in unmatched_tracked_ids:
            track = self.active_tracks[tid]
            track['state'] = TrackState.LOST

        expired: List[int] = []
        for tid, track in self.active_tracks.items():
            if track['time_since_update'] > self.reid_max_age:
                expired.append(tid)
            elif track['time_since_update'] > self.max_age:
                track['state'] = TrackState.LOST

        for tid in expired:
            self.finished_tracks.append(self._serialize_track(self.active_tracks[tid]))
            self.active_tracks.pop(tid, None)

        frame_tracks = []
        for track in self.active_tracks.values():
            if track['state'] != TrackState.TRACKED:
                continue
            if track['last_obs'] != frame_idx:
                continue
            if track['hits'] < self.min_hits:
                continue
            last_obs = track['history'][-1]
            frame_tracks.append(
                {
                    'track_id': track['track_id'],
                    'frame_idx': frame_idx,
                    'cx': last_obs['cx'],
                    'cy': last_obs['cy'],
                    'w': last_obs['w'],
                    'h': last_obs['h'],
                    'score': last_obs['score'],
                }
            )
        return frame_tracks

    def _serialize_track(self, track: Dict) -> Dict:
        end_frame = track['history'][-1]['frame_idx'] if track['history'] else track['last_frame']
        return {
            'track_id': int(track['track_id']),
            'start_frame': int(track['start_frame']),
            'end_frame': int(end_frame),
            'age': int(track['age']),
            'hits': int(track['hits']),
            'history': list(track['history']),
        }

    def get_trajectories(self, include_active: bool = True) -> List[Dict]:
        tracks = list(self.finished_tracks)
        if include_active:
            tracks.extend([self._serialize_track(t) for t in self.active_tracks.values()])
        return tracks
