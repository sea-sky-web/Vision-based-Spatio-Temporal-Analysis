import numpy as np
from typing import List, Dict, Any

from ..utils.motion import predict_linear

# A simplified representation of a track for type hinting
Track = Dict[str, Any]
Detection = Dict[str, Any]

class GeometryMatcher:
    def __init__(self, motion_model: str = 'linear'):
        if motion_model == 'linear':
            self.motion_model = predict_linear
        else:
            raise ValueError(f"Unsupported motion model: {motion_model}")

    def predict_next(self, tracks: List[Track]) -> np.ndarray:
        """
        Predicts the next BEV position for each track.
        """
        return self.motion_model(tracks)

    def spatial_cost(self, tracks: List[Track], detections: List[Detection]) -> np.ndarray:
        """
        Calculates the spatial distance cost matrix between tracks and detections.
        Cost is the Euclidean distance between predicted track positions and detection centers.
        """
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)))

        predicted_positions = self.predict_next(tracks)
        detection_centers = np.array([d['center'] for d in detections])

        # Calculate pairwise Euclidean distance
        # Shape: (num_tracks, num_detections)
        cost_matrix = np.linalg.norm(predicted_positions[:, np.newaxis, :] - detection_centers[np.newaxis, :, :], axis=2)

        return cost_matrix
