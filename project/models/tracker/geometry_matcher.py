from typing import List, Dict
import numpy as np
from project.models.utils.motion import LinearMotionModel
from .track import Track

class GeometryMatcher:
    """
    Manages spatial prediction and computes the geometric cost matrix between
    tracks and detections.
    """
    def __init__(self, motion_model: str = 'linear'):
        """
        Initializes the GeometryMatcher.

        Args:
            motion_model (str): The motion model to use. Currently, only 'linear'
                                is supported.
        """
        if motion_model == 'linear':
            self.motion_model = LinearMotionModel()
        else:
            raise NotImplementedError(f"Motion model '{motion_model}' is not supported.")

    def predict_next_positions(self, tracks: List[Track]) -> np.ndarray:
        """
        Predicts the next position for a list of tracks.

        Args:
            tracks (List[Track]): A list of track objects.

        Returns:
            np.ndarray: A numpy array of shape (N_tracks, 2) with predicted (x, y)
                        coordinates.
        """
        predictions = []
        for track in tracks:
            predicted_pos = self.motion_model.predict_next_position(track)
            predictions.append(predicted_pos)
        return np.array(predictions)

    def spatial_cost(self, tracks: List[Track], detections: List[Dict]) -> np.ndarray:
        """
        Calculates the spatial cost matrix between predicted track positions and
        new detections. The cost is the Euclidean distance.

        Args:
            tracks (List[Track]): A list of N_tracks track objects.
            detections (List[Dict]): A list of M_detections, where each detection
                                     is a dict with a 'center' key.

        Returns:
            np.ndarray: An (N_tracks, M_detections) cost matrix where C[i, j] is
                        the Euclidean distance between the predicted position of
                        track i and the center of detection j.
        """
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)))

        predicted_positions = self.predict_next_positions(tracks)
        detection_centers = np.array([d['center'] for d in detections])

        # Expand dimensions to broadcast for pairwise distance calculation
        # predicted_positions: (N, 1, 2)
        # detection_centers: (1, M, 2)
        dist_matrix = np.linalg.norm(
            predicted_positions[:, np.newaxis, :] - detection_centers[np.newaxis, :, :],
            axis=2
        )

        return dist_matrix

# Example Usage (for verification)
if __name__ == '__main__':
    import torch

    # Create dummy tracks using the canonical Track class
    track1 = Track(track_id=1, initial_detection={'center': (10, 20), 'score': 0.9}, initial_embedding=torch.randn(128))
    track1.update(detection={'center': (12, 22), 'score': 0.9}, embedding=torch.randn(128))

    track2 = Track(track_id=2, initial_detection={'center': (30, 40), 'score': 0.9}, initial_embedding=torch.randn(128))
    track2.update(detection={'center': (28, 39), 'score': 0.9}, embedding=torch.randn(128))

    active_tracks = [track1, track2]

    # Create dummy detections
    current_detections = [
        {'center': (14.5, 24.5), 'score': 0.9}, # Should match track1
        {'center': (50, 60), 'score': 0.8},     # Unmatched
        {'center': (25.5, 37.5), 'score': 0.95} # Should match track2
    ]

    # Initialize the matcher
    geo_matcher = GeometryMatcher(motion_model='linear')

    # Predict next positions
    predicted_pos = geo_matcher.predict_next_positions(active_tracks)
    print("Predicted Positions:\n", predicted_pos)

    # Verification of predictions
    # Track 1 prediction: (12+2, 22+2) = (14, 24)
    # Track 2 prediction: (28-2, 39-1) = (26, 38)
    expected_predictions = np.array([[14., 24.], [26., 38.]])
    assert np.allclose(predicted_pos, expected_predictions), "Prediction mismatch."
    print("✓ Position prediction is correct.")

    # Calculate the spatial cost matrix
    cost_matrix = geo_matcher.spatial_cost(active_tracks, current_detections)
    print("\nSpatial Cost Matrix:\n", cost_matrix)

    # Verification of cost matrix
    det_centers = np.array([d['center'] for d in current_detections])
    dist1_1 = np.linalg.norm(expected_predictions[0] - det_centers[0]) # (14,24) vs (14.5, 24.5)
    dist1_3 = np.linalg.norm(expected_predictions[0] - det_centers[2]) # (14,24) vs (25.5, 37.5)
    dist2_1 = np.linalg.norm(expected_predictions[1] - det_centers[0]) # (26,38) vs (14.5, 24.5)
    dist2_3 = np.linalg.norm(expected_predictions[1] - det_centers[2]) # (26,38) vs (25.5, 37.5)

    assert np.isclose(cost_matrix[0, 0], dist1_1), "Cost matrix calculation error."
    assert np.isclose(cost_matrix[0, 2], dist1_3), "Cost matrix calculation error."
    assert np.isclose(cost_matrix[1, 0], dist2_1), "Cost matrix calculation error."
    assert np.isclose(cost_matrix[1, 2], dist2_3), "Cost matrix calculation error."
    print("✓ Spatial cost matrix calculation is correct.")
    print(f"\nExample cost (track1 -> det1): {cost_matrix[0,0]:.2f}")
    print(f"Example cost (track2 -> det3): {cost_matrix[1,2]:.2f}")

    print("\nGeometryMatcher module implementation is verified.")
