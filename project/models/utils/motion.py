from typing import List, Tuple
import numpy as np
# Import the canonical Track class for the verification script
from project.models.tracker.track import Track

class LinearMotionModel:
    """
    A simple linear motion model that predicts the next position based on
    the most recently observed velocity.
    """
    def predict_next_position(self, track: Track) -> Tuple[float, float]:
        """
        Predicts the next BEV (x, y) position for a single track.

        Args:
            track (Track): The track object with position history and velocity.

        Returns:
            Tuple[float, float]: The predicted (x, y) coordinates.
        """
        if not track.history:
            # Cannot predict without any history
            raise ValueError("Track history is empty, cannot predict next position.")

        # Predict next position by adding the last known velocity
        last_pos = track.history[-1]
        next_pos = (last_pos[0] + track.velocity[0], last_pos[1] + track.velocity[1])

        return next_pos

# Example Usage (for verification)
if __name__ == '__main__':
    import torch

    # Create a dummy track using the canonical Track class
    initial_det = {'center': (10, 20), 'score': 0.9}
    initial_emb = torch.randn(128) # Dummy embedding
    track = Track(track_id=1, initial_detection=initial_det, initial_embedding=initial_emb)
    print(f"Initial Position: {track.center}")

    # Update track with a new detection
    update_det_1 = {'center': (12, 22), 'score': 0.9}
    track.update(update_det_1, initial_emb)
    print(f"After 1st update, Position: {track.center}, Velocity: {track.velocity}")

    # Update track with another detection
    update_det_2 = {'center': (15, 25), 'score': 0.9}
    track.update(update_det_2, initial_emb)
    print(f"After 2nd update, Position: {track.center}, Velocity: {track.velocity}")

    # Initialize the motion model
    motion_model = LinearMotionModel()

    # Predict the next position
    predicted_pos = motion_model.predict_next_position(track)
    print(f"\nPredicted Next Position: {predicted_pos}")

    # Verification
    expected_pos = (15 + 3, 25 + 3) # Last position + last velocity
    assert predicted_pos == expected_pos, \
        f"Prediction failed: expected {expected_pos}, got {predicted_pos}"
    print("✓ Linear motion prediction is correct.")

    # Test another prediction
    update_det_3 = {'center': (17, 26), 'score': 0.9}
    track.update(update_det_3, initial_emb)
    print(f"\nAfter 3rd update, Position: {track.center}, Velocity: {track.velocity}")
    predicted_pos = motion_model.predict_next_position(track)
    print(f"Predicted Next Position: {predicted_pos}")
    expected_pos = (17 + 2, 26 + 1)
    assert predicted_pos == expected_pos, \
        f"Prediction failed: expected {expected_pos}, got {predicted_pos}"
    print("✓ Linear motion prediction is correct.")
