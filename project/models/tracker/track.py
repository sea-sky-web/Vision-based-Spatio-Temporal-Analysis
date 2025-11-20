from typing import List, Dict, Tuple
import torch

class Track:
    """
    Represents a single tracked object, managing its state over time.
    """
    def __init__(self, track_id: int, initial_detection: Dict, initial_embedding: torch.Tensor):
        self.track_id = track_id
        self.center: Tuple[float, float] = initial_detection['center']
        self.box: List[float] = initial_detection.get('box', [])
        self.score: float = initial_detection['score']

        self.history: List[Tuple[float, float]] = [self.center]
        self.velocity: Tuple[float, float] = (0.0, 0.0)

        self.age: int = 1
        self.misses: int = 0

        self.embedding: torch.Tensor = initial_embedding

    def update(self, detection: Dict, embedding: torch.Tensor):
        """Updates the track state with a new detection."""
        self.center = detection['center']
        self.box = detection.get('box', [])
        self.score = detection['score']

        self.history.append(self.center)
        if len(self.history) > 1:
            p1 = self.history[-2]
            p2 = self.history[-1]
            self.velocity = (p2[0] - p1[0], p2[1] - p1[1])

        self.embedding = embedding # Update with the latest embedding
        self.age += 1
        self.misses = 0 # Reset miss counter on successful update

    def mark_missed(self):
        """Increments the miss counter."""
        self.misses += 1
