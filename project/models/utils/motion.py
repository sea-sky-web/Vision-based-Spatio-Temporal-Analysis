import numpy as np
from typing import List, Dict, Any

Track = Dict[str, Any]

def predict_linear(tracks: List[Track]) -> np.ndarray:
    """
    Predicts the next BEV position for each track using a simple linear motion model.
    """
    predictions = []
    for track in tracks:
        if len(track['history']) < 2:
            # Not enough history, predict current location
            predictions.append(track['center'])
        else:
            # Velocity = current_pos - previous_pos
            velocity = np.array(track['center']) - np.array(track['history'][-2])
            prediction = np.array(track['center']) + velocity
            predictions.append(prediction)
    return np.array(predictions)
