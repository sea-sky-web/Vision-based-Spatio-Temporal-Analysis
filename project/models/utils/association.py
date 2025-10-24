import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Set

def build_cost_matrix(space_cost: np.ndarray, app_cost: np.ndarray, w_space: float = 0.6, w_app: float = 0.4) -> np.ndarray:
    """
    Builds the combined cost matrix from spatial and appearance costs.
    :param space_cost: Spatial distance matrix.
    :param app_cost: Appearance similarity matrix (higher is better).
    :param w_space: Weight for the spatial cost.
    :param w_app: Weight for the appearance cost.
    :return: The combined cost matrix.
    """
    # Appearance cost is a similarity, so we convert it to a distance
    app_dist = 1 - app_cost

    # Combine the costs
    combined_cost = w_space * space_cost + w_app * app_dist
    return combined_cost

def match(cost_matrix: np.ndarray, space_thresh: float, app_thresh: float, space_cost: np.ndarray, app_cost: np.ndarray) -> Tuple[np.ndarray, Set[int], Set[int]]:
    """
    Performs matching using the Hungarian algorithm and applies thresholds.
    :param cost_matrix: The combined cost matrix.
    :param space_thresh: Maximum allowed spatial distance for a match.
    :param app_thresh: Minimum required appearance similarity for a match.
    :param space_cost: Original spatial cost matrix for thresholding.
    :param app_cost: Original appearance cost matrix for thresholding.
    :return: A tuple containing:
             - matched_indices: An array of (track_idx, det_idx) pairs.
             - unmatched_track_indices: A set of indices for unmatched tracks.
             - unmatched_det_indices: A set of indices for unmatched detections.
    """
    if cost_matrix.size == 0:
        return np.array([]), set(range(cost_matrix.shape[0])), set(range(cost_matrix.shape[1]))

    # Use the Hungarian algorithm to find the optimal assignment
    track_indices, det_indices = linear_sum_assignment(cost_matrix)

    matched_indices = []
    unmatched_track_indices = set(range(cost_matrix.shape[0]))
    unmatched_det_indices = set(range(cost_matrix.shape[1]))

    for track_idx, det_idx in zip(track_indices, det_indices):
        # Apply thresholds to filter out bad matches
        if space_cost[track_idx, det_idx] < space_thresh and app_cost[track_idx, det_idx] > app_thresh:
            matched_indices.append((track_idx, det_idx))
            unmatched_track_indices.discard(track_idx)
            unmatched_det_indices.discard(det_idx)

    return np.array(matched_indices), unmatched_track_indices, unmatched_det_indices
