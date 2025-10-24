import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple, List

def build_cost_matrix(
    space_cost: np.ndarray,
    app_cost: np.ndarray,
    w_space: float = 0.6,
    w_app: float = 0.4
) -> np.ndarray:
    """
    Builds a combined cost matrix for matching.

    Args:
        space_cost (np.ndarray): Cost matrix based on spatial distance (lower is better).
        app_cost (np.ndarray): Cost matrix based on appearance similarity (higher is better).
        w_space (float): Weight for the spatial cost.
        w_app (float): Weight for the appearance cost.

    Returns:
        np.ndarray: The combined cost matrix where lower values indicate a better match.
    """
    if space_cost.shape != app_cost.shape:
        raise ValueError("Spatial and appearance cost matrices must have the same shape.")

    # Invert appearance cost because linear_sum_assignment minimizes cost
    # A higher similarity (app_cost) should result in a lower final cost.
    inverted_app_cost = 1 - app_cost

    return w_space * space_cost + w_app * inverted_app_cost

def match(
    cost_matrix: np.ndarray,
    space_cost: np.ndarray,
    app_cost: np.ndarray,
    space_thresh: float,
    app_thresh: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Performs optimal matching using the Hungarian algorithm and filters results
    based on spatial and appearance thresholds.

    Args:
        cost_matrix (np.ndarray): The combined cost matrix (N_tracks, M_dets).
        space_cost (np.ndarray): The original spatial cost matrix.
        app_cost (np.ndarray): The original appearance similarity matrix.
        space_thresh (float): Maximum allowed spatial distance for a match.
        app_thresh (float): Minimum required appearance similarity for a match.

    Returns:
        Tuple[List[Tuple[int, int]], List[int], List[int]]:
            - A list of matched (track_idx, det_idx) pairs.
            - A list of unmatched track indices.
            - A list of unmatched detection indices.
    """
    if cost_matrix.size == 0:
        n_tracks, n_dets = cost_matrix.shape
        return [], list(range(n_tracks)), list(range(n_dets))

    # Use the Hungarian algorithm to find the optimal assignment
    track_indices, det_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = []
    unmatched_detections = []

    # Filter the assignments based on the thresholds
    for t_idx, d_idx in zip(track_indices, det_indices):
        valid_space = space_cost[t_idx, d_idx] <= space_thresh
        valid_app = app_cost[t_idx, d_idx] >= app_thresh

        if valid_space and valid_app:
            matches.append((t_idx, d_idx))
        else:
            # The match is invalid, so both are considered unmatched
            unmatched_tracks.append(t_idx)
            unmatched_detections.append(d_idx)

    # Collect all track indices that were not part of any assignment
    assigned_track_indices = {m[0] for m in matches}
    all_track_indices = set(range(cost_matrix.shape[0]))
    unmatched_tracks.extend(list(all_track_indices - assigned_track_indices - set(unmatched_tracks)))

    # Collect all detection indices that were not part of any assignment
    assigned_det_indices = {m[1] for m in matches}
    all_det_indices = set(range(cost_matrix.shape[1]))
    unmatched_detections.extend(list(all_det_indices - assigned_det_indices - set(unmatched_detections)))

    return sorted(matches), sorted(unmatched_tracks), sorted(unmatched_detections)


# Example Usage (for verification)
if __name__ == '__main__':
    # Define weights and thresholds
    w_s, w_a = 0.6, 0.4
    s_thresh, a_thresh = 2.0, 0.6

    # --- Test Case ---
    # T1 should match D1 (low space, high app)
    # T2 should NOT match D2 (low space, but low app)
    # T3 should NOT match D3 (high space, high app)
    # D3 should be unmatched

    # Tracks: T1, T2, T3
    # Detections: D1, D2, D3
    space_costs = np.array([
        [1.0, 5.0, 8.0],  # T1 costs to D1, D2, D3
        [6.0, 1.5, 9.0],  # T2 costs to D1, D2, D3
        [7.0, 4.0, 2.1]   # T3 costs to D1, D2, D3 -> space_cost > thresh
    ])
    app_similarities = np.array([
        [0.9, 0.2, 0.1],  # T1 similarities to D1, D2, D3
        [0.3, 0.4, 0.2],  # T2 similarities to D1, D2, D3
        [0.1, 0.5, 0.95]  # T3 similarities to D1, D2, D3
    ])

    print("--- Verification ---")
    print(f"Space Threshold: {s_thresh}, App Threshold: {a_thresh}\n")

    # 1. Build cost matrix
    final_cost_matrix = build_cost_matrix(space_costs, app_similarities, w_s, w_a)
    print("Combined Cost Matrix:\n", np.round(final_cost_matrix, 2))

    # Expected cost for T1-D1: 0.6*1.0 + 0.4*(1-0.9) = 0.6 + 0.04 = 0.64
    assert np.isclose(final_cost_matrix[0,0], 0.64)
    print("\n✓ Cost matrix calculation is correct.")

    # 2. Perform matching
    matches, u_tracks, u_dets = match(
        final_cost_matrix, space_costs, app_similarities, s_thresh, a_thresh
    )

    print("\n--- Results ---")
    print(f"Matches: {matches}")
    print(f"Unmatched Tracks: {u_tracks}")
    print(f"Unmatched Detections: {u_dets}")

    # 3. Verify results
    expected_matches = [(0, 0)] # T1 -> D1
    expected_unmatched_tracks = [1, 2] # T2, T3
    expected_unmatched_detections = [1, 2] # D2, D3

    assert matches == expected_matches, f"Expected matches {expected_matches}, got {matches}"
    assert u_tracks == expected_unmatched_tracks, f"Expected u_tracks {expected_unmatched_tracks}, got {u_tracks}"
    assert u_dets == expected_unmatched_detections, f"Expected u_dets {expected_unmatched_detections}, got {u_dets}"

    print("\n✓ Association logic is verified.")
