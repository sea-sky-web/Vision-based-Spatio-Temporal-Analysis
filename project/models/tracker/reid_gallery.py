from typing import Dict
import torch

class ReIDGallery:
    """
    Maintains a gallery of appearance embeddings for each track, updated via
    an Exponential Moving Average (EMA), and computes similarity scores.
    """
    def __init__(self, emb_dim: int = 128, alpha: float = 0.2):
        """
        Initializes the ReIDGallery.

        Args:
            emb_dim (int): The dimension of the appearance embeddings.
            alpha (float): The update factor for the EMA. A lower alpha means
                           more weight is given to the historical embedding.
                           ema_embedding = (1 - alpha) * old + alpha * new
        """
        self.emb_dim = emb_dim
        self.alpha = alpha
        self.galleries: Dict[int, torch.Tensor] = {}

    def update(self, track_id: int, embedding: torch.Tensor):
        """
        Updates the gallery for a given track with a new embedding.

        If the track is new, the embedding is added directly.
        If the track already exists, its gallery is updated using EMA.

        Args:
            track_id (int): The ID of the track to update.
            embedding (torch.Tensor): The new appearance embedding of shape (D,).
        """
        if embedding.dim() != 1 or embedding.shape[0] != self.emb_dim:
            raise ValueError(f"Embedding must be a 1D tensor of size {self.emb_dim}")

        if track_id not in self.galleries:
            # Add new track to the gallery
            self.galleries[track_id] = embedding.clone()
        else:
            # Update existing track's gallery using EMA
            self.galleries[track_id] = (1 - self.alpha) * self.galleries[track_id] + \
                                       self.alpha * embedding

    def similarity(self, track_id: int, det_emb: torch.Tensor) -> float:
        """
        Computes the cosine similarity between a track's gallery and a new
        detection's embedding.

        Args:
            track_id (int): The ID of the track in the gallery.
            det_emb (torch.Tensor): The embedding of the new detection (shape D,).

        Returns:
            float: The cosine similarity, a value between -1 and 1.
                   Returns -1.0 if the track_id is not in the gallery.
        """
        if track_id not in self.galleries:
            return -1.0

        gallery_emb = self.galleries[track_id]

        # Cosine similarity: (A . B) / (||A|| * ||B||)
        # Assuming embeddings are L2 normalized, this simplifies to A . B
        similarity = torch.dot(gallery_emb, det_emb).item()

        return similarity

# Example Usage (for verification)
if __name__ == '__main__':
    emb_dim = 128
    alpha = 0.2

    # Initialize the gallery
    gallery = ReIDGallery(emb_dim=emb_dim, alpha=alpha)
    print("ReIDGallery initialized.")

    # Create dummy embeddings (already L2 normalized)
    emb1 = torch.randn(emb_dim)
    emb1 = emb1 / torch.linalg.norm(emb1)

    # Create an embedding that is similar to emb1
    emb2_similar = emb1.clone() + torch.randn(emb_dim) * 0.1
    emb2_similar = emb2_similar / torch.linalg.norm(emb2_similar)

    # Create a dissimilar embedding
    emb3_dissimilar = torch.randn(emb_dim)
    emb3_dissimilar = emb3_dissimilar / torch.linalg.norm(emb3_dissimilar)

    # --- Test Case 1: Add a new track ---
    track_id = 1
    gallery.update(track_id, emb1)
    print(f"\nTrack {track_id} added to gallery.")
    assert torch.equal(gallery.galleries[track_id], emb1), "Failed to add new track."
    print("✓ New track addition is correct.")

    # --- Test Case 2: Update an existing track with a similar embedding ---
    gallery.update(track_id, emb2_similar)
    expected_ema = (1 - alpha) * emb1 + alpha * emb2_similar
    print(f"Track {track_id} updated with a similar embedding.")
    assert torch.allclose(gallery.galleries[track_id], expected_ema), "EMA update failed."
    print("✓ EMA update is correct.")

    # --- Test Case 3: Calculate similarity ---
    # The gallery is now a mix of emb1 and emb2_similar, so it should be
    # highly similar to both, and dissimilar to emb3.
    sim_with_similar = gallery.similarity(track_id, emb2_similar)
    sim_with_dissimilar = gallery.similarity(track_id, emb3_dissimilar)

    print(f"\nSimilarity with a similar vector: {sim_with_similar:.4f}")
    print(f"Similarity with a dissimilar vector: {sim_with_dissimilar:.4f}")

    assert sim_with_similar > sim_with_dissimilar, "Similarity logic is incorrect."
    print("✓ Similarity calculation is correct.")

    # --- Test Case 4: Non-existent track ---
    non_existent_sim = gallery.similarity(999, emb1)
    print(f"\nSimilarity with non-existent track 999: {non_existent_sim}")
    assert non_existent_sim == -1.0, "Non-existent track should return -1.0 similarity."
    print("✓ Handling of non-existent tracks is correct.")

    print("\nReIDGallery module implementation is verified.")
