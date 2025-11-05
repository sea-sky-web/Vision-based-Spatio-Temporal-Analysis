import torch
import torch.nn.functional as F
from typing import Dict, List

class ReIDGallery:
    def __init__(self, emb_dim: int = 128, alpha: float = 0.2):
        """
        Initializes the ReID gallery.
        :param emb_dim: The dimension of the appearance embedding.
        :param alpha: The update rate for the Exponential Moving Average (EMA).
        """
        self.emb_dim = emb_dim
        self.alpha = alpha
        self.gallery: Dict[int, torch.Tensor] = {}

    def update(self, track_id: int, embedding: torch.Tensor):
        """
        Updates the gallery with a new embedding for a given track.
        If the track is new, it's added to the gallery.
        If the track already exists, its embedding is updated using EMA.
        """
        if embedding.dim() == 0 or embedding.nelement() == 0:
            return

        embedding = embedding.squeeze()
        if track_id in self.gallery:
            # Update using EMA
            self.gallery[track_id] = self.alpha * embedding + (1 - self.alpha) * self.gallery[track_id]
        else:
            # Initialize with the first embedding
            self.gallery[track_id] = embedding

        # Ensure the embedding remains L2 normalized
        self.gallery[track_id] = F.normalize(self.gallery[track_id], p=2, dim=0)

    def batch_similarity(self, track_ids: List[int], det_embs: torch.Tensor) -> torch.Tensor:
        """
        Calculates a matrix of cosine similarities between gallery embeddings for
        given track IDs and new detection embeddings.
        :param track_ids: A list of track IDs to compare.
        :param det_embs: A tensor of detection embeddings (num_detections, emb_dim).
        :return: A tensor of similarity scores (num_tracks, num_detections).
        """
        if not track_ids or det_embs.nelement() == 0:
            return torch.empty((len(track_ids), det_embs.shape[0]))

        gallery_embs_list = [self.gallery[tid] for tid in track_ids if tid in self.gallery]
        if not gallery_embs_list:
            return torch.empty((len(track_ids), det_embs.shape[0]))

        gallery_embs = torch.stack(gallery_embs_list)

        # Cosine similarity is the matrix product of L2-normalized vectors
        sim_matrix = torch.matmul(gallery_embs, det_embs.T)
        return sim_matrix

    def remove(self, track_id: int):
        """Removes a track from the gallery."""
        if track_id in self.gallery:
            del self.gallery[track_id]
