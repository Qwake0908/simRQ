from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .distance import DistanceFunction

class ClusteringInitializer(nn.Module):
    """
    Base class for clustering initializers.
    """
    def __init__(self, n_clusters: int, initialize_on_cpu: bool = False):
        super().__init__()
        self.n_clusters = n_clusters
        self.initialize_on_cpu = initialize_on_cpu

    @abstractmethod
    def forward(self, buffer: torch.Tensor) -> torch.Tensor:
        """
        Initialize centroids for clustering algorithms.
        """
        pass

class RandomInitializer(ClusteringInitializer):
    """
    Random initialization for clustering algorithms.
    """
    def forward(self, buffer: torch.Tensor) -> torch.Tensor:
        if self.initialize_on_cpu:
            old_device = buffer.device
            buffer = buffer.to("cpu")

        n_samples = buffer.shape[0]
        indices = torch.randperm(n_samples, device=buffer.device)[: self.n_clusters]
        centroids = buffer[indices].clone().detach()
        
        if self.initialize_on_cpu:
            centroids = centroids.to(old_device)
        return centroids

class KMeansPlusPlusInitInitializer(ClusteringInitializer):
    """
    KMeans++ initialization for clustering algorithms.
    """
    def __init__(
        self,
        n_clusters: int,
        distance_function: DistanceFunction,
        initialize_on_cpu: bool = False,
    ):
        super().__init__(n_clusters=n_clusters, initialize_on_cpu=initialize_on_cpu)
        self.distance_function = distance_function

    def forward(self, buffer: torch.Tensor) -> torch.Tensor:
        if self.initialize_on_cpu:
            old_device = buffer.device
            buffer = buffer.to("cpu")

        n_samples = buffer.shape[0]
        n_features = buffer.shape[1]
        centroids = torch.zeros(
            (self.n_clusters, n_features), dtype=buffer.dtype, device=buffer.device
        )

        # Choose first centroid randomly
        first_centroid_idx = torch.randint(0, n_samples, (1,), device=buffer.device)
        centroids[0] = buffer[first_centroid_idx]

        for i in range(1, self.n_clusters):
            # Compute distances to the nearest existing centroid
            distances = self.distance_function.compute(buffer, centroids[:i])
            # For numerical stability with cosine distance, clamp small negative values to 0
            distances = torch.clamp(distances, min=0.0)
            
            min_distances = torch.min(distances, dim=1)[0]
            if min_distances.sum() == 0:
                centroids[i:] = buffer[
                    torch.randint(
                        0, n_samples, (self.n_clusters - i,), device=buffer.device
                    )
                ]
                break

            # Choose the next centroid with probability proportional to distance
            next_centroid_idx = torch.multinomial(min_distances, num_samples=1)
            centroids[i] = buffer[next_centroid_idx]

        if self.initialize_on_cpu:
            centroids = centroids.to(old_device)
        return centroids
