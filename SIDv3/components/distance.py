from abc import ABC, abstractmethod
from typing import Optional
import torch

class DistanceFunction(ABC):
    @abstractmethod
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute distances between the rows of x and the rows of y.
        Args:
            x: Data points of shape (n1, d)
            y: Centroids of shape (n2, d)
        Returns:
            Distances of shape (n1, n2)
        """
        pass

class CosineDistance(DistanceFunction):
    def compute(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine distances (1 - cosine similarity) between the rows of x and the rows of y.
        """
        assert x.dim() == 2, f"Data must be 2D, got {x.dim()} dimensions"
        assert y.dim() == 2, f"Data must be 2D, got {y.dim()} dimensions"
        assert x.size(1) == y.size(1), f"Data must have the same number of columns"

        x_norm = torch.nn.functional.normalize(x, dim=-1)
        y_norm = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = torch.matmul(x_norm, y_norm.transpose(0, 1))
        return 1.0 - sim_matrix

class SquaredEuclideanDistance(DistanceFunction):
    def compute(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute squared Euclidean distances between the rows of x and the rows of y.
        Uses torch.cdist for GPU-accelerated computation.
        """
        assert x.dim() == 2, f"Data must be 2D, got {x.dim()} dimensions"
        assert y.dim() == 2, f"Data must be 2D, got {y.dim()} dimensions"
        assert x.size(1) == y.size(1), f"Data must have the same number of columns"

        return torch.cdist(x, y, p=2).pow(2)
