import torch
from torch import nn


class OPQRotation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        Q, _ = torch.linalg.qr(torch.randn(dim, dim))
        self.weight = nn.Parameter(Q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._orthogonalize(self.weight)
        return x @ W.t()

    def _orthogonalize(self, W: torch.Tensor) -> torch.Tensor:
        U, _, Vh = torch.linalg.svd(W)
        return U @ Vh
