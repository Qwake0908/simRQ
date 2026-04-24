from typing import List, Optional, Any
import torch
from torch import nn

class MLP(nn.Module):
    """A simple fully-connected neural net for computing predictions."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim_list: Optional[List[int]] = None,
        activation: Any = nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim_list is None:
            hidden_dim_list = []
        
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dim_list:
            layers.append(nn.Linear(current_dim, h_dim, bias=bias))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, output_dim, bias=bias))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
