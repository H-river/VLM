"""Prediction heads for profile2setup v2 Stage 4 model."""

from __future__ import annotations

import torch
from torch import nn


class MultiVariableHeads(nn.Module):
    """Shared trunk with three per-variable linear output heads."""

    def __init__(
        self,
        fused_dim: int = 256,
        num_variables: int = 7,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if fused_dim <= 0:
            raise ValueError(f"fused_dim must be positive, got {fused_dim}")
        if num_variables <= 0:
            raise ValueError(f"num_variables must be positive, got {num_variables}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be >= 0, got {dropout}")

        self.fused_dim = int(fused_dim)

        shared_layers: list[nn.Module] = [
            nn.Linear(self.fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0.0:
            shared_layers.append(nn.Dropout(dropout))
        self.shared = nn.Sequential(*shared_layers)

        self.delta_head = nn.Linear(hidden_dim, num_variables)
        self.absolute_head = nn.Linear(hidden_dim, num_variables)
        self.change_head = nn.Linear(hidden_dim, num_variables)

    def forward(self, fused_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        if fused_embedding.ndim != 2:
            raise ValueError(
                "MultiVariableHeads expects input shape [B, D]; "
                f"got ndim={fused_embedding.ndim} with shape={tuple(fused_embedding.shape)}"
            )
        if fused_embedding.shape[-1] != self.fused_dim:
            raise ValueError(
                f"MultiVariableHeads expects last dimension {self.fused_dim}; "
                f"got {fused_embedding.shape[-1]} with shape={tuple(fused_embedding.shape)}"
            )

        hidden = self.shared(fused_embedding)
        return {
            "delta": self.delta_head(hidden),
            "absolute": self.absolute_head(hidden),
            "change_logits": self.change_head(hidden),
        }
