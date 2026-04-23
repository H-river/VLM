"""Current setup encoder for profile2setup v2 Stage 4 model."""

from __future__ import annotations

import torch
from torch import nn


class SetupEncoder(nn.Module):
    """Small MLP encoder for normalized current setup vectors."""

    def __init__(
        self,
        input_dim: int = 7,
        setup_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if setup_dim <= 0:
            raise ValueError(f"setup_dim must be positive, got {setup_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be >= 0, got {dropout}")

        self.input_dim = int(input_dim)

        layers: list[nn.Module] = [
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden_dim, setup_dim),
                nn.LayerNorm(setup_dim),
                nn.GELU(),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, current_setup: torch.Tensor) -> torch.Tensor:
        if current_setup.ndim != 2:
            raise ValueError(
                "SetupEncoder expects input shape [B, D]; "
                f"got ndim={current_setup.ndim} with shape={tuple(current_setup.shape)}"
            )
        if current_setup.shape[-1] != self.input_dim:
            raise ValueError(
                f"SetupEncoder expects last dimension {self.input_dim}; "
                f"got {current_setup.shape[-1]} with shape={tuple(current_setup.shape)}"
            )

        return self.net(current_setup)
