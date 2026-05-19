"""Intent feature encoder for optional profile2setup reasoning features."""

from __future__ import annotations

import torch
from torch import nn

from legacy.reasoning_vlm.intent_features import INTENT_FEATURE_DIM


class IntentEncoder(nn.Module):
    """Encode fixed-length VLM intent features into a fusion embedding."""

    def __init__(
        self,
        input_dim: int = INTENT_FEATURE_DIM,
        intent_dim: int = 64,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.intent_dim = int(intent_dim)
        hidden = int(hidden_dim or max(self.intent_dim, self.input_dim))

        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.intent_dim <= 0:
            raise ValueError("intent_dim must be positive")

        layers: list[nn.Module] = [
            nn.Linear(self.input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden, self.intent_dim),
                nn.GELU(),
            ]
        )
        self.mlp = nn.Sequential(*layers)

    def forward(self, intent_features: torch.Tensor) -> torch.Tensor:
        if intent_features.ndim != 2:
            raise ValueError(
                "intent_features must have shape [B, input_dim]; "
                f"got {tuple(intent_features.shape)}"
            )
        if intent_features.shape[-1] != self.input_dim:
            raise ValueError(
                "intent_features width must match input_dim; "
                f"got {intent_features.shape[-1]} and {self.input_dim}"
            )
        return self.mlp(intent_features)
