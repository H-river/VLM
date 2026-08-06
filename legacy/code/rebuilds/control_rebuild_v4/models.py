"""Anchored physics-baseline residual models for control rebuild v4."""

from __future__ import annotations

from typing import Any

from control_rebuild_v3.models import residual_block


def anchored_forward_residual_model(torch: Any, input_dim: int) -> Any:
    class AnchoredForwardResidualV4(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 384),
                torch.nn.LayerNorm(384),
                torch.nn.SiLU(),
                residual_block(torch, 384, 0.04),
                residual_block(torch, 384, 0.04),
                residual_block(torch, 384, 0.04),
                torch.nn.LayerNorm(384),
                torch.nn.SiLU(),
                torch.nn.Linear(384, 192),
                torch.nn.SiLU(),
                torch.nn.Linear(192, 5),
            )

        def forward(self, action_features: Any, zero_features: Any) -> Any:
            shape = action_features.shape
            raw = self.network(action_features.reshape(-1, shape[-1])).reshape(
                *shape[:-1], 5
            )
            if shape[-2] != 81:
                raise ValueError(
                    "anchored forward model expects the fixed 81-action grid"
                )
            if zero_features.shape != action_features[:, 40, :].shape:
                raise ValueError("zero-action feature shape does not align")
            # Use the value from the same network pass.  Subtracting a tensor
            # from itself is exactly zero even while dropout is active.
            zero = raw[:, 40:41, :]
            return raw - zero

    return AnchoredForwardResidualV4()
