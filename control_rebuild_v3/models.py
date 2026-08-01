"""Joint-grid forward model and residual-aware inverse ranker."""

from __future__ import annotations

from typing import Any


def require_torch() -> Any:
    import torch

    return torch


def residual_block(torch: Any, width: int, dropout: float) -> Any:
    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.LayerNorm(width),
                torch.nn.Linear(width, width),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(width, width),
            )

        def forward(self, values: Any) -> Any:
            return values + self.layers(values)

    return Block()


def joint_forward_model(
    torch: Any, context_dim: int, basis_dim: int
) -> Any:
    class JointForwardV3(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context = torch.nn.Sequential(
                torch.nn.Linear(context_dim, 320),
                torch.nn.LayerNorm(320),
                torch.nn.SiLU(),
                residual_block(torch, 320, 0.04),
                residual_block(torch, 320, 0.04),
                residual_block(torch, 320, 0.04),
                torch.nn.LayerNorm(320),
                torch.nn.SiLU(),
            )
            self.coefficients = torch.nn.Sequential(
                torch.nn.Linear(320, 256),
                torch.nn.SiLU(),
                torch.nn.Linear(256, basis_dim * 5),
            )
            self.global_coefficients = torch.nn.Parameter(
                torch.zeros(basis_dim, 5)
            )
            self.basis_dim = int(basis_dim)

        def forward(self, context: Any, basis: Any) -> Any:
            coefficients = self.coefficients(self.context(context)).reshape(
                -1, self.basis_dim, 5
            )
            coefficients = coefficients + self.global_coefficients[None, :, :]
            return torch.einsum("af,bfk->bak", basis, coefficients)

    return JointForwardV3()


def inverse_ranker_model(
    torch: Any, context_dim: int, candidate_dim: int, status_dim: int
) -> Any:
    class InverseRankerV3(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context = torch.nn.Sequential(
                torch.nn.Linear(context_dim, 256),
                torch.nn.LayerNorm(256),
                torch.nn.SiLU(),
                residual_block(torch, 256, 0.04),
                residual_block(torch, 256, 0.04),
                torch.nn.Linear(256, 160),
                torch.nn.SiLU(),
            )
            self.candidate = torch.nn.Sequential(
                torch.nn.Linear(candidate_dim, 128),
                torch.nn.LayerNorm(128),
                torch.nn.SiLU(),
                torch.nn.Linear(128, 96),
                torch.nn.SiLU(),
            )
            self.correction = torch.nn.Sequential(
                torch.nn.Linear(256, 128),
                torch.nn.SiLU(),
                torch.nn.Linear(128, 1),
            )
            self.status = torch.nn.Sequential(
                torch.nn.Linear(160 + status_dim, 128),
                torch.nn.SiLU(),
                torch.nn.Linear(128, 3),
            )

        def forward(
            self, context: Any, candidates: Any, status_features: Any
        ) -> tuple[Any, Any]:
            encoded_context = self.context(context)
            encoded_candidates = self.candidate(candidates)
            expanded = encoded_context[:, None, :].expand(
                -1, encoded_candidates.shape[1], -1
            )
            correction = self.correction(
                torch.cat([expanded, encoded_candidates], dim=-1)
            ).squeeze(-1)
            status = self.status(
                torch.cat([encoded_context, status_features], dim=-1)
            )
            return correction, status

    return InverseRankerV3()

