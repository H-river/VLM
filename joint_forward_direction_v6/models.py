"""Neural architecture for shared forward and direction prediction."""

from __future__ import annotations

from typing import Any


def shared_forward_direction_model(
    torch: Any,
    input_dim: int,
    *,
    width: int = 320,
    hidden_dim: int = 192,
    residual_blocks: int = 3,
    dropout: float = 0.03,
) -> Any:
    """Build one encoder with five regression and five classification heads."""

    class ResidualBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(width, width),
                torch.nn.LayerNorm(width),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(width, width),
                torch.nn.LayerNorm(width),
            )
            self.activation = torch.nn.SiLU()

        def forward(self, values: Any) -> Any:
            return self.activation(values + self.network(values))

    class SharedForwardDirectionV6(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = [
                torch.nn.Linear(input_dim, width),
                torch.nn.LayerNorm(width),
                torch.nn.SiLU(),
            ]
            layers.extend(ResidualBlock() for _ in range(residual_blocks))
            layers.extend(
                [
                    torch.nn.Linear(width, hidden_dim),
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.SiLU(),
                ]
            )
            self.encoder = torch.nn.Sequential(*layers)
            self.regression_heads = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, 64),
                        torch.nn.SiLU(),
                        torch.nn.Linear(64, 1),
                    )
                    for _ in range(5)
                ]
            )
            direction_input_dim = hidden_dim + 3
            self.direction_heads = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(direction_input_dim, 64),
                        torch.nn.SiLU(),
                        torch.nn.Linear(64, 3),
                    )
                    for _ in range(5)
                ]
            )
            for head in self.regression_heads:
                torch.nn.init.zeros_(head[-1].weight)
                torch.nn.init.zeros_(head[-1].bias)
            for head in self.direction_heads:
                torch.nn.init.zeros_(head[-1].weight)
                torch.nn.init.zeros_(head[-1].bias)

        def forward(self, values: Any, prior_changes: Any) -> tuple[Any, Any]:
            hidden = self.encoder(values)
            residual = torch.cat(
                [head(hidden) for head in self.regression_heads],
                dim=1,
            )
            changes = prior_changes + residual
            logits = []
            for field, head in enumerate(self.direction_heads):
                change = changes[:, field : field + 1]
                prior = prior_changes[:, field : field + 1]
                boundary_distance = change.abs() - 1.0
                direction_input = torch.cat(
                    [hidden, change, prior, boundary_distance],
                    dim=1,
                )
                threshold_index = torch.where(
                    change < -1.0,
                    torch.zeros_like(change, dtype=torch.long),
                    torch.where(
                        change > 1.0,
                        torch.full_like(change, 2, dtype=torch.long),
                        torch.ones_like(change, dtype=torch.long),
                    ),
                ).squeeze(1)
                threshold_logits = 4.0 * torch.nn.functional.one_hot(
                    threshold_index,
                    num_classes=3,
                ).to(change.dtype)
                logits.append(threshold_logits + head(direction_input))
            return changes, torch.stack(logits, dim=1)

    return SharedForwardDirectionV6()
