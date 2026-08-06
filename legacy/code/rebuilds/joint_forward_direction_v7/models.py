"""Cross-fitted conservative shared forward-direction architecture."""

from __future__ import annotations

from typing import Any


def shared_forward_direction_model_v7(
    torch: Any,
    input_dim: int,
    *,
    width: int = 320,
    hidden_dim: int = 192,
    residual_blocks: int = 3,
    dropout: float = 0.03,
) -> Any:
    """Build one encoder with numerical residual and direction-correction heads."""

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

    class SharedForwardDirectionV7(torch.nn.Module):
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
            self.correction_heads = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim + 3, 64),
                        torch.nn.SiLU(),
                        torch.nn.Linear(64, 3),
                    )
                    for _ in range(5)
                ]
            )
            for head in self.regression_heads:
                torch.nn.init.zeros_(head[-1].weight)
                torch.nn.init.zeros_(head[-1].bias)
            for head in self.correction_heads:
                torch.nn.init.zeros_(head[-1].weight)
                torch.nn.init.zeros_(head[-1].bias)

        def forward(self, values: Any, prior_changes: Any) -> tuple[Any, Any]:
            hidden = self.encoder(values)
            residual = torch.cat(
                [head(hidden) for head in self.regression_heads],
                dim=1,
            )
            full_changes = prior_changes + residual
            correction_logits = []
            for field, head in enumerate(self.correction_heads):
                change = full_changes[:, field : field + 1]
                prior = prior_changes[:, field : field + 1]
                direction_input = torch.cat(
                    [hidden, change, prior, change.abs() - 1.0],
                    dim=1,
                )
                correction_logits.append(head(direction_input))
            return full_changes, torch.stack(correction_logits, dim=1)

    return SharedForwardDirectionV7()

