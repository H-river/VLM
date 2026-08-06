"""Matched-capacity residual models for v11 action-representation ablations."""

from __future__ import annotations

from typing import Any


def build_model(
    torch: Any,
    *,
    representation: str,
    structured_input_dim: int,
    context_input_dim: int,
    config: dict[str, Any],
) -> Any:
    hidden = int(config["hidden"])
    bottleneck = int(config["bottleneck"])
    dropout = float(config["dropout"])
    embedding_dim = int(config["opaque_action_embedding"])

    class ResidualBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.Dropout(dropout),
                torch.nn.LayerNorm(hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, hidden),
            )

        def forward(self, values: Any) -> Any:
            return values + self.block(values)

    class V11ResidualModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.representation = representation
            if representation == "opaque":
                self.action_embedding = torch.nn.Embedding(81, embedding_dim)
                input_dim = context_input_dim + embedding_dim + 5
            elif representation == "structured":
                self.action_embedding = None
                input_dim = structured_input_dim + 5
            else:
                raise ValueError(f"unknown action representation: {representation}")
            layers: list[Any] = [
                torch.nn.Linear(input_dim, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.SiLU(),
            ]
            layers.extend(ResidualBlock() for _ in range(int(config["blocks"])))
            layers.extend(
                [
                    torch.nn.LayerNorm(hidden),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden, bottleneck),
                    torch.nn.SiLU(),
                    torch.nn.Linear(bottleneck, 5),
                ]
            )
            self.network = torch.nn.Sequential(*layers)

        def forward(
            self,
            structured: Any,
            context: Any,
            action_index: Any,
            prior: Any,
        ) -> Any:
            if self.representation == "opaque":
                values = torch.cat(
                    [context, self.action_embedding(action_index), prior],
                    dim=1,
                )
            else:
                values = torch.cat([structured, prior], dim=1)
            residual = self.network(values)
            nonzero = (action_index != 40).to(prior.dtype)[:, None]
            return (prior + residual) * nonzero

    return V11ResidualModel()

