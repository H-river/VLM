"""Neural model used by the v4 direction specialist."""

from __future__ import annotations

from typing import Any


def direction_classifier_v4(torch: Any, input_dim: int) -> Any:
    """Return a shared encoder with five independent three-class heads."""

    class ResidualBlock(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(width, width),
                torch.nn.LayerNorm(width),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.05),
                torch.nn.Linear(width, width),
                torch.nn.LayerNorm(width),
            )
            self.activation = torch.nn.SiLU()

        def forward(self, values: Any) -> Any:
            return self.activation(values + self.network(values))

    class DirectionClassifierV4(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 256),
                torch.nn.LayerNorm(256),
                torch.nn.SiLU(),
                ResidualBlock(256),
                ResidualBlock(256),
                torch.nn.Linear(256, 128),
                torch.nn.LayerNorm(128),
                torch.nn.SiLU(),
            )
            self.heads = torch.nn.ModuleList(
                [torch.nn.Linear(128, 3) for _ in range(5)]
            )

        def forward(self, values: Any) -> Any:
            hidden = self.encoder(values)
            return torch.stack([head(hidden) for head in self.heads], dim=1)

    return DirectionClassifierV4()
