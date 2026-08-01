"""Grouped physics-structured forward and auxiliary direction model."""

from __future__ import annotations

from typing import Any


def build_structured_forward_model(
    torch: Any,
    config: dict[str, Any],
) -> Any:
    """Build a model respecting setup, state, actuator, and derived groups."""

    class StructuredForwardModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimension = int(config["dimension"])
            hidden = int(config["head_hidden"])

            def encoder(input_dim: int) -> Any:
                return torch.nn.Sequential(
                    torch.nn.Linear(input_dim, dimension),
                    torch.nn.LayerNorm(dimension),
                    torch.nn.GELU(),
                    torch.nn.Linear(dimension, dimension),
                )

            self.setup_encoder = encoder(12)
            self.state_encoder = encoder(5)
            self.derived_encoder = encoder(25)
            self.action_weight = torch.nn.Parameter(
                torch.empty(4, dimension)
            )
            self.action_bias = torch.nn.Parameter(
                torch.empty(4, dimension)
            )
            self.action_identity = torch.nn.Parameter(
                torch.empty(4, dimension)
            )
            self.cls = torch.nn.Parameter(torch.empty(1, 1, dimension))
            self.group_identity = torch.nn.Parameter(
                torch.empty(1, 3, dimension)
            )
            layer = torch.nn.TransformerEncoderLayer(
                d_model=dimension,
                nhead=int(config["heads"]),
                dim_feedforward=int(config["feedforward"]),
                dropout=float(config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.interactions = torch.nn.TransformerEncoder(
                layer,
                num_layers=int(config["layers"]),
                norm=torch.nn.LayerNorm(dimension),
            )
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(dimension, dimension),
                torch.nn.GELU(),
                torch.nn.Dropout(float(config["dropout"])),
            )

            def scalar_head() -> Any:
                return torch.nn.Sequential(
                    torch.nn.Linear(dimension, hidden),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden, 1),
                )

            self.change_heads = torch.nn.ModuleList(
                [scalar_head() for _ in range(5)]
            )
            self.next_heads = torch.nn.ModuleList(
                [scalar_head() for _ in range(5)]
            )
            self.direction_heads = torch.nn.ModuleList(
                [torch.nn.Linear(dimension, 3) for _ in range(5)]
            )
            for parameter in (
                self.action_weight,
                self.action_bias,
                self.action_identity,
                self.cls,
                self.group_identity,
            ):
                torch.nn.init.normal_(parameter, std=0.02)

        def encoded(self, values: Any) -> tuple[Any, Any]:
            setup = self.setup_encoder(values[:, 0:12])
            state = self.state_encoder(values[:, 12:17])
            action_values = values[:, 17:21]
            derived = self.derived_encoder(values[:, 21:46])
            groups = torch.stack([setup, state, derived], dim=1)
            groups = groups + self.group_identity
            actions = (
                action_values[:, :, None] * self.action_weight[None, :, :]
                + self.action_bias[None, :, :]
                + self.action_identity[None, :, :]
            )
            cls = self.cls.expand(len(values), -1, -1)
            encoded = self.interactions(
                torch.cat([cls, groups, actions], dim=1)
            )
            representation = self.trunk(encoded[:, 0])
            nonzero = (
                action_values.abs().sum(dim=1, keepdim=True) > 1e-7
            ).to(representation.dtype)
            return representation, nonzero

        def forward_with_aux(self, values: Any) -> tuple[Any, Any, Any]:
            representation, nonzero = self.encoded(values)
            change = torch.cat(
                [head(representation) for head in self.change_heads],
                dim=1,
            )
            change = change * nonzero
            next_state = torch.cat(
                [head(representation) for head in self.next_heads],
                dim=1,
            )
            direction = torch.stack(
                [head(representation) for head in self.direction_heads],
                dim=1,
            )
            return change, next_state, direction

        def forward(self, values: Any) -> tuple[Any, Any]:
            change, _, direction = self.forward_with_aux(values)
            return change, direction

    return StructuredForwardModel()
