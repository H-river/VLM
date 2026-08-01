"""Compact version-2 specialist architectures."""

from __future__ import annotations

from typing import Any


def require_torch() -> Any:
    import torch

    return torch


class ResidualBlock:
    """Factory namespace because torch is loaded lazily by training scripts."""

    @staticmethod
    def build(torch: Any, width: int, dropout: float) -> Any:
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


def direction_model(torch: Any, input_dim: int = 21) -> Any:
    class DirectionV2(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = torch.nn.Linear(input_dim, 256)
            self.blocks = torch.nn.Sequential(
                ResidualBlock.build(torch, 256, 0.08),
                ResidualBlock.build(torch, 256, 0.08),
            )
            self.output = torch.nn.Sequential(
                torch.nn.LayerNorm(256),
                torch.nn.SiLU(),
                torch.nn.Linear(256, 128),
            )
            self.heads = torch.nn.ModuleList(
                [torch.nn.Linear(128, 3) for _ in range(5)]
            )

        def forward(self, values: Any) -> Any:
            hidden = self.output(self.blocks(self.input(values)))
            return torch.stack([head(hidden) for head in self.heads], dim=1)

    return DirectionV2()


def forward_model(torch: Any, input_dim: int = 46) -> Any:
    class ForwardV2(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = torch.nn.Linear(input_dim, 256)
            self.blocks = torch.nn.Sequential(
                ResidualBlock.build(torch, 256, 0.06),
                ResidualBlock.build(torch, 256, 0.06),
                ResidualBlock.build(torch, 256, 0.06),
            )
            self.hidden = torch.nn.Sequential(
                torch.nn.LayerNorm(256),
                torch.nn.SiLU(),
                torch.nn.Linear(256, 128),
                torch.nn.SiLU(),
            )
            self.change = torch.nn.Linear(128, 5)
            self.log_variance = torch.nn.Linear(128, 5)
            self.direction_heads = torch.nn.ModuleList(
                [torch.nn.Linear(128, 3) for _ in range(5)]
            )

        def forward(self, values: Any) -> tuple[Any, Any, Any]:
            hidden = self.hidden(self.blocks(self.input(values)))
            direction = torch.stack(
                [head(hidden) for head in self.direction_heads], dim=1
            )
            return self.change(hidden), self.log_variance(hidden), direction

    return ForwardV2()


def inverse_ranker(torch: Any, context_dim: int = 31) -> Any:
    class InverseRankerV2(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context = torch.nn.Sequential(
                torch.nn.Linear(context_dim, 256),
                torch.nn.LayerNorm(256),
                torch.nn.SiLU(),
                ResidualBlock.build(torch, 256, 0.05),
                torch.nn.Linear(256, 128),
                torch.nn.SiLU(),
            )
            self.action = torch.nn.Sequential(
                torch.nn.Linear(5, 64),
                torch.nn.SiLU(),
                torch.nn.Linear(64, 64),
                torch.nn.SiLU(),
            )
            self.score = torch.nn.Sequential(
                torch.nn.Linear(192, 128),
                torch.nn.SiLU(),
                torch.nn.Linear(128, 1),
            )
            self.status = torch.nn.Linear(128, 3)

        def forward(
            self, context: Any, actions_and_residual: Any
        ) -> tuple[Any, Any]:
            encoded_context = self.context(context)
            encoded_action = self.action(actions_and_residual)
            expanded = encoded_context[:, None, :].expand(
                -1, encoded_action.shape[1], -1
            )
            scores = self.score(
                torch.cat([expanded, encoded_action], dim=-1)
            ).squeeze(-1)
            return scores, self.status(encoded_context)

    return InverseRankerV2()


def beam_encoder(torch: Any) -> Any:
    class BeamEncoderV2(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, 5, stride=2, padding=2),
                torch.nn.SiLU(),
                torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
                torch.nn.GroupNorm(4, 32),
                torch.nn.SiLU(),
                torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
                torch.nn.GroupNorm(8, 64),
                torch.nn.SiLU(),
                torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
                torch.nn.GroupNorm(8, 128),
                torch.nn.SiLU(),
                torch.nn.AdaptiveAvgPool2d(1),
            )

        def forward(self, image: Any) -> Any:
            return self.features(image).flatten(1)

    return BeamEncoderV2()


def measurement_model(torch: Any) -> Any:
    class MeasurementV2(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = beam_encoder(torch)
            self.calibration = torch.nn.Sequential(
                torch.nn.Linear(4, 32),
                torch.nn.SiLU(),
                torch.nn.Linear(32, 32),
                torch.nn.SiLU(),
            )
            self.output = torch.nn.Sequential(
                torch.nn.Linear(160, 128),
                torch.nn.SiLU(),
                torch.nn.Linear(128, 10),
            )

        def forward(self, image: Any, calibration: Any) -> tuple[Any, Any]:
            values = self.output(
                torch.cat(
                    [self.encoder(image), self.calibration(calibration)], dim=-1
                )
            )
            return values[:, :5], values[:, 5:]

    return MeasurementV2()


def visual_inverse_ranker(torch: Any, setup_dim: int = 12) -> Any:
    class VisualInverseV2(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = beam_encoder(torch)
            self.context = torch.nn.Sequential(
                torch.nn.Linear(128 * 4 + setup_dim + 4, 256),
                torch.nn.LayerNorm(256),
                torch.nn.SiLU(),
                ResidualBlock.build(torch, 256, 0.08),
                torch.nn.Linear(256, 128),
                torch.nn.SiLU(),
            )
            self.action = torch.nn.Sequential(
                torch.nn.Linear(4, 64),
                torch.nn.SiLU(),
                torch.nn.Linear(64, 64),
                torch.nn.SiLU(),
            )
            self.score = torch.nn.Sequential(
                torch.nn.Linear(192, 128),
                torch.nn.SiLU(),
                torch.nn.Linear(128, 1),
            )
            self.status = torch.nn.Linear(128, 3)

        def forward(
            self,
            current_image: Any,
            desired_image: Any,
            setup_and_calibration: Any,
            actions: Any,
        ) -> tuple[Any, Any]:
            current = self.encoder(current_image)
            desired = self.encoder(desired_image)
            pair = torch.cat(
                [
                    current,
                    desired,
                    desired - current,
                    desired * current,
                    setup_and_calibration,
                ],
                dim=-1,
            )
            context = self.context(pair)
            action = self.action(actions)
            expanded = context[:, None, :].expand(-1, action.shape[1], -1)
            scores = self.score(torch.cat([expanded, action], dim=-1)).squeeze(-1)
            return scores, self.status(context)

    return VisualInverseV2()

