"""Spatial analytic-plus-neural measurement model."""

from __future__ import annotations

from typing import Any


def require_torch() -> Any:
    import torch

    return torch


def measurement_model_v3(
    torch: Any, calibration_dim: int = 11, analytic_dim: int = 9
) -> Any:
    class SpatialMeasurementV3(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spatial = torch.nn.Sequential(
                torch.nn.Conv2d(5, 16, 5, stride=2, padding=2),
                torch.nn.GroupNorm(4, 16),
                torch.nn.SiLU(),
                torch.nn.Conv2d(16, 24, 3, stride=2, padding=1),
                torch.nn.GroupNorm(4, 24),
                torch.nn.SiLU(),
                torch.nn.Conv2d(24, 32, 3, stride=2, padding=1),
                torch.nn.GroupNorm(4, 32),
                torch.nn.SiLU(),
                torch.nn.Conv2d(32, 48, 3, stride=2, padding=1),
                torch.nn.GroupNorm(8, 48),
                torch.nn.SiLU(),
                torch.nn.Conv2d(48, 64, 3, stride=2, padding=1),
                torch.nn.GroupNorm(8, 64),
                torch.nn.SiLU(),
                torch.nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.calibration = torch.nn.Sequential(
                torch.nn.Linear(calibration_dim, 64),
                torch.nn.LayerNorm(64),
                torch.nn.SiLU(),
                torch.nn.Linear(64, 64),
                torch.nn.SiLU(),
            )
            self.analytic = torch.nn.Sequential(
                torch.nn.Linear(analytic_dim, 64),
                torch.nn.LayerNorm(64),
                torch.nn.SiLU(),
                torch.nn.Linear(64, 64),
                torch.nn.SiLU(),
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(64 * 4 * 4 + 128, 256),
                torch.nn.LayerNorm(256),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.05),
                torch.nn.Linear(256, 128),
                torch.nn.SiLU(),
                torch.nn.Linear(128, 5),
            )

        def forward(
            self,
            observed: Any,
            linearized: Any,
            valid_mask: Any,
            calibration: Any,
            analytic: Any,
        ) -> Any:
            height, width = observed.shape[-2:]
            y = torch.linspace(
                -1.0, 1.0, height, device=observed.device, dtype=observed.dtype
            )
            x = torch.linspace(
                -1.0, 1.0, width, device=observed.device, dtype=observed.dtype
            )
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            xx = xx[None, None].expand(len(observed), -1, -1, -1)
            yy = yy[None, None].expand(len(observed), -1, -1, -1)
            spatial = self.spatial(
                torch.cat([observed, linearized, valid_mask, xx, yy], dim=1)
            ).flatten(1)
            context = torch.cat(
                [
                    spatial,
                    self.calibration(calibration),
                    self.analytic(analytic),
                ],
                dim=-1,
            )
            return self.head(context)

    return SpatialMeasurementV3()
