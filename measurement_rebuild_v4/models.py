"""Small bounded residual calibrator for frozen v3 measurements."""

from __future__ import annotations

from typing import Any


def measurement_calibrator_v4(torch: Any, input_dim: int) -> Any:
    class MeasurementCalibratorV4(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 160),
                torch.nn.LayerNorm(160),
                torch.nn.SiLU(),
                torch.nn.Linear(160, 160),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.04),
                torch.nn.Linear(160, 96),
                torch.nn.SiLU(),
                torch.nn.Linear(96, 5),
            )
            torch.nn.init.zeros_(self.network[-1].weight)
            torch.nn.init.zeros_(self.network[-1].bias)

        def forward(self, features: Any) -> Any:
            return self.network(features)

    return MeasurementCalibratorV4()
