"""Grouped fixed-grid forward model and runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import (
    action_basis,
    context_vector,
    tolerance_from_current,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_residual_forward_runtime_v9,
)
from specialist_rebuild_v2.common import (
    SETUP_FIELDS,
    STATE_FIELDS,
)


def grouped_context_features(
    contexts: np.ndarray,
) -> np.ndarray:
    """Convert setup plus log-peak current state into derived context features."""

    values = np.asarray(contexts, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 17:
        raise ValueError("grouped forward contexts must have shape [N, 17]")
    output = []
    for context in values:
        setup = {
            field: float(context[index])
            for index, field in enumerate(SETUP_FIELDS)
        }
        current_values = np.asarray(context[12:17], dtype=np.float64).copy()
        current_values[-1] = math.expm1(float(current_values[-1]))
        current = {
            field: float(current_values[index])
            for index, field in enumerate(STATE_FIELDS)
        }
        output.append(context_vector(setup, current))
    return np.asarray(output, dtype=np.float32)


def grouped_forward_surface_model(
    torch: Any,
    input_dim: int,
    coefficient_count: int,
    *,
    width: int,
    blocks: int,
    dropout: float,
) -> Any:
    """Predict five action-basis coefficient vectors from one setup context."""

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

        def forward(self, values: Any) -> Any:
            return torch.nn.functional.silu(values + self.network(values))

    class GroupedForwardSurface(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = torch.nn.Sequential(
                torch.nn.Linear(input_dim, width),
                torch.nn.LayerNorm(width),
                torch.nn.SiLU(),
            )
            self.blocks = torch.nn.Sequential(
                *[ResidualBlock() for _ in range(blocks)]
            )
            self.output = torch.nn.Sequential(
                torch.nn.Linear(width, width),
                torch.nn.SiLU(),
                torch.nn.Linear(width, coefficient_count * 5),
            )
            torch.nn.init.zeros_(self.output[-1].weight)
            torch.nn.init.zeros_(self.output[-1].bias)

        def forward(self, values: Any) -> Any:
            hidden = self.blocks(self.input(values))
            return self.output(hidden).reshape(
                len(values),
                coefficient_count,
                5,
            )

    return GroupedForwardSurface()


class GroupedForwardSurfaceRuntimeV9:
    """Correct a retained 81-action forward surface in one grouped pass."""

    def __init__(
        self,
        torch: Any,
        artifact_path: Path,
        device: Any,
    ) -> None:
        artifact = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=False,
        )
        if artifact.get("model") != "grouped_action_basis_forward_surface_v9":
            raise ValueError("unexpected grouped forward surface artifact")
        self.torch = torch
        self.device = device
        self.artifact = artifact
        self.basis = np.asarray(action_basis(), dtype=np.float32)
        self.pseudoinverse = np.linalg.pinv(self.basis).astype(np.float32)
        config = artifact["config"]
        self.model = grouped_forward_surface_model(
            torch,
            int(config["input_dim"]),
            int(config["coefficient_count"]),
            width=int(config["width"]),
            blocks=int(config["blocks"]),
            dropout=float(config["dropout"]),
        ).to(device)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()
        self.input_mean = np.asarray(
            artifact["input_mean"],
            dtype=np.float32,
        )
        self.input_scale = np.asarray(
            artifact["input_scale"],
            dtype=np.float32,
        )
        self.coefficient_mean = np.asarray(
            artifact["coefficient_mean"],
            dtype=np.float32,
        )
        self.coefficient_scale = np.asarray(
            artifact["coefficient_scale"],
            dtype=np.float32,
        )
        self.field_blend = np.asarray(
            artifact["field_blend"],
            dtype=np.float32,
        )
        self.base, _ = load_residual_forward_runtime_v9(
            Path(str(artifact["base_forward"])).resolve(),
            torch,
            device,
        )

    def predict_changes(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        base = self.base.predict_changes(rows).astype(np.float32)
        contexts = np.asarray(
            [
                [
                    *[float(row["setup"][field]) for field in SETUP_FIELDS],
                    *[
                        (
                            math.log1p(
                                max(
                                    float(row["current_beam_state"][field]),
                                    0.0,
                                )
                            )
                            if field == "peak_intensity"
                            else float(row["current_beam_state"][field])
                        )
                        for field in STATE_FIELDS
                    ],
                ]
                for row in rows
            ],
            dtype=np.float32,
        )
        context = grouped_context_features(contexts)
        base_coefficients = np.einsum(
            "ba,gaf->gbf",
            self.pseudoinverse,
            base,
        ).reshape(len(rows), -1)
        features = np.concatenate(
            [context, base_coefficients],
            axis=1,
        ).astype(np.float32)
        with self.torch.inference_mode():
            standardized = self.model(
                self.torch.as_tensor(
                    (features - self.input_mean) / self.input_scale,
                    dtype=self.torch.float32,
                    device=self.device,
                )
            ).float().cpu().numpy()
        coefficients = (
            standardized * self.coefficient_scale
            + self.coefficient_mean
        )
        correction = np.einsum(
            "ab,gbf->gaf",
            self.basis,
            coefficients,
        ).astype(np.float32)
        prediction = (
            base + correction * self.field_blend[None, None, :]
        )
        zero = np.all(
            np.isclose(self.basis, 0.0, atol=0.0),
            axis=1,
        )
        prediction[:, zero] = 0.0
        return prediction.astype(np.float32)

    def predict_states(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        changes = self.predict_changes(rows)
        current = np.asarray(
            [
                [
                    float(row["current_beam_state"][field])
                    for field in STATE_FIELDS
                ]
                for row in rows
            ],
            dtype=np.float32,
        )
        tolerance = np.stack(
            [
                tolerance_from_current(row["current_beam_state"])
                for row in rows
            ]
        ).astype(np.float32)
        return (
            current[:, None, :]
            + changes * tolerance[:, None, :]
        ).astype(np.float32)


def load_grouped_forward_surface_runtime_v9(
    path: Path,
    torch: Any,
    device: Any,
) -> tuple[GroupedForwardSurfaceRuntimeV9, dict[str, Any]]:
    runtime = GroupedForwardSurfaceRuntimeV9(torch, path, device)
    return runtime, runtime.artifact
