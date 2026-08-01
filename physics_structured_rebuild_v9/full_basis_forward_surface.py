"""Complete ternary-action basis forward surface and runtime."""

from __future__ import annotations

import itertools
import math
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_NORMALIZED, action_basis, tolerance_from_current
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from physics_structured_rebuild_v9.grouped_forward_surface import (
    grouped_context_features,
    grouped_forward_surface_model,
)
from specialist_rebuild_v2.common import (
    SETUP_FIELDS,
    STATE_FIELDS,
    fixed_action_grid,
    forward_feature,
)


def full_ternary_action_basis(
    actions: np.ndarray = ACTION_NORMALIZED,
) -> np.ndarray:
    """Return all nonconstant monomials with per-actuator degree 0, 1, or 2."""

    values = np.asarray(actions, dtype=np.float32)
    exponents = [
        powers
        for powers in itertools.product(range(3), repeat=4)
        if any(power != 0 for power in powers)
    ]
    columns = [
        np.prod(
            np.stack(
                [values[:, index] ** power for index, power in enumerate(powers)],
                axis=1,
            ),
            axis=1,
        )
        for powers in exponents
    ]
    basis = np.stack(columns, axis=1).astype(np.float32)
    if basis.shape != (len(values), 80):
        raise AssertionError("complete ternary action basis must have 80 columns")
    if np.linalg.matrix_rank(basis) != 80:
        raise AssertionError("complete ternary action basis must have rank 80")
    if not np.all(basis[40] == 0.0):
        raise AssertionError("zero-action basis row must be exactly zero")
    return basis


class FullBasisForwardSurfaceRuntimeV9:
    """Predict a complete 81-action correction through 80 basis coefficients."""

    def __init__(
        self,
        artifact_path: Path,
        torch: Any,
        device: Any,
    ) -> None:
        artifact = torch.load(
            artifact_path.resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if artifact.get("model") != "complete_ternary_basis_forward_surface_v9":
            raise ValueError("unexpected complete-basis forward artifact")
        self.artifact = artifact
        self.torch = torch
        self.device = device
        self.full_basis = full_ternary_action_basis()
        self.input_basis = np.asarray(action_basis(), dtype=np.float32)
        self.input_pseudoinverse = np.linalg.pinv(self.input_basis).astype(
            np.float32
        )
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
            artifact["input_mean"], dtype=np.float32
        )
        self.input_scale = np.asarray(
            artifact["input_scale"], dtype=np.float32
        )
        self.coefficient_scale = np.asarray(
            artifact["coefficient_scale"], dtype=np.float32
        )
        self.field_blend = np.asarray(
            artifact["field_blend"], dtype=np.float32
        )
        self.base, _ = load_forward_direction_runtime_v7(
            Path(str(artifact["base_forward"])).resolve(),
            torch,
            device,
        )

    @staticmethod
    def _contexts(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray(
            [
                [
                    *[float(row["setup"][field]) for field in SETUP_FIELDS],
                    *[
                        (
                            math.log1p(
                                max(float(row["current_beam_state"][field]), 0.0)
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

    def predict_correction(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        prior = self.base.predict_changes(rows).astype(np.float32)
        input_coefficients = np.einsum(
            "ba,gaf->gbf", self.input_pseudoinverse, prior
        )
        features = np.concatenate(
            [
                grouped_context_features(self._contexts(rows)),
                input_coefficients.reshape(len(rows), -1),
            ],
            axis=1,
        ).astype(np.float32)
        standardized = (
            features - self.input_mean[None, :]
        ) / self.input_scale[None, :]
        outputs = []
        batch_size = int(self.artifact["config"].get("runtime_batch_size", 128))
        with self.torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                values = self.model(
                    self.torch.as_tensor(
                        standardized[start : start + batch_size],
                        dtype=self.torch.float32,
                        device=self.device,
                    )
                )
                outputs.append(values.float().cpu().numpy())
        coefficients = np.concatenate(outputs, axis=0)
        coefficients *= self.coefficient_scale[None, :, :]
        correction = np.einsum(
            "ab,gbf->gaf", self.full_basis, coefficients
        ).astype(np.float32)
        correction[:, 40, :] = 0.0
        return prior, correction

    def predict_changes(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        prior, correction = self.predict_correction(rows)
        return (
            prior + correction * self.field_blend[None, None, :]
        ).astype(np.float32)

    def predict_states(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        changes = self.predict_changes(rows)
        current = np.asarray(
            [
                [float(row["current_beam_state"][field]) for field in STATE_FIELDS]
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
        return current[:, None, :] + changes * tolerance[:, None, :]


def load_full_basis_forward_surface_runtime_v9(
    path: Path,
    torch: Any,
    device: Any,
) -> tuple[FullBasisForwardSurfaceRuntimeV9, dict[str, Any]]:
    runtime = FullBasisForwardSurfaceRuntimeV9(path, torch, device)
    return runtime, runtime.artifact


class FullBasisForwardSelectorRuntimeV9:
    """Select by action between the accepted ensemble and full-basis model."""

    def __init__(
        self,
        artifact_path: Path,
        torch: Any,
        device: Any,
    ) -> None:
        with artifact_path.resolve().open("rb") as stream:
            artifact = pickle.load(stream)
        if artifact.get("model") != "full_basis_forward_selector_v9":
            raise ValueError("unexpected full-basis selector artifact")
        self.current, _ = load_forward_selector_ensemble_runtime_v9(
            Path(str(artifact["current_selector"])).resolve(),
            torch,
            device,
        )
        self.candidate, _ = load_full_basis_forward_surface_runtime_v9(
            Path(str(artifact["candidate_artifact"])).resolve(),
            torch,
            device,
        )
        self.classifier = artifact["classifier"]
        self.threshold = float(artifact["threshold"])
        self.artifact = artifact
        self.action_grid = fixed_action_grid()

    def predict_changes(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        current = self.current.predict_changes(rows)
        candidate = self.candidate.predict_changes(rows)
        prior = self.current.base.predict_changes(rows)
        engineered = np.asarray(
            [
                [
                    forward_feature(
                        row["setup"], row["current_beam_state"], action
                    )
                    for action in self.action_grid
                ]
                for row in rows
            ],
            dtype=np.float32,
        )
        physical = np.concatenate([engineered, prior], axis=2)
        features = np.concatenate(
            [
                physical.reshape(-1, physical.shape[2]),
                prior.reshape(-1, 5),
                current.reshape(-1, 5),
                candidate.reshape(-1, 5),
                np.abs(candidate - current).reshape(-1, 5),
            ],
            axis=1,
        ).astype(np.float32)
        probability = self.classifier.predict_proba(features)[:, 1]
        choose = probability.reshape(
            len(rows), len(self.action_grid)
        ) >= self.threshold
        return np.where(
            choose[:, :, None], candidate, current
        ).astype(np.float32)

    def predict_states(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        changes = self.predict_changes(rows)
        current = np.asarray(
            [
                [float(row["current_beam_state"][field]) for field in STATE_FIELDS]
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
        return current[:, None, :] + changes * tolerance[:, None, :]


def load_full_basis_forward_selector_runtime_v9(
    path: Path,
    torch: Any,
    device: Any,
) -> tuple[FullBasisForwardSelectorRuntimeV9, dict[str, Any]]:
    runtime = FullBasisForwardSelectorRuntimeV9(path, torch, device)
    return runtime, runtime.artifact
