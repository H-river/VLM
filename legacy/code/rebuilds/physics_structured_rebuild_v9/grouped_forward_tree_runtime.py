"""Runtime for regularized grouped action-basis tree corrections."""

from __future__ import annotations

import math
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import action_basis, tolerance_from_current
from physics_structured_rebuild_v9.grouped_forward_surface import (
    grouped_context_features,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_residual_forward_runtime_v9,
)
from specialist_rebuild_v2.common import SETUP_FIELDS, STATE_FIELDS


class GroupedForwardTreeRuntimeV9:
    """Predict one correction surface per setup with five tree heads."""

    def __init__(self, artifact_path: Path, torch: Any, device: Any) -> None:
        with artifact_path.resolve().open("rb") as stream:
            artifact = pickle.load(stream)
        if artifact.get("model") != "grouped_action_basis_extra_trees_v9":
            raise ValueError("unexpected grouped forward tree artifact")
        self.artifact = artifact
        self.models = list(artifact["models"])
        self.field_blend = np.asarray(
            artifact["field_blend"],
            dtype=np.float32,
        )
        self.basis = np.asarray(action_basis(), dtype=np.float32)
        self.pseudoinverse = np.linalg.pinv(self.basis).astype(np.float32)
        self.base, _ = load_residual_forward_runtime_v9(
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

    def predict_changes(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        prior = self.base.predict_changes(rows).astype(np.float32)
        base_coefficients = np.einsum(
            "ba,gaf->gbf",
            self.pseudoinverse,
            prior,
        )
        features = np.concatenate(
            [
                grouped_context_features(self._contexts(rows)),
                base_coefficients.reshape(len(rows), -1),
            ],
            axis=1,
        ).astype(np.float32)
        correction_coefficients = np.stack(
            [
                np.asarray(model.predict(features), dtype=np.float32)
                for model in self.models
            ],
            axis=2,
        )
        correction = np.einsum(
            "ab,gbf->gaf",
            self.basis,
            correction_coefficients,
        ).astype(np.float32)
        prediction = (
            prior + correction * self.field_blend[None, None, :]
        )
        zero = np.all(self.basis == 0.0, axis=1)
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


def load_grouped_forward_tree_runtime_v9(
    path: Path,
    torch: Any,
    device: Any,
) -> tuple[GroupedForwardTreeRuntimeV9, dict[str, Any]]:
    runtime = GroupedForwardTreeRuntimeV9(path, torch, device)
    return runtime, runtime.artifact

