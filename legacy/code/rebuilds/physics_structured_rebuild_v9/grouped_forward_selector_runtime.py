"""Action-level selector between retained and grouped forward surfaces."""

from __future__ import annotations

import math
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID, tolerance_from_current
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from physics_structured_rebuild_v9.grouped_forward_tree_runtime import (
    load_grouped_forward_tree_runtime_v9,
)
from specialist_rebuild_v2.common import (
    SETUP_FIELDS,
    STATE_FIELDS,
    forward_feature,
)


def grouped_selector_features(
    rows: Sequence[Mapping[str, Any]],
    current: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    """Build runtime-available transition features for both predictions."""

    engineered = np.asarray(
        [
            [
                forward_feature(
                    row["setup"],
                    row["current_beam_state"],
                    action,
                )
                for action in ACTION_GRID
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    current_values = np.asarray(current, dtype=np.float32)
    candidate_values = np.asarray(candidate, dtype=np.float32)
    return np.concatenate(
        [
            engineered,
            current_values,
            candidate_values,
            np.abs(candidate_values - current_values),
        ],
        axis=2,
    ).astype(np.float32)


class GroupedForwardSelectorRuntimeV9:
    """Select the grouped correction independently for each grid action."""

    def __init__(self, path: Path, torch: Any, device: Any) -> None:
        with path.resolve().open("rb") as stream:
            artifact = pickle.load(stream)
        if artifact.get("model") != "hgb_grouped_forward_selector_v9":
            raise ValueError("unexpected grouped forward selector artifact")
        self.artifact = artifact
        self.classifier = artifact["classifier"]
        self.threshold = float(artifact["threshold"])
        self.current, _ = load_forward_selector_ensemble_runtime_v9(
            Path(str(artifact["current_forward"])).resolve(),
            torch,
            device,
        )
        self.candidate, _ = load_grouped_forward_tree_runtime_v9(
            Path(str(artifact["candidate_forward"])).resolve(),
            torch,
            device,
        )

    def predict_changes(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        current = self.current.predict_changes(rows)
        candidate = self.candidate.predict_changes(rows)
        features = grouped_selector_features(
            rows,
            current,
            candidate,
        )
        probability = self.classifier.predict_proba(
            features.reshape(-1, features.shape[-1])
        )[:, 1].reshape(len(rows), len(ACTION_GRID))
        choose_candidate = probability >= self.threshold
        return np.where(
            choose_candidate[:, :, None],
            candidate,
            current,
        ).astype(np.float32)

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


def load_grouped_forward_selector_runtime_v9(
    path: Path,
    torch: Any,
    device: Any,
) -> tuple[GroupedForwardSelectorRuntimeV9, dict[str, Any]]:
    runtime = GroupedForwardSelectorRuntimeV9(path, torch, device)
    return runtime, runtime.artifact

