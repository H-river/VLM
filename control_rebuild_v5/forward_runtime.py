"""Inference runtime for the five-head forward tree v5."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID, tolerance_from_current
from specialist_rebuild_v2.common import (
    STATE_FIELDS,
    forward_feature,
    raw_state_array,
)

ZERO_ACTION_INDEX = next(
    index
    for index, action in enumerate(ACTION_GRID)
    if all(float(value) == 0.0 for value in action.values())
)


class ForwardTreeRuntimeV5:
    """Predict five normalized changes for every registered action."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        if artifact.get("model") != "five_head_hist_gradient_boosting_forward_v5":
            raise ValueError("expected a five-head forward-v5 tree artifact")
        if tuple(artifact.get("state_fields", ())) != tuple(STATE_FIELDS):
            raise ValueError("forward-v5 state-field order differs")
        if len(artifact.get("models", ())) != len(STATE_FIELDS):
            raise ValueError("forward-v5 artifact must contain five heads")
        self.artifact = artifact
        self.models = list(artifact["models"])

    @staticmethod
    def _features(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray(
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

    def predict_changes(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 512,
    ) -> np.ndarray:
        if not rows:
            return np.empty((0, len(ACTION_GRID), 5), dtype=np.float32)
        features = self._features(rows)
        flat = features.reshape(-1, features.shape[-1])
        predictions = np.stack(
            [model.predict(flat) for model in self.models],
            axis=1,
        ).reshape(len(rows), len(ACTION_GRID), len(STATE_FIELDS))
        predictions[:, ZERO_ACTION_INDEX, :] = 0.0
        return predictions.astype(np.float32)

    def predict_states(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 512,
    ) -> np.ndarray:
        changes = self.predict_changes(rows, group_batch=group_batch)
        current = np.asarray(
            [raw_state_array(row["current_beam_state"]) for row in rows],
            dtype=np.float32,
        )
        tolerance = np.asarray(
            [
                tolerance_from_current(row["current_beam_state"])
                for row in rows
            ],
            dtype=np.float32,
        )
        return current[:, None, :] + changes * tolerance[:, None, :]


def load_forward_runtime_v5(
    artifact_path: Path,
    torch: Any | None = None,
    device: Any | None = None,
) -> tuple[ForwardTreeRuntimeV5, dict[str, Any]]:
    del torch, device
    with artifact_path.open("rb") as stream:
        artifact = pickle.load(stream)
    return ForwardTreeRuntimeV5(artifact), artifact

