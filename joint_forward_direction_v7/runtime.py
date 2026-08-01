"""Inference runtime for cross-fitted conservative shared v7."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID, tolerance_from_current
from control_rebuild_v5.forward_runtime import (
    ZERO_ACTION_INDEX,
    load_forward_runtime_v5,
)
from direction_rebuild_v4.data import labels_from_normalized_change
from joint_forward_direction_v7.models import (
    shared_forward_direction_model_v7,
)
from specialist_rebuild_v2.common import (
    CLASSES,
    DIRECTION_FIELDS,
    STATE_FIELDS,
    forward_feature,
    raw_state_array,
)


def gated_direction_indices(
    changes: np.ndarray,
    correction_logits: np.ndarray,
    *,
    minimum_probability: float,
    minimum_margin: float,
) -> np.ndarray:
    """Apply learned class changes only when correction confidence is sufficient."""

    base = labels_from_normalized_change(changes)
    shifted = correction_logits - correction_logits.max(
        axis=-1, keepdims=True
    )
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    ordered = np.sort(probabilities, axis=-1)
    predicted = probabilities.argmax(axis=-1)
    confidence = ordered[..., -1]
    margin = ordered[..., -1] - ordered[..., -2]
    apply = (
        (predicted != base)
        & (confidence >= float(minimum_probability))
        & (margin >= float(minimum_margin))
    )
    return np.where(apply, predicted, base).astype(np.int64)


class SharedForwardDirectionRuntimeV7:
    """Serve calibrated numerical and conservative direction outputs."""

    def __init__(
        self,
        torch: Any,
        artifact: Mapping[str, Any],
        device: Any,
        artifact_path: Path,
    ) -> None:
        if artifact.get("model") != "cross_fitted_shared_forward_direction_v7":
            raise ValueError("expected a cross-fitted shared-v7 artifact")
        if tuple(artifact.get("state_fields", ())) != tuple(STATE_FIELDS):
            raise ValueError("v7 state-field order differs")
        if tuple(artifact.get("direction_fields", ())) != tuple(DIRECTION_FIELDS):
            raise ValueError("v7 direction-field order differs")
        if tuple(artifact.get("classes", ())) != tuple(CLASSES):
            raise ValueError("v7 direction class order differs")
        self.torch = torch
        self.device = device
        self.artifact = artifact
        self.artifact_path = artifact_path.resolve()
        self.input_mean = np.asarray(artifact["input_mean"], dtype=np.float32)
        self.input_scale = np.asarray(artifact["input_scale"], dtype=np.float32)
        self.calibration = dict(artifact["calibration"])
        config = artifact["model_config"]
        self.model = shared_forward_direction_model_v7(
            torch,
            int(config["input_dim"]),
            width=int(config["width"]),
            hidden_dim=int(config["hidden_dim"]),
            residual_blocks=int(config["residual_blocks"]),
            dropout=float(config["dropout"]),
        )
        self.model.load_state_dict(artifact["state_dict"])
        self.model.to(device)
        self.model.eval()
        prior_path = Path(str(artifact["base_forward_artifact"]))
        self.base_forward, _ = load_forward_runtime_v5(prior_path)

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

    def _predict(
        self,
        rows: Sequence[Mapping[str, Any]],
        batch_size: int = 8192,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not rows:
            return (
                np.empty((0, len(ACTION_GRID), 5), dtype=np.float32),
                np.empty((0, len(ACTION_GRID), 5), dtype=np.int64),
                np.empty((0, len(ACTION_GRID), 5, 3), dtype=np.float32),
            )
        features = self._features(rows)
        prior = self.base_forward.predict_changes(rows)
        flat_features = features.reshape(-1, features.shape[-1])
        flat_prior = prior.reshape(-1, prior.shape[-1])
        combined = np.concatenate([flat_features, flat_prior], axis=1)
        combined = (combined - self.input_mean) / self.input_scale
        raw_change_parts = []
        logit_parts = []
        with self.torch.inference_mode():
            for start in range(0, len(combined), batch_size):
                values = self.torch.as_tensor(
                    combined[start : start + batch_size],
                    dtype=self.torch.float32,
                    device=self.device,
                )
                prior_tensor = self.torch.as_tensor(
                    flat_prior[start : start + batch_size],
                    dtype=self.torch.float32,
                    device=self.device,
                )
                changes, logits = self.model(values, prior_tensor)
                raw_change_parts.append(changes.cpu().numpy())
                logit_parts.append(logits.cpu().numpy())
        raw_changes = np.concatenate(raw_change_parts)
        residual = raw_changes - flat_prior
        changes = flat_prior + float(
            self.calibration["residual_blend"]
        ) * residual
        logits = np.concatenate(logit_parts)
        directions = gated_direction_indices(
            changes,
            logits,
            minimum_probability=float(
                self.calibration["direction_minimum_probability"]
            ),
            minimum_margin=float(
                self.calibration["direction_minimum_margin"]
            ),
        )
        changes = changes.reshape(len(rows), len(ACTION_GRID), 5)
        directions = directions.reshape(len(rows), len(ACTION_GRID), 5)
        logits = logits.reshape(len(rows), len(ACTION_GRID), 5, 3)
        changes[:, ZERO_ACTION_INDEX, :] = 0.0
        directions[:, ZERO_ACTION_INDEX, :] = 1
        return (
            changes.astype(np.float32),
            directions.astype(np.int64),
            logits.astype(np.float32),
        )

    def predict_changes(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 512,
    ) -> np.ndarray:
        del group_batch
        return self._predict(rows)[0]

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

    def predict_direction_indices(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        return self._predict(rows)[1]

    def predict_one(
        self,
        setup: Mapping[str, Any],
        current: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> dict[str, Any]:
        action_index = next(
            (
                index
                for index, candidate in enumerate(ACTION_GRID)
                if all(
                    float(action[field]) == float(candidate[field])
                    for field in candidate
                )
            ),
            None,
        )
        if action_index is None:
            raise ValueError("v7 accepts only the registered 81-action grid")
        row = {
            "group_id": "shared_forward_direction_v7",
            "setup": dict(setup),
            "current_beam_state": dict(current),
        }
        changes, directions, logits = self._predict([row])
        selected_logits = logits[0, action_index]
        shifted = selected_logits - selected_logits.max(
            axis=-1, keepdims=True
        )
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        return {
            "directions": {
                field: CLASSES[int(directions[0, action_index, field_index])]
                for field_index, field in enumerate(DIRECTION_FIELDS)
            },
            "correction_probabilities": {
                field: {
                    class_name: float(
                        probabilities[field_index, class_index]
                    )
                    for class_index, class_name in enumerate(CLASSES)
                }
                for field_index, field in enumerate(DIRECTION_FIELDS)
            },
            "normalized_change": {
                field: float(changes[0, action_index, field_index])
                for field_index, field in enumerate(STATE_FIELDS)
            },
            "model_version": str(self.artifact["version"]),
            "simulator_at_inference": False,
        }


def load_forward_direction_runtime_v7(
    artifact_path: Path,
    torch: Any,
    device: Any,
) -> tuple[SharedForwardDirectionRuntimeV7, dict[str, Any]]:
    path = artifact_path.resolve()
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return SharedForwardDirectionRuntimeV7(
        torch,
        artifact,
        device,
        path,
    ), artifact

