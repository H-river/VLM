"""Inference runtime for the shared forward-direction v6 artifact."""

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
from specialist_rebuild_v2.common import (
    CLASSES,
    DIRECTION_FIELDS,
    STATE_FIELDS,
    forward_feature,
    raw_state_array,
)

from joint_forward_direction_v6.models import shared_forward_direction_model


class SharedForwardDirectionRuntimeV6:
    """Serve numerical changes and direction classes from one shared encoder."""

    def __init__(
        self,
        torch: Any,
        artifact: Mapping[str, Any],
        device: Any,
        artifact_path: Path,
    ) -> None:
        if artifact.get("model") != "shared_forward_direction_residual_v6":
            raise ValueError("expected a shared forward-direction v6 artifact")
        if tuple(artifact.get("state_fields", ())) != tuple(STATE_FIELDS):
            raise ValueError("v6 state-field order differs")
        if tuple(artifact.get("direction_fields", ())) != tuple(DIRECTION_FIELDS):
            raise ValueError("v6 direction-field order differs")
        if tuple(artifact.get("classes", ())) != tuple(CLASSES):
            raise ValueError("v6 direction class order differs")
        config = artifact["model_config"]
        self.torch = torch
        self.device = device
        self.artifact = artifact
        self.artifact_path = artifact_path.resolve()
        self.input_mean = np.asarray(artifact["input_mean"], dtype=np.float32)
        self.input_scale = np.asarray(artifact["input_scale"], dtype=np.float32)
        if self.input_mean.shape != self.input_scale.shape:
            raise ValueError("v6 normalization arrays differ")
        self.model = shared_forward_direction_model(
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
        if not prior_path.is_absolute():
            prior_path = (self.artifact_path.parent / prior_path).resolve()
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
    ) -> tuple[np.ndarray, np.ndarray]:
        if not rows:
            return (
                np.empty((0, len(ACTION_GRID), 5), dtype=np.float32),
                np.empty((0, len(ACTION_GRID), 5, 3), dtype=np.float32),
            )
        features = self._features(rows)
        prior = self.base_forward.predict_changes(rows)
        flat_features = features.reshape(-1, features.shape[-1])
        flat_prior = prior.reshape(-1, prior.shape[-1])
        combined = np.concatenate([flat_features, flat_prior], axis=1)
        combined = (combined - self.input_mean) / self.input_scale
        change_parts = []
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
                change, logits = self.model(values, prior_tensor)
                change_parts.append(change.cpu().numpy())
                logit_parts.append(logits.cpu().numpy())
        changes = np.concatenate(change_parts).reshape(
            len(rows), len(ACTION_GRID), 5
        )
        logits = np.concatenate(logit_parts).reshape(
            len(rows), len(ACTION_GRID), 5, 3
        )
        changes[:, ZERO_ACTION_INDEX, :] = 0.0
        logits[:, ZERO_ACTION_INDEX, :, :] = -30.0
        logits[:, ZERO_ACTION_INDEX, :, 1] = 30.0
        return changes.astype(np.float32), logits.astype(np.float32)

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
        return self._predict(rows)[1].argmax(axis=-1).astype(np.int64)

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
            raise ValueError("v6 accepts only the registered 81-action grid")
        row = {
            "group_id": "shared_forward_direction_v6",
            "setup": dict(setup),
            "current_beam_state": dict(current),
        }
        changes, logits = self._predict([row])
        selected_logits = logits[0, action_index]
        shifted = selected_logits - selected_logits.max(
            axis=-1, keepdims=True
        )
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        indices = probabilities.argmax(axis=-1)
        return {
            "directions": {
                field: CLASSES[int(indices[field_index])]
                for field_index, field in enumerate(DIRECTION_FIELDS)
            },
            "probabilities": {
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


def load_forward_direction_runtime_v6(
    artifact_path: Path,
    torch: Any,
    device: Any,
) -> tuple[SharedForwardDirectionRuntimeV6, dict[str, Any]]:
    path = artifact_path.resolve()
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return SharedForwardDirectionRuntimeV6(
        torch,
        artifact,
        device,
        path,
    ), artifact

