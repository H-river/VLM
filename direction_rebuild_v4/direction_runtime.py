"""Inference runtime for the balanced v4 direction classifier."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID
from direction_rebuild_v4.data import CLASSES
from direction_rebuild_v4.models import direction_classifier_v4
from specialist_rebuild_v2.common import DIRECTION_FIELDS, forward_feature


class DirectionRuntimeV4:
    def __init__(
        self,
        torch: Any,
        artifact: Mapping[str, Any],
        device: Any,
        forward_runtime: Any | None = None,
    ) -> None:
        if artifact.get("model") != "balanced_five_head_direction_v4":
            raise ValueError("expected a balanced_five_head_direction_v4 artifact")
        if tuple(artifact.get("classes", ())) != CLASSES:
            raise ValueError("direction artifact class order differs")
        if tuple(artifact.get("direction_fields", ())) != tuple(DIRECTION_FIELDS):
            raise ValueError("direction artifact field order differs")
        self.torch = torch
        self.device = device
        self.artifact = artifact
        self.feature_mode = str(artifact.get("feature_mode", "engineered_46"))
        self.forward_runtime = forward_runtime
        if (
            self.feature_mode == "engineered_46_plus_forward_v4_change_5"
            and self.forward_runtime is None
        ):
            raise ValueError(
                "fused direction artifact requires the pinned forward-v4 runtime"
            )
        self.mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
        self.scale = np.asarray(artifact["feature_scale"], dtype=np.float32)
        self.model = direction_classifier_v4(
            torch,
            int(artifact["input_dim"]),
        ).to(device)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

    def predict_logits(
        self,
        setups: Sequence[Mapping[str, Any]],
        current_states: Sequence[Mapping[str, Any]],
        actions: Sequence[Mapping[str, Any]],
        batch_size: int = 4096,
    ) -> np.ndarray:
        if not (len(setups) == len(current_states) == len(actions)):
            raise ValueError("direction runtime inputs must have equal lengths")
        if not setups:
            return np.empty((0, 5, 3), dtype=np.float32)
        features = np.asarray(
            [
                forward_feature(setup, current, action)
                for setup, current, action in zip(
                    setups,
                    current_states,
                    actions,
                )
            ],
            dtype=np.float32,
        )
        if self.feature_mode == "engineered_46_plus_forward_v4_change_5":
            rows = [
                {
                    "group_id": f"direction_runtime_v4_{index}",
                    "setup": dict(setup),
                    "current_beam_state": dict(current),
                }
                for index, (setup, current) in enumerate(
                    zip(setups, current_states)
                )
            ]
            candidate_changes = self.forward_runtime.predict_changes(rows)
            selected_changes = []
            for index, action in enumerate(actions):
                action_index = next(
                    (
                        candidate_index
                        for candidate_index, candidate in enumerate(ACTION_GRID)
                        if all(
                            float(action[field]) == float(candidate[field])
                            for field in candidate
                        )
                    ),
                    None,
                )
                if action_index is None:
                    raise ValueError(
                        "direction runtime action is outside the 81-action grid"
                    )
                selected_changes.append(candidate_changes[index, action_index])
            features = np.concatenate(
                [
                    features,
                    np.asarray(selected_changes, dtype=np.float32),
                ],
                axis=1,
            )
        elif self.feature_mode != "engineered_46":
            raise ValueError(f"unsupported direction feature mode: {self.feature_mode}")
        scaled = (features - self.mean) / self.scale
        parts = []
        with self.torch.inference_mode():
            for start in range(0, len(scaled), batch_size):
                values = self.torch.as_tensor(
                    scaled[start : start + batch_size],
                    dtype=self.torch.float32,
                    device=self.device,
                )
                parts.append(self.model(values).float().cpu().numpy())
        return np.concatenate(parts)

    def predict_one(
        self,
        setup: Mapping[str, Any],
        current: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> dict[str, Any]:
        logits = self.predict_logits([setup], [current], [action])[0]
        shifted = logits - logits.max(axis=-1, keepdims=True)
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
            "model_version": str(self.artifact["version"]),
            "simulator_at_inference": False,
        }


class TreeDirectionRuntimeV4:
    def __init__(self, artifact: Mapping[str, Any]) -> None:
        if (
            artifact.get("model")
            != "balanced_five_head_hist_gradient_boosting_direction_v4"
        ):
            raise ValueError("expected a balanced five-head tree artifact")
        if tuple(artifact.get("classes", ())) != CLASSES:
            raise ValueError("tree direction artifact class order differs")
        if tuple(artifact.get("direction_fields", ())) != tuple(DIRECTION_FIELDS):
            raise ValueError("tree direction artifact field order differs")
        self.artifact = artifact
        self.models = list(artifact["models"])
        if len(self.models) != len(DIRECTION_FIELDS):
            raise ValueError("tree direction artifact must contain five heads")
        if any(
            tuple(int(value) for value in model.classes_) != (0, 1, 2)
            for model in self.models
        ):
            raise ValueError("tree direction head class order differs")

    def predict_one(
        self,
        setup: Mapping[str, Any],
        current: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> dict[str, Any]:
        feature = forward_feature(setup, current, action)[None, :]
        probabilities = np.stack(
            [model.predict_proba(feature)[0] for model in self.models]
        )
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
            "model_version": str(self.artifact["version"]),
            "simulator_at_inference": False,
        }


def load_direction_runtime_v4(
    artifact_path: Path,
    torch: Any,
    device: Any,
    forward_runtime: Any | None = None,
) -> tuple[DirectionRuntimeV4 | TreeDirectionRuntimeV4, dict[str, Any]]:
    if artifact_path.suffix == ".pkl":
        with artifact_path.open("rb") as stream:
            artifact = pickle.load(stream)
    else:
        artifact = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=False,
        )
    if (
        artifact.get("model")
        == "balanced_five_head_hist_gradient_boosting_direction_v4"
    ):
        return TreeDirectionRuntimeV4(artifact), artifact
    return (
        DirectionRuntimeV4(
            torch,
            artifact,
            device,
            forward_runtime=forward_runtime,
        ),
        artifact,
    )
