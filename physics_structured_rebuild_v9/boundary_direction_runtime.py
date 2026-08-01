"""Runtime for boundary-aware corrections over the accepted direction hybrid."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID
from direction_rebuild_v4.data import labels_from_normalized_change
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    CLASSES,
    DIRECTION_FIELDS,
    HybridDirectionRuntimeV9,
)
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    forward_feature,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def correction_features(
    physical: np.ndarray,
    prior: np.ndarray,
    current_forward: np.ndarray,
    threshold_changes: np.ndarray,
    base_labels: np.ndarray,
) -> np.ndarray:
    one_hot = np.eye(3, dtype=np.float32)[base_labels].reshape(
        len(base_labels),
        -1,
    )
    return np.concatenate(
        [
            physical,
            prior,
            current_forward,
            threshold_changes,
            one_hot,
        ],
        axis=1,
    ).astype(np.float32)


def override_gate_features(
    correction_values: np.ndarray,
    threshold_changes: np.ndarray,
    base_labels: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> np.ndarray:
    """Build truth-free features for deciding whether an override is safe."""
    ordered = np.sort(candidate_probabilities, axis=2)
    candidate_labels = candidate_probabilities.argmax(axis=2)
    candidate_one_hot = np.eye(3, dtype=np.float32)[
        candidate_labels
    ].reshape(len(candidate_labels), -1)
    confidence = ordered[:, :, -1]
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    distance = np.abs(np.abs(threshold_changes) - 1.0)
    disagreement = (candidate_labels != base_labels).astype(np.float32)
    return np.concatenate(
        [
            correction_values,
            candidate_probabilities.reshape(len(candidate_labels), -1),
            candidate_one_hot,
            confidence,
            margin,
            distance,
            disagreement,
            disagreement.sum(axis=1, keepdims=True),
        ],
        axis=1,
    ).astype(np.float32)


def positive_probability(model: Any, values: np.ndarray) -> np.ndarray:
    """Return P(label=1) while tolerating a single-class fitted gate."""
    probability = np.asarray(model.predict_proba(values))
    classes = np.asarray(model.classes_, dtype=np.int64)
    matches = np.flatnonzero(classes == 1)
    if len(matches) == 0:
        return np.zeros(len(values), dtype=np.float32)
    return probability[:, int(matches[0])].astype(np.float32)


class BoundaryDirectionCorrectionRuntimeV9:
    """Correct selected near-boundary fields over a frozen hybrid."""

    def __init__(
        self,
        torch: Any,
        artifact_path: Path,
        device: Any,
    ) -> None:
        path = artifact_path.resolve()
        with path.open("rb") as stream:
            artifact = pickle.load(stream)
        if artifact.get("model") != "boundary_direction_correction_v9":
            raise ValueError("unexpected boundary-direction artifact")
        base_path = Path(str(artifact["base_direction"])).resolve()
        forward_path = Path(str(artifact["current_forward"])).resolve()
        if sha256(base_path) != artifact["base_direction_sha256"]:
            raise ValueError("boundary-direction base checksum differs")
        if sha256(forward_path) != artifact["current_forward_sha256"]:
            raise ValueError("boundary-direction forward checksum differs")
        self.base = HybridDirectionRuntimeV9(torch, base_path, device)
        self.current_forward, _ = load_forward_selector_ensemble_runtime_v9(
            forward_path,
            torch,
            device,
        )
        self.models = list(artifact["models"])
        self.rules = list(artifact["rules"])
        gate_models = artifact.get("gate_models")
        self.gate_models = (
            None if gate_models is None else list(gate_models)
        )
        gate_thresholds = artifact.get("gate_thresholds")
        self.gate_thresholds = (
            None if gate_thresholds is None else list(gate_thresholds)
        )
        if (self.gate_models is None) != (self.gate_thresholds is None):
            raise ValueError("direction override gate artifact is incomplete")
        self.version = str(artifact["version"])

    @staticmethod
    def _physical_features(
        rows: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
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

    def base_grid(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        physical = self._physical_features(rows)
        prior = self.base.shared.predict_changes(rows)
        threshold_changes = self.base.threshold_forward.predict_changes(rows)
        flat_features = physical.reshape(-1, physical.shape[-1])
        tree_features = flat_features
        if self.base.tree_feature_mode != "engineered_46":
            tree_features = np.concatenate(
                [
                    tree_features,
                    prior.reshape(-1, prior.shape[-1]),
                ],
                axis=1,
            )
        if (
            self.base.tree_feature_mode
            == (
                "engineered_46_plus_forward_v7_change_5"
                "_plus_residual_forward_5"
            )
        ):
            if self.base.residual_forward_models is None:
                raise ValueError("boundary-direction residual models missing")
            residual = np.stack(
                [
                    model.predict(tree_features)
                    for model in self.base.residual_forward_models
                ],
                axis=1,
            )
            tree_features = np.concatenate(
                [tree_features, residual.astype(np.float32)],
                axis=1,
            )
        probabilities = np.zeros(
            (len(tree_features), 5, 3),
            dtype=np.float32,
        )
        for field, model in enumerate(self.base.models):
            predicted = model.predict_proba(tree_features)
            probabilities[
                :,
                field,
                np.asarray(model.classes_, dtype=np.int64),
            ] = predicted
        threshold_flat = threshold_changes.reshape(-1, 5)
        threshold = labels_from_normalized_change(threshold_flat)
        ordered = np.sort(probabilities, axis=2)
        tree_class = probabilities.argmax(axis=2)
        distance = np.abs(np.abs(threshold_flat) - 1.0)
        boundary_limit = np.asarray(
            [
                calibration["boundary_limit"]
                for calibration in self.base.field_calibration
            ],
            dtype=np.float32,
        )
        confidence_limit = np.asarray(
            [
                calibration["confidence_limit"]
                for calibration in self.base.field_calibration
            ],
            dtype=np.float32,
        )
        margin_limit = np.asarray(
            [
                calibration["margin_limit"]
                for calibration in self.base.field_calibration
            ],
            dtype=np.float32,
        )
        apply = (
            (distance <= boundary_limit[None, :])
            & (ordered[:, :, -1] >= confidence_limit[None, :])
            & (
                ordered[:, :, -1] - ordered[:, :, -2]
                >= margin_limit[None, :]
            )
            & (tree_class != threshold)
        )
        base_labels = np.where(apply, tree_class, threshold).astype(np.int64)
        return physical, prior, threshold_changes, base_labels

    def predict_grid(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not rows:
            empty = np.empty((0, len(ACTION_GRID), 5), dtype=np.int64)
            return empty, empty.astype(np.bool_)
        physical, prior, threshold_changes, base_labels = self.base_grid(rows)
        current_forward = self.current_forward.predict_changes(rows)
        values = correction_features(
            physical.reshape(-1, physical.shape[-1]),
            prior.reshape(-1, 5),
            current_forward.reshape(-1, 5),
            threshold_changes.reshape(-1, 5),
            base_labels,
        )
        probabilities = np.zeros((len(values), 5, 3), dtype=np.float32)
        for field, model in enumerate(self.models):
            predicted = model.predict_proba(values)
            probabilities[
                :,
                field,
                np.asarray(model.classes_, dtype=np.int64),
            ] = predicted
        ordered = np.sort(probabilities, axis=2)
        model_class = probabilities.argmax(axis=2)
        distance = np.abs(
            np.abs(threshold_changes.reshape(-1, 5)) - 1.0
        )
        selected = base_labels.copy()
        applied = np.zeros_like(selected, dtype=np.bool_)
        if self.gate_models is not None:
            gate_values = override_gate_features(
                values,
                threshold_changes.reshape(-1, 5),
                base_labels,
                probabilities,
            )
            for field, (model, threshold) in enumerate(
                zip(self.gate_models, self.gate_thresholds, strict=True)
            ):
                if threshold is None:
                    continue
                gate_probability = positive_probability(model, gate_values)
                mask = (
                    (gate_probability >= float(threshold))
                    & (model_class[:, field] != selected[:, field])
                )
                selected[mask, field] = model_class[mask, field]
                applied[mask, field] = True
            shape = (len(rows), len(ACTION_GRID), 5)
            return selected.reshape(shape), applied.reshape(shape)
        for field, rule in enumerate(self.rules):
            if rule is None:
                continue
            mask = (
                (distance[:, field] <= float(rule["boundary_limit"]))
                & (
                    ordered[:, field, -1]
                    >= float(rule["confidence_limit"])
                )
                & (
                    ordered[:, field, -1]
                    - ordered[:, field, -2]
                    >= float(rule["margin_limit"])
                )
                & (model_class[:, field] != selected[:, field])
            )
            selected[mask, field] = model_class[mask, field]
            applied[mask, field] = True
        shape = (len(rows), len(ACTION_GRID), 5)
        return selected.reshape(shape), applied.reshape(shape)

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
                    for field in ACTION_FIELDS
                )
            ),
            None,
        )
        if action_index is None:
            raise ValueError(
                "boundary-aware direction accepts only the 81-action grid"
            )
        row = {
            "group_id": "boundary_direction_correction_v9",
            "setup": dict(setup),
            "current_beam_state": dict(current),
        }
        predicted, applied = self.predict_grid([row])
        labels = predicted[0, action_index]
        correction = applied[0, action_index]
        return {
            "directions": {
                field: CLASSES[int(labels[index])]
                for index, field in enumerate(DIRECTION_FIELDS)
            },
            "boundary_correction_applied": {
                field: bool(correction[index])
                for index, field in enumerate(DIRECTION_FIELDS)
            },
            "model_version": self.version,
            "simulator_at_inference": False,
        }


def load_boundary_direction_correction_runtime_v9(
    path: Path,
    torch: Any,
    device: Any,
) -> tuple[BoundaryDirectionCorrectionRuntimeV9, dict[str, Any]]:
    with path.resolve().open("rb") as stream:
        artifact = pickle.load(stream)
    return BoundaryDirectionCorrectionRuntimeV9(torch, path, device), artifact
