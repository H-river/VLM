"""Risk-based selector between the retained and grouped forward surfaces."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID, tolerance_from_current
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from physics_structured_rebuild_v9.grouped_forward_selector_runtime import (
    grouped_selector_features,
)
from physics_structured_rebuild_v9.grouped_forward_tree_runtime import (
    load_grouped_forward_tree_runtime_v9,
)
from specialist_rebuild_v2.common import STATE_FIELDS


def grouped_risk_features(
    rows: Sequence[Mapping[str, Any]],
    current: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    """Build action features plus whole-surface disagreement summaries."""

    local = grouped_selector_features(rows, current, candidate)
    current_values = np.asarray(current, dtype=np.float32)
    candidate_values = np.asarray(candidate, dtype=np.float32)
    difference = np.abs(candidate_values - current_values)
    summary = np.concatenate(
        [
            difference.mean(axis=1),
            difference.std(axis=1),
            difference.max(axis=1),
            current_values.std(axis=1),
            candidate_values.std(axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    repeated = np.repeat(summary[:, None, :], len(ACTION_GRID), axis=1)
    return np.concatenate([local, repeated], axis=2).astype(np.float32)


def selector_score(
    artifact: Mapping[str, Any],
    features: np.ndarray,
) -> np.ndarray:
    """Return the calibrated expert-advantage score."""

    flat = np.asarray(features, dtype=np.float32).reshape(
        -1,
        features.shape[-1],
    )
    candidate_probability = artifact[
        "candidate_success_classifier"
    ].predict_proba(flat)[:, 1]
    current_probability = artifact[
        "current_success_classifier"
    ].predict_proba(flat)[:, 1]
    probability_advantage = candidate_probability - current_probability
    current_log_error = artifact["current_error_regressor"].predict(flat)
    candidate_log_error = artifact["candidate_error_regressor"].predict(flat)
    error_advantage = current_log_error - candidate_log_error
    score_kind = str(artifact["score_kind"])
    if score_kind == "probability_advantage":
        score = probability_advantage
    elif score_kind == "error_advantage":
        score = error_advantage
    elif score_kind.startswith("hybrid_"):
        probability_scale = artifact["probability_scale"]
        error_scale = artifact["error_scale"]
        probability_z = (
            probability_advantage - float(probability_scale["mean"])
        ) / float(probability_scale["std"])
        error_z = (
            error_advantage - float(error_scale["mean"])
        ) / float(error_scale["std"])
        score = probability_z + float(artifact["hybrid_weight"]) * error_z
    else:
        raise ValueError("unexpected grouped forward risk score")
    return np.asarray(score, dtype=np.float32).reshape(
        features.shape[0],
        features.shape[1],
    )


class GroupedForwardRiskSelectorRuntimeV9:
    """Choose one complete prediction per action from learned failure risk."""

    def __init__(self, path: Path, torch: Any, device: Any) -> None:
        with path.resolve().open("rb") as stream:
            artifact = pickle.load(stream)
        if artifact.get("model") != "hgb_grouped_forward_risk_selector_v9":
            raise ValueError("unexpected grouped forward risk selector")
        self.artifact = artifact
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
        features = grouped_risk_features(rows, current, candidate)
        score = selector_score(self.artifact, features)
        choose_candidate = score >= self.threshold
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
            current[:, None, :] + changes * tolerance[:, None, :]
        ).astype(np.float32)


def load_grouped_forward_risk_selector_runtime_v9(
    path: Path,
    torch: Any,
    device: Any,
) -> tuple[GroupedForwardRiskSelectorRuntimeV9, dict[str, Any]]:
    runtime = GroupedForwardRiskSelectorRuntimeV9(path, torch, device)
    return runtime, runtime.artifact
