"""Runtime for group-balanced physical-success inverse action ranking."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID, MOVEMENT
from control_rebuild_v4.inverse_data import state_mapping
from physics_structured_rebuild_v9.inverse_lightgbm_runtime import (
    normalized_per_request,
    ranker_features,
)
from physics_structured_rebuild_v9.inverse_runtime_v8 import (
    load_inverse_runtime_v8,
)
from specialist_rebuild_v2.common import inverse_context


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def apply_gate(
    base_indices: np.ndarray,
    candidate_indices: np.ndarray,
    probabilities: np.ndarray,
    model_scores: np.ndarray,
    rule: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(len(base_indices))
    candidate_probability = probabilities[positions, candidate_indices]
    base_probability = probabilities[positions, base_indices]
    score_advantage = (
        model_scores[positions, candidate_indices]
        - model_scores[positions, base_indices]
    )
    use_candidate = (
        (candidate_indices != base_indices)
        & (
            candidate_probability
            >= float(rule["candidate_probability_min"])
        )
        & (
            candidate_probability - base_probability
            >= float(rule["probability_advantage_min"])
        )
        & (base_probability <= float(rule["base_probability_max"]))
        & (
            score_advantage
            >= float(rule["model_score_advantage_min"])
        )
    )
    selected = np.where(use_candidate, candidate_indices, base_indices)
    return selected.astype(np.int64), use_candidate


class InverseSuccessRankerRuntimeV9:
    """Rank 81 actions by learned physical success with a protected base gate."""

    def __init__(
        self,
        artifact_path: Path,
        torch: Any,
        device: Any,
    ) -> None:
        path = artifact_path.resolve()
        with path.open("rb") as stream:
            artifact = pickle.load(stream)
        if artifact.get("model") != "group_balanced_inverse_success_v9":
            raise ValueError("unexpected inverse-success artifact")
        base_path = Path(str(artifact["base_inverse_artifact"])).resolve()
        if sha256(base_path) != artifact["base_inverse_artifact_sha256"]:
            raise ValueError("inverse-success base checksum differs")
        self.base, _ = load_inverse_runtime_v8(base_path, torch, device)
        self.classifier = artifact["classifier"]
        self.regressor = artifact["regressor"]
        self.quality_weight = float(artifact["quality_weight"])
        self.model_weight = float(artifact["model_weight"])
        self.rule = dict(artifact["gate_rule"])
        self.version = str(artifact["version"])

    def score_feature_arrays(
        self,
        contexts: np.ndarray,
        desired: np.ndarray,
        candidate_states: np.ndarray,
        *,
        batch_size: int = 128,
    ) -> dict[str, Any]:
        contexts_out = np.asarray(contexts, dtype=np.float32)
        desired_out = np.asarray(desired, dtype=np.float32)
        candidates = np.asarray(candidate_states, dtype=np.float32)
        base = self.base.score_feature_arrays(
            contexts_out,
            desired_out,
            candidates,
            batch_size=batch_size,
        )
        base_scores = np.asarray(base["scores"], dtype=np.float32)
        features = ranker_features(
            contexts_out,
            desired_out,
            candidates,
            base_scores,
            include_action_basis=True,
        )
        flat = features.reshape(-1, features.shape[-1])
        raw_probability = np.asarray(
            self.classifier.predict(flat, raw_score=True),
            dtype=np.float32,
        ).reshape(len(features), len(ACTION_GRID))
        predicted_log_quality = np.asarray(
            self.regressor.predict(flat),
            dtype=np.float32,
        ).reshape(len(features), len(ACTION_GRID))
        probability_score = normalized_per_request(raw_probability)
        quality_score = normalized_per_request(-predicted_log_quality)
        model_scores = (
            probability_score + self.quality_weight * quality_score
        )
        blended = (
            normalized_per_request(base_scores)
            + self.model_weight * normalized_per_request(model_scores)
        )
        candidate_indices = (
            blended - 1e-7 * MOVEMENT[None, :]
        ).argmax(axis=1)
        base_indices = np.asarray(base["selected_indices"], dtype=np.int64)
        probabilities = sigmoid(raw_probability)
        selected, applied = apply_gate(
            base_indices,
            candidate_indices,
            probabilities,
            model_scores,
            self.rule,
        )
        return {
            **base,
            "selected_indices": selected,
            "selected_actions": [
                ACTION_GRID[int(index)] for index in selected
            ],
            "base_selected_indices": base_indices,
            "candidate_indices": candidate_indices,
            "success_probabilities": probabilities,
            "predicted_log_quality": predicted_log_quality,
            "model_scores": model_scores.astype(np.float32),
            "blended_scores": blended.astype(np.float32),
            "gate_applied": applied,
            "quality_weight": self.quality_weight,
            "model_weight": self.model_weight,
        }

    def score_requests(
        self,
        setups: Sequence[Mapping[str, Any]],
        context_current: np.ndarray,
        context_desired: np.ndarray,
        candidate_states: np.ndarray,
        feature_desired: np.ndarray | None = None,
        batch_size: int = 128,
    ) -> dict[str, Any]:
        desired_context = np.asarray(context_desired, dtype=np.float32)
        desired_features = (
            desired_context
            if feature_desired is None
            else np.asarray(feature_desired, dtype=np.float32)
        )
        current = np.asarray(context_current, dtype=np.float32)
        contexts = np.asarray(
            [
                inverse_context(
                    setups[index],
                    state_mapping(current[index]),
                    state_mapping(desired_context[index]),
                )
                for index in range(len(current))
            ],
            dtype=np.float32,
        )
        return self.score_feature_arrays(
            contexts,
            desired_features,
            np.asarray(candidate_states, dtype=np.float32),
            batch_size=batch_size,
        )


def load_inverse_success_ranker_runtime_v9(
    artifact_path: Path,
    torch: Any,
    device: Any,
) -> tuple[InverseSuccessRankerRuntimeV9, dict[str, Any]]:
    path = artifact_path.resolve()
    with path.open("rb") as stream:
        artifact = pickle.load(stream)
    return InverseSuccessRankerRuntimeV9(path, torch, device), artifact
