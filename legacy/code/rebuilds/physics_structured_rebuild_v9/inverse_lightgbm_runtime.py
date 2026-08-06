"""Runtime for the LightGBM re-ranker over frozen transformer inverse scores."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import (
    ACTION_GRID,
    ACTION_NORMALIZED,
    MOVEMENT,
    action_basis,
    inverse_candidate_features,
)
from control_rebuild_v4.inverse_data import state_mapping
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


def normalized_per_request(values: np.ndarray) -> np.ndarray:
    values_out = np.asarray(values, dtype=np.float32)
    mean = values_out.mean(axis=1, keepdims=True)
    scale = values_out.std(axis=1, keepdims=True)
    return (values_out - mean) / np.maximum(scale, 1e-6)


def rank_positions(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, axis=1), axis=1)
    return order.astype(np.float32) / max(values.shape[1] - 1, 1)


def ranker_features(
    contexts: np.ndarray,
    desired: np.ndarray,
    candidate_states: np.ndarray,
    base_scores: np.ndarray,
    *,
    include_action_basis: bool = False,
) -> np.ndarray:
    candidate, _, status = inverse_candidate_features(
        candidate_states,
        desired,
    )
    repeated_context = np.repeat(
        np.asarray(contexts, dtype=np.float32)[:, None, :],
        len(ACTION_GRID),
        axis=1,
    )
    repeated_status = np.repeat(
        status[:, None, :],
        len(ACTION_GRID),
        axis=1,
    )
    normalized_base = normalized_per_request(base_scores)
    positions = rank_positions(base_scores)
    top_gap = base_scores.max(axis=1, keepdims=True) - base_scores
    columns = [
        repeated_context,
        candidate,
        repeated_status,
        normalized_base[:, :, None],
        positions[:, :, None],
        top_gap[:, :, None],
    ]
    if include_action_basis:
        action_features = np.concatenate(
            [ACTION_NORMALIZED, action_basis()],
            axis=1,
        ).astype(np.float32)
        columns.append(
            np.broadcast_to(
                action_features[None, :, :],
                (len(contexts), *action_features.shape),
            )
        )
    features = np.concatenate(columns, axis=2)
    return features.astype(np.float32)


class InverseLightGBMRankerRuntimeV9:
    """Re-rank 81 candidates while retaining transformer status prediction."""

    def __init__(
        self,
        artifact_path: Path,
        torch: Any,
        device: Any,
    ) -> None:
        path = artifact_path.resolve()
        with path.open("rb") as stream:
            artifact = pickle.load(stream)
        if artifact.get("model") != "lightgbm_inverse_reranker_v9":
            raise ValueError("unexpected LightGBM inverse artifact")
        base_path = Path(str(artifact["base_inverse_artifact"])).resolve()
        if sha256(base_path) != artifact["base_inverse_artifact_sha256"]:
            raise ValueError("LightGBM inverse base checksum differs")
        self.base, _ = load_inverse_runtime_v8(base_path, torch, device)
        self.ranker = artifact["ranker"]
        self.ranker_weight = float(artifact["ranker_weight"])
        self.include_action_basis = bool(
            artifact.get("include_action_basis", False)
        )
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
            include_action_basis=self.include_action_basis,
        )
        ranker_raw = self.ranker.predict(
            features.reshape(-1, features.shape[-1])
        ).reshape(len(features), len(ACTION_GRID))
        scores = (
            normalized_per_request(base_scores)
            + self.ranker_weight * normalized_per_request(ranker_raw)
        )
        selected = (scores - 1e-7 * MOVEMENT[None, :]).argmax(axis=1)
        return {
            **base,
            "selected_indices": selected,
            "selected_actions": [
                ACTION_GRID[int(index)] for index in selected
            ],
            "scores": scores.astype(np.float32),
            "transformer_scores": base_scores,
            "lightgbm_scores": ranker_raw.astype(np.float32),
            "ranker_weight": self.ranker_weight,
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
        candidates = np.asarray(candidate_states, dtype=np.float32)
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
            candidates,
            batch_size=batch_size,
        )


def load_inverse_lightgbm_runtime_v9(
    artifact_path: Path,
    torch: Any,
    device: Any,
) -> tuple[InverseLightGBMRankerRuntimeV9, dict[str, Any]]:
    path = artifact_path.resolve()
    with path.open("rb") as stream:
        artifact = pickle.load(stream)
    return InverseLightGBMRankerRuntimeV9(path, torch, device), artifact
