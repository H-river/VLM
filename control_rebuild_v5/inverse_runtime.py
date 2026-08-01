"""Runtime for the candidate-feasibility numerical inverse tree v5."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import (
    ACTION_GRID,
    MOVEMENT,
    inverse_candidate_features,
)
from control_rebuild_v4.inverse_data import state_mapping
from specialist_rebuild_v2.common import inverse_context


class InverseTreeRuntimeV5:
    """Rank the fixed 81 actions and classify request reachability."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        if artifact.get("model") != "candidate_feasibility_inverse_tree_v5":
            raise ValueError("expected a candidate-feasibility inverse-v5 artifact")
        self.artifact = artifact
        self.candidate_model = artifact["candidate_model"]
        self.cost_correction_model = artifact["cost_correction_model"]
        self.status_model = artifact["status_model"]
        self.alpha = float(artifact["candidate_logit_alpha"])
        self.cost_correction_beta = float(artifact["cost_correction_beta"])
        self.statuses = list(artifact["statuses"])
        candidate_classes = tuple(
            int(value) for value in self.candidate_model.classes_
        )
        if candidate_classes != (0, 1):
            raise ValueError("inverse-v5 candidate class order differs")
        status_classes = tuple(int(value) for value in self.status_model.classes_)
        if status_classes != (0, 1, 2):
            raise ValueError("inverse-v5 status class order differs")

    def score_requests(
        self,
        setups: Sequence[Mapping[str, Any]],
        context_current: np.ndarray,
        context_desired: np.ndarray,
        candidate_states: np.ndarray,
        feature_desired: np.ndarray | None = None,
        batch_size: int = 512,
    ) -> dict[str, Any]:
        current = np.asarray(context_current, dtype=np.float32)
        desired_context = np.asarray(context_desired, dtype=np.float32)
        desired_features = (
            desired_context
            if feature_desired is None
            else np.asarray(feature_desired, dtype=np.float32)
        )
        candidates = np.asarray(candidate_states, dtype=np.float32)
        length = len(current)
        if not (
            len(setups)
            == len(desired_context)
            == len(desired_features)
            == len(candidates)
            == length
        ):
            raise ValueError("all inverse-v5 request arrays must have equal length")
        contexts = np.asarray(
            [
                inverse_context(
                    setups[index],
                    state_mapping(current[index]),
                    state_mapping(desired_context[index]),
                )
                for index in range(length)
            ],
            dtype=np.float32,
        )

        score_parts = []
        probability_parts = []
        status_probability_parts = []
        cost_parts = []
        for start in range(0, length, batch_size):
            stop = min(start + batch_size, length)
            candidate, cost, status = inverse_candidate_features(
                candidates[start:stop],
                desired_features[start:stop],
            )
            repeated_context = np.repeat(
                contexts[start:stop, None, :],
                len(ACTION_GRID),
                axis=1,
            )
            candidate_input = np.concatenate(
                [repeated_context, candidate],
                axis=-1,
            ).reshape(-1, contexts.shape[1] + candidate.shape[-1])
            probability = self.candidate_model.predict_proba(
                candidate_input
            )[:, 1].reshape(stop - start, len(ACTION_GRID))
            cost_correction = self.cost_correction_model.predict(
                candidate_input
            ).reshape(stop - start, len(ACTION_GRID))
            clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
            logit = np.log(clipped) - np.log1p(-clipped)
            status_input = np.concatenate(
                [contexts[start:stop], status],
                axis=1,
            )
            status_probability = self.status_model.predict_proba(status_input)
            estimated_cost = np.maximum(
                cost + self.cost_correction_beta * cost_correction,
                0.0,
            )
            score_parts.append(-estimated_cost + self.alpha * logit)
            probability_parts.append(probability)
            status_probability_parts.append(status_probability)
            cost_parts.append(cost)

        scores = np.concatenate(score_parts)
        candidate_probability = np.concatenate(probability_parts)
        status_probability = np.concatenate(status_probability_parts)
        costs = np.concatenate(cost_parts)
        selected = (scores - 1e-7 * MOVEMENT[None, :]).argmax(axis=1)
        status_indices = status_probability.argmax(axis=1)
        return {
            "selected_indices": selected,
            "selected_actions": [ACTION_GRID[int(index)] for index in selected],
            "predicted_status_indices": status_indices,
            "predicted_statuses": [
                self.statuses[int(index)] for index in status_indices
            ],
            "scores": scores,
            "base_costs": costs,
            "candidate_success_probabilities": candidate_probability,
            "cost_correction_beta": self.cost_correction_beta,
            "status_probabilities": status_probability,
            "status_logits": np.log(np.clip(status_probability, 1e-8, 1.0)),
        }


def load_inverse_runtime_v5(
    artifact_path: Path,
    torch: Any | None = None,
    device: Any | None = None,
) -> tuple[InverseTreeRuntimeV5, dict[str, Any]]:
    del torch, device
    with artifact_path.open("rb") as stream:
        artifact = pickle.load(stream)
    return InverseTreeRuntimeV5(artifact), artifact
