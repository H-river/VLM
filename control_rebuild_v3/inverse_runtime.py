"""Reusable numerical inverse-control runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import (
    ACTION_GRID,
    MOVEMENT,
    inverse_candidate_features,
)
from control_rebuild_v3.models import inverse_ranker_model
from specialist_rebuild_v2.common import (
    STATE_FIELDS,
    inverse_context,
    raw_state_array,
)


def state_mapping(values: np.ndarray) -> dict[str, float]:
    return {
        field: float(values[index])
        for index, field in enumerate(STATE_FIELDS)
    }


class InverseControlRuntime:
    """Score 81 forward-predicted candidates and classify reachability."""

    def __init__(
        self,
        torch: Any,
        artifact: Mapping[str, Any],
        device: Any,
    ) -> None:
        if artifact.get("model") != "residual_corrected_inverse_control_v3":
            raise ValueError(
                "expected a residual_corrected_inverse_control_v3 artifact"
            )
        self.torch = torch
        self.device = device
        self.artifact = artifact
        self.alpha = float(artifact["correction_alpha"])
        self.statuses = list(artifact["statuses"])
        self.model = inverse_ranker_model(
            torch,
            int(artifact["context_dim"]),
            int(artifact["candidate_dim"]),
            int(artifact["status_dim"]),
        ).to(device)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

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
            raise ValueError("all inverse request arrays must have equal length")
        context = np.asarray(
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
        context = (
            context - np.asarray(self.artifact["context_mean"])
        ) / np.asarray(self.artifact["context_scale"])

        scores, corrections, logits, costs = [], [], [], []
        with self.torch.inference_mode():
            for start in range(0, length, batch_size):
                index = slice(start, start + batch_size)
                candidate, cost, status = inverse_candidate_features(
                    candidates[index], desired_features[index]
                )
                candidate = (
                    candidate - np.asarray(self.artifact["candidate_mean"])
                ) / np.asarray(self.artifact["candidate_scale"])
                status = (
                    status - np.asarray(self.artifact["status_mean"])
                ) / np.asarray(self.artifact["status_scale"])
                correction, status_logits = self.model(
                    self.torch.as_tensor(
                        context[index],
                        dtype=self.torch.float32,
                        device=self.device,
                    ),
                    self.torch.as_tensor(
                        candidate,
                        dtype=self.torch.float32,
                        device=self.device,
                    ),
                    self.torch.as_tensor(
                        status,
                        dtype=self.torch.float32,
                        device=self.device,
                    ),
                )
                correction_np = correction.float().cpu().numpy()
                scores.append(-cost + self.alpha * correction_np)
                corrections.append(correction_np)
                logits.append(status_logits.float().cpu().numpy())
                costs.append(cost)
        score = np.concatenate(scores)
        correction = np.concatenate(corrections)
        status_logits = np.concatenate(logits)
        cost = np.concatenate(costs)
        adjusted = score - 1e-7 * MOVEMENT[None, :]
        selected = adjusted.argmax(axis=1)
        status_index = status_logits.argmax(axis=1)
        return {
            "selected_indices": selected,
            "selected_actions": [ACTION_GRID[int(index)] for index in selected],
            "predicted_status_indices": status_index,
            "predicted_statuses": [
                self.statuses[int(index)] for index in status_index
            ],
            "scores": score,
            "base_costs": cost,
            "learned_corrections": correction,
            "status_logits": status_logits,
        }


def load_inverse_runtime(
    artifact_path: Path, torch: Any, device: Any
) -> tuple[InverseControlRuntime, dict[str, Any]]:
    artifact = torch.load(
        artifact_path, map_location="cpu", weights_only=False
    )
    return InverseControlRuntime(torch, artifact, device), artifact
