"""Runtime for measurement-augmented numerical inverse control v4."""

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
from control_rebuild_v4.inverse_data import state_mapping
from specialist_rebuild_v2.common import inverse_context


class InverseControlRuntimeV4:
    """Score all 81 candidate actions and classify target reachability."""

    def __init__(
        self,
        torch: Any,
        artifact: Mapping[str, Any],
        device: Any,
    ) -> None:
        if artifact.get("model") not in {
            "measurement_augmented_inverse_control_v4",
            "visual_sensor_residual_scorer_v4",
        }:
            raise ValueError(
                "expected a v4 numerical inverse or visual scorer artifact"
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
            context - np.asarray(self.artifact["context_mean"], dtype=np.float32)
        ) / np.asarray(self.artifact["context_scale"], dtype=np.float32)

        score_parts, correction_parts, logit_parts, cost_parts = [], [], [], []
        with self.torch.inference_mode():
            for start in range(0, length, batch_size):
                index = slice(start, start + batch_size)
                candidate, cost, status = inverse_candidate_features(
                    candidates[index], desired_features[index]
                )
                candidate = (
                    candidate
                    - np.asarray(self.artifact["candidate_mean"], dtype=np.float32)
                ) / np.asarray(self.artifact["candidate_scale"], dtype=np.float32)
                status = (
                    status - np.asarray(self.artifact["status_mean"], dtype=np.float32)
                ) / np.asarray(self.artifact["status_scale"], dtype=np.float32)
                correction, logits = self.model(
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
                correction_out = correction.float().cpu().numpy()
                score_parts.append(-cost + self.alpha * correction_out)
                correction_parts.append(correction_out)
                logit_parts.append(logits.float().cpu().numpy())
                cost_parts.append(cost)
        scores = np.concatenate(score_parts)
        corrections = np.concatenate(correction_parts)
        status_logits = np.concatenate(logit_parts)
        costs = np.concatenate(cost_parts)
        selected = (scores - 1e-7 * MOVEMENT[None, :]).argmax(axis=1)
        status_indices = status_logits.argmax(axis=1)
        return {
            "selected_indices": selected,
            "selected_actions": [ACTION_GRID[int(index)] for index in selected],
            "predicted_status_indices": status_indices,
            "predicted_statuses": [
                self.statuses[int(index)] for index in status_indices
            ],
            "scores": scores,
            "base_costs": costs,
            "learned_corrections": corrections,
            "status_logits": status_logits,
        }


def load_inverse_runtime_v4(
    artifact_path: Path, torch: Any, device: Any
) -> tuple[InverseControlRuntimeV4, dict[str, Any]]:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("model") != "measurement_augmented_inverse_control_v4":
        raise ValueError("expected a measurement_augmented_inverse_control_v4 artifact")
    return InverseControlRuntimeV4(torch, artifact, device), artifact


def load_visual_scorer_runtime_v4(
    artifact_path: Path, torch: Any, device: Any
) -> tuple[InverseControlRuntimeV4, dict[str, Any]]:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("model") != "visual_sensor_residual_scorer_v4":
        raise ValueError("expected a visual_sensor_residual_scorer_v4 artifact")
    return InverseControlRuntimeV4(torch, artifact, device), artifact
