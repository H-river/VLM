"""Runtime adapter for the frozen v8 Set-Transformer inverse ranker."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID, MOVEMENT
from control_rebuild_v4.inverse_data import state_mapping
from specialist_rebuild_v2.common import STATUSES, inverse_context
from tabm_transformer_rebuild_v8.inverse_features import (
    inverse_features_torch,
)
from tabm_transformer_rebuild_v8.models import build_inverse_model


class InverseSetTransformerRuntimeV8:
    """Rank all 81 predicted actions with the frozen v8 set model."""

    def __init__(
        self,
        torch: Any,
        artifact: Mapping[str, Any],
        device: Any,
    ) -> None:
        if artifact.get("model") != "transformer_numerical_inverse_v8":
            raise ValueError("expected a transformer numerical inverse-v8 artifact")
        if artifact.get("architecture") != "transformer":
            raise ValueError("inverse-v8 artifact architecture differs")
        self.torch = torch
        self.device = device
        self.artifact = artifact
        self.context_mean = np.asarray(
            artifact["context_mean"],
            dtype=np.float32,
        )
        self.context_scale = np.asarray(
            artifact["context_scale"],
            dtype=np.float32,
        )
        self.candidate_mean = torch.as_tensor(
            artifact["candidate_mean"],
            dtype=torch.float32,
            device=device,
        )
        self.candidate_scale = torch.as_tensor(
            artifact["candidate_scale"],
            dtype=torch.float32,
            device=device,
        )
        self.status_mean = torch.as_tensor(
            artifact["status_mean"],
            dtype=torch.float32,
            device=device,
        )
        self.status_scale = torch.as_tensor(
            artifact["status_scale"],
            dtype=torch.float32,
            device=device,
        )
        self.correction_weight = float(artifact["correction_weight"])
        self.model = build_inverse_model(
            torch,
            "transformer",
            context_dim=len(self.context_mean),
            candidate_dim=len(self.candidate_mean),
            status_dim=len(self.status_mean),
            candidate_count=len(ACTION_GRID),
            config=dict(artifact["architecture_config"]),
        ).to(device)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

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
        candidates_out = np.asarray(candidate_states, dtype=np.float32)
        if (
            contexts_out.ndim != 2
            or contexts_out.shape[1] != len(self.context_mean)
            or desired_out.shape != (len(contexts_out), 5)
            or candidates_out.shape
            != (len(contexts_out), len(ACTION_GRID), 5)
        ):
            raise ValueError("inverse-v8 feature-array shapes differ")

        score_parts = []
        cost_parts = []
        status_probability_parts = []
        self.model.eval()
        with self.torch.inference_mode():
            for start in range(0, len(contexts_out), batch_size):
                stop = min(start + batch_size, len(contexts_out))
                context = self.torch.as_tensor(
                    (
                        contexts_out[start:stop] - self.context_mean
                    )
                    / self.context_scale,
                    dtype=self.torch.float32,
                    device=self.device,
                )
                states = self.torch.as_tensor(
                    candidates_out[start:stop],
                    dtype=self.torch.float32,
                    device=self.device,
                )
                desired_values = self.torch.as_tensor(
                    desired_out[start:stop],
                    dtype=self.torch.float32,
                    device=self.device,
                )
                candidate, cost, status_features = inverse_features_torch(
                    self.torch,
                    states,
                    desired_values,
                )
                candidate = (
                    candidate - self.candidate_mean
                ) / self.candidate_scale
                status_features = (
                    status_features - self.status_mean
                ) / self.status_scale
                correction, status_logits = self.model(
                    context,
                    candidate,
                    status_features,
                )
                scores = (
                    -cost.float()
                    + self.correction_weight * correction.float()
                )
                score_parts.append(scores.cpu().numpy())
                cost_parts.append(cost.float().cpu().numpy())
                status_probability_parts.append(
                    status_logits.float().softmax(dim=-1).cpu().numpy()
                )
        scores = np.concatenate(score_parts)
        costs = np.concatenate(cost_parts)
        status_probability = np.concatenate(status_probability_parts)
        selected = (scores - 1e-7 * MOVEMENT[None, :]).argmax(axis=1)
        status_indices = status_probability.argmax(axis=1)
        return {
            "selected_indices": selected,
            "selected_actions": [
                ACTION_GRID[int(index)] for index in selected
            ],
            "predicted_status_indices": status_indices,
            "predicted_statuses": [
                STATUSES[int(index)] for index in status_indices
            ],
            "scores": scores,
            "base_costs": costs,
            "correction_weight": self.correction_weight,
            "status_probabilities": status_probability,
            "status_logits": np.log(
                np.clip(status_probability, 1e-8, 1.0)
            ),
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
        current = np.asarray(context_current, dtype=np.float32)
        desired_context = np.asarray(context_desired, dtype=np.float32)
        desired_features = (
            desired_context
            if feature_desired is None
            else np.asarray(feature_desired, dtype=np.float32)
        )
        candidates = np.asarray(candidate_states, dtype=np.float32)
        if not (
            len(setups)
            == len(current)
            == len(desired_context)
            == len(desired_features)
            == len(candidates)
        ):
            raise ValueError("all inverse-v8 request arrays must have equal length")
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


def load_inverse_runtime_v8(
    artifact_path: Path,
    torch: Any,
    device: Any,
) -> tuple[InverseSetTransformerRuntimeV8, dict[str, Any]]:
    artifact = torch.load(
        artifact_path.resolve(),
        map_location="cpu",
        weights_only=False,
    )
    return InverseSetTransformerRuntimeV8(torch, artifact, device), artifact
