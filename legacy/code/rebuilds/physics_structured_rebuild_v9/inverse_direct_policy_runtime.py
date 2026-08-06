"""Runtime for a direct 81-action inverse policy blended with the v9 ranker."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID, MOVEMENT
from control_rebuild_v4.inverse_data import state_mapping
from physics_structured_rebuild_v9.inverse_lightgbm_runtime import (
    normalized_per_request,
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


def build_direct_policy(torch: Any, input_dimension: int, hidden: int) -> Any:
    """Build the small residual-free policy saved by the training script."""

    return torch.nn.Sequential(
        torch.nn.Linear(input_dimension, hidden),
        torch.nn.LayerNorm(hidden),
        torch.nn.SiLU(),
        torch.nn.Dropout(0.05),
        torch.nn.Linear(hidden, hidden),
        torch.nn.LayerNorm(hidden),
        torch.nn.SiLU(),
        torch.nn.Dropout(0.05),
        torch.nn.Linear(hidden, len(ACTION_GRID)),
    )


class DirectInversePolicyRuntimeV9:
    """Blend context-only action logits with the frozen physics ranker."""

    def __init__(
        self,
        artifact_path: Path,
        torch: Any,
        device: Any,
    ) -> None:
        path = artifact_path.resolve()
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("model") != "direct_inverse_policy_v9":
            raise ValueError("unexpected direct inverse-policy artifact")
        base_path = Path(str(artifact["base_inverse_artifact"])).resolve()
        if sha256(base_path) != artifact["base_inverse_artifact_sha256"]:
            raise ValueError("direct inverse-policy base checksum differs")
        self.base, _ = load_inverse_runtime_v8(base_path, torch, device)
        self.torch = torch
        self.device = device
        self.context_mean = np.asarray(
            artifact["context_mean"],
            dtype=np.float32,
        )
        self.context_scale = np.asarray(
            artifact["context_scale"],
            dtype=np.float32,
        )
        self.policy_weight = float(artifact["policy_weight"])
        self.model = build_direct_policy(
            torch,
            len(self.context_mean),
            int(artifact["hidden_dimension"]),
        ).to(device)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()
        self.version = str(artifact["version"])

    def policy_logits(
        self,
        contexts: np.ndarray,
        *,
        batch_size: int = 1024,
    ) -> np.ndarray:
        contexts_out = np.asarray(contexts, dtype=np.float32)
        if (
            contexts_out.ndim != 2
            or contexts_out.shape[1] != len(self.context_mean)
        ):
            raise ValueError("direct inverse-policy context shape differs")
        parts = []
        self.model.eval()
        with self.torch.inference_mode():
            for start in range(0, len(contexts_out), batch_size):
                values = self.torch.as_tensor(
                    (
                        contexts_out[start : start + batch_size]
                        - self.context_mean
                    )
                    / self.context_scale,
                    dtype=self.torch.float32,
                    device=self.device,
                )
                parts.append(self.model(values).float().cpu().numpy())
        return np.concatenate(parts).astype(np.float32)

    def score_feature_arrays(
        self,
        contexts: np.ndarray,
        desired: np.ndarray,
        candidate_states: np.ndarray,
        *,
        batch_size: int = 128,
    ) -> dict[str, Any]:
        base = self.base.score_feature_arrays(
            contexts,
            desired,
            candidate_states,
            batch_size=batch_size,
        )
        base_scores = np.asarray(base["scores"], dtype=np.float32)
        policy_logits = self.policy_logits(
            contexts,
            batch_size=max(batch_size, 512),
        )
        scores = (
            normalized_per_request(base_scores)
            + self.policy_weight * normalized_per_request(policy_logits)
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
            "policy_logits": policy_logits,
            "policy_weight": self.policy_weight,
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
            candidate_states,
            batch_size=batch_size,
        )


def load_direct_inverse_policy_runtime_v9(
    artifact_path: Path,
    torch: Any,
    device: Any,
) -> tuple[DirectInversePolicyRuntimeV9, dict[str, Any]]:
    path = artifact_path.resolve()
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return DirectInversePolicyRuntimeV9(path, torch, device), artifact
