"""Runtime for the validation-calibrated forward-control ensemble."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import (
    ACTION_GRID,
    context_vector,
    tolerance_from_current,
)
from control_rebuild_v3.models import joint_forward_model
from specialist_rebuild_v2.common import forward_feature, raw_state_array
from specialist_rebuild_v2.models import forward_model


class ForwardControlRuntime:
    """Load both frozen components once and predict all 81 actions jointly."""

    def __init__(
        self,
        torch: Any,
        artifact: Mapping[str, Any],
        device: Any,
    ) -> None:
        if artifact.get("model") != "blended_forward_control_v3":
            raise ValueError("expected a blended_forward_control_v3 artifact")
        self.torch = torch
        self.device = device
        self.alpha = float(artifact["blend_alpha"])
        self.legacy = artifact["legacy_component"]
        self.joint = artifact["joint_component"]

        self.legacy_model = forward_model(
            torch, int(self.legacy["input_dim"])
        ).to(device)
        self.legacy_model.load_state_dict(self.legacy["state_dict"])
        self.legacy_model.eval()

        self.joint_model = joint_forward_model(
            torch,
            int(self.joint["context_dim"]),
            int(self.joint["basis_dim"]),
        ).to(device)
        self.joint_model.load_state_dict(self.joint["state_dict"])
        self.joint_model.eval()
        self.joint_basis = torch.as_tensor(
            np.asarray(self.joint["action_basis"], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )

    def _legacy_changes(
        self,
        rows: Sequence[Mapping[str, Any]],
        transition_batch: int,
    ) -> np.ndarray:
        features = np.asarray(
            [
                forward_feature(
                    row["setup"], row["current_beam_state"], action
                )
                for row in rows
                for action in ACTION_GRID
            ],
            dtype=np.float32,
        )
        mean = np.asarray(self.legacy["mean"], dtype=np.float32)
        scale = np.asarray(self.legacy["scale"], dtype=np.float32)
        normalized = (features - mean) / scale
        coefficient = np.asarray(
            self.legacy["baseline_coef"], dtype=np.float32
        )
        intercept = np.asarray(
            self.legacy["baseline_intercept"], dtype=np.float32
        )
        baseline = normalized @ coefficient.T + intercept
        residual_parts = []
        with self.torch.inference_mode():
            for start in range(0, len(normalized), transition_batch):
                values = self.torch.as_tensor(
                    normalized[start : start + transition_batch],
                    dtype=self.torch.float32,
                    device=self.device,
                )
                residual, _, _ = self.legacy_model(values)
                residual_parts.append(residual.float().cpu().numpy())
        return (baseline + np.concatenate(residual_parts)).reshape(
            len(rows), len(ACTION_GRID), 5
        )

    def _joint_changes(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int,
    ) -> np.ndarray:
        context = np.asarray(
            [
                context_vector(row["setup"], row["current_beam_state"])
                for row in rows
            ],
            dtype=np.float32,
        )
        mean = np.asarray(self.joint["context_mean"], dtype=np.float32)
        scale = np.asarray(self.joint["context_scale"], dtype=np.float32)
        normalized = (context - mean) / scale
        parts = []
        with self.torch.inference_mode():
            for start in range(0, len(normalized), group_batch):
                values = self.torch.as_tensor(
                    normalized[start : start + group_batch],
                    dtype=self.torch.float32,
                    device=self.device,
                )
                parts.append(
                    self.joint_model(values, self.joint_basis)
                    .float()
                    .cpu()
                    .numpy()
                )
        return np.concatenate(parts)

    def predict_changes(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 256,
        transition_batch: int = 8192,
    ) -> np.ndarray:
        legacy = self._legacy_changes(rows, transition_batch)
        if self.alpha == 0.0:
            return legacy.astype(np.float32)
        joint = self._joint_changes(rows, group_batch)
        return (
            (1.0 - self.alpha) * legacy + self.alpha * joint
        ).astype(np.float32)

    def predict_states(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 256,
        transition_batch: int = 8192,
    ) -> np.ndarray:
        changes = self.predict_changes(
            rows,
            group_batch=group_batch,
            transition_batch=transition_batch,
        )
        current = np.asarray(
            [raw_state_array(row["current_beam_state"]) for row in rows],
            dtype=np.float32,
        )
        tolerance = np.asarray(
            [
                tolerance_from_current(row["current_beam_state"])
                for row in rows
            ],
            dtype=np.float32,
        )
        return current[:, None, :] + changes * tolerance[:, None, :]


def load_forward_runtime(
    artifact_path: Path, torch: Any, device: Any
) -> tuple[ForwardControlRuntime, dict[str, Any]]:
    artifact = torch.load(
        artifact_path, map_location="cpu", weights_only=False
    )
    return ForwardControlRuntime(torch, artifact, device), artifact
