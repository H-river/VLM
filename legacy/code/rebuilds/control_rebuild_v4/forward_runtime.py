"""Runtime for the zero-anchored physics-residual forward model v4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID, tolerance_from_current
from control_rebuild_v4.models import anchored_forward_residual_model
from specialist_rebuild_v2.common import forward_feature, raw_state_array

ZERO_ACTION_INDEX = next(
    index
    for index, action in enumerate(ACTION_GRID)
    if all(float(value) == 0.0 for value in action.values())
)


class ForwardControlRuntimeV4:
    """Predict normalized five-state changes for all 81 allowed actions.

    A normalized change of one means one task tolerance.  The saved linear
    baseline and neural residual are both evaluated relative to the zero-action
    feature vector, so the zero action returns exactly five zeros.
    """

    def __init__(
        self,
        torch: Any,
        artifact: Mapping[str, Any],
        device: Any,
    ) -> None:
        if artifact.get("model") != "anchored_physics_residual_forward_v4":
            raise ValueError(
                "expected an anchored_physics_residual_forward_v4 artifact"
            )
        self.torch = torch
        self.device = device
        self.mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
        self.scale = np.asarray(artifact["feature_scale"], dtype=np.float32)
        self.coefficient = np.asarray(artifact["ridge_coefficient"], dtype=np.float32)
        self.residual_alpha = float(artifact.get("residual_alpha", 1.0))
        self.model = anchored_forward_residual_model(
            torch, int(artifact["input_dim"])
        ).to(device)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

    def _normalized_features(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        features = np.asarray(
            [
                [
                    forward_feature(row["setup"], row["current_beam_state"], action)
                    for action in ACTION_GRID
                ]
                for row in rows
            ],
            dtype=np.float32,
        )
        return ((features - self.mean) / self.scale).astype(np.float32)

    def predict_changes(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 256,
    ) -> np.ndarray:
        if not rows:
            return np.empty((0, len(ACTION_GRID), 5), dtype=np.float32)
        features = self._normalized_features(rows)
        zero = features[:, ZERO_ACTION_INDEX, :].copy()
        baseline = np.einsum(
            "gaf,kf->gak",
            features - zero[:, None, :],
            self.coefficient,
        ).astype(np.float32)
        residual_parts = []
        with self.torch.inference_mode():
            for start in range(0, len(features), group_batch):
                stop = min(start + group_batch, len(features))
                residual_parts.append(
                    self.model(
                        self.torch.as_tensor(
                            features[start:stop],
                            dtype=self.torch.float32,
                            device=self.device,
                        ),
                        self.torch.as_tensor(
                            zero[start:stop],
                            dtype=self.torch.float32,
                            device=self.device,
                        ),
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
        changes = baseline + self.residual_alpha * np.concatenate(residual_parts)
        changes[:, ZERO_ACTION_INDEX, :] = 0.0
        return changes.astype(np.float32)

    def predict_states(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 256,
    ) -> np.ndarray:
        changes = self.predict_changes(rows, group_batch=group_batch)
        current = np.asarray(
            [raw_state_array(row["current_beam_state"]) for row in rows],
            dtype=np.float32,
        )
        tolerance = np.asarray(
            [tolerance_from_current(row["current_beam_state"]) for row in rows],
            dtype=np.float32,
        )
        return current[:, None, :] + changes * tolerance[:, None, :]


def load_forward_runtime_v4(
    artifact_path: Path, torch: Any, device: Any
) -> tuple[ForwardControlRuntimeV4, dict[str, Any]]:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    return ForwardControlRuntimeV4(torch, artifact, device), artifact
