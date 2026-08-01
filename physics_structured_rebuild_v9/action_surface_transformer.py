"""Explicit 81-action Transformer forward surface and runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import tolerance_from_current
from physics_structured_rebuild_v9.forward_runtime import (
    load_residual_forward_runtime_v9,
)
from specialist_rebuild_v2.common import (
    STATE_FIELDS,
    fixed_action_grid,
    forward_feature,
)


def build_action_surface_transformer(
    torch: Any,
    config: Mapping[str, Any],
) -> Any:
    """Build a model with one context token and 81 explicit action tokens."""

    class ActionSurfaceTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimension = int(config["dimension"])
            self.context_encoder = torch.nn.Sequential(
                torch.nn.Linear(int(config["context_dim"]), dimension),
                torch.nn.LayerNorm(dimension),
                torch.nn.GELU(),
                torch.nn.Linear(dimension, dimension),
            )
            self.action_encoder = torch.nn.Sequential(
                torch.nn.Linear(int(config["action_dim"]), dimension),
                torch.nn.LayerNorm(dimension),
                torch.nn.GELU(),
                torch.nn.Linear(dimension, dimension),
            )
            self.action_identity = torch.nn.Parameter(
                torch.empty(1, int(config["action_count"]), dimension)
            )
            self.context_identity = torch.nn.Parameter(
                torch.empty(1, 1, dimension)
            )
            layer = torch.nn.TransformerEncoderLayer(
                d_model=dimension,
                nhead=int(config["heads"]),
                dim_feedforward=int(config["feedforward"]),
                dropout=float(config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.interactions = torch.nn.TransformerEncoder(
                layer,
                num_layers=int(config["layers"]),
                norm=torch.nn.LayerNorm(dimension),
            )
            self.output = torch.nn.Sequential(
                torch.nn.Linear(dimension * 2, dimension),
                torch.nn.GELU(),
                torch.nn.Dropout(float(config["dropout"])),
                torch.nn.Linear(dimension, 5),
            )
            torch.nn.init.normal_(self.action_identity, std=0.02)
            torch.nn.init.normal_(self.context_identity, std=0.02)
            torch.nn.init.zeros_(self.output[-1].weight)
            torch.nn.init.zeros_(self.output[-1].bias)

        def forward(self, context: Any, actions: Any) -> Any:
            context_token = (
                self.context_encoder(context)[:, None, :]
                + self.context_identity
            )
            action_tokens = (
                self.action_encoder(actions) + self.action_identity
            )
            encoded = self.interactions(
                torch.cat([context_token, action_tokens], dim=1)
            )
            global_context = encoded[:, :1, :].expand(
                -1, int(config["action_count"]), -1
            )
            return self.output(
                torch.cat([encoded[:, 1:, :], global_context], dim=2)
            )

    return ActionSurfaceTransformer()


class ActionSurfaceTransformerRuntimeV9:
    """Predict a complete fixed action surface using an explicit token model."""

    def __init__(
        self,
        artifact_path: Path,
        torch: Any,
        device: Any,
    ) -> None:
        artifact = torch.load(
            artifact_path.resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if artifact.get("model") != "explicit_81_action_surface_transformer_v9":
            raise ValueError("unexpected action-surface Transformer artifact")
        self.artifact = artifact
        self.torch = torch
        self.device = device
        self.model = build_action_surface_transformer(
            torch, artifact["config"]
        ).to(device)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()
        self.feature_mean = np.asarray(
            artifact["feature_mean"], dtype=np.float32
        )
        self.feature_scale = np.asarray(
            artifact["feature_scale"], dtype=np.float32
        )
        self.residual_scale = np.asarray(
            artifact["residual_scale"], dtype=np.float32
        )
        self.field_blend = np.asarray(
            artifact["field_blend"], dtype=np.float32
        )
        self.base, _ = load_residual_forward_runtime_v9(
            Path(str(artifact["base_forward"])).resolve(),
            torch,
            device,
        )
        self.actions = fixed_action_grid()

    def predict_correction(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        prior = self.base.predict_changes(rows).astype(np.float32)
        features = np.asarray(
            [
                [
                    np.concatenate(
                        [
                            forward_feature(
                                row["setup"],
                                row["current_beam_state"],
                                action,
                            ),
                            prior[row_index, action_index],
                        ]
                    )
                    for action_index, action in enumerate(self.actions)
                ]
                for row_index, row in enumerate(rows)
            ],
            dtype=np.float32,
        )
        standardized = (
            features - self.feature_mean[None, None, :]
        ) / self.feature_scale[None, None, :]
        context = standardized[:, 0, :17]
        action_values = standardized[:, :, 17:]
        corrections = []
        batch_size = int(self.artifact["config"].get("runtime_batch_size", 64))
        with self.torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                raw = self.model(
                    self.torch.as_tensor(
                        context[start : start + batch_size],
                        dtype=self.torch.float32,
                        device=self.device,
                    ),
                    self.torch.as_tensor(
                        action_values[start : start + batch_size],
                        dtype=self.torch.float32,
                        device=self.device,
                    ),
                )
                corrections.append(raw.float().cpu().numpy())
        correction = np.concatenate(corrections, axis=0)
        correction *= self.residual_scale[None, None, :]
        correction[:, 40, :] = 0.0
        return prior, correction.astype(np.float32)

    def predict_changes(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        prior, correction = self.predict_correction(rows)
        return (
            prior
            + correction * self.field_blend[None, None, :]
        ).astype(np.float32)

    def predict_states(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        changes = self.predict_changes(rows)
        current = np.asarray(
            [
                [float(row["current_beam_state"][field]) for field in STATE_FIELDS]
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
        return current[:, None, :] + changes * tolerance[:, None, :]


def load_action_surface_transformer_runtime_v9(
    path: Path,
    torch: Any,
    device: Any,
) -> tuple[ActionSurfaceTransformerRuntimeV9, dict[str, Any]]:
    runtime = ActionSurfaceTransformerRuntimeV9(path, torch, device)
    return runtime, runtime.artifact
