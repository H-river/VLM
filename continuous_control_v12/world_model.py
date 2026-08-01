"""Numerical residual world model and calibrated ensemble runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    OUTPUT_FIELDS,
    POSITION_FIELDS,
    SETUP_CONTEXT_FIELDS,
    Bounds,
    action_vector,
    assert_no_q_star,
    metrics_vector,
    position_vector,
    tolerance_vector,
)


def structured_features(
    setup_context: Mapping[str, Any],
    positions_mm: Mapping[str, Any] | Sequence[float],
    current_metrics: Mapping[str, Any] | Sequence[float],
    actions_mm: np.ndarray | Sequence[float],
    bounds: Bounds,
    image_embedding: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Physical values, squares, pairs, and position-action interactions."""

    assert_no_q_star(
        {
            "setup_context": setup_context,
            "positions_mm": positions_mm,
            "current_metrics": current_metrics,
            "actions_mm": np.asarray(actions_mm).tolist(),
        }
    )
    actions = np.asarray(actions_mm, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError("actions must have shape [N, 4]")
    setup = np.asarray(
        [float(setup_context[field]) for field in SETUP_CONTEXT_FIELDS],
        dtype=np.float64,
    )
    positions = position_vector(positions_mm)
    metrics = metrics_vector(current_metrics)
    transformed_metrics = metrics.copy()
    transformed_metrics[4] = np.log1p(max(float(metrics[4]), 0.0))
    action_normalized = actions / bounds.action_high[None, :]
    pair = np.stack(
        [
            action_normalized[:, left] * action_normalized[:, right]
            for left in range(4)
            for right in range(left + 1, 4)
        ],
        axis=1,
    )
    position_scale = np.maximum(
        np.maximum(np.abs(bounds.position_low), np.abs(bounds.position_high)),
        1e-9,
    )
    position_normalized = positions / position_scale
    base = np.concatenate([setup, positions, transformed_metrics])
    repeated = np.broadcast_to(base[None, :], (len(actions), len(base)))
    parts = [
        repeated,
        action_normalized,
        np.square(action_normalized),
        pair,
        action_normalized * position_normalized[None, :],
    ]
    if image_embedding is not None:
        embedding = np.asarray(image_embedding, dtype=np.float32)
        if embedding.ndim != 1:
            raise ValueError("image embedding must be one-dimensional")
        parts.append(
            np.broadcast_to(embedding[None, :], (len(actions), len(embedding)))
        )
    return np.concatenate(
        parts,
        axis=1,
    ).astype(np.float32)


def image_embedding_from_intensity(intensity: np.ndarray) -> np.ndarray:
    """Fixed 4x4 normalized pooling used only by the image-conditioning ablation."""

    values = np.asarray(intensity, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("image conditioning requires one finite 2-D intensity")
    height_edges = np.linspace(0, values.shape[0], 5, dtype=int)
    width_edges = np.linspace(0, values.shape[1], 5, dtype=int)
    peak = max(float(values.max()), 1e-30)
    pooled = [
        float(
            values[
                height_edges[y] : height_edges[y + 1],
                width_edges[x] : width_edges[x + 1],
            ].mean()
            / peak
        )
        for y in range(4)
        for x in range(4)
    ]
    pooled.extend([np.log1p(max(float(values.sum()), 0.0)), np.log(peak)])
    return np.asarray(pooled, dtype=np.float32)


def build_member(torch: Any, input_dim: int, config: Mapping[str, Any]) -> Any:
    hidden = int(config["hidden"])
    dropout = float(config["dropout"])

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.Dropout(dropout),
                torch.nn.LayerNorm(hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, hidden),
            )

        def forward(self, value: Any) -> Any:
            return value + self.layers(value)

    class NumericalForwardMember(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = [
                torch.nn.Linear(input_dim, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.SiLU(),
            ]
            layers.extend(Block() for _ in range(int(config["blocks"])))
            self.trunk = torch.nn.Sequential(*layers)
            self.metric_delta = torch.nn.Linear(hidden, 5)
            self.log_variance = torch.nn.Linear(hidden, 5)
            self.auxiliary = torch.nn.Linear(hidden, 4)

        def forward(
            self,
            features: Any,
            action_nonzero: Any | None = None,
        ) -> tuple[Any, Any, Any]:
            encoded = self.trunk(features)
            nonzero = (
                features.new_ones((len(features), 1))
                if action_nonzero is None
                else action_nonzero.to(features.dtype).reshape(-1, 1)
            )
            return (
                self.metric_delta(encoded) * nonzero,
                self.log_variance(encoded).clamp(-8.0, 4.0),
                self.auxiliary(encoded),
            )

    return NumericalForwardMember()


def transition_arrays(
    rows: Sequence[Mapping[str, Any]],
    bounds: Bounds,
    *,
    data_dir: Path | None = None,
    image_conditioning: bool = False,
) -> dict[str, Any]:
    features = []
    targets = []
    tolerances = []
    auxiliary = []
    auxiliary_mask = []
    group_ids = []
    regimes = []
    sampling = []
    no_op = []
    for row in rows:
        assert_no_q_star(
            {
                "setup_context": row["setup_context"],
                "simulator_fixed": row["simulator_fixed"],
                "positions_mm": row["positions_mm"],
                "metrics": row["metrics"],
                "action_mm": row["action_mm"],
            }
        )
        action = action_vector(row["action_mm"])
        embedding = None
        if image_conditioning:
            reference = row.get("image_ref")
            if data_dir is None or reference is None:
                raise ValueError(
                    "image-conditioning ablation requires stored image references"
                )
            with np.load(data_dir / str(reference), allow_pickle=False) as image:
                embedding = image_embedding_from_intensity(image["intensity"])
        feature = structured_features(
            row["setup_context"],
            row["positions_mm"],
            row["metrics"],
            action,
            bounds,
            image_embedding=embedding,
        )[0]
        current = metrics_vector(row["metrics"])
        next_metrics = metrics_vector(row["next_metrics"])
        tolerance = tolerance_vector(current)
        aux = row["auxiliary"]
        captured = aux.get("captured_power")
        clipping = aux.get("clipping_fraction")
        camera = aux.get("camera_boundary_indicator")
        actuator = aux.get("actuator_limit_indicator")
        values = [
            0.0 if captured is None else np.log(max(float(captured), 1e-30)),
            0.0 if clipping is None else float(clipping),
            0.0 if camera is None else float(bool(camera)),
            0.0 if actuator is None else float(bool(actuator)),
        ]
        masks = [
            captured is not None,
            clipping is not None,
            camera is not None,
            actuator is not None,
        ]
        features.append(feature)
        targets.append((next_metrics - current) / tolerance)
        tolerances.append(tolerance)
        auxiliary.append(values)
        auxiliary_mask.append(masks)
        group_ids.append(str(row["group_id"]))
        regimes.append(str(row.get("regime", "from_manifest")))
        sampling.append(str(row["sampling"]["kind"]))
        no_op.append(bool(np.all(np.abs(action) <= 1e-12)))
    return {
        "features": np.asarray(features, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "tolerances": np.asarray(tolerances, dtype=np.float32),
        "auxiliary": np.asarray(auxiliary, dtype=np.float32),
        "auxiliary_mask": np.asarray(auxiliary_mask, dtype=bool),
        "group_ids": group_ids,
        "regimes": regimes,
        "sampling": sampling,
        "no_op": np.asarray(no_op, dtype=bool),
    }


class ForwardEnsemble:
    """Serialized ensemble exposing F(s, a) residual, auxiliaries, uncertainty."""

    def __init__(self, artifact: Mapping[str, Any], torch: Any, device: Any) -> None:
        if artifact.get("version") != "continuous_forward_model_v12":
            raise ValueError("unexpected v12 forward artifact")
        if artifact.get("q_star_in_inputs") is not False:
            raise ValueError("forward artifact does not attest q* exclusion")
        self.torch = torch
        self.device = device
        self.bounds = Bounds(
            action_low=np.asarray(artifact["bounds"]["action_low"]),
            action_high=np.asarray(artifact["bounds"]["action_high"]),
            position_low=np.asarray(artifact["bounds"]["position_low"]),
            position_high=np.asarray(artifact["bounds"]["position_high"]),
        )
        self.mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
        self.scale = np.asarray(artifact["feature_scale"], dtype=np.float32)
        self.image_conditioning = bool(artifact.get("image_conditioning", False))
        self.members = []
        for state in artifact["member_states"]:
            member = build_member(
                torch,
                int(artifact["input_dim"]),
                artifact["model_config"],
            ).to(device)
            member.load_state_dict(state)
            member.eval()
            self.members.append(member)

    def predict(
        self,
        setup_context: Mapping[str, Any],
        positions_mm: Mapping[str, Any] | Sequence[float],
        current_metrics: Mapping[str, Any] | Sequence[float],
        actions_mm: np.ndarray | Sequence[float],
        current_image: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        actions = np.asarray(actions_mm, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        embedding = None
        if self.image_conditioning:
            if current_image is None:
                raise ValueError(
                    "image-conditioned forward model requires current_image"
                )
            embedding = image_embedding_from_intensity(current_image)
        features = structured_features(
            setup_context,
            positions_mm,
            current_metrics,
            actions_mm,
            self.bounds,
            image_embedding=embedding,
        )
        standardized = (features - self.mean) / self.scale
        tensor = self.torch.as_tensor(
            standardized, dtype=self.torch.float32, device=self.device
        )
        nonzero = self.torch.as_tensor(
            np.any(np.abs(actions) > 1e-12, axis=1),
            dtype=self.torch.float32,
            device=self.device,
        )
        deltas, log_variances, auxiliaries = [], [], []
        with self.torch.inference_mode():
            for member in self.members:
                delta, log_variance, auxiliary = member(tensor, nonzero)
                deltas.append(delta.cpu().numpy())
                log_variances.append(log_variance.cpu().numpy())
                auxiliaries.append(auxiliary.cpu().numpy())
        member_delta = np.stack(deltas, axis=0)
        mean_delta = member_delta.mean(axis=0)
        epistemic = member_delta.var(axis=0)
        aleatoric = np.exp(np.stack(log_variances, axis=0)).mean(axis=0)
        current = metrics_vector(current_metrics)
        tolerance = tolerance_vector(current)
        next_metrics = current[None, :] + mean_delta * tolerance[None, :]
        auxiliary_mean = np.stack(auxiliaries, axis=0).mean(axis=0)
        sigmoid = lambda values: 1.0 / (1.0 + np.exp(-values))
        return {
            "next_metric_residual": mean_delta.astype(np.float32),
            "predicted_next_metrics": next_metrics.astype(np.float64),
            "auxiliary_predictions": {
                "captured_power": np.exp(auxiliary_mean[:, 0]),
                "clipping_fraction": np.clip(auxiliary_mean[:, 1], 0.0, 1.0),
                "camera_boundary_probability": sigmoid(auxiliary_mean[:, 2]),
                "actuator_limit_probability": sigmoid(auxiliary_mean[:, 3]),
            },
            "raw_auxiliary_predictions": auxiliary_mean.astype(np.float32),
            "uncertainty": np.sqrt(epistemic + aleatoric).astype(np.float32),
            "member_metric_residuals": member_delta.astype(np.float32),
        }


def load_forward_ensemble(
    path: Path,
    *,
    device_name: str = "cpu",
) -> ForwardEnsemble:
    import torch

    artifact = torch.load(
        path.resolve(), map_location=device_name, weights_only=False
    )
    return ForwardEnsemble(artifact, torch, torch.device(device_name))
