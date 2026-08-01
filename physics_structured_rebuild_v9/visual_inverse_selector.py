"""Feature extraction and runtime selection for complementary visual inverse experts."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import (
    ACTION_GRID,
    ACTION_NORMALIZED,
    MOVEMENT,
    residual_components,
)
from control_rebuild_v3.forward_runtime import load_forward_runtime
from control_rebuild_v3.inverse_runtime import state_mapping
from control_rebuild_v4.forward_runtime import load_forward_runtime_v4
from specialist_rebuild_v2.common import (
    inverse_context,
    setup_array,
)


def _selected_rows(values: np.ndarray, selected: np.ndarray) -> np.ndarray:
    rows = np.arange(len(selected), dtype=np.int64)
    return np.asarray(values)[rows, selected]


def _expert_features(
    result: Mapping[str, Any],
    desired_sensor: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(result["selected_indices"], dtype=np.int64)
    scores = np.asarray(result["scores"], dtype=np.float32)
    costs = np.asarray(result["base_costs"], dtype=np.float32)
    corrections = np.asarray(result["learned_corrections"], dtype=np.float32)
    candidates = np.asarray(
        result["candidate_sensor_states"],
        dtype=np.float32,
    )
    chosen_states = _selected_rows(candidates, selected)
    chosen_scores = _selected_rows(scores, selected)[:, None]
    chosen_costs = _selected_rows(costs, selected)[:, None]
    chosen_corrections = _selected_rows(corrections, selected)[:, None]
    sorted_scores = np.sort(scores, axis=1)
    score_summary = np.stack(
        [
            sorted_scores[:, -1],
            sorted_scores[:, -1] - sorted_scores[:, -2],
            scores.mean(axis=1),
            scores.std(axis=1),
            scores.max(axis=1) - scores.min(axis=1),
            costs.min(axis=1),
            costs.mean(axis=1),
            corrections.max(axis=1),
            corrections.std(axis=1),
        ],
        axis=1,
    )
    signed = residual_components(
        chosen_states,
        np.asarray(desired_sensor, dtype=np.float32),
    )
    return np.concatenate(
        [
            ACTION_NORMALIZED[selected],
            MOVEMENT[selected, None],
            chosen_states,
            signed,
            np.abs(signed),
            chosen_scores,
            chosen_costs,
            chosen_corrections,
            score_summary,
        ],
        axis=1,
    ).astype(np.float32)


def visual_selector_features(
    setups: Sequence[Mapping[str, Any]],
    current_sensor: np.ndarray,
    desired_sensor: np.ndarray,
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
) -> np.ndarray:
    """Describe the request and both experts without using private truth."""

    current = np.asarray(current_sensor, dtype=np.float32)
    desired = np.asarray(desired_sensor, dtype=np.float32)
    current_base = np.asarray(
        primary["current_base_legacy"],
        dtype=np.float32,
    )
    desired_base = np.asarray(
        primary["desired_base_legacy"],
        dtype=np.float32,
    )
    contexts = np.asarray(
        [
            inverse_context(
                setup,
                state_mapping(current_base[index]),
                state_mapping(desired_base[index]),
            )
            for index, setup in enumerate(setups)
        ],
        dtype=np.float32,
    )
    setup_values = np.asarray(
        [setup_array(setup) for setup in setups],
        dtype=np.float32,
    )
    sensor_delta = desired - current
    sensor_derived = np.stack(
        [
            np.linalg.norm(sensor_delta[:, :2], axis=1),
            np.linalg.norm(sensor_delta[:, 2:4], axis=1),
            sensor_delta[:, 4]
            / np.maximum(np.abs(current[:, 4]), np.float32(1e-6)),
        ],
        axis=1,
    )
    primary_features = _expert_features(primary, desired)
    secondary_features = _expert_features(secondary, desired)
    primary_selected = np.asarray(
        primary["selected_indices"],
        dtype=np.int64,
    )
    secondary_selected = np.asarray(
        secondary["selected_indices"],
        dtype=np.int64,
    )
    primary_states = _selected_rows(
        np.asarray(primary["candidate_sensor_states"], dtype=np.float32),
        primary_selected,
    )
    secondary_states = _selected_rows(
        np.asarray(secondary["candidate_sensor_states"], dtype=np.float32),
        secondary_selected,
    )
    cross = np.concatenate(
        [
            (primary_selected == secondary_selected)[:, None],
            ACTION_NORMALIZED[secondary_selected]
            - ACTION_NORMALIZED[primary_selected],
            secondary_states - primary_states,
            np.abs(secondary_states - primary_states),
            np.abs(secondary_features - primary_features),
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate(
        [
            setup_values,
            contexts,
            current,
            desired,
            sensor_delta,
            sensor_derived,
            primary_features,
            secondary_features,
            cross,
        ],
        axis=1,
    ).astype(np.float32)


def load_forward_artifact(
    path: Path,
    torch: Any,
    device: Any,
) -> Any:
    metadata = torch.load(path, map_location="cpu", weights_only=False)
    if metadata.get("model") == "anchored_physics_residual_forward_v4":
        return load_forward_runtime_v4(path, torch, device)[0]
    return load_forward_runtime(path, torch, device)[0]


class VisualInverseSelectorPipelineV9:
    """Select between the retained natural and original v4 visual pipelines."""

    def __init__(
        self,
        base: Any,
        artifact_path: Path,
        torch: Any,
        device: Any,
    ) -> None:
        self.base = base
        with artifact_path.resolve().open("rb") as stream:
            artifact = pickle.load(stream)
        if artifact.get("model") != "hgb_visual_inverse_expert_selector_v9":
            raise ValueError("unexpected visual inverse selector artifact")
        self.artifact = artifact
        self.classifier = artifact["classifier"]
        self.threshold = float(artifact["threshold"])
        self.primary_forward = base.forward
        self.secondary_forward = load_forward_artifact(
            Path(str(artifact["secondary_forward"])).resolve(),
            torch,
            device,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def predict_from_states(
        self,
        setups: Sequence[Mapping[str, Any]],
        current_sensor: np.ndarray,
        desired_sensor: np.ndarray,
        group_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        self.base.forward = self.primary_forward
        primary = self.base.predict_from_states(
            setups,
            current_sensor,
            desired_sensor,
            group_ids=group_ids,
        )
        try:
            self.base.forward = self.secondary_forward
            secondary = self.base.predict_from_states(
                setups,
                current_sensor,
                desired_sensor,
                group_ids=group_ids,
            )
        finally:
            self.base.forward = self.primary_forward
        features = visual_selector_features(
            setups,
            np.asarray(current_sensor, dtype=np.float32),
            np.asarray(desired_sensor, dtype=np.float32),
            primary,
            secondary,
        )
        probability = self.classifier.predict_proba(features)[:, 1]
        choose_secondary = probability >= self.threshold
        if len(choose_secondary) != 1:
            raise ValueError(
                "visual inverse selector currently accepts one request at a time"
            )
        result = secondary if bool(choose_secondary[0]) else primary
        result["visual_selector_probability"] = probability
        result["visual_selector_chose_secondary"] = choose_secondary
        result["decision_source"] = "visual_inverse_expert_selector_v9"
        return result
