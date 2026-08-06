"""Shared numerical contracts for forward and inverse control v3."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    SETUP_FIELDS,
    STATE_FIELDS,
    action_array,
    fixed_action_grid,
    inverse_context,
    raw_state_array,
    setup_array,
    state_array,
)


STATUS_NAMES = ("unique", "ambiguous", "infeasible_within_limits")
STATUS_INDEX = {name: index for index, name in enumerate(STATUS_NAMES)}
ACTION_GRID = fixed_action_grid()
ACTION_VALUES = np.asarray(
    [[float(action[field]) for field in ACTION_FIELDS] for action in ACTION_GRID],
    dtype=np.float32,
)
ACTION_NORMALIZED = ACTION_VALUES / np.asarray(
    [0.05, 0.05, 0.02, 0.02], dtype=np.float32
)
MOVEMENT = np.abs(ACTION_NORMALIZED).sum(axis=1).astype(np.float32)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )


def tolerance_from_current(current: Mapping[str, Any] | np.ndarray) -> np.ndarray:
    values = (
        raw_state_array(current)
        if isinstance(current, Mapping)
        else np.asarray(current, dtype=np.float32)
    )
    return np.asarray(
        [1.0, 1.0, 2.0, 2.0, max(0.05 * abs(float(values[4])), 1e-6)],
        dtype=np.float32,
    )


def context_vector(
    setup: Mapping[str, Any], current: Mapping[str, Any]
) -> np.ndarray:
    setup_values = setup_array(setup)
    current_values = state_array(current)
    pitch_mm = max(float(setup["pixel_size_um"]) / 1000.0, 1e-9)
    focal = max(abs(float(setup["lens_focal_length_mm"])), 1e-9)
    waist = max(abs(float(setup["beam_waist_mm"])), 1e-9)
    sx = max(abs(float(current["sigma_x_px"])), 1e-9)
    sy = max(abs(float(current["sigma_y_px"])), 1e-9)
    derived = np.asarray(
        [
            float(setup["lens_to_camera_mm"]) / focal,
            float(setup["source_to_lens_mm"]) / focal,
            float(setup["lens_to_camera_mm"]) / focal / pitch_mm,
            float(setup["lens_x_offset_mm"]) / pitch_mm,
            float(setup["lens_y_offset_mm"]) / pitch_mm,
            float(setup["camera_x_offset_mm"]) / pitch_mm,
            float(setup["camera_y_offset_mm"]) / pitch_mm,
            math.hypot(
                float(setup["lens_x_offset_mm"]),
                float(setup["lens_y_offset_mm"]),
            )
            / waist,
            math.hypot(
                float(setup["camera_x_offset_mm"]),
                float(setup["camera_y_offset_mm"]),
            )
            / waist,
            sx / sy,
            math.hypot(
                float(current["centroid_x_px"]) - 511.5,
                float(current["centroid_y_px"]) - 511.5,
            ),
        ],
        dtype=np.float32,
    )
    return np.concatenate([setup_values, current_values, derived])


def action_basis(actions: np.ndarray = ACTION_NORMALIZED) -> np.ndarray:
    """Return a zero-anchored interaction basis for four ternary actuators."""

    values = np.asarray(actions, dtype=np.float32)
    columns: list[np.ndarray] = []
    columns.extend(values[:, index] for index in range(4))
    squared = np.square(values)
    columns.extend(squared[:, index] for index in range(4))
    for left in range(4):
        for right in range(left + 1, 4):
            columns.append(values[:, left] * values[:, right])
    for signed in range(4):
        for gate in range(4):
            if signed != gate:
                columns.append(values[:, signed] * squared[:, gate])
    for left in range(4):
        for right in range(left + 1, 4):
            columns.append(squared[:, left] * squared[:, right])
    columns.extend(
        [
            values[:, 0] * values[:, 1] * values[:, 2],
            values[:, 0] * values[:, 1] * values[:, 3],
            values[:, 0] * values[:, 2] * values[:, 3],
            values[:, 1] * values[:, 2] * values[:, 3],
            values[:, 0] * values[:, 1] * values[:, 2] * values[:, 3],
        ]
    )
    result = np.stack(columns, axis=1).astype(np.float32)
    zero_index = next(
        index
        for index, action in enumerate(ACTION_GRID)
        if all(float(action[field]) == 0.0 for field in ACTION_FIELDS)
    )
    if not np.all(result[zero_index] == 0.0):
        raise AssertionError("zero action basis must be exactly zero")
    return result


def group_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    contexts, currents, tolerances, changes, group_ids = [], [], [], [], []
    for row in rows:
        current = raw_state_array(row["current_beam_state"])
        tolerance = tolerance_from_current(current)
        candidates = np.asarray(
            [
                [float(candidate["change"][field]) for field in STATE_FIELDS]
                for candidate in row["candidates"]
            ],
            dtype=np.float32,
        )
        if len(candidates) != len(ACTION_GRID):
            raise ValueError(f"{row['group_id']}: expected 81 candidates")
        contexts.append(context_vector(row["setup"], row["current_beam_state"]))
        currents.append(current)
        tolerances.append(tolerance)
        changes.append(candidates / tolerance[None, :])
        group_ids.append(str(row["group_id"]))
    return (
        np.asarray(contexts, dtype=np.float32),
        np.asarray(currents, dtype=np.float32),
        np.asarray(tolerances, dtype=np.float32),
        np.asarray(changes, dtype=np.float32),
        group_ids,
    )


def residual_components(
    states: np.ndarray, desired: np.ndarray
) -> np.ndarray:
    states_out = np.asarray(states, dtype=np.float32)
    desired_out = np.asarray(desired, dtype=np.float32)
    peak_scale = np.maximum(np.abs(desired_out[..., 4:5]) * 0.02, 1e-6)
    return np.concatenate(
        [
            (states_out[..., 0:2] - desired_out[..., 0:2]) / 0.5,
            states_out[..., 2:4] - desired_out[..., 2:4],
            (states_out[..., 4:5] - desired_out[..., 4:5]) / peak_scale,
        ],
        axis=-1,
    ).astype(np.float32)


def residual_cost(states: np.ndarray, desired: np.ndarray) -> np.ndarray:
    components = residual_components(states, desired)
    return np.sqrt(np.mean(np.square(components), axis=-1)).astype(np.float32)


def inverse_candidate_features(
    states: np.ndarray, desired: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build candidate and request-level features from forward predictions.

    ``states`` has shape [requests, 81, 5] and ``desired`` has shape
    [requests, 5]. Residual components are dimensionless: one unit is one
    declared inverse-matching tolerance.
    """

    states_out = np.asarray(states, dtype=np.float32)
    desired_out = np.asarray(desired, dtype=np.float32)
    if states_out.ndim == 2:
        states_out = states_out[None, ...]
    if desired_out.ndim == 1:
        desired_out = desired_out[None, ...]
    if states_out.shape[0] != desired_out.shape[0]:
        raise ValueError("states and desired must contain the same requests")
    signed = residual_components(states_out, desired_out[:, None, :])
    absolute = np.abs(signed)
    squared = np.square(signed)
    cost = np.sqrt(np.mean(squared, axis=-1)).astype(np.float32)
    centroid = np.sqrt(np.square(signed[..., 0]) + np.square(signed[..., 1]))
    action = np.broadcast_to(
        ACTION_NORMALIZED[None, :, :],
        (len(states_out), len(ACTION_GRID), 4),
    )
    movement = np.broadcast_to(
        MOVEMENT[None, :, None],
        (len(states_out), len(ACTION_GRID), 1),
    )
    candidate = np.concatenate(
        [
            action,
            signed,
            absolute,
            squared,
            movement,
            cost[..., None],
            absolute.max(axis=-1, keepdims=True),
            centroid[..., None],
        ],
        axis=-1,
    ).astype(np.float32)

    ordered = np.sort(cost, axis=1)
    predicted_match = (
        (centroid <= 1.0)
        & (absolute[..., 2] <= 1.0)
        & (absolute[..., 3] <= 1.0)
        & (absolute[..., 4] <= 1.0)
    )
    status_features = np.column_stack(
        [
            ordered[:, 0],
            ordered[:, 1],
            ordered[:, 2],
            ordered[:, 1] - ordered[:, 0],
            ordered[:, 2] - ordered[:, 0],
            predicted_match.sum(axis=1) / len(ACTION_GRID),
            (cost <= 1.0).sum(axis=1) / len(ACTION_GRID),
            centroid.min(axis=1),
            absolute[..., 2].min(axis=1),
            absolute[..., 3].min(axis=1),
            absolute[..., 4].min(axis=1),
            cost.mean(axis=1),
        ]
    ).astype(np.float32)
    return candidate, cost, status_features


def select_minimum_cost(costs: np.ndarray) -> np.ndarray:
    costs_out = np.asarray(costs)
    if costs_out.ndim == 1:
        return np.asarray(
            min(
                range(len(costs_out)),
                key=lambda index: (
                    float(costs_out[index]),
                    float(MOVEMENT[index]),
                    index,
                ),
            )
        )
    selected = [
        min(
            range(costs_out.shape[1]),
            key=lambda index: (
                float(row[index]),
                float(MOVEMENT[index]),
                index,
            ),
        )
        for row in costs_out
    ]
    return np.asarray(selected, dtype=np.int64)


def inverse_pair_arrays(
    pairs: Sequence[Mapping[str, Any]],
    grids: Mapping[str, Mapping[str, Any]],
    group_position: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group_indices, contexts, desired, positives, statuses = [], [], [], [], []
    for pair in pairs:
        group_id = str(pair["group_id"])
        grid = grids[group_id]
        group_indices.append(int(group_position[group_id]))
        contexts.append(
            inverse_context(
                grid["setup"],
                grid["current_beam_state"],
                pair["desired_beam_state"],
            )
        )
        desired.append(raw_state_array(pair["desired_beam_state"]))
        mask = np.zeros(len(ACTION_GRID), dtype=np.bool_)
        mask[np.asarray(pair["matching_indices"], dtype=np.int64)] = True
        positives.append(mask)
        statuses.append(STATUS_INDEX[str(pair["status"])])
    return (
        np.asarray(group_indices, dtype=np.int64),
        np.asarray(contexts, dtype=np.float32),
        np.asarray(desired, dtype=np.float32),
        np.asarray(positives, dtype=np.bool_),
        np.asarray(statuses, dtype=np.int64),
    )
