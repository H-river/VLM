"""Frozen fields, action representation, labels, and split-hash contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from control_rebuild_v3.common import (
    ACTION_GRID,
    ACTION_NORMALIZED,
    action_basis,
    tolerance_from_current,
)
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    DIRECTION_FIELDS,
    SETUP_FIELDS,
    STATE_FIELDS,
    matching_mask,
    raw_state_array,
)

ZERO_ACTION_INDEX = 40
REGIMES = (
    "ordinary",
    "focusing",
    "clipping",
    "camera_boundary",
    "high_offset_interaction",
    "tolerance_boundary",
)
VISUAL_CONDITIONS = (
    "clean",
    "noise",
    "blur",
    "saturation",
    "asymmetric_gain",
    "crop_boundary",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    return int(
        hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16],
        16,
    )


def setup_hash(setup: Mapping[str, Any]) -> str:
    values = {field: round(float(setup[field]), 12) for field in SETUP_FIELDS}
    return sha256_bytes(canonical_json(values).encode())


def context_hash(
    setup: Mapping[str, Any],
    current: Mapping[str, Any],
) -> str:
    values = {
        "setup": {
            field: round(float(setup[field]), 12) for field in SETUP_FIELDS
        },
        "current": {
            field: round(float(current[field]), 9) for field in STATE_FIELDS
        },
    }
    return sha256_bytes(canonical_json(values).encode())


def action_cardinalities() -> np.ndarray:
    return np.asarray(
        [
            sum(abs(float(action[field])) > 0.0 for field in ACTION_FIELDS)
            for action in ACTION_GRID
        ],
        dtype=np.int64,
    )


def interaction_category(action: Mapping[str, Any]) -> str:
    moved = [
        field for field in ACTION_FIELDS if abs(float(action[field])) > 0.0
    ]
    if not moved:
        return "zero"
    if len(moved) == 1:
        return "single"
    if len(moved) == 2:
        lens_count = sum(field.startswith("lens_") for field in moved)
        return (
            "lens_pair"
            if lens_count == 2
            else "camera_pair"
            if lens_count == 0
            else "cross_pair"
        )
    return "three_way" if len(moved) == 3 else "four_way"


def explicit_action_features() -> np.ndarray:
    """Expose signed motion, masks, cardinality, and interaction structure."""

    values = ACTION_NORMALIZED.astype(np.float32)
    moved = (np.abs(values) > 0).astype(np.float32)
    cardinality = moved.sum(axis=1).astype(np.int64)
    cardinality_one_hot = np.eye(5, dtype=np.float32)[cardinality]
    pair_masks = np.stack(
        [
            moved[:, left] * moved[:, right]
            for left in range(4)
            for right in range(left + 1, 4)
        ],
        axis=1,
    )
    triple_masks = np.stack(
        [
            moved[:, 0] * moved[:, 1] * moved[:, 2],
            moved[:, 0] * moved[:, 1] * moved[:, 3],
            moved[:, 0] * moved[:, 2] * moved[:, 3],
            moved[:, 1] * moved[:, 2] * moved[:, 3],
        ],
        axis=1,
    )
    four_mask = np.prod(moved, axis=1, keepdims=True)
    features = np.concatenate(
        [
            values,
            np.abs(values),
            moved,
            action_basis(values),
            cardinality_one_hot,
            pair_masks,
            triple_masks,
            four_mask,
        ],
        axis=1,
    ).astype(np.float32)
    if not np.all(features[ZERO_ACTION_INDEX, :4] == 0):
        raise AssertionError("canonical zero action moved unexpectedly")
    return features


def visible_context(
    setup: Mapping[str, Any],
    current: Mapping[str, Any],
) -> np.ndarray:
    """The deployable 17-value input, with log1p peak as in v9."""

    state = raw_state_array(current).astype(np.float32)
    state[-1] = np.log1p(max(float(state[-1]), 0.0))
    return np.concatenate(
        [
            np.asarray(
                [float(setup[field]) for field in SETUP_FIELDS],
                dtype=np.float32,
            ),
            state,
        ]
    )


def group_targets(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    current = raw_state_array(row["current_beam_state"]).astype(np.float32)
    tolerance = tolerance_from_current(current).astype(np.float32)
    states = np.asarray(
        [
            [
                float(candidate["next_state"][field])
                for field in STATE_FIELDS
            ]
            for candidate in row["candidates"]
        ],
        dtype=np.float32,
    )
    normalized_change = (states - current[None, :]) / tolerance[None, :]
    return states, normalized_change.astype(np.float32)


def physical_success(
    candidate_states: np.ndarray,
    target: Sequence[float],
) -> np.ndarray:
    return matching_mask(
        np.asarray(candidate_states, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
    )


def direction_labels(normalized_change: np.ndarray) -> np.ndarray:
    values = np.asarray(normalized_change, dtype=np.float32)
    return np.where(values < -1.0, 0, np.where(values > 1.0, 2, 1)).astype(
        np.int64
    )


def natural_requested_action_index(group_id: str, seed: int) -> int:
    """Deterministic natural request, stratified over cardinality."""

    cardinality = action_cardinalities()
    desired_cardinality = (stable_seed(seed, group_id, "cardinality") % 4) + 1
    eligible = np.flatnonzero(cardinality == desired_cardinality)
    return int(
        eligible[stable_seed(seed, group_id, "action") % len(eligible)]
    )


def validate_action_order(candidates: Sequence[Mapping[str, Any]]) -> None:
    if len(candidates) != len(ACTION_GRID):
        raise ValueError(f"expected 81 candidates, received {len(candidates)}")
    for index, (candidate, expected) in enumerate(zip(candidates, ACTION_GRID)):
        action = candidate["action"]
        if any(
            float(action[field]) != float(expected[field])
            for field in ACTION_FIELDS
        ):
            raise ValueError(f"candidate {index} differs from canonical action grid")

