"""V11 legacy-grid contracts, deterministic group selection, and metrics."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID, tolerance_from_current
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    SETUP_FIELDS,
    STATE_FIELDS,
    forward_feature,
)

ZERO_ACTION_INDEX = 40
LEARNING_CURVE_SIZES = (326, 678, 1200, 2400, 5000, 10500)
OVERFIT_GROUP_SIZES = (32, 64)


def stable_seed(*parts: Any) -> int:
    return int(
        hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16],
        16,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_legacy_action_order(row: Mapping[str, Any]) -> None:
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(ACTION_GRID):
        raise ValueError(f"{row.get('group_id')}: expected 81 candidates")
    for index, (candidate, expected) in enumerate(
        zip(candidates, ACTION_GRID, strict=True)
    ):
        if any(
            float(candidate["action"][field]) != float(expected[field])
            for field in ACTION_FIELDS
        ):
            raise ValueError(
                f"{row.get('group_id')}: action order differs at {index}"
            )


def select_group_rows(
    paths: Sequence[Path],
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the globally lowest stable group hashes from allowed sources."""

    if count < 1:
        raise ValueError("group count must be positive")
    heap: list[tuple[int, int, dict[str, Any]]] = []
    serial = 0
    provenance: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        if "physics_structured_rebuild_v10" in resolved.parts:
            raise ValueError("v11 refuses every v10 data path")
        provenance.append(
            {"path": str(resolved), "sha256": sha256_file(resolved)}
        )
        with resolved.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                group_id = str(row["group_id"])
                if group_id in seen_ids:
                    raise ValueError(f"duplicate group ID across sources: {group_id}")
                seen_ids.add(group_id)
                token = stable_seed(seed, resolved, group_id, "v11_select")
                entry = (-token, serial, row)
                serial += 1
                if len(heap) < count:
                    heapq.heappush(heap, entry)
                elif entry[0] > heap[0][0]:
                    heapq.heapreplace(heap, entry)
    if len(heap) != count:
        raise ValueError(f"requested {count} groups but found only {len(heap)}")
    rows = [entry[2] for entry in heap]
    rows.sort(key=lambda row: stable_seed(seed, row["group_id"], "order"))
    for row in rows:
        validate_legacy_action_order(row)
    return rows, provenance


def natural_action_index(row: Mapping[str, Any], seed: int) -> int:
    declared = row.get("natural_requested_action_index")
    if declared is not None:
        index = int(declared)
        if 0 <= index < len(ACTION_GRID):
            return index
    cardinality = np.asarray(
        [
            sum(abs(float(action[field])) > 0 for field in ACTION_FIELDS)
            for action in ACTION_GRID
        ],
        dtype=np.int64,
    )
    desired = 1 + stable_seed(seed, row["group_id"], "cardinality") % 4
    eligible = np.flatnonzero(cardinality == desired)
    return int(
        eligible[stable_seed(seed, row["group_id"], "natural_action") % len(eligible)]
    )


def rows_to_arrays(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """Flatten grouped rows while retaining group and natural-action indices."""

    features: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    action_indices: list[int] = []
    targets: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    tolerances: list[np.ndarray] = []
    group_indices: list[int] = []
    natural_indices: list[int] = []
    regimes: list[str] = []
    group_ids: list[str] = []
    for group_index, row in enumerate(rows):
        current = np.asarray(
            [float(row["current_beam_state"][field]) for field in STATE_FIELDS],
            dtype=np.float32,
        )
        tolerance = tolerance_from_current(current).astype(np.float32)
        context = np.asarray(
            [
                *[float(row["setup"][field]) for field in SETUP_FIELDS],
                *[
                    float(row["current_beam_state"][field])
                    for field in STATE_FIELDS[:-1]
                ],
                np.log1p(max(float(row["current_beam_state"]["peak_intensity"]), 0.0)),
            ],
            dtype=np.float32,
        )
        natural_indices.append(natural_action_index(row, seed))
        regimes.append(
            str(row.get("regime", row.get("source_category", "unspecified")))
        )
        group_ids.append(str(row["group_id"]))
        for action_index, candidate in enumerate(row["candidates"]):
            feature = forward_feature(
                row["setup"],
                row["current_beam_state"],
                candidate["action"],
            ).astype(np.float32)
            change = np.asarray(
                [float(candidate["change"][field]) for field in STATE_FIELDS],
                dtype=np.float32,
            )
            features.append(feature)
            contexts.append(context)
            action_indices.append(action_index)
            raw_targets.append(change)
            targets.append(change / tolerance)
            tolerances.append(tolerance)
            group_indices.append(group_index)
    return {
        "features": np.asarray(features, dtype=np.float32),
        "contexts": np.asarray(contexts, dtype=np.float32),
        "action_indices": np.asarray(action_indices, dtype=np.int64),
        "targets": np.asarray(targets, dtype=np.float32),
        "raw_targets": np.asarray(raw_targets, dtype=np.float32),
        "tolerances": np.asarray(tolerances, dtype=np.float32),
        "group_indices": np.asarray(group_indices, dtype=np.int64),
        "group_ids": group_ids,
        "natural_action_indices": np.asarray(natural_indices, dtype=np.int64),
        "regimes": regimes,
    }


def action_cardinality() -> np.ndarray:
    return np.asarray(
        [
            sum(abs(float(action[field])) > 0 for field in ACTION_FIELDS)
            for action in ACTION_GRID
        ],
        dtype=np.int64,
    )


def forward_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    arrays: Mapping[str, Any],
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = np.abs(prediction - target)
    strict = np.all(error <= 1.0, axis=1)
    group_count = len(arrays["group_ids"])
    surface = strict.reshape(group_count, len(ACTION_GRID))
    natural = np.asarray(arrays["natural_action_indices"], dtype=np.int64)
    natural_success = surface[np.arange(group_count), natural]
    field_names = list(STATE_FIELDS)
    cardinality = np.tile(action_cardinality(), group_count)
    by_cardinality = {
        str(value): {
            "count": int(np.sum(cardinality == value)),
            "strict_all_five_accuracy": float(
                strict[cardinality == value].mean()
            ),
        }
        for value in range(5)
    }
    regimes = np.asarray(arrays["regimes"], dtype=np.str_)
    by_regime = {}
    for regime in sorted(set(regimes.tolist())):
        mask = regimes == regime
        rows = np.flatnonzero(mask)[:, None] * len(ACTION_GRID) + np.arange(
            len(ACTION_GRID)
        )[None, :]
        by_regime[regime] = {
            "groups": int(mask.sum()),
            "full_surface_strict_accuracy": float(strict[rows.reshape(-1)].mean()),
            "natural_requested_action_accuracy": float(natural_success[mask].mean()),
        }
    return {
        "groups": group_count,
        "transitions": int(len(target)),
        "normalized_mae": float(error.mean()),
        "per_output_tolerance_accuracy": {
            field: float((error[:, index] <= 1.0).mean())
            for index, field in enumerate(field_names)
        },
        "strict_all_five_accuracy": float(strict.mean()),
        "natural_requested_action_accuracy": float(natural_success.mean()),
        "full_81_action_surface_accuracy": float(strict.mean()),
        "by_action_cardinality": by_cardinality,
        "by_regime": by_regime,
    }


def split_hash(group_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(map(str, group_ids))).encode()
    return hashlib.sha256(payload).hexdigest()


def coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    regimes = Counter(
        str(row.get("regime", row.get("source_category", "unspecified")))
        for row in rows
    )
    return {"groups": len(rows), "regime_counts": dict(sorted(regimes.items()))}

