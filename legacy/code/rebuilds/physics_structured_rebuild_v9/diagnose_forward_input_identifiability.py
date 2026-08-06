#!/usr/bin/env python3
"""Audit whether nearly identical forward inputs have inconsistent outcomes.

The audit uses training grids only.  A group is one optical setup and current
five-value beam state with the same fixed grid of 81 legal actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import (
    ACTION_GRID,
    ACTION_VALUES,
    tolerance_from_current,
)
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    SETUP_FIELDS,
    STATE_FIELDS,
)

DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
DEFAULT_OUTPUT = DEFAULT_RUN / "forward_input_identifiability_audit_v9.json"
DEFAULT_NATURAL = DEFAULT_RUN / "clean_nonoverlap_forward_training_features.npz"
DEFAULT_GRIDS = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/train.jsonl",
    REPO_ROOT.parent / "VLM_data/control_rebuild_v4_quickcheck/grids/train.jsonl",
    REPO_ROOT.parent / "VLM_data/control_rebuild_v5_numerical/grids/train.jsonl",
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/targeted_bundle/grids/train.jsonl",
)

TIERS = {
    "near_duplicate": {
        "setup_max_fraction": 0.05,
        "setup_rms_fraction": 0.05,
        "current_max_tolerance_units": 0.10,
    },
    "strict_similar": {
        "setup_max_fraction": 0.10,
        "setup_rms_fraction": 0.05,
        "current_max_tolerance_units": 0.25,
    },
    "exploratory_relaxed": {
        "setup_max_fraction": 0.35,
        "setup_rms_fraction": 0.15,
        "current_max_tolerance_units": 0.50,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, action="append", dest="grids")
    parser.add_argument("--natural-cache", type=Path, default=DEFAULT_NATURAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--example-count", type=int, default=12)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_complexity(index: int) -> int:
    return int(
        sum(
            abs(float(ACTION_GRID[index][field])) > 0.0
            for field in ACTION_FIELDS
        )
    )


def load_json_grids(
    paths: list[Path],
) -> tuple[
    list[str],
    list[str],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    group_ids: list[str] = []
    sources: list[str] = []
    setups: list[np.ndarray] = []
    currents: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    for raw_path in paths:
        path = raw_path.resolve()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                current = np.asarray(
                    [
                        float(row["current_beam_state"][field])
                        for field in STATE_FIELDS
                    ],
                    dtype=np.float64,
                )
                candidates = row["candidates"]
                if len(candidates) != len(ACTION_GRID):
                    raise ValueError(
                        f"{row['group_id']}: expected {len(ACTION_GRID)} actions"
                    )
                candidate_states = []
                for index, candidate in enumerate(candidates):
                    actual_action = np.asarray(
                        [
                            float(candidate["action"][field])
                            for field in ACTION_FIELDS
                        ],
                        dtype=np.float64,
                    )
                    if not np.allclose(
                        actual_action,
                        ACTION_VALUES[index],
                        rtol=0.0,
                        atol=1e-9,
                    ):
                        raise ValueError(
                            f"{row['group_id']}: action order differs at {index}"
                        )
                    if "next_state" in candidate:
                        state = [
                            float(candidate["next_state"][field])
                            for field in STATE_FIELDS
                        ]
                    else:
                        state = [
                            current[field_index]
                            + float(candidate["change"][field])
                            for field_index, field in enumerate(STATE_FIELDS)
                        ]
                    candidate_states.append(state)
                group_ids.append(str(row["group_id"]))
                sources.append(str(path))
                setups.append(
                    np.asarray(
                        [float(row["setup"][field]) for field in SETUP_FIELDS],
                        dtype=np.float64,
                    )
                )
                currents.append(current)
                next_states.append(
                    np.asarray(candidate_states, dtype=np.float64)
                )
    return group_ids, sources, setups, currents, next_states


def load_natural_cache(
    path: Path,
) -> tuple[
    list[str],
    list[str],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    resolved = path.resolve()
    with np.load(resolved, allow_pickle=False) as cache:
        features = np.asarray(cache["grid_features"], dtype=np.float64)
        targets = np.asarray(
            cache["grid_target_normalized"], dtype=np.float64
        )
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
    if len(features) != len(group_ids) * len(ACTION_GRID):
        raise ValueError("natural cache must contain 81 actions per group")
    grouped_features = features.reshape(len(group_ids), len(ACTION_GRID), -1)
    contexts = grouped_features[:, 40, :17]
    setups = contexts[:, : len(SETUP_FIELDS)].copy()
    currents = contexts[:, len(SETUP_FIELDS) :].copy()
    currents[:, -1] = np.expm1(currents[:, -1])
    tolerances = np.stack(
        [tolerance_from_current(current) for current in currents]
    ).astype(np.float64)
    changes = targets.reshape(len(group_ids), len(ACTION_GRID), len(STATE_FIELDS))
    changes *= tolerances[:, None, :]
    states = currents[:, None, :] + changes
    return (
        list(map(str, group_ids)),
        [str(resolved)] * len(group_ids),
        list(setups),
        list(currents),
        list(states),
    )


def pair_metrics(
    left: np.ndarray,
    right: np.ndarray,
    left_current: np.ndarray,
    right_current: np.ndarray,
    setup_scale: np.ndarray,
    left_tolerance: np.ndarray,
    right_tolerance: np.ndarray,
) -> dict[str, Any]:
    setup_difference = np.abs(left - right) / setup_scale
    pair_tolerance = np.maximum(left_tolerance, right_tolerance)
    state_difference = np.abs(left_current - right_current) / pair_tolerance
    return {
        "setup_max_fraction": float(setup_difference.max()),
        "setup_rms_fraction": float(
            np.sqrt(np.mean(np.square(setup_difference)))
        ),
        "current_max_tolerance_units": float(state_difference.max()),
        "current_rms_tolerance_units": float(
            np.sqrt(np.mean(np.square(state_difference)))
        ),
        "setup_difference_fraction_by_field": {
            field: float(setup_difference[index])
            for index, field in enumerate(SETUP_FIELDS)
        },
        "current_difference_tolerance_units_by_field": {
            field: float(state_difference[index])
            for index, field in enumerate(STATE_FIELDS)
        },
    }


def qualifies(metrics: dict[str, Any], tier: dict[str, float]) -> bool:
    return bool(
        metrics["setup_max_fraction"] <= tier["setup_max_fraction"]
        and metrics["setup_rms_fraction"] <= tier["setup_rms_fraction"]
        and metrics["current_max_tolerance_units"]
        <= tier["current_max_tolerance_units"]
    )


def scaled_radius_coordinates(
    setups: np.ndarray,
    currents: np.ndarray,
    setup_scale: np.ndarray,
) -> np.ndarray:
    """Coordinates whose infinity-radius covers the relaxed tier.

    Peak is logarithmic because its tolerance is relative.  A 10% margin is
    used in the radius query, followed by exact filtering.
    """

    relaxed = TIERS["exploratory_relaxed"]
    state_coordinates = np.column_stack(
        [
            currents[:, 0] / 1.0,
            currents[:, 1] / 1.0,
            currents[:, 2] / 2.0,
            currents[:, 3] / 2.0,
            np.log(np.maximum(currents[:, 4], 1e-12)) / 0.05,
        ]
    )
    return np.concatenate(
        [
            setups
            / setup_scale[None, :]
            / relaxed["setup_max_fraction"],
            state_coordinates / relaxed["current_max_tolerance_units"],
        ],
        axis=1,
    )


def pair_detail(
    left: int,
    right: int,
    group_ids: list[str],
    sources: list[str],
    setups: np.ndarray,
    currents: np.ndarray,
    tolerances: np.ndarray,
    next_states: np.ndarray,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    pair_tolerance = np.maximum(tolerances[left], tolerances[right])
    output_difference = np.abs(next_states[left] - next_states[right])
    normalized_output = output_difference / pair_tolerance[None, :]
    changes_left = next_states[left] - currents[left][None, :]
    changes_right = next_states[right] - currents[right][None, :]
    response_difference = np.abs(changes_left - changes_right)
    normalized_response = response_difference / pair_tolerance[None, :]
    violation = normalized_output > 1.0
    response_violation = normalized_response > 1.0
    worst_by_action = normalized_output.max(axis=1)
    worst_index = int(np.argmax(worst_by_action))
    return {
        "left_group_id": group_ids[left],
        "right_group_id": group_ids[right],
        "left_source": sources[left],
        "right_source": sources[right],
        "similarity": metrics,
        "left_setup": {
            field: float(setups[left, index])
            for index, field in enumerate(SETUP_FIELDS)
        },
        "right_setup": {
            field: float(setups[right, index])
            for index, field in enumerate(SETUP_FIELDS)
        },
        "left_current_state": {
            field: float(currents[left, index])
            for index, field in enumerate(STATE_FIELDS)
        },
        "right_current_state": {
            field: float(currents[right, index])
            for index, field in enumerate(STATE_FIELDS)
        },
        "action_comparison_count": int(len(ACTION_GRID)),
        "output_violation_action_count": int(np.any(violation, axis=1).sum()),
        "response_violation_action_count": int(
            np.any(response_violation, axis=1).sum()
        ),
        "output_violation_count_by_field": {
            field: int(violation[:, index].sum())
            for index, field in enumerate(STATE_FIELDS)
        },
        "maximum_output_difference_tolerance_units": float(
            normalized_output.max()
        ),
        "maximum_response_difference_tolerance_units": float(
            normalized_response.max()
        ),
        "worst_action": {
            "index": worst_index,
            "action": ACTION_GRID[worst_index],
            "action_complexity": action_complexity(worst_index),
            "left_resulting_state": {
                field: float(next_states[left, worst_index, field_index])
                for field_index, field in enumerate(STATE_FIELDS)
            },
            "right_resulting_state": {
                field: float(next_states[right, worst_index, field_index])
                for field_index, field in enumerate(STATE_FIELDS)
            },
            "difference_tolerance_units_by_field": {
                field: float(normalized_output[worst_index, field_index])
                for field_index, field in enumerate(STATE_FIELDS)
            },
            "response_difference_tolerance_units_by_field": {
                field: float(normalized_response[worst_index, field_index])
                for field_index, field in enumerate(STATE_FIELDS)
            },
        },
    }


def aggregate_tier(
    pairs: list[tuple[int, int]],
    details: list[dict[str, Any]],
) -> dict[str, Any]:
    if not pairs:
        return {
            "pair_count": 0,
            "action_comparison_count": 0,
            "pair_with_any_output_violation_count": 0,
            "output_violation_action_count": 0,
            "output_violation_action_rate": None,
            "response_violation_action_count": 0,
            "maximum_output_difference_tolerance_units": None,
            "examples": [],
        }
    action_count = len(pairs) * len(ACTION_GRID)
    output_violations = sum(
        int(detail["output_violation_action_count"]) for detail in details
    )
    response_violations = sum(
        int(detail["response_violation_action_count"]) for detail in details
    )
    ordered = sorted(
        details,
        key=lambda item: item["maximum_output_difference_tolerance_units"],
        reverse=True,
    )
    return {
        "pair_count": int(len(pairs)),
        "action_comparison_count": int(action_count),
        "pair_with_any_output_violation_count": int(
            sum(detail["output_violation_action_count"] > 0 for detail in details)
        ),
        "output_violation_action_count": int(output_violations),
        "output_violation_action_rate": float(output_violations / action_count),
        "response_violation_action_count": int(response_violations),
        "maximum_output_difference_tolerance_units": float(
            ordered[0]["maximum_output_difference_tolerance_units"]
        ),
        "examples": ordered,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    started = time.perf_counter()
    grid_paths = [path.resolve() for path in (args.grids or DEFAULT_GRIDS)]
    loaded = load_json_grids(grid_paths)
    natural = load_natural_cache(args.natural_cache)
    group_ids = loaded[0] + natural[0]
    sources = loaded[1] + natural[1]
    setups = np.stack(loaded[2] + natural[2]).astype(np.float64)
    currents = np.stack(loaded[3] + natural[3]).astype(np.float64)
    next_states = np.stack(loaded[4] + natural[4]).astype(np.float64)
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("group identifiers overlap")
    combined = np.concatenate([setups, currents], axis=1)
    exact_context_count = len(combined) - len(np.unique(combined, axis=0))
    setup_low = np.quantile(setups, 0.05, axis=0)
    setup_high = np.quantile(setups, 0.95, axis=0)
    setup_scale = np.maximum(setup_high - setup_low, 1e-12)
    tolerances = np.stack(
        [tolerance_from_current(current) for current in currents]
    ).astype(np.float64)

    coordinates = scaled_radius_coordinates(setups, currents, setup_scale)
    raw_pairs = cKDTree(coordinates).query_pairs(
        1.10,
        p=np.inf,
        output_type="ndarray",
    )
    if raw_pairs.size == 0:
        raw_pairs = np.empty((0, 2), dtype=np.int64)
    metrics_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for left, right in raw_pairs:
        key = (int(left), int(right))
        metrics_by_pair[key] = pair_metrics(
            setups[left],
            setups[right],
            currents[left],
            currents[right],
            setup_scale,
            tolerances[left],
            tolerances[right],
        )

    tier_reports: dict[str, Any] = {}
    for name, threshold in TIERS.items():
        selected_pairs = [
            pair
            for pair, metrics in metrics_by_pair.items()
            if qualifies(metrics, threshold)
        ]
        details = [
            pair_detail(
                left,
                right,
                group_ids,
                sources,
                setups,
                currents,
                tolerances,
                next_states,
                metrics_by_pair[(left, right)],
            )
            for left, right in selected_pairs
        ]
        report = aggregate_tier(selected_pairs, details)
        report["criteria"] = threshold
        report["examples"] = report["examples"][: int(args.example_count)]
        tier_reports[name] = report

    report = {
        "version": "forward_input_identifiability_audit_v9",
        "group_count": int(len(group_ids)),
        "action_count_per_group": int(len(ACTION_GRID)),
        "transition_count": int(len(group_ids) * len(ACTION_GRID)),
        "exact_duplicate_context_pair_count": int(exact_context_count),
        "definitions": {
            "setup_scale": (
                "per-field 95th percentile minus 5th percentile across all "
                "training groups"
            ),
            "setup_difference": (
                "absolute setup difference divided by setup_scale"
            ),
            "current_state_tolerance": {
                "centroid_x_px": 1.0,
                "centroid_y_px": 1.0,
                "sigma_x_px": 2.0,
                "sigma_y_px": 2.0,
                "peak_intensity": "5% of current peak, minimum 1e-6",
            },
            "pair_tolerance": (
                "larger of the two groups' current-state tolerances per field"
            ),
            "output_violation": (
                "same-action resulting states differ by more than one pair "
                "tolerance in at least one field"
            ),
            "response_violation": (
                "same-action induced changes differ by more than one pair "
                "tolerance in at least one field"
            ),
        },
        "setup_robust_span_by_field": {
            field: float(setup_scale[index])
            for index, field in enumerate(SETUP_FIELDS)
        },
        "candidate_radius_pair_count": int(len(raw_pairs)),
        "tiers": tier_reports,
        "conclusion_flags": {
            "strict_identifiability_test_has_pairs": bool(
                tier_reports["strict_similar"]["pair_count"] > 0
            ),
            "near_duplicate_test_has_pairs": bool(
                tier_reports["near_duplicate"]["pair_count"] > 0
            ),
            "exploratory_pairs_are_not_near_duplicates": True,
        },
        "source_contract": {
            "training_grids": [
                {"path": str(path), "sha256": sha256(path)}
                for path in grid_paths
            ],
            "natural_training_cache": {
                "path": str(args.natural_cache.resolve()),
                "sha256": sha256(args.natural_cache.resolve()),
            },
            "validation_files_opened": [],
            "held_out_test_files_opened": [],
            "generated_setups": 0,
            "generated_images": 0,
        },
        "seconds": float(time.perf_counter() - started),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
