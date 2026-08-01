#!/usr/bin/env python3
"""Compare frozen v3 and candidate v4 on the same difficult validation rows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import (
    ACTION_GRID,
    group_arrays,
    inverse_pair_arrays,
    read_jsonl,
    residual_cost,
)
from control_rebuild_v3.evaluate_controlled import selection_metrics
from control_rebuild_v3.forward_runtime import load_forward_runtime
from control_rebuild_v3.inverse_runtime import load_inverse_runtime
from control_rebuild_v3.train_forward import configure, forward_metrics
from control_rebuild_v3.train_inverse import inverse_metrics
from control_rebuild_v4.forward_runtime import load_forward_runtime_v4
from control_rebuild_v4.inverse_data import derived_inverse_pairs
from control_rebuild_v4.inverse_runtime import load_inverse_runtime_v4
from specialist_rebuild_v2.common import STATE_FIELDS, raw_state_array

DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/control_rebuild_v4_numerical"
DEFAULT_V3_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v3_one_seed"
DEFAULT_V4_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"
METRIC_PATHS = {
    "forward_strict_all_five": ("forward", "strict_all_five_success"),
    "forward_only_target_success_feasible": (
        "forward_cost_only",
        "target_success_feasible",
    ),
    "inverse_target_success_feasible": (
        "inverse",
        "target_success_feasible",
    ),
    "inverse_status_accuracy": ("inverse", "status_accuracy"),
}
FORWARD_EXAMPLE_METRIC = "strict_all_five_forward_prediction"
FORWARD_EXAMPLE_DEFINITION = (
    "success means the absolute prediction error is at most one configured "
    "tolerance unit for every one of the five beam-state outputs"
)
INVERSE_EXAMPLE_METRIC = "physical_target_success_feasible"
INVERSE_EXAMPLE_DEFINITION = (
    "for a reachable request, success means the selected action is one of the "
    "ground-truth actions that places all five beam-state outputs within tolerance"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--v3-run", type=Path, default=DEFAULT_V3_RUN)
    parser.add_argument("--v4-run", type=Path, default=DEFAULT_V4_RUN)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expected-groups",
        type=int,
        default=450,
        help="Test-only override for a truncated smoke validation set.",
    )
    return parser.parse_args()


def value_at(row: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = row
    for key in path:
        value = value[key]
    return float(value)


def state_mapping(values: np.ndarray) -> dict[str, float]:
    return {field: float(values[index]) for index, field in enumerate(STATE_FIELDS)}


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    forward: Any,
    inverse: Any,
    include_details: bool = False,
) -> dict[str, Any]:
    _, current, tolerance, target_change, group_ids = group_arrays(rows)
    predicted_change = forward.predict_changes(rows)
    predicted_states = forward.predict_states(rows)
    forward_error = np.abs(predicted_change - target_change)
    forward_success = np.all(forward_error <= 1.0, axis=-1)
    pairs = derived_inverse_pairs(rows)
    grid_map = {str(row["group_id"]): row for row in rows}
    position = {group_id: index for index, group_id in enumerate(group_ids)}
    group_index, _, desired, positives, statuses = inverse_pair_arrays(
        pairs,
        grid_map,
        position,
    )
    selected_truth = np.asarray(
        [
            -1 if row["selected_index"] is None else int(row["selected_index"])
            for row in pairs
        ],
        dtype=np.int64,
    )
    setups = [grid_map[str(row["group_id"])]["setup"] for row in pairs]
    current = np.asarray(
        [
            raw_state_array(grid_map[str(row["group_id"])]["current_beam_state"])
            for row in pairs
        ],
        dtype=np.float32,
    )
    candidate = predicted_states[group_index]
    inverse_result = inverse.score_requests(
        setups,
        current,
        desired,
        candidate,
    )
    selected = np.asarray(inverse_result["selected_indices"], dtype=np.int64)
    inverse_success = positives[np.arange(len(positives)), selected]
    result = {
        "group_count": len(rows),
        "transition_count": len(rows) * 81,
        "pair_count": len(pairs),
        "reachable_pair_count": int(positives.any(axis=1).sum()),
        "forward": forward_metrics(target_change, predicted_change),
        "forward_cost_only": selection_metrics(
            -residual_cost(candidate, desired[:, None, :]),
            positives,
            selected_truth,
        ),
        "inverse": inverse_metrics(
            inverse_result["scores"],
            inverse_result["status_logits"],
            positives,
            statuses,
            selected_truth,
        ),
    }
    if include_details:
        result["_details"] = {
            "rows": list(rows),
            "group_ids": list(group_ids),
            "current": current,
            "tolerance": tolerance,
            "target_change": target_change,
            "predicted_change": predicted_change,
            "forward_error": forward_error,
            "forward_success": forward_success,
            "pairs": pairs,
            "group_index": group_index,
            "desired": desired,
            "positives": positives,
            "statuses": statuses,
            "selected_truth": selected_truth,
            "selected": selected,
            "predicted_statuses": list(inverse_result["predicted_statuses"]),
            "candidate_states": candidate,
            "inverse_success": inverse_success,
        }
    return result


def first_index(mask: np.ndarray) -> int | None:
    values = np.flatnonzero(mask)
    return None if not len(values) else int(values[0])


def forward_example(
    index: int | None,
    v3: Mapping[str, Any],
    v4: Mapping[str, Any],
) -> dict[str, Any] | None:
    if index is None:
        return None
    group_index, action_index = divmod(index, len(ACTION_GRID))
    row = v3["rows"][group_index]
    tolerance = v3["tolerance"][group_index]
    target = v3["target_change"][group_index, action_index]
    v3_prediction = v3["predicted_change"][group_index, action_index]
    v4_prediction = v4["predicted_change"][group_index, action_index]
    return {
        "metric": FORWARD_EXAMPLE_METRIC,
        "metric_definition": FORWARD_EXAMPLE_DEFINITION,
        "group_id": str(row["group_id"]),
        "source_category": str(row["source_category"]),
        "action_index": action_index,
        "action": ACTION_GRID[action_index],
        "target_change": state_mapping(target * tolerance),
        "v3_predicted_change": state_mapping(v3_prediction * tolerance),
        "v4_predicted_change": state_mapping(v4_prediction * tolerance),
        "v3_absolute_error_in_tolerance_units": state_mapping(
            v3["forward_error"][group_index, action_index]
        ),
        "v4_absolute_error_in_tolerance_units": state_mapping(
            v4["forward_error"][group_index, action_index]
        ),
        "v3_strict_all_five_success": bool(
            v3["forward_success"][group_index, action_index]
        ),
        "v4_strict_all_five_success": bool(
            v4["forward_success"][group_index, action_index]
        ),
    }


def inverse_example(
    index: int | None,
    v3: Mapping[str, Any],
    v4: Mapping[str, Any],
) -> dict[str, Any] | None:
    if index is None:
        return None
    pair = v3["pairs"][index]
    v3_selected = int(v3["selected"][index])
    v4_selected = int(v4["selected"][index])
    return {
        "metric": INVERSE_EXAMPLE_METRIC,
        "metric_definition": INVERSE_EXAMPLE_DEFINITION,
        "request_id": str(pair["request_id"]),
        "group_id": str(pair["group_id"]),
        "source_category": str(
            v3["rows"][int(v3["group_index"][index])]["source_category"]
        ),
        "desired_beam_state": state_mapping(v3["desired"][index]),
        "true_status": str(pair["status"]),
        "matching_action_indices": np.flatnonzero(v3["positives"][index]).tolist(),
        "v3_selected_action_index": v3_selected,
        "v3_selected_action": ACTION_GRID[v3_selected],
        "v3_predicted_status": str(v3["predicted_statuses"][index]),
        "v3_predicted_selected_state": state_mapping(
            v3["candidate_states"][index, v3_selected]
        ),
        "v3_physical_target_success": bool(v3["inverse_success"][index]),
        "v4_selected_action_index": v4_selected,
        "v4_selected_action": ACTION_GRID[v4_selected],
        "v4_predicted_status": str(v4["predicted_statuses"][index]),
        "v4_predicted_selected_state": state_mapping(
            v4["candidate_states"][index, v4_selected]
        ),
        "v4_physical_target_success": bool(v4["inverse_success"][index]),
    }


def comparison_examples(
    v3: Mapping[str, Any],
    v4: Mapping[str, Any],
) -> dict[str, Any]:
    forward_v3 = np.asarray(v3["forward_success"], dtype=np.bool_).reshape(-1)
    forward_v4 = np.asarray(v4["forward_success"], dtype=np.bool_).reshape(-1)
    inverse_feasible = np.asarray(v3["positives"], dtype=np.bool_).any(axis=1)
    inverse_v3 = np.asarray(v3["inverse_success"], dtype=np.bool_)
    inverse_v4 = np.asarray(v4["inverse_success"], dtype=np.bool_)
    forward_categories = np.repeat(
        [str(row["source_category"]) for row in v3["rows"]],
        len(ACTION_GRID),
    )
    inverse_categories = np.asarray(
        [
            str(v3["rows"][int(position)]["source_category"])
            for position in v3["group_index"]
        ]
    )

    def block(
        forward_mask: np.ndarray | None = None,
        inverse_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        forward_eligible = (
            np.ones_like(forward_v3, dtype=np.bool_)
            if forward_mask is None
            else forward_mask
        )
        inverse_eligible = inverse_feasible & (
            np.ones_like(inverse_v3, dtype=np.bool_)
            if inverse_mask is None
            else inverse_mask
        )
        return {
            "forward": {
                "v4_improvement": forward_example(
                    first_index(~forward_v3 & forward_v4 & forward_eligible),
                    v3,
                    v4,
                ),
                "v4_regression": forward_example(
                    first_index(forward_v3 & ~forward_v4 & forward_eligible),
                    v3,
                    v4,
                ),
            },
            "inverse": {
                "v4_improvement": inverse_example(
                    first_index(~inverse_v3 & inverse_v4 & inverse_eligible),
                    v3,
                    v4,
                ),
                "v4_regression": inverse_example(
                    first_index(inverse_v3 & ~inverse_v4 & inverse_eligible),
                    v3,
                    v4,
                ),
            },
        }

    categories = sorted(set(forward_categories.tolist()))
    return {
        "definitions": {
            "v4_improvement": "v3 failed the metric and v4 passed it",
            "v4_regression": "v3 passed the metric and v4 failed it",
            FORWARD_EXAMPLE_METRIC: FORWARD_EXAMPLE_DEFINITION,
            INVERSE_EXAMPLE_METRIC: INVERSE_EXAMPLE_DEFINITION,
        },
        "all": block(),
        "by_category": {
            category: block(
                forward_categories == category,
                inverse_categories == category,
            )
            for category in categories
        },
    }


def compare_scope(
    v3: Mapping[str, Any],
    v4: Mapping[str, Any],
) -> dict[str, Any]:
    output = {}
    for name, path in METRIC_PATHS.items():
        baseline = value_at(v3, path)
        candidate = value_at(v4, path)
        output[name] = {
            "v3": baseline,
            "v4": candidate,
            "absolute_delta": candidate - baseline,
        }
    return output


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    v3_run = args.v3_run.resolve()
    v4_run = args.v4_run.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else v4_run / "v3_v4_selection_validation_comparison.json"
    )
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if bool(existing.get("complete")):
            raise RuntimeError(f"refusing to overwrite completed comparison: {output}")

    rows = read_jsonl(data_dir / "grids/val.jsonl")
    if len(rows) != int(args.expected_groups):
        raise ValueError(
            f"expected {args.expected_groups} v4 validation groups, got {len(rows)}"
        )
    category_counts = Counter(str(row["source_category"]) for row in rows)
    torch, device = configure(20260726, args.device)
    v3_forward, _ = load_forward_runtime(
        v3_run / "forward_control_v3_calibrated.pt",
        torch,
        device,
    )
    v3_inverse, _ = load_inverse_runtime(
        v3_run / "inverse_control_v3.pt",
        torch,
        device,
    )
    v4_forward, _ = load_forward_runtime_v4(
        v4_run / "forward_physics_residual_v4.pt",
        torch,
        device,
    )
    v4_inverse, _ = load_inverse_runtime_v4(
        v4_run / "inverse_control_v4.pt",
        torch,
        device,
    )
    started = time.perf_counter()
    models: dict[str, dict[str, Any]] = {"v3": {}, "v4": {}}
    model_details = {}
    for name, forward, inverse in (
        ("v3", v3_forward, v3_inverse),
        ("v4", v4_forward, v4_inverse),
    ):
        all_result = evaluate_rows(
            rows,
            forward,
            inverse,
            include_details=True,
        )
        model_details[name] = all_result.pop("_details")
        models[name]["all"] = all_result
        models[name]["by_category"] = {
            category: evaluate_rows(
                [row for row in rows if str(row["source_category"]) == category],
                forward,
                inverse,
            )
            for category in sorted(category_counts)
        }
    comparison = {
        "all": compare_scope(models["v3"]["all"], models["v4"]["all"]),
        "by_category": {
            category: compare_scope(
                models["v3"]["by_category"][category],
                models["v4"]["by_category"][category],
            )
            for category in sorted(category_counts)
        },
    }
    examples = comparison_examples(model_details["v3"], model_details["v4"])
    result = {
        "comparison_version": "v3_v4_same_failure_focused_selection_validation",
        "scope": (
            f"same {len(rows)} v4 checkpoint-selection validation groups; "
            "not held-out evaluation"
        ),
        "seed": 20260726,
        "device": str(device),
        "category_counts": dict(sorted(category_counts.items())),
        "artifacts": {
            "v3_forward": str((v3_run / "forward_control_v3_calibrated.pt").resolve()),
            "v3_inverse": str((v3_run / "inverse_control_v3.pt").resolve()),
            "v4_forward": str((v4_run / "forward_physics_residual_v4.pt").resolve()),
            "v4_inverse": str((v4_run / "inverse_control_v4.pt").resolve()),
        },
        "models": models,
        "comparison": comparison,
        "examples": examples,
        "held_out_used_for_training_or_selection": 0,
        "complete": True,
        "seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
