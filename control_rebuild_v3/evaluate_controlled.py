#!/usr/bin/env python3
"""Run one controlled held-out evaluation of rebuilt control specialists."""

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
    STATUS_INDEX,
    STATUS_NAMES,
    group_arrays,
    inverse_pair_arrays,
    read_json,
    read_jsonl,
    residual_cost,
    select_minimum_cost,
)
from control_rebuild_v3.forward_runtime import load_forward_runtime
from control_rebuild_v3.inverse_runtime import load_inverse_runtime
from control_rebuild_v3.train_forward import (
    configure,
    forward_metrics,
)
from control_rebuild_v3.train_inverse import inverse_metrics
from control_rebuild_v3.visual_inverse import (
    MeasurementModuleV3,
    VisualInversePipeline,
    true_grid_sensor_states,
)
from measurement_rebuild_v3.common import iter_jsonl
from measurement_rebuild_v3.train import metric_block
from specialist_rebuild_v2.common import (
    STATE_FIELDS,
    matching_mask,
    minimum_motion_index,
    raw_state_array,
)


DEFAULT_CONTROL_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_MEASUREMENT_DATA = (
    REPO_ROOT.parent / "VLM_data/measurement_rebuild_v3"
)
DEFAULT_CONTROL_RUN = (
    REPO_ROOT.parent / "VLM_runs/control_rebuild_v3_one_seed"
)
DEFAULT_MEASUREMENT_RUN = (
    REPO_ROOT.parent / "VLM_runs/measurement_rebuild_v3_one_seed"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/control_rebuild_v3_one_seed/controlled_evaluation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-data", type=Path, default=DEFAULT_CONTROL_DATA
    )
    parser.add_argument(
        "--measurement-data", type=Path, default=DEFAULT_MEASUREMENT_DATA
    )
    parser.add_argument(
        "--control-run", type=Path, default=DEFAULT_CONTROL_RUN
    )
    parser.add_argument(
        "--measurement-run", type=Path, default=DEFAULT_MEASUREMENT_RUN
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("val", "test_iid", "test_ood_physics", "test_visual_stress"),
        default=("test_iid", "test_ood_physics", "test_visual_stress"),
    )
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--visual-batch", type=int, default=256)
    return parser.parse_args()


def selection_metrics(
    scores: np.ndarray,
    positives: np.ndarray,
    selected_truth: np.ndarray,
) -> dict[str, Any]:
    selected = select_minimum_cost(-np.asarray(scores))
    feasible = positives.any(axis=1)
    success = positives[np.arange(len(positives)), selected]
    exact = selected == selected_truth
    return {
        "pair_count": int(len(positives)),
        "reachable_pair_count": int(feasible.sum()),
        "target_success_feasible": float(success[feasible].mean()),
        "target_success_all": float(success.mean()),
        "minimum_movement_exact_feasible": float(exact[feasible].mean()),
    }


def numerical_evaluation(
    split: str,
    control_data: Path,
    forward: Any,
    inverse: Any,
) -> dict[str, Any]:
    rows = read_jsonl(control_data / "grids" / f"{split}.jsonl")
    _, _, _, target_change, group_ids = group_arrays(rows)
    predicted_change = forward.predict_changes(rows)
    predicted_states = forward.predict_states(rows)
    forward_block = forward_metrics(target_change, predicted_change)
    output: dict[str, Any] = {
        "group_count": len(rows),
        "forward": forward_block,
    }

    inverse_path = control_data / "inverse" / f"{split}.jsonl"
    if not inverse_path.is_file():
        output["inverse"] = None
        return output
    pairs = read_jsonl(inverse_path)
    grid_map = {str(row["group_id"]): row for row in rows}
    position = {
        group_id: index for index, group_id in enumerate(group_ids)
    }
    group_index, _, desired, positives, statuses = inverse_pair_arrays(
        pairs, grid_map, position
    )
    selected_truth = np.asarray(
        [
            -1
            if row["selected_index"] is None
            else int(row["selected_index"])
            for row in pairs
        ],
        dtype=np.int64,
    )
    setups = [grid_map[str(row["group_id"])]["setup"] for row in pairs]
    current = np.asarray(
        [
            raw_state_array(
                grid_map[str(row["group_id"])]["current_beam_state"]
            )
            for row in pairs
        ],
        dtype=np.float32,
    )
    candidate = predicted_states[group_index]
    result = inverse.score_requests(
        setups, current, desired, candidate
    )
    learned = inverse_metrics(
        result["scores"],
        result["status_logits"],
        positives,
        statuses,
        selected_truth,
    )
    direct = selection_metrics(
        -residual_cost(candidate, desired[:, None, :]),
        positives,
        selected_truth,
    )
    true_candidates = np.asarray(
        [
            [
                raw_state_array(candidate_row["next_state"])
                for candidate_row in grid_map[str(pair["group_id"])][
                    "candidates"
                ]
            ]
            for pair in pairs
        ],
        dtype=np.float32,
    )
    oracle = selection_metrics(
        -residual_cost(true_candidates, desired[:, None, :]),
        positives,
        selected_truth,
    )
    output["inverse"] = {
        "learned_residual_corrected": learned,
        "forward_cost_only": direct,
        "true_candidate_oracle": oracle,
    }
    return output


def measurement_evaluation(
    split: str,
    measurement_data: Path,
    measurement: MeasurementModuleV3,
    conditions: Sequence[str],
    parameters: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[str, str], np.ndarray]]:
    rows = list(iter_jsonl(measurement_data / "states" / f"{split}.jsonl"))
    predicted = measurement.measure_dataset_states(
        measurement_data, rows, conditions, parameters
    )
    by_condition = {}
    target_parts, predicted_parts = [], []
    for condition in conditions:
        target = np.asarray(
            [raw_state_array(row["target_state"]) for row in rows],
            dtype=np.float32,
        )
        values = np.asarray(
            [predicted[(str(row["state_id"]), condition)] for row in rows],
            dtype=np.float32,
        )
        by_condition[condition] = metric_block(target, values)
        target_parts.append(target)
        predicted_parts.append(values)
    block = {
        "state_count": len(rows),
        "view_count": len(rows) * len(conditions),
        "all_conditions": metric_block(
            np.concatenate(target_parts), np.concatenate(predicted_parts)
        ),
        "by_condition": by_condition,
    }
    return block, rows, predicted


def corrected_visual_requests(
    grids: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], np.ndarray],
    conditions: Sequence[str],
) -> dict[str, Any]:
    grid_map = {str(row["group_id"]): row for row in grids}
    by_group: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in state_rows:
        by_group.setdefault(str(row["group_id"]), {})[
            str(row["role"])
        ] = row

    setups, group_ids = [], []
    current_true, desired_true = [], []
    current_measured, desired_measured = [], []
    positives, selected_truth, statuses, condition_index = [], [], [], []
    for group_id, grid in grid_map.items():
        roles = by_group[group_id]
        current_row = roles["current"]
        true_candidates = true_grid_sensor_states(grid)
        for target_role in ("target_00", "target_01", "target_02"):
            target_row = roles[target_role]
            desired = raw_state_array(target_row["target_state"])
            positive = matching_mask(true_candidates, desired)
            matches = np.flatnonzero(positive).tolist()
            status = (
                "infeasible_within_limits"
                if not matches
                else "unique"
                if len(matches) == 1
                else "ambiguous"
            )
            selected = -1 if not matches else minimum_motion_index(matches)
            for condition_position, condition in enumerate(conditions):
                setups.append(grid["setup"])
                group_ids.append(group_id)
                current_true.append(raw_state_array(current_row["target_state"]))
                desired_true.append(desired)
                current_measured.append(
                    predictions[(str(current_row["state_id"]), condition)]
                )
                desired_measured.append(
                    predictions[(str(target_row["state_id"]), condition)]
                )
                positives.append(positive)
                selected_truth.append(selected)
                statuses.append(STATUS_INDEX[status])
                condition_index.append(condition_position)
    return {
        "setups": setups,
        "group_ids": group_ids,
        "current_true": np.asarray(current_true, dtype=np.float32),
        "desired_true": np.asarray(desired_true, dtype=np.float32),
        "current_measured": np.asarray(current_measured, dtype=np.float32),
        "desired_measured": np.asarray(desired_measured, dtype=np.float32),
        "positives": np.asarray(positives, dtype=np.bool_),
        "selected_truth": np.asarray(selected_truth, dtype=np.int64),
        "statuses": np.asarray(statuses, dtype=np.int64),
        "condition_index": np.asarray(condition_index, dtype=np.int64),
    }


def run_visual_batches(
    pipeline: VisualInversePipeline,
    requests: Mapping[str, Any],
    measured: bool,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_parts, status_parts = [], []
    current_key = "current_measured" if measured else "current_true"
    desired_key = "desired_measured" if measured else "desired_true"
    length = len(requests["setups"])
    for start in range(0, length, batch_size):
        stop = min(start + batch_size, length)
        result = pipeline.predict_from_states(
            requests["setups"][start:stop],
            requests[current_key][start:stop],
            requests[desired_key][start:stop],
            requests["group_ids"][start:stop],
        )
        selected_parts.append(result["selected_indices"])
        status_parts.append(result["predicted_status_indices"])
    return np.concatenate(selected_parts), np.concatenate(status_parts)


def visual_metric_block(
    selected: np.ndarray,
    predicted_status: np.ndarray,
    requests: Mapping[str, Any],
) -> dict[str, Any]:
    positives = requests["positives"]
    statuses = requests["statuses"]
    selected_truth = requests["selected_truth"]
    feasible = positives.any(axis=1)
    success = positives[np.arange(len(positives)), selected]
    exact = selected == selected_truth
    from control_rebuild_v3.train_inverse import macro_f1

    return {
        "request_count": int(len(selected)),
        "reachable_request_count": int(feasible.sum()),
        "specialist_execution": 1.0,
        "physical_target_success_feasible": float(success[feasible].mean()),
        "physical_target_success_all": float(success.mean()),
        "minimum_movement_exact_feasible": float(exact[feasible].mean()),
        "status_accuracy": float((predicted_status == statuses).mean()),
        "status_macro_f1": macro_f1(statuses, predicted_status),
    }


def visual_evaluation(
    split: str,
    control_data: Path,
    pipeline: VisualInversePipeline,
    state_rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], np.ndarray],
    conditions: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    grids = read_jsonl(control_data / "grids" / f"{split}.jsonl")
    requests = corrected_visual_requests(
        grids, state_rows, predictions, conditions
    )
    oracle_selected, oracle_status = run_visual_batches(
        pipeline, requests, measured=False, batch_size=batch_size
    )
    measured_selected, measured_status = run_visual_batches(
        pipeline, requests, measured=True, batch_size=batch_size
    )
    oracle = visual_metric_block(oracle_selected, oracle_status, requests)
    measured = visual_metric_block(
        measured_selected, measured_status, requests
    )
    by_condition = {}
    for index, condition in enumerate(conditions):
        mask = requests["condition_index"] == index
        subset = {
            key: value[mask]
            for key, value in requests.items()
            if isinstance(value, np.ndarray) and len(value) == len(mask)
        }
        by_condition[condition] = {
            "oracle_measurement": visual_metric_block(
                oracle_selected[mask], oracle_status[mask], subset
            ),
            "model_measurement": visual_metric_block(
                measured_selected[mask], measured_status[mask], subset
            ),
        }
    status_counts = Counter(
        STATUS_NAMES[int(index)] for index in requests["statuses"]
    )
    return {
        "corrected_sensor_frame_ground_truth": True,
        "frame_adapter_is_analytic": True,
        "request_status_counts": dict(sorted(status_counts.items())),
        "oracle_measurement": oracle,
        "model_measurement": measured,
        "by_condition": by_condition,
    }


def main() -> None:
    args = parse_args()
    control_data = args.control_data.resolve()
    measurement_data = args.measurement_data.resolve()
    control_run = args.control_run.resolve()
    measurement_run = args.measurement_run.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if bool(existing.get("complete")):
            raise RuntimeError(
                "refusing to overwrite a completed controlled evaluation: "
                f"{output_path}"
            )
    torch, device = configure(20260726, args.device)
    started = time.perf_counter()

    measurement_config = read_json(measurement_data / "config.json")
    conditions = (
        list(args.conditions)
        if args.conditions is not None
        else list(measurement_config["conditions"])
    )
    parameters = measurement_config["condition_parameters"]
    measurement_artifact = measurement_run / "measurement_v3.pt"
    forward_artifact = control_run / "forward_control_v3_calibrated.pt"
    inverse_artifact = control_run / "inverse_control_v3.pt"
    pipeline = VisualInversePipeline(
        torch,
        measurement_artifact,
        forward_artifact,
        inverse_artifact,
        device,
    )
    # The pipeline owns the same measurement model; use that instance.
    measurement = pipeline.measurement

    split_results = {}
    for split in args.splits:
        split_started = time.perf_counter()
        numerical = numerical_evaluation(
            split, control_data, pipeline.forward, pipeline.inverse
        )
        measurement_block, state_rows, predictions = measurement_evaluation(
            split,
            measurement_data,
            measurement,
            conditions,
            parameters,
        )
        visual = visual_evaluation(
            split,
            control_data,
            pipeline,
            state_rows,
            predictions,
            conditions,
            args.visual_batch,
        )
        split_results[split] = {
            "numerical": numerical,
            "measurement": measurement_block,
            "visual_inverse": visual,
            "seconds": time.perf_counter() - split_started,
        }
        partial = {
            "evaluation_version": "control_rebuild_v3_controlled_once",
            "device": str(device),
            "seed": 20260726,
            "splits_requested": list(args.splits),
            "conditions": conditions,
            "split_results": split_results,
            "complete": False,
        }
        output_path.write_text(
            json.dumps(partial, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"completed_split": split, **split_results[split]},
                sort_keys=True,
            ),
            flush=True,
        )

    result = {
        "evaluation_version": "control_rebuild_v3_controlled_once",
        "device": str(device),
        "seed": 20260726,
        "splits_requested": list(args.splits),
        "conditions": conditions,
        "artifacts": {
            "measurement": str(measurement_artifact),
            "forward": str(forward_artifact),
            "inverse": str(inverse_artifact),
        },
        "split_results": split_results,
        "held_out_examples_used_for_training_or_selection": 0,
        "complete": True,
        "seconds": time.perf_counter() - started,
    }
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
