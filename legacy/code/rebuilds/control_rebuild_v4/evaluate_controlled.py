#!/usr/bin/env python3
"""Run one controlled evaluation of all v4 specialist stages."""

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

from control_rebuild_v3.evaluate_controlled import (
    corrected_visual_requests,
    numerical_evaluation,
    run_visual_batches,
    visual_metric_block,
)
from control_rebuild_v3.common import (
    ACTION_GRID,
    STATUS_NAMES,
    group_arrays,
    inverse_pair_arrays,
    read_jsonl,
)
from control_rebuild_v3.train_forward import configure
from control_rebuild_v4.visual_runtime import VisualInversePipelineV4
from measurement_rebuild_v3.common import (
    iter_jsonl,
    measurement_tolerance,
)
from measurement_rebuild_v3.train import metric_block
from specialist_rebuild_v2.common import STATE_FIELDS, raw_state_array

DEFAULT_CONTROL_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_MEASUREMENT_DATA = REPO_ROOT.parent / "VLM_data/measurement_rebuild_v3"
DEFAULT_CONTROL_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"
DEFAULT_MEASUREMENT_V3_RUN = (
    REPO_ROOT.parent / "VLM_runs/measurement_rebuild_v3_one_seed"
)
DEFAULT_MEASUREMENT_V4_RUN = (
    REPO_ROOT.parent / "VLM_runs/measurement_rebuild_v4_one_seed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-data", type=Path, default=DEFAULT_CONTROL_DATA)
    parser.add_argument(
        "--measurement-data", type=Path, default=DEFAULT_MEASUREMENT_DATA
    )
    parser.add_argument("--control-run", type=Path, default=DEFAULT_CONTROL_RUN)
    parser.add_argument(
        "--measurement-v3-run",
        type=Path,
        default=DEFAULT_MEASUREMENT_V3_RUN,
    )
    parser.add_argument(
        "--measurement-v4-run",
        type=Path,
        default=DEFAULT_MEASUREMENT_V4_RUN,
    )
    parser.add_argument(
        "--visual-scorer-name",
        default="visual_sensor_scorer_v4_integrated.pt",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("val", "test_iid", "test_ood_physics", "test_visual_stress"),
        default=("test_iid", "test_ood_physics", "test_visual_stress"),
    )
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--visual-batch", type=int, default=256)
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def state_mapping(values: np.ndarray) -> dict[str, float]:
    return {field: float(values[index]) for index, field in enumerate(STATE_FIELDS)}


def success_failure_examples(
    success: np.ndarray,
    eligible: np.ndarray,
    build: Any,
) -> dict[str, Any]:
    success_indices = np.flatnonzero(success & eligible)
    failure_indices = np.flatnonzero(~success & eligible)
    return {
        "success": (
            None if not len(success_indices) else build(int(success_indices[0]))
        ),
        "failure": (
            None if not len(failure_indices) else build(int(failure_indices[0]))
        ),
    }


def numerical_examples_v4(
    split: str,
    control_data: Path,
    pipeline: VisualInversePipelineV4,
) -> dict[str, Any]:
    rows = read_jsonl(control_data / "grids" / f"{split}.jsonl")
    _, current, tolerance, target_change, group_ids = group_arrays(rows)
    predicted_change = pipeline.forward.predict_changes(rows)
    errors = np.abs(predicted_change - target_change)
    forward_success = np.all(errors <= 1.0, axis=-1).reshape(-1)

    def forward_example(flat_index: int) -> dict[str, Any]:
        group_index, action_index = divmod(flat_index, len(ACTION_GRID))
        return {
            "split": split,
            "group_id": group_ids[group_index],
            "action_index": action_index,
            "action": ACTION_GRID[action_index],
            "target_change": state_mapping(
                target_change[group_index, action_index] * tolerance[group_index]
            ),
            "predicted_change": state_mapping(
                predicted_change[group_index, action_index] * tolerance[group_index]
            ),
            "absolute_error_in_tolerance_units": state_mapping(
                errors[group_index, action_index]
            ),
            "strict_all_five_success": bool(forward_success[flat_index]),
        }

    output = {
        "forward": success_failure_examples(
            forward_success,
            np.ones_like(forward_success, dtype=np.bool_),
            forward_example,
        )
    }
    inverse_path = control_data / "inverse" / f"{split}.jsonl"
    if not inverse_path.is_file():
        output["inverse"] = {"success": None, "failure": None}
        return output

    pairs = read_jsonl(inverse_path)
    grid_map = {str(row["group_id"]): row for row in rows}
    position = {group_id: index for index, group_id in enumerate(group_ids)}
    group_index, _, desired, positives, statuses = inverse_pair_arrays(
        pairs, grid_map, position
    )
    predicted_states = pipeline.forward.predict_states(rows)
    setups = [grid_map[str(pair["group_id"])]["setup"] for pair in pairs]
    current_request = current[group_index]
    candidates = predicted_states[group_index]
    result = pipeline.inverse.score_requests(
        setups,
        current_request,
        desired,
        candidates,
    )
    selected = np.asarray(result["selected_indices"], dtype=np.int64)
    feasible = positives.any(axis=1)
    reached = positives[np.arange(len(pairs)), selected]

    def inverse_example(index: int) -> dict[str, Any]:
        selected_index = int(selected[index])
        matches = np.flatnonzero(positives[index]).tolist()
        request_id = pairs[index].get(
            "request_id",
            pairs[index].get(
                "example_id",
                f"{pairs[index]['group_id']}:inverse:{index:06d}",
            ),
        )
        return {
            "split": split,
            "request_id": str(request_id),
            "group_id": str(pairs[index]["group_id"]),
            "desired_beam_state": state_mapping(desired[index]),
            "true_status": STATUS_NAMES[int(statuses[index])],
            "predicted_status": str(result["predicted_statuses"][index]),
            "matching_action_indices": matches,
            "selected_action_index": selected_index,
            "selected_action": ACTION_GRID[selected_index],
            "predicted_selected_state": state_mapping(
                candidates[index, selected_index]
            ),
            "physical_target_reached": bool(reached[index]),
        }

    output["inverse"] = success_failure_examples(
        reached,
        feasible,
        inverse_example,
    )
    return output


def measurement_evaluation_v4(
    split: str,
    measurement_data: Path,
    pipeline: VisualInversePipelineV4,
    conditions: Sequence[str],
    parameters: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[tuple[str, str], np.ndarray],
]:
    rows = list(iter_jsonl(measurement_data / "states" / f"{split}.jsonl"))
    predicted = pipeline.measure_dataset_states(
        measurement_data, rows, conditions, parameters
    )
    target = np.asarray(
        [raw_state_array(row["target_state"]) for row in rows],
        dtype=np.float32,
    )
    target_parts, predicted_parts = [], []
    by_condition = {}
    example_rows = []
    for condition in conditions:
        values = np.asarray(
            [predicted[(str(row["state_id"]), condition)] for row in rows],
            dtype=np.float32,
        )
        by_condition[condition] = metric_block(target, values)
        tolerances = np.asarray(
            [measurement_tolerance(state) for state in target],
            dtype=np.float32,
        )
        error = np.abs(values - target) / tolerances
        passed = np.all(error <= 1.0, axis=1)
        for index, row in enumerate(rows):
            example_rows.append(
                {
                    "passed": bool(passed[index]),
                    "split": split,
                    "state_id": str(row["state_id"]),
                    "group_id": str(row["group_id"]),
                    "condition": str(condition),
                    "target_beam_state": state_mapping(target[index]),
                    "predicted_beam_state": state_mapping(values[index]),
                    "absolute_error_in_tolerance_units": state_mapping(error[index]),
                    "strict_all_five_success": bool(passed[index]),
                }
            )
        target_parts.append(target)
        predicted_parts.append(values)
    example_pass = np.asarray(
        [row["passed"] for row in example_rows],
        dtype=np.bool_,
    )

    def measurement_example(index: int) -> dict[str, Any]:
        value = dict(example_rows[index])
        value.pop("passed")
        return value

    return (
        {
            "state_count": len(rows),
            "view_count": len(rows) * len(conditions),
            "all_conditions": metric_block(
                np.concatenate(target_parts),
                np.concatenate(predicted_parts),
            ),
            "by_condition": by_condition,
            "examples": success_failure_examples(
                example_pass,
                np.ones_like(example_pass, dtype=np.bool_),
                measurement_example,
            ),
        },
        rows,
        predicted,
    )


def visual_evaluation_v4(
    split: str,
    control_data: Path,
    pipeline: VisualInversePipelineV4,
    state_rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], np.ndarray],
    conditions: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    grids = read_jsonl(control_data / "grids" / f"{split}.jsonl")
    requests = corrected_visual_requests(grids, state_rows, predictions, conditions)
    oracle_selected, oracle_status = run_visual_batches(
        pipeline, requests, measured=False, batch_size=batch_size
    )
    measured_selected, measured_status = run_visual_batches(
        pipeline, requests, measured=True, batch_size=batch_size
    )
    oracle = visual_metric_block(oracle_selected, oracle_status, requests)
    measured = visual_metric_block(measured_selected, measured_status, requests)
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
    feasible = requests["positives"].any(axis=1)
    reached = requests["positives"][
        np.arange(len(measured_selected)), measured_selected
    ]

    def visual_example(index: int) -> dict[str, Any]:
        selected_index = int(measured_selected[index])
        condition = str(conditions[int(requests["condition_index"][index])])
        return {
            "split": split,
            "request_index": index,
            "group_id": str(requests["group_ids"][index]),
            "condition": condition,
            "true_current_sensor_state": state_mapping(requests["current_true"][index]),
            "measured_current_sensor_state": state_mapping(
                requests["current_measured"][index]
            ),
            "true_desired_sensor_state": state_mapping(requests["desired_true"][index]),
            "measured_desired_sensor_state": state_mapping(
                requests["desired_measured"][index]
            ),
            "true_status": STATUS_NAMES[int(requests["statuses"][index])],
            "predicted_status": STATUS_NAMES[int(measured_status[index])],
            "matching_action_indices": np.flatnonzero(
                requests["positives"][index]
            ).tolist(),
            "selected_action_index": selected_index,
            "selected_action": ACTION_GRID[selected_index],
            "physical_target_reached": bool(reached[index]),
        }

    status_counts = Counter(STATUS_NAMES[int(index)] for index in requests["statuses"])
    return {
        "corrected_sensor_frame_ground_truth": True,
        "frame_adapter_is_analytic": True,
        "request_status_counts": dict(sorted(status_counts.items())),
        "oracle_measurement": oracle,
        "model_measurement": measured,
        "by_condition": by_condition,
        "examples": success_failure_examples(
            reached,
            feasible,
            visual_example,
        ),
    }


def main() -> None:
    args = parse_args()
    control_data = args.control_data.resolve()
    measurement_data = args.measurement_data.resolve()
    control_run = args.control_run.resolve()
    measurement_v3_run = args.measurement_v3_run.resolve()
    measurement_v4_run = args.measurement_v4_run.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else control_run / "controlled_evaluation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if output.is_file():
        existing = read_config(output)
        if bool(existing.get("complete")):
            raise RuntimeError(f"refusing to overwrite completed evaluation: {output}")
        previous_splits = list(existing.get("splits_requested", []))
        if previous_splits and previous_splits != list(args.splits):
            raise RuntimeError(
                "partial evaluation requested different splits: "
                f"{previous_splits} versus {list(args.splits)}"
            )

    torch, device = configure(20260726, args.device)
    measurement_config = read_config(measurement_data / "config.json")
    conditions = (
        list(args.conditions)
        if args.conditions is not None
        else list(measurement_config["conditions"])
    )
    parameters = measurement_config["condition_parameters"]
    artifacts = {
        "measurement_v3": measurement_v3_run / "measurement_v3.pt",
        "measurement_calibrator_v4": (
            measurement_v4_run / "measurement_calibrator_v4.pt"
        ),
        "forward_v4": control_run / "forward_physics_residual_v4.pt",
        "inverse_v4": control_run / "inverse_control_v4.pt",
        "visual_scorer_v4": control_run / args.visual_scorer_name,
    }
    pipeline = VisualInversePipelineV4(
        torch,
        artifacts["measurement_v3"],
        artifacts["measurement_calibrator_v4"],
        artifacts["forward_v4"],
        artifacts["inverse_v4"],
        artifacts["visual_scorer_v4"],
        device,
    )
    started = time.perf_counter()
    previous_seconds = float(existing.get("seconds", 0.0))
    split_results = dict(existing.get("split_results", {}))
    for split in args.splits:
        if split in split_results:
            print(json.dumps({"resumed_completed_split": split}), flush=True)
            continue
        split_started = time.perf_counter()
        numerical = numerical_evaluation(
            split, control_data, pipeline.forward, pipeline.inverse
        )
        numerical["examples"] = numerical_examples_v4(split, control_data, pipeline)
        measurement, state_rows, predictions = measurement_evaluation_v4(
            split,
            measurement_data,
            pipeline,
            conditions,
            parameters,
        )
        visual = visual_evaluation_v4(
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
            "measurement": measurement,
            "visual_inverse": visual,
            "seconds": time.perf_counter() - split_started,
        }
        partial = {
            "evaluation_version": "control_rebuild_v4_controlled_once",
            "device": str(device),
            "seed": 20260726,
            "splits_requested": list(args.splits),
            "conditions": conditions,
            "artifacts": {
                key: str(value.resolve()) for key, value in artifacts.items()
            },
            "split_results": split_results,
            "complete": False,
            "seconds": previous_seconds + time.perf_counter() - started,
        }
        output.write_text(
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
        "evaluation_version": "control_rebuild_v4_controlled_once",
        "device": str(device),
        "seed": 20260726,
        "splits_requested": list(args.splits),
        "conditions": conditions,
        "artifacts": {key: str(value.resolve()) for key, value in artifacts.items()},
        "split_results": split_results,
        "held_out_examples_used_for_training_or_selection": 0,
        "held_out_state_examples_evaluated": sum(
            int(split_results[split]["measurement"]["state_count"])
            for split in args.splits
            if split != "val"
        ),
        "complete": True,
        "seconds": previous_seconds + time.perf_counter() - started,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
