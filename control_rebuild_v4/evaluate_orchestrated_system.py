#!/usr/bin/env python3
"""Evaluate saved Qwen decisions through the candidate v4 specialist overlay."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Qwen_orchestration.scripts.evaluate_end_to_end as frozen_evaluator
from control_rebuild_v3.train_forward import configure
from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    DIRECTION_FIELDS,
    change_and_directions,
)

DEFAULT_QWEN_DATA = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "Qwen_orchestration/results/v1/stage2_safe_runtime_v1"
    / "checkpoint-1000_all.jsonl"
)
DEFAULT_CONTROL_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"


class RecordingRuntime:
    """Record dispatch results by decision identity without changing execution."""

    def __init__(self, backend: OrchestratedSpecialistRuntimeV4) -> None:
        self.backend = backend
        self.records: dict[int, dict[str, Any]] = {}

    def dispatch(
        self,
        decision: Mapping[str, Any],
        available_images: Mapping[str, str | Path] | None = None,
    ) -> dict[str, Any]:
        try:
            dispatch = self.backend.dispatch(decision, available_images)
        except Exception as error:
            self.records[id(decision)] = {
                "dispatch": None,
                "error": f"{type(error).__name__}: {error}",
            }
            raise
        self.records[id(decision)] = {
            "dispatch": dispatch,
            "error": None,
        }
        return dispatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN_DATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--control-run", type=Path, default=DEFAULT_CONTROL_RUN)
    parser.add_argument("--overlay-manifest", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--max-per-category",
        type=int,
        help="Bounded smoke subset; omit for the complete 1,600-record validation.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details", type=Path)
    return parser.parse_args()


def subset_by_category(
    rows: list[dict[str, Any]], maximum: int | None
) -> list[dict[str, Any]]:
    if maximum is None:
        return rows
    if maximum < 1:
        raise ValueError("--max-per-category must be positive")
    counts: Counter[str] = Counter()
    output = []
    for row in rows:
        category = str(row["category"])
        if counts[category] < maximum:
            output.append(row)
            counts[category] += 1
    return output


def task_ready_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(row["target_decision"]["task_type"])
        for row in rows
        if row["target_decision"]["status"] == "ready"
    )
    return dict(sorted(counts.items()))


def recorded_specialist_result(
    runtime: RecordingRuntime,
    decision: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if decision is None:
        return None
    record = runtime.records.get(id(decision))
    if record is None:
        return None
    dispatch = record.get("dispatch")
    if (
        not isinstance(dispatch, Mapping)
        or dispatch.get("executed") is not True
        or not isinstance(dispatch.get("result"), Mapping)
    ):
        return None
    return dispatch["result"]


def simulator_forward_truth(
    private: Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, Any]:
    """Calculate physical change for evaluation only, never for inference."""

    from optical_sim.src.experiment_generator import load_yaml as load_sim_yaml
    from optics_understanding_sft.build_dataset import simulator_result
    from optics_understanding_sft.direction_inverse_v1.build_inverse import (
        config_from_visible,
    )

    visible = {**private["setup"], "sensor_resolution_px": [1024, 1024]}
    base = load_sim_yaml(REPO_ROOT / "optical_sim/configs/base_config.yaml")
    config = config_from_visible(visible, base)
    after = simulator_result(config, action)["state"]
    change, directions = change_and_directions(
        private["current_beam_state"],
        after,
    )
    return {
        "change": change,
        "directions": directions,
    }


def direction_all_five_exact(
    truth: Mapping[str, Any],
    predicted: Mapping[str, Any] | None,
) -> bool:
    if predicted is None or not isinstance(predicted.get("directions"), Mapping):
        return False
    return all(
        predicted["directions"].get(field) == truth["directions"][field]
        for field in DIRECTION_FIELDS
    )


def direction_per_field_metrics(
    truth_rows: list[Mapping[str, Any]],
    predicted_rows: list[Mapping[str, Any] | None],
) -> dict[str, dict[str, Any]]:
    """Report accuracy and macro-F1 separately for each direction output."""

    output = {}
    for field in DIRECTION_FIELDS:
        targets = [str(row["directions"][field]) for row in truth_rows]
        predictions = [
            (
                str(row["directions"].get(field, "__failure__"))
                if row is not None
                and isinstance(row.get("directions"), Mapping)
                else "__failure__"
            )
            for row in predicted_rows
        ]
        correct = sum(
            target == predicted
            for target, predicted in zip(targets, predictions, strict=True)
        )
        output[field] = {
            "count": len(targets),
            "correct_count": correct,
            "accuracy": frozen_evaluator.ratio(correct, len(targets)),
            "macro_f1": frozen_evaluator.macro_f1(targets, predictions),
        }
    return output


def physical_forward_direction_metrics(
    canonical: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    private_rows: list[dict[str, Any]],
    runtime: RecordingRuntime,
    truth_function: Any = simulator_forward_truth,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Score direction and forward outputs against cached simulator truth."""

    predictions_by_id = {str(row["example_id"]): row for row in predictions}
    private_by_group = {str(row["group_id"]): row for row in private_rows}
    truth_cache: dict[tuple[str, tuple[float, ...]], dict[str, Any]] = {}
    direction_rows = []
    forward_rows = []
    detail_updates: dict[str, dict[str, Any]] = {}

    for row in canonical:
        target = row["target_decision"]
        if target["status"] != "ready" or target["task_type"] not in {
            "direction_prediction",
            "forward_prediction",
        }:
            continue
        private = private_by_group[str(row["group_id"])]
        action = target["arguments"]["action"]
        cache_key = (
            str(row["group_id"]),
            tuple(float(action[field]) for field in ACTION_FIELDS),
        )
        if cache_key not in truth_cache:
            truth_cache[cache_key] = truth_function(private, action)
        truth = truth_cache[cache_key]
        prediction_row = predictions_by_id.get(str(row["example_id"]))
        predicted_decision = (
            prediction_row.get("parsed_json")
            if prediction_row is not None
            and isinstance(prediction_row.get("parsed_json"), Mapping)
            else None
        )
        oracle_result = recorded_specialist_result(runtime, target)
        predicted_result = recorded_specialist_result(runtime, predicted_decision)
        item = {
            "example_id": str(row["example_id"]),
            "route": str(target["route_name"]),
            "truth": truth,
            "oracle": oracle_result,
            "predicted": predicted_result,
            "current_peak": float(private["current_beam_state"]["peak_intensity"]),
        }
        if target["task_type"] == "direction_prediction":
            direction_rows.append(item)
            detail_updates[item["example_id"]] = {
                "physical_metric": "all_five_directions_exact",
                "physical_success": direction_all_five_exact(truth, predicted_result),
                "correctly_routed_specialist_physical_success": (
                    direction_all_five_exact(truth, oracle_result)
                ),
            }
        else:
            forward_rows.append(item)
            predicted_success = frozen_evaluator.forward_success(
                truth,
                predicted_result,
                item["current_peak"],
            )
            oracle_success = frozen_evaluator.forward_success(
                truth,
                oracle_result,
                item["current_peak"],
            )
            detail_updates[item["example_id"]] = {
                "physical_metric": "all_five_numerical_changes_within_tolerance",
                "physical_success": predicted_success,
                "correctly_routed_specialist_physical_success": oracle_success,
            }

    direction_truth = [row["truth"] for row in direction_rows]
    direction_predicted = [row["predicted"] for row in direction_rows]
    direction_oracle = [row["oracle"] for row in direction_rows]
    direction_predicted_exact = [
        direction_all_five_exact(row["truth"], row["predicted"])
        for row in direction_rows
    ]
    direction_oracle_exact = [
        direction_all_five_exact(row["truth"], row["oracle"]) for row in direction_rows
    ]
    forward_predicted = [
        frozen_evaluator.forward_success(
            row["truth"],
            row["predicted"],
            row["current_peak"],
        )
        for row in forward_rows
    ]
    forward_oracle = [
        frozen_evaluator.forward_success(
            row["truth"],
            row["oracle"],
            row["current_peak"],
        )
        for row in forward_rows
    ]

    direction_by_route = {}
    for route in sorted({row["route"] for row in direction_rows}):
        selected = [row for row in direction_rows if row["route"] == route]
        direction_by_route[route] = {
            "count": len(selected),
            "macro_f1": frozen_evaluator.direction_metric(
                [row["truth"] for row in selected],
                [row["predicted"] for row in selected],
            ),
            "all_five_exact": frozen_evaluator.ratio(
                sum(
                    direction_all_five_exact(row["truth"], row["predicted"])
                    for row in selected
                ),
                len(selected),
            ),
            "per_field": direction_per_field_metrics(
                [row["truth"] for row in selected],
                [row["predicted"] for row in selected],
            ),
        }
    forward_by_route = {}
    for route in sorted({row["route"] for row in forward_rows}):
        selected = [row for row in forward_rows if row["route"] == route]
        successes = [
            frozen_evaluator.forward_success(
                row["truth"],
                row["predicted"],
                row["current_peak"],
            )
            for row in selected
        ]
        forward_by_route[route] = {
            "count": len(selected),
            "strict_all_five_success": frozen_evaluator.ratio(
                sum(successes), len(successes)
            ),
        }
    metrics = {
        "physical_ground_truth_scope": (
            "simulator truth for direction and forward; image ground truth "
            "for measurement; simulator action replay for inverse"
        ),
        "end_to_end_direction_physical_macro_f1": (
            frozen_evaluator.direction_metric(
                direction_truth,
                direction_predicted,
            )
        ),
        "correctly_routed_direction_physical_macro_f1": (
            frozen_evaluator.direction_metric(
                direction_truth,
                direction_oracle,
            )
        ),
        "end_to_end_direction_physical_per_field": (
            direction_per_field_metrics(
                direction_truth,
                direction_predicted,
            )
        ),
        "correctly_routed_direction_physical_per_field": (
            direction_per_field_metrics(
                direction_truth,
                direction_oracle,
            )
        ),
        "end_to_end_direction_physical_all_five_exact": frozen_evaluator.ratio(
            sum(direction_predicted_exact), len(direction_predicted_exact)
        ),
        "correctly_routed_direction_physical_all_five_exact": (
            frozen_evaluator.ratio(
                sum(direction_oracle_exact), len(direction_oracle_exact)
            )
        ),
        "direction_physical_count": len(direction_rows),
        "direction_physical_by_route": direction_by_route,
        "end_to_end_forward_physical_strict_all_five_success": (
            frozen_evaluator.ratio(
                sum(forward_predicted),
                len(forward_predicted),
            )
        ),
        "correctly_routed_forward_physical_strict_all_five_success": (
            frozen_evaluator.ratio(
                sum(forward_oracle),
                len(forward_oracle),
            )
        ),
        "forward_physical_count": len(forward_rows),
        "forward_physical_by_route": forward_by_route,
        "private_forward_ground_truth_simulator_calls": len(truth_cache),
    }
    return metrics, detail_updates


def main() -> None:
    args = parse_args()
    qwen_data = args.qwen_data.resolve()
    control_run = args.control_run.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else control_run / "orchestrated_system_validation.json"
    )
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if bool(existing.get("complete")):
            raise RuntimeError(f"refusing to overwrite completed validation: {output}")

    canonical = frozen_evaluator.read_jsonl(qwen_data / "canonical/val.jsonl")
    canonical = subset_by_category(canonical, args.max_per_category)
    selected_ids = {str(row["example_id"]) for row in canonical}
    predictions = [
        row
        for row in frozen_evaluator.read_jsonl(args.predictions.resolve())
        if str(row["example_id"]) in selected_ids
    ]
    if len(predictions) != len(canonical):
        raise ValueError(
            f"saved Qwen predictions cover {len(predictions)}/{len(canonical)} records"
        )
    private_rows = frozen_evaluator.read_jsonl(
        qwen_data / "private/source_cases/val.jsonl"
    )
    torch, device = configure(20260726, args.device)
    manifest_path = (
        args.overlay_manifest.resolve()
        if args.overlay_manifest is not None
        else control_run / "candidate_overlay_manifest.json"
    )
    runtime = OrchestratedSpecialistRuntimeV4.from_manifest(
        torch,
        manifest_path,
        device,
    )
    recording_runtime = RecordingRuntime(runtime)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    original_runtime = frozen_evaluator.OrchestrationRuntime
    frozen_evaluator.OrchestrationRuntime = lambda: recording_runtime
    try:
        metrics, details = frozen_evaluator.evaluate(
            predictions,
            canonical,
            private_rows,
            qwen_data,
        )
    finally:
        frozen_evaluator.OrchestrationRuntime = original_runtime
    physical_metrics, physical_details = physical_forward_direction_metrics(
        canonical,
        predictions,
        private_rows,
        recording_runtime,
    )
    metrics.update(physical_metrics)
    physical_comparisons = {
        "direction": (
            metrics["end_to_end_direction_physical_macro_f1"],
            metrics["correctly_routed_direction_physical_macro_f1"],
        ),
        "forward": (
            metrics["end_to_end_forward_physical_strict_all_five_success"],
            metrics["correctly_routed_forward_physical_strict_all_five_success"],
        ),
        "measurement": (
            metrics["end_to_end_visual_measurement_strict_all_five_success"],
            metrics["oracle_visual_measurement_strict_all_five_success"],
        ),
        "inverse": (
            metrics["end_to_end_inverse_target_reached_rate"],
            metrics["oracle_inverse_target_reached_rate"],
        ),
    }
    physical_losses = {
        task: max(oracle - predicted, 0.0)
        for task, (predicted, oracle) in physical_comparisons.items()
    }
    metrics["physical_loss_versus_correctly_routed_specialist_by_task"] = (
        physical_losses
    )
    metrics["physical_macro_loss_versus_correctly_routed_specialist"] = sum(
        physical_losses.values()
    ) / len(physical_losses)
    metrics["execution_consistency_gate_passed"] = bool(metrics["promotion_passed"])
    metrics["physical_quality_gate_defined"] = False
    metrics["private_simulator_scoring_calls_total"] = int(
        metrics["private_simulator_scoring_calls"]
    ) + int(metrics["private_forward_ground_truth_simulator_calls"])
    for detail in details:
        detail.update(physical_details.get(str(detail["example_id"]), {}))

    report = {
        "evaluation_version": "saved_qwen_checkpoint1000_plus_specialists_v4",
        "scope": (
            "saved checkpoint-1000 Qwen decisions on frozen in-domain "
            "validation, executed through the candidate v4 overlay"
        ),
        "device": str(device),
        "predictions": str(args.predictions.resolve()),
        "overlay_manifest": str(manifest_path),
        "record_count": len(canonical),
        "task_ready_counts": task_ready_counts(canonical),
        "max_per_category": args.max_per_category,
        "private_scoring_note": (
            "After inference, the evaluator computes one cached physical "
            "direction/forward truth per optical case and replays selected "
            "inverse actions. No inference path calls the simulator."
        ),
        "artifacts": {
            key: str(value["path"]) for key, value in manifest["artifacts"].items()
        },
        "metrics": metrics,
        "complete": True,
        "seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    details_path = (
        args.details.resolve()
        if args.details is not None
        else output.with_suffix(".details.jsonl")
    )
    with details_path.open("w", encoding="utf-8") as stream:
        for row in details:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
