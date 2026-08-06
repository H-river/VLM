#!/usr/bin/env python3
"""Execute generated decisions and compare them with the frozen oracle pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Qwen_orchestration.runtime import OrchestrationRuntime


DIRECTION_FIELDS = (
    "centroid_x",
    "centroid_y",
    "width_x",
    "width_y",
    "peak_intensity",
)
DIRECTION_CLASSES = ("decrease", "no_change", "increase")
CHANGE_TOLERANCES = {
    "centroid_x_px": 1.0,
    "centroid_y_px": 1.0,
    "sigma_x_px": 2.0,
    "sigma_y_px": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--canonical-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--private-source-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--details-jsonl", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def image_bindings(row: Mapping[str, Any], root: Path) -> dict[str, Path]:
    return {f"image_{index}": root / value for index, value in enumerate(row["images"])}


def contains_simulator_use(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("simulator_at_inference") is True:
            return True
        return any(contains_simulator_use(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_simulator_use(item) for item in value)
    return False


def macro_f1(targets: list[str], predictions: list[str]) -> float:
    scores = []
    for label in DIRECTION_CLASSES:
        true_positive = sum(
            target == label and predicted == label
            for target, predicted in zip(targets, predictions, strict=True)
        )
        false_positive = sum(
            target != label and predicted == label
            for target, predicted in zip(targets, predictions, strict=True)
        )
        false_negative = sum(
            target == label and predicted != label
            for target, predicted in zip(targets, predictions, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def direction_metric(
    oracle_results: list[Mapping[str, Any]],
    predicted_results: list[Mapping[str, Any] | None],
) -> float:
    field_scores = []
    for field in DIRECTION_FIELDS:
        targets = [result["directions"][field] for result in oracle_results]
        predictions = [
            (
                result["directions"].get(field, "__failure__")
                if result is not None
                and isinstance(result.get("directions"), Mapping)
                else "__failure__"
            )
            for result in predicted_results
        ]
        field_scores.append(macro_f1(targets, predictions))
    return sum(field_scores) / max(len(field_scores), 1)


def forward_success(
    oracle: Mapping[str, Any],
    predicted: Mapping[str, Any] | None,
    current_peak: float,
) -> bool:
    if predicted is None or not isinstance(predicted.get("change"), Mapping):
        return False
    tolerances = {
        **CHANGE_TOLERANCES,
        "peak_intensity": max(0.05 * abs(current_peak), 1e-6),
    }
    return all(
        key in predicted["change"]
        and abs(float(predicted["change"][key]) - float(oracle["change"][key]))
        <= tolerance
        for key, tolerance in tolerances.items()
    )


def measurement_success(
    result: Mapping[str, Any] | None,
    true_state: Mapping[str, Any],
    setup: Mapping[str, Any],
) -> bool:
    if result is None or not isinstance(result.get("beam_state"), Mapping):
        return False
    measured = result["beam_state"]
    pixel_size_mm = float(setup["pixel_size_um"]) / 1000.0
    true_sensor = dict(true_state)
    true_sensor["centroid_x_px"] = float(true_state["centroid_x_px"]) - (
        float(setup["camera_x_offset_mm"]) / pixel_size_mm
    )
    true_sensor["centroid_y_px"] = float(true_state["centroid_y_px"]) - (
        float(setup["camera_y_offset_mm"]) / pixel_size_mm
    )
    return (
        abs(float(measured["centroid_x_px"]) - float(true_sensor["centroid_x_px"]))
        <= 1.0
        and abs(
            float(measured["centroid_y_px"]) - float(true_sensor["centroid_y_px"])
        )
        <= 1.0
        and abs(float(measured["sigma_x_px"]) - float(true_sensor["sigma_x_px"]))
        <= 2.0
        and abs(float(measured["sigma_y_px"]) - float(true_sensor["sigma_y_px"]))
        <= 2.0
        and abs(
            float(measured["peak_intensity"])
            - float(true_sensor["peak_intensity"])
        )
        <= 0.05 * max(abs(float(true_sensor["peak_intensity"])), 1e-12)
    )


def private_inverse_target_reached(
    setup: Mapping[str, Any],
    action: Mapping[str, Any] | None,
    desired: Mapping[str, Any],
) -> bool:
    """Use the simulator only as a private evaluator, never in dispatch."""
    if action is None:
        return False
    from optical_sim.src.experiment_generator import load_yaml as load_sim_yaml
    from optics_understanding_sft.build_dataset import simulator_result
    from optics_understanding_sft.direction_inverse_v1.build_inverse import (
        config_from_visible,
    )

    visible = {**setup, "sensor_resolution_px": [1024, 1024]}
    base = load_sim_yaml(REPO_ROOT / "optical_sim/configs/base_config.yaml")
    config = config_from_visible(visible, base)
    state = simulator_result(config, action)["state"]
    centroid_distance = math.hypot(
        float(state["centroid_x_px"]) - float(desired["centroid_x_px"]),
        float(state["centroid_y_px"]) - float(desired["centroid_y_px"]),
    )
    peak_scale = max(abs(float(desired["peak_intensity"])), 1e-12)
    return (
        centroid_distance <= 0.5
        and abs(float(state["sigma_x_px"]) - float(desired["sigma_x_px"])) <= 1.0
        and abs(float(state["sigma_y_px"]) - float(desired["sigma_y_px"])) <= 1.0
        and abs(
            float(state["peak_intensity"]) - float(desired["peak_intensity"])
        )
        / peak_scale
        <= 0.02
    )


def ratio(numerator: int, denominator: int) -> float:
    return numerator / max(denominator, 1)


def evaluate(
    predictions: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    private_rows: list[dict[str, Any]],
    image_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions_by_id = {row["example_id"]: row for row in predictions}
    private_by_group = {row["group_id"]: row for row in private_rows}
    runtime = OrchestrationRuntime()
    counts = Counter()
    details: list[dict[str, Any]] = []
    task_items: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in canonical_rows:
        target = row["target_decision"]
        prediction_row = predictions_by_id.get(row["example_id"])
        predicted = prediction_row.get("parsed_json") if prediction_row else None
        target_ready = target["status"] == "ready"
        if target_ready:
            counts["target_ready"] += 1
        if (
            target["status"] == "needs_clarification"
            and isinstance(predicted, Mapping)
            and predicted.get("status") == "ready"
        ):
            counts["silent_execution_with_missing_inputs"] += 1

        bindings = image_bindings(row, image_root)
        oracle_dispatch = (
            runtime.dispatch(target, bindings) if target_ready else None
        )
        predicted_dispatch = None
        dispatch_error = None
        if isinstance(predicted, Mapping):
            try:
                predicted_dispatch = runtime.dispatch(predicted, bindings)
            except Exception as error:
                dispatch_error = f"{type(error).__name__}: {error}"
        if (
            target_ready
            and predicted_dispatch is not None
            and predicted_dispatch.get("executed") is True
        ):
            counts["successful_ready_execution"] += 1
        if predicted_dispatch is not None and contains_simulator_use(predicted_dispatch):
            counts["simulator_calls_during_inference"] += 1

        oracle_result = (
            oracle_dispatch["result"]
            if oracle_dispatch is not None and oracle_dispatch.get("executed")
            else None
        )
        predicted_result = (
            predicted_dispatch["result"]
            if predicted_dispatch is not None and predicted_dispatch.get("executed")
            else None
        )
        if target_ready:
            task_items[target["task_type"]].append(
                {
                    "row": row,
                    "private": private_by_group[row["group_id"]],
                    "oracle": oracle_result,
                    "predicted": predicted_result,
                    "route_exact": (
                        isinstance(predicted, Mapping)
                        and predicted.get("route_name") == target["route_name"]
                    ),
                }
            )
        details.append(
            {
                "example_id": row["example_id"],
                "target_status": target["status"],
                "target_route": target["route_name"],
                "predicted_status": (
                    predicted.get("status") if isinstance(predicted, Mapping) else None
                ),
                "predicted_route": (
                    predicted.get("route_name")
                    if isinstance(predicted, Mapping)
                    else None
                ),
                "executed": bool(
                    predicted_dispatch is not None
                    and predicted_dispatch.get("executed")
                ),
                "dispatch_error": dispatch_error,
            }
        )

    metrics: dict[str, Any] = {
        "record_count": len(canonical_rows),
        "target_ready_count": counts["target_ready"],
        "successful_valid_execution_rate": ratio(
            counts["successful_ready_execution"], counts["target_ready"]
        ),
        "silent_execution_with_missing_inputs_count": counts[
            "silent_execution_with_missing_inputs"
        ],
        "simulator_calls_during_inference_count": counts[
            "simulator_calls_during_inference"
        ],
    }

    direction = task_items["direction_prediction"]
    direction_oracle = [item["oracle"] for item in direction]
    direction_predicted = [item["predicted"] for item in direction]
    metrics["end_to_end_direction_macro_f1"] = direction_metric(
        direction_oracle, direction_predicted
    )
    metrics["oracle_direction_macro_f1"] = direction_metric(
        direction_oracle, direction_oracle
    )

    forward = task_items["forward_prediction"]
    forward_predicted = [
        forward_success(
            item["oracle"],
            item["predicted"],
            float(item["private"]["current_beam_state"]["peak_intensity"]),
        )
        for item in forward
    ]
    metrics["end_to_end_forward_strict_all_five_success"] = ratio(
        sum(forward_predicted), len(forward_predicted)
    )
    metrics["oracle_forward_strict_all_five_success"] = 1.0 if forward else 0.0

    measurement = task_items["beam_profile_measurement"]
    measurement_predicted = [
        measurement_success(
            item["predicted"],
            item["private"]["current_beam_state"],
            item["private"]["setup"],
        )
        for item in measurement
    ]
    measurement_oracle = [
        measurement_success(
            item["oracle"],
            item["private"]["current_beam_state"],
            item["private"]["setup"],
        )
        for item in measurement
    ]
    metrics["end_to_end_visual_measurement_strict_all_five_success"] = ratio(
        sum(measurement_predicted), len(measurement_predicted)
    )
    metrics["oracle_visual_measurement_strict_all_five_success"] = ratio(
        sum(measurement_oracle), len(measurement_oracle)
    )

    inverse = task_items["inverse_control"]
    inverse_predicted, inverse_oracle = [], []
    for item in inverse:
        private = item["private"]
        inverse_predicted.append(
            private_inverse_target_reached(
                private["setup"],
                (
                    item["predicted"].get("selected_action")
                    if item["predicted"] is not None
                    else None
                ),
                private["desired_beam_state"],
            )
        )
        inverse_oracle.append(
            private_inverse_target_reached(
                private["setup"],
                item["oracle"]["selected_action"],
                private["desired_beam_state"],
            )
        )
    metrics["end_to_end_inverse_target_reached_rate"] = ratio(
        sum(inverse_predicted), len(inverse_predicted)
    )
    metrics["oracle_inverse_target_reached_rate"] = ratio(
        sum(inverse_oracle), len(inverse_oracle)
    )
    metrics["private_simulator_scoring_calls"] = 2 * len(inverse)

    comparisons = {
        "direction": (
            metrics["end_to_end_direction_macro_f1"],
            metrics["oracle_direction_macro_f1"],
        ),
        "forward": (
            metrics["end_to_end_forward_strict_all_five_success"],
            metrics["oracle_forward_strict_all_five_success"],
        ),
        "inverse": (
            metrics["end_to_end_inverse_target_reached_rate"],
            metrics["oracle_inverse_target_reached_rate"],
        ),
        "measurement": (
            metrics["end_to_end_visual_measurement_strict_all_five_success"],
            metrics["oracle_visual_measurement_strict_all_five_success"],
        ),
    }
    losses = {
        task: max(oracle - predicted, 0.0)
        for task, (predicted, oracle) in comparisons.items()
    }
    metrics["absolute_loss_versus_oracle_by_task"] = losses
    metrics["macro_absolute_loss_versus_oracle"] = sum(losses.values()) / len(losses)
    metrics["maximum_absolute_loss_versus_oracle"] = max(losses.values())
    loss_gate = 0.02
    loss_gate_epsilon = 1e-12
    metrics["promotion_passed"] = (
        metrics["successful_valid_execution_rate"] >= 0.95
        and metrics["maximum_absolute_loss_versus_oracle"]
        <= loss_gate + loss_gate_epsilon
        and metrics["silent_execution_with_missing_inputs_count"] == 0
        and metrics["simulator_calls_during_inference_count"] == 0
    )
    return metrics, details


def main() -> None:
    args = parse_args()
    metrics, details = evaluate(
        read_jsonl(args.predictions_jsonl),
        read_jsonl(args.canonical_jsonl),
        read_jsonl(args.private_source_jsonl),
        args.image_root,
    )
    report = {
        "predictions_jsonl": str(args.predictions_jsonl.resolve()),
        "canonical_jsonl": str(args.canonical_jsonl.resolve()),
        "private_source_jsonl": str(args.private_source_jsonl.resolve()),
        "private_scoring_note": (
            "Simulator replay is evaluation-only and is not called by the "
            "orchestration runtime or any specialist."
        ),
        "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.details_jsonl:
        args.details_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.details_jsonl.open("w", encoding="utf-8") as stream:
            for item in details:
                stream.write(json.dumps(item, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not metrics["promotion_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
