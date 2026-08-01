#!/usr/bin/env python3
"""Run the frozen specialist-baseline and Qwen-to-v12 evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuous_control_v12.contracts import ACTION_FIELDS, OUTPUT_FIELDS, POSITION_FIELDS, tolerance_vector
from Qwen_orchestration.runtime.errors import ContractError
from Qwen_orchestration.scripts.evaluate_qwen import generate_batch, load_config, load_model
from Qwen_orchestration.v12.adapter import V12Adapter
from Qwen_orchestration.v12.dispatcher import V12OrchestrationRuntime, validate_v12_decision


PHYSICAL_TASKS = (
    "measurement",
    "direction_prediction_v12",
    "forward_prediction_v12",
    "inverse_control_v12",
)
FAILURE_STAGES = (
    "Qwen invalid JSON/schema",
    "wrong status",
    "wrong task/route",
    "missing/wrong argument",
    "unit/sign/order error",
    "image-role error",
    "adapter validation",
    "measurement specialist",
    "v12 forward prediction",
    "direction thresholding",
    "H1 CEM search/planning",
    "actuator boundary",
    "simulator/runtime",
    "checkpoint/config mismatch",
    "unknown",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--qwen-config",
        type=Path,
        default=REPO_ROOT / "Qwen_orchestration/configs/qwen25vl_3b_orchestrator_v12_eval.yaml",
    )
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip()


def image_bindings(case: Mapping[str, Any]) -> dict[str, Path]:
    return {
        f"image_{index}": Path(item["path"]).resolve()
        for index, item in enumerate(case["input_references"])
    }


def validate_freeze(run_dir: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = run_dir / "frozen_manifest.jsonl"
    if sha256(manifest) != str(config["manifest_sha256"]):
        raise RuntimeError("frozen manifest hash mismatch")
    if sha256(Path(config["runtime_config"])) != str(config["runtime_config_sha256"]):
        raise RuntimeError("runtime config changed after manifest freeze")
    if sha256(Path(config["checkpoint"])) != str(config["checkpoint_sha256"]):
        raise RuntimeError("v12 checkpoint changed after manifest freeze")
    for path_key, hash_key in (
        ("protected_test_jsonl", "protected_test_jsonl_sha256"),
        ("protected_manifest", "protected_manifest_sha256"),
        ("protected_validation", "protected_validation_sha256"),
        ("measurement_view_jsonl", "measurement_view_jsonl_sha256"),
    ):
        if sha256(Path(config[path_key])) != str(config[hash_key]):
            raise RuntimeError(f"frozen data artifact mismatch: {config[path_key]}")
    qwen_adapter = Path(config["qwen_adapter"])
    if sha256(qwen_adapter / "adapter_model.safetensors") != str(
        config["qwen_adapter_model_sha256"]
    ):
        raise RuntimeError("Qwen adapter changed after manifest freeze")
    if sha256(qwen_adapter / "adapter_config.json") != str(
        config["qwen_adapter_config_sha256"]
    ):
        raise RuntimeError("Qwen adapter config changed after manifest freeze")
    for collection in ("qwen_base_model_freeze", "source_freeze"):
        for raw_path, expected in config[collection].items():
            path = Path(raw_path)
            if not path.is_file() or sha256(path) != str(expected):
                raise RuntimeError(f"frozen file mismatch: {path}")
    rows = read_jsonl(manifest)
    if len(rows) != int(config["case_count"]):
        raise RuntimeError("frozen manifest case count mismatch")
    for case in rows:
        for reference in case["input_references"]:
            path = Path(reference["path"])
            if not path.is_file() or sha256(path) != reference["sha256"]:
                raise RuntimeError(f"frozen image asset mismatch: {path}")
        validate_v12_decision(
            case["ground_truth"]["decision"], image_bindings(case)
        )
    return rows


def generate_attempt(
    case: Mapping[str, Any],
    prompt_messages: list[dict[str, Any]],
    *,
    processor: Any,
    model: Any,
    dependencies: Mapping[str, Any],
    qwen_config: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "example_id": case["case_id"],
        "group_id": case["group_id"],
        "category": case["route"] or case["task"],
        "prompt": prompt_messages,
        "images": [item["path"] for item in case["input_references"]],
    }
    return generate_batch(
        [row],
        image_root=Path("/"),
        processor=processor,
        model=model,
        deps=dependencies,
        config=qwen_config,
    )[0]


def validation_error(
    generated: Mapping[str, Any], images: Mapping[str, Path]
) -> str | None:
    if generated["parsed_json"] is None:
        return f"JSON parse failure: {generated['parse_error']}"
    try:
        validate_v12_decision(generated["parsed_json"], images)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def qwen_decision(
    case: Mapping[str, Any],
    *,
    processor: Any,
    model: Any,
    dependencies: Mapping[str, Any],
    qwen_config: Mapping[str, Any],
    repair: bool,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    messages = list(case["prompt"])
    attempts = [
        generate_attempt(
            case,
            messages,
            processor=processor,
            model=model,
            dependencies=dependencies,
            qwen_config=qwen_config,
        )
    ]
    error = validation_error(attempts[-1], image_bindings(case))
    if error is not None and repair:
        messages = [
            *messages,
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": attempts[-1]["raw_prediction_text"]}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"The v12 decision was rejected: {error}. Correct only the "
                            "contract error. Use the registered route's exact argument groups "
                            "and only the image roles actually attached. Return only valid "
                            "qwen_orchestration_decision_v12_v1 JSON; never invent values."
                        ),
                    }
                ],
            },
        ]
        attempts.append(
            generate_attempt(
                case,
                messages,
                processor=processor,
                model=model,
                dependencies=dependencies,
                qwen_config=qwen_config,
            )
        )
        error = validation_error(attempts[-1], image_bindings(case))
    return attempts[-1]["parsed_json"], attempts, error


def _within(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    tolerance_reference: Mapping[str, Any] | None = None,
) -> bool:
    av = np.asarray([float(actual[field]) for field in OUTPUT_FIELDS])
    ev = np.asarray([float(expected[field]) for field in OUTPUT_FIELDS])
    reference = expected if tolerance_reference is None else tolerance_reference
    return bool(np.all(np.abs(av - ev) <= tolerance_vector(reference)))


def specialist_success(
    case: Mapping[str, Any], outcome: Mapping[str, Any] | None
) -> bool:
    if outcome is None or outcome.get("status") != "ready" or not outcome.get("executed"):
        return False
    route = str(case["route"])
    if outcome.get("route_name") != route:
        return False
    result = outcome["result"]
    truth = case["ground_truth"]
    if route == "measure_beam_profile_v12":
        return _within(result["beam_state"], truth["beam_state"])
    if "direction" in route:
        return result["directions"] == truth["directions"]
    if "forward" in route:
        return _within(
            result["predicted_next_beam_state"],
            truth["next_beam_state"],
            tolerance_reference=truth["current_beam_state"],
        )
    if "inverse" in route:
        episode = result["episode"]
        return bool(
            result["planner_backend"] == "learned_h1_cem"
            and episode["success"]
            and int(episode["illegal_actions"]) == 0
            and int(episode["steps"]) <= 5
        )
    return False


def direction_field_correctness(
    case: Mapping[str, Any], outcome: Mapping[str, Any] | None
) -> dict[str, bool] | None:
    if "direction" not in str(case["route"]):
        return None
    fields = case["ground_truth"]["directions"]
    if (
        outcome is None
        or outcome.get("route_name") != case["route"]
        or "directions" not in outcome.get("result", {})
    ):
        return {field: False for field in OUTPUT_FIELDS}
    predicted = outcome["result"]["directions"]
    return {field: predicted.get(field) == fields[field] for field in OUTPUT_FIELDS}


def _numbers_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _numbers_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _numbers_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(
        right, (int, float)
    ) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-10)
    return left == right


def canonical_argument_checks(
    predicted: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    adapter: V12Adapter,
) -> dict[str, bool]:
    if predicted is None or predicted.get("status") != "ready":
        return {"arguments": False, "unit": False, "image_roles": False}
    pargs = predicted.get("arguments", {})
    targs = target["arguments"]
    roles = predicted.get("image_roles") == target["image_roles"]
    try:
        converted_p: dict[str, Any] = {}
        converted_t: dict[str, Any] = {}
        for output, value in ((converted_p, pargs), (converted_t, targs)):
            if "setup_context" in value:
                output["setup_context"] = adapter.canonicalize_setup(value["setup_context"])
            if "actuator_position" in value:
                output["actuator_position"] = adapter.canonicalize_position(value["actuator_position"])
            if "current_beam_state" in value:
                output["current_beam_state"] = adapter.canonicalize_state(value["current_beam_state"], "current_beam_state")
            if "target_beam_state" in value:
                output["target_beam_state"] = adapter.canonicalize_state(value["target_beam_state"], "target_beam_state")
            if "continuous_action" in value:
                position = output.get("actuator_position", {field: 0.0 for field in POSITION_FIELDS})
                output["continuous_action"] = adapter.canonicalize_action(value["continuous_action"], position)
            if "image_calibration" in value:
                output["image_calibration"] = dict(value["image_calibration"])
        arguments = _numbers_equal(converted_p, converted_t)
        unit = (
            "continuous_action" not in targs
            or _numbers_equal(
                converted_p.get("continuous_action"), converted_t.get("continuous_action")
            )
        )
    except Exception:
        arguments = unit = False
    return {"arguments": arguments, "unit": unit, "image_roles": roles}


def measurement_component_success(
    case: Mapping[str, Any], outcome: Mapping[str, Any] | None
) -> bool | None:
    if "image" not in str(case["route"]) and case["route"] != "measure_beam_profile_v12":
        return None
    if outcome is None or outcome.get("status") != "ready":
        return False
    result = outcome["result"]
    if case["route"] == "measure_beam_profile_v12":
        return _within(result["beam_state"], case["ground_truth"]["beam_state"])
    measurements = []
    if "measurement" in result:
        measurements.append((result["measurement"]["beam_state"], case["ground_truth"]["current_beam_state"]))
    if "current_measurement" in result:
        measurements.append((result["current_measurement"]["beam_state"], case["ground_truth"]["current_beam_state"]))
    if "target_measurement" in result:
        measurements.append((result["target_measurement"]["beam_state"], case["ground_truth"]["next_beam_state"]))
    return bool(measurements and all(_within(actual, target) for actual, target in measurements))


def measurement_component_errors(
    case: Mapping[str, Any], outcome: Mapping[str, Any] | None
) -> dict[str, dict[str, float]] | None:
    if "image" not in str(case["route"]) and case["route"] != "measure_beam_profile_v12":
        return None
    if outcome is None or outcome.get("status") != "ready":
        return {}
    result = outcome["result"]
    pairs: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    if case["route"] == "measure_beam_profile_v12" and "beam_state" in result:
        pairs.append(("beam", result["beam_state"], case["ground_truth"]["beam_state"]))
    if "measurement" in result:
        pairs.append(("current", result["measurement"]["beam_state"], case["ground_truth"]["current_beam_state"]))
    if "current_measurement" in result:
        pairs.append(("current", result["current_measurement"]["beam_state"], case["ground_truth"]["current_beam_state"]))
    if "target_measurement" in result:
        pairs.append(("target", result["target_measurement"]["beam_state"], case["ground_truth"]["next_beam_state"]))
    output: dict[str, dict[str, float]] = {}
    for label, actual, expected in pairs:
        tolerance = tolerance_vector(expected)
        output[label] = {
            field: abs(float(actual[field]) - float(expected[field])) / float(tolerance[index])
            for index, field in enumerate(OUTPUT_FIELDS)
        }
    return output


def state_counterpart(
    case: Mapping[str, Any], target: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Build the exact state-route counterpart used to isolate v12 downstream quality."""

    route = str(case["route"])
    mapping = {
        "predict_direction_from_image_v12": "predict_direction_from_state_v12",
        "predict_forward_from_image_v12": "predict_forward_from_state_v12",
        "inverse_control_from_images_v12_h1": "inverse_control_from_states_v12_h1",
    }
    state_route = mapping.get(route)
    if state_route is None:
        return None
    arguments = {
        key: value
        for key, value in target["arguments"].items()
        if key != "image_calibration"
    }
    arguments["current_beam_state"] = dict(case["ground_truth"]["current_beam_state"])
    if "inverse" in state_route:
        arguments["target_beam_state"] = dict(case["ground_truth"]["next_beam_state"])
    state_decision = {
        **target,
        "route_name": state_route,
        "arguments": arguments,
        "image_roles": {},
    }
    state_case = {**case, "route": state_route, "modality": "state"}
    return state_case, state_decision


def classify_failure(row: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    if row["e2e_success"]:
        return None, []
    if row["qwen_parse_error"] or row["qwen_validation_error"]:
        return "Qwen invalid JSON/schema", []
    if not row["status_correct"]:
        return "wrong status", []
    if not row["route_correct"] or not row["task_correct"]:
        return "wrong task/route", []
    if not row["reason_correct"]:
        return "missing/wrong argument", []
    if not row["image_roles_correct"]:
        return "image-role error", []
    if not row["unit_conversion_correct"]:
        return "unit/sign/order error", []
    if not row["arguments_correct"]:
        return "missing/wrong argument", []
    if row["dispatch_error"]:
        text = str(row["dispatch_error"])
        if "hash mismatch" in text or "bounds mismatch" in text:
            return "checkpoint/config mismatch", []
        if "ContractError" in text:
            return "adapter validation", []
        return "simulator/runtime", []
    if row["measurement_success"] is False:
        return "measurement specialist", []
    route = str(row["target_route"])
    if "inverse" in route:
        if row["actuator_violations"]:
            return "actuator boundary", []
        if row["planner_exploitation_events"]:
            return "v12 forward prediction", ["H1 CEM search/planning"]
        return "H1 CEM search/planning", []
    if "direction" in route:
        return "v12 forward prediction", ["direction thresholding"]
    if "forward" in route:
        return "v12 forward prediction", []
    if route == "measure_beam_profile_v12":
        return "measurement specialist", []
    return "unknown", []


def wilson(successes: int, count: int) -> tuple[float | None, float | None]:
    if count == 0:
        return None, None
    z = 1.959963984540054
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def route_rows(results: list[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    output: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        if row["target_status"] == "ready":
            output[(row["target_task"], row["modality"])].append(row)
    return output


def four_task_macro(results: list[Mapping[str, Any]], key: str) -> float:
    grouped = route_rows(results)
    task_values = []
    for task in PHYSICAL_TASKS:
        modality_values = [
            statistics.mean(float(row[key]) for row in rows)
            for (candidate, _), rows in grouped.items()
            if candidate == task and rows
        ]
        task_values.append(statistics.mean(modality_values))
    return statistics.mean(task_values)


def bootstrap_macro(
    results: list[Mapping[str, Any]], key: str, *, seed: int, samples: int = 5000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    grouped = route_rows(results)
    values = []
    for _ in range(samples):
        resampled = []
        for rows in grouped.values():
            indices = rng.integers(0, len(rows), size=len(rows))
            resampled.extend(rows[int(index)] for index in indices)
        values.append(four_task_macro(resampled, key))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def summarize(
    results: list[dict[str, Any]], run_dir: Path, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    grouped = route_rows(results)
    metric_rows: list[dict[str, Any]] = []
    for (task, modality), rows in sorted(grouped.items()):
        n = len(rows)
        specialist = sum(bool(row["specialist_success"]) for row in rows)
        e2e = sum(bool(row["e2e_success"]) for row in rows)
        low, high = wilson(e2e, n)
        metric_rows.append(
            {
                "task": task,
                "modality": modality,
                "n": n,
                "specialist_accuracy": specialist / n,
                "e2e_accuracy": e2e / n,
                "qwen_contract_gap": (specialist - e2e) / n,
                "e2e_wilson_95_low": low,
                "e2e_wilson_95_high": high,
            }
        )
    specialist_macro = four_task_macro(results, "specialist_success")
    e2e_macro = four_task_macro(results, "e2e_success")
    macro_ci = bootstrap_macro(results, "e2e_success", seed=2026073111)
    ready = [row for row in results if row["target_status"] == "ready"]
    clarification = [row for row in results if row["target_status"] == "needs_clarification"]
    unsupported = [row for row in results if row["target_status"] == "unsupported"]
    latencies = [float(row["total_latency_seconds"]) for row in results]
    inverse = [row for row in ready if row["target_task"] == "inverse_control_v12"]
    routed_ready = [row for row in ready if row["route_correct"] and row["task_correct"]]
    direction = [row for row in ready if row["target_task"] == "direction_prediction_v12"]
    baseline_direction_fields = [
        value
        for row in direction
        for value in (row["baseline_direction_field_correct"] or {}).values()
    ]
    e2e_direction_fields = [
        value
        for row in direction
        for value in (row["e2e_direction_field_correct"] or {}).values()
    ]
    modality_summary = {}
    for modality in ("state", "image"):
        rows = [row for row in metric_rows if row["modality"] == modality]
        modality_summary[modality] = {
            "specialist_route_balanced_accuracy": statistics.mean(
                float(row["specialist_accuracy"]) for row in rows
            ),
            "e2e_route_balanced_accuracy": statistics.mean(
                float(row["e2e_accuracy"]) for row in rows
            ),
        }
    failure_by_route_stage: dict[str, dict[str, int]] = defaultdict(dict)
    for row in results:
        if row["primary_failure_stage"]:
            route = str(row["target_route"] or row["target_status"])
            stage = str(row["primary_failure_stage"])
            failure_by_route_stage[route][stage] = failure_by_route_stage[route].get(stage, 0) + 1
    inverse_attribution = Counter()
    for row in inverse:
        if row["specialist_success"]:
            continue
        if row["baseline_measurement_success"] is False:
            inverse_attribution["measurement"] += 1
        elif int(row["baseline_planner_exploitation_events"] or 0) > 0:
            inverse_attribution["v12_model_prediction"] += 1
        else:
            inverse_attribution["h1_search_planning"] += 1
    summary = {
        "run_id": run_dir.name,
        "case_count": len(results),
        "ready_count": len(ready),
        "four_task_specialist_macro_accuracy": specialist_macro,
        "four_task_e2e_macro_accuracy": e2e_macro,
        "four_task_qwen_contract_gap": specialist_macro - e2e_macro,
        "four_task_e2e_stratified_bootstrap_95": {"low": macro_ci[0], "high": macro_ci[1], "samples": 5000},
        "ready_micro_specialist_accuracy": statistics.mean(float(row["specialist_success"]) for row in ready),
        "ready_micro_e2e_accuracy": statistics.mean(float(row["e2e_success"]) for row in ready),
        "schema_validity": statistics.mean(float(row["schema_valid"]) for row in results),
        "route_accuracy": statistics.mean(float(row["route_correct"]) for row in ready),
        "argument_extraction_accuracy": statistics.mean(float(row["arguments_correct"]) for row in ready),
        "argument_extraction_given_correct_route": statistics.mean(float(row["arguments_correct"]) for row in routed_ready) if routed_ready else None,
        "unit_conversion_accuracy": statistics.mean(float(row["unit_conversion_correct"]) for row in ready),
        "clarification_accuracy": statistics.mean(float(row["e2e_success"]) for row in clarification) if clarification else None,
        "unsupported_rejection_accuracy": statistics.mean(float(row["e2e_success"]) for row in unsupported) if unsupported else None,
        "image_measurement_specialist_success": statistics.mean(float(row["baseline_measurement_success"]) for row in ready if row["baseline_measurement_success"] is not None),
        "image_measurement_e2e_success": statistics.mean(float(row["e2e_measurement_success"]) for row in ready if row["e2e_measurement_success"] is not None),
        "image_v12_downstream_state_success": statistics.mean(float(row["downstream_v12_success"]) for row in ready if row["downstream_v12_success"] is not None),
        "direction_per_field_specialist_accuracy": statistics.mean(float(value) for value in baseline_direction_fields),
        "direction_per_field_e2e_accuracy": statistics.mean(float(value) for value in e2e_direction_fields),
        "modality_route_balanced": modality_summary,
        "latency_seconds": {
            "average": statistics.mean(latencies),
            "p50": float(np.quantile(latencies, 0.5)),
            "p95": float(np.quantile(latencies, 0.95)),
        },
        "inverse": {
            "specialist_success_rate": statistics.mean(float(row["specialist_success"]) for row in inverse),
            "e2e_success_rate": statistics.mean(float(row["e2e_success"]) for row in inverse),
            "specialist_average_steps": statistics.mean(float(row["baseline_inverse_steps"] or 0) for row in inverse),
            "e2e_average_steps_when_executed": statistics.mean(float(row["e2e_inverse_steps"]) for row in inverse if row["e2e_inverse_steps"] is not None) if any(row["e2e_inverse_steps"] is not None for row in inverse) else None,
            "specialist_actuator_violations": sum(int(row["baseline_actuator_violations"] or 0) for row in inverse),
            "e2e_actuator_violations": sum(int(row["e2e_actuator_violations"] or 0) for row in inverse),
            "specialist_failure_attribution": dict(inverse_attribution),
        },
        "by_task_modality": metric_rows,
        "failure_counts": dict(Counter(row["primary_failure_stage"] for row in results if row["primary_failure_stage"])),
        "failure_by_route_stage": dict(sorted(failure_by_route_stage.items())),
        "elapsed_seconds": metadata["elapsed_seconds"],
    }
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
        writer.writerow(
            {
                "task": "four_task_macro",
                "modality": "balanced",
                "n": len(ready),
                "specialist_accuracy": specialist_macro,
                "e2e_accuracy": e2e_macro,
                "qwen_contract_gap": specialist_macro - e2e_macro,
                "e2e_wilson_95_low": macro_ci[0],
                "e2e_wilson_95_high": macro_ci[1],
            }
        )
    failures = [row for row in results if row["primary_failure_stage"]]
    failure_fields = [
        "case_id", "target_route", "modality", "primary_failure_stage", "secondary_failure_tags", "qwen_validation_error", "dispatch_error"
    ]
    with (run_dir / "failure_analysis.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=failure_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in failure_fields} for row in failures)
    (run_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report(run_dir, summary, metadata)
    return summary


def report(run_dir: Path, summary: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    table = [
        "| Task | Modality | N | Specialist accuracy | E2E accuracy | Qwen/contract gap | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_task_modality"]:
        table.append(
            f"| {row['task']} | {row['modality']} | {row['n']} | {row['specialist_accuracy']:.2%} | "
            f"{row['e2e_accuracy']:.2%} | {row['qwen_contract_gap']:.2%} | "
            f"[{row['e2e_wilson_95_low']:.2%}, {row['e2e_wilson_95_high']:.2%}] |"
        )
    ci = summary["four_task_e2e_stratified_bootstrap_95"]
    table.append(
        f"| Four-task macro | Balanced | {summary['ready_count']} | "
        f"{summary['four_task_specialist_macro_accuracy']:.2%} | "
        f"{summary['four_task_e2e_macro_accuracy']:.2%} | "
        f"{summary['four_task_qwen_contract_gap']:.2%} | [{ci['low']:.2%}, {ci['high']:.2%}] |"
    )
    failures = summary["failure_counts"]
    ordered_failures = sorted(failures.items(), key=lambda item: (-item[1], item[0]))
    bottleneck = ordered_failures[0][0] if ordered_failures else "none"
    priorities = {
        "Qwen invalid JSON/schema": "adapt Qwen explicitly to the v12 JSON contract and exact route argument groups",
        "wrong status": "strengthen ready versus clarification versus unsupported classification",
        "wrong task/route": "improve v12 task and state/image route classification",
        "missing/wrong argument": "improve exact field copying and clarification reason extraction",
        "unit/sign/order error": "add unit-preserving extraction examples without changing the deterministic adapter",
        "image-role error": "improve current/target image-role grounding",
        "measurement specialist": "improve or recalibrate the guarded measurement specialist on protected image regimes",
        "v12 forward prediction": "increase forward-model data/quality in a future separately trained checkpoint",
        "direction thresholding": "audit near-threshold delta calibration while keeping the locked criterion",
        "H1 CEM search/planning": "study H1 CEM search coverage without enabling H3",
        "actuator boundary": "improve constrained H1 sampling near absolute limits",
        "simulator/runtime": "repair the recorded simulator/runtime failure before another run",
        "adapter validation": "correct the v12 contract input at the adapter boundary",
        "checkpoint/config mismatch": "restore the exact frozen artifact set",
        "unknown": "inspect the retained per-case outcome and logs",
    }
    top_priorities = [priorities.get(name, name) for name, _ in ordered_failures[:3]]
    modality = summary["modality_route_balanced"]
    state_image_e2e_gap = (
        modality["state"]["e2e_route_balanced_accuracy"]
        - modality["image"]["e2e_route_balanced_accuracy"]
    )
    inverse_attribution = summary["inverse"]["specialist_failure_attribution"]
    conditional_argument_text = (
        "n/a"
        if summary["argument_extraction_given_correct_route"] is None
        else f"{summary['argument_extraction_given_correct_route']:.2%}"
    )
    limitations = (
        f"This is a single-seed diagnostic with {metadata['ready_per_route']} ready cases per route. The Qwen adapter was trained on the v1 route contract, not v12, and is not formally promoted. "
        "The v12 checkpoint used only 128 training groups; its preregistered H3 accumulation gate failed, so H3 was excluded. Image-route assets were deterministically materialized from protected numerical test states because the protected v12 manifest stored no images."
    )
    text = f"""# Qwen + continuous-action v12 evaluation report

Run ID: `{run_dir.name}`  
Start: `{metadata['start_timestamp']}`  
End: `{metadata['end_timestamp']}`  
Elapsed: `{metadata['elapsed_seconds']:.1f}` seconds  
Cases: `{summary['case_count']}` (`{summary['ready_count']}` ready)

## Main results

{chr(10).join(table)}

The v12 correct-route specialist four-task macro accuracy is **{summary['four_task_specialist_macro_accuracy']:.2%}**. Qwen + contract + adapter + v12 E2E macro accuracy is **{summary['four_task_e2e_macro_accuracy']:.2%}**. The measured Qwen/contract gap is **{summary['four_task_qwen_contract_gap']:.2%}**.

Ready micro specialist/E2E accuracy is {summary['ready_micro_specialist_accuracy']:.2%}/{summary['ready_micro_e2e_accuracy']:.2%}. Schema validity is {summary['schema_validity']:.2%}, routing accuracy {summary['route_accuracy']:.2%}, argument extraction {summary['argument_extraction_accuracy']:.2%} overall and {conditional_argument_text} conditional on a correct route, and unit conversion {summary['unit_conversion_accuracy']:.2%}. Clarification and unsupported rejection accuracy are {summary['clarification_accuracy']:.2%}/{summary['unsupported_rejection_accuracy']:.2%}.

Direction per-field specialist/E2E accuracy is {summary['direction_per_field_specialist_accuracy']:.2%}/{summary['direction_per_field_e2e_accuracy']:.2%}; the table uses strict all-five accuracy. Route-balanced state/image E2E accuracy is {modality['state']['e2e_route_balanced_accuracy']:.2%}/{modality['image']['e2e_route_balanced_accuracy']:.2%}, a state-minus-image gap of {state_image_e2e_gap:.2%}.

Image measurement strict success is {summary['image_measurement_specialist_success']:.2%} on the correct-route specialist and {summary['image_measurement_e2e_success']:.2%} after Qwen routing. The ground-truth-state v12 downstream counterpart succeeds on {summary['image_v12_downstream_state_success']:.2%}; per-image normalized measurement errors, the direct-state downstream outcome, and the full image E2E outcome are all retained per case.

Inverse Learned H1 specialist/E2E success is {summary['inverse']['specialist_success_rate']:.2%}/{summary['inverse']['e2e_success_rate']:.2%}, specialist mean steps {summary['inverse']['specialist_average_steps']:.2f}, and specialist/E2E actuator violations {summary['inverse']['specialist_actuator_violations']}/{summary['inverse']['e2e_actuator_violations']}. Correct-route inverse failure attribution is `{json.dumps(inverse_attribution, ensure_ascii=False)}`; model-prediction attribution requires recorded planner-exploitation evidence, otherwise a valid H1 miss is assigned to search/planning.

Latency average/P50/P95 is {summary['latency_seconds']['average']:.2f}/{summary['latency_seconds']['p50']:.2f}/{summary['latency_seconds']['p95']:.2f} seconds.

## Failure localization

Primary failure counts: `{json.dumps(dict(ordered_failures), ensure_ascii=False)}`. The largest observed stage is **{bottleneck}**. Route-by-stage equivalent confusion statistics are `{json.dumps(summary['failure_by_route_stage'], ensure_ascii=False)}`. Failures are assigned exactly one primary stage; secondary tags are retained in `failure_analysis.csv`.

## Acceptance and interpretation

Integration acceptance passed before this frozen run: schema/registry validation, unit/order/bounds/no-op tests, state/image v12 dispatch, a non-trivial Learned H1 closed-loop episode, checkpoint/config integrity, and old-route regressions. Formal cases were then run without changing code, config, checkpoint, or manifest.

The routing-versus-argument split is given by route accuracy ({summary['route_accuracy']:.2%}) versus conditional argument accuracy ({conditional_argument_text}). State/image and specialist/E2E gaps are quantified above and in the table. Inverse failures tagged `v12 forward prediction` had planner-exploitation evidence; remaining valid-planner misses are attributed to H1 search/planning.

Top priorities from the three largest observed stages are: {'; '.join(top_priorities) or 'none observed'}.

## Known limitations

{limitations}

Exact reproduction:

```bash
/home/jiamo/miniconda3/envs/optics_qlora/bin/python -m Qwen_orchestration.scripts.evaluate_v12_e2e --run-dir {run_dir}
```
"""
    (run_dir / "evaluation_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    # Fail before creating any formal outputs if a runtime dependency needed by
    # the guarded image measurement path is absent from the Qwen environment.
    for dependency in ("torch", "scipy"):
        try:
            importlib.import_module(dependency)
        except ModuleNotFoundError as error:
            raise RuntimeError(
                f"formal runtime dependency is missing: {dependency}"
            ) from error
    started = time.perf_counter()
    start_timestamp = datetime.now(timezone.utc).isoformat()
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    for output in (
        "case_results.jsonl",
        "metrics.csv",
        "failure_analysis.csv",
        "evaluation_report.md",
        "evaluation.log",
    ):
        if (run_dir / output).exists():
            raise RuntimeError(f"refusing to overwrite formal output: {run_dir / output}")
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.qwen_config.resolve() != Path(run_config["qwen_config"]).resolve():
        raise RuntimeError("--qwen-config differs from the frozen Qwen config")
    if sha256(args.qwen_config.resolve()) != str(run_config["qwen_config_sha256"]):
        raise RuntimeError("Qwen config changed after manifest freeze")
    cases = validate_freeze(run_dir, run_config)
    if args.max_cases is not None:
        if args.max_cases < 1:
            raise ValueError("--max-cases must be positive")
        cases = cases[: args.max_cases]
    qwen_config = load_config(args.qwen_config.resolve())
    processor, model, dependencies, qwen_adapter = load_model(qwen_config, None)
    adapter = V12Adapter()
    runtime = V12OrchestrationRuntime(adapter)
    results: list[dict[str, Any]] = []
    log_path = run_dir / "evaluation.log"
    with log_path.open("w", encoding="utf-8") as log:
        for index, case in enumerate(cases):
            case_started = time.perf_counter()
            target = case["ground_truth"]["decision"]
            bindings = image_bindings(case)
            baseline = None
            baseline_error = None
            try:
                baseline = runtime.dispatch(
                    target,
                    bindings,
                    run_id=f"{run_dir.name}:{case['case_id']}:baseline",
                    execution_context=case["execution_context"],
                )
            except Exception as error:
                baseline_error = f"{type(error).__name__}: {error}"
            downstream_outcome = None
            downstream_error = None
            counterpart = state_counterpart(case, target)
            if counterpart is not None:
                state_case, state_decision = counterpart
                try:
                    downstream_outcome = runtime.dispatch(
                        state_decision,
                        {},
                        run_id=f"{run_dir.name}:{case['case_id']}:downstream",
                        execution_context=case["execution_context"],
                    )
                except Exception as error:
                    downstream_error = f"{type(error).__name__}: {error}"
            predicted, attempts, qwen_error = qwen_decision(
                case,
                processor=processor,
                model=model,
                dependencies=dependencies,
                qwen_config=qwen_config,
                repair=not args.no_repair,
            )
            e2e_outcome = None
            dispatch_error = None
            if qwen_error is None and predicted is not None and predicted["status"] == "ready":
                try:
                    e2e_outcome = runtime.dispatch(
                        predicted,
                        bindings,
                        run_id=f"{run_dir.name}:{case['case_id']}:e2e",
                        execution_context=case["execution_context"],
                    )
                except Exception as error:
                    dispatch_error = f"{type(error).__name__}: {error}"
            status_correct = predicted is not None and predicted.get("status") == target["status"]
            task_correct = predicted is not None and predicted.get("task_type") == target["task_type"]
            route_correct = predicted is not None and predicted.get("route_name") == target["route_name"]
            reason_correct = predicted is not None and predicted.get("reason") == target["reason"]
            checks = canonical_argument_checks(predicted, target, adapter)
            schema_valid = qwen_error is None
            baseline_success = specialist_success(case, baseline) if target["status"] == "ready" else True
            e2e_specialist_success = specialist_success(case, e2e_outcome) if target["status"] == "ready" else False
            if target["status"] == "ready":
                e2e_success = bool(
                    schema_valid
                    and status_correct
                    and task_correct
                    and route_correct
                    and checks["arguments"]
                    and checks["unit"]
                    and checks["image_roles"]
                    and e2e_outcome is not None
                    and e2e_outcome.get("executed_backend") == "v12"
                    and e2e_specialist_success
                )
            elif target["status"] == "needs_clarification":
                e2e_success = bool(
                    schema_valid
                    and status_correct
                    and task_correct
                    and reason_correct
                    and set(predicted.get("missing_fields", [])) == set(target["missing_fields"])
                    and e2e_outcome is None
                )
            else:
                e2e_success = bool(
                    schema_valid
                    and status_correct
                    and task_correct
                    and reason_correct
                    and e2e_outcome is None
                )
            baseline_measurement_success = measurement_component_success(case, baseline)
            e2e_measurement_success = measurement_component_success(case, e2e_outcome)
            measurement_success = (
                e2e_measurement_success
                if e2e_outcome is not None
                else baseline_measurement_success
            )
            downstream_success = (
                specialist_success(state_case, downstream_outcome)
                if counterpart is not None
                else None
            )
            baseline_episode = (
                baseline["result"]["episode"]
                if baseline and "inverse" in str(case["route"])
                else None
            )
            e2e_episode = (
                e2e_outcome["result"]["episode"]
                if e2e_outcome
                and e2e_outcome.get("route_name") == case["route"]
                and "inverse" in str(case["route"])
                else None
            )
            episode = e2e_episode if e2e_episode is not None else baseline_episode
            baseline_direction = direction_field_correctness(case, baseline)
            e2e_direction = direction_field_correctness(case, e2e_outcome)
            row = {
                "case_id": case["case_id"],
                "group_id": case["group_id"],
                "target_status": target["status"],
                "target_task": target["task_type"],
                "target_route": target["route_name"],
                "modality": case["modality"],
                "schema_valid": schema_valid,
                "status_correct": status_correct,
                "task_correct": task_correct,
                "route_correct": route_correct,
                "reason_correct": reason_correct,
                "arguments_correct": checks["arguments"] if target["status"] == "ready" else True,
                "unit_conversion_correct": checks["unit"] if target["status"] == "ready" else True,
                "image_roles_correct": checks["image_roles"] if target["status"] == "ready" else True,
                "specialist_success": baseline_success,
                "e2e_specialist_success": e2e_specialist_success,
                "e2e_success": e2e_success,
                "measurement_success": measurement_success,
                "baseline_measurement_success": baseline_measurement_success,
                "e2e_measurement_success": e2e_measurement_success,
                "baseline_measurement_normalized_errors": measurement_component_errors(case, baseline),
                "e2e_measurement_normalized_errors": measurement_component_errors(case, e2e_outcome),
                "downstream_v12_success": downstream_success,
                "downstream_outcome": downstream_outcome,
                "downstream_error": downstream_error,
                "specialist_called_e2e": e2e_outcome is not None,
                "qwen_parse_error": attempts[-1]["parse_error"],
                "qwen_validation_error": qwen_error,
                "qwen_attempts": attempts,
                "predicted_decision": predicted,
                "baseline_outcome": baseline,
                "e2e_outcome": e2e_outcome,
                "baseline_error": baseline_error,
                "dispatch_error": dispatch_error,
                "baseline_direction_field_correct": baseline_direction,
                "e2e_direction_field_correct": e2e_direction,
                "baseline_inverse_steps": None if baseline_episode is None else int(baseline_episode["steps"]),
                "e2e_inverse_steps": None if e2e_episode is None else int(e2e_episode["steps"]),
                "baseline_actuator_violations": None if baseline_episode is None else int(baseline_episode["illegal_actions"]),
                "e2e_actuator_violations": None if e2e_episode is None else int(e2e_episode["illegal_actions"]),
                "baseline_planner_exploitation_events": None if baseline_episode is None else int(baseline_episode.get("planner_exploitation_events", 0)),
                "inverse_steps": 0 if episode is None else int(episode["steps"]),
                "actuator_violations": 0 if episode is None else int(episode["illegal_actions"]),
                "planner_exploitation_events": 0 if episode is None else int(episode.get("planner_exploitation_events", 0)),
                "total_latency_seconds": time.perf_counter() - case_started,
                "primary_failure_stage": None,
                "secondary_failure_tags": [],
            }
            primary, secondary = classify_failure(row)
            row["primary_failure_stage"] = primary
            row["secondary_failure_tags"] = secondary
            results.append(row)
            log.write(
                json.dumps(
                    {
                        "index": index,
                        "case_id": case["case_id"],
                        "e2e_success": e2e_success,
                        "specialist_success": baseline_success,
                        "failure": primary,
                        "latency_seconds": row["total_latency_seconds"],
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            log.flush()
            print(
                f"[{index+1}/{len(cases)}] {case['case_id']} specialist={int(baseline_success)} "
                f"e2e={int(e2e_success)} latency={row['total_latency_seconds']:.2f}s",
                flush=True,
            )
    write_jsonl(run_dir / "case_results.jsonl", results)
    end_timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {
        "version": "qwen_v12_evaluation_metadata_v1",
        "run_id": run_dir.name,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "elapsed_seconds": time.perf_counter() - started,
        "git_commit": git_value("rev-parse", "HEAD"),
        "working_tree_diff_summary": git_value("status", "--short"),
        "qwen_adapter": str(qwen_adapter),
        "qwen_config": str(args.qwen_config.resolve()),
        "checkpoint": str(adapter.checkpoint_path),
        "checkpoint_sha256": adapter.checkpoint_hash,
        "manifest_sha256": run_config["manifest_sha256"],
        "random_seeds": run_config["random_seeds"],
        "ready_per_route": run_config["ready_per_route"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "bitsandbytes", "lm-format-enforcer", "numpy", "scipy", "Pillow")
            },
        },
        "stochastic_repeats": 1,
        "scientific_claim": "single_seed_diagnostic",
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = summarize(results, run_dir, metadata)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
