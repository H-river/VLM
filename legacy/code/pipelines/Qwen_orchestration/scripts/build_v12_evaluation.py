#!/usr/bin/env python3
"""Build and freeze the group-disjoint Qwen + v12 evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    OUTPUT_FIELDS,
    POSITION_FIELDS,
    stable_seed,
    tolerance_vector,
)
from Qwen_orchestration.v12.adapter import V12Adapter
from Qwen_orchestration.v12.prompt_contract import prompt


ROUTES = (
    "measure_beam_profile_v12",
    "predict_direction_from_state_v12",
    "predict_direction_from_image_v12",
    "predict_forward_from_state_v12",
    "predict_forward_from_image_v12",
    "inverse_control_from_states_v12_h1",
    "inverse_control_from_images_v12_h1",
)

# Every executable contract that can change the meaning of a frozen case is
# hash-pinned.  The evaluator refuses to start when any entry drifts.
SOURCE_FREEZE_PATHS = (
    "Qwen_orchestration/configs/qwen25vl_3b_orchestrator_v12_eval.yaml",
    "Qwen_orchestration/v12/runtime_config.json",
    "Qwen_orchestration/v12/model_registry.yaml",
    "Qwen_orchestration/v12/orchestration_decision_v12.schema.json",
    "Qwen_orchestration/v12/orchestration_generation_v12.schema.json",
    "Qwen_orchestration/v12/adapter.py",
    "Qwen_orchestration/v12/dispatcher.py",
    "Qwen_orchestration/v12/prompt_contract.py",
    "Qwen_orchestration/scripts/build_v12_evaluation.py",
    "Qwen_orchestration/scripts/evaluate_v12_e2e.py",
    "Qwen_orchestration/scripts/smoke_v12_integration.py",
    "continuous_control_v12/contracts.py",
    "continuous_control_v12/world_model.py",
    "continuous_control_v12/mpc.py",
    "control_rebuild_v3/visual_inverse.py",
    "control_rebuild_v4/visual_runtime.py",
)

QWEN_BASE_MODEL = Path("/home/jiamo/HF_models/Qwen2.5-VL-3B-Instruct")
QWEN_BASE_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
TASK_BY_ROUTE = {
    "measure_beam_profile_v12": "measurement",
    "predict_direction_from_state_v12": "direction_prediction_v12",
    "predict_direction_from_image_v12": "direction_prediction_v12",
    "predict_forward_from_state_v12": "forward_prediction_v12",
    "predict_forward_from_image_v12": "forward_prediction_v12",
    "inverse_control_from_states_v12_h1": "inverse_control_v12",
    "inverse_control_from_images_v12_h1": "inverse_control_v12",
}
MODALITY_BY_ROUTE = {
    route: ("image" if "image" in route or route == "measure_beam_profile_v12" else "state")
    for route in ROUTES
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ready-per-route", type=int, default=25)
    parser.add_argument("--clarification", type=int, default=20)
    parser.add_argument("--unsupported", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026073107)
    parser.add_argument("--v12-split", choices=("development", "test"), default="test")
    parser.add_argument("--exclude-group", action="append", default=[])
    parser.add_argument("--exclude-measurement-group", action="append", default=[])
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def q4(
    values: Mapping[str, Any],
    source_fields: tuple[str, ...],
    *,
    unit: str = "mm",
) -> dict[str, Any]:
    scale = 1000.0 if unit in {"um", "µm", "μm"} else 1.0
    return {
        "values": {
            target: float(values[source] * scale)
            for target, source in zip(
                ("lens_x", "lens_y", "camera_x", "camera_y"),
                source_fields,
                strict=True,
            )
        },
        "unit": unit,
    }


def direction_truth(current: Mapping[str, Any], following: Mapping[str, Any]) -> dict[str, str]:
    delta = np.asarray(
        [float(following[field]) - float(current[field]) for field in OUTPUT_FIELDS]
    )
    normalized = delta / tolerance_vector(current)
    return {
        field: (
            "decrease" if normalized[index] < -1.0 else "increase" if normalized[index] > 1.0 else "unchanged"
        )
        for index, field in enumerate(OUTPUT_FIELDS)
    }


def _ordered_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[str(row["sampling"]["kind"])].append(row)
    for kind in by_kind:
        by_kind[kind].sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{kind}:{row['transition_id']}".encode()
            ).hexdigest()
        )
    priority = [
        "no_op",
        "axis_aligned",
        "paired",
        "multi_axis",
        "near_zero_fine",
        "sobol",
        "legacy_grid",
        "closed_loop_trajectory",
    ]
    output: list[dict[str, Any]] = []
    position = 0
    while any(by_kind.values()):
        kind = priority[position % len(priority)]
        position += 1
        if by_kind.get(kind):
            output.append(by_kind[kind].pop(0))
    return output


def render_gaussian(
    metrics_sensor: Mapping[str, Any],
    path: Path,
    *,
    gamma: float = 1.1,
    stored_resolution: int = 512,
    source_resolution: int = 1024,
    calibration_high: float | None = None,
) -> dict[str, Any]:
    coordinates = (
        (np.arange(stored_resolution, dtype=np.float64) + 0.5)
        * source_resolution
        / stored_resolution
        - 0.5
    )
    xx, yy = np.meshgrid(coordinates, coordinates)
    sx = max(float(metrics_sensor["sigma_x_px"]), 1e-6)
    sy = max(float(metrics_sensor["sigma_y_px"]), 1e-6)
    high = (
        float(metrics_sensor["peak_intensity"])
        if calibration_high is None
        else float(calibration_high)
    )
    if high <= 0.0 or float(metrics_sensor["peak_intensity"]) > high * (1.0 + 1e-12):
        raise ValueError("render calibration ceiling must cover the metric peak")
    linear = (float(metrics_sensor["peak_intensity"]) / high) * np.exp(
        -0.5
        * (
            np.square((xx - float(metrics_sensor["centroid_x_px"])) / sx)
            + np.square((yy - float(metrics_sensor["centroid_y_px"])) / sy)
        )
    )
    encoded = np.power(np.clip(linear, 0.0, 1.0), gamma)
    raw = np.rint(encoded * np.iinfo(np.uint16).max).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(raw).save(path)
    return {
        "linear_intensity_low": 0.0,
        "linear_intensity_high": high,
        "gamma": gamma,
        "source_sensor_resolution_px": [source_resolution, source_resolution],
    }


def lab_to_sensor(
    metrics: Mapping[str, Any], positions: Mapping[str, Any], setup: Mapping[str, Any]
) -> dict[str, float]:
    output = {field: float(metrics[field]) for field in OUTPUT_FIELDS}
    pitch_mm = float(setup["pixel_size_um"]) * 1e-3
    output["centroid_x_px"] -= float(positions["camera_x_mm"]) / pitch_mm
    output["centroid_y_px"] -= float(positions["camera_y_mm"]) / pitch_mm
    return output


def decision(
    route: str,
    arguments: Mapping[str, Any],
    image_roles: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "qwen_orchestration_decision_v12_v1",
        "status": "ready",
        "task_type": TASK_BY_ROUTE[route],
        "route_name": route,
        "arguments": dict(arguments),
        "image_roles": dict(image_roles),
        "missing_fields": [],
        "reason": None,
    }


def _state_text(value: Mapping[str, Any]) -> str:
    return canonical_json({field: float(value[field]) for field in OUTPUT_FIELDS})


def _setup_text(value: Mapping[str, Any]) -> str:
    return canonical_json(value)


def _position_text(value: Mapping[str, Any]) -> str:
    return canonical_json(value)


def build_v12_case(
    *,
    route: str,
    row: Mapping[str, Any],
    index: int,
    output_dir: Path,
    seed: int,
    split: str,
) -> dict[str, Any]:
    unit = ("mm", "um", "µm", "canonical")[index % 4]
    position = q4(row["positions_mm"], POSITION_FIELDS, unit="mm")
    action = q4(row["action_mm"], ACTION_FIELDS, unit=unit)
    setup = dict(row["setup_context"])
    base_args: dict[str, Any] = {
        "setup_context": setup,
        "actuator_position": position,
    }
    image_roles: dict[str, str] = {}
    input_refs: list[dict[str, Any]] = []
    if "direction" in route or "forward" in route:
        base_args["continuous_action"] = action
    if route.endswith("state_v12"):
        base_args["current_beam_state"] = dict(row["metrics"])
    if route == "inverse_control_from_states_v12_h1":
        base_args["current_beam_state"] = dict(row["metrics"])
        base_args["target_beam_state"] = dict(row["next_metrics"])

    if "image" in route:
        current_path = output_dir / "images" / f"{route}_{index:03d}_current.png"
        current_calibration = render_gaussian(row["metrics_sensor_frame"], current_path)
        base_args["image_calibration"] = current_calibration
        image_roles["current_beam"] = "image_0"
        input_refs.append(
            {"role": "current_beam", "path": str(current_path.resolve()), "sha256": sha256(current_path)}
        )
        if route == "inverse_control_from_images_v12_h1":
            target_path = output_dir / "images" / f"{route}_{index:03d}_target.png"
            target_sensor = lab_to_sensor(row["next_metrics"], row["positions_mm"], setup)
            # One shared calibration is a route contract. Use a common ceiling
            # while preserving each image's physical peak as its encoded amplitude.
            common = max(
                float(row["metrics"]["peak_intensity"]),
                float(row["next_metrics"]["peak_intensity"]),
            )
            current_calibration = render_gaussian(
                row["metrics_sensor_frame"], current_path, calibration_high=common
            )
            target_calibration = render_gaussian(
                target_sensor, target_path, calibration_high=common
            )
            if target_calibration != current_calibration:
                raise RuntimeError("inverse images did not receive one shared calibration")
            # The common-ceiling render replaces the initial current image, so
            # freeze the final bytes rather than the provisional render.
            input_refs[0]["sha256"] = sha256(current_path)
            base_args["image_calibration"] = current_calibration
            image_roles["target_beam"] = "image_1"
            input_refs.append(
                {"role": "target_beam", "path": str(target_path.resolve()), "sha256": sha256(target_path)}
            )

    if "direction" in route:
        asks = (
            "Classify the change of every beam metric as increase, decrease, or unchanged",
            "Determine the five v12 metric directions after this move",
            "Report the signed direction category for each next-state metric",
            "Route this continuous move to the five-field direction predictor",
        )
    elif "forward" in route:
        asks = (
            "Predict the next five-metric beam state",
            "Estimate all five beam metrics after applying this continuous move",
            "Run one-step v12 forward prediction for the requested action",
            "Return the continuous-control next state in the locked metric order",
        )
    else:
        asks = (
            "Use Learned H1 CEM closed-loop inverse control to reach the target beam",
            "Drive the current beam toward the target with the registered H1 controller",
            "Run the v12 one-step-replanning inverse route for this target",
            "Select the Learned H1 CEM closed-loop inverse specialist",
        )
    ask = asks[index % len(asks)]
    modality = MODALITY_BY_ROUTE[route]
    details = [
        f"{ask} using continuous-control v12.",
        f"setup_context={_setup_text(setup)}",
        f"actuator_position={_position_text(position)}",
    ]
    if "direction" in route or "forward" in route:
        details.append(f"continuous_action={canonical_json(action)}")
    if modality == "state":
        details.append(f"current_beam_state={_state_text(row['metrics'])}")
        if "inverse" in route:
            details.append(f"target_beam_state={_state_text(row['next_metrics'])}")
    else:
        details.append(
            "image_calibration=" + canonical_json(base_args["image_calibration"])
        )
        details.append(
            "The first image is current_beam."
            + (" The second image is target_beam." if "inverse" in route else "")
        )
    target = decision(route, base_args, image_roles)
    ground_truth = {
        "decision": target,
        "current_beam_state": dict(row["metrics"]),
        "next_beam_state": dict(row["next_metrics"]),
        "directions": direction_truth(row["metrics"], row["next_metrics"]),
    }
    return {
        "case_id": f"ready_{route}_{index:03d}",
        "group_id": str(row["group_id"]),
        "task": TASK_BY_ROUTE[route],
        "route": route,
        "modality": modality,
        "prompt": prompt("\n".join(details), len(input_refs)),
        "input_references": input_refs,
        "canonical_arguments": base_args,
        "ground_truth": ground_truth,
        "scoring_rule": (
            "strict_all_five_direction"
            if "direction" in route
            else "strict_all_five_forward_tolerance"
            if "forward" in route
            else "strict_all_five_inverse_within_5_steps"
        ),
        "random_seed": stable_seed(seed, route, index),
        "data_split": split,
        "provenance": {
            "dataset": "corrected_128_16_16_v2",
            "transition_id": row["transition_id"],
            "generator_version": row["generator_version"],
        },
        "execution_context": {
            "simulator_fixed": row["simulator_fixed"],
            "planner_seed": stable_seed(seed, row["transition_id"], "matched_h1"),
        },
    }


def build_measurement_case(
    row: Mapping[str, Any], index: int, seed: int, split: str
) -> dict[str, Any]:
    root = Path("/home/jiamo/VLM_data/measurement_rebuild_v3")
    image_path = (root / str(row["base_image"])).resolve()
    calibration = {
        "linear_intensity_low": float(row["image_calibration"]["linear_intensity_low"]),
        "linear_intensity_high": float(row["image_calibration"]["linear_intensity_high"]),
        "gamma": 1.0,
        "source_sensor_resolution_px": list(row["image_calibration"]["source_sensor_resolution_px"]),
    }
    target = decision(
        "measure_beam_profile_v12",
        {"image_calibration": calibration},
        {"beam": "image_0"},
    )
    asks = (
        "Measure the five-metric beam profile in the attached calibrated image.",
        "Extract all five locked beam metrics from this calibrated frame.",
        "Route the attached beam image through the measurement specialist.",
        "Return the calibrated centroid, widths, and peak intensity for this beam.",
    )
    text = (
        asks[index % len(asks)] + "\n"
        f"image_calibration={canonical_json(calibration)}\n"
        "The attached image has role beam."
    )
    return {
        "case_id": f"ready_measure_beam_profile_v12_{index:03d}",
        "group_id": str(row["group_id"]),
        "task": "measurement",
        "route": "measure_beam_profile_v12",
        "modality": "image",
        "prompt": prompt(text, 1),
        "input_references": [
            {"role": "beam", "path": str(image_path), "sha256": sha256(image_path)}
        ],
        "canonical_arguments": target["arguments"],
        "ground_truth": {"decision": target, "beam_state": dict(row["target_state"])},
        "scoring_rule": "strict_all_five_measurement_tolerance",
        "random_seed": stable_seed(seed, "measurement", index),
        "data_split": split,
        "provenance": {
            "dataset": "measurement_rebuild_v3_one_seed",
            "state_id": row["state_id"],
            "view_id": row["view_id"],
        },
        "execution_context": {},
    }


def nonready_cases(clarification: int, unsupported: int, seed: int) -> list[dict[str, Any]]:
    clarification_templates = [
        ("Classify v12 direction for [0.03,0,0,0]; the action's unit was not stated.", ["continuous_action.unit"], "missing_unit", "direction_prediction_v12"),
        ("Predict the next v12 state from the supplied setup and state, but the continuous move is absent.", ["continuous_action"], "missing_action", "forward_prediction_v12"),
        ("Start H1 inverse control from this current state; no desired beam state is present.", ["target_beam_state"], "missing_target", "inverse_control_v12"),
        ("Use the image forward route, although no current beam image was attached.", ["current_beam"], "missing_image", "forward_prediction_v12"),
        ("Find v12 directions for a 30 um lens-x move; setup parameters were not supplied.", ["setup_context"], "missing_setup", "direction_prediction_v12"),
        ("The action is lens-x minus 2e-2 for v12 direction, with no unit metadata.", ["continuous_action.unit"], "missing_unit", "direction_prediction_v12"),
        ("I gave setup and current metrics for a one-step forecast but omitted the action vector.", ["continuous_action"], "missing_action", "forward_prediction_v12"),
        ("Please run state-based v12 inverse control; the target metrics are missing.", ["target_beam_state"], "missing_target", "inverse_control_v12"),
        ("Forecast from a current beam frame that I forgot to include.", ["current_beam"], "missing_image", "forward_prediction_v12"),
        ("Apply a -10 μm camera-y move for direction prediction without any optical setup.", ["setup_context"], "missing_setup", "direction_prediction_v12"),
        ("A four-value continuous direction action is provided, but its scale is unspecified.", ["continuous_action.unit"], "missing_unit", "direction_prediction_v12"),
        ("Run v12 forward prediction; setup and state are known, action is not.", ["continuous_action"], "missing_action", "forward_prediction_v12"),
        ("Move the current state toward a desired state, but I have not supplied that desired state.", ["target_beam_state"], "missing_target", "inverse_control_v12"),
        ("Use image-to-next-state v12 with no image input available.", ["current_beam"], "missing_image", "forward_prediction_v12"),
        ("For a 0.01 mm camera-x action, classify direction; setup_context is unavailable.", ["setup_context"], "missing_setup", "direction_prediction_v12"),
        ("Interpret [3e-2,0,0,0] as a direction action, but ask me which unit first.", ["continuous_action.unit"], "missing_unit", "direction_prediction_v12"),
        ("Estimate the next beam state after an action that was never specified.", ["continuous_action"], "missing_action", "forward_prediction_v12"),
        ("Use Learned H1 from the current metrics, with the destination metrics omitted.", ["target_beam_state"], "missing_target", "inverse_control_v12"),
        ("I requested image-based forward prediction but supplied zero images.", ["current_beam"], "missing_image", "forward_prediction_v12"),
        ("Predict directions for 20 um camera-x motion without wavelength or the rest of the setup.", ["setup_context"], "missing_setup", "direction_prediction_v12"),
    ]
    unsupported_templates = [
        ("Plan three steps jointly with H3 rather than the registered one-step controller.", "h3_forbidden"),
        ("Fine-tune the v12 ensemble on these evaluation examples before scoring.", "training_request"),
        ("Command an extra mirror_z actuator by 100 um.", "unsupported_actuator"),
        ("Discard the pinned weights and initialize a random substitute checkpoint.", "unsafe_checkpoint_request"),
        ("Return a Jones-vector polarization prediction for the laser.", "unsupported_task"),
        ("Switch inverse control to a horizon-three objective.", "h3_forbidden"),
        ("Retrain the forward specialist using protected test groups.", "training_request"),
        ("Add a fifth delta component for a mirror tilt axis.", "unsupported_actuator"),
        ("Overwrite the frozen model artifact with arbitrary weights.", "unsafe_checkpoint_request"),
        ("Infer phase and polarization, which are outside the beam-state contract.", "unsupported_task"),
        ("Use multi-step H3 MPC for the target images.", "h3_forbidden"),
        ("Modify model structure and optimize it against this manifest.", "training_request"),
        ("Move an unregistered stage_z actuator.", "unsupported_actuator"),
        ("Replace the verified checkpoint because I want different answers.", "unsafe_checkpoint_request"),
        ("Predict the optical field's complex phase map.", "unsupported_task"),
    ]
    rows: list[dict[str, Any]] = []
    for index in range(clarification):
        text, missing, reason, task_type = clarification_templates[index % len(clarification_templates)]
        target = {
            "schema_version": "qwen_orchestration_decision_v12_v1",
            "status": "needs_clarification",
            "task_type": task_type,
            "route_name": None,
            "arguments": {},
            "image_roles": {},
            "missing_fields": missing,
            "reason": reason,
        }
        rows.append(
            {
                "case_id": f"clarification_{index:03d}",
                "group_id": f"clarification_{index:03d}",
                "task": "needs_clarification",
                "route": None,
                "modality": "text",
                "prompt": prompt(text),
                "input_references": [],
                "canonical_arguments": {},
                "ground_truth": {"decision": target, "reason_category": reason},
                "scoring_rule": "status_and_missing_field_category",
                "random_seed": stable_seed(seed, "clarification", index),
                "data_split": "frozen_contract_probe",
                "provenance": {"dataset": "held_out_template_families_v12_v1"},
                "execution_context": {},
            }
        )
    for index in range(unsupported):
        text, reason = unsupported_templates[index % len(unsupported_templates)]
        target = {
            "schema_version": "qwen_orchestration_decision_v12_v1",
            "status": "unsupported",
            "task_type": None,
            "route_name": None,
            "arguments": {},
            "image_roles": {},
            "missing_fields": [],
            "reason": reason,
        }
        rows.append(
            {
                "case_id": f"unsupported_{index:03d}",
                "group_id": f"unsupported_{index:03d}",
                "task": "unsupported",
                "route": None,
                "modality": "text",
                "prompt": prompt(text),
                "input_references": [],
                "canonical_arguments": {},
                "ground_truth": {"decision": target, "reason_category": reason},
                "scoring_rule": "unsupported_rejection_without_specialist",
                "random_seed": stable_seed(seed, "unsupported", index),
                "data_split": "frozen_contract_probe",
                "provenance": {"dataset": "held_out_template_families_v12_v1"},
                "execution_context": {},
            }
        )
    return rows


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    if args.ready_per_route < 1 or args.clarification < 0 or args.unsupported < 0:
        raise ValueError("case counts must be non-negative and ready-per-route positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True)
    adapter = V12Adapter()
    protected_root = (
        REPO_ROOT / adapter.runtime_config["protected_data"]["root"]
    ).resolve()
    test_path = protected_root / f"transitions/{args.v12_split}.jsonl"
    if args.v12_split == "test" and sha256(test_path) != adapter.runtime_config["protected_data"]["test_jsonl_sha256"]:
        raise RuntimeError("protected v12 test JSONL hash mismatch")
    excluded = set(args.exclude_group)
    excluded_measurement = set(args.exclude_measurement_group)
    protected_manifest = protected_root / "manifest.json"
    protected_validation = protected_root / "validation.json"
    if sha256(protected_manifest) != adapter.runtime_config["protected_data"]["manifest_sha256"]:
        raise RuntimeError("protected v12 manifest hash mismatch")
    validation = json.loads(protected_validation.read_text(encoding="utf-8"))
    if not validation.get("complete") or int(validation.get("cross_split_group_overlap", -1)) != 0:
        raise RuntimeError("protected v12 data is not complete and group-disjoint")
    if validation["reports"][args.v12_split]["jsonl_sha256"] != sha256(test_path):
        raise RuntimeError("protected v12 split hash differs from validation report")
    v12_rows = _ordered_rows(
        [row for row in read_jsonl(test_path) if str(row["group_id"]) not in excluded],
        args.seed,
    )
    image_eligible = [
        row
        for row in v12_rows
        if all(
            96.0 < float(state[field]) < 928.0
            for state in (
                row["metrics_sensor_frame"],
                lab_to_sensor(row["next_metrics"], row["positions_mm"], row["setup_context"]),
            )
            for field in ("centroid_x_px", "centroid_y_px")
        )
    ]
    if len(v12_rows) < args.ready_per_route or len(image_eligible) < args.ready_per_route:
        raise RuntimeError("protected test data lacks enough eligible state/image cases")
    measurement_split = "test_iid" if args.v12_split == "test" else "val"
    measurement_rows = [
        row
        for row in read_jsonl(
            Path(f"/home/jiamo/VLM_data/measurement_rebuild_v3/views/{measurement_split}.jsonl")
        )
        if row["condition"] == "clean" and str(row["group_id"]) not in excluded_measurement
    ]
    measurement_root = Path("/home/jiamo/VLM_data/measurement_rebuild_v3")
    measurement_view_path = measurement_root / f"views/{measurement_split}.jsonl"
    train_measurement_groups = {
        str(row["group_id"])
        for row in read_jsonl(measurement_root / "views/train.jsonl")
    }
    selected_measurement_groups = {str(row["group_id"]) for row in measurement_rows}
    if train_measurement_groups & selected_measurement_groups:
        raise RuntimeError("measurement evaluation groups overlap measurement training groups")
    measurement_rows.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}:{row['view_id']}".encode()
        ).hexdigest()
    )
    if len(measurement_rows) < args.ready_per_route:
        raise RuntimeError("protected measurement test data lacks enough clean cases")

    cases: list[dict[str, Any]] = []
    for route in ROUTES:
        if route == "measure_beam_profile_v12":
            cases.extend(
                build_measurement_case(row, index, args.seed, measurement_split)
                for index, row in enumerate(measurement_rows[: args.ready_per_route])
            )
            continue
        source = image_eligible if "image" in route else v12_rows
        cases.extend(
            build_v12_case(
                route=route,
                row=row,
                index=index,
                output_dir=output_dir,
                seed=args.seed,
                split=args.v12_split,
            )
            for index, row in enumerate(source[: args.ready_per_route])
        )
    cases.extend(nonready_cases(args.clarification, args.unsupported, args.seed))
    expected = len(ROUTES) * args.ready_per_route + args.clarification + args.unsupported
    if len(cases) != expected or len({row["case_id"] for row in cases}) != expected:
        raise RuntimeError("evaluation manifest cardinality/ID check failed")

    manifest_path = output_dir / "frozen_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in cases:
            stream.write(canonical_json(row) + "\n")
    manifest_hash = sha256(manifest_path)
    (output_dir / "frozen_manifest.sha256").write_text(
        f"{manifest_hash}  frozen_manifest.jsonl\n", encoding="utf-8"
    )
    source_freeze = {
        str((REPO_ROOT / relative).resolve()): sha256((REPO_ROOT / relative).resolve())
        for relative in SOURCE_FREEZE_PATHS
    }
    qwen_base_freeze = {
        str((QWEN_BASE_MODEL / filename).resolve()): sha256(
            (QWEN_BASE_MODEL / filename).resolve()
        )
        for filename in QWEN_BASE_FILES
    }
    qwen_adapter = Path(
        "/home/jiamo/VLM_runs/qwen_orchestrator_v1_stage2_schema_refinement_v2_seed20260724/checkpoint-1000"
    )
    run_config = {
        "version": "qwen_v12_evaluation_run_config_v2",
        "frozen": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "case_count": len(cases),
        "ready_per_route": args.ready_per_route,
        "clarification_count": args.clarification,
        "unsupported_count": args.unsupported,
        "seed": args.seed,
        "git_commit": git_value("rev-parse", "HEAD"),
        "working_tree_diff_summary": git_value("status", "--short"),
        "checkpoint": str(adapter.checkpoint_path),
        "checkpoint_sha256": adapter.checkpoint_hash,
        "model_config_sha256": adapter.model_config_hash,
        "normalization_config_sha256": adapter.normalization_config_hash,
        "planner_config": adapter.planner_config,
        "planner_config_sha256": adapter.planner_config_hash,
        "runtime_config": str(adapter.runtime_config_path),
        "runtime_config_sha256": sha256(adapter.runtime_config_path),
        "qwen_config": str(
            (REPO_ROOT / "Qwen_orchestration/configs/qwen25vl_3b_orchestrator_v12_eval.yaml").resolve()
        ),
        "qwen_config_sha256": source_freeze[
            str((REPO_ROOT / "Qwen_orchestration/configs/qwen25vl_3b_orchestrator_v12_eval.yaml").resolve())
        ],
        "qwen_adapter": str(qwen_adapter),
        "qwen_adapter_model_sha256": sha256(qwen_adapter / "adapter_model.safetensors"),
        "qwen_adapter_config_sha256": sha256(qwen_adapter / "adapter_config.json"),
        "qwen_base_model": str(QWEN_BASE_MODEL),
        "qwen_base_model_freeze": qwen_base_freeze,
        "source_freeze": source_freeze,
        "protected_test_jsonl": str(test_path),
        "protected_test_jsonl_sha256": sha256(test_path),
        "protected_manifest": str(protected_manifest),
        "protected_manifest_sha256": sha256(protected_manifest),
        "protected_validation": str(protected_validation),
        "protected_validation_sha256": sha256(protected_validation),
        "measurement_view_jsonl": str(measurement_view_path),
        "measurement_view_jsonl_sha256": sha256(measurement_view_path),
        "v12_split": args.v12_split,
        "excluded_groups": sorted(excluded),
        "excluded_measurement_groups": sorted(excluded_measurement),
        "python": sys.version,
        "platform": platform.platform(),
        "random_seeds": {"manifest": args.seed, "per_case": "stable sha256-derived"},
    }
    run_config_path = output_dir / "run_config.json"
    run_config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    audit = f"""# V12 Qwen integration audit

- Legacy registry remains isolated at `Qwen_orchestration/configs/model_registry.yaml` with seven v1 routes and the unchanged 81-action contract.
- V12 registry contains seven distinct ready routes: one measurement, state/image direction, state/image forward, and state/image inverse H1.
- V12 action order is `{list(ACTION_FIELDS)}` in canonical mm. Bounds are lens +/-0.05 mm and camera +/-0.02 mm per step; all four absolute positions are limited to +/-3 mm by the repository sampling-domain contract.
- V12 model input is 35 canonical features: eight setup values, four positions, five state values, four normalized actions, four action squares, six pairwise action products, and four position-action products. The adapter calls the checkpoint runtime's canonical `structured_features` path; Qwen never emits these engineered features.
- Output order is `{list(OUTPUT_FIELDS)}`. Targets and decoded deltas use current-state tolerances `[1 px, 1 px, 2 px, 2 px, 5% peak with 1e-6 floor]`.
- Checkpoint: `{adapter.checkpoint_path}` (`{adapter.checkpoint_hash}`), three MLP members, model config hash `{adapter.model_config_hash}`, normalization hash `{adapter.normalization_config_hash}`.
- Auxiliary order confirmed by runtime: log captured power, clipping fraction, camera-boundary logit, actuator-limit logit.
- Measurement backend is guarded measurement v4: measurement v3 plus v4 calibrator inside gamma [0.75, 1.05], otherwise calibrated analytic moments. Metric order and tolerance match v12.
- Learned H1 CEM is frozen to population 256, 32 elites, 5 iterations, horizon 1, max 5 closed-loop steps, strict all-five max normalized error <= 1. Seed is fixed per case. H3 is not registered.
- Protected v12 test data contains 16 groups, has zero train/development overlap, and is hash-pinned. The original manifest did not store images; this freeze deterministically materializes image-route assets from protected test states before any formal result is observed.
- The Qwen checkpoint is an evaluation candidate rather than a formally promoted v12-trained adapter; this limitation is retained for the final report.
- The manifest, Qwen/v12 checkpoints, Qwen base-model shards, schemas, registries, prompts, adapters, measurement code, continuous-control runtime, and evaluation scripts are all SHA-256 pinned in `run_config.json`; the evaluator checks every hash before model loading.
"""
    (output_dir / "audit.md").write_text(audit, encoding="utf-8")
    os.chmod(manifest_path, 0o444)
    os.chmod(run_config_path, 0o444)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "cases": len(cases),
                "manifest_sha256": manifest_hash,
                "ready_per_route": args.ready_per_route,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
