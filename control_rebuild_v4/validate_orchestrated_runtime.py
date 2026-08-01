#!/usr/bin/env python3
"""Validate every frozen Qwen route against the candidate v4 execution overlay."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from measurement_rebuild_v3.train import metric_block
from Qwen_orchestration.runtime.numerics import legacy_to_sensor
from specialist_rebuild_v2.common import raw_state_array

DEFAULT_QWEN_DATA = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_CONTROL_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"
ROUTES = (
    "measure_beam_profile_v1",
    "predict_direction_from_state_v1",
    "predict_direction_from_image_v1",
    "predict_forward_from_state_v1",
    "predict_forward_from_image_v1",
    "select_inverse_action_from_states_v1",
    "select_inverse_action_from_images_v1",
)
ZERO_ACTION = {
    "lens_x_delta_mm": 0.0,
    "lens_y_delta_mm": 0.0,
    "camera_x_delta_mm": 0.0,
    "camera_y_delta_mm": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN_DATA)
    parser.add_argument("--control-run", type=Path, default=DEFAULT_CONTROL_RUN)
    parser.add_argument("--overlay-manifest", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def first_ready_record_by_route(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            decision = row["target_decision"]
            route = decision["route_name"]
            if (
                decision["status"] == "ready"
                and route in ROUTES
                and route not in records
            ):
                records[str(route)] = row
            if len(records) == len(ROUTES):
                break
    missing = set(ROUTES) - set(records)
    if missing:
        raise ValueError(f"canonical validation lacks ready routes: {sorted(missing)}")
    return records


def direct_measurement_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (
                row["category"] == "measure_beam_profile_v1"
                and row["target_decision"]["status"] == "ready"
            ):
                records.append(row)
    if not records:
        raise ValueError("canonical validation has no direct measurement records")
    return records


def source_cases(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    output = {str(row["group_id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError("private source cases contain duplicate group ids")
    return output


def validate_output(route: str, output: dict[str, Any]) -> None:
    if output.get("status") != "ready":
        raise ValueError(f"{route}: overlay did not return ready")
    if output.get("route_name") != route:
        raise ValueError(f"{route}: returned route differs from input route")
    if output.get("executed") is not True:
        raise ValueError(f"{route}: specialist was not executed")
    if output.get("specialist_generation") != "v4_candidate_overlay":
        raise ValueError(f"{route}: candidate generation marker is absent")
    result = output.get("result", {})
    if result.get("simulator_at_inference") is not False:
        raise ValueError(f"{route}: simulator-free inference invariant failed")
    json.dumps(output, allow_nan=False)


def main() -> None:
    args = parse_args()
    qwen_data = args.qwen_data.resolve()
    control_run = args.control_run.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else control_run / "orchestrated_runtime_validation.json"
    )
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if bool(existing.get("complete")):
            raise RuntimeError(f"refusing to overwrite completed validation: {output}")
    records = first_ready_record_by_route(qwen_data / "canonical/val.jsonl")
    measurement_records = direct_measurement_records(qwen_data / "canonical/val.jsonl")
    private_cases = source_cases(qwen_data / "private/source_cases/val.jsonl")
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
    started = time.perf_counter()
    route_results = []
    for route in ROUTES:
        row = records[route]
        available_images = {
            f"image_{index}": qwen_data / relative
            for index, relative in enumerate(row["images"])
        }
        executed = runtime.dispatch(row["target_decision"], available_images)
        validate_output(route, executed)
        result = executed["result"]
        route_results.append(
            {
                "route_name": route,
                "example_id": row["example_id"],
                "passed": True,
                "model_version": result.get("model_version"),
                "simulator_at_inference": result["simulator_at_inference"],
                "output_fields": sorted(result),
            }
        )
    measurement_target = []
    measurement_prediction = []
    measurement_sources: dict[str, int] = {}
    for row in measurement_records:
        source = private_cases[str(row["group_id"])]
        target_sensor = legacy_to_sensor(
            source["current_beam_state"],
            source["setup"],
            ZERO_ACTION,
        )
        executed = runtime.dispatch(
            row["target_decision"],
            {"image_0": qwen_data / row["images"][0]},
        )
        validate_output("measure_beam_profile_v1", executed)
        measured = executed["result"]
        source_name = str(measured["measurement_source"])
        measurement_sources[source_name] = measurement_sources.get(source_name, 0) + 1
        measurement_target.append(raw_state_array(target_sensor))
        measurement_prediction.append(raw_state_array(measured["beam_state"]))
    direct_metrics = metric_block(
        np.asarray(measurement_target, dtype=np.float32),
        np.asarray(measurement_prediction, dtype=np.float32),
    )
    result = {
        "validation_version": "qwen_v1_to_specialists_v4_contract_validation",
        "scope": (
            "one canonical in-domain ready request per registered route; "
            "this validates execution contracts, not task accuracy"
        ),
        "device": str(device),
        "overlay_manifest": str(manifest_path),
        "routes_expected": len(ROUTES),
        "routes_passed": len(route_results),
        "all_routes_passed": len(route_results) == len(ROUTES),
        "simulator_inference_calls": 0,
        "direct_measurement_validation": {
            "scope": (
                "all clean Qwen in-domain measurement validation images using "
                "only the frozen four-field public calibration contract"
            ),
            "count": len(measurement_records),
            "measurement_source_counts": measurement_sources,
            **direct_metrics,
        },
        "route_results": route_results,
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
