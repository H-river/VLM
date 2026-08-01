#!/usr/bin/env python3
"""Diagnose measurement, forward, ranker, and visual-scorer inverse variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Qwen_orchestration.scripts.evaluate_end_to_end import (
    private_inverse_target_reached,
    read_jsonl,
)
from control_rebuild_v3.common import ACTION_GRID, tolerance_from_current
from control_rebuild_v3.train_forward import configure
from control_rebuild_v3.visual_inverse import sensor_to_base_legacy
from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_residual_forward_runtime_v9,
)
from physics_structured_rebuild_v9.inverse_runtime_v8 import (
    load_inverse_runtime_v8,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_COMBINED_NATURAL_INVERSE_V9,
    DEFAULT_NATURAL_GRID_FORWARD_STATE,
    DEFAULT_TRANSFORMER_INVERSE_V8,
)
from specialist_rebuild_v2.common import STATE_FIELDS, raw_state_array

DEFAULT_QWEN = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_OVERLAY = (
    REPO_ROOT.parent
    / "VLM_runs/direction_rebuild_v4_tree_one_seed"
    / "candidate_overlay_direction_v4_manifest.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "visual_inverse_modular_diagnostic.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--v4-overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument(
        "--natural-forward",
        type=Path,
        default=DEFAULT_NATURAL_GRID_FORWARD_STATE,
    )
    parser.add_argument(
        "--base-inverse",
        type=Path,
        default=DEFAULT_TRANSFORMER_INVERSE_V8,
    )
    parser.add_argument(
        "--adapted-inverse",
        type=Path,
        default=DEFAULT_COMBINED_NATURAL_INVERSE_V9,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def role_path(
    data_dir: Path,
    row: dict[str, Any],
    role: str,
) -> Path:
    image_roles = row["target_decision"]["image_roles"]
    image_name = str(image_roles[role])
    index = int(image_name.removeprefix("image_"))
    return (data_dir / str(row["images"][index])).resolve()


def exact_measurement_count(
    predicted: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    per_field = np.zeros((len(predicted), len(STATE_FIELDS)), dtype=np.bool_)
    for index in range(len(predicted)):
        tolerance = tolerance_from_current(
            {
                field: float(truth[index, field_index])
                for field_index, field in enumerate(STATE_FIELDS)
            }
        )
        per_field[index] = (
            np.abs(predicted[index] - truth[index]) <= tolerance
        )
    return per_field, np.all(per_field, axis=1)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    data_dir = args.qwen_data.resolve()
    canonical = [
        row
        for row in read_jsonl(data_dir / "canonical/val.jsonl")
        if row["target_decision"].get("route_name")
        == "select_inverse_action_from_images_v1"
    ]
    if len(canonical) != 150:
        raise ValueError("expected 150 visual inverse validation requests")
    private_by_group = {
        str(row["group_id"]): row
        for row in read_jsonl(data_dir / "private/source_cases/val.jsonl")
    }
    private_rows = [private_by_group[str(row["group_id"])] for row in canonical]
    setups = [row["target_decision"]["arguments"]["setup"] for row in canonical]

    torch, device = configure(int(args.seed), args.device)
    backend = OrchestratedSpecialistRuntimeV4.from_manifest(
        torch,
        args.v4_overlay.resolve(),
        device,
    )
    visual = backend.visual
    original_forward = visual.forward
    original_inverse = visual.inverse
    natural_forward, _ = load_residual_forward_runtime_v9(
        args.natural_forward.resolve(),
        torch,
        device,
    )
    base_inverse, _ = load_inverse_runtime_v8(
        args.base_inverse.resolve(),
        torch,
        device,
    )
    adapted_inverse, _ = load_inverse_runtime_v8(
        args.adapted_inverse.resolve(),
        torch,
        device,
    )

    current_sensor = []
    desired_sensor = []
    for index, row in enumerate(canonical):
        calibration = row["target_decision"]["arguments"]["image_calibration"]
        current = visual.measure_image(
            role_path(data_dir, row, "current_beam"),
            calibration,
        )
        desired = visual.measure_image(
            role_path(data_dir, row, "desired_beam"),
            calibration,
        )
        current_sensor.append(current["beam_state"])
        desired_sensor.append(desired["beam_state"])
        if (index + 1) % 25 == 0:
            print(
                json.dumps(
                    {"measured_requests": index + 1, "count": len(canonical)},
                    sort_keys=True,
                ),
                flush=True,
            )
    current_sensor_array = np.asarray(current_sensor, dtype=np.float32)
    desired_sensor_array = np.asarray(desired_sensor, dtype=np.float32)

    visual.forward = original_forward
    visual.inverse = original_inverse
    baseline = visual.predict_from_states(
        setups,
        current_sensor_array,
        desired_sensor_array,
        group_ids=[str(row["example_id"]) for row in canonical],
    )
    visual.forward = natural_forward
    visual.inverse = adapted_inverse
    natural = visual.predict_from_states(
        setups,
        current_sensor_array,
        desired_sensor_array,
        group_ids=[str(row["example_id"]) for row in canonical],
    )
    natural_base = base_inverse.score_requests(
        setups,
        np.asarray(natural["current_base_legacy"], dtype=np.float32),
        np.asarray(natural["desired_base_legacy"], dtype=np.float32),
        np.asarray(natural["candidate_legacy_states"], dtype=np.float32),
    )
    variants = {
        "v4_visual_scorer_v4_forward": np.asarray(
            baseline["selected_indices"],
            dtype=np.int64,
        ),
        "v4_numerical_ranker_v4_forward": np.asarray(
            baseline["numerical_selected_indices"],
            dtype=np.int64,
        ),
        "v4_visual_scorer_natural_forward": np.asarray(
            natural["selected_indices"],
            dtype=np.int64,
        ),
        "base_v8_ranker_natural_forward": np.asarray(
            natural_base["selected_indices"],
            dtype=np.int64,
        ),
        "adapted_v8_ranker_natural_forward": np.asarray(
            natural["numerical_selected_indices"],
            dtype=np.int64,
        ),
    }
    success_by_variant: dict[str, np.ndarray] = {}
    cached_success: dict[tuple[str, int], bool] = {}
    for name, selected in variants.items():
        success = []
        for index, action_index in enumerate(selected):
            group_id = str(canonical[index]["group_id"])
            key = (group_id, int(action_index))
            if key not in cached_success:
                private = private_rows[index]
                cached_success[key] = private_inverse_target_reached(
                    private["setup"],
                    ACTION_GRID[int(action_index)],
                    private["desired_beam_state"],
                )
            success.append(cached_success[key])
        success_by_variant[name] = np.asarray(success, dtype=np.bool_)
        print(
            json.dumps(
                {
                    "variant": name,
                    "success_count": int(success_by_variant[name].sum()),
                    "count": len(canonical),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    true_current = np.stack(
        [raw_state_array(row["current_beam_state"]) for row in private_rows]
    )
    true_desired = np.stack(
        [raw_state_array(row["desired_beam_state"]) for row in private_rows]
    )
    measured_current_legacy = sensor_to_base_legacy(
        current_sensor_array,
        setups,
    )
    measured_desired_legacy = sensor_to_base_legacy(
        desired_sensor_array,
        setups,
    )
    current_per_field, current_exact = exact_measurement_count(
        measured_current_legacy,
        true_current,
    )
    desired_per_field, desired_exact = exact_measurement_count(
        measured_desired_legacy,
        true_desired,
    )
    variant_metrics = {
        name: {
            "success_count": int(success.sum()),
            "success_rate": float(success.mean()),
        }
        for name, success in success_by_variant.items()
    }
    union = np.logical_or.reduce(list(success_by_variant.values()))
    report = {
        "version": "visual_inverse_modular_diagnostic_v9_one_seed",
        "count": len(canonical),
        "variants": variant_metrics,
        "all_variant_oracle_union": {
            "success_count": int(union.sum()),
            "success_rate": float(union.mean()),
        },
        "measurement": {
            "current_all_five_count": int(current_exact.sum()),
            "desired_all_five_count": int(desired_exact.sum()),
            "both_all_five_count": int((current_exact & desired_exact).sum()),
            "current_per_field_count": {
                field: int(current_per_field[:, index].sum())
                for index, field in enumerate(STATE_FIELDS)
            },
            "desired_per_field_count": {
                field: int(desired_per_field[:, index].sum())
                for index, field in enumerate(STATE_FIELDS)
            },
        },
        "source_contract": {
            "qwen_validation": str(
                (data_dir / "canonical/val.jsonl").resolve()
            ),
            "v4_overlay": str(args.v4_overlay.resolve()),
            "natural_forward": str(args.natural_forward.resolve()),
            "base_inverse": str(args.base_inverse.resolve()),
            "adapted_inverse": str(args.adapted_inverse.resolve()),
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
