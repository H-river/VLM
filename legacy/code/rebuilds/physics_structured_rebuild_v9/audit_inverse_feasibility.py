#!/usr/bin/env python3
"""Measure exhaustive 81-action inverse feasibility on existing requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v5.simulator_grid import simulate_fixed_action_grid
from optical_sim.src.experiment_generator import load_yaml as load_sim_yaml
from optical_sim.src.optical_elements import setup_from_dict
from optics_understanding_sft.direction_inverse_v1.build_inverse import (
    config_from_visible,
)

DEFAULT_CANONICAL = (
    REPO_ROOT.parent
    / "VLM_data/qwen_orchestration/v1/canonical/val.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "inverse_exhaustive_feasibility_audit.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_errors(
    state: Mapping[str, Any],
    desired: Mapping[str, Any],
) -> np.ndarray:
    centroid_distance = math.hypot(
        float(state["centroid_x_px"]) - float(desired["centroid_x_px"]),
        float(state["centroid_y_px"]) - float(desired["centroid_y_px"]),
    )
    peak_scale = max(abs(float(desired["peak_intensity"])), 1e-12)
    return np.asarray(
        [
            centroid_distance / 0.5,
            abs(float(state["sigma_x_px"]) - float(desired["sigma_x_px"])),
            abs(float(state["sigma_y_px"]) - float(desired["sigma_y_px"])),
            (
                abs(
                    float(state["peak_intensity"])
                    - float(desired["peak_intensity"])
                )
                / peak_scale
                / 0.02
            ),
        ],
        dtype=np.float64,
    )


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    canonical = args.canonical.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")

    rows = []
    with canonical.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("category") == "select_inverse_action_from_states_v1":
                rows.append(row)
    if not rows:
        raise ValueError("no state inverse requests found")
    group_ids = [str(row["group_id"]) for row in rows]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("state inverse requests must contain unique groups")

    base = load_sim_yaml(REPO_ROOT / "optical_sim/configs/base_config.yaml")
    records = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        arguments = row["target_decision"]["arguments"]
        setup_values = dict(arguments["setup"])
        desired = dict(arguments["desired_beam_state"])
        visible = {
            **setup_values,
            "sensor_resolution_px": [1024, 1024],
        }
        setup = setup_from_dict(config_from_visible(visible, base))
        simulated = simulate_fixed_action_grid(setup)
        errors = np.stack(
            [
                normalized_errors(item["state"], desired)
                for item in simulated
            ]
        )
        maximum_error = np.max(errors, axis=1)
        successful = maximum_error <= 1.0
        best_index = int(np.argmin(maximum_error))
        records.append(
            {
                "example_id": str(row["example_id"]),
                "group_id": str(row["group_id"]),
                "feasible": bool(np.any(successful)),
                "successful_action_count": int(np.sum(successful)),
                "best_action_index": best_index,
                "best_maximum_normalized_error": float(
                    maximum_error[best_index]
                ),
                "best_normalized_errors": {
                    "centroid_distance": float(errors[best_index, 0]),
                    "sigma_x": float(errors[best_index, 1]),
                    "sigma_y": float(errors[best_index, 2]),
                    "peak_intensity": float(errors[best_index, 3]),
                },
            }
        )
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(
                json.dumps(
                    {
                        "processed": index + 1,
                        "total": len(rows),
                        "feasible_so_far": int(
                            sum(record["feasible"] for record in records)
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    feasible = int(sum(record["feasible"] for record in records))
    successful_counts = [
        int(record["successful_action_count"]) for record in records
    ]
    best_scores = [
        float(record["best_maximum_normalized_error"])
        for record in records
    ]
    report = {
        "version": "inverse_exhaustive_feasibility_audit_v9",
        "evaluation_contract": {
            "action_count": 81,
            "centroid_distance_tolerance_px": 0.5,
            "sigma_x_tolerance_px": 1.0,
            "sigma_y_tolerance_px": 1.0,
            "peak_intensity_relative_tolerance": 0.02,
            "success_requires_all_conditions": True,
        },
        "unique_state_requests": len(records),
        "duplicated_state_plus_image_request_count": 2 * len(records),
        "feasible_unique_count": feasible,
        "feasible_unique_rate": feasible / len(records),
        "feasible_duplicated_count": 2 * feasible,
        "feasible_duplicated_rate": feasible / len(records),
        "successful_action_count": summarize(successful_counts),
        "best_maximum_normalized_error": summarize(best_scores),
        "records": records,
        "seconds": time.perf_counter() - started,
        "source_contract": {
            "canonical_validation": str(canonical),
            "canonical_validation_sha256": sha256(canonical),
            "generated_setups": 0,
            "generated_images": 0,
            "used_for_training": False,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "unique_count": len(records),
                "feasible_count": feasible,
                "feasible_rate": feasible / len(records),
                "seconds": report["seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
