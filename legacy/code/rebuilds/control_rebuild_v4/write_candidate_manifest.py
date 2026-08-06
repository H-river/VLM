#!/usr/bin/env python3
"""Write a hash-pinned Qwen-to-v4 candidate execution manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v4.orchestrated_runtime import (
    DIRECTION_V4_ROUTE_ARTIFACTS,
    DIRECTION_V4_ROUTE_BACKENDS,
    ROUTE_ARTIFACTS,
    ROUTE_BACKENDS,
)

DEFAULT_CONTROL_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"
DEFAULT_MEASUREMENT_V3_RUN = (
    REPO_ROOT.parent / "VLM_runs/measurement_rebuild_v3_one_seed"
)
DEFAULT_MEASUREMENT_V4_RUN = (
    REPO_ROOT.parent / "VLM_runs/measurement_rebuild_v4_one_seed"
)
QWEN_REGISTRY = REPO_ROOT / "Qwen_orchestration/configs/model_registry.yaml"
QWEN_SCHEMA = (
    REPO_ROOT / "Qwen_orchestration/schemas/orchestration_decision.schema.json"
)
QWEN_FREEZE = REPO_ROOT / "Qwen_orchestration/freeze/baseline_manifest.json"
QWEN_SELECTION = (
    REPO_ROOT
    / "Qwen_orchestration/results/v1/stage2_safe_runtime_v1/full_selection.json"
)
DIRECTION_ARTIFACT = (
    REPO_ROOT
    / "optics_understanding_sft/direction_inverse_v1/results"
    / "direction_small_v1/direction_small_ensemble.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--direction-artifact",
        type=Path,
        help="Optional balanced direction-v4 artifact; leaves v1 frozen if omitted.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned(path: Path) -> dict[str, Any]:
    value = path.resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return {
        "path": str(value),
        "size": value.stat().st_size,
        "sha256": sha256(value),
    }


def main() -> None:
    args = parse_args()
    control_run = args.control_run.resolve()
    if args.output is not None:
        output = args.output.resolve()
    elif args.direction_artifact is not None:
        output = control_run / "candidate_overlay_direction_v4_manifest.json"
    else:
        output = control_run / "candidate_overlay_manifest.json"
    freeze = json.loads(QWEN_FREEZE.read_text(encoding="utf-8"))
    selection = json.loads(QWEN_SELECTION.read_text(encoding="utf-8"))
    artifacts: dict[str, dict[str, Any]] = {
        "measurement_v3": pinned(
            args.measurement_v3_run.resolve() / "measurement_v3.pt"
        ),
        "measurement_calibrator_v4": pinned(
            args.measurement_v4_run.resolve() / "measurement_calibrator_v4.pt"
        ),
        "forward_v4": pinned(control_run / "forward_physics_residual_v4.pt"),
        "inverse_v4": pinned(control_run / "inverse_control_v4.pt"),
        "visual_scorer_v4": pinned(control_run / args.visual_scorer_name),
    }
    if args.direction_artifact is None:
        artifacts["direction_v1"] = pinned(DIRECTION_ARTIFACT)
        route_backends = ROUTE_BACKENDS
        route_artifacts = ROUTE_ARTIFACTS
        manifest_version = "qwen_to_specialists_v4_candidate_manifest"
        specialist_generation = "control_rebuild_v4_one_seed"
    else:
        artifacts["direction_v4"] = pinned(args.direction_artifact.resolve())
        route_backends = DIRECTION_V4_ROUTE_BACKENDS
        route_artifacts = DIRECTION_V4_ROUTE_ARTIFACTS
        manifest_version = (
            "qwen_to_specialists_v4_direction_candidate_manifest"
        )
        specialist_generation = "direction_rebuild_v4_quickcheck_one_seed"
    routes = {
        route: {
            "backend": route_backends[route],
            "artifacts": list(route_artifacts[route]),
        }
        for route in route_backends
    }
    manifest = {
        "manifest_version": manifest_version,
        "status": "validation_candidate",
        "frozen_qwen": {
            "freeze_id": freeze["freeze_id"],
            "checkpoint_step": 1000,
            "formally_promoted": selection["selected_checkpoint"] is not None,
            "registry": pinned(QWEN_REGISTRY),
            "decision_schema": pinned(QWEN_SCHEMA),
        },
        "specialist_generation": specialist_generation,
        "seed": 20260726,
        "action_grid_size": 81,
        "simulator_at_inference": False,
        "held_out_test_used_for_training_or_selection": 0,
        "direct_measurement_policy": {
            "learned_gamma_min": 0.75,
            "learned_gamma_max": 1.05,
            "out_of_range_backend": "calibrated_analytic_moments",
        },
        "artifacts": artifacts,
        "routes": routes,
        "complete": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
