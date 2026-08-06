#!/usr/bin/env python3
"""Run the single final end-to-end replay for protected cycle candidates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joint_forward_direction_v7.evaluate_system as base_evaluator
from physics_structured_rebuild_v9.boundary_direction_runtime import sha256
from physics_structured_rebuild_v9.orchestrated_gated_cycle_runtime import (
    BOUNDARY_STATE_DIRECTION,
    INVERSE_SUCCESS_RANKER,
    OrchestratedGatedCycleRuntimeV9,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "orchestrated_gated_cycle_v9_validation.json"


def main() -> None:
    output = DEFAULT_OUTPUT.resolve()
    details = output.with_suffix(".details.jsonl")
    if output.exists() or details.exists():
        raise RuntimeError(f"refusing to overwrite completed output: {output}")
    original_class = base_evaluator.OrchestratedSharedRuntimeV7
    original_argv = sys.argv
    base_evaluator.OrchestratedSharedRuntimeV7 = (
        OrchestratedGatedCycleRuntimeV9
    )
    sys.argv = [
        original_argv[0],
        "--output",
        str(output),
        "--details",
        str(details),
    ]
    try:
        base_evaluator.main()
    finally:
        base_evaluator.OrchestratedSharedRuntimeV7 = original_class
        sys.argv = original_argv

    report = json.loads(output.read_text(encoding="utf-8"))
    report["evaluation_version"] = "saved_qwen_plus_gated_cycle_v9"
    report["scope"] = (
        "Accepted forward-selector system plus a protected boundary correction "
        "for state direction and a protected group-balanced ranker for "
        "numerical inverse. Image direction, direct forward, visual inverse, "
        "measurement, Qwen decisions, and all other routes remain frozen."
    )
    report["candidate_artifacts"] = {
        "boundary_state_direction": {
            "path": str(BOUNDARY_STATE_DIRECTION.resolve()),
            "sha256": sha256(BOUNDARY_STATE_DIRECTION.resolve()),
        },
        "inverse_success_ranker": {
            "path": str(INVERSE_SUCCESS_RANKER.resolve()),
            "sha256": sha256(INVERSE_SUCCESS_RANKER.resolve()),
        },
    }
    report["accepted_baseline"] = str(
        (
            DEFAULT_RUN
            / "orchestrated_forward_selector_ensemble_v9_validation.json"
        ).resolve()
    )
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = report["metrics"]
    print(
        json.dumps(
            {
                "output": str(output),
                "record_count": report["record_count"],
                "direction_all_five_exact": metrics[
                    "end_to_end_direction_physical_all_five_exact"
                ],
                "forward_strict_all_five": metrics[
                    "end_to_end_forward_physical_strict_all_five_success"
                ],
                "forward_correctly_routed": metrics[
                    "correctly_routed_forward_physical_strict_all_five_success"
                ],
                "inverse_target_reached": metrics[
                    "end_to_end_inverse_target_reached_rate"
                ],
                "inverse_correctly_routed": metrics[
                    "oracle_inverse_target_reached_rate"
                ],
                "measurement_strict_all_five": metrics[
                    "end_to_end_visual_measurement_strict_all_five_success"
                ],
                "direction_by_route": metrics[
                    "direction_physical_by_route"
                ],
                "inverse_by_route": metrics["inverse_physical_by_route"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
