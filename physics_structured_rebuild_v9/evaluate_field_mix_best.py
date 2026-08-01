#!/usr/bin/env python3
"""Evaluate field-mixed direction with the accumulated best v9 system."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joint_forward_direction_v7.evaluate_system as base_evaluator
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_FIELD_MIX_IMAGE_HYBRID,
    DEFAULT_FIELD_MIX_STATE_HYBRID,
    DEFAULT_RESIDUAL_FORWARD_IMAGE,
    DEFAULT_RESIDUAL_FORWARD_STATE,
    DEFAULT_TRANSFORMER_INVERSE_V8,
    OrchestratedFieldMixBestForwardInverseRuntimeV9,
)

DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "orchestrated_field_mix_best_v9_validation.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    output = DEFAULT_OUTPUT.resolve()
    details = output.with_suffix(".details.jsonl")
    if output.exists() or details.exists():
        raise RuntimeError(f"refusing to overwrite completed output: {output}")
    original_class = base_evaluator.OrchestratedSharedRuntimeV7
    original_argv = sys.argv
    base_evaluator.OrchestratedSharedRuntimeV7 = (
        OrchestratedFieldMixBestForwardInverseRuntimeV9
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
    report["evaluation_version"] = (
        "saved_qwen_plus_field_mix_direction_and_best_forward_inverse_v9"
    )
    report["scope"] = (
        "direction uses route- and field-specific gates over a selected "
        "HistGradientBoosting and Extra Trees field mix; forward uses "
        "route-specific residual-tree corrections; numerical-state inverse "
        "uses the frozen v8 ranker with residual-forward enumeration; visual "
        "inverse, measurement, Qwen, and all deployed defaults remain frozen"
    )
    artifact_paths = {
        "field_mix_state_direction_v9": DEFAULT_FIELD_MIX_STATE_HYBRID,
        "field_mix_image_direction_v9": DEFAULT_FIELD_MIX_IMAGE_HYBRID,
        "residual_state_forward_v9": DEFAULT_RESIDUAL_FORWARD_STATE,
        "residual_image_forward_v9": DEFAULT_RESIDUAL_FORWARD_IMAGE,
        "transformer_numerical_inverse_v8": DEFAULT_TRANSFORMER_INVERSE_V8,
    }
    for name, path in artifact_paths.items():
        report["artifacts"][name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path.resolve()),
        }
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = report["metrics"]
    print(
        json.dumps(
            {
                "output": str(output),
                "details": str(details),
                "record_count": report["record_count"],
                "direction_all_five_exact": metrics[
                    "end_to_end_direction_physical_all_five_exact"
                ],
                "direction_macro_f1": metrics[
                    "end_to_end_direction_physical_macro_f1"
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
                "measurement_strict_all_five": metrics[
                    "end_to_end_visual_measurement_strict_all_five_success"
                ],
                "forward_by_route": metrics["forward_physical_by_route"],
                "inverse_by_route": metrics["inverse_physical_by_route"],
                "direction_by_route": metrics[
                    "direction_physical_by_route"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
