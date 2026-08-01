#!/usr/bin/env python3
"""Test improved forward enumeration with the frozen v8 inverse ranker."""

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
    DEFAULT_FIELDWISE_IMAGE_HYBRID,
    DEFAULT_FIELDWISE_STATE_HYBRID,
    DEFAULT_RESIDUAL_FORWARD_IMAGE,
    DEFAULT_RESIDUAL_FORWARD_STATE,
    DEFAULT_TRANSFORMER_INVERSE_V8,
    OrchestratedBestForwardInverseRuntimeV9,
)

DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "orchestrated_best_forward_inverse_v9_validation.json"
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
        OrchestratedBestForwardInverseRuntimeV9
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
        "saved_qwen_plus_best_v9_with_residual_forward_inverse_enumeration"
    )
    report["scope"] = (
        "best accumulated v9 candidate, with numerical-state inverse "
        "candidate states changed from frozen v5 forward to calibrated "
        "residual forward v9; v8 inverse ranker remains frozen"
    )
    artifact_paths = {
        "fieldwise_state_direction_v9": DEFAULT_FIELDWISE_STATE_HYBRID,
        "fieldwise_image_direction_v9": DEFAULT_FIELDWISE_IMAGE_HYBRID,
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
                "direction_all_five_exact": metrics[
                    "end_to_end_direction_physical_all_five_exact"
                ],
                "forward_strict_all_five": metrics[
                    "end_to_end_forward_physical_strict_all_five_success"
                ],
                "inverse_target_reached": metrics[
                    "end_to_end_inverse_target_reached_rate"
                ],
                "inverse_oracle_target_reached": metrics[
                    "oracle_inverse_target_reached_rate"
                ],
                "inverse_by_route": metrics["inverse_physical_by_route"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
