#!/usr/bin/env python3
"""Evaluate targeted direction plus v8 numerical inverse in the full system."""

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
    DEFAULT_TARGETED_TREE_IMAGE_HYBRID,
    DEFAULT_TARGETED_TREE_STATE_HYBRID,
    DEFAULT_TRANSFORMER_INVERSE_V8,
    OrchestratedTargetedTreeV8InverseRuntimeV9,
)

DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "orchestrated_targeted_direction_v8_inverse_validation.json"
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
        OrchestratedTargetedTreeV8InverseRuntimeV9
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
        "saved_qwen_plus_targeted_direction_v9_plus_inverse_v8"
    )
    report["scope"] = (
        "direction uses the targeted-data v9 tree hybrid; numerical-state "
        "inverse uses the frozen v8 Set Transformer with frozen v5 forward; "
        "visual inverse, forward, measurement, and Qwen remain frozen"
    )
    report["artifacts"]["targeted_tree_state_direction_v9"] = {
        "path": str(DEFAULT_TARGETED_TREE_STATE_HYBRID.resolve()),
        "sha256": sha256(DEFAULT_TARGETED_TREE_STATE_HYBRID.resolve()),
    }
    report["artifacts"]["targeted_tree_image_direction_v9"] = {
        "path": str(DEFAULT_TARGETED_TREE_IMAGE_HYBRID.resolve()),
        "sha256": sha256(DEFAULT_TARGETED_TREE_IMAGE_HYBRID.resolve()),
    }
    report["artifacts"]["transformer_numerical_inverse_v8"] = {
        "path": str(DEFAULT_TRANSFORMER_INVERSE_V8.resolve()),
        "sha256": sha256(DEFAULT_TRANSFORMER_INVERSE_V8.resolve()),
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
