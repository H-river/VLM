#!/usr/bin/env python3
"""Write a hash-pinned manifest for the evaluated shared-v6 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/joint_forward_direction_v6_one_seed"
)
DEFAULT_V5_RUN = (
    REPO_ROOT.parent / "VLM_runs/control_rebuild_v5_one_seed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--v5-run", type=Path, default=DEFAULT_V5_RUN)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    v5_run = args.v5_run.resolve()
    manifest = {
        "manifest_version": "shared_forward_direction_v6_evaluated_candidate",
        "complete": True,
        "status": "evaluated_not_promoted",
        "seed": 20260729,
        "simulator_at_inference": False,
        "scope": {
            "candidate_replacements": [
                "predict_forward_from_state_v1",
                "predict_forward_from_image_v1",
                "predict_direction_from_state_v1",
                "predict_direction_from_image_v1",
            ],
            "unchanged_during_evaluation": [
                "Qwen orchestration",
                "measurement",
                "numerical inverse",
                "visual inverse",
            ],
        },
        "artifacts": {
            "shared_forward_direction_v6": entry(
                run_dir / "shared_forward_direction_v6.pt"
            ),
            "base_forward_v5": entry(v5_run / "forward_tree_v5.pkl"),
            "inverse_v5": entry(v5_run / "inverse_tree_v5.pkl"),
        },
        "validation_reports": {
            "specialist": entry(
                run_dir / "shared_forward_direction_v6_summary.json"
            ),
            "system": entry(
                run_dir / "orchestrated_system_shared_v6_validation.json"
            ),
        },
        "held_out_test_files_opened": [],
        "production_approved": False,
    }
    output = run_dir / "shared_forward_direction_v6_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(output), "sha256": sha256(output)}))


if __name__ == "__main__":
    main()

