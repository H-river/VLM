#!/usr/bin/env python3
"""Write a hash-pinned numerical-v5 validation overlay manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    REPO_ROOT.parent
    / "VLM_runs/direction_rebuild_v4_tree_one_seed"
    / "candidate_overlay_direction_v4_manifest.json"
)
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v5_one_seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
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
    base_path = args.base_manifest.resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    artifacts = dict(base["artifacts"])
    artifacts["forward_v5"] = entry(run_dir / "forward_tree_v5.pkl")
    artifacts["inverse_v5"] = entry(run_dir / "inverse_tree_v5.pkl")
    routes = json.loads(json.dumps(base["routes"]))
    routes["predict_forward_from_state_v1"] = {
        "backend": "five_head_forward_tree_v5",
        "artifacts": ["forward_v5"],
    }
    routes["predict_forward_from_image_v1"] = {
        "backend": "guarded_measurement_v4_then_five_head_forward_tree_v5",
        "artifacts": [
            "measurement_v3",
            "measurement_calibrator_v4",
            "forward_v5",
        ],
    }
    routes["select_inverse_action_from_states_v1"] = {
        "backend": "forward_tree_v5_then_candidate_feasibility_inverse_v5",
        "artifacts": ["forward_v5", "inverse_v5"],
    }
    manifest = {
        "manifest_version": "qwen_to_specialists_v5_numerical_validation_overlay",
        "complete": True,
        "status": "validation_candidate_not_production",
        "seed": 20260728,
        "action_grid_size": 81,
        "simulator_at_inference": False,
        "scope": {
            "replaced": [
                "forward prediction from state",
                "forward prediction from image after frozen measurement",
                "inverse action selection from numerical states",
            ],
            "unchanged": [
                "Qwen orchestration",
                "measurement",
                "direction",
                "visual inverse",
            ],
        },
        "base_direction_v4_manifest": {
            **entry(base_path),
            "manifest_version": base["manifest_version"],
        },
        "frozen_qwen": base["frozen_qwen"],
        "direct_measurement_policy": base["direct_measurement_policy"],
        "artifacts": artifacts,
        "routes": routes,
        "validation_reports": {
            "forward": entry(run_dir / "forward_tree_v5_summary.json"),
            "inverse": entry(run_dir / "inverse_tree_v5_summary.json"),
            "system": entry(
                run_dir / "orchestrated_system_numerical_v5_validation.json"
            ),
        },
        "held_out_test_files_opened": [],
    }
    output = run_dir / "candidate_overlay_numerical_v5_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()

