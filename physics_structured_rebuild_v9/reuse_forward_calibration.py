#!/usr/bin/env python3
"""Apply a frozen forward calibration to a newly trained compatible tree."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_structured_rebuild_v9.calibrate_system_direction import sha256

ROUTES = {
    "forward_state.pkl": "predict_forward_from_state_v1",
    "forward_image.pkl": "predict_forward_from_image_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--tree-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_dir = args.reference_dir.resolve()
    tree_path = args.tree_artifact.resolve()
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "forward_calibration_reuse_summary.json"
    outputs = {name: output_dir / name for name in ROUTES}
    if report_path.exists() or any(path.exists() for path in outputs.values()):
        raise RuntimeError(f"refusing to overwrite output: {output_dir}")
    with tree_path.open("rb") as stream:
        tree = pickle.load(stream)
    if tree.get("model") not in {
        "v7_stacked_five_head_hist_gradient_boosting_forward_v9",
        "v7_stacked_five_head_extra_trees_forward_v9",
        "v7_stacked_five_head_lightgbm_forward_v9",
    }:
        raise ValueError("unexpected residual-forward tree artifact")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, route in ROUTES.items():
        reference_path = reference_dir / name
        with reference_path.open("rb") as stream:
            reference = pickle.load(stream)
        if reference.get("model") != "frozen_v7_plus_residual_tree_v9":
            raise ValueError("unexpected reference calibration artifact")
        if reference.get("route_scope") != route:
            raise ValueError("reference route differs")
        artifact = {
            **reference,
            "version": f"reused_round2_{reference['version']}",
            "tree_artifact": str(tree_path),
            "tree_artifact_sha256": sha256(tree_path),
            "field_tree_source": {
                field: "primary"
                for field in reference["field_tree_source"]
            },
            "secondary_tree_artifact": None,
            "secondary_tree_artifact_sha256": None,
            "calibration_reused_from": str(reference_path.resolve()),
            "calibration_reused_from_sha256": sha256(
                reference_path.resolve()
            ),
            "system_validation_used_for_training": False,
            "held_out_test_used": False,
        }
        with outputs[name].open("wb") as stream:
            pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
        artifacts[route] = {
            "path": str(outputs[name]),
            "sha256": sha256(outputs[name]),
            "field_blend": artifact["field_blend"],
        }
    report = {
        "version": "reused_forward_calibration_round2_v9_one_seed",
        "artifacts": artifacts,
        "source_contract": {
            "reference_dir": str(reference_dir),
            "tree_artifact": str(tree_path),
            "system_validation_used_for_training": False,
            "validation_files_opened": [],
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
