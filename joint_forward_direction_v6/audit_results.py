#!/usr/bin/env python3
"""Audit specialist, system, integrity, and 75-percent target gates."""

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


def gate(
    name: str,
    value: float,
    threshold: float,
    relation: str = ">=",
) -> dict[str, Any]:
    passed = value >= threshold if relation == ">=" else value <= threshold
    return {
        "name": name,
        "value": float(value),
        "threshold": float(threshold),
        "relation": relation,
        "margin": (
            float(value - threshold)
            if relation == ">="
            else float(threshold - value)
        ),
        "passed": bool(passed),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    v5_run = args.v5_run.resolve()
    specialist = json.loads(
        (run_dir / "shared_forward_direction_v6_summary.json").read_text()
    )
    system = json.loads(
        (
            run_dir / "orchestrated_system_shared_v6_validation.json"
        ).read_text()
    )
    old_system = json.loads(
        (
            v5_run / "orchestrated_system_numerical_v5_validation.json"
        ).read_text()
    )
    manifest = json.loads(
        (run_dir / "shared_forward_direction_v6_manifest.json").read_text()
    )
    validation = specialist["validation"]
    v6 = validation["shared_forward_direction_v6"]
    references = validation["references"]
    old_forward = v6["old_iid"]["forward"]["strict_all_five_success"]
    difficult_forward = v6["difficult"]["forward"][
        "strict_all_five_success"
    ]
    old_direction = v6["old_iid"]["direction"]["joint_exact"]
    difficult_direction = v6["difficult"]["direction"]["joint_exact"]
    reference_forward_old = references["forward_v5"]["old_iid"][
        "strict_all_five_success"
    ]
    reference_forward_difficult = references["forward_v5"]["difficult"][
        "strict_all_five_success"
    ]
    reference_direction_old = references["direction_tree_v4"]["old_iid"][
        "joint_exact"
    ]
    reference_direction_difficult = references["direction_tree_v4"][
        "difficult"
    ]["joint_exact"]
    system_metrics = system["metrics"]
    old_system_metrics = old_system["metrics"]

    comparison_gates = [
        gate(
            "specialist_forward_old_non_regression",
            old_forward,
            reference_forward_old - 0.005,
        ),
        gate(
            "specialist_forward_difficult_improves",
            difficult_forward,
            reference_forward_difficult,
        ),
        gate(
            "specialist_direction_old_improves",
            old_direction,
            reference_direction_old,
        ),
        gate(
            "specialist_direction_difficult_improves",
            difficult_direction,
            reference_direction_difficult,
        ),
        gate(
            "system_forward_non_regression",
            system_metrics[
                "end_to_end_forward_physical_strict_all_five_success"
            ],
            old_system_metrics[
                "end_to_end_forward_physical_strict_all_five_success"
            ],
        ),
        gate(
            "system_direction_non_regression",
            system_metrics[
                "end_to_end_direction_physical_all_five_exact"
            ],
            old_system_metrics[
                "end_to_end_direction_physical_all_five_exact"
            ],
        ),
    ]
    target_gates = [
        gate("target_75_forward_old", old_forward, 0.75),
        gate("target_75_forward_difficult", difficult_forward, 0.75),
        gate("target_75_direction_old", old_direction, 0.75),
        gate("target_75_direction_difficult", difficult_direction, 0.75),
        gate(
            "target_75_system_forward",
            system_metrics[
                "end_to_end_forward_physical_strict_all_five_success"
            ],
            0.75,
        ),
        gate(
            "target_75_system_direction",
            system_metrics[
                "end_to_end_direction_physical_all_five_exact"
            ],
            0.75,
        ),
    ]
    artifact = manifest["artifacts"]["shared_forward_direction_v6"]
    artifact_path = Path(artifact["path"])
    integrity = {
        "artifact_exists": artifact_path.is_file(),
        "artifact_size_matches": (
            artifact_path.is_file()
            and artifact_path.stat().st_size == int(artifact["size"])
        ),
        "artifact_sha256_matches": (
            artifact_path.is_file()
            and sha256(artifact_path) == artifact["sha256"]
        ),
        "specialist_complete": specialist.get("held_out_test_used") is False,
        "system_complete": system.get("complete") is True,
        "no_simulator_at_inference": (
            system.get("simulator_at_inference") is False
        ),
        "held_out_test_files_untouched": (
            specialist["source_contract"]["held_out_files_opened"] == []
        ),
    }
    promotion_passed = (
        all(row["passed"] for row in comparison_gates)
        and all(row["passed"] for row in target_gates)
        and all(integrity.values())
    )
    report = {
        "audit_version": "shared_forward_direction_v6_one_seed",
        "complete": True,
        "integrity": integrity,
        "comparison_gates": comparison_gates,
        "target_75_gates": target_gates,
        "comparison_passed_count": sum(
            row["passed"] for row in comparison_gates
        ),
        "target_75_passed_count": sum(
            row["passed"] for row in target_gates
        ),
        "promotion_passed": bool(promotion_passed),
        "production_approved": False,
        "decision": (
            "promote"
            if promotion_passed
            else "retain current v5 forward and v4 direction deployment"
        ),
    }
    output = run_dir / "shared_forward_direction_v6_audit.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
