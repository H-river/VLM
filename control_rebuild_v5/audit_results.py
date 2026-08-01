#!/usr/bin/env python3
"""Audit v5 numerical data, artifacts, metrics, and declared targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/control_rebuild_v5_numerical"
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v5_one_seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
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
        "passed": bool(passed),
        "margin": (
            float(value - threshold)
            if relation == ">="
            else float(threshold - value)
        ),
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    data_manifest = json.loads(
        (data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    forward = json.loads(
        (run_dir / "forward_tree_v5_summary.json").read_text(encoding="utf-8")
    )
    inverse = json.loads(
        (run_dir / "inverse_tree_v5_summary.json").read_text(encoding="utf-8")
    )
    system = json.loads(
        (
            run_dir / "orchestrated_system_numerical_v5_validation.json"
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (
            run_dir / "candidate_overlay_numerical_v5_manifest.json"
        ).read_text(encoding="utf-8")
    )

    forward_metrics = forward["validation"]["forward_tree_v5"]
    difficult = forward_metrics["difficult"]["overall"]
    old = forward_metrics["old_iid"]["overall"]
    fields = difficult["per_field_tolerance_pass"]
    four = forward_metrics["difficult"]["by_action_complexity"]["4"][
        "strict_all_five_success"
    ]
    forward_gates = [
        gate("old_iid_strict_all_five", old["strict_all_five_success"], 0.45),
        gate(
            "difficult_strict_all_five",
            difficult["strict_all_five_success"],
            0.40,
        ),
        gate("four_component_strict_all_five", four, 0.25),
        gate("difficult_peak", fields["peak_intensity"], 0.70),
        gate("difficult_centroid_x", fields["centroid_x_px"], 0.75),
        gate("difficult_centroid_y", fields["centroid_y_px"], 0.75),
        gate("difficult_width_x", fields["sigma_x_px"], 0.92),
        gate("difficult_width_y", fields["sigma_y_px"], 0.92),
        gate(
            "difficult_average_worst_field_error",
            difficult["worst_field_error_in_tolerance_units"],
            1.50,
            "<=",
        ),
    ]
    inverse_metrics = inverse["validation"]["inverse_tree_v5"]
    inverse_gates = [
        gate(
            "iid_clean_target_reached",
            inverse_metrics["iid_clean"]["target_success_feasible"],
            0.50,
        ),
        gate(
            "difficult_clean_target_reached",
            inverse_metrics["difficult_clean"]["target_success_feasible"],
            0.65,
        ),
        gate(
            "difficult_paired_error_target_reached",
            inverse_metrics["difficult_measurement_augmented"][
                "target_success_feasible"
            ],
            0.50,
        ),
        gate(
            "difficult_status_accuracy",
            inverse_metrics["difficult_clean"]["status_accuracy"],
            0.82,
        ),
    ]
    system_metrics = system["metrics"]
    inverse_by_route = system_metrics["inverse_physical_by_route"]
    numerical_inverse = inverse_by_route[
        "select_inverse_action_from_states_v1"
    ]
    system_gates = [
        gate(
            "system_forward_strict_all_five",
            system_metrics[
                "end_to_end_forward_physical_strict_all_five_success"
            ],
            0.40,
        ),
        gate(
            "system_numerical_inverse_target_reached",
            numerical_inverse["end_to_end_target_reached"],
            0.60,
        ),
        gate(
            "system_numerical_inverse_correct_route",
            numerical_inverse["correctly_routed_target_reached"],
            0.60,
        ),
        gate(
            "system_successful_valid_execution",
            system_metrics["successful_valid_execution_rate"],
            0.98,
        ),
    ]

    artifact_checks = {}
    for key in ("forward_v5", "inverse_v5"):
        expected = manifest["artifacts"][key]
        path = Path(expected["path"])
        artifact_checks[key] = {
            "exists": path.is_file(),
            "size_matches": path.is_file()
            and path.stat().st_size == int(expected["size"]),
            "sha256_matches": path.is_file()
            and sha256(path) == str(expected["sha256"]),
        }
    all_gates = [*forward_gates, *inverse_gates, *system_gates]
    integrity_passed = (
        data_manifest.get("verification", {}).get("complete") is True
        and int(data_manifest["verification"]["groups"]) == 6000
        and all(
            all(values.values()) for values in artifact_checks.values()
        )
        and not forward.get("held_out_test_used")
        and not inverse.get("held_out_test_used")
        and system.get("simulator_at_inference") is False
    )
    report = {
        "audit_version": "control_rebuild_v5_numerical_one_seed",
        "data_integrity_passed": integrity_passed,
        "data_groups": data_manifest.get("verification", {}).get("groups"),
        "data_transitions": data_manifest.get("verification", {}).get(
            "transitions"
        ),
        "artifact_checks": artifact_checks,
        "forward_gates": forward_gates,
        "inverse_gates": inverse_gates,
        "system_gates": system_gates,
        "passed_gate_count": sum(row["passed"] for row in all_gates),
        "total_gate_count": len(all_gates),
        "all_engineering_targets_passed": all(
            row["passed"] for row in all_gates
        ),
        "held_out_test_files_opened": [],
        "production_approved": False,
        "complete": True,
    }
    output = run_dir / "numerical_v5_audit.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

