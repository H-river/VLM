#!/usr/bin/env python3
"""Audit the targeted candidate and write a frozen comparison report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT.parent / "VLM_data/joint_forward_direction_targeted_v7"
BASE_RUN = REPO_ROOT.parent / "VLM_runs/joint_forward_direction_v7_one_seed"
TARGET_RUN = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_targeted_one_seed"
)
BASE_SUMMARY = BASE_RUN / "shared_forward_direction_v7_summary.json"
TARGET_SUMMARY = TARGET_RUN / "shared_forward_direction_v7_summary.json"
BASE_SYSTEM = BASE_RUN / "orchestrated_system_shared_v7_validation.json"
TARGET_SYSTEM = (
    TARGET_RUN / "orchestrated_system_shared_v7_validation.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_checksums(root: Path) -> tuple[int, list[str]]:
    failures = []
    lines = (root / "checksums.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            failures.append(relative)
    return len(lines), failures


def metric_values(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary["validation"]["shared_forward_direction_v7"]
    return {
        "forward_old": float(
            metrics["forward"]["old_iid"]["overall"][
                "strict_all_five_success"
            ]
        ),
        "forward_difficult": float(
            metrics["forward"]["difficult"]["overall"][
                "strict_all_five_success"
            ]
        ),
        "forward_old_high_complexity": float(
            metrics["forward"]["old_iid"]["by_action_complexity"][
                "three_or_four"
            ]["strict_all_five_success"]
        ),
        "forward_difficult_high_complexity": float(
            metrics["forward"]["difficult"]["by_action_complexity"][
                "three_or_four"
            ]["strict_all_five_success"]
        ),
        "direction_old": float(
            metrics["direction"]["old_iid"]["overall"]["joint_exact"]
        ),
        "direction_difficult": float(
            metrics["direction"]["difficult"]["overall"]["joint_exact"]
        ),
    }


def system_values(report: dict[str, Any]) -> dict[str, float]:
    metrics = report["metrics"]
    return {
        "forward_end_to_end": float(
            metrics["end_to_end_forward_physical_strict_all_five_success"]
        ),
        "forward_correctly_routed": float(
            metrics[
                "correctly_routed_forward_physical_strict_all_five_success"
            ]
        ),
        "direction_end_to_end_exact": float(
            metrics["end_to_end_direction_physical_all_five_exact"]
        ),
        "direction_end_to_end_macro_f1": float(
            metrics["end_to_end_direction_physical_macro_f1"]
        ),
        "inverse_end_to_end": float(
            metrics["end_to_end_inverse_target_reached_rate"]
        ),
    }


def comparison(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, dict[str, float]]:
    return {
        key: {
            "before": before[key],
            "after": after[key],
            "absolute_delta": after[key] - before[key],
        }
        for key in before
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    manifest = read_json(DATA_ROOT / "manifest.json")
    base_summary = read_json(BASE_SUMMARY)
    target_summary = read_json(TARGET_SUMMARY)
    base_system = read_json(BASE_SYSTEM)
    target_system = read_json(TARGET_SYSTEM)
    checksum_count, checksum_failures = verify_checksums(DATA_ROOT)
    artifact = Path(target_summary["artifact"])
    base_specialist = metric_values(base_summary)
    target_specialist = metric_values(target_summary)
    base_end_to_end = system_values(base_system)
    target_end_to_end = system_values(target_system)
    specialist_comparison = comparison(
        base_specialist,
        target_specialist,
    )
    system_comparison = comparison(base_end_to_end, target_end_to_end)

    expected_categories = read_json(DATA_ROOT / "config.json")[
        "category_counts"
    ]
    integrity = {
        "dataset_complete": (
            int(manifest["train_groups"]) == 3000
            and int(manifest["train_transitions"]) == 243000
        ),
        "category_counts_exact": (
            manifest["category_counts"] == expected_categories
        ),
        "checksums_verified": not checksum_failures,
        "checksum_file_count": checksum_count,
        "artifact_hash_matches_summary": (
            sha256(artifact) == target_summary["artifact_sha256"]
        ),
        "system_artifact_hash_matches": (
            target_system["artifacts"]["shared_forward_direction_v7"][
                "sha256"
            ]
            == sha256(artifact)
        ),
        "system_evaluation_complete": (
            target_system.get("complete") is True
            and int(target_system["record_count"]) == 1600
        ),
        "no_simulator_during_inference": (
            int(
                target_system["metrics"][
                    "simulator_calls_during_inference_count"
                ]
            )
            == 0
        ),
        "held_out_test_unused": (
            target_summary.get("held_out_test_used") is False
            and target_summary["source_contract"]["held_out_files_opened"]
            == []
            and int(manifest["held_out_test_groups_used"]) == 0
        ),
        "frozen_validation_unmodified": (
            manifest["frozen_validation_files_modified"] is False
        ),
    }
    non_regression_gates = {
        "forward_old": (
            target_specialist["forward_old"]
            >= base_specialist["forward_old"]
        ),
        "forward_difficult": (
            target_specialist["forward_difficult"]
            >= base_specialist["forward_difficult"]
        ),
        "forward_old_high_complexity": (
            target_specialist["forward_old_high_complexity"]
            >= base_specialist["forward_old_high_complexity"]
        ),
        "forward_difficult_high_complexity": (
            target_specialist["forward_difficult_high_complexity"]
            >= base_specialist["forward_difficult_high_complexity"]
        ),
        "system_forward_end_to_end": (
            target_end_to_end["forward_end_to_end"]
            >= base_end_to_end["forward_end_to_end"]
        ),
        "system_forward_correctly_routed": (
            target_end_to_end["forward_correctly_routed"]
            >= base_end_to_end["forward_correctly_routed"]
        ),
        "system_direction_exact": (
            target_end_to_end["direction_end_to_end_exact"]
            >= base_end_to_end["direction_end_to_end_exact"]
        ),
        "system_direction_macro_f1": (
            target_end_to_end["direction_end_to_end_macro_f1"]
            >= base_end_to_end["direction_end_to_end_macro_f1"]
        ),
    }
    target_75 = {
        "forward_old": target_specialist["forward_old"] >= 0.75,
        "forward_difficult": (
            target_specialist["forward_difficult"] >= 0.75
        ),
        "direction_old": target_specialist["direction_old"] >= 0.75,
        "direction_difficult": (
            target_specialist["direction_difficult"] >= 0.75
        ),
    }
    inverse_prerequisite = (
        target_specialist["forward_old"] >= 0.65
        and target_specialist["forward_difficult"] >= 0.65
        and non_regression_gates["system_forward_end_to_end"]
    )
    accepted = all(
        bool(value) for value in integrity.values() if isinstance(value, bool)
    ) and all(non_regression_gates.values())
    audit = {
        "version": "joint_forward_direction_v7_targeted_final_audit",
        "complete": True,
        "dataset": {
            "path": str(DATA_ROOT),
            "manifest": manifest,
            "checksum_failures": checksum_failures,
        },
        "artifacts": {
            "candidate": str(artifact),
            "candidate_sha256": sha256(artifact),
            "summary": str(TARGET_SUMMARY),
            "system_evaluation": str(TARGET_SYSTEM),
        },
        "specialist_comparison": specialist_comparison,
        "system_comparison": system_comparison,
        "integrity": integrity,
        "non_regression_gates": non_regression_gates,
        "target_75_reached": target_75,
        "candidate_accepted_for_manual_review": accepted,
        "inverse_prerequisite_reached": inverse_prerequisite,
        "inverse_retrained": False,
        "inverse_note": (
            "Numerical inverse remains frozen because forward must first "
            "reach at least 65% strict success on both validation sets."
            if not inverse_prerequisite
            else "Forward prerequisite passed; inverse retraining is now due."
        ),
        "deployment_modified": False,
    }
    audit_path = TARGET_RUN / "final_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Targeted shared forward-direction result",
        "",
        "No deployed manifest was changed.",
        "",
        "## Specialist comparison",
        "",
        "| Metric | Before | After | Change |",
        "|---|---:|---:|---:|",
    ]
    for key, values in specialist_comparison.items():
        lines.append(
            f"| {key} | {pct(values['before'])} | "
            f"{pct(values['after'])} | "
            f"{100.0 * values['absolute_delta']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## End-to-end comparison",
            "",
            "| Metric | Before | After | Change |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, values in system_comparison.items():
        lines.append(
            f"| {key} | {pct(values['before'])} | "
            f"{pct(values['after'])} | "
            f"{100.0 * values['absolute_delta']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Manual-review candidate accepted: `{accepted}`",
            f"- Forward prerequisite for inverse: `{inverse_prerequisite}`",
            f"- 75% target reached on all four requested metrics: "
            f"`{all(target_75.values())}`",
            "- Numerical inverse retrained: `False`",
            "- Deployment modified: `False`",
            "",
        ]
    )
    (TARGET_RUN / "FINAL_RESULTS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
