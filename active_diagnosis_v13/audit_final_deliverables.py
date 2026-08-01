#!/usr/bin/env python3
"""Audit the final v13 report, decision chain, provenance, and required artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

VERSION = "active_diagnosis_v13_final_deliverables_audit_v6"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.resolve().read_text().splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    report_path = args.report.resolve()
    paths = {
        "report": report_path,
        "gate": run / "development/gate_a_diagnosis.json",
        "resolution": run / "development/branch_a_resolution.json",
        "protected_manifest": run / "protected/protected_once_manifest.json",
        "protected_summary": run / "protected/protected_confirmatory_summary.json",
        "protected_direct": run / "protected/control/direct.jsonl",
        "protected_oracle": run / "protected/control/oracle_known.jsonl",
        "protected_probe": run / "protected/control/probe_replan.jsonl",
        "final_decision": run / "final_synthesis/final_research_decision.json",
        "progress_audit": run / "development/progress_log_audit.json",
        "command_audit": run / "development/reproduction_command_audit.json",
        "integrity_audit": run / "development/artifact_integrity_postfreeze.json",
        "source_snapshot": run / "development/postfreeze_source_snapshot.json",
        "hourly_timeline": run / "hourly_timeline.json",
        "pytest_report": run / "development/pytest_final.xml",
        "exact_commands": run / "commands/exact_commands.md",
    }
    fresh_suite = run / "development/fresh_setup_holdout/evaluation_suite_manifest.json"
    if fresh_suite.exists():
        paths["fresh_holdout"] = run / "development/frozen_sequential_fresh_holdout.json"
        paths["cohort_shift"] = (
            run / "development/frozen_sequential_setup_cohort_shift.json"
        )
    fresh_suite_b = (
        run / "development/fresh_setup_holdout_b/evaluation_suite_manifest.json"
    )
    if fresh_suite_b.exists():
        paths["fresh_holdout_b"] = (
            run / "development/frozen_sequential_fresh_holdout_b.json"
        )
    fresh_matched_suite = (
        run / "development/fresh_setup_holdout_matched/evaluation_suite_manifest.json"
    )
    if fresh_matched_suite.exists():
        paths["fresh_matched_suite"] = fresh_matched_suite
        paths["fresh_matched"] = (
            run / "development/frozen_sequential_fresh_holdout_matched.json"
        )
        paths["fresh_matched_seed2"] = (
            run
            / "development/frozen_sequential_fresh_holdout_matched_seed_2026080202.json"
        )
        paths["fresh_replication"] = (
            run / "development/frozen_sequential_fresh_holdout_replication.json"
        )
    fresh_seed_a = (
        run / "development/frozen_sequential_fresh_holdout_seed_2026080202.json"
    )
    fresh_seed_b = (
        run / "development/frozen_sequential_fresh_holdout_b_seed_2026080202.json"
    )
    if fresh_seed_a.exists() or fresh_seed_b.exists():
        paths["fresh_seed_a"] = fresh_seed_a
        paths["fresh_seed_b"] = fresh_seed_b
        paths["fresh_seed_setup"] = (
            run
            / "development/frozen_sequential_fresh_holdout_seed_setup_robustness.json"
        )
        paths["fresh_failure_stability"] = (
            run
            / "development/frozen_sequential_fresh_holdout_failure_stability.json"
        )
    candidate_third_decision_path = (
        run / "development/boundary_candidate_third_seed_decision.json"
    )
    if candidate_third_decision_path.exists():
        paths["candidate_third_decision"] = candidate_third_decision_path
        candidate_third_preview = _read(candidate_third_decision_path)
        if candidate_third_preview.get("launch_seed3") is True:
            paths["candidate_seed3_audit"] = (
                run
                / "development/boundary_candidate_coverage_seed_2026080103_audit.json"
            )
            paths["candidate_three_seed"] = (
                run / "development/boundary_candidate_coverage_three_seed.json"
            )
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"missing final artifacts: {missing}")
    gate = _read(paths["gate"])
    resolution = _read(paths["resolution"])
    protected_manifest = _read(paths["protected_manifest"])
    protected_summary = _read(paths["protected_summary"])
    protected_rows = {
        "direct": _read_jsonl(paths["protected_direct"]),
        "oracle_known": _read_jsonl(paths["protected_oracle"]),
        "probe_replan": _read_jsonl(paths["protected_probe"]),
    }
    protected_rows_by_key = {
        mode: {
            (row["case_id"], float(row["evaluator_only_true_gain"])): row
            for row in rows
        }
        for mode, rows in protected_rows.items()
    }
    protected_episode_keys = set(protected_rows_by_key["direct"])
    protected_fault_keys = {
        key for key in protected_episode_keys if abs(key[1] - 1.0) > 1e-12
    }
    decision = _read(paths["final_decision"])
    progress = _read(paths["progress_audit"])
    commands = _read(paths["command_audit"])
    integrity = _read(paths["integrity_audit"])
    snapshot = _read(paths["source_snapshot"])
    hourly = _read(paths["hourly_timeline"])
    pytest_root = ET.parse(paths["pytest_report"]).getroot()
    pytest_suites = (
        [pytest_root]
        if pytest_root.tag == "testsuite"
        else list(pytest_root.findall(".//testsuite"))
    )
    pytest_totals = {
        field: sum(int(suite.attrib.get(field, 0)) for suite in pytest_suites)
        for field in ("tests", "failures", "errors")
    }
    fresh_holdout = (
        None if "fresh_holdout" not in paths else _read(paths["fresh_holdout"])
    )
    fresh_replication = (
        None
        if "fresh_replication" not in paths
        else _read(paths["fresh_replication"])
    )
    fresh_seed_setup = (
        None
        if "fresh_seed_setup" not in paths
        else _read(paths["fresh_seed_setup"])
    )
    fresh_failure_stability = (
        None
        if "fresh_failure_stability" not in paths
        else _read(paths["fresh_failure_stability"])
    )
    cohort_shift = None if "cohort_shift" not in paths else _read(paths["cohort_shift"])
    fresh_matched = (
        None if "fresh_matched" not in paths else _read(paths["fresh_matched"])
    )
    fresh_matched_manifest = (
        None
        if "fresh_matched_suite" not in paths
        else _read(paths["fresh_matched_suite"])
    )
    candidate_third_decision = (
        None
        if "candidate_third_decision" not in paths
        else _read(paths["candidate_third_decision"])
    )
    candidate_seed3_audit = (
        None
        if "candidate_seed3_audit" not in paths
        else _read(paths["candidate_seed3_audit"])
    )
    candidate_three_seed = (
        None
        if "candidate_three_seed" not in paths
        else _read(paths["candidate_three_seed"])
    )
    report_text = report_path.read_text()
    report_text_flat = " ".join(report_text.split())
    report_exact_id_blocks = [
        [line for line in block.splitlines() if line]
        for block in re.findall(r"```text\n(.*?)```", report_text, flags=re.DOTALL)
    ]
    temporal_evidence = decision["ranked_bottlenecks"][0]["evidence"]
    candidate_evidence = decision["ranked_bottlenecks"][1]["evidence"]
    hourly_markers = [f"T+{hour}–T+{hour + 1}" for hour in range(12)]
    checks = {
        "exactly_one_primary_branch_a": bool(
            gate["exactly_one_primary_branch_selected"]
            and gate["selected_primary_branch"] == "A"
        ),
        "gate_decomposition_supports_branch_a": bool(
            gate["fault_impact"]["passes"] is True
            and gate["recoverability"]["passes"] is True
            and gate["observability"]["passes"] is False
            and gate["control_value"]["passes"] is True
        ),
        "gate_decomposition_metrics_are_exact": bool(
            abs(gate["fault_impact"]["success_drop"] - 23 / 120) < 1e-12
            and abs(
                gate["recoverability"]["success_gain_over_direct_fault"] - 15 / 120
            )
            < 1e-12
            and abs(gate["observability"]["classification_accuracy"] - 118 / 150)
            < 1e-12
            and abs(
                gate["control_value"]["success_gain_over_direct_fault"] - 12 / 120
            )
            < 1e-12
            and gate["fault_impact"]["matched_group_bootstrap_95"]["low"] > 0.0
            and gate["recoverability"]["matched_group_bootstrap_95"]["low"]
            > 0.0
            and gate["control_value"]["matched_group_bootstrap_95"]["low"]
            > 0.0
        ),
        "resolution_is_development_only_branch_a": bool(
            resolution["selected_primary_branch"] == "A"
            and resolution.get("protected_set_used_for_selection") is False
        ),
        "protected_once_complete": bool(
            protected_manifest["status"] == "complete"
            and len(protected_manifest["jobs"]) == 3
            and all(
                int(job["records"]) == 90 and int(job["returncode"]) == 0
                for job in protected_manifest["jobs"]
            )
        ),
        "protected_never_reselected": bool(
            protected_manifest["protected_set_used_for_selection"] is False
            and protected_summary["protected_set_used_for_selection"] is False
            and protected_summary["branch_reselected"] is False
        ),
        "protected_confirmatory_metrics_are_exact": bool(
            protected_summary["fault_impact"]["fault_direct_success"]
            == 43 / 72
            and protected_summary["recoverability"]["oracle_fault_success"]
            == 52 / 72
            and protected_summary["control_value"]["probe_fault_success"]
            == 52 / 72
            and protected_summary["control_value"]["success_gain_over_direct"]
            == 9 / 72
            and protected_summary["control_value"]["matched_recoveries"] == 12
            and protected_summary["control_value"]["matched_regressions"] == 3
            and protected_summary["control_value"]["matched_group_bootstrap_95"][
                "low"
            ]
            > 0.0
        ),
        "protected_raw_rows_match_confirmatory_summary": bool(
            all(len(rows) == 90 for rows in protected_rows.values())
            and all(len(rows) == 90 for rows in protected_rows_by_key.values())
            and all(
                set(rows) == protected_episode_keys
                for rows in protected_rows_by_key.values()
            )
            and len(protected_fault_keys) == 72
            and sum(
                bool(protected_rows_by_key["direct"][key]["strict_success"])
                for key in protected_episode_keys
            )
            == 57
            and sum(
                bool(protected_rows_by_key["oracle_known"][key]["strict_success"])
                for key in protected_episode_keys
            )
            == 66
            and sum(
                bool(protected_rows_by_key["probe_replan"][key]["strict_success"])
                for key in protected_episode_keys
            )
            == 66
            and sum(
                bool(protected_rows_by_key["direct"][key]["strict_success"])
                for key in protected_fault_keys
            )
            == 43
            and sum(
                bool(protected_rows_by_key["oracle_known"][key]["strict_success"])
                for key in protected_fault_keys
            )
            == 52
            and sum(
                bool(protected_rows_by_key["probe_replan"][key]["strict_success"])
                for key in protected_fault_keys
            )
            == 52
            and sum(
                not bool(protected_rows_by_key["direct"][key]["strict_success"])
                and bool(
                    protected_rows_by_key["probe_replan"][key]["strict_success"]
                )
                for key in protected_fault_keys
            )
            == 12
            and sum(
                bool(protected_rows_by_key["direct"][key]["strict_success"])
                and not bool(
                    protected_rows_by_key["probe_replan"][key]["strict_success"]
                )
                for key in protected_fault_keys
            )
            == 3
        ),
        "protected_selection_is_sha_bound_to_frozen_resolution": bool(
            protected_manifest["selection_source_sha256"]
            == _sha256(paths["resolution"])
            == protected_summary["selection_source_sha256"]
            and protected_manifest["selected_probe"]["classifier_bundle_sha256"]
            == resolution["selected_probe"]["classifier_bundle_sha256"]
            == protected_summary["frozen_probe"]["classifier_bundle_sha256"]
        ),
        "final_decision_preserves_branch_a": bool(
            decision["selected_primary_branch"] == "A"
            and decision["protected_set_used_for_reselection"] is False
        ),
        "final_decision_reproduction_commands_are_resolved": bool(
            decision["version"] == "active_diagnosis_v13_final_research_decision_v4"
            and all(
                str(run / relative)
                in decision["reproduction_commands"]["horizon"]
                for relative in (
                    "development/control/direct.jsonl",
                    "development/control/direct_budget6.jsonl",
                    "development/control/direct_budget8.jsonl",
                    "development/control/probe_symmetric_pair_f0p1_no_residual.jsonl",
                    "development/control/probe_nores_budget6.jsonl",
                    "development/control/probe_nores_budget8.jsonl",
                    "development/control_horizon_curve.json",
                )
            )
            and "exact six-input command"
            in decision["reproduction_commands"]["fresh_failure_stability"]
            and str(run / "commands/exact_commands.md")
            in decision["reproduction_commands"]["fresh_failure_stability"]
        ),
        "final_temporal_and_candidate_evidence_is_complete": bool(
            temporal_evidence["planner_seed_count"] == 5
            and temporal_evidence["sequential_rule_all_observed_seeds_positive"]
            is True
            and temporal_evidence["sequential_rule_seed_group_interval_pp"][0]
            > 0.0
            and temporal_evidence["sequential_rule_recovery_seed_stability"]
            == {
                "planner_seeds": 5,
                "recoveries_by_seed": [25, 29, 24, 25, 23],
                "stable_recoveries_all_seeds": 13,
                "recoveries_any_seed": 42,
                "stable_to_any_recovery_ratio": 13 / 42,
            }
            and temporal_evidence["fresh_nonprotected_seed_setup_robustness"][
                "all_suite_seed_cells_positive"
            ]
            is True
            and temporal_evidence["fresh_nonprotected_seed_setup_robustness"][
                "planner_seeds"
            ]
            == 2
            and temporal_evidence["fresh_nonprotected_seed_setup_robustness"][
                "setups"
            ]
            == 90
            and temporal_evidence["fresh_nonprotected_seed_setup_robustness"][
                "two_way_gain_interval_pp"
            ][0]
            > 0.0
            and candidate_evidence["proposal_seed_count"] == 3
            and candidate_evidence[
                "proposal_union_successes_by_seed_at_maximum_common_budget"
            ]
            == {
                "seed_2026080101": 7,
                "seed_2026080102": 5,
                "seed_2026080103": 3,
            }
            and candidate_evidence["stable_proposal_union_successes_all_seeds"]
            == 1
            and candidate_evidence["proposal_union_successes_any_seed"] == 10
            and candidate_evidence[
                "stable_proposal_recoveries_beyond_temporal8_all_seeds"
            ]
            == []
            and candidate_evidence[
                "minimum_seed_candidate_union_fault_upper_bound_points"
            ]
            == 2.5
        ),
        "progress_log_passes_12h_audit": bool(
            progress["passes"]
            and progress["required_end_at_or_after"]
            == "2026-08-01T13:00:51+08:00"
            and float(progress["elapsed_seconds"]) >= 12 * 60 * 60
        ),
        "reproduction_commands_pass": bool(commands["passes"]),
        "full_regression_junit_passes": bool(
            pytest_totals["tests"] >= 102
            and pytest_totals["failures"] == 0
            and pytest_totals["errors"] == 0
        ),
        "development_integrity_passes": bool(integrity["passes"]),
        "source_snapshot_does_not_traverse_protected": bool(
            snapshot["protected_directory_traversed"] is False
        ),
        "all_twelve_hours_have_machine_readable_evidence": bool(
            int(hourly["hours"]) == 12
            and int(hourly["hours_with_evidence"]) == 12
            and hourly["all_hours_have_evidence"]
        ),
        "report_names_branch_a": "Branch A" in report_text,
        "report_is_explicitly_final": report_text.startswith(
            "# Active diagnosis v13 — final report\n"
        ),
        "report_has_ranked_bottleneck_table": "| Rank | Bottleneck |" in report_text,
        "report_has_actionable_next_decision": bool(
            "Next research decision" in report_text
            and "30 external groups (ten per stratum)" in report_text_flat
            and "two preregistered planner seeds" in report_text_flat
            and "identical rule versus fixed four" in report_text_flat
            and "capped at eight" in report_text_flat
            and "two-way seed/group bootstrap lower bound" in report_text_flat
            and "step or saturation noninferiority margin" in report_text_flat
            and "no threshold retuning" in report_text_flat
        ),
        "report_has_exact_failing_setup_ids": bool(
            "Exact residual failure IDs" in report_text
            and len(report_exact_id_blocks) >= 2
            and report_exact_id_blocks[0]
            == decision["exact_failing_setup_ids"][
                "probe_fault_failures_after_budget8"
            ]
            and report_exact_id_blocks[1]
            == decision["exact_failing_setup_ids"][
                "neither_temporal8_nor_candidate_union_recovers"
            ]
        ),
        "report_has_reproduction_commands": "Reproduction commands" in report_text,
        "report_records_final_test_and_command_counts": bool(
            "102/102" in report_text and "166/166" in report_text
        ),
        "report_records_final_temporal_and_candidate_results": bool(
            "+21.0" in report_text
            and "+18.06" in report_text
            and "7/5/3" in report_text
            and "stable/any" in report_text
            and "zero all-three-seed-stable complementary cases" in report_text
            and "Thirteen of 42" in report_text
            and "1/10" in report_text
            and "eligible failure sets" in report_text_flat
        ),
        "report_states_no_jobs_remain_active": (
            "No simulator, model-training, or protected job remains active."
            in report_text
        ),
        "report_has_all_twelve_hour_markers": all(
            marker in report_text for marker in hourly_markers
        ),
        "fresh_holdout_is_one_shot_and_disjoint_if_launched": bool(
            fresh_holdout is None
            or (
                fresh_holdout["protected_set_used"] is False
                and fresh_holdout["selection_or_retuning_on_fresh_suite"] is False
                and int(
                    fresh_holdout["setup_independence"]["group_id_overlap"]
                )
                == 0
                and int(
                    fresh_holdout["setup_independence"]["setup_hash_overlap"]
                )
                == 0
                and decision["ranked_bottlenecks"][0]["evidence"][
                    "fresh_nonprotected_setup_holdout"
                ]
                is not None
            )
        ),
        "fresh_replication_is_pooled_and_disjoint_if_launched": bool(
            fresh_replication is None
            or (
                fresh_replication["protected_set_used"] is False
                and fresh_replication[
                    "selection_or_retuning_on_fresh_suites"
                ]
                is False
                and int(fresh_replication["suite_count"]) == 2
                and int(fresh_replication["independent_setup_groups"]) == 60
                and int(fresh_replication["cross_suite_group_id_overlap"]) == 0
                and int(fresh_replication["cross_suite_setup_hash_overlap"]) == 0
                and decision["ranked_bottlenecks"][0]["evidence"][
                    "fresh_nonprotected_setup_replication"
                ]
                is not None
            )
        ),
        "fresh_seed_setup_is_balanced_and_positive_if_launched": bool(
            fresh_seed_setup is None
            or (
                fresh_seed_setup["protected_set_used"] is False
                and fresh_seed_setup[
                    "selection_or_retuning_on_fresh_suites"
                ]
                is False
                and int(fresh_seed_setup["balanced_seed_by_group_shape"][0]) >= 2
                and int(fresh_seed_setup["balanced_seed_by_group_shape"][1])
                == int(fresh_seed_setup["independent_setup_groups"])
                and int(fresh_seed_setup["independent_setup_groups"]) >= 60
                and int(fresh_seed_setup["independent_setup_groups"]) % 30 == 0
                and int(fresh_seed_setup["cross_suite_group_id_overlap"]) == 0
                and int(fresh_seed_setup["cross_suite_setup_hash_overlap"]) == 0
                and decision["ranked_bottlenecks"][0]["evidence"][
                    "fresh_nonprotected_seed_setup_robustness"
                ]
                is not None
            )
        ),
        "fresh_failure_stability_is_exact_and_descriptive_if_launched": bool(
            fresh_failure_stability is None
            or (
                fresh_failure_stability["protected_set_used"] is False
                and fresh_failure_stability[
                    "selection_or_retuning_on_fresh_suites"
                ]
                is False
                and int(fresh_failure_stability["fault_episodes_per_seed"]) >= 240
                and int(fresh_failure_stability["fault_episodes_per_seed"]) % 120
                == 0
                and decision["ranked_bottlenecks"][0]["evidence"][
                    "fresh_exact_failure_stability"
                ]
                is not None
            )
        ),
        "fresh_cohort_shift_is_disjoint_and_descriptive": bool(
            cohort_shift is None
            or (
                cohort_shift["protected_set_used"] is False
                and cohort_shift["selection_or_retuning_on_cohorts"] is False
                and all(
                    int(value) == 0
                    for value in cohort_shift["pairwise_group_id_overlaps"].values()
                )
                and decision["ranked_bottlenecks"][0]["evidence"][
                    "fresh_setup_cohort_shift"
                ]
                is not None
            )
        ),
        "difficulty_matched_fresh_suite_is_outcome_free_if_launched": bool(
            fresh_matched is None
            or (
                fresh_matched_manifest is not None
                and fresh_matched_manifest["outcome_free_difficulty_matching"][
                    "protected_set_used"
                ]
                is False
                and fresh_matched_manifest["outcome_free_difficulty_matching"][
                    "selection_or_retuning_on_candidate_outcomes"
                ]
                is False
                and fresh_matched["protected_set_used"] is False
                and fresh_matched["selection_or_retuning_on_fresh_suite"] is False
                and int(fresh_matched["setup_independence"]["group_id_overlap"])
                == 0
                and int(fresh_matched["setup_independence"]["setup_hash_overlap"])
                == 0
                and fresh_seed_setup is not None
                and int(fresh_seed_setup["independent_setup_groups"]) >= 90
                and decision["ranked_bottlenecks"][0]["evidence"][
                    "fresh_nonprotected_difficulty_matched_holdout"
                ]
                is not None
                and decision["ranked_bottlenecks"][0]["evidence"][
                    "fresh_nonprotected_difficulty_matched_holdout"
                ]["outcome_free_selection"]
                is True
            )
        ),
        "candidate_third_seed_is_complete_if_triggered": bool(
            candidate_third_decision is None
            or candidate_third_decision.get("launch_seed3") is not True
            or (
                candidate_seed3_audit is not None
                and candidate_three_seed is not None
                and int(candidate_seed3_audit["episodes"]) == 20
                and candidate_seed3_audit["protected_set_used"] is False
                and candidate_seed3_audit["budgets"] == [8, 24, 48, 96]
                and candidate_seed3_audit["planner_root_seed_field_complete"]
                is True
                and candidate_seed3_audit["planner_root_seed_values"]
                == [2026080103]
                and candidate_three_seed["protected_set_used"] is False
                and candidate_three_seed["proposal_seeds"]
                == [
                    "seed_2026080101",
                    "seed_2026080102",
                    "seed_2026080103",
                ]
                and candidate_three_seed["common_budgets"] == [8, 24, 48, 96]
                and candidate_three_seed["audit_provenance"][
                    "seed_2026080101"
                ]["planner_root_seed_field_complete"]
                is False
                and candidate_three_seed["audit_provenance"][
                    "seed_2026080101"
                ]["planner_root_seed_values"]
                == []
                and all(
                    candidate_three_seed["audit_provenance"][label][
                        "planner_root_seed_field_complete"
                    ]
                    is True
                    and candidate_three_seed["audit_provenance"][label][
                        "planner_root_seed_values"
                    ]
                    == [root_seed]
                    for label, root_seed in (
                        ("seed_2026080102", 2026080102),
                        ("seed_2026080103", 2026080103),
                    )
                )
                and len(
                    candidate_three_seed["stable_union_success_ids_all_seeds"]
                )
                <= len(candidate_three_seed["union_success_ids_any_seed"])
                and decision["ranked_bottlenecks"][1]["evidence"][
                    "proposal_seed_count"
                ]
                == 3
                and decision["ranked_bottlenecks"][1]["evidence"][
                    "stable_proposal_union_successes_all_seeds"
                ]
                == len(
                    candidate_three_seed["stable_union_success_ids_all_seeds"]
                )
                and decision["ranked_bottlenecks"][1]["evidence"][
                    "proposal_union_successes_any_seed"
                ]
                == len(candidate_three_seed["union_success_ids_any_seed"])
            )
        ),
    }
    artifacts = {
        name: {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }
    report = {
        "version": VERSION,
        "protected_set_used": False,
        "checks": checks,
        "errors": [name for name, passed in checks.items() if not passed],
        "passes": all(checks.values()),
        "artifacts": artifacts,
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
