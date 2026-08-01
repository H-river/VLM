#!/usr/bin/env python3
"""Synthesize the final v13 research decision from frozen and post-freeze evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from active_diagnosis_v13.analyze_control_step_budget import _read_jsonl

VERSION = "active_diagnosis_v13_final_research_decision_v4"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _episode_id(row: dict[str, Any]) -> str:
    return f"{row['case_id']}__g{float(row['evaluator_only_true_gain']):g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    development = run / "development"
    temporal_seed_path = development / "temporal_five_seed_statistics.json"
    if not temporal_seed_path.exists():
        temporal_seed_path = development / "temporal_seed_statistics.json"
    artifacts = {
        "gate": _read(development / "gate_a_diagnosis.json"),
        "resolution": _read(development / "branch_a_resolution.json"),
        "protected": _read(run / "protected/protected_confirmatory_summary.json"),
        "temporal_base": _read(development / "control_step_budget_analysis.json"),
        "temporal_seed": _read(temporal_seed_path),
        "horizon": _read(development / "control_horizon_curve.json"),
        "candidate": _read(development / "boundary_candidate_coverage_audit.json"),
        "overlap": _read(development / "boundary_recovery_overlap.json"),
        "residual": _read(development / "oof_candidate_residual_calibration.json"),
        "postfreeze": _read(
            run
            / "fallback_preparation/postfreeze_synthesis/postfreeze_research_decision.json"
        ),
    }
    horizon_seed_path = development / "control_horizon_three_seed.json"
    if horizon_seed_path.exists():
        artifacts["horizon_seed"] = _read(horizon_seed_path)
    sequential_path = development / "sequential_horizon_rule_five_seed.json"
    if not sequential_path.exists():
        sequential_path = development / "sequential_horizon_rule_three_seed.json"
    if sequential_path.exists():
        artifacts["sequential"] = _read(sequential_path)
    candidate_seed_path = development / "boundary_candidate_coverage_three_seed.json"
    if not candidate_seed_path.exists():
        candidate_seed_path = development / "boundary_candidate_coverage_two_seed.json"
    if candidate_seed_path.exists():
        artifacts["candidate_seed"] = _read(candidate_seed_path)
    fresh_holdout_path = development / "frozen_sequential_fresh_holdout.json"
    if fresh_holdout_path.exists():
        artifacts["fresh_holdout"] = _read(fresh_holdout_path)
    fresh_holdout_b_path = development / "frozen_sequential_fresh_holdout_b.json"
    if fresh_holdout_b_path.exists():
        artifacts["fresh_holdout_b"] = _read(fresh_holdout_b_path)
    fresh_replication_path = (
        development / "frozen_sequential_fresh_holdout_replication.json"
    )
    if fresh_replication_path.exists():
        artifacts["fresh_replication"] = _read(fresh_replication_path)
    fresh_matched_path = development / "frozen_sequential_fresh_holdout_matched.json"
    if fresh_matched_path.exists():
        artifacts["fresh_matched"] = _read(fresh_matched_path)
        artifacts["fresh_matched_manifest"] = _read(
            development
            / "fresh_setup_holdout_matched/evaluation_suite_manifest.json"
        )
    fresh_seed_setup_path = (
        development / "frozen_sequential_fresh_holdout_seed_setup_robustness.json"
    )
    if fresh_seed_setup_path.exists():
        artifacts["fresh_seed_setup"] = _read(fresh_seed_setup_path)
    fresh_failure_path = (
        development / "frozen_sequential_fresh_holdout_failure_stability.json"
    )
    if fresh_failure_path.exists():
        artifacts["fresh_failure_stability"] = _read(fresh_failure_path)
    cohort_shift_path = development / "frozen_sequential_setup_cohort_shift.json"
    if cohort_shift_path.exists():
        artifacts["cohort_shift"] = _read(cohort_shift_path)
    if artifacts["gate"]["selected_primary_branch"] != "A":
        raise ValueError("final synthesis requires the frozen Branch-A decision")
    for name in (
        "gate",
        "resolution",
        "temporal_base",
        "temporal_seed",
        "horizon",
        "candidate",
        "candidate_seed",
        "fresh_holdout",
        "fresh_holdout_b",
        "fresh_replication",
        "fresh_matched",
        "fresh_matched_manifest",
        "fresh_seed_setup",
        "fresh_failure_stability",
        "cohort_shift",
        "overlap",
        "residual",
        "postfreeze",
        "horizon_seed",
        "sequential",
    ):
        if name not in artifacts:
            continue
        artifact = artifacts[name]
        if artifact.get("protected_set_used") is True or artifact.get(
            "protected_set_used_for_selection"
        ) is True:
            raise ValueError(f"{name} uses protected evidence for selection")
    if artifacts["protected"].get("protected_set_used_for_selection") is not False:
        raise ValueError("protected confirmation was used for reselection")
    matched_manifest = artifacts.get("fresh_matched_manifest")
    if matched_manifest is not None:
        matching = matched_manifest.get("outcome_free_difficulty_matching", {})
        if matching.get("protected_set_used") is not False or matching.get(
            "selection_or_retuning_on_candidate_outcomes"
        ) is not False:
            raise ValueError(
                "difficulty-matched fresh cohort lacks outcome-free provenance"
            )

    temporal_seed = artifacts["temporal_seed"]
    horizon = artifacts["horizon"]
    candidate = artifacts["candidate"]
    overlap = artifacts["overlap"]
    residual = artifacts["residual"]
    postfreeze = artifacts["postfreeze"]
    probe_curve = horizon["arms"]["probe"]["curve"]
    best_probe_success = max(
        float(row["summary"]["fault"]["strict_success"]) for row in probe_curve
    )
    earliest_best_horizon = min(
        int(row["horizon"])
        for row in probe_curve
        if float(row["summary"]["fault"]["strict_success"]) == best_probe_success
    )
    horizon_seed = artifacts.get("horizon_seed")
    sequential = artifacts.get("sequential")
    candidate_seed = artifacts.get("candidate_seed")
    temporal8_partition = overlap["partitions"][
        "budget8_temporal_vs_max_candidate_union"
    ]
    temporal8_recovery_ids = set(temporal8_partition["both"]) | set(
        temporal8_partition["temporal_only"]
    )
    stable_candidate_ids = (
        set()
        if candidate_seed is None
        else set(candidate_seed["stable_union_success_ids_all_seeds"])
    )
    any_candidate_ids = (
        set()
        if candidate_seed is None
        else set(candidate_seed["union_success_ids_any_seed"])
    )
    sequential_recovery_sets = (
        []
        if sequential is None
        else [
            set(row["sequential_recovery_episode_ids"])
            for row in sequential["seeds"]
        ]
    )
    sequential_recovery_any = (
        set() if not sequential_recovery_sets else set.union(*sequential_recovery_sets)
    )
    sequential_recovery_stable = (
        set()
        if not sequential_recovery_sets
        else set.intersection(*sequential_recovery_sets)
    )
    fresh_holdout = artifacts.get("fresh_holdout")
    fresh_holdout_b = artifacts.get("fresh_holdout_b")
    fresh_replication = artifacts.get("fresh_replication")
    fresh_matched = artifacts.get("fresh_matched")
    fresh_matched_manifest = artifacts.get("fresh_matched_manifest")
    fresh_seed_setup = artifacts.get("fresh_seed_setup")
    fresh_failure_stability = artifacts.get("fresh_failure_stability")
    cohort_shift = artifacts.get("cohort_shift")
    fresh_generalization_supported = None
    if fresh_seed_setup is not None:
        fresh_generalization_supported = bool(
            fresh_seed_setup["all_suite_seed_cells_positive_over_fixed4"]
            and fresh_seed_setup["all_planner_seeds_positive_over_fixed4"]
            and float(
                fresh_seed_setup[
                    "two_way_seed_group_bootstrap_gain_over_fixed4"
                ]["low"]
            )
            > 0.0
        )
    elif fresh_replication is not None:
        fresh_generalization_supported = bool(
            fresh_replication["all_suites_positive_over_fixed4"]
            and float(
                fresh_replication["pooled_group_bootstrap_gain_over_fixed4"]["low"]
            )
            > 0.0
        )
    elif fresh_holdout is not None:
        fresh_generalization_supported = bool(
            float(fresh_holdout["frozen_sequential_gain_over_fixed4"]["low"])
            > 0.0
        )
    recommended_maximum_horizon = earliest_best_horizon
    if horizon_seed is not None:
        robust_horizon = horizon_seed["arms"]["probe"][
            "earliest_horizon_matching_every_seed_h8_success"
        ]
        if robust_horizon is not None:
            recommended_maximum_horizon = int(robust_horizon)
    maximum_candidate = candidate["nested_curve"][-1]
    maximum_pairing = candidate["maximum_budget_pairing"]
    candidate_union_successes = (
        len(maximum_pairing["both_success"])
        + len(maximum_pairing["uniform_only_success"])
        + len(maximum_pairing["conditioned_only_success"])
    )
    guarded = postfreeze["ranked_bottlenecks"][2]
    qwen = postfreeze["ranked_bottlenecks"][3]
    original_reranking = residual["candidate_reranking"]["original"]
    constant_reranking = residual["candidate_reranking"]["constant_oof"]
    ridge_reranking = residual["candidate_reranking"]["ridge_oof"]

    ranked = [
        {
            "rank": 1,
            "bottleneck": (
                "the fixed four-step stopping horizon truncates still-improving control"
                if fresh_generalization_supported is not False
                else "temporal continuation headroom does not generalize reliably to fresh setups"
            ),
            "evidence": {
                "fresh_generalization_supported": fresh_generalization_supported,
                "base_seed_fixed6_fault_gain_pp": 100.0
                * float(
                    artifacts["temporal_base"]["arms"]["probe"][
                        "fixed_budget6_over_budget4_fault"
                    ]["estimate"]
                ),
                "planner_seed_count": len(temporal_seed["per_seed"]),
                "fixed6_all_observed_seeds_positive": bool(
                    temporal_seed["all_observed_seeds_positive"]["fixed6"]
                ),
                "adaptive_all_observed_seeds_positive": bool(
                    temporal_seed["all_observed_seeds_positive"]["adaptive"]
                ),
                "adaptive_seed_group_interval_pp": [
                    100.0
                    * float(
                        temporal_seed["two_way_seed_group_bootstrap"]["adaptive"][key]
                    )
                    for key in ("low", "high")
                ],
                "earliest_base_seed_horizon_at_maximum_success": earliest_best_horizon,
                "base_seed_maximum_fault_success": best_probe_success,
                "horizon_planner_seed_count": (
                    1
                    if horizon_seed is None
                    else len(horizon_seed["planner_seeds"])
                ),
                "multi_seed_all_positive_h6_vs_h4": (
                    None
                    if horizon_seed is None
                    else bool(
                        horizon_seed["arms"]["probe"][
                            "all_seeds_positive_h6_vs_h4"
                        ]
                    )
                ),
                "multi_seed_earliest_horizon_matching_every_seed_h8_success": (
                    None
                    if horizon_seed is None
                    else horizon_seed["arms"]["probe"][
                        "earliest_horizon_matching_every_seed_h8_success"
                    ]
                ),
                "sequential_rule": (
                    None
                    if sequential is None
                    else sequential["frozen_full_selection_seed_rule"]
                ),
                "sequential_rule_seed_group_interval_pp": (
                    None
                    if sequential is None
                    else [
                        100.0
                        * float(
                            sequential[
                                "two_way_seed_group_bootstrap_gain_over_fixed4"
                            ][key]
                        )
                        for key in ("low", "high")
                    ]
                ),
                "sequential_rule_all_observed_seeds_positive": (
                    None
                    if sequential is None
                    else bool(sequential["all_observed_seeds_positive_over_fixed4"])
                ),
                "sequential_rule_recovery_seed_stability": (
                    None
                    if not sequential_recovery_sets
                    else {
                        "planner_seeds": len(sequential_recovery_sets),
                        "recoveries_by_seed": [
                            len(values) for values in sequential_recovery_sets
                        ],
                        "stable_recoveries_all_seeds": len(
                            sequential_recovery_stable
                        ),
                        "recoveries_any_seed": len(sequential_recovery_any),
                        "stable_to_any_recovery_ratio": (
                            1.0
                            if not sequential_recovery_any
                            else len(sequential_recovery_stable)
                            / len(sequential_recovery_any)
                        ),
                    }
                ),
                "fresh_nonprotected_setup_holdout": (
                    None
                    if fresh_holdout is None
                    else {
                        "setups": int(
                            fresh_holdout["setup_independence"]["fresh_groups"]
                        ),
                        "group_id_overlap": int(
                            fresh_holdout["setup_independence"]["group_id_overlap"]
                        ),
                        "setup_hash_overlap": int(
                            fresh_holdout["setup_independence"]["setup_hash_overlap"]
                        ),
                        "gain_over_fixed4_pp": 100.0
                        * float(
                            fresh_holdout[
                                "frozen_sequential_gain_over_fixed4"
                            ]["estimate"]
                        ),
                        "gain_interval_pp": [
                            100.0
                            * float(
                                fresh_holdout[
                                    "frozen_sequential_gain_over_fixed4"
                                ][key]
                            )
                            for key in ("low", "high")
                        ],
                        "recoveries": int(fresh_holdout["recoveries"]),
                        "regressions": int(fresh_holdout["regressions"]),
                        "probe_gain_accuracy": float(
                            fresh_holdout["probe_gain_classification"][
                                "overall_accuracy"
                            ]
                        ),
                        "boundary_fault_gain_accuracy": float(
                            fresh_holdout["probe_gain_classification"][
                                "boundary_fault_accuracy"
                            ]
                        ),
                    }
                ),
                "fresh_nonprotected_setup_replication": (
                    None
                    if fresh_replication is None
                    else {
                        "suites": int(fresh_replication["suite_count"]),
                        "setups": int(
                            fresh_replication["independent_setup_groups"]
                        ),
                        "all_suites_positive": bool(
                            fresh_replication["all_suites_positive_over_fixed4"]
                        ),
                        "pooled_gain_over_fixed4_pp": 100.0
                        * float(
                            fresh_replication[
                                "pooled_group_bootstrap_gain_over_fixed4"
                            ]["estimate"]
                        ),
                        "pooled_gain_interval_pp": [
                            100.0
                            * float(
                                fresh_replication[
                                    "pooled_group_bootstrap_gain_over_fixed4"
                                ][key]
                            )
                            for key in ("low", "high")
                        ],
                        "total_recoveries": int(
                            fresh_replication["total_recoveries"]
                        ),
                        "total_regressions": int(
                            fresh_replication["total_regressions"]
                        ),
                        "mean_probe_gain_accuracy": float(
                            fresh_replication["mean_probe_gain_accuracy"]
                        ),
                        "minimum_probe_gain_accuracy": float(
                            fresh_replication["minimum_probe_gain_accuracy"]
                        ),
                    }
                ),
                "fresh_nonprotected_difficulty_matched_holdout": (
                    None
                    if fresh_matched is None
                    else {
                        "setups": int(
                            fresh_matched["setup_independence"]["fresh_groups"]
                        ),
                        "group_id_overlap": int(
                            fresh_matched["setup_independence"]["group_id_overlap"]
                        ),
                        "setup_hash_overlap": int(
                            fresh_matched["setup_independence"]["setup_hash_overlap"]
                        ),
                        "outcome_free_selection": bool(
                            fresh_matched_manifest is not None
                            and fresh_matched_manifest[
                                "outcome_free_difficulty_matching"
                            ]["selection_or_retuning_on_candidate_outcomes"]
                            is False
                        ),
                        "mean_absolute_log1p_distance_mismatch": (
                            None
                            if fresh_matched_manifest is None
                            else float(
                                fresh_matched_manifest[
                                    "outcome_free_difficulty_matching"
                                ]["mean_absolute_log1p_distance_mismatch"]
                            )
                        ),
                        "maximum_absolute_log1p_distance_mismatch": (
                            None
                            if fresh_matched_manifest is None
                            else float(
                                fresh_matched_manifest[
                                    "outcome_free_difficulty_matching"
                                ]["maximum_absolute_log1p_distance_mismatch"]
                            )
                        ),
                        "gain_over_fixed4_pp": 100.0
                        * float(
                            fresh_matched[
                                "frozen_sequential_gain_over_fixed4"
                            ]["estimate"]
                        ),
                        "gain_interval_pp": [
                            100.0
                            * float(
                                fresh_matched[
                                    "frozen_sequential_gain_over_fixed4"
                                ][key]
                            )
                            for key in ("low", "high")
                        ],
                        "recoveries": int(fresh_matched["recoveries"]),
                        "regressions": int(fresh_matched["regressions"]),
                    }
                ),
                "fresh_nonprotected_seed_setup_robustness": (
                    None
                    if fresh_seed_setup is None
                    else {
                        "suites": len(fresh_seed_setup["suites"]),
                        "planner_seeds": len(
                            fresh_seed_setup["planner_root_seeds"]
                        ),
                        "setups": int(
                            fresh_seed_setup["independent_setup_groups"]
                        ),
                        "all_suite_seed_cells_positive": bool(
                            fresh_seed_setup[
                                "all_suite_seed_cells_positive_over_fixed4"
                            ]
                        ),
                        "all_planner_seeds_positive": bool(
                            fresh_seed_setup[
                                "all_planner_seeds_positive_over_fixed4"
                            ]
                        ),
                        "two_way_gain_over_fixed4_pp": 100.0
                        * float(
                            fresh_seed_setup[
                                "two_way_seed_group_bootstrap_gain_over_fixed4"
                            ]["estimate"]
                        ),
                        "two_way_gain_interval_pp": [
                            100.0
                            * float(
                                fresh_seed_setup[
                                    "two_way_seed_group_bootstrap_gain_over_fixed4"
                                ][key]
                            )
                            for key in ("low", "high")
                        ],
                    }
                ),
                "fresh_exact_failure_stability": (
                    None
                    if fresh_failure_stability is None
                    else {
                        "fault_episodes_per_seed": int(
                            fresh_failure_stability["fault_episodes_per_seed"]
                        ),
                        "failures_by_seed": fresh_failure_stability[
                            "failures_by_seed"
                        ],
                        "stable_failures": len(
                            fresh_failure_stability[
                                "stable_failure_ids_all_seeds"
                            ]
                        ),
                        "stable_failure_groups": len(
                            fresh_failure_stability[
                                "stable_failure_group_ids_all_seeds"
                            ]
                        ),
                        "fault_outcome_agreement_rate": float(
                            fresh_failure_stability[
                                "fault_outcome_agreement_rate"
                            ]
                        ),
                    }
                ),
                "fresh_setup_cohort_shift": (
                    None
                    if cohort_shift is None
                    else {
                        "comparisons_to_original": cohort_shift[
                            "comparisons_to_reference"
                        ],
                        "pairwise_group_id_overlaps": cohort_shift[
                            "pairwise_group_id_overlaps"
                        ],
                        "interpretation": (
                            "The unconstrained fresh A/B cohorts have easier fixed-four "
                            "baselines and lower median initial distances than the original. "
                            "The outcome-free within-stratum matched cohort, when present, "
                            "separately tests the frozen rule after reducing that distance shift."
                        ),
                    }
                ),
            },
            "estimated_achievable_gain": {
                "kind": (
                    "observed_frozen_sequential_rule_multi_seed_multi_setup_gain"
                    if fresh_seed_setup is not None
                    else "observed_frozen_sequential_rule_two_fresh_suite_pooled_gain"
                    if fresh_replication is not None
                    else "observed_frozen_sequential_rule_fresh_setup_holdout_gain"
                    if fresh_holdout is not None
                    else "observed_held_sequential_rule_multi_seed_development_gain"
                    if sequential is not None
                    else "observed_held_rule_multi_seed_development_gain"
                ),
                "fault_success_points": 100.0
                * float(
                    (
                        fresh_seed_setup[
                            "two_way_seed_group_bootstrap_gain_over_fixed4"
                        ]["estimate"]
                        if fresh_seed_setup is not None
                        else fresh_replication[
                            "pooled_group_bootstrap_gain_over_fixed4"
                        ]["estimate"]
                        if fresh_replication is not None
                        else fresh_holdout["frozen_sequential_gain_over_fixed4"][
                            "estimate"
                        ]
                        if fresh_holdout is not None
                        else sequential[
                            "two_way_seed_group_bootstrap_gain_over_fixed4"
                        ]["estimate"]
                        if sequential is not None
                        else temporal_seed["two_way_seed_group_bootstrap"]["adaptive"][
                            "estimate"
                        ]
                    )
                ),
            },
            "smallest_recommended_next_experiment": (
                (
                    "On an externally held setup generator or hardware, use 30 groups "
                    "(ten per stratum) and two preregistered planner seeds for the same "
                    f"visible rule versus four steps, capped at horizon {recommended_maximum_horizon}; "
                    "require a positive two-way bootstrap lower bound, score total steps "
                    "and saturation under predeclared margins, and do not retune."
                    if fresh_generalization_supported is True
                    else "Keep the four-step default and mine the exact fresh-suite rule failures; "
                    "test one visible distribution-shift guard on a new preregistered suite before "
                    "any horizon increase."
                    if fresh_generalization_supported is False
                    else f"On new non-protected setup groups, preregister the frozen visible "
                    f"sequential rule and compare it with four steps, capped at horizon "
                    f"{recommended_maximum_horizon}; score strict success, total steps, and "
                    "saturation without retuning the rule."
                )
            ),
        },
        {
            "rank": 2,
            "bottleneck": "remaining boundary failures need better action-sequence coverage, not broad CEM knob changes",
            "evidence": {
                "selected_failures": int(candidate["episodes"]),
                "maximum_candidate_budget": int(maximum_candidate["candidate_budget"]),
                "uniform_best_of_k_successes": int(
                    maximum_candidate["proposals"]["uniform_feasible"][
                        "strict_successes"
                    ]
                ),
                "conditioned_best_of_k_successes": int(
                    maximum_candidate["proposals"]["boundary_conditioned"][
                        "strict_successes"
                    ]
                ),
                "proposal_union_successes": candidate_union_successes,
                "proposal_seed_count": (
                    1
                    if candidate_seed is None
                    else len(candidate_seed["proposal_seeds"])
                ),
                "proposal_union_successes_by_seed_at_maximum_common_budget": (
                    None
                    if candidate_seed is None
                    else candidate_seed["proposal_union_successes_by_seed"]
                ),
                "stable_proposal_union_successes_all_seeds": (
                    None
                    if candidate_seed is None
                    else len(candidate_seed["stable_union_success_ids_all_seeds"])
                ),
                "proposal_union_successes_any_seed": (
                    None
                    if candidate_seed is None
                    else len(candidate_seed["union_success_ids_any_seed"])
                ),
                "stable_to_any_proposal_union_recovery_ratio": (
                    None
                    if candidate_seed is None
                    else (
                        1.0
                        if not candidate_seed["union_success_ids_any_seed"]
                        else len(candidate_seed["stable_union_success_ids_all_seeds"])
                        / len(candidate_seed["union_success_ids_any_seed"])
                    )
                ),
                "stable_proposal_recoveries_beyond_temporal8_all_seeds": (
                    None
                    if candidate_seed is None
                    else sorted(stable_candidate_ids - temporal8_recovery_ids)
                ),
                "any_seed_proposal_recoveries_beyond_temporal8": (
                    None
                    if candidate_seed is None
                    else sorted(any_candidate_ids - temporal8_recovery_ids)
                ),
                "proposal_union_seed_stability_by_budget": (
                    None
                    if candidate_seed is None
                    else [
                        {
                            "candidate_budget": int(row["candidate_budget"]),
                            "successes_by_seed": row["proposal_union"][
                                "successes_by_seed"
                            ],
                            "stable_successes": len(
                                row["proposal_union"][
                                    "stable_success_ids_all_seeds"
                                ]
                            ),
                            "successes_any_seed": len(
                                row["proposal_union"]["success_ids_any_seed"]
                            ),
                            "stable_to_any_success_ratio": float(
                                row["proposal_union"][
                                    "stable_to_any_success_ratio"
                                ]
                            ),
                        }
                        for row in candidate_seed["seed_robustness_curve"]
                        if "proposal_union" in row
                    ]
                ),
                "proposal_union_maximum_budget_recovery_slices": (
                    None
                    if candidate_seed is None
                    else candidate_seed.get("maximum_budget_recovery_slices")
                ),
                "proposal_union_maximum_budget_seed_recovery_frequency": (
                    None
                    if candidate_seed is None
                    else candidate_seed.get(
                        "maximum_budget_seed_recovery_frequency"
                    )
                ),
                "minimum_seed_candidate_union_fault_upper_bound_points": (
                    None
                    if candidate_seed is None
                    else float(
                        candidate_seed[
                            "minimum_seed_candidate_union_fault_upper_bound_points"
                        ]
                    )
                ),
                "conditioned_minus_uniform_success_interval": [
                    float(
                        maximum_candidate["conditioned_minus_uniform_success"][key]
                    )
                    for key in ("low", "high")
                ],
                "conditioning_reach": candidate["conditioning_reach"],
                "candidate_only_beyond_temporal8": int(
                    overlap["mechanism_interpretation"]["candidate_only_recoveries"]
                ),
                "temporal8_only": int(
                    overlap["mechanism_interpretation"]["temporal_only_recoveries"]
                ),
                "shared_with_temporal8": int(
                    overlap["mechanism_interpretation"]["shared_recoveries"]
                ),
                "neither_mechanism_recovers": len(
                    overlap["partitions"][
                        "budget8_temporal_vs_max_candidate_union"
                    ]["neither"]
                ),
            },
            "estimated_achievable_gain": {
                "kind": "candidate_only_incremental_simulator_oracle_upper_bound_beyond_temporal8",
                "fault_success_points": 100.0
                * int(overlap["mechanism_interpretation"]["candidate_only_recoveries"])
                / 120.0,
            },
            "smallest_recommended_next_experiment": (
                "Serialize the simulator-best sequences only as evaluator labels, fit a visible "
                "boundary action-proposal model on development groups, and require held-group "
                "coverage gain before changing H1 or CEM scoring."
            ),
        },
        {
            "rank": 3,
            "bottleneck": "visible residual calibration does not repair retained-candidate ordering",
            "evidence": {
                "original_strict_success": float(original_reranking["strict_success"]),
                "constant_strict_success": float(constant_reranking["strict_success"]),
                "ridge_strict_success": float(ridge_reranking["strict_success"]),
                "original_actual_best_selection": float(
                    original_reranking["actual_best_selection_rate"]
                ),
                "constant_actual_best_selection": float(
                    constant_reranking["actual_best_selection_rate"]
                ),
                "ridge_actual_best_selection": float(
                    ridge_reranking["actual_best_selection_rate"]
                ),
            },
            "estimated_achievable_gain": {
                "kind": "supported_no_positive_gain_from_tested_calibrators",
                "fault_success_points": 0.0,
            },
            "smallest_recommended_next_experiment": (
                "Do not deploy a residual reranker; collect new multi-step boundary transitions "
                "and test whether forward-model retraining improves held-group candidate coverage."
            ),
        },
        {
            "rank": 4,
            "bottleneck": "continuous gain compensation adds actuator risk without robust incremental value",
            "evidence": guarded["evidence"],
            "estimated_achievable_gain": guarded["estimated_achievable_gain"],
            "smallest_recommended_next_experiment": guarded["smallest_next_experiment"],
        },
        {
            "rank": 5,
            "bottleneck": "larger Qwen capacity is not the limiting observable",
            "evidence": qwen["evidence"],
            "estimated_achievable_gain": qwen["estimated_achievable_gain"],
            "smallest_recommended_next_experiment": qwen["smallest_next_experiment"],
        },
    ]
    probe_budget8 = _read_jsonl(development / "control/probe_nores_budget8.jsonl")
    direct_budget8 = _read_jsonl(development / "control/direct_budget8.jsonl")
    exact_ids = {
        "probe_fault_failures_after_budget8": sorted(
            _episode_id(row)
            for row in probe_budget8
            if float(row["evaluator_only_true_gain"]) != 1.0
            and not bool(row["strict_success"])
        ),
        "direct_fault_failures_after_budget8": sorted(
            _episode_id(row)
            for row in direct_budget8
            if float(row["evaluator_only_true_gain"]) != 1.0
            and not bool(row["strict_success"])
        ),
        "candidate_neither_proposal_succeeds_at_maximum_budget": maximum_pairing[
            "neither_success"
        ],
        "candidate_uniform_only_success_at_maximum_budget": maximum_pairing[
            "uniform_only_success"
        ],
        "candidate_conditioned_only_success_at_maximum_budget": maximum_pairing[
            "conditioned_only_success"
        ],
        "neither_temporal8_nor_candidate_union_recovers": overlap["partitions"][
            "budget8_temporal_vs_max_candidate_union"
        ]["neither"],
    }
    if fresh_holdout is not None:
        exact_ids["fresh_suite_a_sequential_fault_failures"] = fresh_holdout[
            "exact_ids"
        ]["frozen_sequential_fault_failures"]
    if fresh_holdout_b is not None:
        exact_ids["fresh_suite_b_sequential_fault_failures"] = fresh_holdout_b[
            "exact_ids"
        ]["frozen_sequential_fault_failures"]
    if fresh_matched is not None:
        exact_ids["fresh_matched_sequential_fault_failures"] = fresh_matched[
            "exact_ids"
        ]["frozen_sequential_fault_failures"]
    if fresh_failure_stability is not None:
        exact_ids["fresh_stable_sequential_fault_failures_both_seeds"] = (
            fresh_failure_stability["stable_failure_ids_all_seeds"]
        )
        exact_ids["fresh_planner_seed_discordant_failures"] = (
            fresh_failure_stability["planner_seed_discordant_failure_ids"]
        )
    report = {
        "version": VERSION,
        "role": "final_research_decision_primary_branch_not_reselected",
        "selected_primary_branch": "A",
        "branch_reason": artifacts["gate"]["branch_reason"],
        "exactly_one_primary_branch_selected": bool(
            artifacts["gate"]["exactly_one_primary_branch_selected"]
        ),
        "protected_set_used_for_reselection": False,
        "recommended_policy_now": {
            "policy": "probe_symmetric_pair_f0p1_no_residual",
            "status": "keep the frozen Branch-A policy",
            "reason": (
                "it passed the registered development resolution and the single protected "
                "confirmation; all temporal and forensic results are supporting evidence only"
            ),
        },
        "ranked_bottlenecks": ranked,
        "exact_failing_setup_ids": exact_ids,
        "actionable_next_decision": {
            "priority": ranked[0]["smallest_recommended_next_experiment"],
            "secondary": ranked[1]["smallest_recommended_next_experiment"],
            "fresh_evidence_supports_sequential_horizon_change": (
                fresh_generalization_supported
            ),
            "stop_more_broad_cem_knob_sweeps": True,
            "stop_scaling_qwen_on_current_visible_features": True,
            "stop_deploying_continuous_gain_compensation_without_a_safety_utility": True,
            "do_not_retrain_frozen_h1_from_this_run": True,
        },
        "reproduction_commands": {
            "temporal_seed": (
                "python -m active_diagnosis_v13.analyze_temporal_seed_statistics "
                f"--output-dir {development} --seed-report "
                f"{development/('control_budget_five_seed.json' if 'five_seed' in temporal_seed_path.name else 'control_budget_three_seed.json')} --output "
                f"{temporal_seed_path}"
            ),
            "horizon": (
                "python -m active_diagnosis_v13.analyze_control_horizon_curve "
                f"--direct-budget4 {development/'control/direct.jsonl'} "
                f"--direct-budget6 {development/'control/direct_budget6.jsonl'} "
                f"--direct-budget8 {development/'control/direct_budget8.jsonl'} "
                f"--probe-budget4 {development/'control/probe_symmetric_pair_f0p1_no_residual.jsonl'} "
                f"--probe-budget6 {development/'control/probe_nores_budget6.jsonl'} "
                f"--probe-budget8 {development/'control/probe_nores_budget8.jsonl'} "
                f"--output {development/'control_horizon_curve.json'}"
            ),
            "boundary_candidate_audit": (
                "python -m active_diagnosis_v13.audit_boundary_candidate_coverage "
                f"--records {development/'boundary_candidate_coverage_records.jsonl'} "
                "--budgets 8 24 48 96 192 "
                f"--output {development/'boundary_candidate_coverage_audit.json'}"
            ),
            "residual_calibration": (
                "python -m active_diagnosis_v13.analyze_oof_candidate_residual_calibration "
                "--episode-dir runs/v12_mpc_h1_h3_diagnosis_20260731_111829/episodes/primary "
                "--v12-config continuous_control_v12/config_v12_semantics_v2.json "
                f"--predictions {development/'oof_candidate_residual_predictions.jsonl'} "
                f"--output {development/'oof_candidate_residual_calibration.json'}"
            ),
            "fresh_failure_stability": (
                None
                if fresh_failure_stability is None
                else "See the exact six-input command under balanced planner-seed "
                f"replication in {run/'commands/exact_commands.md'}."
            ),
        },
        "limitations": [
            "The primary policy remains the already-frozen Branch-A policy; post-freeze evidence cannot reselect it.",
            "Candidate best-of-k results use evaluator-only simulator scores on selected failures and are upper bounds, not policy results.",
            (
                "The temporal continuation rule was selected on seed-1 development groups and only planner-seed, not new-setup, replication is available in this run."
                if fresh_holdout is None
                else (
                    f"The {len(fresh_seed_setup['suites'])} fresh-setup temporal replications are mutually independent by group ID and setup hash and use {len(fresh_seed_setup['planner_root_seeds'])} planner seeds, but remain in the same simulator family."
                    if fresh_seed_setup is not None
                    else "The two fresh-setup temporal replications are mutually independent by group ID and setup hash but use the same simulator family."
                    if fresh_replication is not None
                    else "The fresh-setup temporal replication is independent by group ID and setup hash but uses the same simulator family."
                )
            ),
            (
                "The two random fresh cohorts have 10–15 point higher fixed-four success and 6.9–9.4 lower median initial normalized distance than the original cohort; their pooled gain is therefore new-setup, not equal-difficulty, replication."
                if cohort_shift is not None
                else "Fresh setup difficulty has not been directly compared with the original cohort."
            ),
            (
                "Candidate coverage was measured with one proposal seed."
                if candidate_seed is None
                else "Candidate coverage proposal-seed replication still uses the same selected development failures."
            ),
        ],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "final_research_decision.json", report)
    lines = [
        "# Final v13 research decision",
        "",
        "Primary Branch A remains frozen; protected evidence was confirmatory and was not used for reselection.",
        "",
        "## Ranked bottlenecks",
        "",
        "| Rank | Bottleneck | Estimated gain | Smallest recommended next experiment |",
        "|---:|---|---:|---|",
    ]
    for row in ranked:
        estimate = row["estimated_achievable_gain"]
        lines.append(
            f"| {row['rank']} | {row['bottleneck']} | "
            f"{float(estimate['fault_success_points']):+.1f} pp ({estimate['kind']}) | "
            f"{row['smallest_recommended_next_experiment']} |"
        )
    lines.extend(
        [
            "",
            "## Actionable decision",
            "",
            f"1. {report['actionable_next_decision']['priority']}",
            f"2. {report['actionable_next_decision']['secondary']}",
            "3. Keep H1 and the frozen discrete Branch-A policy unchanged for this result.",
        ]
    )
    (output / "final_research_decision.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
