#!/usr/bin/env python3
"""Synthesize the post-freeze development decision and ranked next experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


VERSION = "active_diagnosis_v13_postfreeze_decision_v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text())


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _interval(block: dict[str, Any]) -> list[float]:
    return [float(block["low"]), float(block["high"])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    development = run / "development"
    artifacts = {
        "taxonomy": _read(development / "failure_taxonomy/matched_failure_taxonomy.json"),
        "candidate": _read(
            run / "fallback_preparation/candidate_coverage_dev/candidate_coverage.json"
        ),
        "forward": _read(
            run
            / "fallback_preparation/forward_calibration_dev/forward_uncertainty_calibration.json"
        ),
        "risk": _read(development / "cem_risk_sensitivity.json"),
        "population": _read(development / "cem_population48_three_seed.json"),
        "diversity": _read(development / "cem_elite_diversity_0p25.json"),
        "feasible": _read(development / "cem_feasible_proposal_resample8.json"),
        "guarded": _read(development / "guarded_margin_0p125_eight_seed.json"),
        "guarded_frontier": _read(
            development / "guarded_margin_eight_seed_frontier.json"
        ),
        "qwen": _read(run / "branch_a/qwen_closed_loop_validation_comparison.json"),
        "resolution": _read(development / "branch_a_resolution.json"),
    }
    for name, artifact in artifacts.items():
        if artifact.get("protected_set_used") is True or artifact.get(
            "protected_set_used_for_selection"
        ) is True:
            raise ValueError(f"{name} uses protected evidence")

    taxonomy = artifacts["taxonomy"]
    non_nominal = taxonomy["non_nominal"]
    boundary = taxonomy["by_stratum"]["reachable_boundary_or_clipping"]
    candidate = artifacts["candidate"]["all_closed_loop_decisions"]
    forward = artifacts["forward"]
    risk = artifacts["risk"]
    population = artifacts["population"]
    diversity = artifacts["diversity"]
    feasible = artifacts["feasible"]
    guarded = artifacts["guarded"]
    qwen = artifacts["qwen"]
    qwen_policies = {row["policy"]: row for row in qwen["policies"]}

    controls = [
        {
            "treatment": "uncertainty weight 0/0.1/0.25",
            "scope": "150 matched development episodes",
            "fault_success_difference_pp": 0.0,
            "interval_95_pp": None,
            "safety_or_geometry_effect": (
                "15 and 17 plans change at weights 0 and 0.25, but all 120 fault outcomes are identical"
            ),
            "decision": "behaviorally active but zero fault-outcome leverage",
        },
        {
            "treatment": "CEM population 24 to 48",
            "scope": "three matched planner seeds",
            "fault_success_difference_pp": 100.0
            * float(population["fault_success_gain"]["mean"]),
            "interval_95_pp": [
                100.0
                * float(
                    population["fault_success_gain"]["planner_seed_bootstrap_mean_95"][
                        "low"
                    ]
                ),
                100.0
                * float(
                    population["fault_success_gain"]["planner_seed_bootstrap_mean_95"][
                        "high"
                    ]
                ),
            ],
            "safety_or_geometry_effect": (
                "seed effects +3.33/-2.50/+2.50 pp; boundary effects +5.0/-7.5/0.0 pp"
            ),
            "decision": "small, directionally unstable, below five points",
        },
        {
            "treatment": "minimum normalized elite distance 0.25",
            "scope": "150 matched development episodes",
            "fault_success_difference_pp": 100.0
            * float(diversity["fault"]["success_difference"]),
            "interval_95_pp": [
                100.0 * value
                for value in _interval(diversity["fault"]["matched_group_bootstrap_95"])
            ],
            "safety_or_geometry_effect": (
                f"minimum elite separation rises to {diversity['on_planner_diagnostics']['mean_elite_minimum_normalized_action_distance']:.3f}; "
                f"saturation changes {100*diversity['fault']['off_saturation_episode_rate']:.1f}% to "
                f"{100*diversity['fault']['on_saturation_episode_rate']:.1f}%"
            ),
            "decision": "effective diversity treatment, harmful control result",
        },
        {
            "treatment": "eight-attempt feasible proposal resampling",
            "scope": "150 matched development episodes",
            "fault_success_difference_pp": 100.0
            * float(feasible["fault"]["success_difference"]),
            "interval_95_pp": [
                100.0 * value
                for value in _interval(feasible["fault"]["matched_group_bootstrap_95"])
            ],
            "safety_or_geometry_effect": (
                "residual infeasible proposal mass falls to zero; fault saturation changes "
                f"{100*feasible['fault']['off_saturation_episode_rate']:.1f}% to "
                f"{100*feasible['fault']['on_saturation_episode_rate']:.1f}%"
            ),
            "decision": "cleaner sampler, small uncertain gain, zero boundary gain",
        },
    ]

    ranked = [
        {
            "rank": 1,
            "bottleneck": "successful trajectories remain absent in boundary and multi-step failures",
            "evidence": {
                "fault_episodes_unrecovered_even_with_oracle_gain": int(
                    non_nominal["categories"]["unrecovered_even_with_oracle_gain"]
                ),
                "boundary_unrecovered_even_with_oracle_gain": int(
                    boundary["categories"]["unrecovered_even_with_oracle_gain"]
                ),
                "retained_top10_best_of_k_gain_pp": 100.0
                * float(candidate["best_of_k_success_gain"]),
                "population48_three_seed_mean_gain_pp": 100.0
                * float(population["fault_success_gain"]["mean"]),
                "diversity_gain_pp": 100.0
                * float(diversity["fault"]["success_difference"]),
                "feasible_proposal_gain_pp": 100.0
                * float(feasible["fault"]["success_difference"]),
                "feasible_proposal_boundary_gain_pp": 100.0
                * float(
                    feasible["by_stratum"]["reachable_boundary_or_clipping"][
                        "success_difference"
                    ]
                ),
            },
            "estimated_achievable_gain": {
                "kind": "oracle_failure_upper_bound_not_observed_gain",
                "fault_success_points": 100.0
                * int(non_nominal["categories"]["unrecovered_even_with_oracle_gain"])
                / int(non_nominal["episodes"]),
            },
            "smallest_next_experiment": (
                "On only the exact development boundary episodes unrecovered even with oracle gain, "
                "execute simulator-scored candidates from a boundary-conditioned action proposal and "
                "report best-of-budget coverage before changing H1 or training a reranker."
            ),
        },
        {
            "rank": 2,
            "bottleneck": "centroid forward error dominates strict nominal failures",
            "evidence": {
                "strict_nominal_failures": int(
                    forward["strict_all_five_failure_decomposition"]["failed_episodes"]
                ),
                "centroid_x_failure_count": int(
                    forward["strict_all_five_failure_decomposition"][
                        "metric_failure_counts"
                    ]["centroid_x_px"]
                ),
                "centroid_x_error_uncertainty_spearman": float(
                    forward["per_metric_forward_calibration"]["centroid_x_px"][
                        "absolute_error_uncertainty_spearman"
                    ]
                ),
            },
            "estimated_achievable_gain": {
                "kind": "nominal_failure_contribution_upper_bound",
                "success_points": 100.0
                * int(
                    forward["strict_all_five_failure_decomposition"][
                        "metric_failure_counts"
                    ]["centroid_x_px"]
                )
                / int(forward["episodes"]),
            },
            "smallest_next_experiment": (
                "Fit a group-held-out residual calibrator for centroid x/y and rescore the exact "
                "nominal failures; require candidate-order improvement before retraining H1."
            ),
        },
        {
            "rank": 3,
            "bottleneck": "continuous gain compensation buys little robust value for substantial actuator risk",
            "evidence": {
                "guarded_gain_over_frozen_discrete_mean_pp": 100.0
                * float(guarded["gain_over_discrete"]["mean"]),
                "guarded_gain_over_frozen_discrete_seed_95_pp": [
                    100.0
                    * float(
                        guarded["gain_over_discrete"]["planner_seed_bootstrap_mean_95"][
                            key
                        ]
                    )
                    for key in ("low", "high")
                ],
                "guarded_fault_saturation_mean_percent": 100.0
                * float(guarded["fault_saturation_episode_rate"]["mean"]),
                "net_added_saturation_per_net_recovery": float(
                    guarded["aggregate_seed_episode_tradeoff"][
                        "net_added_saturation_per_net_recovery"
                    ]
                ),
            },
            "estimated_achievable_gain": {
                "kind": "observed_eight_seed_mean_over_frozen_discrete",
                "fault_success_points": 100.0
                * float(guarded["gain_over_discrete"]["mean"]),
            },
            "smallest_next_experiment": (
                "Keep the frozen discrete gain classes; if continuous compensation is revisited, "
                "optimize a preregistered success-saturation utility on new development groups."
            ),
        },
        {
            "rank": 4,
            "bottleneck": "larger Qwen policy is not the limiting estimator capacity",
            "evidence": {
                "qwen_validation_fault_success": float(
                    qwen_policies["probe_qwen_dev_validation"]["fault_success"]
                ),
                "frozen_linear_validation_fault_success": float(
                    qwen_policies["probe_symmetric_pair_f0p1_no_residual"][
                        "fault_success"
                    ]
                ),
                "qwen_boundary_fault_success": float(
                    qwen_policies["probe_qwen_dev_validation"]["by_stratum"][
                        "reachable_boundary_or_clipping"
                    ]["strict_success"]
                ),
            },
            "estimated_achievable_gain": {
                "kind": "observed_validation_difference_vs_frozen_linear",
                "fault_success_points": 100.0
                * (
                    float(qwen_policies["probe_qwen_dev_validation"]["fault_success"])
                    - float(
                        qwen_policies["probe_symmetric_pair_f0p1_no_residual"][
                            "fault_success"
                        ]
                    )
                ),
            },
            "smallest_next_experiment": (
                "Do not scale the language model; first add genuinely observable boundary evidence "
                "or improve action coverage."
            ),
        },
    ]
    exact_ids = {
        "fault_unrecovered_even_with_oracle_gain": taxonomy[
            "exact_episode_ids_by_category"
        ]["unrecovered_even_with_oracle_gain"],
        "boundary_unrecovered_even_with_oracle_gain": [
            episode_id
            for episode_id in taxonomy["exact_episode_ids_by_category"][
                "unrecovered_even_with_oracle_gain"
            ]
            if "_02_" in episode_id
        ],
        "nominal_forward_failures": [
            row["case_id"]
            for row in forward["strict_all_five_failure_decomposition"]["exact_failures"]
        ],
    }
    report = {
        "version": VERSION,
        "role": "postfreeze_development_synthesis_primary_branch_not_reselected",
        "split": "development_only",
        "protected_set_used": False,
        "selected_primary_branch": "A",
        "recommended_current_policy": {
            "policy": "probe_symmetric_pair_f0p1_no_residual",
            "reason": (
                "frozen discrete Branch-A estimator retains robust control value without the "
                "guarded mean's large and nonreplicating saturation tradeoff"
            ),
            "protected_result_used_for_reselection": False,
        },
        "controlled_cem_ablation_table": controls,
        "ranked_bottlenecks": ranked,
        "exact_failing_setup_ids": exact_ids,
        "actionable_decision": {
            "stop_broad_cem_knob_sweeps": True,
            "do_not_train_more_qwen_for_current_features": True,
            "do_not_replace_frozen_discrete_with_guarded_mean": True,
            "next_experiment": ranked[0]["smallest_next_experiment"],
        },
        "reproduction_commands": {
            "risk": (
                "python -m active_diagnosis_v13.analyze_cem_risk_sensitivity --gate-dir "
                f"{development} --arm 0:direct_uncertainty_off --arm 0.1:direct "
                f"--arm 0.25:direct_uncertainty_high --output {development/'cem_risk_sensitivity.json'}"
            ),
            "population": (
                "python -m active_diagnosis_v13.run_cem_seed_robustness --gate-dir "
                f"{development} --config active_diagnosis_v13/config_v13.json --population 48 "
                "--base-policy-name direct_population_48 --seed-policy-prefix direct_population48_seed "
                "--seeds 2026080101 2026080102 2026080103 --max-workers 2 "
                f"--output {development/'cem_population48_three_seed.json'}"
            ),
            "diversity": (
                "python -m active_diagnosis_v13.run_cem_diversity_ablation --gate-dir "
                f"{development} --config active_diagnosis_v13/config_v13.json --minimum-distance 0.25 "
                f"--max-workers 2 --output {development/'cem_elite_diversity_0p25.json'}"
            ),
            "feasible_proposals": (
                "python -m active_diagnosis_v13.run_cem_feasible_proposal_ablation --gate-dir "
                f"{development} --config active_diagnosis_v13/config_v13.json --resample-attempts 8 "
                f"--max-workers 2 --output {development/'cem_feasible_proposal_resample8.json'}"
            ),
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "postfreeze_research_decision.json", report)
    lines = [
        "# Post-freeze development research decision",
        "",
        "Primary Branch A remains frozen; protected results were not used for reselection.",
        "",
        "## Controlled CEM ablations",
        "",
        "| Treatment | Fault difference | 95% interval | Decision |",
        "|---|---:|---:|---|",
    ]
    for row in controls:
        interval = row["interval_95_pp"]
        interval_text = (
            "n/a" if interval is None else f"[{interval[0]:+.1f},{interval[1]:+.1f}]"
        )
        lines.append(
            f"| {row['treatment']} | {row['fault_success_difference_pp']:+.1f} pp | "
            f"{interval_text} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Ranked bottlenecks",
            "",
            "| Rank | Bottleneck | Headroom | Smallest next experiment |",
            "|---:|---|---:|---|",
        ]
    )
    for row in ranked:
        estimate = row["estimated_achievable_gain"]
        value = estimate.get("fault_success_points", estimate.get("success_points"))
        lines.append(
            f"| {row['rank']} | {row['bottleneck']} | {float(value):+.1f} pp "
            f"({estimate['kind']}) | {row['smallest_next_experiment']} |"
        )
    (output_dir / "postfreeze_research_decision.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
