#!/usr/bin/env python3
"""Synthesize ranked H1/CEM bottleneck evidence from development artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


VERSION = "active_diagnosis_v13_ranked_bottleneck_synthesis_v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--candidate-coverage", type=Path, required=True)
    parser.add_argument("--forward-calibration", type=Path, required=True)
    parser.add_argument("--cem-curve", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    candidate = _read(args.candidate_coverage)
    forward = _read(args.forward_calibration)
    taxonomy = _read(root / "failure_taxonomy/matched_failure_taxonomy.json")
    cem = _read(args.cem_curve)
    for name, artifact in (
        ("candidate coverage", candidate),
        ("forward calibration", forward),
        ("failure taxonomy", taxonomy),
        ("CEM curve", cem),
    ):
        if artifact.get("protected_set_used", False) or artifact.get(
            "protected_set_used_for_selection", False
        ):
            raise ValueError(f"{name} uses protected evidence")

    non_nominal = taxonomy["non_nominal"]
    categories = non_nominal["categories"]
    unrecovered = int(categories.get("unrecovered_even_with_oracle_gain", 0))
    missed_recoverable = int(
        categories.get("probe_missed_oracle_recoverable_failure", 0)
    )
    boundary = taxonomy["by_stratum"]["reachable_boundary_or_clipping"]
    multistep = taxonomy["by_stratum"]["multi_step_reachable_interior"]
    forward_failures = forward["strict_all_five_failure_decomposition"]
    metric_counts = forward_failures["metric_failure_counts"]
    calibration = forward["per_metric_forward_calibration"]
    all_candidates = candidate["all_closed_loop_decisions"]
    first_candidates = candidate["initial_decisions_only"]
    cem_arms = cem["arms"]
    best_cem = max(cem_arms, key=lambda row: float(row["fault"]["strict_success"]))

    ranked = [
        {
            "rank": 1,
            "bottleneck": "gain-aware planning still lacks a successful action/control trajectory",
            "evidence": {
                "fault_episodes_unrecovered_even_with_oracle_gain": unrecovered,
                "fault_episodes": int(non_nominal["episodes"]),
                "share_of_fault_episodes": unrecovered / int(non_nominal["episodes"]),
                "probe_missed_oracle_recoverable_cases": missed_recoverable,
            },
            "estimated_achievable_gain": {
                "kind": "upper_bound_not_point_estimate",
                "fault_success_points": 100.0 * unrecovered / int(non_nominal["episodes"]),
                "interpretation": "maximum remaining headroom if every oracle-gain failure were fixed",
            },
            "smallest_next_experiment": (
                "Replay the exact unrecovered IDs with simulator-scored actions from an expanded, "
                "boundary-aware candidate distribution; keep H1 frozen and measure best-of-budget coverage."
            ),
        },
        {
            "rank": 2,
            "bottleneck": "boundary/clipping and multi-step cases concentrate failures",
            "evidence": {
                "boundary_fault_success_direct": boundary["direct_success"],
                "boundary_fault_success_oracle": boundary["oracle_success"],
                "boundary_fault_success_probe": boundary["probe_replan_success"],
                "boundary_unrecovered_even_with_oracle": int(
                    boundary["categories"].get("unrecovered_even_with_oracle_gain", 0)
                ),
                "multistep_unrecovered_even_with_oracle": int(
                    multistep["categories"].get("unrecovered_even_with_oracle_gain", 0)
                ),
            },
            "estimated_achievable_gain": {
                "kind": "slice_upper_bound",
                "fault_success_points": 100.0
                * int(boundary["categories"].get("unrecovered_even_with_oracle_gain", 0))
                / int(non_nominal["episodes"]),
                "interpretation": "upper bound from the dominant boundary slice only",
            },
            "smallest_next_experiment": (
                "For the boundary IDs, compare the frozen sampler with a feasibility-preserving "
                "truncated proposal and report matched coverage, saturation, and success."
            ),
        },
        {
            "rank": 3,
            "bottleneck": "centroid forward error dominates strict nominal failures",
            "evidence": {
                "strict_nominal_failures": int(forward_failures["failed_episodes"]),
                "centroid_x_failure_count": int(metric_counts["centroid_x_px"]),
                "centroid_y_failure_count": int(metric_counts["centroid_y_px"]),
                "centroid_x_mean_absolute_normalized_error": calibration["centroid_x_px"][
                    "mean_absolute_normalized_error"
                ],
                "centroid_y_mean_absolute_normalized_error": calibration["centroid_y_px"][
                    "mean_absolute_normalized_error"
                ],
                "centroid_x_error_uncertainty_spearman": calibration["centroid_x_px"][
                    "absolute_error_uncertainty_spearman"
                ],
                "centroid_y_error_uncertainty_spearman": calibration["centroid_y_px"][
                    "absolute_error_uncertainty_spearman"
                ],
            },
            "estimated_achievable_gain": {
                "kind": "nominal_failure_contribution_upper_bound",
                "success_points": 100.0
                * int(metric_counts["centroid_x_px"])
                / int(forward["episodes"]),
                "episodes": int(metric_counts["centroid_x_px"]),
                "interpretation": (
                    "nominal-set upper bound; centroid-x appears in five of seven strict failures"
                ),
            },
            "smallest_next_experiment": (
                "Without retraining during this run, fit a held-out residual calibration for centroid "
                "x/y and replay only the seven exact nominal failures with corrected candidate scores."
            ),
        },
        {
            "rank": 4,
            "bottleneck": "CEM sampling budget offers modest but unconfirmed headroom",
            "evidence": {
                "best_population": int(best_cem["population"]),
                "best_fault_success": float(best_cem["fault"]["strict_success"]),
                "best_gain_over_population_24": float(
                    best_cem["fault_success_gain_over_population_24"]
                ),
                "population_24_fault_success": next(
                    float(row["fault"]["strict_success"])
                    for row in cem_arms
                    if int(row["population"]) == 24
                ),
            },
            "estimated_achievable_gain": {
                "kind": "observed_matched_development_gain",
                "fault_success_points": 100.0
                * float(best_cem["fault_success_gain_over_population_24"]),
                "interpretation": "observed sensitivity, not a guaranteed causal gain",
            },
            "smallest_next_experiment": (
                "Repeat populations 24 and the best budget on two additional matched planner seeds."
            ),
        },
        {
            "rank": 5,
            "bottleneck": "top-k reranking has too little existing-candidate headroom",
            "evidence": {
                "initial_top1_success": first_candidates["h1_cem_top1_strict_success"],
                "initial_best10_success": first_candidates[
                    "simulator_oracle_best_of_k_strict_success"
                ],
                "all_decision_best10_gain": all_candidates["best_of_k_success_gain"],
                "all_decision_hard_negative_recoveries": all_candidates[
                    "matched_hard_negative_recoveries"
                ],
                "branch_b_gate_passes": bool(
                    all_candidates["best_of_k_success_gain"] >= 0.05
                    or all_candidates["matched_hard_negative_recoveries"] >= 5
                ),
            },
            "estimated_achievable_gain": {
                "kind": "simulator_oracle_best_of_retained_top10",
                "success_points": 100.0 * all_candidates["best_of_k_success_gain"],
                "interpretation": "upper bound for reranking the retained candidates",
            },
            "smallest_next_experiment": (
                "Do not train a reranker; retain the two hard negatives and focus on generating "
                "better candidates or correcting centroid predictions."
            ),
        },
    ]
    exact_ids = {
        "nominal_strict_failures": [
            row["case_id"] for row in forward_failures["exact_failures"]
        ],
        "fault_unrecovered_even_with_oracle_gain": taxonomy[
            "exact_episode_ids_by_category"
        ].get("unrecovered_even_with_oracle_gain", []),
        "probe_missed_oracle_recoverable_failure": taxonomy[
            "exact_episode_ids_by_category"
        ].get("probe_missed_oracle_recoverable_failure", []),
    }
    report = {
        "version": VERSION,
        "role": "supporting_development_forensics_primary_branch_not_reselected",
        "split": "development_only",
        "protected_set_used": False,
        "ranked_bottlenecks": ranked,
        "exact_failing_setup_ids": exact_ids,
        "reproduction_commands": {
            "matched_fault_controls": taxonomy["reproduction_command"],
            "candidate_coverage": (
                f"{os.sys.executable} -m active_diagnosis_v13.analyze_candidate_coverage "
                "--episode-dir runs/v12_mpc_h1_h3_diagnosis_20260731_111829/episodes/primary "
                f"--output-dir {args.candidate_coverage.resolve().parent}"
            ),
            "forward_calibration": (
                f"{os.sys.executable} -m active_diagnosis_v13.analyze_forward_calibration "
                "--config active_diagnosis_v13/config_v13.json "
                "--episode-dir runs/v12_mpc_h1_h3_diagnosis_20260731_111829/episodes/primary "
                f"--output-dir {args.forward_calibration.resolve().parent}"
            ),
        },
        "limitations": [
            "This is supporting development-only forensic evidence and cannot reselect the primary branch.",
            "Upper bounds are labeled separately from observed matched gains.",
            "The retained top-10 audit is nominal-gain prior-run evidence, not a v13 hidden-gain counterfactual.",
        ],
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "ranked_bottlenecks.json", report)
    lines = [
        "# Ranked development-only H1/CEM bottlenecks",
        "",
        "This supporting forensic table does not reselect the primary branch.",
        "",
        "| Rank | Bottleneck | Estimated headroom | Smallest next experiment |",
        "|---:|---|---|---|",
    ]
    for row in ranked:
        estimate = row["estimated_achievable_gain"]
        value = estimate.get("fault_success_points", estimate.get("success_points"))
        estimate_text = (
            f"{float(value):.1f} pp ({estimate['kind']})"
            if value is not None
            else f"{estimate.get('episodes')} episodes ({estimate['kind']})"
        )
        lines.append(
            f"| {row['rank']} | {row['bottleneck']} | {estimate_text} | "
            f"{row['smallest_next_experiment']} |"
        )
    (output_dir / "ranked_bottlenecks.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
