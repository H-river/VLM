#!/usr/bin/env python3
"""Plot development robustness, confidence gating, and forensic summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed = json.loads(
        (run / "development" / "seed_robustness" / "three_seed_summary.json").read_text()
    )
    confidence = json.loads((run / "development" / "confidence_gate.json").read_text())
    learning = json.loads(
        (run / "development" / "learning_curve" / "gain_learning_curve.json").read_text()
    )
    calibration = json.loads(
        (
            run
            / "fallback_preparation"
            / "forward_calibration_dev"
            / "forward_uncertainty_calibration.json"
        ).read_text()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(seed["seeds"]))
    width = 0.25
    for offset, (key, label) in enumerate(
        (
            ("fault_direct_success", "Direct H1"),
            ("fault_oracle_success", "Oracle gain"),
            ("fault_probe_replan_success", "Probe + replan"),
        )
    ):
        axes[0].bar(
            x + (offset - 1) * width,
            [100 * row[key] for row in seed["seeds"]],
            width,
            label=label,
        )
    axes[0].set_xticks(x, [str(row["seed"])[-2:] for row in seed["seeds"]])
    axes[0].set(xlabel="Planner seed suffix", ylabel="Fault strict success (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    thresholds = [row for row in confidence["thresholds"] if row["confidence_threshold"] <= 1]
    axes[1].plot(
        [row["confidence_threshold"] for row in thresholds],
        [100 * row["fault_success_gain_over_direct"] for row in thresholds],
        marker="o",
        label="Control gain",
    )
    axes[1].plot(
        [row["confidence_threshold"] for row in thresholds],
        [100 * row["estimated_gain_usage_rate"] for row in thresholds],
        marker="s",
        label="Estimate usage",
    )
    axes[1].axhline(5, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Confidence threshold", ylabel="Percentage points / percent")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "robustness_and_confidence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    curve = learning["curve"]
    axes[0].errorbar(
        [row["training_records"] for row in curve],
        [100 * row["mean_accuracy"] for row in curve],
        yerr=[
            [100 * (row["mean_accuracy"] - row["empirical_95_low"]) for row in curve],
            [100 * (row["empirical_95_high"] - row["mean_accuracy"]) for row in curve],
        ],
        marker="o",
        capsize=4,
    )
    axes[0].axhline(80, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Training records", ylabel="Group-held-out gain accuracy (%)")
    axes[0].grid(alpha=0.25)

    metrics = list(calibration["per_metric_forward_calibration"])
    x = np.arange(len(metrics))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        [calibration["per_metric_forward_calibration"][key]["mean_absolute_normalized_error"] for key in metrics],
        width,
        label="Absolute error",
    )
    axes[1].bar(
        x + width / 2,
        [calibration["per_metric_forward_calibration"][key]["mean_predicted_uncertainty"] for key in metrics],
        width,
        label="Predicted uncertainty",
    )
    axes[1].set_xticks(x, [key.replace("_px", "").replace("peak_intensity", "peak") for key in metrics], rotation=25)
    axes[1].set_ylabel("Tolerance-normalized units")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "learning_and_forward_calibration.png", dpi=180)
    plt.close(fig)

    cem_path = run / "development" / "cem_population_curve.json"
    bottleneck_path = (
        run / "fallback_preparation" / "bottleneck_synthesis" / "ranked_bottlenecks.json"
    )
    if cem_path.exists() and bottleneck_path.exists():
        cem = json.loads(cem_path.read_text())
        bottlenecks = json.loads(bottleneck_path.read_text())
        arms = sorted(cem["arms"], key=lambda row: int(row["population"]))
        populations = [int(row["population"]) for row in arms]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        axes[0].plot(
            populations,
            [100 * float(row["fault"]["strict_success"]) for row in arms],
            marker="o",
            label="All fault episodes",
        )
        axes[0].plot(
            populations,
            [
                100
                * float(
                    row["by_stratum"]["reachable_boundary_or_clipping"][
                        "strict_success"
                    ]
                )
                for row in arms
            ],
            marker="s",
            label="Boundary/clipping",
        )
        axes[0].set_xscale("log", base=2)
        axes[0].set_xticks(populations, [str(value) for value in populations])
        axes[0].set(xlabel="CEM population", ylabel="Strict success (%)")
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)

        ranked = bottlenecks["ranked_bottlenecks"]
        labels = [f"{row['rank']}. {row['bottleneck']}" for row in ranked]
        values = []
        kinds = []
        for row in ranked:
            estimate = row["estimated_achievable_gain"]
            values.append(
                float(
                    estimate.get(
                        "fault_success_points", estimate.get("success_points", 0.0)
                    )
                )
            )
            kinds.append(str(estimate["kind"]))
        y = np.arange(len(ranked))
        colors = ["#d95f02" if "upper_bound" in kind else "#1b9e77" for kind in kinds]
        axes[1].barh(y, values, color=colors)
        axes[1].set_yticks(y, labels, fontsize=7)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Estimated headroom (percentage points)")
        axes[1].grid(axis="x", alpha=0.25)
        axes[1].set_title("Orange = upper bound; green = observed")
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.91, wspace=0.62)
        fig.savefig(output / "cem_budget_and_ranked_bottlenecks.png", dpi=180)
        plt.close(fig)

    refinement_path = run / "development" / "branch_a_refinement_five_seed.json"
    if refinement_path.exists():
        refinement = json.loads(refinement_path.read_text())
        rows = refinement["seeds"]
        x = np.arange(len(rows))
        width = 0.34
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
        axes[0].bar(
            x - width / 2,
            [100 * row["direct_fault_success"] for row in rows],
            width,
            label="Direct",
        )
        axes[0].bar(
            x + width / 2,
            [100 * row["refinement_fault_success"] for row in rows],
            width,
            label="No-residual probe",
        )
        labels = [str(row["seed"])[-2:] for row in rows]
        axes[0].set_xticks(x, labels)
        axes[0].set(xlabel="Planner seed suffix", ylabel="Fault strict success (%)")
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)
        gains = [100 * row["refinement_control_value"] for row in rows]
        axes[1].plot(x, gains, marker="o")
        axes[1].axhline(5, color="black", linestyle="--", linewidth=1)
        interval = refinement["control_value"]["planner_seed_bootstrap_mean_95"]
        axes[1].axhspan(
            100 * interval["low"],
            100 * interval["high"],
            color="#1f77b4",
            alpha=0.15,
            label="Mean seed-bootstrap 95%",
        )
        axes[1].set_xticks(x, labels)
        axes[1].set(xlabel="Planner seed suffix", ylabel="Control gain (percentage points)")
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "branch_a_five_seed_robustness.png", dpi=180)
        plt.close(fig)

    posterior_path = run / "development" / "probability_mean_policy_comparison.json"
    if posterior_path.exists():
        comparison = json.loads(posterior_path.read_text())
        rows = comparison["comparisons"]
        direct = 100 * float(comparison["direct_fault_success"])
        labels = ["Direct", "Full linear", "No residual", "Posterior mean"]
        values = [direct, *[100 * float(row["fault_success"]) for row in rows]]
        lower = [0.0]
        upper = [0.0]
        for row in rows:
            interval = row["matched_group_bootstrap_95"]
            value = 100 * float(row["fault_success"])
            low = direct + 100 * float(interval["low"])
            high = direct + 100 * float(interval["high"])
            lower.append(value - low)
            upper.append(high - value)
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        x = np.arange(len(labels))
        colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]
        axes[0].bar(x, values, color=colors)
        axes[0].errorbar(
            x,
            values,
            yerr=[lower, upper],
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1,
        )
        axes[0].axhline(direct + 5, color="black", linestyle="--", linewidth=1)
        axes[0].set_xticks(x, labels, rotation=15)
        axes[0].set_ylabel("Fault strict success (%)")
        axes[0].grid(axis="y", alpha=0.25)

        for row, label, color in zip(rows, labels[1:], colors[1:]):
            axes[1].scatter(
                100 * float(row["saturation_episode_rate"]),
                100 * float(row["fault_success_gain_over_direct"]),
                s=70,
                color=color,
            )
            axes[1].annotate(
                label,
                (
                    100 * float(row["saturation_episode_rate"]),
                    100 * float(row["fault_success_gain_over_direct"]),
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axes[1].axhline(5, color="black", linestyle="--", linewidth=1)
        axes[1].set(
            xlabel="Saturation episode rate (%)",
            ylabel="Fault success gain over direct (pp)",
        )
        axes[1].grid(alpha=0.25)
        fig.suptitle(
            "Development-only estimator tradeoff; bars show group-bootstrap 95% vs direct",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output / "posterior_mean_control_tradeoff.png", dpi=180)
        plt.close(fig)

    seven_seed_path = run / "development" / "branch_a_refinement_seven_seed.json"
    if seven_seed_path.exists():
        seven = json.loads(seven_seed_path.read_text())
        rows = seven["seeds"]
        x = np.arange(len(rows))
        width = 0.25
        labels = [str(row["seed"])[-2:] for row in rows]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for offset, (key, label) in enumerate(
            (
                ("direct_fault_success", "Direct"),
                ("oracle_fault_success", "Oracle gain"),
                ("refinement_fault_success", "No-residual probe"),
            )
        ):
            axes[0].bar(
                x + (offset - 1) * width,
                [100 * float(row[key]) for row in rows],
                width,
                label=label,
            )
        axes[0].set_xticks(x, labels)
        axes[0].set(xlabel="Planner seed suffix", ylabel="Fault strict success (%)")
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)
        for key, label, marker in (
            ("fault_impact", "Fault impact", "o"),
            ("oracle_recovery", "Oracle recovery", "s"),
            ("refinement_control_value", "No-residual control value", "^"),
        ):
            axes[1].plot(
                x,
                [100 * float(row[key]) for row in rows],
                marker=marker,
                label=label,
            )
        axes[1].axhline(5, color="black", linestyle="--", linewidth=1)
        axes[1].set_xticks(x, labels)
        axes[1].set(
            xlabel="Planner seed suffix",
            ylabel="Gate component (percentage points)",
        )
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "branch_a_seven_seed_robustness.png", dpi=180)
        plt.close(fig)

    protected_path = run / "protected" / "protected_confirmatory_summary.json"
    resolution_path = run / "development" / "branch_a_resolution.json"
    if protected_path.exists() and resolution_path.exists():
        protected = json.loads(protected_path.read_text())
        resolution = json.loads(resolution_path.read_text())
        if protected.get("role") != "single_confirmatory_protected_evaluation_no_reselection":
            raise ValueError("protected summary is not marked confirmatory/no-reselection")
        if protected.get("branch_reselected") is not False:
            raise ValueError("protected plot refuses a reselected branch")
        development = resolution["development_evidence"]
        component_labels = ("Fault impact", "Oracle recovery", "Control value")
        development_values = (
            100.0 * float(
                json.loads((run / "development" / "gate_a_diagnosis.json").read_text())[
                    "fault_impact"
                ]["success_drop"]
            ),
            100.0 * float(
                json.loads((run / "development" / "gate_a_diagnosis.json").read_text())[
                    "recoverability"
                ]["success_gain_over_direct_fault"]
            ),
            100.0 * float(development["fault_success_gain_over_direct"]),
        )
        protected_values = (
            100.0 * float(protected["fault_impact"]["success_drop"]),
            100.0 * float(protected["recoverability"]["success_gain_over_direct"]),
            100.0 * float(protected["control_value"]["success_gain_over_direct"]),
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        x = np.arange(len(component_labels))
        width = 0.36
        axes[0].bar(x - width / 2, development_values, width, label="Development")
        axes[0].bar(x + width / 2, protected_values, width, label="Protected once")
        axes[0].axhline(5.0, color="black", linestyle="--", linewidth=1)
        axes[0].set_xticks(x, component_labels, rotation=12)
        axes[0].set_ylabel("Percentage points")
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)

        summaries = protected["control_summaries"]
        gains = sorted(summaries["direct"]["by_gain"], key=float)
        x = np.arange(len(gains), dtype=np.float64)
        width = 0.25
        for offset, (mode, label) in enumerate(
            (
                ("direct", "Direct H1"),
                ("oracle_known", "Oracle gain"),
                ("probe_replan", "Frozen probe"),
            )
        ):
            axes[1].bar(
                x + (offset - 1) * width,
                [
                    100.0 * float(summaries[mode]["by_gain"][gain]["strict_success"])
                    for gain in gains
                ],
                width,
                label=label,
            )
        axes[1].set_xticks(x, gains)
        axes[1].set_ylim(0, 105)
        axes[1].set(
            xlabel="Hidden physical gain",
            ylabel="Protected strict success (%)",
        )
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(fontsize=8)
        fig.suptitle(
            "Frozen Branch A: development selection and one protected confirmation",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output / "development_protected_confirmation.png", dpi=180)
        plt.close(fig)

    qwen_comparison_path = run / "branch_a" / "qwen_closed_loop_validation_comparison.json"
    hybrid_comparison_path = (
        run / "development" / "guarded_probability_mean_policy_comparison.json"
    )
    if qwen_comparison_path.exists() and hybrid_comparison_path.exists():
        qwen = json.loads(qwen_comparison_path.read_text())
        hybrid = json.loads(hybrid_comparison_path.read_text())
        qwen_labels = {
            "direct": "Direct",
            "probe_symmetric_pair_f0p1_no_residual": "Frozen linear",
            "probe_symmetric_pair_f0p1_reduced_mlp": "Small MLP",
            "probe_qwen_dev_validation": "Qwen 3B",
        }
        qwen_rows = [
            row for row in qwen["policies"] if row["policy"] in qwen_labels
        ]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        x = np.arange(len(qwen_rows))
        axes[0].bar(
            x,
            [100.0 * float(row["fault_success"]) for row in qwen_rows],
            color=("#7f7f7f", "#1f77b4", "#ff7f0e", "#9467bd"),
        )
        axes[0].set_xticks(
            x,
            [qwen_labels[row["policy"]] for row in qwen_rows],
            rotation=15,
        )
        axes[0].set_ylabel("Validation-subset fault success (%)")
        axes[0].set_title("Six group-held-out development groups")
        axes[0].grid(axis="y", alpha=0.25)

        hybrid_labels = {
            "probe_symmetric_pair_f0p1_no_residual": "Frozen discrete",
            "probe_no_residual_probability_mean": "Posterior mean",
            "probe_no_residual_guarded_probability_mean": "Guarded mean",
        }
        for row in hybrid["comparisons"]:
            label = hybrid_labels.get(row["policy"])
            if label is None:
                continue
            axes[1].scatter(
                100.0 * float(row["saturation_episode_rate"]),
                100.0 * float(row["fault_success_gain_over_direct"]),
                s=78,
            )
            axes[1].annotate(
                label,
                (
                    100.0 * float(row["saturation_episode_rate"]),
                    100.0 * float(row["fault_success_gain_over_direct"]),
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axes[1].axhline(5.0, color="black", linestyle="--", linewidth=1)
        axes[1].set(
            xlabel="All-episode saturation rate (%)",
            ylabel="Development fault gain over direct (pp)",
        )
        axes[1].set_title("Continuous-control success/risk tradeoff")
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "qwen_and_guarded_hybrid_tradeoffs.png", dpi=180)
        plt.close(fig)

    margin_seed_path = run / "development" / "guarded_margin_0p125_eight_seed.json"
    unbounded_seed_path = (
        run / "development" / "guarded_probability_mean_eight_seed.json"
    )
    if margin_seed_path.exists() and unbounded_seed_path.exists():
        margin = json.loads(margin_seed_path.read_text())
        unbounded = json.loads(unbounded_seed_path.read_text())
        if margin.get("protected_set_used") is not False or unbounded.get(
            "protected_set_used"
        ) is not False:
            raise ValueError("guarded seed plot requires development-only summaries")
        margin_rows = margin["seeds"]
        unbounded_by_seed = {int(row["seed"]): row for row in unbounded["seeds"]}
        if set(unbounded_by_seed) != {int(row["seed"]) for row in margin_rows}:
            raise ValueError("guarded seed summaries do not contain the same seeds")
        x = np.arange(len(margin_rows))
        labels = [str(row["seed"])[-2:] for row in margin_rows]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        axes[0].plot(
            x,
            [100.0 * float(row["control_value_over_direct"]) for row in margin_rows],
            marker="o",
            label="Margin 0.125 vs direct",
        )
        axes[0].plot(
            x,
            [100.0 * float(row["gain_over_discrete"]) for row in margin_rows],
            marker="s",
            label="Margin 0.125 vs frozen discrete",
        )
        axes[0].axhline(
            5.0,
            color="black",
            linestyle="--",
            linewidth=1,
            label="5 pp Gate threshold (vs direct)",
        )
        axes[0].axhline(
            0.0,
            color="#666666",
            linestyle=":",
            linewidth=1,
            label="No change vs frozen discrete",
        )
        axes[0].set_xticks(x, labels)
        axes[0].set(
            xlabel="Planner seed suffix",
            ylabel="Fault success difference (pp)",
        )
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)

        axes[1].plot(
            x,
            [
                100.0 * float(row["discrete_fault_saturation_episode_rate"])
                for row in margin_rows
            ],
            marker="^",
            label="Frozen discrete",
        )
        axes[1].plot(
            x,
            [100.0 * float(row["fault_saturation_episode_rate"]) for row in margin_rows],
            marker="o",
            label="Margin 0.125",
        )
        axes[1].plot(
            x,
            [
                100.0
                * float(
                    unbounded_by_seed[int(row["seed"])][
                        "fault_saturation_episode_rate"
                    ]
                )
                for row in margin_rows
            ],
            marker="s",
            label="Unbounded guarded mean",
        )
        axes[1].set_xticks(x, labels)
        axes[1].set(
            xlabel="Planner seed suffix",
            ylabel="Fault saturation episode rate (%)",
        )
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8)
        fig.suptitle(
            "Development-only guarded continuous control across eight planner seeds",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output / "guarded_margin_eight_seed_tradeoff.png", dpi=180)
        plt.close(fig)

    synthesis_path = (
        run
        / "fallback_preparation"
        / "postfreeze_synthesis"
        / "postfreeze_research_decision.json"
    )
    diversity_path = run / "development" / "cem_elite_diversity_0p25.json"
    feasible_path = run / "development" / "cem_feasible_proposal_resample8.json"
    population_path = run / "development" / "cem_population48_three_seed.json"
    if all(
        path.exists()
        for path in (synthesis_path, diversity_path, feasible_path, population_path)
    ):
        synthesis = json.loads(synthesis_path.read_text())
        diversity = json.loads(diversity_path.read_text())
        feasible = json.loads(feasible_path.read_text())
        population = json.loads(population_path.read_text())
        if any(
            artifact.get("protected_set_used") is not False
            for artifact in (synthesis, diversity, feasible, population)
        ):
            raise ValueError("post-freeze CEM plot requires development-only artifacts")
        table = synthesis["controlled_cem_ablation_table"]
        labels = ("Risk\nweight", "Population\n24 to 48", "Elite\ndiversity", "Feasible\nproposal")
        fault = np.asarray(
            [float(row["fault_success_difference_pp"]) for row in table],
            dtype=np.float64,
        )
        boundary = np.asarray(
            [
                0.0,
                100.0
                * float(
                    np.mean(
                        [
                            row["by_stratum"]["reachable_boundary_or_clipping"][
                                "population_success"
                            ]
                            - row["by_stratum"]["reachable_boundary_or_clipping"][
                                "reference_success"
                            ]
                            for row in population["seeds"]
                        ]
                    )
                ),
                100.0
                * float(
                    diversity["by_stratum"]["reachable_boundary_or_clipping"][
                        "success_difference"
                    ]
                ),
                100.0
                * float(
                    feasible["by_stratum"]["reachable_boundary_or_clipping"][
                        "success_difference"
                    ]
                ),
            ],
            dtype=np.float64,
        )
        intervals = [row["interval_95_pp"] for row in table]
        lower = np.asarray(
            [0.0 if interval is None else fault[index] - float(interval[0]) for index, interval in enumerate(intervals)]
        )
        upper = np.asarray(
            [0.0 if interval is None else float(interval[1]) - fault[index] for index, interval in enumerate(intervals)]
        )
        x = np.arange(len(labels))
        width = 0.36
        fig, axis = plt.subplots(figsize=(9.5, 5.2))
        axis.bar(
            x - width / 2,
            fault,
            width,
            yerr=np.vstack([lower, upper]),
            capsize=4,
            label="All fault episodes",
        )
        axis.bar(x + width / 2, boundary, width, label="Boundary/clipping slice")
        axis.axhline(0.0, color="#555555", linewidth=1)
        axis.axhline(5.0, color="black", linestyle="--", linewidth=1, label="+5 pp target")
        axis.set_xticks(x, labels)
        axis.set_ylabel("Strict-success difference (percentage points)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
        axis.set_title(
            "Development-only post-freeze CEM controls: no robust boundary fix"
        )
        fig.tight_layout()
        fig.savefig(output / "postfreeze_cem_controls.png", dpi=180)
        plt.close(fig)

    step_budget_path = run / "development" / "control_step_budget_analysis.json"
    if step_budget_path.exists():
        budget = json.loads(step_budget_path.read_text())
        if budget.get("protected_set_used") is not False:
            raise ValueError("control-step plot requires development-only evidence")
        arms = ("direct", "oracle", "probe")
        labels = ("Direct", "Oracle gain", "Frozen probe")
        policies = (
            ("baseline_budget4", "Four steps"),
            ("fixed_budget6", "Fixed six steps"),
            ("group_oof_adaptive_budget", "Group-OOF adaptive"),
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        x = np.arange(len(arms))
        width = 0.25
        for offset, (policy, label) in enumerate(policies):
            summaries = [
                (
                    budget["arms"][arm][policy]["summary"]
                    if policy == "group_oof_adaptive_budget"
                    else budget["arms"][arm][policy]
                )
                for arm in arms
            ]
            axes[0].bar(
                x + (offset - 1) * width,
                [100.0 * float(row["fault"]["strict_success"]) for row in summaries],
                width,
                label=label,
            )
            axes[1].bar(
                x + (offset - 1) * width,
                [
                    100.0
                    * float(
                        row["fault_by_stratum"]["reachable_boundary_or_clipping"][
                            "strict_success"
                        ]
                    )
                    for row in summaries
                ],
                width,
                label=label,
            )
        for axis, title in zip(
            axes,
            ("All nonnominal episodes", "Boundary/clipping nonnominal episodes"),
            strict=True,
        ):
            axis.set_xticks(x, labels)
            axis.set_ylim(0, 105)
            axis.set_ylabel("Strict success (%)")
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.suptitle(
            "Development-only temporal budget: exact fixed arms and group-OOF switch",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output / "control_step_budget_tradeoff.png", dpi=180)
        plt.close(fig)

    boundary_candidate_path = run / "development" / "boundary_candidate_coverage.json"
    if boundary_candidate_path.exists():
        coverage = json.loads(boundary_candidate_path.read_text())
        if coverage.get("protected_set_used") is not False:
            raise ValueError("boundary candidate plot requires development-only evidence")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for proposal, label, marker in (
            ("uniform_feasible", "Uniform feasible", "o"),
            ("boundary_conditioned", "Boundary conditioned", "s"),
        ):
            curve = coverage["proposals"][proposal]["coverage_curve"]
            budgets = [int(row["candidate_budget"]) for row in curve]
            axes[0].plot(
                budgets,
                [100.0 * float(row["simulator_best_of_k_strict_success"]) for row in curve],
                marker=marker,
                label=label,
            )
            axes[1].plot(
                budgets,
                [float(row["median_best_terminal_distance"]) for row in curve],
                marker=marker,
                label=label,
            )
        axes[0].set(
            xlabel="Two-step candidate budget k",
            ylabel="Simulator best-of-k strict coverage (%)",
        )
        axes[1].set(
            xlabel="Two-step candidate budget k",
            ylabel="Median best terminal distance",
        )
        for axis in axes:
            axis.set_xscale("log", base=2)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        fig.suptitle(
            "Exact 20 development boundary oracle failures; evaluator-only coverage bound",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output / "boundary_candidate_coverage.png", dpi=180)
        plt.close(fig)

    horizon_path = run / "development" / "control_horizon_curve.json"
    temporal_seed_path = run / "development" / "temporal_five_seed_statistics.json"
    if not temporal_seed_path.exists():
        temporal_seed_path = run / "development" / "temporal_seed_statistics.json"
    if horizon_path.exists() and temporal_seed_path.exists():
        horizon = json.loads(horizon_path.read_text())
        temporal_seed = json.loads(temporal_seed_path.read_text())
        if horizon.get("protected_set_used") is not False or temporal_seed.get(
            "protected_set_used"
        ) is not False:
            raise ValueError("temporal robustness plot requires development-only evidence")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for arm, label, marker in (
            ("direct", "Direct", "o"),
            ("probe", "Frozen probe", "s"),
        ):
            curve = horizon["arms"][arm]["curve"]
            axes[0].plot(
                [int(row["horizon"]) for row in curve],
                [100.0 * float(row["summary"]["fault"]["strict_success"]) for row in curve],
                marker=marker,
                label=label,
            )
        axes[0].set_xticks(range(4, 9))
        axes[0].set(
            xlabel="Maximum control steps",
            ylabel="Fault strict success (%)",
            title="Matched base-seed horizon curve",
        )
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)

        seed_rows = temporal_seed["per_seed"]
        x = np.arange(len(seed_rows))
        width = 0.34
        axes[1].bar(
            x - width / 2,
            [100.0 * float(row["fixed6_fault_difference"]) for row in seed_rows],
            width,
            label="Fixed six",
        )
        axes[1].bar(
            x + width / 2,
            [100.0 * float(row["adaptive_fault_difference"]) for row in seed_rows],
            width,
            label="Held visible rule",
        )
        adaptive_interval = temporal_seed["two_way_seed_group_bootstrap"]["adaptive"]
        axes[1].axhspan(
            100.0 * float(adaptive_interval["low"]),
            100.0 * float(adaptive_interval["high"]),
            color="#ff7f0e",
            alpha=0.13,
            label="Adaptive two-way 95%",
        )
        axes[1].axhline(5.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_xticks(x, [str(row["seed"])[-2:] for row in seed_rows])
        axes[1].set(
            xlabel="Planner seed suffix",
            ylabel="Fault success difference (pp)",
            title=f"Frozen rule on {len(seed_rows)} planner seeds",
        )
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(fontsize=8)
        fig.suptitle(
            "Development-only temporal-control robustness and stopping plateau",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output / "temporal_horizon_and_seed_robustness.png", dpi=180)
        plt.close(fig)

    candidate_seed_path = (
        run / "development" / "boundary_candidate_coverage_three_seed.json"
    )
    if not candidate_seed_path.exists():
        candidate_seed_path = (
            run / "development" / "boundary_candidate_coverage_two_seed.json"
        )
    if candidate_seed_path.exists():
        candidate_seed = json.loads(candidate_seed_path.read_text())
        if candidate_seed.get("protected_set_used") is not False:
            raise ValueError("candidate seed plot requires development-only evidence")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        rows = candidate_seed["seed_robustness_curve"]
        budgets = [int(row["candidate_budget"]) for row in rows]
        proposal = "uniform_feasible"
        labels = sorted(rows[0]["proposals"][proposal]["successes_by_seed"])
        for label, marker in zip(labels, ("o", "s", "^", "D"), strict=False):
            axes[0].plot(
                budgets,
                [
                    int(row["proposals"][proposal]["successes_by_seed"][label])
                    for row in rows
                ],
                marker=marker,
                label=label.replace("seed_", "seed "),
            )
        axes[0].set(
            xlabel="Two-step candidate budget k",
            ylabel="Selected failures recovered (of 20)",
            title="Per-seed best-of-k coverage",
        )
        axes[0].legend(fontsize=8)

        union_rows = [row.get("proposal_union") for row in rows]
        if all(row is not None for row in union_rows):
            axes[1].plot(
                budgets,
                [len(row["success_ids_any_seed"]) for row in union_rows],
                marker="o",
                label="Recovered by any seed",
            )
            axes[1].plot(
                budgets,
                [len(row["stable_success_ids_all_seeds"]) for row in union_rows],
                marker="s",
                label="Recovered by every seed",
            )
            axes[1].fill_between(
                budgets,
                [int(row["minimum_successes"]) for row in union_rows],
                [int(row["maximum_successes"]) for row in union_rows],
                color="#7f7f7f",
                alpha=0.2,
                label="Per-seed count range",
            )
            axes[1].set(
                xlabel="Two-step candidate budget k",
                ylabel="Distinct selected failures (of 20)",
                title="Proposal-union stability across seeds",
            )
        else:
            conditioned = "boundary_conditioned"
            for label, marker in zip(labels, ("o", "s", "^", "D"), strict=False):
                axes[1].plot(
                    budgets,
                    [
                        int(
                            row["proposals"][conditioned]["successes_by_seed"][
                                label
                            ]
                        )
                        for row in rows
                    ],
                    marker=marker,
                    label=label.replace("seed_", "seed "),
                )
            axes[1].set(
                xlabel="Two-step candidate budget k",
                ylabel="Selected failures recovered (of 20)",
                title="Boundary-conditioned proposal",
            )
        for axis in axes:
            axis.set_xscale("log", base=2)
            axis.set_xticks(budgets, [str(value) for value in budgets])
            axis.set_ylim(0, 20.5)
            axis.grid(alpha=0.25)
        axes[1].legend(fontsize=8)
        fig.suptitle(
            "Evaluator-only boundary coverage across independent proposal seeds",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output / "boundary_candidate_seed_robustness.png", dpi=180)
        plt.close(fig)

    sequential_path = run / "development" / "sequential_horizon_rule_five_seed.json"
    if not sequential_path.exists():
        sequential_path = run / "development" / "sequential_horizon_rule_three_seed.json"
    if sequential_path.exists():
        sequential = json.loads(sequential_path.read_text())
        if sequential.get("protected_set_used") is not False:
            raise ValueError("sequential horizon plot requires development-only evidence")
        rows = sequential["seeds"]
        x = np.arange(len(rows))
        width = 0.2
        policies = (
            ("fixed4", "Fixed 4"),
            ("fixed6", "Fixed 6"),
            ("fixed8", "Fixed 8"),
            ("sequential_rule", "Visible sequential"),
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for offset, (key, label) in enumerate(policies):
            centers = x + (offset - 1.5) * width
            axes[0].bar(
                centers,
                [100.0 * float(row["fault_success"][key]) for row in rows],
                width,
                label=label,
            )
            axes[1].bar(
                centers,
                [100.0 * float(row["boundary_fault_success"][key]) for row in rows],
                width,
                label=label,
            )
        labels = [str(row["seed"]).replace("seed_", "")[-2:] for row in rows]
        for axis, title in zip(
            axes,
            ("All nonnominal episodes", "Boundary/clipping nonnominal episodes"),
            strict=True,
        ):
            axis.set_xticks(x, labels)
            axis.set_ylim(0, 105)
            axis.set(
                xlabel="Planner seed suffix",
                ylabel="Strict success (%)",
                title=title,
            )
            axis.grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.suptitle(
            "Frozen visible sequential stopping rule versus fixed horizons",
            fontsize=10,
        )
        fig.tight_layout()
        filename = (
            "sequential_horizon_rule_five_seed.png"
            if len(rows) >= 5
            else "sequential_horizon_rule_three_seed.png"
        )
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    fresh_path = run / "development" / "frozen_sequential_fresh_holdout.json"
    if fresh_path.exists():
        fresh = json.loads(fresh_path.read_text())
        if fresh.get("protected_set_used") is not False or fresh.get(
            "selection_or_retuning_on_fresh_suite"
        ) is not False:
            raise ValueError("fresh temporal plot requires one-shot non-protected evidence")
        policies = fresh["policies"]
        order = ("fixed4", "fixed6", "fixed8", "frozen_sequential")
        labels = ("Fixed 4", "Fixed 6", "Fixed 8", "Frozen\nsequential")
        x = np.arange(len(order))
        width = 0.36
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        axes[0].bar(
            x - width / 2,
            [100.0 * float(policies[key]["fault"]["strict_success"]) for key in order],
            width,
            label="All nonnominal",
        )
        axes[0].bar(
            x + width / 2,
            [
                100.0 * float(policies[key]["boundary_fault"]["strict_success"])
                for key in order
            ],
            width,
            label="Boundary/clipping",
        )
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0, 105)
        axes[0].set_ylabel("Strict success (%)")
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)

        axes[1].bar(
            x,
            [float(policies[key]["overall"]["mean_control_steps"]) for key in order],
            color="#4c78a8",
            label="Mean controls",
        )
        saturation_axis = axes[1].twinx()
        saturation_values = [
            100.0 * float(policies[key]["fault"]["saturation_episode_rate"])
            for key in order
        ]
        saturation_axis.plot(
            x,
            saturation_values,
            color="#e45756",
            marker="o",
            label="Fault saturation",
        )
        axes[1].set_xticks(x, labels)
        axes[1].set_ylabel("Mean control steps")
        saturation_axis.set_ylabel("Fault saturation episodes (%)")
        saturation_axis.set_ylim(0.0, max(10.0, 1.5 * max(saturation_values)))
        axes[1].grid(axis="y", alpha=0.25)
        handles1, labels1 = axes[1].get_legend_handles_labels()
        handles2, labels2 = saturation_axis.get_legend_handles_labels()
        axes[1].legend(handles1 + handles2, labels1 + labels2, fontsize=8)
        fig.suptitle(
            "Frozen rule on 30 fresh group- and setup-disjoint simulator setups",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output / "frozen_sequential_fresh_holdout.png", dpi=180)
        plt.close(fig)

    fresh_replication_path = (
        run / "development" / "frozen_sequential_fresh_holdout_replication.json"
    )
    if fresh_replication_path.exists():
        replication = json.loads(fresh_replication_path.read_text())
        if replication.get("protected_set_used") is not False or replication.get(
            "selection_or_retuning_on_fresh_suites"
        ) is not False:
            raise ValueError("fresh replication plot requires frozen one-shot evidence")
        rows = replication["suites"]
        x = np.arange(len(rows))
        width = 0.36
        labels = [str(row["suite"]).replace("suite_", "Suite ").upper() for row in rows]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        axes[0].bar(
            x - width / 2,
            [100.0 * float(row["fixed4_fault_success"]) for row in rows],
            width,
            label="Fixed 4",
        )
        axes[0].bar(
            x + width / 2,
            [100.0 * float(row["fault_success"]) for row in rows],
            width,
            label="Frozen sequential",
        )
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0, 105)
        axes[0].set_ylabel("Fault strict success (%)")
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)

        gains = [100.0 * float(row["fault_gain_over_fixed4"]) for row in rows]
        axes[1].bar(x, gains, color="#f58518")
        interval = replication["pooled_group_bootstrap_gain_over_fixed4"]
        axes[1].axhspan(
            100.0 * float(interval["low"]),
            100.0 * float(interval["high"]),
            color="#f58518",
            alpha=0.15,
            label="Pooled 60-group 95%",
        )
        axes[1].axhline(5.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_xticks(x, labels)
        axes[1].set_ylabel("Gain over fixed 4 (pp)")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(fontsize=8)
        fig.suptitle(
            "Frozen sequential rule across two mutually disjoint fresh suites",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output / "frozen_sequential_fresh_replication.png", dpi=180)
        plt.close(fig)

    cohort_shift_path = (
        run / "development" / "frozen_sequential_setup_cohort_shift.json"
    )
    if cohort_shift_path.exists():
        shift = json.loads(cohort_shift_path.read_text())
        if shift.get("protected_set_used") is not False or shift.get(
            "selection_or_retuning_on_cohorts"
        ) is not False:
            raise ValueError("cohort shift plot requires descriptive non-protected evidence")
        preferred = ["original", "fresh_a", "fresh_b", "fresh_matched"]
        labels = [label for label in preferred if label in shift["cohorts"]]
        rows = [shift["cohorts"][label] for label in labels]
        x = np.arange(len(labels), dtype=np.float64)
        width = 0.36
        display = [
            f"{label.replace('_', ' ').title()}\nmedian d={row['initial_distance']['median']:.1f}"
            for label, row in zip(labels, rows, strict=True)
        ]
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
        axes[0].bar(
            x - width / 2,
            [100.0 * float(row["fault"]["fixed4_success"]) for row in rows],
            width,
            label="Fixed 4",
        )
        axes[0].bar(
            x + width / 2,
            [
                100.0 * float(row["fault"]["frozen_sequential_success"])
                for row in rows
            ],
            width,
            label="Frozen sequential",
        )
        axes[0].set_xticks(x, display)
        axes[0].set_ylim(0, 105)
        axes[0].set_ylabel("Fault strict success (%)")
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)
        gains = [
            100.0 * float(row["fault"]["frozen_sequential_gain_over_fixed4"])
            for row in rows
        ]
        axes[1].bar(x, gains, color="#ff7f0e")
        axes[1].axhline(5.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_xticks(x, display)
        axes[1].set_ylabel("Frozen sequential gain over fixed 4 (pp)")
        axes[1].grid(axis="y", alpha=0.25)
        fig.suptitle(
            "Setup-cohort difficulty shift and outcome-free matching",
            fontsize=10.5,
        )
        fig.tight_layout()
        fig.savefig(output / "frozen_sequential_setup_cohort_shift.png", dpi=180)
        plt.close(fig)

    matched_suite_path = (
        run
        / "development"
        / "fresh_setup_holdout_matched"
        / "evaluation_suite_manifest.json"
    )
    if matched_suite_path.exists():
        matched = json.loads(matched_suite_path.read_text())
        matching = matched["outcome_free_difficulty_matching"]
        if matching.get("protected_set_used") is not False or matching.get(
            "selection_or_retuning_on_candidate_outcomes"
        ) is not False:
            raise ValueError("matched setup plot requires outcome-free non-protected selection")
        rows = matching["matches"]
        strata = sorted({row["stratum"] for row in rows})
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
        for stratum in strata:
            selected = [row for row in rows if row["stratum"] == stratum]
            axes[0].scatter(
                [row["reference_initial_normalized_distance"] for row in selected],
                [row["selected_initial_normalized_distance"] for row in selected],
                label=stratum.replace("_", " "),
                s=45,
            )
        all_distances = [
            float(row[key])
            for row in rows
            for key in (
                "reference_initial_normalized_distance",
                "selected_initial_normalized_distance",
            )
        ]
        low = min(all_distances) * 0.9
        high = max(all_distances) * 1.1
        axes[0].plot([low, high], [low, high], "k--", linewidth=1)
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set(
            xlabel="Original development initial distance",
            ylabel="Matched fresh initial distance",
            xlim=(low, high),
            ylim=(low, high),
        )
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=7)
        mismatches = sorted(
            float(row["absolute_log1p_distance_mismatch"]) for row in rows
        )
        axes[1].bar(np.arange(len(mismatches)), mismatches, color="#9467bd")
        axes[1].set(
            xlabel="Selected setup pair, sorted",
            ylabel="Absolute log1p distance mismatch",
        )
        axes[1].grid(axis="y", alpha=0.25)
        fig.suptitle(
            "Outcome-free within-stratum difficulty matching for the fresh cohort",
            fontsize=10.5,
        )
        fig.tight_layout()
        fig.savefig(output / "fresh_setup_difficulty_matching.png", dpi=180)
        plt.close(fig)

    fresh_seed_setup_path = (
        run
        / "development"
        / "frozen_sequential_fresh_holdout_seed_setup_robustness.json"
    )
    if fresh_seed_setup_path.exists():
        robustness = json.loads(fresh_seed_setup_path.read_text())
        if robustness.get("protected_set_used") is not False or robustness.get(
            "selection_or_retuning_on_fresh_suites"
        ) is not False:
            raise ValueError("fresh seed/setup plot requires frozen non-protected evidence")
        suites = robustness["suites"]
        seeds = robustness["planner_root_seeds"]
        cell_rows = {
            (row["suite"], int(row["planner_root_seed"])): row
            for row in robustness["per_suite_seed"]
        }
        x = np.arange(len(suites), dtype=np.float64)
        width = 0.34
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
        for index, seed in enumerate(seeds):
            offset = (index - (len(seeds) - 1) / 2.0) * width
            axes[0].bar(
                x + offset,
                [
                    100.0 * float(cell_rows[(suite, int(seed))]["fault_gain_over_fixed4"])
                    for suite in suites
                ],
                width,
                label=f"Planner seed …{str(seed)[-4:]}",
            )
        axes[0].axhline(0.0, color="black", linewidth=0.9)
        axes[0].set_xticks(x, [str(suite).replace("suite_", "Suite ").upper() for suite in suites])
        axes[0].set(
            ylabel="Frozen sequential gain over fixed 4 (pp)",
            xlabel="Independent fresh setup suite",
        )
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)

        seed_rows = {int(row["planner_root_seed"]): row for row in robustness["per_seed"]}
        pooled = [
            100.0 * float(seed_rows[int(seed)]["fault_gain_over_fixed4"])
            for seed in seeds
        ]
        axes[1].bar(np.arange(len(seeds)), pooled, color="#2ca02c", alpha=0.85)
        interval = robustness["two_way_seed_group_bootstrap_gain_over_fixed4"]
        estimate = 100.0 * float(interval["estimate"])
        axes[1].axhline(estimate, color="black", linewidth=1.2, label="Two-way estimate")
        axes[1].axhspan(
            100.0 * float(interval["low"]),
            100.0 * float(interval["high"]),
            color="#2ca02c",
            alpha=0.16,
            label="Seed × setup bootstrap 95%",
        )
        axes[1].axhline(0.0, color="black", linewidth=0.9)
        axes[1].set_xticks(
            np.arange(len(seeds)), [f"…{str(seed)[-4:]}" for seed in seeds]
        )
        axes[1].set(
            xlabel="Planner seed suffix",
            ylabel=(
                "Gain pooled across "
                f"{robustness['independent_setup_groups']} setups (pp)"
            ),
        )
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(fontsize=8)
        fig.suptitle(
            "Frozen sequential rule: balanced "
            f"{len(seeds)}-seed × {robustness['independent_setup_groups']}-fresh-setup robustness",
            fontsize=10.5,
        )
        fig.tight_layout()
        fig.savefig(
            output / "frozen_sequential_fresh_seed_setup_robustness.png", dpi=180
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
