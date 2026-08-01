#!/usr/bin/env python3
"""Plot Gate A probe and control results from machine-readable artifacts."""

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
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    gate_dir = args.gate_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    probes = json.loads((gate_dir / "probes" / "probe_summary.json").read_text())
    safety_path = gate_dir / "probes" / "probe_safety_summary.json"
    safety = json.loads(safety_path.read_text()) if safety_path.exists() else None
    safety_by_key = {
        (str(row["design"]), float(row["fraction"])): row
        for row in ([] if safety is None else safety["cells"])
    }
    diagnosis = json.loads((gate_dir / "gate_a_diagnosis.json").read_text())

    designs = sorted({row["design"] for row in probes["designs"]})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for design in designs:
        rows = sorted(
            [row for row in probes["designs"] if row["design"] == design],
            key=lambda row: float(row["fraction"]),
        )
        axes[0].plot(
            [100.0 * float(row["fraction"]) for row in rows],
            [100.0 * float(row["gain_classification_accuracy"]) for row in rows],
            marker="o",
            label=design,
        )
        axes[1].plot(
            [100.0 * float(row["fraction"]) for row in rows],
            [
                float(
                    safety_by_key.get(
                        (str(row["design"]), float(row["fraction"])), row
                    ).get(
                        "mean_peak_beam_state_disturbance",
                        row["mean_absolute_beam_state_disturbance"],
                    )
                )
                for row in rows
            ],
            marker="o",
            label=design,
        )
    axes[0].axhline(80.0, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Probe fraction of command range (%)", ylabel="Gain accuracy (%)")
    axes[1].set(
        xlabel="Probe fraction of command range (%)",
        ylabel=(
            "Mean peak transient beam disturbance (tolerances)"
            if safety is not None
            else "Mean final beam disturbance (tolerances)"
        ),
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "probe_observability_and_disturbance.png", dpi=180)
    plt.close(fig)

    summaries = diagnosis["control_summaries"]
    gains = sorted(
        summaries["direct"]["by_gain"], key=float
    )
    modes = ("direct", "oracle_known", "probe_replan")
    labels = ("Direct H1", "Oracle-known gain", "Probe + replan")
    x = np.arange(len(gains), dtype=np.float64)
    width = 0.25
    fig, axis = plt.subplots(figsize=(9, 4.8))
    for index, (mode, label) in enumerate(zip(modes, labels, strict=True)):
        axis.bar(
            x + (index - 1) * width,
            [100.0 * summaries[mode]["by_gain"][gain]["strict_success"] for gain in gains],
            width=width,
            label=label,
        )
    axis.set_xticks(x, gains)
    axis.set_ylim(0, 105)
    axis.set_xlabel("Hidden physical gain")
    axis.set_ylabel("Strict success (%)")
    axis.set_title(f"Gate A development result: branch {diagnosis['selected_primary_branch']}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "control_success_by_gain.png", dpi=180)
    plt.close(fig)

    tradeoff_path = gate_dir / "probe_design_control_summary.json"
    if tradeoff_path.exists():
        tradeoff = json.loads(tradeoff_path.read_text())
        cells = tradeoff["cells"]
        if int(tradeoff["completed_cells"]) != int(tradeoff["expected_cells"]):
            raise ValueError("probe tradeoff plot requires the complete control matrix")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        accuracies = np.asarray(
            [100.0 * float(row["gain_classification_accuracy"]) for row in cells]
        )
        transient = np.asarray(
            [
                float(row["transient_probe_safety"]["mean_peak_beam_state_disturbance"])
                for row in cells
            ]
        )
        gains = np.asarray(
            [100.0 * float(row["fault_success_gain_over_direct"]) for row in cells]
        )
        steps = np.asarray([float(row["mean_total_additional_steps"]) for row in cells])
        success = np.asarray(
            [
                100.0 * float(row["fault_success_after_estimation_and_replanning"])
                for row in cells
            ]
        )
        left = axes[0].scatter(
            transient,
            gains,
            c=accuracies,
            cmap="viridis",
            vmin=min(accuracies),
            vmax=max(80.0, max(accuracies)),
            s=62,
        )
        axes[1].scatter(
            steps,
            success,
            c=accuracies,
            cmap="viridis",
            vmin=min(accuracies),
            vmax=max(80.0, max(accuracies)),
            s=62,
        )
        abbreviations = {
            "single_positive": "+",
            "single_negative": "−",
            "symmetric_pair": "±",
            "repeated_positive": "++",
        }
        for index, row in enumerate(cells):
            label = f"{abbreviations[str(row['design'])]}{100 * float(row['fraction']):g}%"
            axes[0].annotate(label, (transient[index], gains[index]), xytext=(4, 3), textcoords="offset points", fontsize=7)
            axes[1].annotate(label, (steps[index], success[index]), xytext=(4, 3), textcoords="offset points", fontsize=7)
        axes[0].axhline(5.0, color="black", linestyle="--", linewidth=1)
        axes[0].set(
            xlabel="Mean peak transient disturbance (tolerances)",
            ylabel="Fault success gain over direct (points)",
        )
        axes[1].set(
            xlabel="Mean total additional steps",
            ylabel="Fault strict success (%)",
        )
        axes[0].grid(alpha=0.25)
        axes[1].grid(alpha=0.25)
        fig.colorbar(left, ax=axes, label="Five-class gain accuracy (%)", shrink=0.9)
        fig.subplots_adjust(left=0.07, right=0.93, bottom=0.13, top=0.95, wspace=0.27)
        fig.savefig(output_dir / "probe_control_tradeoffs.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
