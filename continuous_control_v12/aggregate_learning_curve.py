#!/usr/bin/env python3
"""Aggregate nested-group v12 forward runs into CSV and plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from continuous_control_v12.contracts import OUTPUT_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.models_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite aggregate: {output_dir}")
    output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for report_path in sorted(root.rglob("continuous_forward_v12_*g.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        groups = int(report["train_groups"])
        row: dict[str, Any] = {
            "train_groups": groups,
            "fit_normalized_mae": report["fit"]["normalized_mae"],
            "development_normalized_mae": report["development"][
                "normalized_mae"
            ],
            "fit_strict_all_five_accuracy": report["fit"][
                "strict_all_five_accuracy"
            ],
            "development_strict_all_five_accuracy": report["development"][
                "strict_all_five_accuracy"
            ],
            "elapsed_seconds": report["elapsed_seconds"],
            "report": str(report_path),
        }
        for split in ("fit", "development"):
            for field in OUTPUT_FIELDS:
                row[f"{split}_mae_{field}"] = report[split][
                    "per_output_normalized_mae"
                ][field]
                row[f"{split}_accuracy_{field}"] = report[split][
                    "per_output_tolerance_accuracy"
                ][field]
        diagnostics_path = (
            report_path.parent
            / "diagnostics_test"
            / "forward_diagnostics.json"
        )
        if diagnostics_path.exists():
            diagnostics = json.loads(
                diagnostics_path.read_text(encoding="utf-8")
            )
            row["finite_predictions"] = diagnostics["finite_predictions"]
            row["test_normalized_mae"] = diagnostics["one_step"][
                "normalized_mae"
            ]
            row["test_strict_all_five_accuracy"] = diagnostics["one_step"][
                "strict_all_five_accuracy"
            ]
            paired = diagnostics["paired_action_and_jacobian"]
            row["paired_direction_sign_accuracy"] = paired[
                "direction_sign_accuracy"
            ]
            row["median_relative_directional_jacobian_error"] = paired[
                "median_relative_directional_jacobian_error"
            ]
            for horizon in ("1", "3", "5"):
                row[f"rollout_h{horizon}_normalized_mae"] = diagnostics[
                    "rollout_error_by_horizon"
                ][horizon]["normalized_mae"]
                row[f"rollout_h{horizon}_strict_all_five_accuracy"] = (
                    diagnostics["rollout_error_by_horizon"][horizon][
                        "strict_all_five_accuracy"
                    ]
                )
            for field in OUTPUT_FIELDS:
                row[f"test_mae_{field}"] = diagnostics["one_step"][
                    "per_output_normalized_mae"
                ][field]
                row[f"test_accuracy_{field}"] = diagnostics["one_step"][
                    "per_output_tolerance_accuracy"
                ][field]
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no nested-group reports under {root}")
    rows.sort(key=lambda row: row["train_groups"])
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output_dir / "forward_learning_curve.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "forward_learning_curve.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [row["train_groups"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key, label in (
        ("fit_normalized_mae", "fit"),
        ("development_normalized_mae", "development"),
        ("test_normalized_mae", "test"),
    ):
        if all(key in row for row in rows):
            axes[0].plot(
                groups, [row[key] for row in rows], marker="o", label=label
            )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(groups, labels=[str(value) for value in groups])
    axes[0].set_xlabel("independent train groups")
    axes[0].set_ylabel("normalized MAE")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    for key, label in (
        ("fit_strict_all_five_accuracy", "fit"),
        ("development_strict_all_five_accuracy", "development"),
        ("test_strict_all_five_accuracy", "test"),
    ):
        if all(key in row for row in rows):
            axes[1].plot(
                groups, [row[key] for row in rows], marker="o", label=label
            )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(groups, labels=[str(value) for value in groups])
    axes[1].set_xlabel("independent train groups")
    axes[1].set_ylabel("strict all-five accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "forward_learning_curve.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6))
    for field in OUTPUT_FIELDS:
        key = f"test_mae_{field}"
        if all(key in row for row in rows):
            axis.plot(
                groups,
                [row[key] for row in rows],
                marker="o",
                label=field,
            )
    axis.set_xscale("log", base=2)
    axis.set_xticks(groups, labels=[str(value) for value in groups])
    axis.set_xlabel("independent train groups")
    axis.set_ylabel("test normalized MAE")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "per_output_learning_curve.png", dpi=160)
    plt.close(figure)
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
