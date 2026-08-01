"""Export compact human-readable tables from the full diagnostic summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


CONTROLLERS = ("oracle_h1", "oracle_h3", "learned_h1", "learned_h3")
EPISODE_METRICS = (
    "groups",
    "strict_success_rate",
    "any_improvement_rate",
    "predicted_improvement_actual_worsening_rate",
    "mean_final_normalized_distance",
    "median_final_normalized_distance",
    "median_normalized_distance_reduction",
    "mean_final_to_initial_distance_ratio",
    "median_steps",
    "median_cumulative_motion_l1_mm",
    "boundary_clipping_failure_rate",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _episode_rows(
    values: Mapping[str, Mapping[str, Any]],
    grouping_name: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for group, controllers in values.items():
        for controller in CONTROLLERS:
            summary = controllers[controller]
            row: dict[str, Any] = {}
            if grouping_name is not None:
                row[grouping_name] = group
            row["controller"] = controller
            row.update(
                {metric: summary[metric] for metric in EPISODE_METRICS}
            )
            interval = summary["strict_success_group_bootstrap_95"]
            row["strict_success_ci95_low"] = interval["low"]
            row["strict_success_ci95_high"] = interval["high"]
            for output, rate in summary[
                "per_output_final_tolerance_failure_rate"
            ].items():
                row[f"failure_rate_{output}"] = rate
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError("refusing to export tables from incomplete results")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        args.output_dir / "overall_summary.csv",
        _episode_rows({"overall": summary["overall"]}),
    )
    _write_csv(
        args.output_dir / "stratum_summary.csv",
        _episode_rows(summary["by_stratum"], "stratum"),
    )
    _write_csv(
        args.output_dir / "initial_distance_band_summary.csv",
        _episode_rows(
            summary["by_initial_distance_band"],
            "initial_distance_band",
        ),
    )

    action_rows: list[dict[str, Any]] = []
    for controller in CONTROLLERS:
        for category, values in summary[
            "ordinary_vs_action_bound_adjacent_steps"
        ][controller].items():
            action_rows.append(
                {
                    "controller": controller,
                    "step_category": category,
                    **values,
                }
            )
    _write_csv(
        args.output_dir / "ordinary_vs_action_bound_steps.csv",
        action_rows,
    )
    print(
        json.dumps(
            {
                "event": "summary_tables_exported",
                "output_dir": str(args.output_dir.resolve()),
                "tables": 4,
            }
        )
    )


if __name__ == "__main__":
    main()
