#!/usr/bin/env python3
"""Merge per-probe observability and closed-loop development metrics.

The input control files are produced with output names of the form
``probe_<design>_f<fraction-with-p-for-decimal>``.  This script never reads the
protected split and refuses incomplete or mismatched development cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "active_diagnosis_v13_probe_control_summary_v2"


def fraction_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def output_name(design: str, fraction: float) -> str:
    return f"probe_{design}_f{fraction_label(fraction)}"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _rate(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(row["strict_success"]) for row in rows]))


def _group_bootstrap_difference(
    reference: list[dict[str, Any]], policy: list[dict[str, Any]], seed: int
) -> dict[str, float]:
    def grouped(rows: list[dict[str, Any]]) -> dict[str, float]:
        values: defaultdict[str, list[float]] = defaultdict(list)
        for row in rows:
            values[str(row["group_id"])].append(float(row["strict_success"]))
        return {group: float(np.mean(items)) for group, items in values.items()}

    left, right = grouped(reference), grouped(policy)
    groups = np.asarray(sorted(set(left) & set(right)))
    if len(groups) == 0:
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan")}
    differences = np.asarray([right[group] - left[group] for group in groups])
    rng = np.random.default_rng(seed)
    estimates = [
        float(np.mean(differences[rng.integers(0, len(groups), len(groups))]))
        for _ in range(4000)
    ]
    low, high = np.quantile(estimates, (0.025, 0.975))
    return {"estimate": float(differences.mean()), "low": float(low), "high": float(high)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    probe_summary = json.loads(
        (root / "probes" / "probe_summary.json").read_text(encoding="utf-8")
    )
    safety_path = root / "probes" / "probe_safety_summary.json"
    safety = (
        json.loads(safety_path.read_text(encoding="utf-8")) if safety_path.exists() else None
    )
    safety_by_key = {
        (str(cell["design"]), float(cell["fraction"])): cell
        for cell in ([] if safety is None else safety["cells"])
    }
    direct = _jsonl(root / "control" / "direct.jsonl")
    nominal = 1.0
    direct_fault = [
        row for row in direct if float(row["evaluator_only_true_gain"]) != nominal
    ]
    if len(direct_fault) != 120:
        raise ValueError(f"expected 120 direct fault episodes, found {len(direct_fault)}")
    direct_keys = {
        (str(row["case_id"]), float(row["evaluator_only_true_gain"]))
        for row in direct_fault
    }
    cells = []
    for cell_index, observable in enumerate(probe_summary["designs"]):
        design = str(observable["design"])
        fraction = float(observable["fraction"])
        name = output_name(design, fraction)
        path = root / "control" / f"{name}.jsonl"
        selected_key = (
            str(probe_summary["selected"]["selected_design"]),
            float(probe_summary["selected"]["selected_fraction"]),
        )
        uses_selected_alias = False
        if not path.exists() and (design, fraction) == selected_key:
            path = root / "control" / "probe_replan.jsonl"
            uses_selected_alias = True
        if not path.exists():
            if args.allow_incomplete:
                continue
            raise FileNotFoundError(path)
        rows = _jsonl(path)
        if len(rows) != 150:
            if args.allow_incomplete:
                continue
            raise ValueError(f"{path} has {len(rows)} rows, expected 150")
        expected_policy_name = "probe_replan" if uses_selected_alias else name
        if any(str(row.get("policy_name")) != expected_policy_name for row in rows):
            raise ValueError(f"policy name mismatch in {path}")
        if any(
            str(row.get("probe_design")) != design
            or float(row.get("probe_fraction")) != fraction
            for row in rows
        ):
            raise ValueError(f"probe identity mismatch in {path}")
        suffixes = {int(str(row["case_id"]).rsplit("_", 1)[1]) for row in rows}
        if any(suffix >= 10 for suffix in suffixes):
            raise ValueError(f"protected case present in {path}")
        fault = [
            row for row in rows if float(row["evaluator_only_true_gain"]) != nominal
        ]
        keys = {
            (str(row["case_id"]), float(row["evaluator_only_true_gain"]))
            for row in fault
        }
        if len(rows) == 150 and keys != direct_keys:
            raise ValueError(f"fault episode mismatch in {path}")
        direct_by_key = {
            (str(row["case_id"]), float(row["evaluator_only_true_gain"])): row
            for row in direct_fault
        }
        matched = [
            row
            for row in fault
            if (str(row["case_id"]), float(row["evaluator_only_true_gain"]))
            in direct_by_key
        ]
        recoveries = sum(
            not bool(
                direct_by_key[
                    (str(row["case_id"]), float(row["evaluator_only_true_gain"]))
                ]["strict_success"]
            )
            and bool(row["strict_success"])
            for row in matched
        )
        regressions = sum(
            bool(
                direct_by_key[
                    (str(row["case_id"]), float(row["evaluator_only_true_gain"]))
                ]["strict_success"]
            )
            and not bool(row["strict_success"])
            for row in matched
        )
        fault_rate = _rate(fault) if fault else float("nan")
        cells.append(
            {
                **{
                    key: observable[key]
                    for key in (
                        "design",
                        "fraction",
                        "gain_classification_accuracy",
                        "per_gain_accuracy",
                        "residual_signal_to_noise_ratio",
                        "mean_absolute_beam_state_disturbance",
                        "mean_signed_target_cost_disturbance",
                        "saturation_rate",
                        "constraint_violations",
                        "additional_probe_steps",
                    )
                },
                "control_file": str(path),
                "uses_selected_probe_replan_alias": uses_selected_alias,
                "episodes": len(rows),
                "fault_episodes": len(fault),
                "final_success_after_estimation_and_replanning": (
                    _rate(rows) if rows else float("nan")
                ),
                "fault_success_after_estimation_and_replanning": fault_rate,
                "fault_success_gain_over_direct": fault_rate - _rate(direct_fault),
                "matched_group_bootstrap_95": _group_bootstrap_difference(
                    direct_fault, fault, 2026080110 + cell_index
                ),
                "matched_direct_failure_recoveries": int(recoveries),
                "matched_direct_success_regressions": int(regressions),
                "mean_total_additional_steps": (
                    float(np.mean([int(row["total_additional_steps"]) for row in rows]))
                    if rows
                    else float("nan")
                ),
                "gain_classification_accuracy_in_control_replay": (
                    float(np.mean([bool(row["gain_classification_correct"]) for row in rows]))
                    if rows
                    else float("nan")
                ),
                "saturation_episode_rate_in_control": (
                    float(np.mean([int(row["saturation_count"]) > 0 for row in rows]))
                    if rows
                    else float("nan")
                ),
                "constraint_violations_in_control": int(
                    sum(int(row["constraint_violation_count"]) for row in rows)
                ),
                "transient_probe_safety": safety_by_key.get((design, fraction)),
            }
        )
    selected = probe_summary["selected"]
    selected_key = (
        str(selected["selected_design"]),
        float(selected["selected_fraction"]),
    )
    selected_cells = [
        cell
        for cell in cells
        if (str(cell["design"]), float(cell["fraction"])) == selected_key
    ]
    csv_path = root / "probe_design_control_summary.csv"
    csv_temporary = csv_path.with_suffix(f".csv.tmp.{os.getpid()}")
    csv_fields = (
        "design",
        "fraction",
        "gain_classification_accuracy",
        "residual_signal_to_noise_ratio",
        "mean_peak_transient_disturbance",
        "mean_final_beam_state_disturbance",
        "probe_saturation_rate",
        "probe_constraint_violations",
        "fault_success_after_estimation_and_replanning",
        "fault_success_gain_over_direct",
        "fault_gain_bootstrap_95_low",
        "fault_gain_bootstrap_95_high",
        "control_saturation_episode_rate",
        "control_constraint_violations",
        "additional_probe_steps",
        "mean_total_additional_steps",
    )
    with csv_temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for cell in cells:
            transient_safety = cell["transient_probe_safety"]
            interval = cell["matched_group_bootstrap_95"]
            writer.writerow(
                {
                    "design": cell["design"],
                    "fraction": cell["fraction"],
                    "gain_classification_accuracy": cell[
                        "gain_classification_accuracy"
                    ],
                    "residual_signal_to_noise_ratio": cell[
                        "residual_signal_to_noise_ratio"
                    ],
                    "mean_peak_transient_disturbance": transient_safety[
                        "mean_peak_beam_state_disturbance"
                    ],
                    "mean_final_beam_state_disturbance": transient_safety[
                        "mean_final_beam_state_disturbance"
                    ],
                    "probe_saturation_rate": cell["saturation_rate"],
                    "probe_constraint_violations": cell["constraint_violations"],
                    "fault_success_after_estimation_and_replanning": cell[
                        "fault_success_after_estimation_and_replanning"
                    ],
                    "fault_success_gain_over_direct": cell[
                        "fault_success_gain_over_direct"
                    ],
                    "fault_gain_bootstrap_95_low": interval["low"],
                    "fault_gain_bootstrap_95_high": interval["high"],
                    "control_saturation_episode_rate": cell[
                        "saturation_episode_rate_in_control"
                    ],
                    "control_constraint_violations": cell[
                        "constraint_violations_in_control"
                    ],
                    "additional_probe_steps": cell["additional_probe_steps"],
                    "mean_total_additional_steps": cell[
                        "mean_total_additional_steps"
                    ],
                }
            )
    os.replace(csv_temporary, csv_path)
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "selection_was_frozen_before_control_sensitivity": True,
        "direct_fault_success": _rate(direct_fault),
        "completed_cells": len(cells),
        "expected_cells": len(probe_summary["designs"]),
        "selected_probe": selected,
        "probe_transient_safety_present": safety is not None,
        "selected_cell_control_metrics": selected_cells[0] if selected_cells else None,
        "compact_csv": str(csv_path),
        "cells": cells,
    }
    output = root / "probe_design_control_summary.json"
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
