#!/usr/bin/env python3
"""Summarize the one protected evaluation without reopening model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.compare_control_policies import _bootstrap
from active_diagnosis_v13.run_protected_once import validate_protected_rows


VERSION = "active_diagnosis_v13_protected_confirmatory_summary_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _rate(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(row["strict_success"]) for row in rows]))


def _block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "strict_success": _rate(rows),
        "mean_final_distance": float(
            np.mean([float(row["final_normalized_distance"]) for row in rows])
        ),
        "mean_total_additional_steps": float(
            np.mean([int(row["total_additional_steps"]) for row in rows])
        ),
        "saturation_episode_rate": float(
            np.mean([int(row["saturation_count"]) > 0 for row in rows])
        ),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_gain: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gain[f"{float(row['evaluator_only_true_gain']):g}"].append(row)
        by_stratum[str(row["stratum"])].append(row)
    return {
        "overall": _block(rows),
        "by_gain": {
            key: _block(value)
            for key, value in sorted(by_gain.items(), key=lambda item: float(item[0]))
        },
        "by_stratum": {key: _block(value) for key, value in sorted(by_stratum.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected-dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    root = args.protected_dir.resolve()
    manifest = json.loads(
        (root / "protected_once_manifest.json").read_text(encoding="utf-8")
    )
    freeze_path = args.freeze.resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not freeze.get("frozen"):
        raise ValueError("protected run and freeze must both be complete")
    if manifest.get("selection_source_sha256") != _sha256(freeze_path):
        raise ValueError("protected manifest does not match the supplied freeze")
    if manifest.get("selected_probe") != freeze.get("selected_probe"):
        raise ValueError("protected manifest probe does not match the supplied freeze")
    modes = {
        mode: _jsonl(root / "control" / f"{mode}.jsonl")
        for mode in ("direct", "oracle_known", "probe_replan")
    }
    support_errors = [
        error
        for mode, rows in modes.items()
        for error in validate_protected_rows(rows, mode)
    ]
    if support_errors:
        raise ValueError("invalid protected support: " + "; ".join(support_errors))
    nominal = [
        row for row in modes["direct"] if float(row["evaluator_only_true_gain"]) == 1.0
    ]
    fault = {
        mode: [row for row in rows if float(row["evaluator_only_true_gain"]) != 1.0]
        for mode, rows in modes.items()
    }
    direct_by_key = {
        (str(row["case_id"]), float(row["evaluator_only_true_gain"])): row
        for row in fault["direct"]
    }
    probe_by_key = {
        (str(row["case_id"]), float(row["evaluator_only_true_gain"])): row
        for row in fault["probe_replan"]
    }
    oracle_by_key = {
        (str(row["case_id"]), float(row["evaluator_only_true_gain"])): row
        for row in fault["oracle_known"]
    }
    if set(direct_by_key) != set(probe_by_key) or set(direct_by_key) != set(
        oracle_by_key
    ):
        raise ValueError("protected fault arms are not episode-matched")
    direct_fault_rate = _rate(fault["direct"])
    report: dict[str, Any] = {
        "version": VERSION,
        "role": "single_confirmatory_protected_evaluation_no_reselection",
        "selection_source": str(freeze_path),
        "selection_source_sha256": _sha256(freeze_path),
        "frozen_primary_branch": freeze["selected_primary_branch"],
        "frozen_probe": freeze["selected_probe"],
        "protected_set_used_for_selection": False,
        "fault_impact": {
            "nominal_direct_success": _rate(nominal),
            "fault_direct_success": direct_fault_rate,
            "success_drop": _rate(nominal) - direct_fault_rate,
            "matched_group_bootstrap_95": _bootstrap(
                fault["direct"], nominal, 2026080121
            ),
        },
        "recoverability": {
            "oracle_fault_success": _rate(fault["oracle_known"]),
            "success_gain_over_direct": _rate(fault["oracle_known"]) - direct_fault_rate,
            "matched_group_bootstrap_95": _bootstrap(
                fault["direct"], fault["oracle_known"], 2026080122
            ),
            "matched_recoveries": int(
                sum(
                    not bool(direct_by_key[key]["strict_success"])
                    and bool(oracle_by_key[key]["strict_success"])
                    for key in direct_by_key
                )
            ),
        },
        "observability": {
            "gain_classification_accuracy": float(
                np.mean(
                    [
                        bool(row["gain_classification_correct"])
                        for row in modes["probe_replan"]
                    ]
                )
            ),
        },
        "control_value": {
            "probe_fault_success": _rate(fault["probe_replan"]),
            "success_gain_over_direct": _rate(fault["probe_replan"]) - direct_fault_rate,
            "matched_group_bootstrap_95": _bootstrap(
                fault["direct"], fault["probe_replan"], 2026080123
            ),
            "matched_recoveries": int(
                sum(
                    not bool(direct_by_key[key]["strict_success"])
                    and bool(probe_by_key[key]["strict_success"])
                    for key in direct_by_key
                )
            ),
            "matched_regressions": int(
                sum(
                    bool(direct_by_key[key]["strict_success"])
                    and not bool(probe_by_key[key]["strict_success"])
                    for key in direct_by_key
                )
            ),
        },
        "control_summaries": {mode: _summarize(rows) for mode, rows in modes.items()},
        "branch_reselected": False,
    }
    output = root / "protected_confirmatory_summary.json"
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
