#!/usr/bin/env python3
"""Taxonomize the frozen policy's one protected confirmation without reselection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.run_protected_once import validate_protected_rows


VERSION = "active_diagnosis_v13_protected_failure_taxonomy_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _category(
    direct: dict[str, Any], oracle: dict[str, Any], probe: dict[str, Any]
) -> str:
    direct_success = bool(direct["strict_success"])
    oracle_success = bool(oracle["strict_success"])
    probe_success = bool(probe["strict_success"])
    if direct_success and probe_success:
        return "direct_success_preserved"
    if direct_success and not probe_success:
        return "probe_induced_regression"
    if not direct_success and oracle_success and probe_success:
        return "probe_recovered_oracle_recoverable_failure"
    if not direct_success and oracle_success and not probe_success:
        return "probe_missed_oracle_recoverable_failure"
    if not direct_success and not oracle_success and probe_success:
        return "probe_recovered_beyond_oracle_policy"
    return "unrecovered_even_with_oracle_gain"


def _block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
        "direct_success": float(np.mean([row["direct_success"] for row in rows])),
        "oracle_success": float(np.mean([row["oracle_success"] for row in rows])),
        "probe_replan_success": float(
            np.mean([row["probe_replan_success"] for row in rows])
        ),
        "gain_classification_accuracy": float(
            np.mean([row["gain_classification_correct"] for row in rows])
        ),
        "probe_saturation_episode_rate": float(
            np.mean([row["probe_saturation_count"] > 0 for row in rows])
        ),
        "mean_probe_total_additional_steps": float(
            np.mean([row["probe_total_additional_steps"] for row in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected-dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.protected_dir.resolve()
    freeze_path = args.freeze.resolve()
    freeze = json.loads(freeze_path.read_text())
    manifest = json.loads((root / "protected_once_manifest.json").read_text())
    summary = json.loads((root / "protected_confirmatory_summary.json").read_text())
    if manifest.get("status") != "complete" or not freeze.get("frozen"):
        raise ValueError("protected run and freeze must both be complete")
    if manifest.get("selection_source_sha256") != _sha256(freeze_path):
        raise ValueError("protected manifest does not match the supplied freeze")
    if manifest.get("selected_probe") != freeze.get("selected_probe"):
        raise ValueError("protected manifest probe does not match the freeze")
    if summary.get("role") != "single_confirmatory_protected_evaluation_no_reselection":
        raise ValueError("protected summary is not confirmatory/no-reselection")
    if summary.get("branch_reselected") is not False:
        raise ValueError("protected branch was reselected")
    by_mode = {
        mode: {_key(row): row for row in _jsonl(root / "control" / f"{mode}.jsonl")}
        for mode in ("direct", "oracle_known", "probe_replan")
    }
    support_errors = [
        error
        for mode, rows in by_mode.items()
        for error in validate_protected_rows(list(rows.values()), mode)
    ]
    if support_errors:
        raise ValueError("invalid protected support: " + "; ".join(support_errors))
    key_sets = [set(rows) for rows in by_mode.values()]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("protected control arms are not exactly episode-matched")

    episodes = []
    for key in sorted(key_sets[0]):
        direct, oracle, probe = (by_mode[mode][key] for mode in by_mode)
        episodes.append(
            {
                "episode_id": f"{key[0]}__g{key[1]:g}",
                "case_id": key[0],
                "true_gain_evaluator_only": key[1],
                "stratum": str(direct["stratum"]),
                "category": _category(direct, oracle, probe),
                "direct_success": bool(direct["strict_success"]),
                "oracle_success": bool(oracle["strict_success"]),
                "probe_replan_success": bool(probe["strict_success"]),
                "gain_classification_correct": bool(
                    probe["gain_classification_correct"]
                ),
                "estimated_gain": float(probe["gain_belief"]),
                "direct_final_distance": float(direct["final_normalized_distance"]),
                "oracle_final_distance": float(oracle["final_normalized_distance"]),
                "probe_final_distance": float(probe["final_normalized_distance"]),
                "probe_total_additional_steps": int(probe["total_additional_steps"]),
                "direct_saturation_count": int(direct["saturation_count"]),
                "oracle_saturation_count": int(oracle["saturation_count"]),
                "probe_saturation_count": int(probe["saturation_count"]),
            }
        )
    non_nominal = [row for row in episodes if row["true_gain_evaluator_only"] != 1.0]
    by_gain: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in non_nominal:
        by_gain[f"{row['true_gain_evaluator_only']:g}"].append(row)
        by_stratum[row["stratum"]].append(row)
    labels = [0.5, 0.75, 1.0, 1.25, 1.5]
    label_index = {value: index for index, value in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for row in episodes:
        confusion[
            label_index[row["true_gain_evaluator_only"]],
            label_index[row["estimated_gain"]],
        ] += 1
    correct = [row for row in non_nominal if row["gain_classification_correct"]]
    incorrect = [row for row in non_nominal if not row["gain_classification_correct"]]
    category_cases: defaultdict[str, list[str]] = defaultdict(list)
    for row in non_nominal:
        category_cases[row["category"]].append(row["episode_id"])
    report = {
        "version": VERSION,
        "role": "posthoc_confirmatory_failure_interpretation_no_reselection",
        "selection_source": str(freeze_path),
        "selection_source_sha256": _sha256(freeze_path),
        "frozen_primary_branch": freeze["selected_primary_branch"],
        "frozen_probe": freeze["selected_probe"],
        "protected_set_used_for_selection": False,
        "branch_reselected": False,
        "matched_episodes": len(episodes),
        "non_nominal": _block(non_nominal),
        "by_gain": {
            key: _block(rows)
            for key, rows in sorted(by_gain.items(), key=lambda item: float(item[0]))
        },
        "by_stratum": {key: _block(rows) for key, rows in sorted(by_stratum.items())},
        "estimation_outcome_slice": {
            "correct": _block(correct),
            "incorrect": _block(incorrect),
        },
        "confusion_matrix_labels": labels,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "exact_episode_ids_by_category": {
            key: sorted(values) for key, values in sorted(category_cases.items())
        },
        "episodes": episodes,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "episodes"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
