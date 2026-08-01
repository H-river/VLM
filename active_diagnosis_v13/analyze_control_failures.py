#!/usr/bin/env python3
"""Create a matched taxonomy of direct, oracle, and probe-control outcomes."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "active_diagnosis_v13_matched_control_failure_taxonomy_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _category(
    direct: dict[str, Any], oracle: dict[str, Any], probe: dict[str, Any]
) -> str:
    d = bool(direct["strict_success"])
    o = bool(oracle["strict_success"])
    p = bool(probe["strict_success"])
    if d and p:
        return "direct_success_preserved"
    if d and not p:
        return "probe_induced_regression"
    if not d and o and p:
        return "probe_recovered_oracle_recoverable_failure"
    if not d and o and not p:
        return "probe_missed_oracle_recoverable_failure"
    if not d and not o and p:
        return "probe_recovered_beyond_oracle_policy"
    return "unrecovered_even_with_oracle_gain"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    by_mode = {
        mode: {_key(row): row for row in _jsonl(root / "control" / f"{mode}.jsonl")}
        for mode in ("direct", "oracle_known", "probe_replan")
    }
    if any(len(rows) != 150 for rows in by_mode.values()):
        raise ValueError(
            "all three matched control arms must contain exactly 150 episodes"
        )
    keys = sorted(set.intersection(*(set(rows) for rows in by_mode.values())))
    if len(keys) != 150:
        raise ValueError(f"matched key count is {len(keys)}, expected 150")
    episodes = []
    for key in keys:
        d, o, p = (by_mode[mode][key] for mode in by_mode)
        category = _category(d, o, p)
        episodes.append(
            {
                "case_id": key[0],
                "true_gain_evaluator_only": key[1],
                "stratum": d["stratum"],
                "category": category,
                "direct_success": bool(d["strict_success"]),
                "oracle_success": bool(o["strict_success"]),
                "probe_replan_success": bool(p["strict_success"]),
                "gain_classification_correct": bool(p["gain_classification_correct"]),
                "estimated_gain": float(p["gain_belief"]),
                "direct_final_distance": float(d["final_normalized_distance"]),
                "oracle_final_distance": float(o["final_normalized_distance"]),
                "probe_final_distance": float(p["final_normalized_distance"]),
                "probe_total_additional_steps": int(p["total_additional_steps"]),
                "direct_saturation_count": int(d["saturation_count"]),
                "oracle_saturation_count": int(o["saturation_count"]),
                "probe_saturation_count": int(p["saturation_count"]),
            }
        )
    non_nominal = [row for row in episodes if row["true_gain_evaluator_only"] != 1.0]
    by_gain: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in non_nominal:
        by_gain[f"{row['true_gain_evaluator_only']:g}"].append(row)
        by_stratum[str(row["stratum"])].append(row)

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
            "mean_probe_total_additional_steps": float(
                np.mean([row["probe_total_additional_steps"] for row in rows])
            ),
        }

    category_cases: defaultdict[str, list[str]] = defaultdict(list)
    for row in non_nominal:
        category_cases[row["category"]].append(
            f"{row['case_id']}__g{row['true_gain_evaluator_only']:g}"
        )
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "matched_episodes": len(episodes),
        "non_nominal": block(non_nominal),
        "by_gain": {
            key: block(rows)
            for key, rows in sorted(by_gain.items(), key=lambda item: float(item[0]))
        },
        "by_stratum": {key: block(rows) for key, rows in sorted(by_stratum.items())},
        "exact_episode_ids_by_category": {
            key: sorted(values) for key, values in sorted(category_cases.items())
        },
        "reproduction_command": (
            f"{os.sys.executable} -m active_diagnosis_v13.run_gate_a control "
            f"--config {Path('active_diagnosis_v13/config_v13.json').resolve()} "
            f"--output-dir {root} --mode MODE"
        ),
        "episodes": episodes,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "matched_failure_taxonomy.json"
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    hard_negative = output_dir / "hard_negative_examples.jsonl"
    with hard_negative.open("w") as stream:
        for row in episodes:
            if row["category"] not in {"direct_success_preserved"}:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
