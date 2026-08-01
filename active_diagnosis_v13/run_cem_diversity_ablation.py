#!/usr/bin/env python3
"""Run a controlled development-only CEM elite-diversity on/off ablation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.compare_control_policies import _bootstrap
from continuous_control_v12.contracts import ACTION_FIELDS


VERSION = "active_diagnosis_v13_cem_elite_diversity_ablation_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _run_arm(
    root: Path,
    config: Path,
    minimum_distance: float,
    output_name: str,
    log_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "active_diagnosis_v13.run_gate_a",
        "control",
        "--config",
        str(config),
        "--output-dir",
        str(root),
        "--mode",
        "direct",
        "--elite-min-distance-override",
        str(minimum_distance),
        "--output-name",
        output_name,
    ]
    output = root / "control" / f"{output_name}.jsonl"
    log_path = log_dir / f"{output_name}.log"
    started = time.time()
    with log_path.open("a", buffering=1) as stream:
        stream.write(json.dumps({"event": "launch", "command": command}) + "\n")
        process = subprocess.run(
            command,
            cwd=Path.cwd(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        elapsed = time.time() - started
        stream.write(
            json.dumps(
                {"event": "exit", "returncode": process.returncode, "elapsed_seconds": elapsed}
            )
            + "\n"
        )
    records = len(_jsonl(output)) if output.exists() else 0
    return {
        "minimum_normalized_distance": minimum_distance,
        "output_name": output_name,
        "output_path": str(output),
        "log_path": str(log_path),
        "records": records,
        "returncode": process.returncode,
        "elapsed_seconds": time.time() - started,
    }


def _rate(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(row["strict_success"]) for row in rows]))


def _commands(row: dict[str, Any]) -> list[list[float]]:
    return [
        [float(step["command_mm"][field]) for field in ACTION_FIELDS]
        for step in row["trace"]
    ]


def _diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    means: list[float] = []
    minimums: list[float] = []
    constrained: list[bool] = []
    for row in rows:
        for step in row["trace"]:
            diagnostic = step.get("planner_diagnostics")
            if diagnostic is None:
                raise ValueError("diversity arm is missing planner diagnostics")
            for iteration in diagnostic["iteration_history"]:
                means.append(
                    float(iteration["elite_mean_pairwise_normalized_action_distance"])
                )
                minimums.append(
                    float(iteration["elite_minimum_pairwise_normalized_action_distance"])
                )
                constrained.append(bool(iteration["diversity_constraint_satisfied"]))
    return {
        "planner_iterations": len(means),
        "mean_elite_pairwise_normalized_action_distance": float(np.mean(means)),
        "mean_elite_minimum_normalized_action_distance": float(np.mean(minimums)),
        "minimum_elite_minimum_normalized_action_distance": float(np.min(minimums)),
        "diversity_constraint_satisfied_iteration_rate": float(np.mean(constrained)),
    }


def _summarize(
    root: Path,
    off_name: str,
    on_name: str,
    minimum_distance: float,
) -> dict[str, Any]:
    original_rows = _jsonl(root / "control/direct.jsonl")
    off_rows = _jsonl(root / "control" / f"{off_name}.jsonl")
    on_rows = _jsonl(root / "control" / f"{on_name}.jsonl")
    original = {_key(row): row for row in original_rows}
    off = {_key(row): row for row in off_rows}
    on = {_key(row): row for row in on_rows}
    if (
        any(len(rows) != 150 for rows in (original_rows, off_rows, on_rows))
        or any(len(rows) != 150 for rows in (original, off, on))
        or set(original) != set(off)
        or set(original) != set(on)
    ):
        raise ValueError("diversity policies are incomplete, duplicated, or unmatched")
    if any(int(case_id.rsplit("_", 1)[1]) >= 10 for case_id, _ in original):
        raise ValueError("protected case present in diversity ablation")
    default_mismatches = [
        key
        for key in original
        if _commands(original[key]) != _commands(off[key])
        or bool(original[key]["strict_success"]) != bool(off[key]["strict_success"])
        or float(original[key]["final_normalized_distance"])
        != float(off[key]["final_normalized_distance"])
    ]
    fault_keys = sorted(key for key in off if key[1] != 1.0)
    off_fault = [off[key] for key in fault_keys]
    on_fault = [on[key] for key in fault_keys]
    recoveries = [
        key
        for key in fault_keys
        if not bool(off[key]["strict_success"]) and bool(on[key]["strict_success"])
    ]
    regressions = [
        key
        for key in fault_keys
        if bool(off[key]["strict_success"]) and not bool(on[key]["strict_success"])
    ]
    by_gain: defaultdict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    by_stratum: defaultdict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for key in fault_keys:
        pair = (off[key], on[key])
        by_gain[f"{key[1]:g}"].append(pair)
        by_stratum[str(off[key]["stratum"])].append(pair)

    def paired_slice(
        pairs: list[tuple[dict[str, Any], dict[str, Any]]]
    ) -> dict[str, Any]:
        reference = [pair[0] for pair in pairs]
        alternate = [pair[1] for pair in pairs]
        return {
            "episodes": len(pairs),
            "off_success": _rate(reference),
            "on_success": _rate(alternate),
            "success_difference": _rate(alternate) - _rate(reference),
            "off_saturation_episode_rate": float(
                np.mean([int(row["saturation_count"]) > 0 for row in reference])
            ),
            "on_saturation_episode_rate": float(
                np.mean([int(row["saturation_count"]) > 0 for row in alternate])
            ),
        }

    return {
        "version": VERSION,
        "role": "supporting_development_cem_diversity_primary_branch_not_reselected",
        "split": "development_only",
        "protected_set_used": False,
        "minimum_normalized_elite_distance": minimum_distance,
        "off_policy": off_name,
        "on_policy": on_name,
        "off_replay_exactly_matches_frozen_direct": not default_mismatches,
        "default_replay_mismatch_episode_ids": [
            f"{case_id}__g{gain:g}" for case_id, gain in default_mismatches
        ],
        "fault": {
            **paired_slice(list(zip(off_fault, on_fault, strict=True))),
            "matched_group_bootstrap_95": _bootstrap(off_fault, on_fault, 2026080190),
            "recoveries": len(recoveries),
            "regressions": len(regressions),
            "exact_recovery_episode_ids": [
                f"{case_id}__g{gain:g}" for case_id, gain in recoveries
            ],
            "exact_regression_episode_ids": [
                f"{case_id}__g{gain:g}" for case_id, gain in regressions
            ],
        },
        "by_gain": {
            key: paired_slice(value)
            for key, value in sorted(by_gain.items(), key=lambda item: float(item[0]))
        },
        "by_stratum": {
            key: paired_slice(value) for key, value in sorted(by_stratum.items())
        },
        "off_planner_diagnostics": _diagnostics(off_rows),
        "on_planner_diagnostics": _diagnostics(on_rows),
        "interpretation_constraint": (
            "controlled development-only CEM ablation; no H1 retraining and no protected selection"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--minimum-distance", type=float, default=0.25)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_distance <= 0 or args.max_workers <= 0:
        raise ValueError("minimum distance and max workers must be positive")
    root = args.gate_dir.resolve()
    config = args.config.resolve()
    output = args.output.resolve()
    off_name = "direct_elite_diversity_off_postfreeze"
    on_name = f"direct_elite_diversity_{str(args.minimum_distance).replace('.', 'p')}"
    log_dir = output.parent / f"{output.stem}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    manifest: dict[str, Any] = {
        "version": VERSION,
        "status": "running",
        "split": "development_only",
        "protected_set_used": False,
        "minimum_distance": args.minimum_distance,
        "off_policy": off_name,
        "on_policy": on_name,
        "completed": [],
    }
    _atomic_json(manifest_path, manifest)
    arms = ((0.0, off_name), (args.minimum_distance, on_name))
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(_run_arm, root, config, distance, name, log_dir): name
            for distance, name in arms
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            manifest["completed"].append(result)
            if result["returncode"] != 0 or result["records"] != 150:
                failures.append(result)
            _atomic_json(manifest_path, manifest)
            print(json.dumps({"event": "cem_diversity_arm_complete", **result}), flush=True)
    manifest["failures"] = failures
    manifest["status"] = "failed" if failures else "complete"
    _atomic_json(manifest_path, manifest)
    if failures:
        raise SystemExit(1)
    report = _summarize(root, off_name, on_name, args.minimum_distance)
    _atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
