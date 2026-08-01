#!/usr/bin/env python3
"""Run and summarize matched direct/oracle/probe control over planner seeds."""

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


VERSION = "active_diagnosis_v13_three_seed_control_robustness_v1"
MODES = ("direct", "oracle_known", "probe_replan")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _name(mode: str, seed: int, base_seed: int) -> str:
    return mode if seed == base_seed else f"{mode}_seed_{seed}"


def _run(
    root: Path, config: Path, mode: str, seed: int, base_seed: int, log_dir: Path
) -> dict[str, Any]:
    name = _name(mode, seed, base_seed)
    output = root / "control" / f"{name}.jsonl"
    if seed == base_seed:
        records = len(_jsonl(output)) if output.exists() else 0
        return {
            "mode": mode,
            "seed": seed,
            "output_name": name,
            "output_path": str(output),
            "records": records,
            "returncode": 0 if records == 150 else 1,
            "reused_base_seed": True,
        }
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
        mode,
        "--output-name",
        name,
        "--root-seed-override",
        str(seed),
    ]
    log_path = log_dir / f"{name}.log"
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
        stream.write(
            json.dumps(
                {
                    "event": "exit",
                    "returncode": process.returncode,
                    "elapsed_seconds": time.time() - started,
                }
            )
            + "\n"
        )
    records = len(_jsonl(output)) if output.exists() else 0
    return {
        "mode": mode,
        "seed": seed,
        "output_name": name,
        "output_path": str(output),
        "log_path": str(log_path),
        "records": records,
        "returncode": process.returncode,
        "elapsed_seconds": time.time() - started,
        "reused_base_seed": False,
    }


def _summarize(root: Path, seeds: list[int], base_seed: int) -> dict[str, Any]:
    seed_reports = []
    for seed in seeds:
        rows = {
            mode: _jsonl(root / "control" / f"{_name(mode, seed, base_seed)}.jsonl")
            for mode in MODES
        }
        if any(len(value) != 150 for value in rows.values()):
            raise ValueError(f"seed {seed} is incomplete")
        nominal = [
            row for row in rows["direct"] if float(row["evaluator_only_true_gain"]) == 1.0
        ]
        fault = {
            mode: [row for row in value if float(row["evaluator_only_true_gain"]) != 1.0]
            for mode, value in rows.items()
        }
        rate = lambda values: float(np.mean([bool(row["strict_success"]) for row in values]))
        direct_rate = rate(fault["direct"])
        seed_reports.append(
            {
                "seed": seed,
                "nominal_direct_success": rate(nominal),
                "fault_direct_success": direct_rate,
                "fault_oracle_success": rate(fault["oracle_known"]),
                "fault_probe_replan_success": rate(fault["probe_replan"]),
                "fault_impact": rate(nominal) - direct_rate,
                "oracle_recovery": rate(fault["oracle_known"]) - direct_rate,
                "probe_control_value": rate(fault["probe_replan"]) - direct_rate,
                "probe_gain_classification_accuracy": float(
                    np.mean(
                        [bool(row["gain_classification_correct"]) for row in rows["probe_replan"]]
                    )
                ),
            }
        )
    aggregate: dict[str, Any] = {}
    for metric in (
        "nominal_direct_success",
        "fault_direct_success",
        "fault_oracle_success",
        "fault_probe_replan_success",
        "fault_impact",
        "oracle_recovery",
        "probe_control_value",
        "probe_gain_classification_accuracy",
    ):
        values = np.asarray([row[metric] for row in seed_reports], dtype=np.float64)
        aggregate[metric] = {
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "standard_deviation": float(values.std(ddof=1)),
        }
    return {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "seeds": seed_reports,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    config = args.config.resolve()
    base_seed = int(json.loads(config.read_text(encoding="utf-8"))["root_seed"])
    seeds = list(dict.fromkeys(args.seeds))
    if base_seed not in seeds:
        raise ValueError("seed list must include the preregistered base seed")
    work = root / "seed_robustness"
    log_dir = work / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    base_reuse = []
    for mode in MODES:
        output = root / "control" / f"{mode}.jsonl"
        records = len(_jsonl(output)) if output.exists() else 0
        base_reuse.append(
            {
                "mode": mode,
                "seed": base_seed,
                "output_name": mode,
                "output_path": str(output),
                "records_at_launch": records,
                "reused_base_seed": True,
            }
        )
    manifest: dict[str, Any] = {
        "version": VERSION,
        "status": "running",
        "seeds": seeds,
        "base_seed": base_seed,
        "max_workers": args.max_workers,
        "base_reuse": base_reuse,
        "completed": [],
    }
    manifest_path = work / "manifest.json"
    _atomic_json(manifest_path, manifest)
    jobs = [(mode, seed) for seed in seeds if seed != base_seed for mode in MODES]
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(_run, root, config, mode, seed, base_seed, log_dir): (mode, seed)
            for mode, seed in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            manifest["completed"].append(result)
            if result["returncode"] != 0 or result["records"] != 150:
                failures.append(result)
            _atomic_json(manifest_path, manifest)
            print(json.dumps({"event": "seed_mode_complete", **result}), flush=True)
    manifest["failures"] = failures
    manifest["status"] = "failed" if failures else "complete"
    _atomic_json(manifest_path, manifest)
    if failures:
        raise SystemExit(1)
    report = _summarize(root, seeds, base_seed)
    _atomic_json(work / "three_seed_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
