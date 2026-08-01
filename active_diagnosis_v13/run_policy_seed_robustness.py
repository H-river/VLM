#!/usr/bin/env python3
"""Run one development probe policy over existing matched direct planner seeds."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "active_diagnosis_v13_policy_seed_robustness_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy_name(seed: int, base_seed: int, base_name: str, prefix: str) -> str:
    return base_name if seed == base_seed else f"{prefix}_{seed}"


def _direct_name(seed: int, base_seed: int) -> str:
    return "direct" if seed == base_seed else f"direct_seed_{seed}"


def _run_one(
    root: Path,
    config: Path,
    model: Path,
    seed: int,
    output_name: str,
    design: str,
    fraction: float,
    log_dir: Path,
) -> dict[str, Any]:
    output = root / "control" / f"{output_name}.jsonl"
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
        "probe_replan",
        "--probe-model",
        str(model),
        "--probe-design",
        design,
        "--probe-fraction",
        str(fraction),
        "--output-name",
        output_name,
        "--root-seed-override",
        str(seed),
    ]
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
        "seed": seed,
        "output_name": output_name,
        "output_path": str(output),
        "log_path": str(log_path),
        "records": records,
        "returncode": process.returncode,
        "elapsed_seconds": time.time() - started,
    }


def _rate(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(row["strict_success"]) for row in rows]))


def _summarize(
    root: Path,
    seeds: list[int],
    base_seed: int,
    base_name: str,
    prefix: str,
    model: Path,
) -> dict[str, Any]:
    reports = []
    for seed in seeds:
        direct = _jsonl(root / "control" / f"{_direct_name(seed, base_seed)}.jsonl")
        policy_name = _policy_name(seed, base_seed, base_name, prefix)
        policy = _jsonl(root / "control" / f"{policy_name}.jsonl")
        if len(direct) != 150 or len(policy) != 150:
            raise ValueError(f"seed {seed} is incomplete")
        direct_fault = [
            row for row in direct if float(row["evaluator_only_true_gain"]) != 1.0
        ]
        policy_fault = [
            row for row in policy if float(row["evaluator_only_true_gain"]) != 1.0
        ]
        direct_rate = _rate(direct_fault)
        policy_rate = _rate(policy_fault)
        reports.append(
            {
                "seed": seed,
                "policy": policy_name,
                "direct_fault_success": direct_rate,
                "policy_fault_success": policy_rate,
                "control_value": policy_rate - direct_rate,
                "fault_saturation_episode_rate": float(
                    np.mean([int(row["saturation_count"]) > 0 for row in policy_fault])
                ),
                "mean_total_additional_steps": float(
                    np.mean([int(row["total_additional_steps"]) for row in policy])
                ),
                "mean_absolute_gain_error": float(
                    np.mean(
                        [
                            abs(
                                float(row["gain_belief"])
                                - float(row["evaluator_only_true_gain"])
                            )
                            for row in policy
                        ]
                    )
                ),
            }
        )
    values = np.asarray([row["control_value"] for row in reports], dtype=np.float64)
    rng = np.random.default_rng(2026080131)
    bootstrap = np.asarray(
        [
            np.mean(values[rng.integers(0, len(values), len(values))])
            for _ in range(10000)
        ]
    )
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "model": str(model),
        "model_sha256": _sha256(model),
        "seeds": reports,
        "control_value": {
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "standard_deviation": float(values.std(ddof=1)),
            "planner_seed_bootstrap_mean_95": {
                "estimate": float(values.mean()),
                "low": float(low),
                "high": float(high),
                "resampling_unit": "planner_seed",
                "samples": 10000,
            },
            "passes_five_points_every_seed": bool(np.all(values >= 0.05)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probe-model", type=Path, required=True)
    parser.add_argument("--base-policy-name", required=True)
    parser.add_argument("--seed-policy-prefix", required=True)
    parser.add_argument("--probe-design", default="symmetric_pair")
    parser.add_argument("--probe-fraction", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    config = args.config.resolve()
    model = args.probe_model.resolve()
    output = args.output.resolve()
    base_seed = int(json.loads(config.read_text())["root_seed"])
    seeds = list(dict.fromkeys(args.seeds))
    if base_seed not in seeds:
        raise ValueError("seeds must include the configured base seed")
    base_rows = _jsonl(root / "control" / f"{args.base_policy_name}.jsonl")
    if len(base_rows) != 150:
        raise ValueError("base policy must be complete before seed expansion")
    log_dir = output.parent / f"{output.stem}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    manifest: dict[str, Any] = {
        "version": VERSION,
        "status": "running",
        "split": "development_only",
        "protected_set_used": False,
        "model": str(model),
        "model_sha256": _sha256(model),
        "seeds": seeds,
        "base_seed": base_seed,
        "base_policy_name": args.base_policy_name,
        "seed_policy_prefix": args.seed_policy_prefix,
        "max_workers": args.max_workers,
        "completed": [],
    }
    _atomic_json(manifest_path, manifest)
    jobs = [seed for seed in seeds if seed != base_seed]
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                _run_one,
                root,
                config,
                model,
                seed,
                _policy_name(
                    seed,
                    base_seed,
                    args.base_policy_name,
                    args.seed_policy_prefix,
                ),
                args.probe_design,
                args.probe_fraction,
                log_dir,
            ): seed
            for seed in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            manifest["completed"].append(result)
            if result["returncode"] != 0 or result["records"] != 150:
                failures.append(result)
            _atomic_json(manifest_path, manifest)
            print(json.dumps({"event": "policy_seed_complete", **result}), flush=True)
    manifest["failures"] = failures
    manifest["status"] = "failed" if failures else "complete"
    _atomic_json(manifest_path, manifest)
    if failures:
        raise SystemExit(1)
    report = _summarize(
        root,
        seeds,
        base_seed,
        args.base_policy_name,
        args.seed_policy_prefix,
        model,
    )
    _atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
