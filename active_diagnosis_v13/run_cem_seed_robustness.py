#!/usr/bin/env python3
"""Run and summarize a matched development-only CEM population seed check."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "active_diagnosis_v13_cem_population_seed_robustness_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _direct_name(seed: int, base_seed: int) -> str:
    return "direct" if seed == base_seed else f"direct_seed_{seed}"


def _population_name(
    seed: int, base_seed: int, base_policy_name: str, seed_policy_prefix: str
) -> str:
    return (
        base_policy_name
        if seed == base_seed
        else f"{seed_policy_prefix}_{seed}"
    )


def _run_one(
    root: Path,
    config: Path,
    seed: int,
    population: int,
    output_name: str,
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
        "direct",
        "--population-override",
        str(population),
        "--root-seed-override",
        str(seed),
        "--output-name",
        output_name,
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
        elapsed = time.time() - started
        stream.write(
            json.dumps(
                {
                    "event": "exit",
                    "returncode": process.returncode,
                    "elapsed_seconds": elapsed,
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


def _seed_summary(values: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [
            np.mean(values[rng.integers(0, len(values), len(values))])
            for _ in range(10000)
        ],
        dtype=np.float64,
    )
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return {
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
    }


def _summarize(
    root: Path,
    seeds: list[int],
    base_seed: int,
    population: int,
    base_policy_name: str,
    seed_policy_prefix: str,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for seed in seeds:
        reference_name = _direct_name(seed, base_seed)
        alternate_name = _population_name(
            seed, base_seed, base_policy_name, seed_policy_prefix
        )
        reference_rows = _jsonl(root / "control" / f"{reference_name}.jsonl")
        alternate_rows = _jsonl(root / "control" / f"{alternate_name}.jsonl")
        reference = {_key(row): row for row in reference_rows}
        alternate = {_key(row): row for row in alternate_rows}
        if (
            len(reference_rows) != 150
            or len(alternate_rows) != 150
            or len(reference) != 150
            or len(alternate) != 150
            or set(reference) != set(alternate)
        ):
            raise ValueError(f"seed {seed} policies are incomplete, duplicated, or unmatched")
        if any(int(case_id.rsplit("_", 1)[1]) >= 10 for case_id, _ in reference):
            raise ValueError(f"protected case present for seed {seed}")
        fault_keys = [key for key in reference if key[1] != 1.0]
        reference_fault = [reference[key] for key in fault_keys]
        alternate_fault = [alternate[key] for key in fault_keys]
        reference_success = _rate(reference_fault)
        alternate_success = _rate(alternate_fault)
        recoveries = sorted(
            key
            for key in fault_keys
            if not bool(reference[key]["strict_success"])
            and bool(alternate[key]["strict_success"])
        )
        regressions = sorted(
            key
            for key in fault_keys
            if bool(reference[key]["strict_success"])
            and not bool(alternate[key]["strict_success"])
        )
        gains = sorted({key[1] for key in fault_keys})
        strata = sorted({str(reference[key]["stratum"]) for key in fault_keys})
        reports.append(
            {
                "seed": seed,
                "reference_policy": reference_name,
                "population_policy": alternate_name,
                "reference_fault_success": reference_success,
                "population_fault_success": alternate_success,
                "fault_success_gain": alternate_success - reference_success,
                "matched_fault_recoveries": len(recoveries),
                "matched_fault_regressions": len(regressions),
                "exact_recovery_episode_ids": [
                    f"{case_id}__g{gain:g}" for case_id, gain in recoveries
                ],
                "exact_regression_episode_ids": [
                    f"{case_id}__g{gain:g}" for case_id, gain in regressions
                ],
                "reference_fault_saturation_episode_rate": float(
                    np.mean(
                        [int(reference[key]["saturation_count"]) > 0 for key in fault_keys]
                    )
                ),
                "population_fault_saturation_episode_rate": float(
                    np.mean(
                        [int(alternate[key]["saturation_count"]) > 0 for key in fault_keys]
                    )
                ),
                "by_gain": {
                    f"{gain:g}": {
                        "reference_success": _rate(
                            [reference[key] for key in fault_keys if key[1] == gain]
                        ),
                        "population_success": _rate(
                            [alternate[key] for key in fault_keys if key[1] == gain]
                        ),
                    }
                    for gain in gains
                },
                "by_stratum": {
                    stratum: {
                        "reference_success": _rate(
                            [
                                reference[key]
                                for key in fault_keys
                                if str(reference[key]["stratum"]) == stratum
                            ]
                        ),
                        "population_success": _rate(
                            [
                                alternate[key]
                                for key in fault_keys
                                if str(alternate[key]["stratum"]) == stratum
                            ]
                        ),
                    }
                    for stratum in strata
                },
            }
        )
    values = np.asarray(
        [row["fault_success_gain"] for row in reports], dtype=np.float64
    )
    recovery_seeds: defaultdict[str, list[int]] = defaultdict(list)
    regression_seeds: defaultdict[str, list[int]] = defaultdict(list)
    for row in reports:
        for episode_id in row["exact_recovery_episode_ids"]:
            recovery_seeds[episode_id].append(int(row["seed"]))
        for episode_id in row["exact_regression_episode_ids"]:
            regression_seeds[episode_id].append(int(row["seed"]))
    return {
        "version": VERSION,
        "role": "supporting_development_cem_sensitivity_primary_branch_not_reselected",
        "split": "development_only",
        "protected_set_used": False,
        "baseline_population": 24,
        "alternate_population": population,
        "seeds": reports,
        "fault_success_gain": _seed_summary(values, 2026080150),
        "passes_five_points_every_seed": bool(np.all(values >= 0.05)),
        "cross_seed_episode_stability": {
            "recoveries": [
                {
                    "episode_id": episode_id,
                    "seed_count": len(seed_values),
                    "seeds": seed_values,
                }
                for episode_id, seed_values in sorted(
                    recovery_seeds.items(), key=lambda item: (-len(item[1]), item[0])
                )
            ],
            "regressions": [
                {
                    "episode_id": episode_id,
                    "seed_count": len(seed_values),
                    "seeds": seed_values,
                }
                for episode_id, seed_values in sorted(
                    regression_seeds.items(), key=lambda item: (-len(item[1]), item[0])
                )
            ],
        },
        "interpretation_constraint": (
            "matched exploratory CEM sensitivity only; no H1 retraining and no protected selection"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--population", type=int, required=True)
    parser.add_argument("--base-policy-name", required=True)
    parser.add_argument("--seed-policy-prefix", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    config = args.config.resolve()
    output = args.output.resolve()
    base_seed = int(json.loads(config.read_text())["root_seed"])
    seeds = list(dict.fromkeys(args.seeds))
    if base_seed not in seeds:
        raise ValueError("seeds must include the configured base seed")
    if len(seeds) < 2:
        raise ValueError("at least two planner seeds are required")
    if args.population <= 1:
        raise ValueError("population must exceed one")
    if args.max_workers <= 0:
        raise ValueError("max workers must be positive")
    base_path = root / "control" / f"{args.base_policy_name}.jsonl"
    if len(_jsonl(base_path)) != 150:
        raise ValueError("base population policy must contain 150 episodes")
    log_dir = output.parent / f"{output.stem}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    manifest: dict[str, Any] = {
        "version": VERSION,
        "status": "running",
        "split": "development_only",
        "protected_set_used": False,
        "config": str(config),
        "config_sha256": _sha256(config),
        "population": args.population,
        "seeds": seeds,
        "base_seed": base_seed,
        "base_policy_name": args.base_policy_name,
        "seed_policy_prefix": args.seed_policy_prefix,
        "max_workers": args.max_workers,
        "completed": [],
    }
    _atomic_json(manifest_path, manifest)
    jobs = [seed for seed in seeds if seed != base_seed]
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                _run_one,
                root,
                config,
                seed,
                args.population,
                _population_name(
                    seed,
                    base_seed,
                    args.base_policy_name,
                    args.seed_policy_prefix,
                ),
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
            print(json.dumps({"event": "cem_seed_complete", **result}), flush=True)
    manifest["failures"] = failures
    manifest["status"] = "failed" if failures else "complete"
    _atomic_json(manifest_path, manifest)
    if failures:
        raise SystemExit(1)
    report = _summarize(
        root,
        seeds,
        base_seed,
        args.population,
        args.base_policy_name,
        args.seed_policy_prefix,
    )
    _atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
