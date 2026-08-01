#!/usr/bin/env python3
"""Replicate six-step direct/frozen-probe controls on planner seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.analyze_control_step_budget import (
    _assert_replay,
    _key,
    _read_jsonl,
    _switched_row,
)

VERSION = "active_diagnosis_v13_control_budget_seed_robustness_v2"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run(command: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    return {
        "command": command,
        "log": str(log_path.resolve()),
        "returncode": int(result.returncode),
    }


def _rate(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(row["strict_success"]) for row in rows]))


def _seed_summary(
    *,
    seed: int,
    baseline_direct: list[dict[str, Any]],
    budget6_direct: list[dict[str, Any]],
    baseline_probe: list[dict[str, Any]],
    budget6_probe: list[dict[str, Any]],
    rule: dict[str, Any],
) -> dict[str, Any]:
    arms = {
        "direct": ({_key(row): row for row in baseline_direct}, {_key(row): row for row in budget6_direct}),
        "probe": ({_key(row): row for row in baseline_probe}, {_key(row): row for row in budget6_probe}),
    }
    for name, (baseline, treatment) in arms.items():
        if len(baseline) != 150 or len(treatment) != 150 or set(baseline) != set(treatment):
            raise ValueError(f"seed {seed} {name} does not contain 150 matched rows")
        mismatches = _assert_replay(baseline, treatment)
        if mismatches:
            raise ValueError(f"seed {seed} {name} prefix mismatch: {mismatches[:3]}")
    minimum = float(rule["minimum_last_step_improvement"])
    maximum_raw = rule["maximum_final_distance"]
    maximum = float("inf") if maximum_raw == "infinity" else float(maximum_raw)
    adaptive_probe = {
        key: _switched_row(
            arms["probe"][0][key],
            arms["probe"][1][key],
            minimum_last_improvement=minimum,
            maximum_final_distance=maximum,
        )
        for key in arms["probe"][0]
    }
    output = {"seed": seed, "first_four_trace_exact": True, "arms": {}}
    for name, (baseline, treatment) in arms.items():
        fault_keys = [key for key in baseline if key[1] != 1.0]
        boundary_keys = [
            key
            for key in fault_keys
            if str(baseline[key]["stratum"]) == "reachable_boundary_or_clipping"
        ]
        output["arms"][name] = {
            "baseline_fault_success": _rate([baseline[key] for key in fault_keys]),
            "budget6_fault_success": _rate([treatment[key] for key in fault_keys]),
            "fault_success_difference": _rate([treatment[key] for key in fault_keys])
            - _rate([baseline[key] for key in fault_keys]),
            "baseline_boundary_fault_success": _rate([baseline[key] for key in boundary_keys]),
            "budget6_boundary_fault_success": _rate([treatment[key] for key in boundary_keys]),
            "boundary_fault_success_difference": _rate(
                [treatment[key] for key in boundary_keys]
            )
            - _rate([baseline[key] for key in boundary_keys]),
            "baseline_fault_saturation_rate": float(
                np.mean([int(baseline[key]["saturation_count"]) > 0 for key in fault_keys])
            ),
            "budget6_fault_saturation_rate": float(
                np.mean([int(treatment[key]["saturation_count"]) > 0 for key in fault_keys])
            ),
            "recoveries": sum(
                not bool(baseline[key]["strict_success"])
                and bool(treatment[key]["strict_success"])
                for key in fault_keys
            ),
            "regressions": sum(
                bool(baseline[key]["strict_success"])
                and not bool(treatment[key]["strict_success"])
                for key in fault_keys
            ),
        }
    probe_baseline = arms["probe"][0]
    fault_keys = [key for key in probe_baseline if key[1] != 1.0]
    boundary_keys = [
        key
        for key in fault_keys
        if str(probe_baseline[key]["stratum"]) == "reachable_boundary_or_clipping"
    ]
    output["adaptive_probe"] = {
        "rule": rule,
        "selected_continuations": sum(
            bool(row["continuation_selected"]) for row in adaptive_probe.values()
        ),
        "fault_success": _rate([adaptive_probe[key] for key in fault_keys]),
        "fault_success_difference": _rate([adaptive_probe[key] for key in fault_keys])
        - _rate([probe_baseline[key] for key in fault_keys]),
        "boundary_fault_success": _rate([adaptive_probe[key] for key in boundary_keys]),
        "boundary_fault_success_difference": _rate(
            [adaptive_probe[key] for key in boundary_keys]
        )
        - _rate([probe_baseline[key] for key in boundary_keys]),
        "fault_saturation_rate": float(
            np.mean([int(adaptive_probe[key]["saturation_count"]) > 0 for key in fault_keys])
        ),
        "mean_control_steps": float(
            np.mean([int(row["control_steps"]) for row in adaptive_probe.values()])
        ),
    }
    return output


def _seed_bootstrap(values: list[float], seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = [
        float(np.mean(array[rng.integers(0, len(array), len(array))]))
        for _ in range(4000)
    ]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "low": float(low),
        "high": float(high),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-model", type=Path, required=True)
    parser.add_argument("--probe-design", required=True)
    parser.add_argument("--probe-fraction", type=float, required=True)
    parser.add_argument("--seed1-analysis", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if 2026080101 in args.seeds:
        raise ValueError("seed 2026080101 is the preexisting base arm, not a worker seed")
    seed1_analysis = json.loads(args.seed1_analysis.resolve().read_text())
    if seed1_analysis.get("protected_set_used") is not False:
        raise ValueError("seed-1 budget analysis must be development-only")
    rule = seed1_analysis["arms"]["probe"]["exploratory_full_development_rule"][
        "selected"
    ]["rule"]
    commands = []
    for seed in args.seeds:
        for mode in ("direct", "probe_replan"):
            name = (
                f"direct_budget6_seed_{seed}"
                if mode == "direct"
                else f"probe_nores_budget6_seed_{seed}"
            )
            command = [
                str(args.python.resolve()),
                "-m",
                "active_diagnosis_v13.run_gate_a",
                "control",
                "--config",
                str(args.config.resolve()),
                "--output-dir",
                str(args.output_dir.resolve()),
                "--split",
                "development",
                "--mode",
                mode,
                "--root-seed-override",
                str(seed),
                "--max-control-steps-override",
                "6",
                "--output-name",
                name,
            ]
            if mode == "probe_replan":
                command.extend(
                    [
                        "--probe-model",
                        str(args.probe_model.resolve()),
                        "--probe-design",
                        args.probe_design,
                        "--probe-fraction",
                        f"{args.probe_fraction:g}",
                    ]
                )
            commands.append((seed, mode, name, command))
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                _run,
                command,
                args.logs_dir / f"{name}.log",
            ): (seed, mode, name)
            for seed, mode, name, command in commands
        }
        for future in as_completed(futures):
            seed, mode, name = futures[future]
            result = future.result()
            result.update({"seed": seed, "mode": mode, "name": name})
            results.append(result)
            _atomic_json(
                args.manifest,
                {
                    "version": VERSION,
                    "protected_set_used": False,
                    "completed": sorted(results, key=lambda row: (row["seed"], row["mode"])),
                },
            )
            if result["returncode"] != 0:
                raise RuntimeError(f"seed worker failed: {result}")
    control = args.output_dir.resolve() / "control"
    seeds = [2026080101, *sorted(set(args.seeds))]
    summaries = []
    for seed in seeds:
        suffix = "" if seed == 2026080101 else f"_seed_{seed}"
        summaries.append(
            _seed_summary(
                seed=seed,
                baseline_direct=_read_jsonl(control / f"direct{suffix}.jsonl"),
                budget6_direct=_read_jsonl(control / f"direct_budget6{suffix}.jsonl"),
                baseline_probe=_read_jsonl(
                    control
                    / (
                        "probe_symmetric_pair_f0p1_no_residual.jsonl"
                        if seed == 2026080101
                        else f"probe_nores{suffix}.jsonl"
                    )
                ),
                budget6_probe=_read_jsonl(control / f"probe_nores_budget6{suffix}.jsonl"),
                rule=rule,
            )
        )
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "frozen_seed1_visible_rule": rule,
        "seeds": summaries,
        "seed_bootstrap": {
            "direct_fixed6_fault_difference": _seed_bootstrap(
                [row["arms"]["direct"]["fault_success_difference"] for row in summaries],
                2026080161,
            ),
            "probe_fixed6_fault_difference": _seed_bootstrap(
                [row["arms"]["probe"]["fault_success_difference"] for row in summaries],
                2026080162,
            ),
            "probe_adaptive_fault_difference": _seed_bootstrap(
                [row["adaptive_probe"]["fault_success_difference"] for row in summaries],
                2026080163,
            ),
        },
        "interpretation_guard": (
            "The continuation rule was selected on seed 1 development only, then held fixed "
            f"for {len(summaries) - 1} additional planner seeds. No protected data or policy "
            "reselection is allowed."
        ),
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
