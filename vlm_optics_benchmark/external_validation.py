#!/usr/bin/env python3
"""Run and analyze the preregistered one-shot external controller validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from active_diagnosis_v13.analyze_control_horizon_curve import _cap
from active_diagnosis_v13.analyze_sequential_horizon_rule import _sequential_cap
from active_diagnosis_v13.run_gate_a import (
    _assert_frozen_full_cases_outside_oof_map,
    _execute_control_episode,
    _load_runtime,
)


VERSION = "vlm_optics_external_validation_v1"
EXPECTED_STRATA = (
    "one_step_reachable_interior",
    "multi_step_reachable_interior",
    "reachable_boundary_or_clipping",
)
STRATUM_LABELS = {
    "one_step_reachable_interior": "A_nominal_nonboundary",
    "multi_step_reachable_interior": "B_hidden_gain_nonboundary",
    "reachable_boundary_or_clipping": "C_hidden_gain_boundary_clipping",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def selected_gain(case: Mapping[str, Any]) -> float:
    """Return the preregistered single gain for an external setup."""

    stratum = str(case["stratum"])
    if stratum == EXPECTED_STRATA[0]:
        return 1.0
    if stratum not in EXPECTED_STRATA[1:]:
        raise ValueError(f"unexpected external stratum: {stratum}")
    index = int(str(case["case_id"]).rsplit("_", 1)[1])
    return (0.5, 0.75, 1.25, 1.5)[index % 4]


def validate_suite(suite: Mapping[str, Any]) -> None:
    cases = list(suite["cases"])
    if len(cases) != 30:
        raise ValueError("external suite must contain exactly 30 cases")
    counts = Counter(str(case["stratum"]) for case in cases)
    if counts != Counter({stratum: 10 for stratum in EXPECTED_STRATA}):
        raise ValueError(f"external suite strata mismatch: {dict(counts)}")
    groups = [str(case["group_id"]) for case in cases]
    hashes = [str(case["setup_hash"]) for case in cases]
    if len(groups) != len(set(groups)) or len(hashes) != len(set(hashes)):
        raise ValueError("external setup groups and hashes must be unique")
    validation = suite["validation"]
    if int(validation["prior_group_id_overlap"]) != 0:
        raise ValueError("external suite reports prior group overlap")
    if int(validation["prior_setup_hash_overlap"]) != 0:
        raise ValueError("external suite reports prior setup-hash overlap")
    if int(validation["initial_strict_successes"]) != 0:
        raise ValueError("external suite contains an initially successful case")


def _assert_preregistration(
    preregistration: Mapping[str, Any],
    *,
    planner_seed: int,
    suite: Path,
    config: Path,
    probe_model: Path,
    rule_report: Path,
) -> None:
    expected = preregistration["controller_configuration"]
    if int(planner_seed) not in list(map(int, preregistration["planner_seeds"])):
        raise ValueError("planner seed was not preregistered")
    identities = preregistration["frozen_source_identities"]
    checks = {
        config.resolve(): str(identities["v13_config_sha256"]),
        probe_model.resolve(): str(identities["probe_model_sha256"]),
        rule_report.resolve(): str(identities["sequential_rule_report_sha256"]),
    }
    for path, expected_hash in checks.items():
        if sha256_file(path) != expected_hash:
            raise ValueError(f"frozen source hash mismatch: {path}")
    if Path(str(preregistration["external_suite"]["output_path"])).resolve() != suite.resolve():
        raise ValueError("suite path does not match preregistration")
    if int(expected["max_horizon"]) != 8:
        raise ValueError("only the preregistered horizon-eight source run is allowed")
    if str(expected["mode"]) != "probe_replan":
        raise ValueError("unexpected preregistered controller mode")


def run_external(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"one-shot output already exists: {args.output}")
    prereg = json.loads(args.preregistration.resolve().read_text(encoding="utf-8"))
    suite = json.loads(args.suite.resolve().read_text(encoding="utf-8"))
    validate_suite(suite)
    _assert_preregistration(
        prereg,
        planner_seed=args.planner_seed,
        suite=args.suite,
        config=args.config,
        probe_model=args.probe_model,
        rule_report=args.rule_report,
    )

    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    config = {
        **config,
        "root_seed": int(args.planner_seed),
        "baseline": {
            **config["baseline"],
            "evaluation_suite": str(args.suite.resolve()),
            "evaluation_suite_sha256": sha256_file(args.suite),
            "max_control_steps": 8,
        },
    }
    _, bounds, model = _load_runtime(config)
    classifier = joblib.load(args.probe_model.resolve())
    cases = list(suite["cases"])
    _assert_frozen_full_cases_outside_oof_map(cases, classifier)
    probe_selection = {
        "selected_design": "symmetric_pair",
        "selected_fraction": 0.1,
        "classifier_bundle": str(args.probe_model.resolve()),
        "classifier_bundle_sha256": sha256_file(args.probe_model),
    }
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        gain = selected_gain(case)
        row = _execute_control_episode(
            case=case,
            true_gain=gain,
            mode="probe_replan",
            config=config,
            bounds=bounds,
            model=model,
            probe_selection=probe_selection,
            classifier_bundle=classifier,
            split="development",
            policy_name=f"external_budget8_seed_{args.planner_seed}",
            use_frozen_full_probe_model=True,
        )
        row["external_stratum"] = STRATUM_LABELS[str(case["stratum"])]
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "external_episode_complete",
                    "planner_seed": args.planner_seed,
                    "episode": index,
                    "episodes": len(cases),
                    "case_id": row["case_id"],
                    "gain": gain,
                    "success_at_8": row["strict_success"],
                    "steps": row["control_steps"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.output)


def _episode_id(row: Mapping[str, Any], planner_seed: int) -> str:
    return f"{row['case_id']}__g{float(row['evaluator_only_true_gain']):g}__seed{planner_seed}"


def derive_arms(row: Mapping[str, Any], rule: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    fixed = _cap(dict(row), 4)
    maximum_raw = rule["maximum_final_distance"]
    maximum = float("inf") if maximum_raw == "infinity" else float(maximum_raw)
    sequential = _sequential_cap(
        row,
        minimum_improvement=float(rule["minimum_last_step_improvement"]),
        maximum_distance=maximum,
    )
    sequential["constraint_violation_count"] = int(row["constraint_violation_count"])
    fixed["constraint_violation_count"] = int(row["constraint_violation_count"])
    return fixed, sequential


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = np.asarray([bool(row["strict_success"]) for row in rows], dtype=float)
    steps = np.asarray([int(row["control_steps"]) for row in rows], dtype=int)
    distances = np.asarray([float(row["final_normalized_distance"]) for row in rows])
    saturation = np.asarray([int(row["saturation_count"]) > 0 for row in rows], dtype=float)
    constraints = np.asarray([int(row["constraint_violation_count"]) for row in rows])
    conditional = {}
    for step in sorted(set(map(int, steps))):
        mask = steps == step
        conditional[str(step)] = {
            "episodes": int(mask.sum()),
            "strict_success": float(successes[mask].mean()),
        }
    return {
        "episodes": len(rows),
        "strict_success": float(successes.mean()),
        "mean_executed_control_steps": float(steps.mean()),
        "median_executed_control_steps": float(np.median(steps)),
        "executed_control_step_distribution": {
            str(step): int(np.sum(steps == step)) for step in sorted(set(map(int, steps)))
        },
        "success_conditional_on_executed_control_steps": conditional,
        "saturation_episode_rate": float(saturation.mean()),
        "saturation_episode_count": int(saturation.sum()),
        "hard_constraint_violations": int(constraints.sum()),
        "mean_final_normalized_target_distance": float(distances.mean()),
        "median_final_normalized_target_distance": float(np.median(distances)),
        "final_normalized_target_distance_quantiles": {
            "q10": float(np.quantile(distances, 0.1)),
            "q25": float(np.quantile(distances, 0.25)),
            "q50": float(np.quantile(distances, 0.5)),
            "q75": float(np.quantile(distances, 0.75)),
            "q90": float(np.quantile(distances, 0.9)),
        },
    }


def _paired_bootstrap(deltas: np.ndarray, seed: int, samples: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    draws = np.mean(deltas[rng.integers(0, len(deltas), size=(samples, len(deltas)))], axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(deltas.mean()),
        "low": float(low),
        "high": float(high),
        "samples": samples,
        "seed": seed,
        "resampling_unit": "paired_planner_seed_by_setup_episode",
    }


def _mcnemar_exact(recoveries: int, regressions: int) -> float:
    discordant = recoveries + regressions
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(recoveries, regressions) + 1)) / (2**discordant)
    return float(min(1.0, 2.0 * tail))


def _trajectory(source: Sequence[Mapping[str, Any]], arms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: dict[int, list[float]] = defaultdict(list)
    for raw, arm in zip(source, arms, strict=True):
        horizon = int(arm["control_steps"])
        trace = raw["trace"][:horizon]
        values[0].append(float(raw["initial_normalized_distance"]))
        last = float(raw["initial_normalized_distance"])
        for step in range(1, 9):
            if step <= len(trace):
                last = float(trace[step - 1]["actual_target_cost"])
            values[step].append(last)
    return [
        {
            "step": step,
            "mean_normalized_target_distance": float(np.mean(distances)),
            "median_normalized_target_distance": float(np.median(distances)),
            "episodes_observed_or_carried_forward": len(distances),
        }
        for step, distances in sorted(values.items())
    ]


def _failure_category(raw: Mapping[str, Any], sequential: Mapping[str, Any]) -> str:
    decisions = list(sequential.get("sequential_rule_decisions", []))
    trace = list(raw["trace"])
    if int(sequential["saturation_count"]) > 0:
        return "boundary_interaction"
    if int(sequential["control_steps"]) < 8 and not bool(sequential["strict_success"]):
        if decisions and not bool(decisions[-1]["continued"]):
            return "premature_stopping"
    costs = [float(step["actual_target_cost"]) for step in trace[: int(sequential["control_steps"])]]
    if len(costs) >= 4 and sum(np.diff(costs) > 0.0) >= 2:
        return "oscillation"
    if trace and np.mean(
        [abs(float(step["predicted_target_cost"]) - float(step["actual_target_cost"])) for step in trace]
    ) > 1.0:
        return "planner_failure"
    if not trace:
        return "measurement_failure"
    return "premature_continuation"


def analyze(args: argparse.Namespace) -> None:
    prereg = json.loads(args.preregistration.resolve().read_text(encoding="utf-8"))
    rule_report = json.loads(args.rule_report.resolve().read_text(encoding="utf-8"))
    rule = rule_report["frozen_full_selection_seed_rule"]
    expected_rule = prereg["controller_configuration"]["continuation_rule"]
    if rule != expected_rule:
        raise ValueError("rule report no longer matches the preregistration")
    seeds = list(map(int, prereg["planner_seeds"]))
    if len(args.source) != len(seeds):
        raise ValueError("one source file is required per preregistered seed")

    raw_rows: list[dict[str, Any]] = []
    seed_rows: dict[int, list[dict[str, Any]]] = {}
    for seed, path in zip(seeds, args.source, strict=True):
        rows = read_jsonl(path.resolve())
        if len(rows) != 30 or {int(row["planner_root_seed"]) for row in rows} != {seed}:
            raise ValueError(f"source for seed {seed} is incomplete or mislabeled")
        seed_rows[seed] = rows
        raw_rows.extend(rows)

    fixed_rows: list[dict[str, Any]] = []
    sequential_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    recoveries: list[str] = []
    regressions: list[str] = []
    per_seed: dict[str, Any] = {}
    per_stratum: dict[str, Any] = {}
    for seed in seeds:
        current_fixed, current_sequential = [], []
        for raw in seed_rows[seed]:
            fixed, sequential = derive_arms(raw, rule)
            fixed["planner_seed"] = seed
            sequential["planner_seed"] = seed
            fixed["external_stratum"] = raw["external_stratum"]
            sequential["external_stratum"] = raw["external_stratum"]
            current_fixed.append(fixed)
            current_sequential.append(sequential)
            eid = _episode_id(raw, seed)
            recovered = not bool(fixed["strict_success"]) and bool(sequential["strict_success"])
            regressed = bool(fixed["strict_success"]) and not bool(sequential["strict_success"])
            if recovered:
                recoveries.append(eid)
            if regressed:
                regressions.append(eid)
            episode_rows.append(
                {
                    "episode_id": eid,
                    "case_id": raw["case_id"],
                    "setup_id": raw["group_id"],
                    "setup_hash": next(
                        str(case["setup_hash"])
                        for case in json.loads(args.suite.read_text())["cases"]
                        if str(case["case_id"]) == str(raw["case_id"])
                    ),
                    "planner_seed": seed,
                    "stratum": raw["external_stratum"],
                    "true_gain_evaluator_only": raw["evaluator_only_true_gain"],
                    "fixed4_success": fixed["strict_success"],
                    "sequential_success": sequential["strict_success"],
                    "paired_delta": int(sequential["strict_success"]) - int(fixed["strict_success"]),
                    "fixed4_steps": fixed["control_steps"],
                    "sequential_steps": sequential["control_steps"],
                    "fixed4_saturation_count": fixed["saturation_count"],
                    "sequential_saturation_count": sequential["saturation_count"],
                    "fixed4_final_distance": fixed["final_normalized_distance"],
                    "sequential_final_distance": sequential["final_normalized_distance"],
                    "recovery": recovered,
                    "regression": regressed,
                }
            )
        fixed_rows.extend(current_fixed)
        sequential_rows.extend(current_sequential)
        fixed_rate = float(np.mean([row["strict_success"] for row in current_fixed]))
        seq_rate = float(np.mean([row["strict_success"] for row in current_sequential]))
        per_seed[str(seed)] = {
            "fixed4": _summary(current_fixed),
            "sequential": _summary(current_sequential),
            "paired_percentage_point_difference": 100.0 * (seq_rate - fixed_rate),
        }

    for label in STRATUM_LABELS.values():
        fixed = [row for row in fixed_rows if row["external_stratum"] == label]
        sequential = [row for row in sequential_rows if row["external_stratum"] == label]
        per_stratum[label] = {
            "fixed4": _summary(fixed),
            "sequential": _summary(sequential),
            "paired_percentage_point_difference": 100.0
            * (np.mean([row["strict_success"] for row in sequential]) - np.mean([row["strict_success"] for row in fixed])),
            "by_seed": {
                str(seed): {
                    "fixed4_strict_success": float(np.mean([row["strict_success"] for row in fixed if row["planner_seed"] == seed])),
                    "sequential_strict_success": float(np.mean([row["strict_success"] for row in sequential if row["planner_seed"] == seed])),
                }
                for seed in seeds
            },
        }

    deltas = np.asarray(
        [int(seq["strict_success"]) - int(fixed["strict_success"]) for fixed, seq in zip(fixed_rows, sequential_rows, strict=True)],
        dtype=float,
    )
    overall_fixed = _summary(fixed_rows)
    overall_sequential = _summary(sequential_rows)
    saturation_delta = (
        float(overall_sequential["saturation_episode_rate"])
        - float(overall_fixed["saturation_episode_rate"])
    )
    constraint_delta = int(overall_sequential["hard_constraint_violations"]) - int(
        overall_fixed["hard_constraint_violations"]
    )
    gate_criteria = prereg["go_no_go"]["backbone_freeze_gate"]
    criteria = {
        "overall_improvement_at_least_10pp": float(deltas.mean())
        >= float(gate_criteria["minimum_overall_paired_success_improvement"]),
        "nonnegative_for_both_planner_seeds": all(
            float(row["paired_percentage_point_difference"]) >= 0.0 for row in per_seed.values()
        ),
        "matched_regressions_at_most_2": len(regressions)
        <= int(gate_criteria["maximum_matched_regressions"]),
        "saturation_not_materially_increased": saturation_delta
        <= float(gate_criteria["maximum_saturation_episode_rate_increase"]),
        "hard_constraints_not_increased": constraint_delta
        <= int(gate_criteria["maximum_hard_constraint_violation_increase"]),
    }
    promoted = all(criteria.values())
    taxonomy = Counter()
    if not promoted:
        for raw, seq in zip(raw_rows, sequential_rows, strict=True):
            if not bool(seq["strict_success"]):
                taxonomy[_failure_category(raw, seq)] += 1

    report = {
        "version": VERSION,
        "evaluation_role": "new_frozen_external_evaluation",
        "protected_set_used": False,
        "selection_or_retuning_on_external_results": False,
        "episodes": len(fixed_rows),
        "unique_setups": len({row["group_id"] for row in fixed_rows}),
        "planner_seeds": seeds,
        "controller_configuration": prereg["controller_configuration"],
        "overall": {
            "fixed4": overall_fixed,
            "sequential": overall_sequential,
            "paired_percentage_point_difference": 100.0 * float(deltas.mean()),
        },
        "by_seed": per_seed,
        "by_stratum": per_stratum,
        "matched": {
            "recoveries": len(recoveries),
            "regressions": len(regressions),
            "recovery_episode_ids": recoveries,
            "regression_episode_ids": regressions,
            "mcnemar_exact_two_sided_p": _mcnemar_exact(len(recoveries), len(regressions)),
        },
        "paired_success_bootstrap_95": _paired_bootstrap(
            deltas,
            seed=int(prereg["statistical_analyses"]["bootstrap_seed"]),
            samples=int(prereg["statistical_analyses"]["bootstrap_samples"]),
        ),
        "target_distance_trajectory": {
            "fixed4": _trajectory(raw_rows, fixed_rows),
            "sequential": _trajectory(raw_rows, sequential_rows),
            "post_stop_values_are_carried_forward": True,
        },
        "safety_deltas": {
            "saturation_episode_rate": saturation_delta,
            "hard_constraint_violation_count": constraint_delta,
        },
        "backbone_freeze_gate": {
            "criteria": criteria,
            "passed": promoted,
            "promoted_backbone": "frozen_sequential_max8" if promoted else "fixed4_h1_cem",
        },
        "failure_taxonomy_if_gate_failed": dict(sorted(taxonomy.items())),
        "source_identities": {
            "preregistration": str(args.preregistration.resolve()),
            "preregistration_sha256": sha256_file(args.preregistration),
            "suite": str(args.suite.resolve()),
            "suite_sha256": sha256_file(args.suite),
            "rule_report": str(args.rule_report.resolve()),
            "rule_report_sha256": sha256_file(args.rule_report),
            "source_budget8": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.source
            ],
        },
    }
    atomic_json(args.output.resolve(), report)
    args.episodes_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.episodes_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    print(json.dumps(report["backbone_freeze_gate"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--preregistration", type=Path, required=True)
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--probe-model", type=Path, required=True)
    run.add_argument("--rule-report", type=Path, required=True)
    run.add_argument("--planner-seed", type=int, required=True)
    run.add_argument("--output", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--preregistration", type=Path, required=True)
    analyze_parser.add_argument("--suite", type=Path, required=True)
    analyze_parser.add_argument("--rule-report", type=Path, required=True)
    analyze_parser.add_argument("--source", action="append", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--episodes-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        run_external(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()

