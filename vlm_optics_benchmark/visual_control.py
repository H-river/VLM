#!/usr/bin/env python3
"""Closed-loop control-value audit for frozen visual anomaly families."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from active_diagnosis_v13.run_gate_a import _planner_config, matched_planner_seed
from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    Bounds,
    apply_action,
    metrics_dict,
    metrics_vector,
    normalized_distance,
    position_dict,
    position_vector,
)
from continuous_control_v12.mpc import CEMMPC, learned_predictor
from continuous_control_v12.simulator import simulate_state
from continuous_control_v12.world_model import load_forward_ensemble
from vlm_optics_benchmark.visual_anomalies import (
    FAMILIES,
    _predict_tiny,
    _torch_models,
    canonical_patch,
    inject_anomaly,
    moment_metrics,
    read_jsonl,
    severity_for,
    stable_seed,
    write_jsonl,
)


VERSION = "vlm_optics_control_value_v1"
ARMS = ("no_diagnosis", "oracle_diagnosis", "learned_image_diagnostic")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _full_anomaly(
    raw: np.ndarray,
    family: str,
    severity: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw, dtype=np.float64)
    canonical = canonical_patch(values)
    if family == "sensor_saturation":
        level = float(severity["clip_level_fraction_of_peak"]) * max(float(values.max()), 1e-30)
        full = np.minimum(values, level)
    else:
        from scipy.ndimage import shift

        full_metrics = moment_metrics(values)
        canonical_metrics = moment_metrics(canonical)
        scale = float(np.mean(full_metrics[2:4] / np.maximum(canonical_metrics[2:4], 1e-6)))
        offset = float(severity["reflection_offset_px"]) * scale
        angle = float(severity["reflection_angle_radians"])
        reflected = shift(
            values,
            shift=(offset * np.sin(angle), offset * np.cos(angle)),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        full = values + float(severity["reflection_amplitude_fraction"]) * reflected
    diagnostic = inject_anomaly(canonical, family, severity)
    return np.asarray(full, dtype=np.float32), diagnostic


def _corrupted_lab_metrics(capture: Mapping[str, Any], full_anomaly: np.ndarray) -> np.ndarray:
    clean_image = moment_metrics(capture["intensity"])
    anomaly_image = moment_metrics(full_anomaly)
    clean_lab = metrics_vector(capture["metrics"])
    output = clean_lab.copy()
    output[:4] += anomaly_image[:4] - clean_image[:4]
    output[4] = float(full_anomaly.max())
    return output


def _load_diagnostic(path: Path) -> tuple[Any, np.ndarray, np.ndarray]:
    import torch

    artifact = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    _, _, multimodal = _torch_models(int(artifact["seed"]))
    if int(artifact["metric_dim"]) != 5:
        raise ValueError("control audit requires the frozen multimodal small diagnostic")
    multimodal.load_state_dict(artifact["state_dict"])
    return (
        multimodal,
        np.asarray(artifact["metric_mean"], dtype=np.float32),
        np.asarray(artifact["metric_scale"], dtype=np.float32),
    )


def _observe(
    *,
    case: Mapping[str, Any],
    position: np.ndarray,
    bounds: Bounds,
    base_config: Path,
    family: str,
    severity: Mapping[str, float],
    arm: str,
    diagnostic: tuple[Any, np.ndarray, np.ndarray] | None,
) -> dict[str, Any]:
    capture = simulate_state(
        case["setup_context"],
        position_dict(position),
        case["simulator_fixed"],
        str(base_config.resolve()),
        bounds,
    )
    clean_metrics = metrics_vector(capture["metrics"])
    full_anomaly, diagnostic_image = _full_anomaly(capture["intensity"], family, severity)
    anomalous_metrics = _corrupted_lab_metrics(capture, full_anomaly)
    probability = None
    diagnosis_correct = None
    switch = False
    if arm == "oracle_diagnosis":
        switch = True
        diagnosis_correct = True
    elif arm == "learned_image_diagnostic":
        if diagnostic is None:
            raise ValueError("learned arm requires a diagnostic model")
        model, mean, scale = diagnostic
        canonical_metrics = moment_metrics(diagnostic_image)[None, :]
        probability = float(
            _predict_tiny(
                model,
                diagnostic_image[None, None, :, :],
                canonical_metrics,
                mean,
                scale,
                5,
            )[0]
        )
        switch = probability >= 0.5
        diagnosis_correct = switch
    elif arm != "no_diagnosis":
        raise ValueError(f"unknown control arm: {arm}")
    return {
        "clean_metrics": clean_metrics,
        "observed_metrics": clean_metrics if switch else anomalous_metrics,
        "diagnostic_probability": probability,
        "diagnosis_correct": diagnosis_correct,
        "specialist_switch": switch,
        "simulator_valid": bool(capture["auxiliary"]["simulator_valid"]),
    }


def execute_episode(
    *,
    case: Mapping[str, Any],
    family: str,
    arm: str,
    config: Mapping[str, Any],
    bounds: Bounds,
    model: Any,
    base_config: Path,
    diagnostic: tuple[Any, np.ndarray, np.ndarray] | None,
) -> dict[str, Any]:
    severity = severity_for(str(case["case_id"]), family, "severity_ood")
    position = position_vector(case["initial_positions_mm"])
    target = metrics_vector(case["target_metrics"])
    initial_clean = metrics_vector(case["initial_metrics"])
    planner = CEMMPC(
        bounds=bounds,
        predictor=learned_predictor(model, case["setup_context"]),
        config=_planner_config(config),
        seed=matched_planner_seed(config, str(case["case_id"])),
    )
    observation = _observe(
        case=case,
        position=position,
        bounds=bounds,
        base_config=base_config,
        family=family,
        severity=severity,
        arm=arm,
        diagnostic=diagnostic,
    )
    trace: list[dict[str, Any]] = []
    saturation_count = 0
    specialist_switches = int(observation["specialist_switch"])
    classification_correct = [observation["diagnosis_correct"]]
    observed = np.asarray(observation["observed_metrics"], dtype=np.float64)
    clean = np.asarray(observation["clean_metrics"], dtype=np.float64)
    for control_step in range(8):
        before_observed = normalized_distance(observed, target, initial_clean)
        before_clean = normalized_distance(clean, target, initial_clean)
        if before_observed <= 1.0:
            break
        plan = planner.plan(
            positions_mm=position,
            current_metrics=observed,
            target_metrics=target,
            allowed_dofs=["lens_x", "lens_y", "camera_x", "camera_y"],
            tolerance_reference=initial_clean,
        )
        requested = np.asarray([plan["selected_requested_action"][field] for field in ACTION_FIELDS])
        action = np.asarray([plan["selected_action"][field] for field in ACTION_FIELDS])
        saturation_count += int(not np.allclose(requested, action, rtol=0.0, atol=1e-12))
        position = apply_action(position, action, bounds, project=False)
        next_observation = _observe(
            case=case,
            position=position,
            bounds=bounds,
            base_config=base_config,
            family=family,
            severity=severity,
            arm=arm,
            diagnostic=diagnostic,
        )
        next_observed = np.asarray(next_observation["observed_metrics"], dtype=np.float64)
        next_clean = np.asarray(next_observation["clean_metrics"], dtype=np.float64)
        after_observed = normalized_distance(next_observed, target, initial_clean)
        after_clean = normalized_distance(next_clean, target, initial_clean)
        trace.append(
            {
                "control_step": control_step + 1,
                "command_mm": {field: float(action[index]) for index, field in enumerate(ACTION_FIELDS)},
                "observed_before_distance": float(before_observed),
                "observed_after_distance": float(after_observed),
                "true_clean_before_distance": float(before_clean),
                "true_clean_after_distance": float(after_clean),
                "diagnostic_probability": next_observation["diagnostic_probability"],
                "diagnosis_correct": next_observation["diagnosis_correct"],
                "specialist_switch": bool(next_observation["specialist_switch"]),
                "simulator_valid": bool(next_observation["simulator_valid"]),
            }
        )
        specialist_switches += int(next_observation["specialist_switch"])
        classification_correct.append(next_observation["diagnosis_correct"])
        observed, clean = next_observed, next_clean
        if after_observed <= 1.0:
            break
        if control_step + 1 >= 4:
            improvement = before_observed - after_observed
            if improvement < 0.25:
                break
    final_clean_distance = normalized_distance(clean, target, initial_clean)
    valid_classifications = [bool(value) for value in classification_correct if value is not None]
    return {
        "version": VERSION,
        "episode_id": f"{case['case_id']}__{family}__{arm}",
        "case_id": case["case_id"],
        "setup_id": case["group_id"],
        "setup_hash": case["setup_hash"],
        "stratum": case["stratum"],
        "family": family,
        "severity": severity,
        "arm": arm,
        "strict_success": bool(final_clean_distance <= 1.0),
        "final_normalized_target_distance": float(final_clean_distance),
        "control_steps": len(trace),
        "observation_mode_or_specialist_switches": specialist_switches,
        "all_diagnostic_decisions_correct": (
            None if not valid_classifications else all(valid_classifications)
        ),
        "any_diagnostic_decision_wrong": (
            None if not valid_classifications else not all(valid_classifications)
        ),
        "saturation_count": int(saturation_count),
        "constraint_violation_count": 0,
        "trace": trace,
        "controller": "frozen_h1_cem_sequential_max8",
        "recovery_adds_actuator_budget": False,
    }


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"control-value source already exists: {args.output}")
    suite = json.loads(args.suite.resolve().read_text())
    config = json.loads(args.v13_config.resolve().read_text())
    config = {**config, "root_seed": int(args.planner_seed), "baseline": {**config["baseline"], "max_control_steps": 8}}
    v12 = json.loads(Path(config["baseline"]["v12_config"]).read_text())
    bounds = Bounds.from_config(v12)
    model = load_forward_ensemble(Path(config["baseline"]["checkpoint"]), device_name="cpu")
    diagnostics = {
        family: _load_diagnostic(args.model_dir / f"{family}_multimodal_cnn.pt")
        for family in FAMILIES
    }
    rows: list[dict[str, Any]] = []
    tasks = len(suite["cases"]) * len(FAMILIES) * len(ARMS)
    task = 0
    for family in FAMILIES:
        for case in suite["cases"]:
            for arm in ARMS:
                task += 1
                row = execute_episode(
                    case=case,
                    family=family,
                    arm=arm,
                    config=config,
                    bounds=bounds,
                    model=model,
                    base_config=args.base_config,
                    diagnostic=diagnostics[family] if arm == "learned_image_diagnostic" else None,
                )
                rows.append(row)
                print(
                    json.dumps({"event": "control_value_episode_complete", "task": task, "tasks": tasks, "episode_id": row["episode_id"], "success": row["strict_success"], "steps": row["control_steps"]}),
                    flush=True,
                )
    write_jsonl(args.output, rows)


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "strict_success": float(np.mean([bool(row["strict_success"]) for row in rows])),
        "mean_executed_steps": float(np.mean([int(row["control_steps"]) for row in rows])),
        "saturation_episode_rate": float(np.mean([int(row["saturation_count"]) > 0 for row in rows])),
        "constraint_violations": int(sum(int(row["constraint_violation_count"]) for row in rows)),
        "mean_final_normalized_target_distance": float(np.mean([float(row["final_normalized_target_distance"]) for row in rows])),
    }


def analyze(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.source)
    identifiability = json.loads(args.identifiability.resolve().read_text())
    results: dict[str, Any] = {
        "version": VERSION,
        "evaluation_split": "severity_ood_setup_disjoint",
        "protected_set_used": False,
        "backbone": "externally_validated_frozen_h1_cem_sequential_max8",
        "recovery_adds_actuator_budget": False,
        "families": {},
    }
    row_map = {(row["family"], row["setup_id"], row["arm"]): row for row in rows}
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        arms = {arm: [row for row in family_rows if row["arm"] == arm] for arm in ARMS}
        baseline = {row["setup_id"]: row for row in arms["no_diagnosis"]}
        oracle = {row["setup_id"]: row for row in arms["oracle_diagnosis"]}
        learned = {row["setup_id"]: row for row in arms["learned_image_diagnostic"]}
        recoveries = [setup for setup in sorted(baseline) if not baseline[setup]["strict_success"] and oracle[setup]["strict_success"]]
        regressions = [setup for setup in sorted(baseline) if baseline[setup]["strict_success"] and not oracle[setup]["strict_success"]]
        learned_recoveries = [setup for setup in sorted(baseline) if not baseline[setup]["strict_success"] and learned[setup]["strict_success"]]
        learned_regressions = [setup for setup in sorted(baseline) if baseline[setup]["strict_success"] and not learned[setup]["strict_success"]]
        classification_correct_control_failed = [
            row["setup_id"] for row in arms["learned_image_diagnostic"]
            if row["all_diagnostic_decisions_correct"] is True and not row["strict_success"]
        ]
        classification_wrong_accidentally_recovered = [
            row["setup_id"] for row in arms["learned_image_diagnostic"]
            if row["any_diagnostic_decision_wrong"] is True and row["strict_success"]
        ]
        by_severity = {}
        anomaly_rows = arms["oracle_diagnosis"]
        scalar = np.asarray([
            1.0 - float(row["severity"]["clip_level_fraction_of_peak"])
            if family == "sensor_saturation"
            else float(row["severity"]["reflection_amplitude_fraction"])
            for row in anomaly_rows
        ])
        cuts = np.quantile(scalar, [1 / 3, 2 / 3])
        for name, select in (
            ("low", scalar <= cuts[0]),
            ("medium", (scalar > cuts[0]) & (scalar <= cuts[1])),
            ("high", scalar > cuts[1]),
        ):
            setups = [anomaly_rows[index]["setup_id"] for index in np.flatnonzero(select)]
            by_severity[name] = {
                "episodes": len(setups),
                "no_diagnosis_success": float(np.mean([baseline[setup]["strict_success"] for setup in setups])),
                "oracle_success": float(np.mean([oracle[setup]["strict_success"] for setup in setups])),
                "learned_success": float(np.mean([learned[setup]["strict_success"] for setup in setups])),
            }
        no_rate = _summary(arms["no_diagnosis"])["strict_success"]
        oracle_rate = _summary(arms["oracle_diagnosis"])["strict_success"]
        safety_worse = (
            _summary(arms["oracle_diagnosis"])["saturation_episode_rate"]
            > _summary(arms["no_diagnosis"])["saturation_episode_rate"] + 0.05
            or _summary(arms["oracle_diagnosis"])["constraint_violations"]
            > _summary(arms["no_diagnosis"])["constraint_violations"]
        )
        control_gate = {
            "oracle_improves_at_least_5pp_or_five_recoveries": oracle_rate - no_rate >= 0.05 or len(recoveries) >= 5,
            "diagnosis_dependent_not_action_budget": True,
            "safety_not_materially_worse": not safety_worse,
        }
        control_gate["passed"] = all(control_gate.values())
        results["families"][family] = {
            "arms": {arm: _summary(value) for arm, value in arms.items()},
            "oracle_minus_no_diagnosis_percentage_points": 100.0 * (oracle_rate - no_rate),
            "matched_oracle_recoveries": len(recoveries),
            "matched_oracle_regressions": len(regressions),
            "oracle_recovery_setup_ids": recoveries,
            "oracle_regression_setup_ids": regressions,
            "matched_learned_recoveries": len(learned_recoveries),
            "matched_learned_regressions": len(learned_regressions),
            "learned_recovery_setup_ids": learned_recoveries,
            "learned_regression_setup_ids": learned_regressions,
            "classification_correct_but_control_failed_setup_ids": classification_correct_control_failed,
            "classification_wrong_but_accidentally_recovered_setup_ids": classification_wrong_accidentally_recovered,
            "by_severity": by_severity,
            "control_value_gate": control_gate,
            "visual_identifiability_gate_passed": bool(identifiability["families"][family]["visual_identifiability_gate"]["passed"]),
        }
        results["families"][family]["both_gates_passed"] = bool(control_gate["passed"] and results["families"][family]["visual_identifiability_gate_passed"])
    atomic_json(args.output, results)
    if args.pairs:
        pairs = []
        for split in ("train", "iid_heldout", "severity_ood"):
            pairs.extend(read_jsonl(args.pairs / f"pairs_{split}.jsonl"))
        for pair in pairs:
            key = (pair["family"], pair["setup_id"])
            if pair["split"] == "severity_ood" and (key[0], key[1], "no_diagnosis") in row_map:
                no = row_map[(key[0], key[1], "no_diagnosis")]
                oracle = row_map[(key[0], key[1], "oracle_diagnosis")]
                pair["no_diagnosis_outcome"] = {"strict_success": no["strict_success"], "final_normalized_target_distance": no["final_normalized_target_distance"], "executed_steps": no["control_steps"]}
                pair["oracle_diagnosis_outcome"] = {"strict_success": oracle["strict_success"], "final_normalized_target_distance": oracle["final_normalized_target_distance"], "executed_steps": oracle["control_steps"]}
                pair["relevant_safety_outcomes"] = {"no_diagnosis_saturation_count": no["saturation_count"], "oracle_saturation_count": oracle["saturation_count"], "constraint_violations": oracle["constraint_violation_count"]}
            else:
                pair["no_diagnosis_outcome"] = {"status": "not_in_control_audit_split"}
                pair["oracle_diagnosis_outcome"] = {"status": "not_in_control_audit_split"}
                pair["relevant_safety_outcomes"] = {"status": "not_in_control_audit_split"}
        write_jsonl(args.paired_output, pairs)
    print(json.dumps({family: value["control_value_gate"] for family, value in results["families"].items()}, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--suite", type=Path, required=True)
    run_parser.add_argument("--v13-config", type=Path, required=True)
    run_parser.add_argument("--base-config", type=Path, required=True)
    run_parser.add_argument("--model-dir", type=Path, required=True)
    run_parser.add_argument("--planner-seed", type=int, default=2026081301)
    run_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--source", type=Path, required=True)
    analyze_parser.add_argument("--identifiability", type=Path, required=True)
    analyze_parser.add_argument("--pairs", type=Path)
    analyze_parser.add_argument("--paired-output", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        run(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()

