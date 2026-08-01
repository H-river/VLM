#!/usr/bin/env python3
"""Run the development-only hidden-gain Gate A decomposition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.contracts import assert_policy_visible
from active_diagnosis_v13.faults import (
    command_for_desired_physical_delta,
    effective_planning_bounds,
    realize_hidden_gain_step,
)
from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    OUTPUT_FIELDS,
    Bounds,
    action_dict,
    metrics_dict,
    metrics_vector,
    normalized_distance,
    normalized_error,
    position_dict,
    position_vector,
    stable_seed,
    tolerance_vector,
)
from continuous_control_v12.mpc import CEMMPC, learned_predictor
from continuous_control_v12.simulator import simulate_state
from continuous_control_v12.world_model import load_forward_ensemble

VERSION = "active_diagnosis_v13_gate_a_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.resolve().read_text(encoding="utf-8"))
    if config.get("version") != "active_diagnosis_v13_preregistered_v1":
        raise ValueError("unexpected v13 config version")
    baseline = config["baseline"]
    for path_key, hash_key in (
        ("v12_config", "v12_config_sha256"),
        ("base_simulator_config", "base_simulator_config_sha256"),
        ("checkpoint", "checkpoint_sha256"),
        ("training_manifest", "training_manifest_sha256"),
        ("evaluation_suite", "evaluation_suite_sha256"),
    ):
        source = Path(str(baseline[path_key])).resolve()
        actual = _sha256(source)
        if actual != str(baseline[hash_key]):
            raise ValueError(f"locked source hash mismatch: {source}")
    return config


def _case_index(case: Mapping[str, Any]) -> int:
    return int(str(case["case_id"]).rsplit("_", 1)[1])


def _load_frozen_decision(freeze_path: Path | None) -> dict[str, Any]:
    if freeze_path is None or not freeze_path.exists():
        raise ValueError("protected evaluation requires a frozen decision")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not bool(freeze.get("frozen", False)):
        raise ValueError("protected evaluation requires frozen=true")
    if bool(freeze.get("protected_set_used_for_selection", True)):
        raise ValueError("protected evaluation requires a development-only freeze")
    if freeze.get("selected_primary_branch") not in {"P", "A", "B", "C"}:
        raise ValueError("protected evaluation freeze has no valid primary branch")
    return freeze


def _assert_frozen_probe_identity(
    freeze: Mapping[str, Any],
    model_path: Path,
    design: str | None,
    fraction: float | None,
) -> None:
    selected = freeze.get("selected_probe")
    if not isinstance(selected, Mapping):
        raise ValueError("protected probe evaluation requires a frozen probe")
    frozen_model = Path(str(selected["classifier_bundle"])).resolve()
    if model_path.resolve() != frozen_model:
        raise ValueError("protected probe model does not match the frozen decision")
    if _sha256(frozen_model) != str(selected["classifier_bundle_sha256"]):
        raise ValueError("frozen protected probe classifier hash mismatch")
    if design != str(selected["selected_design"]):
        raise ValueError("protected probe design does not match the frozen decision")
    if fraction is None or not np.isclose(
        float(fraction), float(selected["selected_fraction"]), rtol=0.0, atol=1e-12
    ):
        raise ValueError("protected probe fraction does not match the frozen decision")


def _validate_frozen_full_probe_override(
    *,
    enabled: bool,
    mode: str | None,
    split: str,
    probe_model: Path | None,
    gain_predictions: Path | None,
) -> None:
    if enabled and (
        mode != "probe_replan"
        or split != "development"
        or probe_model is None
        or gain_predictions is not None
    ):
        raise ValueError(
            "use-frozen-full-probe-model requires development probe_replan with an "
            "explicit external model and no precomputed predictions"
        )


def _assert_frozen_full_cases_outside_oof_map(
    cases: Sequence[Mapping[str, Any]], classifier_bundle: Mapping[str, Any]
) -> None:
    overlap = sorted(
        {str(case["case_id"]) for case in cases}
        & set(map(str, classifier_bundle["case_to_fold"]))
    )
    if overlap:
        raise ValueError(
            "frozen full probe model override is only for cases outside the "
            f"OOF selection map: {overlap[:3]}"
        )


def _select_cases(
    config: Mapping[str, Any],
    split: str,
    cases_per_stratum: int | None,
    freeze_path: Path | None,
) -> list[dict[str, Any]]:
    if split == "protected":
        _load_frozen_decision(freeze_path)
    suite = json.loads(
        Path(str(config["baseline"]["evaluation_suite"]))
        .resolve()
        .read_text(encoding="utf-8")
    )
    selected = []
    counts: defaultdict[str, int] = defaultdict(int)
    for case in suite["cases"]:
        index = _case_index(case)
        belongs = 0 <= index <= 9 if split == "development" else 10 <= index <= 15
        if not belongs:
            continue
        stratum = str(case["stratum"])
        if cases_per_stratum is not None and counts[stratum] >= cases_per_stratum:
            continue
        selected.append(case)
        counts[stratum] += 1
    expected = 30 if split == "development" else 18
    if cases_per_stratum is None and len(selected) != expected:
        raise ValueError(f"unexpected {split} case count: {len(selected)}")
    if len({str(row["group_id"]) for row in selected}) != len(selected):
        raise ValueError("case groups are not independent")
    return selected


def _load_runtime(config: Mapping[str, Any]) -> tuple[dict[str, Any], Bounds, Any]:
    v12_config = json.loads(
        Path(str(config["baseline"]["v12_config"])).read_text(encoding="utf-8")
    )
    bounds = Bounds.from_config(v12_config)
    bounds.validate()
    model = load_forward_ensemble(
        Path(str(config["baseline"]["checkpoint"])), device_name="cpu"
    )
    return v12_config, bounds, model


def _probe_commands(
    design: str,
    fraction: float,
    bounds: Bounds,
    axis_signs: Sequence[float],
) -> list[np.ndarray]:
    magnitude = bounds.action_high * float(fraction)
    positive = magnitude * np.asarray(axis_signs, dtype=np.float64)
    if design == "single_positive":
        return [positive]
    if design == "single_negative":
        return [-positive]
    if design == "symmetric_pair":
        return [positive, -positive]
    if design == "repeated_positive":
        return [positive, positive]
    raise ValueError(f"unknown probe design: {design}")


def _probe_feature_names(steps: int) -> list[str]:
    names = []
    blocks = (
        "observed_delta",
        "predicted_delta",
        "prediction_residual",
        "uncertainty_normalized_residual",
        "ensemble_uncertainty",
    )
    for step in range(steps):
        for block in blocks:
            names.extend(
                f"step_{step + 1}.{block}.{field}" for field in OUTPUT_FIELDS
            )
        names.append(f"step_{step + 1}.gain_ratio_projection")
    if steps == 2:
        for field in OUTPUT_FIELDS:
            names.append(f"repeat_consistency.{field}")
    return names


def _execute_probe(
    *,
    case: Mapping[str, Any],
    true_gain: float,
    design: str,
    fraction: float,
    config: Mapping[str, Any],
    bounds: Bounds,
    model: Any,
) -> dict[str, Any]:
    commands = _probe_commands(
        design,
        fraction,
        bounds,
        config["probes"]["axis_signs"],
    )
    true_position = position_vector(case["initial_positions_mm"])
    commanded_belief = true_position.copy()
    initial_metrics = metrics_vector(case["initial_metrics"])
    metrics = initial_metrics.copy()
    target = metrics_vector(case["target_metrics"])
    floor = float(config["probes"]["measurement_uncertainty_floor_normalized"])
    feature: list[float] = []
    step_records = []
    accepted_commands = []
    deltas = []
    saturation_count = 0
    started = time.perf_counter()
    for step_index, command in enumerate(commands, start=1):
        visible_input = {
            "setup_context": case["setup_context"],
            "commanded_position_belief": position_dict(commanded_belief),
            "observed_metrics": metrics_dict(metrics),
            "command": action_dict(command),
        }
        assert_policy_visible(visible_input)
        prediction = model.predict(
            case["setup_context"], commanded_belief, metrics, command
        )
        predicted = prediction["predicted_next_metrics"][0]
        uncertainty = np.asarray(prediction["uncertainty"][0], dtype=np.float64)
        gain_step = realize_hidden_gain_step(
            true_position=true_position,
            commanded_position_belief=commanded_belief,
            requested_command=command,
            true_gain=true_gain,
            bounds=bounds,
        )
        capture = simulate_state(
            case["setup_context"],
            position_dict(gain_step.next_true_position),
            case["simulator_fixed"],
            str(config["baseline"]["base_simulator_config"]),
            bounds,
        )
        observed = metrics_vector(capture["metrics"])
        tolerance = tolerance_vector(metrics)
        observed_delta = (observed - metrics) / tolerance
        predicted_delta = (predicted - metrics) / tolerance
        residual = (observed - predicted) / tolerance
        total_uncertainty = np.sqrt(np.square(uncertainty) + floor**2)
        standardized = residual / np.maximum(total_uncertainty, 1e-9)
        denominator = float(np.dot(predicted_delta, predicted_delta))
        ratio = float(
            np.dot(observed_delta, predicted_delta) / max(denominator, 1e-12)
        )
        feature.extend(observed_delta.tolist())
        feature.extend(predicted_delta.tolist())
        feature.extend(residual.tolist())
        feature.extend(standardized.tolist())
        feature.extend(uncertainty.tolist())
        feature.append(ratio)
        deltas.append(observed_delta)
        accepted_commands.append(gain_step.accepted_command.tolist())
        saturation_count += int(
            gain_step.step_saturated or gain_step.absolute_position_saturated
        )
        step_records.append(
            {
                "step": step_index,
                "visible_input": visible_input,
                "predicted_metrics": metrics_dict(predicted),
                "observed_metrics": metrics_dict(observed),
                "observed_signed_delta_normalized": dict(
                    zip(OUTPUT_FIELDS, map(float, observed_delta), strict=True)
                ),
                "h1_prediction_residual_normalized": dict(
                    zip(OUTPUT_FIELDS, map(float, residual), strict=True)
                ),
                "ensemble_plus_measurement_uncertainty": dict(
                    zip(OUTPUT_FIELDS, map(float, total_uncertainty), strict=True)
                ),
                "residual_signal_to_noise": float(
                    np.linalg.norm(residual)
                    / max(np.linalg.norm(total_uncertainty), 1e-12)
                ),
                "gain_step_audit": gain_step.audit_dict(),
                "simulator_valid": bool(capture["auxiliary"]["simulator_valid"]),
                "clipping_fraction": float(
                    capture["auxiliary"]["clipping_fraction"] or 0.0
                ),
                "camera_boundary_indicator": bool(
                    capture["auxiliary"]["camera_boundary_indicator"]
                ),
            }
        )
        true_position = gain_step.next_true_position
        commanded_belief = gain_step.next_commanded_position_belief
        metrics = observed
    if len(deltas) == 2:
        feature.extend((deltas[1] - deltas[0]).tolist())
    initial_target_cost = normalized_distance(initial_metrics, target, initial_metrics)
    final_target_cost = normalized_distance(metrics, target, initial_metrics)
    state_disturbance = normalized_distance(metrics, initial_metrics, initial_metrics)
    policy_record = {
        "case_id": case["case_id"],
        "group_id": case["group_id"],
        "design": design,
        "fraction": float(fraction),
        "feature_names": _probe_feature_names(len(commands)),
        "feature_vector": list(map(float, feature)),
    }
    assert_policy_visible(policy_record)
    return {
        "version": VERSION,
        "record_kind": "probe",
        "record_id": f"{case['case_id']}__g{true_gain:g}__{design}__f{fraction:g}",
        "case_id": case["case_id"],
        "group_id": case["group_id"],
        "stratum": case["stratum"],
        "regime": case["regime"],
        "evaluator_only_true_gain": float(true_gain),
        "design": design,
        "fraction": float(fraction),
        "probe_steps": len(commands),
        "policy_record": policy_record,
        "accepted_commands": accepted_commands,
        "initial_target_cost": float(initial_target_cost),
        "final_target_cost_after_probe": float(final_target_cost),
        "signed_target_cost_disturbance": float(
            final_target_cost - initial_target_cost
        ),
        "absolute_beam_state_disturbance": float(state_disturbance),
        "cumulative_realized_motion_l1_mm": float(
            sum(
                np.abs(
                    [row["gain_step_audit"]["realized_delta_mm"][field] for field in ACTION_FIELDS]
                ).sum()
                for row in step_records
            )
        ),
        "saturation_count": int(saturation_count),
        "constraint_violation_count": 0,
        "all_simulations_valid": all(row["simulator_valid"] for row in step_records),
        "step_records": step_records,
        "final_observed_metrics": metrics_dict(metrics),
        "final_commanded_position_belief_mm": position_dict(commanded_belief),
        "evaluator_only_final_true_position_mm": position_dict(true_position),
        "wall_runtime_seconds": float(time.perf_counter() - started),
    }


def run_probes(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    _, bounds, model = _load_runtime(config)
    cases = _select_cases(
        config, args.split, args.cases_per_stratum, args.freeze
    )
    output = args.output_dir / "probes" / (
        "records.jsonl"
        if args.shard_count == 1
        else f"records_shard_{args.shard_index:02d}_of_{args.shard_count:02d}.jsonl"
    )
    completed = {row["record_id"] for row in _read_jsonl(output)}
    designs = args.design or list(config["probes"]["designs"])
    fractions = args.fraction or list(config["probes"]["safe_range_fractions"])
    gains = list(map(float, config["fault"]["gain_hypotheses"]))
    tasks = len(cases) * len(designs) * len(fractions) * len(gains)
    task = 0
    for case in cases:
        for gain in gains:
            for design in designs:
                for fraction in fractions:
                    task += 1
                    if (task - 1) % args.shard_count != args.shard_index:
                        continue
                    record_id = (
                        f"{case['case_id']}__g{gain:g}__{design}__f{float(fraction):g}"
                    )
                    if record_id in completed:
                        continue
                    record = _execute_probe(
                        case=case,
                        true_gain=gain,
                        design=design,
                        fraction=float(fraction),
                        config=config,
                        bounds=bounds,
                        model=model,
                    )
                    _append_jsonl(output, record)
                    print(
                        json.dumps(
                            {
                                "event": "probe_complete",
                                "task": task,
                                "tasks": tasks,
                                "record_id": record_id,
                                "runtime_seconds": record["wall_runtime_seconds"],
                                "disturbance": record["absolute_beam_state_disturbance"],
                            }
                        ),
                        flush=True,
                    )


def _probe_summary(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    grouped: defaultdict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["design"]), float(row["fraction"]))].append(row)
    summaries = []
    bundles: dict[tuple[str, float], Any] = {}
    order = list(config["probes"]["probe_selection_order_tiebreak"])
    for (design, fraction), selected in sorted(grouped.items()):
        X = np.asarray(
            [row["policy_record"]["feature_vector"] for row in selected],
            dtype=np.float64,
        )
        y = np.asarray(
            [f"{float(row['evaluator_only_true_gain']):g}" for row in selected],
            dtype=np.str_,
        )
        groups = np.asarray([str(row["group_id"]) for row in selected])
        unique_groups = sorted(set(groups.tolist()))
        folds = min(5, len(unique_groups))
        if folds < 2:
            raise ValueError("probe analysis needs at least two setup groups")
        splitter = GroupKFold(n_splits=folds)
        predictions = np.empty_like(y)
        fold_models = []
        case_to_fold: dict[str, int] = {}
        for fold, (train, validation) in enumerate(splitter.split(X, y, groups)):
            pipeline = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=3000,
                            random_state=int(config["root_seed"]),
                        ),
                    ),
                ]
            )
            pipeline.fit(X[train], y[train])
            predictions[validation] = pipeline.predict(X[validation])
            fold_models.append(pipeline)
            for index in validation:
                case_to_fold[str(selected[index]["case_id"])] = fold
        full_model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=int(config["root_seed"]),
                    ),
                ),
            ]
        ).fit(X, y)
        accuracy = float(np.mean(predictions == y))
        feature_class_means = np.stack(
            [X[y == label].mean(axis=0) for label in sorted(set(y.tolist()))]
        )
        between = float(np.mean(np.var(feature_class_means, axis=0)))
        within = float(
            np.mean(
                [np.mean(np.var(X[y == label], axis=0)) for label in sorted(set(y.tolist()))]
            )
        )
        per_gain = {
            str(gain): float(np.mean(predictions[y == gain] == y[y == gain]))
            for gain in sorted(set(y.tolist()), key=float)
        }
        bundle = {
            "version": VERSION,
            "design": design,
            "fraction": fraction,
            "feature_names": selected[0]["policy_record"]["feature_names"],
            "fold_models": fold_models,
            "case_to_fold": case_to_fold,
            "full_model": full_model,
        }
        model_dir = output_dir / "probes" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        fraction_label = f"{fraction:g}".replace(".", "p")
        cell_model_path = model_dir / f"{design}__f{fraction_label}.joblib"
        joblib.dump(bundle, cell_model_path)
        summary = {
            "design": design,
            "fraction": fraction,
            "groups": len(unique_groups),
            "records": len(selected),
            "gain_classification_accuracy": accuracy,
            "per_gain_accuracy": per_gain,
            "residual_signal_to_noise_ratio": float(
                np.sqrt(between / max(within, 1e-12))
            ),
            "mean_absolute_beam_state_disturbance": float(
                np.mean([abs(float(row["absolute_beam_state_disturbance"])) for row in selected])
            ),
            "mean_signed_target_cost_disturbance": float(
                np.mean([float(row["signed_target_cost_disturbance"]) for row in selected])
            ),
            "saturation_rate": float(
                np.mean([int(row["saturation_count"]) > 0 for row in selected])
            ),
            "constraint_violations": int(
                sum(int(row["constraint_violation_count"]) for row in selected)
            ),
            "additional_probe_steps": int(selected[0]["probe_steps"]),
            "final_success_after_estimation_and_replanning": None,
            "classifier_bundle": str(cell_model_path.resolve()),
            "classifier_bundle_sha256": _sha256(cell_model_path),
            "oof_predictions": [
                {
                    "case_id": str(row["case_id"]),
                    "evaluator_only_true_gain": float(label),
                    "estimated_gain": float(prediction),
                }
                for row, label, prediction in zip(selected, y, predictions, strict=True)
            ],
        }
        summaries.append(summary)
        bundles[(design, fraction)] = bundle
    ranked = sorted(
        summaries,
        key=lambda row: (
            -float(row["gain_classification_accuracy"]),
            float(row["mean_absolute_beam_state_disturbance"]),
            int(row["additional_probe_steps"]),
            order.index(str(row["design"])),
            float(row["fraction"]),
        ),
    )
    best = ranked[0]
    model_path = output_dir / "probes" / "selected_probe_model.joblib"
    joblib.dump(bundles[(str(best["design"]), float(best["fraction"]))], model_path)
    selection = {
        "version": VERSION,
        "selection_split": "development_only",
        "selected_design": best["design"],
        "selected_fraction": best["fraction"],
        "development_gain_classification_accuracy": best[
            "gain_classification_accuracy"
        ],
        "selection_rule": "accuracy_then_disturbance_then_steps_then_preregistered_order",
        "classifier_bundle": str(model_path.resolve()),
        "classifier_bundle_sha256": _sha256(model_path),
        "protected_set_used": False,
    }
    return {"version": VERSION, "designs": summaries, "selected": selection}


def analyze_probes(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    paths = sorted((args.output_dir / "probes").glob("records*.jsonl"))
    rows_by_id = {
        row["record_id"]: row
        for path in paths
        for row in _read_jsonl(path)
    }
    rows = list(rows_by_id.values())
    if not rows:
        raise ValueError("no probe records")
    summary = _probe_summary(rows, config, args.output_dir)
    _atomic_json(args.output_dir / "probes" / "probe_summary.json", summary)
    _atomic_json(args.output_dir / "probes" / "selected_probe.json", summary["selected"])
    print(json.dumps(summary["selected"], sort_keys=True), flush=True)


def _planner_config(config: Mapping[str, Any]) -> dict[str, Any]:
    baseline = config["baseline"]
    planner = {
        "horizon": 1,
        "population": int(baseline["population"]),
        "elites": int(baseline["elites"]),
        "cem_iterations": int(baseline["cem_iterations"]),
        "mean_error_weight": float(baseline["mean_error_weight"]),
        "movement_weight": float(baseline["movement_weight"]),
        "limit_penalty": float(baseline["limit_penalty"]),
        "boundary_penalty": float(baseline["boundary_penalty"]),
        "uncertainty_weight": float(baseline["uncertainty_weight"]),
        "candidate_audit_top_k": 0,
        "candidate_audit_reference_k": 0,
    }
    if "elite_min_normalized_distance" in baseline:
        planner["elite_min_normalized_distance"] = float(
            baseline["elite_min_normalized_distance"]
        )
    if "feasible_proposal_resample_attempts" in baseline:
        planner["feasible_proposal_resample_attempts"] = int(
            baseline["feasible_proposal_resample_attempts"]
        )
    return planner


def matched_planner_seed(config: Mapping[str, Any], case_id: str) -> int:
    """Return a candidate-sampling seed independent of hidden conditions."""

    return stable_seed(config["root_seed"], case_id, "matched_h1")


def _replay_estimated_position(
    initial: np.ndarray,
    commands: Sequence[Sequence[float]],
    gain_belief: float,
    bounds: Bounds,
) -> np.ndarray:
    estimated_true = initial.copy()
    command_belief = initial.copy()
    for command in commands:
        step = realize_hidden_gain_step(
            true_position=estimated_true,
            commanded_position_belief=command_belief,
            requested_command=command,
            true_gain=gain_belief,
            bounds=bounds,
        )
        estimated_true = step.next_true_position
        command_belief = step.next_commanded_position_belief
    return estimated_true


def _execute_control_episode(
    *,
    case: Mapping[str, Any],
    true_gain: float,
    mode: str,
    config: Mapping[str, Any],
    bounds: Bounds,
    model: Any,
    probe_selection: Mapping[str, Any] | None,
    classifier_bundle: Any | None,
    split: str,
    policy_name: str,
    precomputed_gain_prediction: Mapping[str, Any] | None = None,
    confidence_threshold: float | None = None,
    use_frozen_full_probe_model: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    initial_position = position_vector(case["initial_positions_mm"])
    true_position = initial_position.copy()
    command_position = initial_position.copy()
    estimated_position = initial_position.copy()
    metrics = metrics_vector(case["initial_metrics"])
    initial_metrics = metrics.copy()
    target = metrics_vector(case["target_metrics"])
    probe_steps = 0
    probe_record = None
    gain_belief: float | None = None
    raw_gain_estimate: float | None = None
    gain_estimate_confidence: float | None = None
    confidence_fallback_to_nominal = False
    gain_prediction_source: str | None = None
    probe_commands: list[list[float]] = []
    saturation_count = 0
    if mode == "oracle_known":
        gain_belief = float(true_gain)
    elif mode == "probe_replan":
        if probe_selection is None or (
            classifier_bundle is None and precomputed_gain_prediction is None
        ):
            raise ValueError("probe_replan requires a classifier or precomputed prediction")
        probe_record = _execute_probe(
            case=case,
            true_gain=true_gain,
            design=str(probe_selection["selected_design"]),
            fraction=float(probe_selection["selected_fraction"]),
            config=config,
            bounds=bounds,
            model=model,
        )
        probe_steps = int(probe_record["probe_steps"])
        probe_commands = list(probe_record["accepted_commands"])
        features = np.asarray(
            probe_record["policy_record"]["feature_vector"], dtype=np.float64
        )[None, :]
        if precomputed_gain_prediction is not None:
            from active_diagnosis_v13.build_decision_dataset import prompt_for_record

            expected_prompt = prompt_for_record(
                probe_record,
                int(precomputed_gain_prediction["serialization_seed"]),
                precomputed_gain_prediction.get("retained_feature_indices"),
            )
            expected_hash = hashlib.sha256(expected_prompt.encode()).hexdigest()
            if expected_hash != str(
                precomputed_gain_prediction["visible_prompt_sha256"]
            ):
                raise ValueError("precomputed prediction visible-prompt hash mismatch")
            estimated = precomputed_gain_prediction.get("estimated_gain")
            valid = bool(precomputed_gain_prediction.get("valid_gain_prediction"))
            gain_belief = (
                float(estimated)
                if valid
                else float(config["fault"]["nominal_gain"])
            )
            raw_gain_estimate = None if estimated is None else float(estimated)
            confidence_fallback_to_nominal = not valid
            gain_prediction_source = "precomputed_visible_prompt_with_nominal_parse_fallback"
        elif split == "development" and not use_frozen_full_probe_model:
            fold = classifier_bundle["case_to_fold"][str(case["case_id"])]
            classifier = classifier_bundle["fold_models"][fold]
            raw_gain_estimate = float(classifier.predict(features)[0])
            gain_estimate_confidence = float(np.max(classifier.predict_proba(features)[0]))
            confidence_fallback_to_nominal = bool(
                confidence_threshold is not None
                and gain_estimate_confidence < confidence_threshold
            )
            gain_belief = (
                float(config["fault"]["nominal_gain"])
                if confidence_fallback_to_nominal
                else raw_gain_estimate
            )
            gain_prediction_source = "group_out_of_fold_classifier"
        else:
            classifier = classifier_bundle["full_model"]
            raw_gain_estimate = float(classifier.predict(features)[0])
            gain_estimate_confidence = float(np.max(classifier.predict_proba(features)[0]))
            confidence_fallback_to_nominal = bool(
                confidence_threshold is not None
                and gain_estimate_confidence < confidence_threshold
            )
            gain_belief = (
                float(config["fault"]["nominal_gain"])
                if confidence_fallback_to_nominal
                else raw_gain_estimate
            )
            gain_prediction_source = (
                "frozen_full_classifier"
                if split == "protected"
                else "frozen_full_classifier_fresh_nonprotected"
            )
        true_position = position_vector(
            probe_record["evaluator_only_final_true_position_mm"]
        )
        command_position = position_vector(
            probe_record["final_commanded_position_belief_mm"]
        )
        estimated_position = _replay_estimated_position(
            initial_position, probe_commands, gain_belief, bounds
        )
        metrics = metrics_vector(probe_record["final_observed_metrics"])
        saturation_count += int(probe_record["saturation_count"])
    elif mode != "direct":
        raise ValueError(f"unknown control mode: {mode}")
    planner_bounds = (
        bounds
        if gain_belief is None
        else effective_planning_bounds(bounds, gain_belief)
    )
    planner = CEMMPC(
        bounds=planner_bounds,
        predictor=learned_predictor(model, case["setup_context"]),
        config=_planner_config(config),
        # Candidate sampling is matched by case and must not depend on the
        # evaluator-only hidden gain or on which controller is being scored.
        seed=matched_planner_seed(config, str(case["case_id"])),
    )
    trace = []
    for control_step in range(int(config["baseline"]["max_control_steps"])):
        before = normalized_distance(metrics, target, initial_metrics)
        if before <= 1.0:
            break
        planning_position = command_position if gain_belief is None else estimated_position
        visible_plan_input = {
            "setup_context": case["setup_context"],
            "position_belief": position_dict(planning_position),
            "observed_metrics": metrics_dict(metrics),
            "target_metrics": metrics_dict(target),
            "gain_belief": gain_belief,
        }
        assert_policy_visible(visible_plan_input)
        plan = planner.plan(
            positions_mm=planning_position,
            current_metrics=metrics,
            target_metrics=target,
            allowed_dofs=["lens_x", "lens_y", "camera_x", "camera_y"],
            tolerance_reference=initial_metrics,
        )
        desired_or_command = np.asarray(
            [plan["selected_action"][field] for field in ACTION_FIELDS],
            dtype=np.float64,
        )
        command = (
            desired_or_command
            if gain_belief is None
            else command_for_desired_physical_delta(
                desired_or_command, gain_belief, command_position, bounds
            )
        )
        actual_step = realize_hidden_gain_step(
            true_position=true_position,
            commanded_position_belief=command_position,
            requested_command=command,
            true_gain=true_gain,
            bounds=bounds,
        )
        if gain_belief is not None:
            estimated_step = realize_hidden_gain_step(
                true_position=estimated_position,
                commanded_position_belief=command_position,
                requested_command=command,
                true_gain=gain_belief,
                bounds=bounds,
            )
            estimated_position = estimated_step.next_true_position
        capture = simulate_state(
            case["setup_context"],
            position_dict(actual_step.next_true_position),
            case["simulator_fixed"],
            str(config["baseline"]["base_simulator_config"]),
            bounds,
        )
        observed = metrics_vector(capture["metrics"])
        after = normalized_distance(observed, target, initial_metrics)
        saturation_count += int(
            actual_step.step_saturated or actual_step.absolute_position_saturated
        )
        trace.append(
            {
                "control_step": control_step + 1,
                "visible_plan_input": visible_plan_input,
                "command_mm": action_dict(command),
                "predicted_next_metrics": plan["predicted_next_metrics"],
                "observed_next_metrics": metrics_dict(observed),
                "before_target_cost": float(before),
                "predicted_target_cost": float(
                    plan["predicted_normalized_target_distance"]
                ),
                "actual_target_cost": float(after),
                "actual_step_audit": actual_step.audit_dict(),
                **(
                    {
                        "planner_diagnostics": {
                            "elite_min_normalized_distance": float(
                                config["baseline"].get(
                                    "elite_min_normalized_distance", 0.0
                                )
                            ),
                            "feasible_proposal_resample_attempts": int(
                                config["baseline"].get(
                                    "feasible_proposal_resample_attempts", 0
                                )
                            ),
                            "iteration_history": plan["iteration_history"],
                        }
                    }
                    if "elite_min_normalized_distance" in config["baseline"]
                    or "feasible_proposal_resample_attempts" in config["baseline"]
                    else {}
                ),
            }
        )
        true_position = actual_step.next_true_position
        command_position = actual_step.next_commanded_position_belief
        metrics = observed
    final_distance = normalized_distance(metrics, target, initial_metrics)
    return {
        "version": VERSION,
        "record_kind": "control_episode",
        "record_id": f"{case['case_id']}__g{true_gain:g}__{policy_name}",
        "case_id": case["case_id"],
        "group_id": case["group_id"],
        "stratum": case["stratum"],
        "regime": case["regime"],
        "mode": mode,
        "policy_name": policy_name,
        "planner_root_seed": int(config["root_seed"]),
        "planner_config": _planner_config(config),
        "evaluator_only_true_gain": float(true_gain),
        "gain_belief": gain_belief,
        "raw_gain_estimate": raw_gain_estimate,
        "gain_estimate_confidence": gain_estimate_confidence,
        "confidence_threshold": confidence_threshold,
        "confidence_fallback_to_nominal": confidence_fallback_to_nominal,
        "gain_prediction_source": gain_prediction_source,
        "gain_classification_correct": (
            None if mode != "probe_replan" else bool(gain_belief == true_gain)
        ),
        "probe_design": None if probe_selection is None else probe_selection.get("selected_design"),
        "probe_fraction": None if probe_selection is None else probe_selection.get("selected_fraction"),
        "probe_steps": probe_steps,
        "control_steps": len(trace),
        "total_additional_steps": probe_steps + len(trace),
        "initial_normalized_distance": float(
            normalized_distance(initial_metrics, target, initial_metrics)
        ),
        "final_normalized_distance": float(final_distance),
        "strict_success": bool(final_distance <= 1.0),
        "saturation_count": int(saturation_count),
        "constraint_violation_count": 0,
        "trace": trace,
        "probe_record": probe_record,
        "final_metrics": metrics_dict(metrics),
        "final_commanded_position_belief_mm": position_dict(command_position),
        "evaluator_only_final_true_position_mm": position_dict(true_position),
        "wall_runtime_seconds": float(time.perf_counter() - started),
    }


def run_control(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    import joblib

    if args.confidence_threshold is not None:
        if args.mode != "probe_replan" or not 0.0 <= args.confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be in [0,1] for probe_replan")
    _validate_frozen_full_probe_override(
        enabled=args.use_frozen_full_probe_model,
        mode=args.mode,
        split=args.split,
        probe_model=args.probe_model,
        gain_predictions=args.gain_predictions,
    )
    _, bounds, model = _load_runtime(config)
    cases = _select_cases(config, args.split, args.cases_per_stratum, args.freeze)
    precomputed_by_key: dict[tuple[str, float], Mapping[str, Any]] = {}
    if args.gain_predictions is not None:
        if args.mode != "probe_replan" or args.split != "development":
            raise ValueError(
                "precomputed gain predictions are only valid for development probe_replan"
            )
        for row in _read_jsonl(args.gain_predictions.resolve()):
            if bool(row.get("protected_set_used", False)):
                raise ValueError("protected prediction cannot enter development control")
            key = (str(row["case_id"]), float(row["true_gain_evaluator_only"]))
            if key in precomputed_by_key:
                raise ValueError(f"duplicate precomputed gain prediction: {key}")
            precomputed_by_key[key] = row
        case_ids = {case_id for case_id, _ in precomputed_by_key}
        cases = [case for case in cases if str(case["case_id"]) in case_ids]
        expected = {
            (str(case["case_id"]), gain)
            for case in cases
            for gain in map(float, config["fault"]["gain_hypotheses"])
        }
        if set(precomputed_by_key) != expected:
            raise ValueError("precomputed predictions do not form complete case-by-gain groups")
    probe_selection = None
    classifier_bundle = None
    policy_name = args.output_name or args.mode
    if "/" in policy_name or ".." in policy_name:
        raise ValueError("output name must be a simple filename stem")
    if args.mode == "probe_replan":
        if args.probe_model is not None:
            if args.probe_design is None or args.probe_fraction is None:
                raise ValueError(
                    "external probe model requires --probe-design and --probe-fraction"
                )
            model_path = args.probe_model.resolve()
            if args.split == "protected":
                _assert_frozen_probe_identity(
                    _load_frozen_decision(args.freeze),
                    model_path,
                    args.probe_design,
                    args.probe_fraction,
                )
            probe_selection = {
                "selected_design": args.probe_design,
                "selected_fraction": float(args.probe_fraction),
                "classifier_bundle": str(model_path),
                "classifier_bundle_sha256": _sha256(model_path),
            }
        else:
            if args.split == "protected":
                raise ValueError(
                    "protected probe evaluation requires the explicit frozen model"
                )
            selection_path = args.output_dir / "probes" / "selected_probe.json"
            probe_selection = json.loads(selection_path.read_text(encoding="utf-8"))
            model_path = Path(str(probe_selection["classifier_bundle"]))
            if _sha256(model_path) != str(probe_selection["classifier_bundle_sha256"]):
                raise ValueError("probe classifier hash mismatch")
        if args.gain_predictions is None:
            classifier_bundle = joblib.load(model_path)
            if args.use_frozen_full_probe_model:
                _assert_frozen_full_cases_outside_oof_map(cases, classifier_bundle)
    output = args.output_dir / "control" / f"{policy_name}.jsonl"
    completed = {row["record_id"] for row in _read_jsonl(output)}
    gains = list(map(float, config["fault"]["gain_hypotheses"]))
    tasks = len(cases) * len(gains)
    task = 0
    for case in cases:
        for gain in gains:
            task += 1
            record_id = f"{case['case_id']}__g{gain:g}__{policy_name}"
            if record_id in completed:
                continue
            result = _execute_control_episode(
                case=case,
                true_gain=gain,
                mode=args.mode,
                config=config,
                bounds=bounds,
                model=model,
                probe_selection=probe_selection,
                classifier_bundle=classifier_bundle,
                split=args.split,
                policy_name=policy_name,
                precomputed_gain_prediction=precomputed_by_key.get(
                    (str(case["case_id"]), float(gain))
                ),
                confidence_threshold=args.confidence_threshold,
                use_frozen_full_probe_model=args.use_frozen_full_probe_model,
            )
            _append_jsonl(output, result)
            print(
                json.dumps(
                    {
                        "event": "control_complete",
                        "task": task,
                        "tasks": tasks,
                        "record_id": record_id,
                        "success": result["strict_success"],
                        "final_distance": result["final_normalized_distance"],
                        "runtime_seconds": result["wall_runtime_seconds"],
                    }
                ),
                flush=True,
            )


def _group_bootstrap_difference(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    seed: int,
    samples: int = 4000,
) -> dict[str, float]:
    def by_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        grouped: defaultdict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[str(row["group_id"])].append(float(row["strict_success"]))
        return {key: float(np.mean(value)) for key, value in grouped.items()}

    left_g, right_g = by_group(left), by_group(right)
    groups = sorted(set(left_g) & set(right_g))
    values = np.asarray([right_g[group] - left_g[group] for group in groups])
    rng = np.random.default_rng(seed)
    estimates = [float(np.mean(values[rng.integers(0, len(values), len(values))])) for _ in range(samples)]
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"estimate": float(values.mean()), "low": float(low), "high": float(high)}


def select_gate_branch(A: bool, B: bool, C: bool, D: bool) -> tuple[str, str]:
    """Return the single preregistered branch for the four Gate-A checks."""

    if A and B and C and D:
        return "P", "active diagnosis passed all four preregistered Gate A checks"
    if A and B and not C:
        return (
            "A",
            "fault is impactful and recoverable but current probes are not observable enough",
        )
    if A and (not D or not B):
        return (
            "B",
            "active diagnosis has insufficient recovery/control value; test execute-only reranking",
        )
    return (
        "C",
        "hidden gain does not create sufficient actionable headroom; isolate the H1/CEM bottleneck",
    )


def aggregate_gate_a(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    probe_summary_path = args.output_dir / "probes" / "probe_summary.json"
    probe_summary = json.loads(probe_summary_path.read_text(encoding="utf-8"))
    modes = {
        mode: _read_jsonl(args.output_dir / "control" / f"{mode}.jsonl")
        for mode in ("direct", "oracle_known", "probe_replan")
    }
    if any(not rows for rows in modes.values()):
        raise ValueError("all three control modes must complete before Gate A")
    nominal = float(config["fault"]["nominal_gain"])
    direct_nominal = [row for row in modes["direct"] if float(row["evaluator_only_true_gain"]) == nominal]
    direct_fault = [row for row in modes["direct"] if float(row["evaluator_only_true_gain"]) != nominal]
    oracle_fault = [row for row in modes["oracle_known"] if float(row["evaluator_only_true_gain"]) != nominal]
    probe_fault = [row for row in modes["probe_replan"] if float(row["evaluator_only_true_gain"]) != nominal]
    rate = lambda rows: float(np.mean([bool(row["strict_success"]) for row in rows]))
    nominal_rate = rate(direct_nominal)
    direct_fault_rate = rate(direct_fault)
    oracle_fault_rate = rate(oracle_fault)
    probe_fault_rate = rate(probe_fault)
    impact = nominal_rate - direct_fault_rate
    recovery = oracle_fault_rate - direct_fault_rate
    control_value = probe_fault_rate - direct_fault_rate
    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        by_gain: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        by_stratum: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_gain[f"{float(row['evaluator_only_true_gain']):g}"].append(row)
            by_stratum[str(row["stratum"])].append(row)
        block = lambda values: {
            "episodes": len(values),
            "strict_success": rate(values),
            "mean_final_distance": float(
                np.mean([float(row["final_normalized_distance"]) for row in values])
            ),
            "mean_total_additional_steps": float(
                np.mean([int(row["total_additional_steps"]) for row in values])
            ),
            "saturation_episode_rate": float(
                np.mean([int(row["saturation_count"]) > 0 for row in values])
            ),
        }
        return {
            "overall": block(rows),
            "by_gain": {key: block(value) for key, value in sorted(by_gain.items(), key=lambda item: float(item[0]))},
            "by_stratum": {key: block(value) for key, value in sorted(by_stratum.items())},
        }
    direct_by_key = {
        (str(row["case_id"]), float(row["evaluator_only_true_gain"])): row
        for row in direct_fault
    }
    oracle_by_key = {
        (str(row["case_id"]), float(row["evaluator_only_true_gain"])): row
        for row in oracle_fault
    }
    probe_by_key = {
        (str(row["case_id"]), float(row["evaluator_only_true_gain"])): row
        for row in probe_fault
    }
    common = sorted(set(direct_by_key) & set(oracle_by_key) & set(probe_by_key))
    oracle_recoveries = sum(
        not bool(direct_by_key[key]["strict_success"])
        and bool(oracle_by_key[key]["strict_success"])
        for key in common
    )
    probe_recoveries = sum(
        not bool(direct_by_key[key]["strict_success"])
        and bool(probe_by_key[key]["strict_success"])
        for key in common
    )
    selected = probe_summary["selected"]
    accuracy = float(selected["development_gain_classification_accuracy"])
    thresholds = config["gate_a"]
    A = impact >= float(thresholds["meaningful_fault_impact_success_drop"])
    B = recovery >= float(thresholds["meaningful_oracle_recovery_success_gain"])
    C = accuracy >= float(thresholds["minimum_probe_gain_classification_accuracy"])
    D = control_value >= float(thresholds["minimum_probe_replan_success_gain"])
    branch, reason = select_gate_branch(A, B, C, D)
    diagnosis = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "fault_impact": {
            "question": "Does hidden gain drift materially reduce direct H1-CEM success?",
            "nominal_direct_success": nominal_rate,
            "fault_direct_success": direct_fault_rate,
            "success_drop": impact,
            "matched_group_bootstrap_95": _group_bootstrap_difference(
                direct_fault, direct_nominal, int(config["root_seed"])
            ),
            "passes": A,
        },
        "recoverability": {
            "question": "Does oracle-known-gain planning recover the lost success?",
            "oracle_known_gain_success": oracle_fault_rate,
            "success_gain_over_direct_fault": recovery,
            "matched_fault_episode_recoveries": int(oracle_recoveries),
            "matched_group_bootstrap_95": _group_bootstrap_difference(
                direct_fault, oracle_fault, int(config["root_seed"]) + 1
            ),
            "passes": B,
        },
        "observability": {
            "question": "Can safe probe observations distinguish the gain hypotheses?",
            "selected_probe": selected,
            "classification_accuracy": accuracy,
            "passes": C,
        },
        "control_value": {
            "question": "Does probe plus estimation plus replanning improve final strict success?",
            "probe_replan_success": probe_fault_rate,
            "success_gain_over_direct_fault": control_value,
            "matched_fault_episode_recoveries": int(probe_recoveries),
            "matched_group_bootstrap_95": _group_bootstrap_difference(
                direct_fault, probe_fault, int(config["root_seed"]) + 2
            ),
            "mean_total_additional_steps": float(
                np.mean([int(row["total_additional_steps"]) for row in probe_fault])
            ),
            "passes": D,
        },
        "selected_primary_branch": branch,
        "branch_reason": reason,
        "exactly_one_primary_branch_selected": True,
        "control_summaries": {
            mode: summarize(rows) for mode, rows in modes.items()
        },
    }
    _atomic_json(args.output_dir / "gate_a_diagnosis.json", diagnosis)
    freeze = {
        "version": VERSION,
        "frozen": True,
        "selected_primary_branch": branch,
        "branch_reason": reason,
        "selected_probe": selected,
        "config_sha256": _sha256(args.config.resolve()),
        "gate_a_diagnosis_sha256": _sha256(args.output_dir / "gate_a_diagnosis.json"),
        "protected_set_used_for_selection": False,
    }
    _atomic_json(args.output_dir / "frozen_decision.json", freeze)
    print(json.dumps(diagnosis, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("probes", "analyze-probes", "control", "aggregate"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_v13.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "protected"), default="development")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--cases-per-stratum", type=int)
    parser.add_argument("--design", action="append")
    parser.add_argument("--fraction", action="append", type=float)
    parser.add_argument("--mode", choices=("direct", "oracle_known", "probe_replan"))
    parser.add_argument("--probe-model", type=Path)
    parser.add_argument("--probe-design")
    parser.add_argument("--probe-fraction", type=float)
    parser.add_argument("--use-frozen-full-probe-model", action="store_true")
    parser.add_argument("--output-name")
    parser.add_argument("--root-seed-override", type=int)
    parser.add_argument("--gain-predictions", type=Path)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--population-override", type=int)
    parser.add_argument("--cem-iterations-override", type=int)
    parser.add_argument("--max-control-steps-override", type=int)
    parser.add_argument("--uncertainty-weight-override", type=float)
    parser.add_argument("--elite-min-distance-override", type=float)
    parser.add_argument("--feasible-proposal-resample-attempts-override", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard count)")
    args.output_dir = args.output_dir.resolve()
    config = _load_config(args.config)
    if args.root_seed_override is not None:
        if args.phase != "control":
            raise ValueError("root-seed-override is only valid for control robustness")
        config = {**config, "root_seed": int(args.root_seed_override)}
    control_overrides = {
        "population": args.population_override,
        "cem_iterations": args.cem_iterations_override,
        "max_control_steps": args.max_control_steps_override,
        "uncertainty_weight": args.uncertainty_weight_override,
        "elite_min_normalized_distance": args.elite_min_distance_override,
        "feasible_proposal_resample_attempts": (
            args.feasible_proposal_resample_attempts_override
        ),
    }
    if any(value is not None for value in control_overrides.values()):
        if args.phase != "control":
            raise ValueError("planner overrides are only valid for control ablations")
        baseline = dict(config["baseline"])
        for key, value in control_overrides.items():
            if value is not None:
                baseline[key] = value
        if (
            int(baseline["population"]) <= 0
            or int(baseline["cem_iterations"]) <= 0
            or int(baseline["max_control_steps"]) <= 0
        ):
            raise ValueError("CEM population, iterations, and control steps must be positive")
        if float(baseline["uncertainty_weight"]) < 0:
            raise ValueError("uncertainty weight must be non-negative")
        if float(baseline.get("elite_min_normalized_distance", 0.0)) < 0:
            raise ValueError("elite minimum distance must be non-negative")
        if int(baseline.get("feasible_proposal_resample_attempts", 0)) < 0:
            raise ValueError("feasible proposal resample attempts must be non-negative")
        config = {**config, "baseline": baseline}
    if args.phase == "probes":
        run_probes(args, config)
    elif args.phase == "analyze-probes":
        analyze_probes(args, config)
    elif args.phase == "control":
        if args.mode is None:
            raise ValueError("control phase requires --mode")
        run_control(args, config)
    else:
        aggregate_gate_a(args, config)


if __name__ == "__main__":
    main()
