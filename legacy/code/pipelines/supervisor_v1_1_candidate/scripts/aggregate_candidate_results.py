#!/usr/bin/env python3
"""Aggregate candidate-only Qwen, intervention, ablation, and baseline results."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_vl_supervisor_v1.evaluate_offline import (
    evaluate_records,
    parse_supervisor_json,
)

BASE = ROOT / "supervisor_v1_1_candidate"
SEEDS = (2026080101, 2026080102, 2026080103)
DIAGNOSES = ("nominal", "sensor_saturation", "secondary_reflection")
POLICIES = ("standard", "lower_exposure_reacquire", "primary_spot")
ACTIONS = ("reacquire", "switch_measurement", "execute", "continue", "stop")
BOOTSTRAP_SEED = 2026085101
BOOTSTRAP_REPLICATES = 4000
INTERVENTIONS = (
    "blank_image",
    "cross_class_image_shuffle",
    "random_image_shuffle_2026084101",
    "random_image_shuffle_2026084102",
    "random_image_shuffle_2026084103",
    "metrics_blank",
    "metrics_shuffle",
    "goal_history_budget_empty",
    "goal_history_budget_shuffle",
    "goal_empty",
    "history_empty",
    "budget_empty",
    "goal_shuffle",
    "history_shuffle",
    "budget_shuffle",
)
STRUCTURAL_NO_OP_INTERVENTIONS = frozenset(("history_empty", "history_shuffle", "budget_shuffle"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_new_json(path: Path, value: Any) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError(f"existing JSON does not match deterministic recomputation: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def summary(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    return {
        "n": len(data),
        "mean": statistics.fmean(data),
        "std_population": statistics.pstdev(data) if len(data) > 1 else 0.0,
        "min": min(data),
        "max": max(data),
    }


def prediction_paths() -> dict[str, list[Path]]:
    values: dict[str, list[Path]] = {
        "full": [BASE / "artifacts/predictions" / f"full_seed_{seed}.jsonl" for seed in SEEDS],
        "image_only_qwen": [BASE / "artifacts/predictions" / f"image_only_seed_{seed}.jsonl" for seed in SEEDS],
        "metrics_only_qwen": [BASE / "artifacts/predictions" / f"metrics_only_seed_{seed}.jsonl" for seed in SEEDS],
    }
    for name in INTERVENTIONS:
        if name in STRUCTURAL_NO_OP_INTERVENTIONS:
            # These SFT files are byte-identical to full dev. Reusing the exact
            # deterministic generations is stronger than spending GPU time on
            # an input that did not change.
            values[f"intervention_{name}"] = [
                BASE / "artifacts/predictions" / f"full_seed_{seed}.jsonl" for seed in SEEDS
            ]
        else:
            values[f"intervention_{name}"] = [
                BASE / "artifacts/predictions/interventions" / f"{name}_seed_{seed}.jsonl" for seed in SEEDS
            ]
    baseline = json.loads((BASE / "reports/unified_baselines.json").read_text(encoding="utf-8"))
    for name, model in baseline["models"].items():
        values[f"baseline_{name}"] = [ROOT / run["predictions"] for run in model["runs"]]
    missing = [str(path) for paths in values.values() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing prediction files: {missing}")
    return values


def combine_predictions(name: str, paths: list[Path]) -> tuple[list[dict[str, Any]], Path]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(read_jsonl(path))
    destination = BASE / "artifacts/predictions/combined" / f"{name}.jsonl"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in records),
            encoding="utf-8",
        )
    elif read_jsonl(destination) != records:
        raise RuntimeError(f"existing combined predictions do not match inputs: {destination}")
    return records, destination


def compact_evaluation(report: dict[str, Any]) -> dict[str, Any]:
    metric_keys = (
        "coverage_rate", "valid_json_rate", "diagnosis_balanced_accuracy", "diagnosis_macro_f1",
        "measurement_policy_accuracy", "supervisor_action_macro_f1", "joint_exact_accuracy",
    )
    diagnosis_matrices = [row["confusion_matrices"]["diagnosis"] for row in report["per_seed"]]
    diagnosis_labels = list(diagnosis_matrices[0]["labels"])
    if any(list(matrix["labels"]) != diagnosis_labels for matrix in diagnosis_matrices[1:]):
        raise RuntimeError("diagnosis confusion-matrix labels differ across seeds")
    diagnosis_sum = np.sum(
        np.asarray([matrix["matrix"] for matrix in diagnosis_matrices], dtype=np.int64), axis=0
    ).tolist()
    return {
        "per_seed": [
            {"seed": row["seed"], **{key: row[key] for key in metric_keys},
             "diagnosis_confusion_matrix": row["confusion_matrices"]["diagnosis"],
             "diagnosis_per_class": row["diagnosis_per_class"]}
            for row in report["per_seed"]
        ],
        "aggregate_metrics": {key: report["aggregate"]["metrics"][key] for key in metric_keys},
        "aggregate_diagnosis_confusion_matrix": {
            "labels": diagnosis_labels,
            "summed_matrix": diagnosis_sum,
            "summed_scored_count": sum(int(matrix["scored_count"]) for matrix in diagnosis_matrices),
        },
        "aggregate_diagnosis_per_class": report["aggregate"]["diagnosis_per_class"],
    }


def parsed_by_seed(records: list[dict[str, Any]], sample_ids: list[str]) -> dict[int, dict[str, dict[str, str] | None]]:
    result: dict[int, dict[str, dict[str, str] | None]] = defaultdict(dict)
    expected = set(sample_ids)
    for row in records:
        sample_id = str(row["sample_id"])
        seed = int(row["seed"])
        if sample_id not in expected or sample_id in result[seed]:
            raise RuntimeError(f"invalid prediction identity for seed={seed}, sample={sample_id}")
        try:
            parsed = parse_supervisor_json(row["prediction"])
        except Exception:
            parsed = None
        result[seed][sample_id] = parsed
    for seed, rows in result.items():
        if set(rows) != expected:
            raise RuntimeError(f"seed {seed} does not cover complete candidate dev")
    return dict(result)


def metric_vector(
    manifest: list[dict[str, Any]], parsed: dict[str, dict[str, str] | None], indices: list[int]
) -> dict[str, float] | None:
    target_diag = [manifest[index]["target"]["diagnosis"] for index in indices]
    if any(label not in target_diag for label in DIAGNOSES):
        return None
    target_policy = [manifest[index]["target"]["measurement_policy"] for index in indices]
    target_action = [manifest[index]["target"]["supervisor_action"] for index in indices]
    predictions = [parsed[manifest[index]["sample_id"]] for index in indices]
    pred_diag = [row["diagnosis"] if row else "<invalid>" for row in predictions]
    pred_policy = [row["measurement_policy"] if row else "<invalid>" for row in predictions]
    pred_action = [row["supervisor_action"] if row else "<invalid>" for row in predictions]
    supported_actions = [action for action in ACTIONS if action in set(target_action)]
    return {
        "diagnosis_balanced_accuracy": float(balanced_accuracy_score(target_diag, pred_diag)),
        "diagnosis_macro_f1": float(f1_score(target_diag, pred_diag, labels=list(DIAGNOSES), average="macro", zero_division=0)),
        "diagnosis_accuracy": float(accuracy_score(target_diag, pred_diag)),
        "measurement_policy_accuracy": float(accuracy_score(target_policy, pred_policy)),
        "supervisor_action_macro_f1": float(f1_score(target_action, pred_action, labels=supported_actions, average="macro", zero_division=0)),
        "joint_exact_accuracy": float(np.mean([
            prediction is not None
            and prediction["diagnosis"] == diagnosis
            and prediction["measurement_policy"] == policy
            and prediction["supervisor_action"] == action
            for prediction, diagnosis, policy, action in zip(predictions, target_diag, target_policy, target_action, strict=True)
        ])),
        "valid_json_rate": float(np.mean([prediction is not None for prediction in predictions])),
    }


def interval(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "point_estimate": float(np.mean(array)),
        "ci95_percentile": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        "bootstrap_std": float(np.std(array)),
        "accepted_replicates": len(values),
    }


def bootstrap_conditions(
    manifest: list[dict[str, Any]], parsed: dict[str, dict[int, dict[str, dict[str, str] | None]]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_ids = [row["sample_id"] for row in manifest]
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(manifest):
        clusters[row["setup_hash"]].append(index)
    cluster_ids = sorted(clusters)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: list[list[int]] = []
    attempts = 0
    while len(draws) < BOOTSTRAP_REPLICATES:
        attempts += 1
        selected = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        indices = [index for cluster in selected for index in clusters[str(cluster)]]
        labels = {manifest[index]["target"]["diagnosis"] for index in indices}
        if labels == set(DIAGNOSES):
            draws.append(indices)
        if attempts > BOOTSTRAP_REPLICATES * 10:
            raise RuntimeError("could not obtain bootstrap replicates containing every diagnosis")

    boot: dict[str, Any] = {}
    per_condition_draw_values: dict[str, list[dict[str, float]]] = {}
    for name, seed_map in parsed.items():
        seed_values = []
        for indices in draws:
            values = [metric_vector(manifest, rows, indices) for rows in seed_map.values()]
            if any(value is None for value in values):
                raise RuntimeError("unexpected missing bootstrap class")
            keys = list(values[0])
            seed_values.append({key: statistics.fmean(value[key] for value in values) for key in keys})
        per_condition_draw_values[name] = seed_values
        boot[name] = {
            "seeds": sorted(seed_map),
            "cluster_unit": "setup_hash preserving complete counterfactual pairs",
            "clusters": len(cluster_ids),
            "replicates": BOOTSTRAP_REPLICATES,
            "attempts": attempts,
            "metrics": {key: interval([row[key] for row in seed_values]) for key in seed_values[0]},
        }

    full_draws = per_condition_draw_values["full"]
    differences: dict[str, Any] = {}
    for name, values in per_condition_draw_values.items():
        if name == "full":
            continue
        metrics = {}
        for key in values[0]:
            # Positive values mean Full is better (an absolute drop under the intervention/ablation/baseline).
            deltas = [full[key] - condition[key] for full, condition in zip(full_draws, values, strict=True)]
            metrics[key] = interval(deltas)
        differences[name] = {"direction": "full_minus_condition", "metrics": metrics}
    return boot, differences


def training_summary(modality: str) -> dict[str, Any]:
    runs = []
    for seed in SEEDS:
        run_dir = BASE / "artifacts/training" / f"{modality}_seed_{seed}"
        manifest = json.loads((run_dir / "run_manifest.latest.json").read_text(encoding="utf-8"))
        log_rows = read_jsonl(Path(manifest["training_result"]["log_jsonl"]))
        train_rows = [row for row in log_rows if "loss" in row and "eval_loss" not in row]
        eval_rows = [row for row in log_rows if "eval_loss" in row]
        eval_curve = [{"step": (index + 1) * 25, "loss": float(row["eval_loss"])} for index, row in enumerate(eval_rows)]
        best = min(eval_curve, key=lambda row: (row["loss"], row["step"]))
        best_step = int(re.search(r"checkpoint-(\d+)$", manifest["training_result"]["best_dev_checkpoint"]).group(1))
        if best_step != best["step"]:
            raise RuntimeError(f"best checkpoint selection mismatch for {modality} seed {seed}")
        final_window = [float(row["loss"]) for row in train_rows[-25:]]
        best_window = [float(row["loss"]) for row in train_rows[max(0, best_step - 25):best_step]]
        adapter = run_dir / "final_adapter/adapter_model.safetensors"
        final_dev = eval_curve[-1]["loss"]
        runs.append({
            "seed": seed,
            "best_step": best_step,
            "best_dev_loss": best["loss"],
            "final_dev_loss": final_dev,
            "reported_whole_training_loss": float(manifest["training_result"]["metrics"]["train_loss"]),
            "trailing_25_step_train_loss_at_best": statistics.fmean(best_window),
            "trailing_25_step_final_train_loss": statistics.fmean(final_window),
            "best_generalization_gap_dev_minus_train_window": best["loss"] - statistics.fmean(best_window),
            "final_generalization_gap_dev_minus_train_window": final_dev - statistics.fmean(final_window),
            "dev_loss_increase_from_best_fraction": (final_dev - best["loss"]) / best["loss"],
            "eval_loss_curve": eval_curve,
            "wall_seconds": float(manifest["training_result"]["runtime_wall_seconds"]),
            "gpu_peak_allocated_bytes": int(manifest["gpu"]["peak_allocated_bytes"]),
            "gpu_peak_reserved_bytes": int(manifest["gpu"]["peak_reserved_bytes"]),
            "config_sha256": manifest["config_sha256"],
            "adapter_model_sha256": sha256(adapter),
            "checkpoint_path": manifest["training_result"]["best_dev_checkpoint"],
            "final_adapter_path": str((run_dir / "final_adapter").relative_to(ROOT)),
        })
    return {
        "runs": runs,
        "aggregate": {
            key: summary(run[key] for run in runs)
            for key in (
                "best_step", "best_dev_loss", "final_dev_loss", "reported_whole_training_loss",
                "trailing_25_step_final_train_loss", "final_generalization_gap_dev_minus_train_window",
                "wall_seconds", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes",
            )
        },
    }


def full_agreement(manifest: list[dict[str, Any]], seed_map: dict[int, dict[str, dict[str, str] | None]]) -> dict[str, Any]:
    rows = []
    for target in manifest:
        sample_id = target["sample_id"]
        predictions = [seed_map[seed][sample_id] for seed in SEEDS]
        diagnoses = [prediction["diagnosis"] if prediction else "<invalid>" for prediction in predictions]
        truth = target["target"]["diagnosis"]
        wrong = sum(prediction != truth for prediction in diagnoses)
        exact = [
            prediction is not None and all(prediction[field] == target["target"][field] for field in ("diagnosis", "measurement_policy", "supervisor_action"))
            for prediction in predictions
        ]
        rows.append({
            "sample_id": sample_id,
            "setup_hash": target["setup_hash"],
            "counterfactual_pair_id": target["counterfactual_pair_id"],
            "anomaly_family": target["provenance"]["anomaly_family"],
            "truth": truth,
            "predicted_diagnoses_by_seed": dict(zip(map(str, SEEDS), diagnoses, strict=True)),
            "diagnosis_wrong_seed_count": wrong,
            "joint_wrong_seed_count": 3 - sum(exact),
            "three_seed_diagnosis_agreement": len(set(diagnoses)) == 1,
        })
    return {
        "records": len(rows),
        "three_seed_diagnosis_agreement_count": sum(row["three_seed_diagnosis_agreement"] for row in rows),
        "three_seed_diagnosis_agreement_rate": statistics.fmean(row["three_seed_diagnosis_agreement"] for row in rows),
        "wrong_seed_count_histogram": dict(sorted(Counter(row["diagnosis_wrong_seed_count"] for row in rows).items())),
        "repeated_errors_wrong_in_at_least_two_seeds": [row for row in rows if row["diagnosis_wrong_seed_count"] >= 2],
        "all_records": rows,
    }


def generation_telemetry(paths: list[Path]) -> dict[str, Any] | None:
    reports = [path.with_suffix(".report.json") for path in paths]
    if not all(path.is_file() for path in reports):
        return None
    rows = [json.loads(path.read_text(encoding="utf-8"))["telemetry"] for path in reports]
    return {
        "latency_seconds_total": summary(row["generation_latency_seconds_total_this_invocation"] for row in rows),
        "gpu_peak_allocated_bytes": summary(row["gpu_peak_allocated_bytes"] for row in rows),
        "gpu_peak_reserved_bytes": summary(row["gpu_peak_reserved_bytes"] for row in rows),
    }


def main() -> None:
    manifest_path = BASE / "manifests/manifest_dev.jsonl"
    manifest = read_jsonl(manifest_path)
    paths_by_condition = prediction_paths()
    evaluations: dict[str, Any] = {}
    parsed: dict[str, dict[int, dict[str, dict[str, str] | None]]] = {}
    artifacts: dict[str, Any] = {}
    for name, paths in paths_by_condition.items():
        records, combined = combine_predictions(name, paths)
        expected_seeds = sorted({int(row["seed"]) for row in records})
        report = evaluate_records(manifest, records, expected_seeds=expected_seeds)
        report_path = BASE / "reports/evaluations" / f"{name}.json"
        atomic_new_json(report_path, report)
        evaluations[name] = compact_evaluation(report)
        parsed[name] = parsed_by_seed(records, [row["sample_id"] for row in manifest])
        artifacts[name] = {
            "source_prediction_files": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in paths],
            "combined": {"path": str(combined.relative_to(ROOT)), "sha256": sha256(combined)},
            "evaluation": {"path": str(report_path.relative_to(ROOT)), "sha256": sha256(report_path)},
            "generation_telemetry": generation_telemetry(paths),
        }
        intervention_name = name.removeprefix("intervention_")
        if intervention_name in STRUCTURAL_NO_OP_INTERVENTIONS:
            source = BASE / "sft/interventions" / intervention_name / "sft_dev.jsonl"
            full_source = BASE / "sft/sft_dev.jsonl"
            if sha256(source) != sha256(full_source):
                raise RuntimeError(f"declared structural no-op is not byte-identical: {intervention_name}")
            artifacts[name]["prediction_reuse"] = {
                "reason": "intervention SFT is byte-identical to Full dev; no model rerun was required",
                "intervention_sft_sha256": sha256(source),
                "full_sft_sha256": sha256(full_source),
            }

    bootstrap, paired_differences = bootstrap_conditions(manifest, parsed)
    training = {name: training_summary(name) for name in ("full", "image_only", "metrics_only")}
    random_names = [f"intervention_random_image_shuffle_{seed}" for seed in (2026084101, 2026084102, 2026084103)]
    random_shuffle_summary = {
        key: summary(evaluations[name]["aggregate_metrics"][key]["mean"] for name in random_names)
        for key in evaluations[random_names[0]]["aggregate_metrics"]
    }
    result = {
        "version": "supervisor_v1_1_candidate_aggregate_v1",
        "status": "NOT SEALED — FROZEN EVALUATION DISABLED",
        "evidence_scope": "candidate train/dev engineering evidence only",
        "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": sha256(manifest_path), "records": len(manifest)},
        "evaluations": evaluations,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "method": "nonparametric setup-cluster resampling preserving all records and complete counterfactual pairs in a selected setup; percentile CI",
            "conditions": bootstrap,
            "paired_full_minus_condition": paired_differences,
        },
        "full_three_seed_agreement": full_agreement(manifest, parsed["full"]),
        "training": training,
        "random_image_shuffle_across_three_fixed_permutations": random_shuffle_summary,
        "artifacts": artifacts,
        "diagnosis_is_primary_capability": True,
        "policy_action_joint_interpretation": "deterministic diagnosis routing consistency, not three independent reasoning tasks",
        "frozen_iid_ood_or_protected_used": False,
        "formal_frozen_evaluation_run": False,
    }
    destination = BASE / "reports/candidate_results.json"
    atomic_new_json(destination, result)
    print(json.dumps({
        "output": str(destination),
        "full": evaluations["full"]["aggregate_metrics"],
        "repeated_full_errors": len(result["full_three_seed_agreement"]["repeated_errors_wrong_in_at_least_two_seeds"]),
    }, indent=2))


if __name__ == "__main__":
    main()
