#!/usr/bin/env python3
"""Model-free candidate-only offline evaluation for H1 meta generations."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from continuous_control_v12.contracts import Bounds
from specialist_rebuild_v2.common import ACTION_FIELDS

from .compiler import compile_guidance
from .contracts import MetaContractError, load_protocol_json, parse_meta_output
from .training import PACKAGE_ROOT, _guard_candidate_path


SCALAR_FIELDS = (
    "decision",
    "observation_request",
    "objective_profile",
    "mask_profile",
    "step_scale",
    "risk_mode",
    "confidence",
)
DIRECTION_VALUES = tuple(
    load_protocol_json("guidance_codebook.json")["output_space"][
        "directions_per_actuator"
    ]
)
OUTPUT_SPACE = load_protocol_json("guidance_codebook.json")["output_space"]
META_PROTOCOL = load_protocol_json("meta_controller_protocol.json")
PREREGISTERED_SEEDS = (2026080201, 2026080202, 2026080203)
EXPECTED_DEV_RECORDS = int(META_PROTOCOL["data"]["dev_records"])
EXPECTED_CANDIDATE_EVAL_RECORDS = int(
    META_PROTOCOL["data"]["candidate_eval_episodes"]
)

RegretHook = Callable[[str, Mapping[str, Any], Mapping[str, Any] | None, Mapping[str, Any]], float | None]
CompilerHook = Callable[[Mapping[str, Any]], bool]


class OfflineEvaluationError(ValueError):
    pass


def _classes_for(field: str) -> tuple[str, ...]:
    key = {
        "decision": "decisions",
        "observation_request": "observation_requests",
        "objective_profile": "objective_profiles",
        "mask_profile": "mask_profiles",
        "step_scale": "step_scales",
        "risk_mode": "risk_modes",
        "confidence": "confidences",
    }[field]
    return tuple(OUTPUT_SPACE[key])


def _confusion(
    truth: Sequence[str], prediction: Sequence[str], classes: Sequence[str]
) -> dict[str, dict[str, int]]:
    predicted_classes = list(classes) + (["<invalid>"] if "<invalid>" in prediction else [])
    return {
        actual: {
            predicted: sum(
                left == actual and right == predicted
                for left, right in zip(truth, prediction, strict=True)
            )
            for predicted in predicted_classes
        }
        for actual in classes
    }


def _classification_metrics(
    truth: Sequence[str], prediction: Sequence[str], classes: Sequence[str]
) -> dict[str, Any]:
    if len(truth) != len(prediction) or not truth:
        raise OfflineEvaluationError("classification vectors must be nonempty and aligned")
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in classes:
        tp = sum(left == label and right == label for left, right in zip(truth, prediction, strict=True))
        fp = sum(left != label and right == label for left, right in zip(truth, prediction, strict=True))
        fn = sum(left == label and right != label for left, right in zip(truth, prediction, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": sum(value == label for value in truth),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
    return {
        "accuracy": sum(left == right for left, right in zip(truth, prediction, strict=True))
        / len(truth),
        "macro_f1": statistics.fmean(f1_values),
        "per_class": per_class,
        "confusion_matrix": _confusion(truth, prediction, classes),
    }


def _manifest_id(record: Mapping[str, Any], index: int) -> str:
    for key in ("sample_id", "record_id", "example_id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    raise OfflineEvaluationError(f"manifest record {index} has no stable ID")


def _manifest_target(record: Mapping[str, Any], sample_id: str) -> dict[str, Any]:
    for key in ("oracle_output", "target", "meta_target", "target_configuration"):
        value = record.get(key)
        if isinstance(value, Mapping):
            try:
                return parse_meta_output(value).to_dict()
            except MetaContractError as exc:
                raise OfflineEvaluationError(
                    f"manifest {sample_id} has invalid {key}: {exc}"
                ) from exc
    raise OfflineEvaluationError(f"manifest {sample_id} has no strict meta target")


def _normalise_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_split: str,
    expected_count: int,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        sample_id = _manifest_id(record, index)
        if sample_id in output:
            raise OfflineEvaluationError(f"duplicate manifest sample ID: {sample_id}")
        split = record.get("split")
        if split != expected_split:
            raise OfflineEvaluationError(
                f"manifest {sample_id}: expected split={expected_split!r}, got {split!r}"
            )
        output[sample_id] = {
            "sample_id": sample_id,
            "target": _manifest_target(record, sample_id),
            "record": record,
        }
    if not output:
        raise OfflineEvaluationError("manifest is empty")
    if len(output) != expected_count:
        raise OfflineEvaluationError(
            f"{expected_split} manifest must contain exactly {expected_count} unique "
            f"records, got {len(output)}"
        )
    return output


def _normalise_predictions(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    expected_seeds: Sequence[int],
) -> dict[int, dict[str, Mapping[str, Any]]]:
    grouped: dict[int, dict[str, Mapping[str, Any]]] = {
        int(seed): {} for seed in expected_seeds
    }
    for index, record in enumerate(records):
        sample_id = record.get("sample_id", record.get("record_id"))
        if not isinstance(sample_id, str) or sample_id not in manifest:
            raise OfflineEvaluationError(
                f"prediction record {index} references unknown sample ID {sample_id!r}"
            )
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in grouped:
            raise OfflineEvaluationError(
                f"prediction {sample_id}: seed must be one of {list(expected_seeds)}"
            )
        if sample_id in grouped[seed]:
            raise OfflineEvaluationError(
                f"duplicate prediction for seed={seed}, sample_id={sample_id}"
            )
        if not isinstance(record.get("prediction"), str):
            raise OfflineEvaluationError(f"prediction {sample_id}: prediction must be raw text")
        grouped[seed][sample_id] = record
    expected_ids = set(manifest)
    for seed, supplied in grouped.items():
        supplied_ids = set(supplied)
        if supplied_ids != expected_ids:
            missing = sorted(expected_ids - supplied_ids)
            extra = sorted(supplied_ids - expected_ids)
            raise OfflineEvaluationError(
                f"seed={seed} predictions must provide one unique row for every "
                f"manifest record; missing={missing}, extra={extra}"
            )
    expected_total = len(expected_ids) * len(grouped)
    if len(records) != expected_total:
        raise OfflineEvaluationError(
            f"prediction grid must contain exactly {expected_total} rows, got "
            f"{len(records)}"
        )
    return grouped


def _default_compiler_hook(prediction: Mapping[str, Any]) -> bool:
    bounds = Bounds(
        action_low=np.asarray([-0.05, -0.05, -0.02, -0.02]),
        action_high=np.asarray([0.05, 0.05, 0.02, 0.02]),
        position_low=np.full(4, -3.0),
        position_high=np.full(4, 3.0),
    )
    compile_guidance(prediction, default_bounds=bounds)
    return True


def _regret_from_record(record: Mapping[str, Any]) -> float | None:
    value = record.get("configuration_regret")
    if value is None:
        selected = record.get("selected_configuration_cost")
        oracle = record.get("oracle_configuration_cost")
        if selected is not None and oracle is not None:
            value = float(selected) - float(oracle)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise OfflineEvaluationError("configuration regret must be finite")
    if number < -1e-12:
        raise OfflineEvaluationError("configuration regret must not be negative")
    return max(number, 0.0)


def _seed_metrics(
    *,
    seed: int,
    manifest: Mapping[str, Mapping[str, Any]],
    supplied: Mapping[str, Mapping[str, Any]],
    regret_hook: RegretHook | None,
    compiler_hook: CompilerHook,
    seed_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    error_codes: Counter[str] = Counter()
    regrets: list[float] = []
    latencies: list[float] = []
    for sample_id in sorted(manifest):
        entry = manifest[sample_id]
        prediction_record = supplied.get(sample_id)
        if prediction_record is None:
            raise OfflineEvaluationError(
                f"seed={seed} is missing the required prediction for {sample_id}"
            )
        parsed: dict[str, Any] | None = None
        parse_error: str | None = None
        compiled_valid = False
        try:
            parsed = parse_meta_output(prediction_record["prediction"]).to_dict()
        except MetaContractError as exc:
            parse_error = exc.code
        latency = prediction_record.get("latency_seconds")
        if latency is not None:
            number = float(latency)
            if not math.isfinite(number) or number < 0:
                raise OfflineEvaluationError(
                    f"prediction {sample_id}: latency must be finite and nonnegative"
                )
            latencies.append(number)
        if parse_error is not None:
            error_codes[parse_error] += 1
        if parsed is not None:
            try:
                compiled_valid = bool(compiler_hook(parsed))
            except Exception:  # compiler rejection is an evaluated failure
                compiled_valid = False

        regret: float | None = None
        if regret_hook is not None:
            regret = regret_hook(sample_id, entry["target"], parsed, entry["record"])
        else:
            regret = _regret_from_record(prediction_record)
        if regret is not None:
            regret = float(regret)
            if not math.isfinite(regret) or regret < -1e-12:
                raise OfflineEvaluationError(
                    f"configuration regret hook returned invalid value for {sample_id}"
                )
            regrets.append(max(regret, 0.0))
        rows.append(
            {
                "sample_id": sample_id,
                "target": entry["target"],
                "prediction": parsed,
                "parse_error": parse_error,
                "compiled_valid": compiled_valid,
            }
        )

    total = len(rows)
    valid = sum(row["prediction"] is not None for row in rows)
    field_metrics: dict[str, Any] = {}
    for field in SCALAR_FIELDS:
        truth = [row["target"][field] for row in rows]
        predicted = [
            row["prediction"][field] if row["prediction"] is not None else "<invalid>"
            for row in rows
        ]
        field_metrics[field] = _classification_metrics(truth, predicted, _classes_for(field))

    direction_truth: list[str] = []
    direction_prediction: list[str] = []
    per_actuator: dict[str, Any] = {}
    for actuator in ACTION_FIELDS:
        truth = [row["target"]["directional_prior"][actuator] for row in rows]
        predicted = [
            row["prediction"]["directional_prior"][actuator]
            if row["prediction"] is not None
            else "<invalid>"
            for row in rows
        ]
        per_actuator[actuator] = _classification_metrics(
            truth, predicted, DIRECTION_VALUES
        )
        direction_truth.extend(truth)
        direction_prediction.extend(predicted)
    direction_metrics = _classification_metrics(
        direction_truth, direction_prediction, DIRECTION_VALUES
    )
    direction_metrics["per_actuator"] = per_actuator

    full_exact = sum(
        row["prediction"] == row["target"] for row in rows
    ) / total
    reason_exact = sum(
        row["prediction"] is not None
        and row["prediction"]["reason_codes"] == row["target"]["reason_codes"]
        for row in rows
    ) / total
    confidence_distribution = Counter(
        row["prediction"]["confidence"]
        for row in rows
        if row["prediction"] is not None
    )
    metadata = dict(seed_metadata or {})
    required_training_metadata = {
        "best_dev_loss": metadata.get("best_dev_loss"),
        "final_dev_loss": metadata.get("final_dev_loss"),
        "wall_time_seconds": metadata.get("wall_time_seconds"),
        "peak_memory_bytes": metadata.get("peak_memory_bytes"),
    }
    return {
        "seed": seed,
        "records": total,
        "predictions_supplied": len(supplied),
        "prediction_coverage_rate": len(supplied) / total,
        "valid_json_rate": valid / total,
        "invalid_or_unknown_field_rate": (total - valid) / total,
        "parse_error_counts": dict(sorted(error_codes.items())),
        "decision_macro_f1": field_metrics["decision"]["macro_f1"],
        "full_configuration_exact_match": full_exact,
        "reason_codes_exact_match": reason_exact,
        "field_metrics": field_metrics,
        "objective_profile_accuracy": field_metrics["objective_profile"]["accuracy"],
        "mask_profile_accuracy": field_metrics["mask_profile"]["accuracy"],
        "step_scale_accuracy": field_metrics["step_scale"]["accuracy"],
        "risk_mode_accuracy": field_metrics["risk_mode"]["accuracy"],
        "direction_accuracy": direction_metrics["accuracy"],
        "direction_macro_f1": direction_metrics["macro_f1"],
        "direction_metrics": direction_metrics,
        "confidence_distribution": {
            key: confidence_distribution.get(key, 0)
            for key in _classes_for("confidence")
        },
        "compiled_guidance_validity_rate": sum(row["compiled_valid"] for row in rows)
        / total,
        "compiled_guidance_valid_count": sum(row["compiled_valid"] for row in rows),
        "configuration_regret": {
            "evaluated_count": len(regrets),
            "coverage_rate": len(regrets) / total,
            "mean": statistics.fmean(regrets) if regrets else None,
            "median": statistics.median(regrets) if regrets else None,
            "maximum": max(regrets) if regrets else None,
        },
        "latency_seconds": {
            "evaluated_count": len(latencies),
            "mean": statistics.fmean(latencies) if latencies else None,
            "maximum": max(latencies) if latencies else None,
        },
        "training_run": required_training_metadata,
    }


def _evaluate_prediction_set(
    *,
    manifest_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int],
    expected_split: str,
    expected_count: int,
    version: str,
    scope: str,
    regret_hook: RegretHook | None = None,
    compiler_hook: CompilerHook | None = None,
    seed_metadata: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if tuple(expected_seeds) != PREREGISTERED_SEEDS:
        raise OfflineEvaluationError(
            "offline evaluator requires all three preregistered training seeds"
        )
    manifest = _normalise_manifest(
        manifest_records,
        expected_split=expected_split,
        expected_count=expected_count,
    )
    predictions = _normalise_predictions(prediction_records, manifest, expected_seeds)
    compiler_hook = compiler_hook or _default_compiler_hook
    per_seed = [
        _seed_metrics(
            seed=seed,
            manifest=manifest,
            supplied=predictions[seed],
            regret_hook=regret_hook,
            compiler_hook=compiler_hook,
            seed_metadata=None if seed_metadata is None else seed_metadata.get(seed),
        )
        for seed in expected_seeds
    ]
    aggregate_fields = (
        "valid_json_rate",
        "invalid_or_unknown_field_rate",
        "decision_macro_f1",
        "full_configuration_exact_match",
        "objective_profile_accuracy",
        "mask_profile_accuracy",
        "step_scale_accuracy",
        "risk_mode_accuracy",
        "direction_accuracy",
        "direction_macro_f1",
        "compiled_guidance_validity_rate",
    )
    aggregate = {
        field: {
            "mean": statistics.fmean(float(row[field]) for row in per_seed),
            "std_population": statistics.pstdev(float(row[field]) for row in per_seed),
        }
        for field in aggregate_fields
    }
    regrets = [
        row["configuration_regret"]["mean"]
        for row in per_seed
        if row["configuration_regret"]["mean"] is not None
    ]
    aggregate["configuration_regret_mean"] = {
        "mean": statistics.fmean(regrets) if regrets else None,
        "std_population": statistics.pstdev(regrets) if regrets else None,
        "seeds_with_regret": len(regrets),
    }
    return {
        "version": version,
        "scope": scope,
        "formal_frozen_evaluation_enabled": False,
        "expected_seeds": list(expected_seeds),
        "manifest_split": expected_split,
        "manifest_record_count": len(manifest),
        "prediction_grid_complete": True,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def evaluate_meta_predictions(
    *,
    manifest_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int],
    regret_hook: RegretHook | None = None,
    compiler_hook: CompilerHook | None = None,
    seed_metadata: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate the complete preregistered 24-row candidate dev split only."""

    return _evaluate_prediction_set(
        manifest_records=manifest_records,
        prediction_records=prediction_records,
        expected_seeds=expected_seeds,
        expected_split="dev",
        expected_count=EXPECTED_DEV_RECORDS,
        version="qwen_h1_meta_v0_offline_evaluation_v1",
        scope="complete 24-record candidate dev; model-free evaluator",
        regret_hook=regret_hook,
        compiler_hook=compiler_hook,
        seed_metadata=seed_metadata,
    )


def evaluate_candidate_eval_predictions(
    *,
    manifest_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int],
    regret_hook: RegretHook | None = None,
    compiler_hook: CompilerHook | None = None,
    seed_metadata: Mapping[int, Mapping[str, Any]] | None = None,
    require_preregistered_cardinality: bool = True,
) -> dict[str, Any]:
    """Score a complete 36-row candidate-eval grid for ablation diagnostics.

    This deliberately separate entry point prevents candidate-eval rows from
    being mislabeled as the preregistered full-dev offline result.
    """

    return _evaluate_prediction_set(
        manifest_records=manifest_records,
        prediction_records=prediction_records,
        expected_seeds=expected_seeds,
        expected_split="candidate_eval",
        expected_count=(
            EXPECTED_CANDIDATE_EVAL_RECORDS
            if require_preregistered_cardinality
            else len(manifest_records)
        ),
        version="qwen_h1_meta_v0_candidate_eval_diagnostics_v1",
        scope="complete 36-record candidate_eval; ablation diagnostics only",
        regret_hook=regret_hook,
        compiler_hook=compiler_hook,
        seed_metadata=seed_metadata,
    )


def _read_records(path: Path) -> list[dict[str, Any]]:
    path = _guard_candidate_path(path, role="offline_input")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise OfflineEvaluationError(f"empty input: {path}")
    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        whole = None
    if isinstance(whole, list):
        records = whole
    elif isinstance(whole, dict) and isinstance(whole.get("records"), list):
        records = whole["records"]
    elif whole is None:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = [whole]
    if not all(isinstance(row, dict) for row in records):
        raise OfflineEvaluationError(f"records must be JSON objects: {path}")
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--seed-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = None
    if args.seed_metadata is not None:
        raw = json.loads(
            _guard_candidate_path(args.seed_metadata, role="seed_metadata").read_text(
                encoding="utf-8"
            )
        )
        metadata = {int(key): value for key, value in raw.items()}
    report = evaluate_meta_predictions(
        manifest_records=_read_records(args.manifest),
        prediction_records=_read_records(args.predictions),
        expected_seeds=(2026080201, 2026080202, 2026080203),
        seed_metadata=metadata,
    )
    output = _guard_candidate_path(args.output, role="offline_output", must_exist=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "candidate_offline_evaluation_written", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
