#!/usr/bin/env python3
"""Strict, model-free offline evaluation for the Qwen-VL supervisor.

The evaluator never runs a model.  It joins already-captured generations to a
manifest by ``sample_id``, validates each generation as one canonical JSON
object, and computes the frozen supervisor metrics.  Manifest target fields
that are explicitly masked are omitted from every metric denominator.

Prediction JSONL contract::

    {"sample_id": "...", "prediction": "{...}", "seed": 7}

``seed`` is optional.  If present, each seed is evaluated independently and
the report also contains across-seed aggregate statistics. Formal evaluation
must pass either ``--evaluation-config`` or ``--expected-seeds``; development
smoke evaluation may omit both.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import DIAGNOSES, MEASUREMENT_POLICIES, SUPERVISOR_ACTIONS


TARGET_FIELDS = ("diagnosis", "measurement_policy", "supervisor_action")
FROZEN_ENUMS: dict[str, tuple[str, ...]] = {
    "diagnosis": DIAGNOSES,
    "measurement_policy": MEASUREMENT_POLICIES,
    "supervisor_action": SUPERVISOR_ACTIONS,
}
INVALID_PREDICTION_LABEL = "<invalid>"


class EvaluationError(ValueError):
    """Raised when evaluation inputs violate the frozen contract."""


class PredictionFormatError(EvaluationError):
    """Raised when generated text is not one canonical supervisor JSON object."""


def _reject_constant(value: str) -> None:
    raise PredictionFormatError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PredictionFormatError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def parse_supervisor_json(text: str) -> dict[str, str]:
    """Parse one complete generated string under the frozen output contract.

    Leading/trailing JSON whitespace is accepted.  Markdown fences, prose,
    multiple values, duplicate keys, non-finite constants, extra/missing keys,
    and values outside the frozen enums are rejected.
    """

    if not isinstance(text, str):
        raise PredictionFormatError("prediction must be a string")
    if not text.strip():
        raise PredictionFormatError("prediction is empty")

    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    start = len(text) - len(text.lstrip())
    try:
        parsed, end = decoder.raw_decode(text, idx=start)
    except PredictionFormatError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PredictionFormatError(f"invalid JSON: {exc}") from exc
    if text[end:].strip():
        raise PredictionFormatError("trailing content after the JSON object")
    if not isinstance(parsed, dict):
        raise PredictionFormatError("prediction must be a JSON object")

    actual_keys = set(parsed)
    expected_keys = set(TARGET_FIELDS)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise PredictionFormatError(f"prediction keys do not match contract; missing={missing}, extra={extra}")

    canonical: dict[str, str] = {}
    for field in TARGET_FIELDS:
        value = parsed[field]
        if not isinstance(value, str):
            raise PredictionFormatError(f"{field} must be a string")
        if value not in FROZEN_ENUMS[field]:
            raise PredictionFormatError(
                f"invalid {field}={value!r}; expected one of {list(FROZEN_ENUMS[field])}"
            )
        canonical[field] = value
    return canonical


def _load_records(path: str | Path, *, kind: str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise EvaluationError(f"{kind} file does not exist: {source}")
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise EvaluationError(f"{kind} file is empty: {source}")

    records: Any
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            records = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"invalid {kind} JSON: {exc}") from exc
    elif stripped.startswith("{"):
        # A JSON object might be a wrapper/single record, but JSONL also starts
        # with ``{``.  Prefer a whole-file object only when parsing consumes it.
        try:
            whole = json.loads(text)
        except json.JSONDecodeError:
            whole = None
        if isinstance(whole, dict) and "records" in whole:
            records = whole["records"]
        elif isinstance(whole, dict):
            records = [whole]
        else:
            records = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(f"invalid {kind} JSONL at line {line_number}: {exc}") from exc
                records.append(row)
    else:
        raise EvaluationError(f"{kind} must be a JSON array/object or JSONL")

    if not isinstance(records, list):
        raise EvaluationError(f"{kind} records must be a list")
    checked: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise EvaluationError(f"{kind} record {index} is not a JSON object")
        checked.append(row)
    return checked


def _validate_mask_fields(fields: Iterable[Any], *, sample_id: str) -> set[str]:
    masked: set[str] = set()
    for field in fields:
        if not isinstance(field, str) or field not in TARGET_FIELDS:
            raise EvaluationError(f"sample {sample_id}: invalid masked target field {field!r}")
        masked.add(field)
    return masked


def _extract_target(record: Mapping[str, Any], sample_id: str) -> tuple[dict[str, str], set[str]]:
    raw_target = record.get("target", {})
    if raw_target is None:
        raw_target = {}
    if not isinstance(raw_target, Mapping):
        raise EvaluationError(f"sample {sample_id}: target must be an object")

    masked: set[str] = set()
    for container in (record, raw_target):
        explicit = container.get("masked_target_fields")
        if explicit is not None:
            if not isinstance(explicit, list):
                raise EvaluationError(f"sample {sample_id}: masked_target_fields must be a list")
            masked.update(_validate_mask_fields(explicit, sample_id=sample_id))

        for key in ("field_mask", "target_mask", "supervision_mask"):
            mask = container.get(key)
            if mask is None:
                continue
            if not isinstance(mask, Mapping):
                raise EvaluationError(f"sample {sample_id}: {key} must be an object")
            unknown = set(mask) - set(TARGET_FIELDS)
            if unknown:
                raise EvaluationError(f"sample {sample_id}: {key} has unknown fields {sorted(unknown)}")
            for field, is_supervised in mask.items():
                if not isinstance(is_supervised, bool):
                    raise EvaluationError(f"sample {sample_id}: {key}.{field} must be boolean")
                if not is_supervised:
                    masked.add(field)

    target: dict[str, str] = {}
    for field in TARGET_FIELDS:
        if field in raw_target:
            value = raw_target[field]
        elif f"target_{field}" in record:
            value = record[f"target_{field}"]
        elif field in record:
            value = record[field]
        else:
            value = None

        if value is None:
            masked.add(field)
            continue
        if field in masked:
            # A masked value is deliberately not inspected or scored.  This is
            # important for blinded/frozen cohorts whose target may be a token.
            continue
        if not isinstance(value, str) or value not in FROZEN_ENUMS[field]:
            raise EvaluationError(
                f"sample {sample_id}: invalid unmasked target {field}={value!r}"
            )
        target[field] = value

    if not target:
        raise EvaluationError(f"sample {sample_id}: all target fields are masked")
    return target, masked


def _metadata(record: Mapping[str, Any], key: str) -> Any:
    value = record.get(key)
    if value is not None:
        return value
    provenance = record.get("provenance")
    if isinstance(provenance, Mapping):
        value = provenance.get(key)
        if value is not None:
            return value
    target = record.get("target")
    if isinstance(target, Mapping):
        provenance = target.get("provenance")
        if isinstance(provenance, Mapping):
            return provenance.get(key)
    return None


def _normalise_manifest(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise EvaluationError(f"manifest record {index}: sample_id must be a non-empty string")
        if sample_id in manifest:
            raise EvaluationError(f"duplicate manifest sample_id: {sample_id}")
        target, masked = _extract_target(record, sample_id)
        manifest[sample_id] = {
            "sample_id": sample_id,
            "target": target,
            "masked_target_fields": sorted(masked),
            "anomaly_family": _metadata(record, "anomaly_family"),
            "width_quartile": _metadata(record, "width_quartile"),
            "boundary_status": _metadata(record, "boundary_status"),
            "severity_bucket": _metadata(record, "severity_bucket"),
        }
    if not manifest:
        raise EvaluationError("manifest contains no records")
    return manifest


_MISSING_SEED = object()
EXPECTED_SEED_TOKEN = re.compile(r"^-?[0-9]+$")


def _seed_sort_key(seed: Any) -> tuple[str, str]:
    if seed is _MISSING_SEED:
        return ("", "")
    return (type(seed).__name__, json.dumps(seed, sort_keys=True, ensure_ascii=False))


def _normalise_predictions(
    records: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[Any, str]] = set()
    count = 0
    for index, record in enumerate(records):
        count += 1
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise EvaluationError(f"prediction record {index}: sample_id must be a non-empty string")
        if sample_id not in manifest:
            raise EvaluationError(f"prediction references unknown sample_id: {sample_id}")
        if "prediction" not in record:
            raise EvaluationError(f"prediction record {index}: missing prediction string")
        prediction = record["prediction"]
        if not isinstance(prediction, str):
            raise EvaluationError(f"prediction record {index}: prediction must be a string")

        seed = record.get("seed", _MISSING_SEED)
        if seed is not _MISSING_SEED and (isinstance(seed, bool) or not isinstance(seed, (int, str))):
            raise EvaluationError(f"prediction record {index}: seed must be an integer or string")
        key = (seed, sample_id)
        if key in seen:
            shown_seed = None if seed is _MISSING_SEED else seed
            raise EvaluationError(f"duplicate prediction for sample_id={sample_id}, seed={shown_seed!r}")
        seen.add(key)

        try:
            parsed = parse_supervisor_json(prediction)
            parse_error = None
        except PredictionFormatError as exc:
            parsed = None
            parse_error = str(exc)
        grouped[seed].append(
            {
                "sample_id": sample_id,
                "prediction": parsed,
                "parse_error": parse_error,
                "was_supplied": True,
                "manifest": manifest[sample_id],
            }
        )
    if count == 0:
        raise EvaluationError("predictions contain no records")

    # A missing prediction is a model failure, not an omitted denominator.
    # Complete every observed seed to the manifest before metrics and slices
    # are computed.  ``was_supplied`` keeps coverage distinct from validity.
    all_sample_ids = set(manifest)
    for seed, rows in grouped.items():
        supplied_ids = {row["sample_id"] for row in rows}
        for sample_id in sorted(all_sample_ids - supplied_ids):
            rows.append(
                {
                    "sample_id": sample_id,
                    "prediction": None,
                    "parse_error": "missing_prediction",
                    "was_supplied": False,
                    "manifest": manifest[sample_id],
                }
            )
    return grouped


def _normalise_expected_seeds(
    expected_seeds: Sequence[int | str] | None,
) -> list[int | str] | None:
    if expected_seeds is None:
        return None
    normalised: list[int | str] = []
    for index, seed in enumerate(expected_seeds):
        if isinstance(seed, bool) or not isinstance(seed, (int, str)):
            raise EvaluationError(
                f"expected seed {index} must be an integer or string, got {seed!r}"
            )
        if isinstance(seed, str) and not seed:
            raise EvaluationError(f"expected seed {index} must not be empty")
        if seed in normalised:
            raise EvaluationError(f"duplicate expected seed: {seed!r}")
        normalised.append(seed)
    if not normalised:
        raise EvaluationError("expected seed set must not be empty")
    return normalised


def _display_seed(seed: Any) -> int | str | None:
    return None if seed is _MISSING_SEED else seed


def _enforce_seed_set(
    grouped: Mapping[Any, Any], expected_seeds: Sequence[int | str] | None
) -> list[Any]:
    expected = _normalise_expected_seeds(expected_seeds)
    if expected is None:
        return sorted(grouped, key=_seed_sort_key)
    observed = list(grouped)
    missing = [seed for seed in expected if seed not in observed]
    extra = [seed for seed in observed if seed not in expected]
    if missing or extra:
        raise EvaluationError(
            "prediction seed set mismatch; "
            f"missing={missing!r}, extra={[_display_seed(seed) for seed in extra]!r}"
        )
    return list(expected)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _confusion(
    rows: Sequence[Mapping[str, Any]], field: str
) -> tuple[list[str], list[list[int]], list[tuple[str, str]]]:
    labels = [*FROZEN_ENUMS[field], INVALID_PREDICTION_LABEL]
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    pairs: list[tuple[str, str]] = []
    for row in rows:
        target = row["manifest"]["target"]
        if field not in target:
            continue
        truth = target[field]
        parsed = row["prediction"]
        prediction = parsed[field] if parsed is not None else INVALID_PREDICTION_LABEL
        matrix[index[truth]][index[prediction]] += 1
        pairs.append((truth, prediction))
    return labels, matrix, pairs


def _classification_metrics(
    pairs: Sequence[tuple[str, str]], labels: Sequence[str]
) -> tuple[dict[str, dict[str, float | int | None]], float | None, float | None]:
    per_class: dict[str, dict[str, float | int | None]] = {}
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        tp = sum(truth == label and prediction == label for truth, prediction in pairs)
        fp = sum(truth != label and prediction == label for truth, prediction in pairs)
        fn = sum(truth == label and prediction != label for truth, prediction in pairs)
        support = tp + fn
        predicted = tp + fp
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        if precision is None or recall is None or precision + recall == 0:
            f1 = 0.0 if support else None
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        per_class[label] = {
            "support": support,
            "predicted": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        # Macro metrics average classes represented by a supervised target.
        # This excludes unlabelled continue/stop classes while a prediction of
        # either still counts as an error for its true class.
        if support:
            assert recall is not None and f1 is not None
            recalls.append(recall)
            f1s.append(f1)
    return per_class, _mean(recalls), _mean(f1s)


def _accuracy(pairs: Sequence[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    return sum(truth == prediction for truth, prediction in pairs) / len(pairs)


def _core_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid_count = sum(row["prediction"] is not None for row in rows)
    report: dict[str, Any] = {
        "num_predictions": len(rows),
        "valid_json_count": valid_count,
        "valid_json_rate": valid_count / len(rows) if rows else None,
    }

    pairs_by_field: dict[str, list[tuple[str, str]]] = {}
    confusion_matrices: dict[str, Any] = {}
    per_class_by_field: dict[str, Any] = {}
    macro_f1_by_field: dict[str, float | None] = {}
    balanced_by_field: dict[str, float | None] = {}
    for field in TARGET_FIELDS:
        labels, matrix, pairs = _confusion(rows, field)
        pairs_by_field[field] = pairs
        per_class, balanced_accuracy, macro_f1 = _classification_metrics(pairs, FROZEN_ENUMS[field])
        per_class_by_field[field] = per_class
        balanced_by_field[field] = balanced_accuracy
        macro_f1_by_field[field] = macro_f1
        confusion_matrices[field] = {
            "labels": labels,
            "matrix": matrix,
            "scored_count": len(pairs),
        }

    joint_correct = 0
    joint_count = 0
    for row in rows:
        target = row["manifest"]["target"]
        if not target:
            continue
        joint_count += 1
        parsed = row["prediction"]
        if parsed is not None and all(parsed[field] == value for field, value in target.items()):
            joint_correct += 1

    report.update(
        {
            "diagnosis_scored_count": len(pairs_by_field["diagnosis"]),
            "diagnosis_balanced_accuracy": balanced_by_field["diagnosis"],
            "diagnosis_macro_f1": macro_f1_by_field["diagnosis"],
            "diagnosis_per_class": per_class_by_field["diagnosis"],
            "measurement_policy_scored_count": len(pairs_by_field["measurement_policy"]),
            "measurement_policy_accuracy": _accuracy(pairs_by_field["measurement_policy"]),
            "supervisor_action_scored_count": len(pairs_by_field["supervisor_action"]),
            "supervisor_action_macro_f1": macro_f1_by_field["supervisor_action"],
            "joint_scored_count": joint_count,
            "joint_exact_accuracy": joint_correct / joint_count if joint_count else None,
            "confusion_matrices": confusion_matrices,
        }
    )
    return report


def _slice_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boundary" if value else "non_boundary"
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _group_slices(
    rows: Sequence[Mapping[str, Any]], metadata_key: str, *, reflection_only: bool = False
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if reflection_only:
            target_diagnosis = row["manifest"]["target"].get("diagnosis")
            family = row["manifest"].get("anomaly_family")
            if target_diagnosis != "secondary_reflection" and "reflection" not in str(family).lower():
                continue
        value = row["manifest"].get(metadata_key)
        if value is None:
            continue
        grouped[_slice_name(value)].append(row)
    return {name: _core_metrics(grouped[name]) for name in sorted(grouped)}


def _seed_report(
    rows: Sequence[Mapping[str, Any]], *, seed: Any, manifest_size: int
) -> dict[str, Any]:
    result = _core_metrics(rows)
    supplied_count = sum(bool(row.get("was_supplied", True)) for row in rows)
    result["seed"] = None if seed is _MISSING_SEED else seed
    result["manifest_count"] = manifest_size
    result["supplied_prediction_count"] = supplied_count
    result["coverage_rate"] = supplied_count / manifest_size
    result["missing_prediction_count"] = manifest_size - supplied_count
    result["slices"] = {
        "anomaly_family": _group_slices(rows, "anomaly_family"),
        "reflection_width_quartile": _group_slices(rows, "width_quartile", reflection_only=True),
        "boundary_status": _group_slices(rows, "boundary_status"),
        "severity_bucket": _group_slices(rows, "severity_bucket"),
    }
    invalid = [
        {"sample_id": row["sample_id"], "error": row["parse_error"]}
        for row in rows
        if row["parse_error"] is not None
    ]
    result["invalid_predictions"] = invalid
    return result


PRIMARY_SCALARS = (
    "coverage_rate",
    "valid_json_rate",
    "diagnosis_balanced_accuracy",
    "diagnosis_macro_f1",
    "measurement_policy_accuracy",
    "supervisor_action_macro_f1",
    "joint_exact_accuracy",
)

SLICE_SCALARS = (
    "valid_json_rate",
    "diagnosis_balanced_accuracy",
    "diagnosis_macro_f1",
    "measurement_policy_accuracy",
    "supervisor_action_macro_f1",
    "joint_exact_accuracy",
)

SLICE_GROUPS = (
    "anomaly_family",
    "reflection_width_quartile",
    "boundary_status",
    "severity_bucket",
)


def _summary_statistics(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _aggregate_seed_reports(seed_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in PRIMARY_SCALARS:
        values = [float(report[name]) for report in seed_reports if report.get(name) is not None]
        metrics[name] = _summary_statistics(values)

    diagnosis_per_class: dict[str, Any] = {}
    for label in FROZEN_ENUMS["diagnosis"]:
        diagnosis_per_class[label] = {}
        for name in ("precision", "recall", "f1"):
            values = [
                float(report["diagnosis_per_class"][label][name])
                for report in seed_reports
                if report["diagnosis_per_class"][label][name] is not None
            ]
            diagnosis_per_class[label][name] = _summary_statistics(values)
    return {
        "num_seeds": len(seed_reports),
        "metrics": metrics,
        "diagnosis_per_class": diagnosis_per_class,
        "slices": _aggregate_seed_slices(seed_reports),
    }


def _sum_confusion_matrices(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for field in TARGET_FIELDS:
        matrices = [report["confusion_matrices"][field] for report in reports]
        labels = list(matrices[0]["labels"])
        if any(list(matrix["labels"]) != labels for matrix in matrices[1:]):
            raise EvaluationError(f"cannot aggregate {field} confusion matrices with different labels")
        size = len(labels)
        summed = [[0 for _ in range(size)] for _ in range(size)]
        for matrix in matrices:
            values = matrix["matrix"]
            if len(values) != size or any(len(row) != size for row in values):
                raise EvaluationError(f"cannot aggregate malformed {field} confusion matrix")
            for row_index in range(size):
                for column_index in range(size):
                    summed[row_index][column_index] += int(values[row_index][column_index])
        scored_counts = [int(matrix["scored_count"]) for matrix in matrices]
        aggregate[field] = {
            "labels": labels,
            "summed_matrix": summed,
            "summed_scored_count": sum(scored_counts),
            "scored_count_per_seed": _summary_statistics(scored_counts),
        }
    return aggregate


def _aggregate_core_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise EvaluationError("cannot aggregate an empty subgroup report list")
    metrics = {
        name: _summary_statistics(
            [float(report[name]) for report in reports if report.get(name) is not None]
        )
        for name in SLICE_SCALARS
    }
    counts = {
        name: _summary_statistics([int(report[name]) for report in reports])
        for name in (
            "num_predictions",
            "valid_json_count",
            "diagnosis_scored_count",
            "measurement_policy_scored_count",
            "supervisor_action_scored_count",
            "joint_scored_count",
        )
    }
    diagnosis_per_class: dict[str, Any] = {}
    for label in FROZEN_ENUMS["diagnosis"]:
        diagnosis_per_class[label] = {}
        for name in ("support", "predicted", "precision", "recall", "f1"):
            values = [
                report["diagnosis_per_class"][label][name]
                for report in reports
                if report["diagnosis_per_class"][label][name] is not None
            ]
            diagnosis_per_class[label][name] = _summary_statistics(
                [float(value) for value in values]
            )
    return {
        "num_seeds_with_subgroup": len(reports),
        "metrics": metrics,
        "counts": counts,
        "diagnosis_per_class": diagnosis_per_class,
        "summed_confusion_matrices": _sum_confusion_matrices(reports),
    }


def _aggregate_seed_slices(seed_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for group_name in SLICE_GROUPS:
        slice_names = sorted(
            {
                slice_name
                for seed_report in seed_reports
                for slice_name in seed_report["slices"][group_name]
            }
        )
        aggregate[group_name] = {}
        for slice_name in slice_names:
            matching = [
                seed_report["slices"][group_name][slice_name]
                for seed_report in seed_reports
                if slice_name in seed_report["slices"][group_name]
            ]
            result = _aggregate_core_reports(matching)
            result["seeds_present"] = [
                seed_report["seed"]
                for seed_report in seed_reports
                if slice_name in seed_report["slices"][group_name]
            ]
            aggregate[group_name][slice_name] = result
    return aggregate


def evaluate_records(
    manifest_records: Iterable[Mapping[str, Any]],
    prediction_records: Iterable[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int | str] | None = None,
) -> dict[str, Any]:
    """Evaluate prediction records joined to manifest records by sample ID."""

    manifest = _normalise_manifest(manifest_records)
    grouped = _normalise_predictions(prediction_records, manifest)
    expected = _normalise_expected_seeds(expected_seeds)
    seed_order = _enforce_seed_set(grouped, expected)
    seed_reports = [
        _seed_report(grouped[seed], seed=seed, manifest_size=len(manifest))
        for seed in seed_order
    ]
    return {
        "schema_version": "qwen_vl_supervisor_offline_evaluation_v1",
        "frozen_enums": {field: list(values) for field, values in FROZEN_ENUMS.items()},
        "manifest_count": len(manifest),
        "seed_enforcement": {
            "enabled": expected is not None,
            "expected_seeds": expected,
        },
        "per_seed": seed_reports,
        "aggregate": _aggregate_seed_reports(seed_reports),
    }


def evaluate_files(
    manifest_path: str | Path,
    predictions_path: str | Path,
    *,
    expected_seeds: Sequence[int | str] | None = None,
) -> dict[str, Any]:
    return evaluate_records(
        _load_records(manifest_path, kind="manifest"),
        _load_records(predictions_path, kind="predictions"),
        expected_seeds=expected_seeds,
    )


def expected_seeds_from_evaluation_config(path: str | Path) -> list[int | str]:
    source = Path(path)
    if not source.is_file():
        raise EvaluationError(f"evaluation config does not exist: {source}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependency error
        raise EvaluationError("PyYAML is required for --evaluation-config") from exc
    try:
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise EvaluationError(f"cannot read evaluation config {source}: {exc}") from exc
    if not isinstance(config, Mapping):
        raise EvaluationError("evaluation config must contain one YAML object")
    finalization = config.get("finalization")
    if not isinstance(finalization, Mapping):
        raise EvaluationError("evaluation config has no finalization object")
    seeds = finalization.get("training_seeds")
    if not isinstance(seeds, list):
        raise EvaluationError("evaluation config finalization.training_seeds must be a list")
    return _normalise_expected_seeds(seeds) or []


def parse_expected_seed_tokens(tokens: Sequence[str]) -> list[int | str]:
    parsed: list[int | str] = []
    for raw in tokens:
        for token in raw.split(","):
            value = token.strip()
            if not value:
                raise EvaluationError("--expected-seeds contains an empty token")
            parsed.append(int(value) if EXPECTED_SEED_TOKEN.fullmatch(value) else value)
    return _normalise_expected_seeds(parsed) or []


def _write_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="manifest JSON/JSONL path")
    parser.add_argument("--predictions", required=True, help="captured prediction JSONL path")
    parser.add_argument("--output", required=True, help="evaluation report JSON path")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument(
        "--evaluation-config",
        help="frozen evaluation YAML whose finalization.training_seeds must match predictions",
    )
    seeds.add_argument(
        "--expected-seeds",
        nargs="+",
        metavar="SEED",
        help="exact formal seed set (space- or comma-separated); omitted for smoke mode",
    )
    args = parser.parse_args(argv)

    expected_seeds: list[int | str] | None = None
    if args.evaluation_config is not None:
        expected_seeds = expected_seeds_from_evaluation_config(args.evaluation_config)
    elif args.expected_seeds is not None:
        expected_seeds = parse_expected_seed_tokens(args.expected_seeds)
    report = evaluate_files(
        args.manifest,
        args.predictions,
        expected_seeds=expected_seeds,
    )
    _write_report(report, args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()), "num_seeds": report["aggregate"]["num_seeds"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
