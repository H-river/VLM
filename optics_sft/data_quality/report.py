"""Build structured quality reports for physics/text SFT datasets."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
PHYSICS_SCHEMA_PATH = ROOT / "optics_sft" / "data_schema" / "physics_sft_sample_schema.json"

CONTROL_KEYS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
DEFAULT_QUALITY_GATES = {
    "schema_error_count_max": 0,
    "leakage_failure_count_max": 0,
    "split_sample_id_overlap_max": 0,
    "duplicate_setup_hash_rate_max": 0.02,
    "label_improvement_rate_min": 0.90,
    "control_within_bounds_rate_min": 0.98,
    "confidence_in_range_rate_min": 0.99,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            rows.append(row)
    return rows


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_schema_validator() -> tuple[Any | None, str]:
    try:
        import jsonschema
    except ImportError:
        return None, "jsonschema_not_installed"

    schema = json.loads(PHYSICS_SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema), "enabled"


def validate_text_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    required_top_level = ("sample_id", "sample_type", "modality", "prompt_inputs", "target", "messages")
    examples: list[dict[str, str]] = []
    error_count = 0
    for row in rows:
        row_errors: list[str] = []
        for key in required_top_level:
            if key not in row:
                row_errors.append(f"missing top-level field {key}")
        if row.get("modality") != "text":
            row_errors.append("modality must be 'text'")
        prompt_inputs = row.get("prompt_inputs")
        if isinstance(prompt_inputs, Mapping):
            if not isinstance(prompt_inputs.get("safe_setup_metadata"), Mapping):
                row_errors.append("prompt_inputs.safe_setup_metadata must be an object")
            observations = prompt_inputs.get("text_observations")
            if not isinstance(observations, Mapping):
                row_errors.append("prompt_inputs.text_observations must be an object")
        error_count += len(row_errors)
        for message in row_errors[: max(0, 5 - len(examples))]:
            examples.append(
                {
                    "sample_id": str(row.get("sample_id", "")),
                    "sample_type": str(row.get("sample_type", "")),
                    "path": "",
                    "message": message,
                }
            )
    return {"status": "text_rows", "error_count": error_count, "examples": examples}


def validate_schema_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    physics_rows = [row for row in rows if row.get("modality") != "text"]
    text_rows = [row for row in rows if row.get("modality") == "text"]

    physics_result = {"status": "skipped", "error_count": 0, "examples": []}
    if physics_rows:
        validator, status = load_schema_validator()
        if validator is None:
            physics_result = {"status": status, "error_count": 0, "examples": []}
        else:
            examples: list[dict[str, str]] = []
            error_count = 0
            for row in physics_rows:
                row_errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
                error_count += len(row_errors)
                for error in row_errors[: max(0, 5 - len(examples))]:
                    examples.append(
                        {
                            "sample_id": str(row.get("sample_id", "")),
                            "sample_type": str(row.get("sample_type", "")),
                            "path": ".".join(str(part) for part in error.path),
                            "message": error.message,
                        }
                    )
            physics_result = {"status": status, "error_count": error_count, "examples": examples}

    text_result = validate_text_rows(text_rows) if text_rows else {"status": "skipped", "error_count": 0, "examples": []}
    total_errors = int(physics_result["error_count"]) + int(text_result["error_count"])
    return {
        "status": "mixed" if physics_rows and text_rows else physics_result["status"],
        "error_count": total_errors,
        "physics": physics_result,
        "text": text_result,
        "examples": (physics_result.get("examples", []) + text_result.get("examples", []))[:5],
    }


def audit_leakage_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    from optics_sft.physics.prompt_builder import assert_prompt_inputs_safe

    failures: list[str] = []
    for row in rows:
        try:
            assert_prompt_inputs_safe(row)
        except ValueError as exc:
            failures.append(f"{row.get('sample_id', '<missing-id>')}: {exc}")
    return {"failure_count": len(failures), "examples": failures[:10]}


def split_overlap(train_rows: list[Mapping[str, Any]], val_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    train_ids = {str(row.get("sample_id")) for row in train_rows if row.get("sample_id")}
    val_ids = {str(row.get("sample_id")) for row in val_rows if row.get("sample_id")}
    overlap = sorted(train_ids & val_ids)
    return {
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "train_unique_sample_ids": len(train_ids),
        "val_unique_sample_ids": len(val_ids),
        "sample_id_overlap_count": len(overlap),
        "sample_id_overlap_examples": overlap[:10],
    }


def _state_vector(state: Mapping[str, Any] | None) -> dict[str, float]:
    if not isinstance(state, Mapping):
        return {}
    values: dict[str, float] = {}
    for key in ("centroid_x_px", "centroid_y_px", "sigma_x_px", "sigma_y_px", "peak_intensity"):
        candidate = state.get(key)
        if is_number(candidate):
            values[key] = round(float(candidate), 4)
            continue
        nested = state.get("centroid_px")
        if key.startswith("centroid_") and isinstance(nested, Mapping):
            axis = key.split("_", 1)[1]
            nested_value = nested.get(axis[0])
            if is_number(nested_value):
                values[key] = round(float(nested_value), 4)
    return values


def setup_hash(row: Mapping[str, Any]) -> str:
    prompt_inputs = row.get("prompt_inputs")
    payload: dict[str, Any] = {}
    if isinstance(prompt_inputs, Mapping):
        metadata = prompt_inputs.get("safe_setup_metadata")
        if isinstance(metadata, Mapping):
            payload["safe_setup_metadata"] = metadata
        observations = prompt_inputs.get("text_observations")
        if isinstance(observations, Mapping):
            payload["text_observations"] = observations
    private_eval = row.get("private_eval")
    if isinstance(private_eval, Mapping):
        payload["current_state"] = _state_vector(private_eval.get("current_state"))
        payload["target_state"] = _state_vector(private_eval.get("target_state"))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def duplicate_setup_stats(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    hashes = [setup_hash(row) for row in rows]
    counts = Counter(hashes)
    duplicate_groups = sum(1 for count in counts.values() if count > 1)
    duplicate_rows = sum(count - 1 for count in counts.values() if count > 1)
    total = len(hashes)
    return {
        "row_count": total,
        "unique_setup_hash_count": len(counts),
        "duplicate_group_count": duplicate_groups,
        "duplicate_row_count": duplicate_rows,
        "duplicate_setup_hash_rate": duplicate_rows / total if total else 0.0,
    }


def _control_plan(row: Mapping[str, Any]) -> dict[str, Any] | None:
    target = row.get("target")
    if isinstance(target, Mapping) and isinstance(target.get("control_plan"), Mapping):
        return target["control_plan"]
    label = row.get("label")
    if isinstance(label, Mapping) and isinstance(label.get("control_plan"), Mapping):
        return label["control_plan"]
    return None


def _actuator_limits(row: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    prompt_inputs = row.get("prompt_inputs")
    if not isinstance(prompt_inputs, Mapping):
        return {}
    metadata = prompt_inputs.get("safe_setup_metadata")
    if not isinstance(metadata, Mapping):
        return {}
    limits_obj = metadata.get("actuator_limits")
    if not isinstance(limits_obj, Mapping):
        return {}
    limits: dict[str, tuple[float, float]] = {}
    for key in CONTROL_KEYS:
        spec = limits_obj.get(key)
        if isinstance(spec, Mapping) and is_number(spec.get("min")) and is_number(spec.get("max")):
            limits[key] = (float(spec["min"]), float(spec["max"]))
    return limits


def label_sanity(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    checked = 0
    improved = 0
    within_bounds = 0
    bounds_checked = 0
    confidence_ok = 0
    confidence_checked = 0
    missing_private_eval = 0
    examples: list[dict[str, Any]] = []

    for row in rows:
        sample_type = row.get("sample_type")
        if sample_type != "inverse_control":
            continue
        checked += 1
        private_eval = row.get("private_eval")
        if not isinstance(private_eval, Mapping):
            missing_private_eval += 1
            continue

        initial = private_eval.get("initial_error_norm_px")
        post = private_eval.get("post_action_error_norm_px")
        if is_number(initial) and is_number(post) and float(post) <= float(initial) + 1e-6:
            improved += 1
        elif len(examples) < 5:
            examples.append(
                {
                    "sample_id": row.get("sample_id"),
                    "issue": "post_action_error_not_improved",
                    "initial_error_norm_px": initial,
                    "post_action_error_norm_px": post,
                }
            )

        plan = _control_plan(row)
        limits = _actuator_limits(row)
        if isinstance(plan, Mapping) and limits:
            bounds_checked += 1
            in_bounds = True
            for key, (low, high) in limits.items():
                value = plan.get(key)
                if not is_number(value) or float(value) < low or float(value) > high:
                    in_bounds = False
                    break
            if in_bounds:
                within_bounds += 1
            elif len(examples) < 5:
                examples.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "issue": "control_plan_out_of_bounds",
                        "control_plan": plan,
                        "limits": limits,
                    }
                )

        target = row.get("target")
        confidence = target.get("confidence") if isinstance(target, Mapping) else None
        if is_number(confidence):
            confidence_checked += 1
            if 0.0 <= float(confidence) <= 1.0:
                confidence_ok += 1

    return {
        "inverse_control_rows_checked": checked,
        "missing_private_eval_count": missing_private_eval,
        "label_improvement_rate": improved / checked if checked else None,
        "control_within_bounds_rate": within_bounds / bounds_checked if bounds_checked else None,
        "confidence_in_range_rate": confidence_ok / confidence_checked if confidence_checked else None,
        "examples": examples,
    }


def distribution_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    sample_types = Counter(str(row.get("sample_type", "unknown")) for row in rows)
    initial_errors: list[float] = []
    post_errors: list[float] = []
    control_magnitudes: list[float] = []

    for row in rows:
        private_eval = row.get("private_eval")
        if isinstance(private_eval, Mapping):
            initial = private_eval.get("initial_error_norm_px")
            post = private_eval.get("post_action_error_norm_px")
            if is_number(initial):
                initial_errors.append(float(initial))
            if is_number(post):
                post_errors.append(float(post))
        plan = _control_plan(row)
        if isinstance(plan, Mapping):
            lens_mag = 0.0
            for key in ("lens_x_delta_mm", "lens_y_delta_mm"):
                value = plan.get(key)
                if is_number(value):
                    lens_mag += abs(float(value))
            control_magnitudes.append(lens_mag)

    def summarize(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p90": None}
        ordered = sorted(values)
        return {
            "count": len(values),
            "min": ordered[0],
            "max": ordered[-1],
            "mean": statistics.fmean(values),
            "p50": ordered[len(ordered) // 2],
            "p90": ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))],
        }

    return {
        "sample_type_counts": dict(sample_types),
        "initial_error_norm_px": summarize(initial_errors),
        "post_action_error_norm_px": summarize(post_errors),
        "lens_control_magnitude_mm": summarize(control_magnitudes),
    }


def text_prompt_stats(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    for row in rows:
        messages = row.get("messages")
        if isinstance(messages, list) and len(messages) >= 2:
            user = messages[0].get("content") if isinstance(messages[0], Mapping) else None
            assistant = messages[1].get("content") if isinstance(messages[1], Mapping) else None
            if isinstance(user, str):
                prompt_lengths.append(len(user))
            if isinstance(assistant, str):
                completion_lengths.append(len(assistant))
            continue
        user_prompt = row.get("user_prompt")
        assistant_completion = row.get("assistant_completion")
        if isinstance(user_prompt, str):
            prompt_lengths.append(len(user_prompt))
        if isinstance(assistant_completion, str):
            completion_lengths.append(len(assistant_completion))

    def summarize(values: list[int]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
        }

    return {
        "user_prompt_char_length": summarize(prompt_lengths),
        "assistant_completion_char_length": summarize(completion_lengths),
    }


def evaluate_gates(report: Mapping[str, Any], gates: Mapping[str, float | int] | None = None) -> dict[str, Any]:
    thresholds = dict(DEFAULT_QUALITY_GATES)
    if gates:
        thresholds.update(gates)

    schema = report.get("schema_validation", {})
    leakage = report.get("leakage_audit", {})
    overlap = report.get("split_overlap", {})
    duplicates = report.get("duplicate_setup_stats", {})
    sanity = report.get("label_sanity", {})

    checks = {
        "schema_valid": int(schema.get("error_count", 0)) <= int(thresholds["schema_error_count_max"]),
        "no_leakage": int(leakage.get("failure_count", 0)) <= int(thresholds["leakage_failure_count_max"]),
        "no_split_overlap": int(overlap.get("sample_id_overlap_count", 0))
        <= int(thresholds["split_sample_id_overlap_max"]),
        "duplicate_setup_hash_rate_ok": float(duplicates.get("duplicate_setup_hash_rate", 0.0))
        <= float(thresholds["duplicate_setup_hash_rate_max"]),
        "label_improvement_rate_ok": _rate_at_least(
            sanity.get("label_improvement_rate"),
            float(thresholds["label_improvement_rate_min"]),
        ),
        "control_within_bounds_rate_ok": _rate_at_least_or_missing(
            sanity.get("control_within_bounds_rate"),
            float(thresholds["control_within_bounds_rate_min"]),
        ),
        "confidence_in_range_rate_ok": _rate_at_least(
            sanity.get("confidence_in_range_rate"),
            float(thresholds["confidence_in_range_rate_min"]),
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
    }


def _rate_at_least(value: Any, minimum: float) -> bool:
    if value is None:
        return False
    if not is_number(value):
        return False
    return float(value) >= minimum


def _rate_at_least_or_missing(value: Any, minimum: float) -> bool:
    if value is None:
        return True
    return _rate_at_least(value, minimum)


def quality_gate_passed(report: Mapping[str, Any], gates: Mapping[str, float | int] | None = None) -> bool:
    return bool(evaluate_gates(report, gates)["passed"])


def build_quality_report(
    *,
    train_jsonl: Path | None = None,
    val_jsonl: Path | None = None,
    extra_jsonls: Mapping[str, Path] | None = None,
    dataset_name: str = "unknown",
    gates: Mapping[str, float | int] | None = None,
) -> dict[str, Any]:
    splits: dict[str, list[dict[str, Any]]] = {}
    if train_jsonl is not None:
        splits["train"] = read_jsonl(train_jsonl)
    if val_jsonl is not None:
        splits["val"] = read_jsonl(val_jsonl)
    if extra_jsonls:
        for split_name, path in extra_jsonls.items():
            splits[split_name] = read_jsonl(path)

    all_rows = [row for rows in splits.values() for row in rows]
    report: dict[str, Any] = {
        "dataset_name": dataset_name,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "schema_validation": validate_schema_rows(all_rows),
        "leakage_audit": audit_leakage_rows(all_rows),
        "duplicate_setup_stats": duplicate_setup_stats(all_rows),
        "label_sanity": label_sanity(all_rows),
        "distribution_summary": distribution_summary(all_rows),
        "text_prompt_stats": text_prompt_stats(all_rows),
    }
    if "train" in splits and "val" in splits:
        report["split_overlap"] = split_overlap(splits["train"], splits["val"])
    report["quality_gates"] = evaluate_gates(report, gates)
    return report


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
