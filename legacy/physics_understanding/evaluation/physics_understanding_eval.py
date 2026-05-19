"""Evaluation for physics-understanding diagnostic probe predictions."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from legacy.physics_understanding.evaluation.physics_understanding_schema import (
    CANONICAL_VARIABLE_ORDER,
    INPUT_MODES,
    load_probe_jsonl,
)
from profile2setup.llm_api.validator import validate_all_variable_dicts, validate_llm_output
from profile2setup.training.normalization import load_variables_config

CHANGE_DIRECTIONS = {"increase", "decrease", "unchanged"}
NOT_APPLICABLE_STATUSES = {"not_applicable", "skipped", "dry_run"}
FALLBACK_TOLERANCES = {
    "source_to_lens": 0.01,
    "lens_to_camera": 0.01,
    "focal_length": 0.005,
    "lens_x": 0.0005,
    "lens_y": 0.0005,
    "camera_x": 0.0005,
    "camera_y": 0.0005,
}


def _load_tolerances(variables_config: dict) -> dict[str, float]:
    variables = variables_config.get("variables") or {}
    tolerances: dict[str, float] = {}
    for name in CANONICAL_VARIABLE_ORDER:
        spec = variables.get(name) or {}
        tolerances[name] = float(spec.get("tolerance", FALLBACK_TOLERANCES[name]))
    return tolerances


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} row must be a JSON object")
            rows.append(row)
    return rows


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _prediction_rows_by_probe_id(rows: list[dict]) -> tuple[dict[str, dict], list[str]]:
    out: dict[str, dict] = {}
    warnings: list[str] = []
    for index, row in enumerate(rows, start=1):
        probe_id = row.get("probe_id") or row.get("record_id")
        if not isinstance(probe_id, str) or not probe_id:
            warnings.append(f"prediction row {index} missing probe_id")
            continue
        if probe_id in out:
            warnings.append(f"duplicate prediction for probe_id={probe_id}; using the last row")
        out[probe_id] = row
    return out, warnings


def _parse_prediction(row: dict | None) -> tuple[dict | None, bool]:
    if row is None:
        return None, False
    prediction = row.get("prediction")
    if isinstance(prediction, dict):
        return prediction, True
    raw = row.get("raw_response")
    if not isinstance(raw, str) or not raw.strip():
        return None, False
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None, False
    return (parsed, True) if isinstance(parsed, dict) else (None, False)


def _row_is_format_applicable(row: dict | None) -> bool:
    if row is None:
        return False
    status = row.get("status")
    if status in NOT_APPLICABLE_STATUSES:
        return False
    return True


def _canonical_ok(parsed: dict | None) -> bool:
    if parsed is None:
        return False
    try:
        validate_all_variable_dicts(parsed)
    except Exception:
        return False
    return True


def _schema_ok(parsed: dict | None) -> bool:
    if parsed is None:
        return False
    try:
        validate_llm_output(parsed)
    except Exception:
        return False
    return True


def _clean_delta(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        return {name: float(value[name]) for name in CANONICAL_VARIABLE_ORDER}
    except Exception:
        return None


def _direction_from_delta(delta: dict[str, float], tolerances: dict[str, float]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in CANONICAL_VARIABLE_ORDER:
        value = float(delta[name])
        tolerance = float(tolerances[name])
        if value > tolerance:
            out[name] = "increase"
        elif value < -tolerance:
            out[name] = "decrease"
        else:
            out[name] = "unchanged"
    return out


def _direction_from_understanding(prediction: dict) -> dict[str, str] | None:
    understanding = prediction.get("setup_understanding")
    if not isinstance(understanding, dict):
        return None
    value = understanding.get("change_direction")
    if not isinstance(value, dict):
        return None
    out: dict[str, str] = {}
    for name in CANONICAL_VARIABLE_ORDER:
        direction = value.get(name)
        if direction not in CHANGE_DIRECTIONS:
            return None
        out[name] = str(direction)
    return out


def _prediction_direction(prediction: dict | None, tolerances: dict[str, float]) -> dict[str, str] | None:
    if prediction is None:
        return None
    delta = _clean_delta(prediction.get("predicted_delta"))
    if delta is not None:
        return _direction_from_delta(delta, tolerances)
    return _direction_from_understanding(prediction)


def _changed_from_direction(direction: dict[str, str] | None) -> set[str] | None:
    if direction is None:
        return None
    return {
        name for name in CANONICAL_VARIABLE_ORDER if direction.get(name) != "unchanged"
    }


def _changed_from_prediction(prediction: dict | None, tolerances: dict[str, float]) -> set[str] | None:
    direction = _prediction_direction(prediction, tolerances)
    if direction is not None:
        return _changed_from_direction(direction)
    if prediction is None:
        return None
    understanding = prediction.get("setup_understanding")
    if not isinstance(understanding, dict):
        return None
    changed = understanding.get("changed_variables")
    if not isinstance(changed, list):
        return None
    return {name for name in changed if name in CANONICAL_VARIABLE_ORDER}


def _has_full_setup(prediction: dict | None) -> bool:
    if prediction is None:
        return False
    setup = prediction.get("predicted_setup")
    if not isinstance(setup, dict):
        return False
    try:
        return all(name in setup and isinstance(float(setup[name]), float) for name in CANONICAL_VARIABLE_ORDER)
    except Exception:
        return False


def _rejection_reason(prediction: dict | None) -> str:
    if prediction is None:
        return ""
    value = prediction.get("rejection_reason")
    return value if isinstance(value, str) else ""


def _keyword_hit(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(str(keyword).lower() in lowered for keyword in keywords)


def _is_rejected_or_keyword(prediction: dict | None, keywords: list[str]) -> bool:
    if prediction is None:
        return False
    if prediction.get("valid") is False:
        return True
    return _keyword_hit(_rejection_reason(prediction), keywords)


def _expected_direction_items(probe: dict) -> list[tuple[str, str]]:
    expected = probe.get("expected_change_direction")
    if not isinstance(expected, dict):
        return []
    return [
        (name, expected[name])
        for name in CANONICAL_VARIABLE_ORDER
        if expected.get(name) in CHANGE_DIRECTIONS
    ]


def _fixed_variable_violations(probe: dict, direction: dict[str, str] | None) -> tuple[int, int]:
    if direction is None:
        return 0, 0
    fixed = probe.get("fixed_variables")
    if not isinstance(fixed, list):
        return 0, 0
    total = 0
    violations = 0
    for name in fixed:
        if name not in CANONICAL_VARIABLE_ORDER:
            continue
        total += 1
        if direction.get(name) != "unchanged":
            violations += 1
    return violations, total


def _allowed_variable_violation(probe: dict, changed: set[str] | None) -> tuple[int, int]:
    allowed = probe.get("allowed_variables")
    if allowed is None:
        return 0, 0
    if changed is None:
        return 0, 0
    allowed_set = {name for name in allowed if name in CANONICAL_VARIABLE_ORDER}
    return (1 if any(name not in allowed_set for name in changed) else 0, 1)


def _signature(direction: dict[str, str] | None) -> tuple[str, ...] | None:
    if direction is None:
        return None
    return tuple(direction[name] for name in CANONICAL_VARIABLE_ORDER)


def _probe_success(probe: dict, parsed: dict | None, tolerances: dict[str, float]) -> bool | None:
    if parsed is None:
        return False
    if probe.get("expected_valid") is False:
        return _is_rejected_or_keyword(parsed, probe.get("expected_rejection_keywords") or [])
    if parsed.get("valid") is not True:
        return False
    direction = _prediction_direction(parsed, tolerances)
    changed = _changed_from_direction(direction)
    fixed_violations, fixed_total = _fixed_variable_violations(probe, direction)
    if fixed_total and fixed_violations:
        return False
    allowed_violation, allowed_total = _allowed_variable_violation(probe, changed)
    if allowed_total and allowed_violation:
        return False
    expected_items = _expected_direction_items(probe)
    if expected_items and direction is None:
        return False
    for name, expected in expected_items:
        if direction is None or direction.get(name) != expected:
            return False
    return True


def _empty_running_counts() -> dict[str, int]:
    return {
        "rows": 0,
        "format_denominator": 0,
        "valid_json": 0,
        "schema_valid": 0,
        "canonical": 0,
        "probe_success": 0,
        "probe_success_denominator": 0,
        "fixed_violations": 0,
        "fixed_total": 0,
        "allowed_violations": 0,
        "allowed_total": 0,
        "direction_correct": 0,
        "direction_total": 0,
        "forced_predictions": 0,
        "forced_total": 0,
        "contradiction_correct": 0,
        "contradiction_total": 0,
        "conflict_correct": 0,
        "conflict_total": 0,
    }


def _counts_to_metrics(counts: dict[str, int]) -> dict[str, Any]:
    return {
        "rows": int(counts["rows"]),
        "format_denominator": int(counts["format_denominator"]),
        "valid_json_rate": _rate(counts["valid_json"], counts["format_denominator"]),
        "schema_valid_rate": _rate(counts["schema_valid"], counts["format_denominator"]),
        "canonical_variable_rate": _rate(counts["canonical"], counts["format_denominator"]),
        "success_rate": _rate(counts["probe_success"], counts["probe_success_denominator"]),
        "success_count": int(counts["probe_success"]),
        "success_denominator": int(counts["probe_success_denominator"]),
        "fixed_variable_violation_rate": _rate(counts["fixed_violations"], counts["fixed_total"]),
        "fixed_variable_violations": int(counts["fixed_violations"]),
        "fixed_variable_total": int(counts["fixed_total"]),
        "allowed_variable_violation_rate": _rate(counts["allowed_violations"], counts["allowed_total"]),
        "allowed_variable_violations": int(counts["allowed_violations"]),
        "allowed_variable_total": int(counts["allowed_total"]),
        "direction_accuracy": _rate(counts["direction_correct"], counts["direction_total"]),
        "direction_correct": int(counts["direction_correct"]),
        "direction_total": int(counts["direction_total"]),
        "forced_prediction_rate": _rate(counts["forced_predictions"], counts["forced_total"]),
        "forced_predictions": int(counts["forced_predictions"]),
        "forced_prediction_total": int(counts["forced_total"]),
        "contradiction_detection_accuracy": _rate(
            counts["contradiction_correct"], counts["contradiction_total"]
        ),
        "contradiction_correct": int(counts["contradiction_correct"]),
        "contradiction_total": int(counts["contradiction_total"]),
        "prompt_image_conflict_detection_accuracy": _rate(
            counts["conflict_correct"], counts["conflict_total"]
        ),
        "conflict_correct": int(counts["conflict_correct"]),
        "conflict_total": int(counts["conflict_total"]),
    }


def _semantic_key(probe: dict) -> tuple[Any, ...]:
    expected_changed = probe.get("expected_changed_variables")
    expected_direction = probe.get("expected_change_direction")
    changed_key = tuple(expected_changed) if isinstance(expected_changed, list) else None
    if isinstance(expected_direction, dict):
        direction_key = tuple(
            (name, expected_direction.get(name))
            for name in CANONICAL_VARIABLE_ORDER
        )
    else:
        direction_key = None
    return probe.get("base_record_id"), changed_key, direction_key


def _paraphrase_consistency(
    probes: list[dict],
    parsed_by_probe_id: dict[str, dict | None],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict]] = defaultdict(list)
    for probe in probes:
        if probe.get("probe_type") == "paraphrase_consistency":
            groups[_semantic_key(probe)].append(probe)
    group_scores: list[float] = []
    row_weighted_correct = 0
    row_weighted_total = 0
    scored_groups = 0
    for group in groups.values():
        signatures: list[tuple[str, ...]] = []
        for probe in group:
            parsed = parsed_by_probe_id.get(probe["probe_id"])
            sig = _signature(_prediction_direction(parsed, tolerances))
            if sig is not None:
                signatures.append(sig)
        if len(signatures) < 2:
            continue
        counts = Counter(signatures)
        modal = max(counts.values())
        score = modal / len(signatures)
        group_scores.append(float(score))
        row_weighted_correct += modal
        row_weighted_total += len(signatures)
        scored_groups += 1
    return {
        "score": (sum(group_scores) / len(group_scores)) if group_scores else None,
        "row_weighted_score": _rate(row_weighted_correct, row_weighted_total),
        "groups": int(scored_groups),
        "rows": int(row_weighted_total),
    }


def _prompt_sensitivity(
    probes: list[dict],
    parsed_by_probe_id: dict[str, dict | None],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for probe in probes:
        if probe.get("probe_type") == "prompt_sensitivity" and isinstance(probe.get("base_record_id"), str):
            groups[probe["base_record_id"]].append(probe)
    successes = 0
    scored = 0
    details: list[dict] = []
    for base_record_id, group in groups.items():
        if len(group) < 2:
            continue
        signatures = []
        constraint_results = []
        for probe in group:
            parsed = parsed_by_probe_id.get(probe["probe_id"])
            direction = _prediction_direction(parsed, tolerances)
            signatures.append(_signature(direction))
            fixed_v, fixed_t = _fixed_variable_violations(probe, direction)
            changed = _changed_from_direction(direction)
            allowed_v, allowed_t = _allowed_variable_violation(probe, changed)
            constraint_results.append((fixed_v == 0 if fixed_t else True) and (allowed_v == 0 if allowed_t else True))
        scorable_signatures = [sig for sig in signatures if sig is not None]
        if len(scorable_signatures) < 2:
            continue
        differs = len(set(scorable_signatures)) > 1
        constraints_ok = all(constraint_results)
        group_success = bool(differs and constraints_ok)
        successes += int(group_success)
        scored += 1
        if len(details) < 20:
            details.append(
                {
                    "base_record_id": base_record_id,
                    "success": group_success,
                    "outputs_differ": differs,
                    "constraints_ok": constraints_ok,
                }
            )
    return {
        "score": _rate(successes, scored),
        "successful_groups": int(successes),
        "groups": int(scored),
        "examples": details,
    }


def _opposite_direction(direction: str) -> str | None:
    if direction == "increase":
        return "decrease"
    if direction == "decrease":
        return "increase"
    return None


def _direction_flip_accuracy(
    probes: list[dict],
    parsed_by_probe_id: dict[str, dict | None],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    by_base: dict[str, list[dict]] = defaultdict(list)
    for probe in probes:
        if isinstance(probe.get("base_record_id"), str) and isinstance(probe.get("expected_change_direction"), dict):
            by_base[probe["base_record_id"]].append(probe)

    correct = 0
    total = 0
    pairs = 0
    for group in by_base.values():
        for i, left in enumerate(group):
            left_expected = left.get("expected_change_direction")
            if not isinstance(left_expected, dict):
                continue
            for right in group[i + 1 :]:
                right_expected = right.get("expected_change_direction")
                if not isinstance(right_expected, dict):
                    continue
                flipped_vars = []
                for name in CANONICAL_VARIABLE_ORDER:
                    opposite = _opposite_direction(str(left_expected.get(name)))
                    if opposite is not None and right_expected.get(name) == opposite:
                        flipped_vars.append(name)
                if not flipped_vars:
                    continue
                left_pred = _prediction_direction(parsed_by_probe_id.get(left["probe_id"]), tolerances)
                right_pred = _prediction_direction(parsed_by_probe_id.get(right["probe_id"]), tolerances)
                if left_pred is None or right_pred is None:
                    continue
                pairs += 1
                for name in flipped_vars:
                    total += 1
                    predicted_opposite = _opposite_direction(left_pred.get(name, "unchanged"))
                    correct += int(predicted_opposite is not None and right_pred.get(name) == predicted_opposite)
    return {
        "accuracy": _rate(correct, total),
        "correct": int(correct),
        "total": int(total),
        "pairs": int(pairs),
    }


def _accumulate_counts(
    counts: dict[str, int],
    *,
    probe: dict,
    row: dict | None,
    parsed: dict | None,
    json_valid: bool,
    schema_valid: bool,
    canonical_valid: bool,
    tolerances: dict[str, float],
) -> None:
    counts["rows"] += 1
    if _row_is_format_applicable(row):
        counts["format_denominator"] += 1
        counts["valid_json"] += int(json_valid)
        counts["schema_valid"] += int(schema_valid)
        counts["canonical"] += int(canonical_valid)

    success = _probe_success(probe, parsed, tolerances)
    if row is not None and row.get("status") not in NOT_APPLICABLE_STATUSES:
        counts["probe_success_denominator"] += 1
        counts["probe_success"] += int(bool(success))

    direction = _prediction_direction(parsed, tolerances)
    fixed_v, fixed_t = _fixed_variable_violations(probe, direction)
    counts["fixed_violations"] += fixed_v
    counts["fixed_total"] += fixed_t

    changed = _changed_from_direction(direction)
    allowed_v, allowed_t = _allowed_variable_violation(probe, changed)
    counts["allowed_violations"] += allowed_v
    counts["allowed_total"] += allowed_t

    for name, expected_direction in _expected_direction_items(probe):
        if direction is not None:
            counts["direction_total"] += 1
            counts["direction_correct"] += int(direction.get(name) == expected_direction)

    if probe.get("expected_valid") is False and row is not None and row.get("status") not in NOT_APPLICABLE_STATUSES:
        counts["forced_total"] += 1
        counts["forced_predictions"] += int(parsed is not None and parsed.get("valid") is True and _has_full_setup(parsed))

    if probe.get("probe_type") == "contradiction_detection" and probe.get("expected_valid") is False:
        if row is not None and row.get("status") not in NOT_APPLICABLE_STATUSES:
            counts["contradiction_total"] += 1
            counts["contradiction_correct"] += int(
                _is_rejected_or_keyword(parsed, probe.get("expected_rejection_keywords") or [])
            )

    if probe.get("probe_type") == "prompt_image_conflict":
        if row is not None and row.get("status") not in NOT_APPLICABLE_STATUSES:
            counts["conflict_total"] += 1
            counts["conflict_correct"] += int(
                _is_rejected_or_keyword(parsed, probe.get("expected_rejection_keywords") or [])
            )


def evaluate_physics_understanding(
    *,
    predictions_path,
    probes_path,
    out_path=None,
    variables_config_path="profile2setup/configs/variables.yaml",
    markdown_out_path=None,
) -> dict[str, Any]:
    predictions_file = Path(predictions_path)
    probes_file = Path(probes_path)
    variables_config = load_variables_config(variables_config_path)
    tolerances = _load_tolerances(variables_config)
    probes = load_probe_jsonl(probes_file)
    prediction_rows = _load_jsonl(predictions_file)
    predictions_by_id, warnings = _prediction_rows_by_probe_id(prediction_rows)

    parsed_by_probe_id: dict[str, dict | None] = {}
    row_by_probe_id: dict[str, dict | None] = {}
    overall_counts = _empty_running_counts()
    by_probe_type: dict[str, dict[str, int]] = defaultdict(_empty_running_counts)
    by_input_mode: dict[str, dict[str, int]] = defaultdict(_empty_running_counts)
    status_counts = Counter(str(row.get("status")) for row in prediction_rows)
    missing_predictions = 0

    for probe in probes:
        row = predictions_by_id.get(probe["probe_id"])
        if row is None:
            missing_predictions += 1
            warnings.append(f"missing prediction for probe_id={probe['probe_id']}")
        parsed, json_valid = _parse_prediction(row)
        schema_valid = _schema_ok(parsed)
        canonical_valid = _canonical_ok(parsed) if json_valid else False
        parsed_by_probe_id[probe["probe_id"]] = parsed
        row_by_probe_id[probe["probe_id"]] = row

        kwargs = {
            "probe": probe,
            "row": row,
            "parsed": parsed,
            "json_valid": json_valid,
            "schema_valid": schema_valid,
            "canonical_valid": canonical_valid,
            "tolerances": tolerances,
        }
        _accumulate_counts(overall_counts, **kwargs)
        _accumulate_counts(by_probe_type[probe["probe_type"]], **kwargs)
        _accumulate_counts(by_input_mode[probe["input_mode"]], **kwargs)

    per_probe_type_count = Counter(probe["probe_type"] for probe in probes)
    per_probe_type_metrics = {
        probe_type: _counts_to_metrics(by_probe_type[probe_type])
        for probe_type in sorted(per_probe_type_count)
    }
    per_probe_type_success_rate = {
        probe_type: per_probe_type_metrics[probe_type]["success_rate"]
        for probe_type in sorted(per_probe_type_count)
    }
    input_ablation_summary = {
        input_mode: _counts_to_metrics(by_input_mode[input_mode])
        for input_mode in sorted(INPUT_MODES)
    }

    report = {
        "predictions_path": str(predictions_file),
        "probes_path": str(probes_file),
        "variables_config_path": str(variables_config_path),
        "prediction_rows": int(len(prediction_rows)),
        "probe_rows": int(len(probes)),
        "matched_probe_rows": int(len(probes) - missing_predictions),
        "missing_prediction_rows": int(missing_predictions),
        "status_counts": dict(sorted(status_counts.items())),
        "warnings": warnings[:100],
        "format_metrics": {
            "valid_json_rate": _counts_to_metrics(overall_counts)["valid_json_rate"],
            "schema_valid_rate": _counts_to_metrics(overall_counts)["schema_valid_rate"],
            "canonical_variable_rate": _counts_to_metrics(overall_counts)["canonical_variable_rate"],
            "format_denominator": int(overall_counts["format_denominator"]),
            "valid_json_count": int(overall_counts["valid_json"]),
            "schema_valid_count": int(overall_counts["schema_valid"]),
            "canonical_variable_count": int(overall_counts["canonical"]),
        },
        "probe_level_metrics": {
            "per_probe_type_count": dict(sorted(per_probe_type_count.items())),
            "per_probe_type_success_rate": per_probe_type_success_rate,
            "per_probe_type_metrics": per_probe_type_metrics,
        },
        "language_physics_metrics": {
            "paraphrase_consistency_score": _paraphrase_consistency(
                probes, parsed_by_probe_id, tolerances
            ),
            "prompt_sensitivity_score": _prompt_sensitivity(
                probes, parsed_by_probe_id, tolerances
            ),
            "fixed_variable_violation_rate": _counts_to_metrics(overall_counts)[
                "fixed_variable_violation_rate"
            ],
            "fixed_variable_violations": int(overall_counts["fixed_violations"]),
            "fixed_variable_total": int(overall_counts["fixed_total"]),
            "allowed_variable_violation_rate": _counts_to_metrics(overall_counts)[
                "allowed_variable_violation_rate"
            ],
            "allowed_variable_violations": int(overall_counts["allowed_violations"]),
            "allowed_variable_total": int(overall_counts["allowed_total"]),
            "contradiction_detection_accuracy": _counts_to_metrics(overall_counts)[
                "contradiction_detection_accuracy"
            ],
            "contradiction_correct": int(overall_counts["contradiction_correct"]),
            "contradiction_total": int(overall_counts["contradiction_total"]),
            "prompt_image_conflict_detection_accuracy": _counts_to_metrics(overall_counts)[
                "prompt_image_conflict_detection_accuracy"
            ],
            "conflict_correct": int(overall_counts["conflict_correct"]),
            "conflict_total": int(overall_counts["conflict_total"]),
            "direction_accuracy": _counts_to_metrics(overall_counts)["direction_accuracy"],
            "direction_correct": int(overall_counts["direction_correct"]),
            "direction_total": int(overall_counts["direction_total"]),
            "direction_flip_accuracy": _direction_flip_accuracy(
                probes, parsed_by_probe_id, tolerances
            ),
            "forced_prediction_rate": _counts_to_metrics(overall_counts)["forced_prediction_rate"],
            "forced_predictions": int(overall_counts["forced_predictions"]),
            "forced_prediction_total": int(overall_counts["forced_total"]),
            "input_ablation_summary": input_ablation_summary,
        },
    }

    if out_path is not None:
        out = Path(out_path)
        _save_json(report, out)
        markdown_path = Path(markdown_out_path) if markdown_out_path else out.with_suffix(".md")
        write_physics_understanding_markdown(report, markdown_path)
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metric_sentence(name: str, value: Any, *, high_is_good: bool = True) -> str:
    if value is None:
        return f"{name}: not scored for this prediction set."
    direction = "higher is better" if high_is_good else "lower is better"
    return f"{name}: `{_fmt(value)}` ({direction})."


def write_physics_understanding_markdown(report: dict[str, Any], path) -> None:
    """Write a concise Markdown interpretation of a physics-understanding report."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = report["format_metrics"]
    phys = report["language_physics_metrics"]
    probe_level = report["probe_level_metrics"]
    lines = [
        "# Physics Understanding Evaluation",
        "",
        f"- Predictions: `{report['predictions_path']}`",
        f"- Probes: `{report['probes_path']}`",
        f"- Prediction rows: `{report['prediction_rows']}`",
        f"- Probe rows: `{report['probe_rows']}`",
        f"- Matched probe rows: `{report['matched_probe_rows']}`",
        "",
        "## Format",
        "",
        f"- valid_json_rate: `{_fmt(fmt['valid_json_rate'])}`",
        f"- schema_valid_rate: `{_fmt(fmt['schema_valid_rate'])}`",
        f"- canonical_variable_rate: `{_fmt(fmt['canonical_variable_rate'])}`",
        f"- format denominator: `{fmt['format_denominator']}`",
        "",
        "Interpretation: format metrics only check parseability and schema compliance. "
        "They do not prove prompt-conditioned physics understanding.",
        "",
        "## Probe Counts",
        "",
    ]
    for probe_type, count in probe_level["per_probe_type_count"].items():
        success = probe_level["per_probe_type_success_rate"].get(probe_type)
        lines.append(f"- `{probe_type}`: `{count}` probes, success_rate `{_fmt(success)}`")

    lines.extend(
        [
            "",
            "## Language And Physics Metrics",
            "",
            "- "
            + _metric_sentence(
                "paraphrase_consistency_score",
                phys["paraphrase_consistency_score"]["score"],
            ),
            "- "
            + _metric_sentence(
                "prompt_sensitivity_score",
                phys["prompt_sensitivity_score"]["score"],
            ),
            "- "
            + _metric_sentence(
                "fixed_variable_violation_rate",
                phys["fixed_variable_violation_rate"],
                high_is_good=False,
            ),
            "- "
            + _metric_sentence(
                "allowed_variable_violation_rate",
                phys["allowed_variable_violation_rate"],
                high_is_good=False,
            ),
            "- "
            + _metric_sentence(
                "contradiction_detection_accuracy",
                phys["contradiction_detection_accuracy"],
            ),
            "- "
            + _metric_sentence(
                "prompt_image_conflict_detection_accuracy",
                phys["prompt_image_conflict_detection_accuracy"],
            ),
            "- " + _metric_sentence("direction_accuracy", phys["direction_accuracy"]),
            "- "
            + _metric_sentence(
                "direction_flip_accuracy",
                phys["direction_flip_accuracy"]["accuracy"],
            ),
            "- "
            + _metric_sentence(
                "forced_prediction_rate",
                phys["forced_prediction_rate"],
                high_is_good=False,
            ),
            "",
            "Interpretation: low fixed/allowed-variable violation rates indicate better "
            "constraint following. High contradiction/conflict accuracy indicates the model "
            "can reject inconsistent requests rather than forcing a setup. Paraphrase and "
            "prompt-sensitivity scores are the main checks for whether wording changes affect "
            "outputs in a semantically meaningful way.",
            "",
            "## Input Ablation",
            "",
        ]
    )
    for input_mode, metrics in phys["input_ablation_summary"].items():
        lines.append(
            f"- `{input_mode}`: rows `{metrics['rows']}`, success_rate `{_fmt(metrics['success_rate'])}`, "
            f"direction_accuracy `{_fmt(metrics['direction_accuracy'])}`, "
            f"forced_prediction_rate `{_fmt(metrics['forced_prediction_rate'])}`"
        )
    lines.extend(
        [
            "",
            "Interpretation: compare prompt_only, images_only, prompt_plus_images, conflict, "
            "and shuffled_prompt to separate language use from image-driven behavior. A model "
            "that performs similarly with and without prompts is likely relying mostly on images.",
        ]
    )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"][:20]:
            lines.append(f"- {warning}")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
