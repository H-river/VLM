"""Compute non-LLM understanding proxy metrics for local profile2setup outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from profile2setup.evaluation.param_metrics import load_tolerances
from profile2setup.schema import VARIABLE_ORDER, compute_delta_setup, validate_setup_dict
from profile2setup.training.normalization import denormalize_delta_vector, load_variables_config


CANONICAL_VARIABLE_ORDER = list(VARIABLE_ORDER)

VARIABLE_ALIASES = {
    "source_to_lens": {"source_to_lens", "source to lens", "source-lens", "source distance"},
    "lens_to_camera": {"lens_to_camera", "lens to camera", "lens-camera", "camera distance"},
    "focal_length": {"focal_length", "focal length", "focus", "focal"},
    "lens_x": {"lens_x", "lens x"},
    "lens_y": {"lens_y", "lens y"},
    "camera_x": {"camera_x", "camera x"},
    "camera_y": {"camera_y", "camera y"},
}

GROUP_ALIASES = {
    "camera": {"camera_x", "camera_y"},
    "lens": {"lens_x", "lens_y"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Stage 1A local understanding proxy metrics from model eval JSON."
    )
    parser.add_argument("--model-eval", required=True, help="Path to local model evaluation JSON")
    parser.add_argument("--data", required=True, help="Original profile2setup JSONL data")
    parser.add_argument(
        "--variables-config",
        required=True,
        help="Variables YAML config with ranges and optional tolerances",
    )
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--summary-out", required=True, help="Output Markdown summary path")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _clean_setup(value: Any) -> dict[str, float] | None:
    if value is None or not isinstance(value, dict):
        return None
    if not validate_setup_dict(value):
        return None
    return {name: float(value[name]) for name in CANONICAL_VARIABLE_ORDER}


def _target_delta(record: dict) -> dict[str, float] | None:
    explicit = _clean_setup(record.get("target_delta"))
    if explicit is not None:
        return explicit
    current = _clean_setup(record.get("current_setup"))
    target = _clean_setup(record.get("target_setup"))
    if current is not None and target is not None:
        return compute_delta_setup(current, target)
    return None


def _changed_set(delta: dict[str, float] | None, tolerances: dict[str, float]) -> set[str] | None:
    if delta is None:
        return None
    return {
        name
        for name in CANONICAL_VARIABLE_ORDER
        if abs(float(delta[name])) > float(tolerances[name])
    }


def _direction(value: float, tolerance: float) -> str:
    if value > tolerance:
        return "increase"
    if value < -tolerance:
        return "decrease"
    return "unchanged"


def _direction_map(delta: dict[str, float] | None, tolerances: dict[str, float]) -> dict[str, str] | None:
    if delta is None:
        return None
    return {
        name: _direction(float(delta[name]), float(tolerances[name]))
        for name in CANONICAL_VARIABLE_ORDER
    }


def _resolve_variable_mentions(text: str) -> set[str]:
    found: set[str] = set()
    padded = f" {text} "
    for name, aliases in VARIABLE_ALIASES.items():
        for alias in aliases:
            pattern = r"(?<![a-z0-9_])" + re.escape(alias) + r"(?![a-z0-9_])"
            if re.search(pattern, padded):
                found.add(name)
                break
    for alias, variables in GROUP_ALIASES.items():
        pattern = r"(?<![a-z0-9_])" + re.escape(alias) + r"(?![a-z0-9_])"
        if re.search(pattern, padded):
            found.update(variables)
    return found


def _infer_fixed_variables(prompt: Any) -> set[str]:
    text = re.sub(r"\s+", " ", str(prompt or "").strip().lower())
    if not text:
        return set()

    fixed: set[str] = set()
    if any(phrase in text for phrase in ("keep camera fixed", "camera fixed", "do not move camera")):
        fixed.update({"camera_x", "camera_y"})

    for match in re.finditer(r"(?:keep|hold|leave)\s+(.+?)\s+fixed", text):
        fixed.update(_resolve_variable_mentions(match.group(1)))
    for match in re.finditer(r"do not (?:move|change|adjust)\s+(.+?)(?:[,.]|$)", text):
        fixed.update(_resolve_variable_mentions(match.group(1)))

    for match in re.finditer(r"only (?:change|move|adjust)\s+(.+?)(?:[,.]|$)", text):
        allowed = _resolve_variable_mentions(match.group(1))
        if allowed:
            fixed.update(set(CANONICAL_VARIABLE_ORDER) - allowed)

    return fixed


def _safe_rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int | None]:
    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    if precision is None or recall is None or (precision + recall) == 0.0:
        f1 = None
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))
    return {
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _model_examples_by_id(model_eval: dict) -> tuple[dict[str, dict], list[str]]:
    examples = model_eval.get("examples")
    warnings: list[str] = []
    if not isinstance(examples, list):
        return {}, ["model_eval.examples missing or not a list"]

    by_id: dict[str, dict] = {}
    for idx, example in enumerate(examples):
        if not isinstance(example, dict):
            warnings.append(f"model_eval.examples[{idx}] is not an object")
            continue
        record_id = example.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            warnings.append(f"model_eval.examples[{idx}] missing record_id")
            continue
        by_id[record_id] = example
    return by_id, warnings


def compute_proxy_metrics(
    *,
    model_eval_path: Path,
    data_path: Path,
    variables_config_path: Path,
) -> dict:
    model_eval = _load_json(model_eval_path)
    data_rows = _load_jsonl(data_path)
    variables_config = load_variables_config(variables_config_path)
    tolerances = load_tolerances(variables_config)
    examples_by_id, warnings = _model_examples_by_id(model_eval)

    tp = fp = fn = 0
    direction_correct = direction_total = 0
    fixed_total = fixed_violations = 0
    per_var = {
        name: {"tp": 0, "fp": 0, "fn": 0, "direction_correct": 0, "direction_total": 0}
        for name in CANONICAL_VARIABLE_ORDER
    }

    records_out: list[dict[str, Any]] = []
    counts = {
        "data_records": len(data_rows),
        "model_eval_examples": len(examples_by_id),
        "records_with_model_example": 0,
        "records_missing_model_example": 0,
        "records_with_predicted_delta": 0,
        "records_missing_predicted_delta": 0,
        "records_with_target_delta": 0,
        "records_missing_target_delta": 0,
        "records_evaluated_for_changed_variables": 0,
        "records_evaluated_for_fixed_constraints": 0,
    }

    for idx, record in enumerate(data_rows, start=1):
        record_id = record.get("id") or record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            record_id = f"line_{idx}"

        example = examples_by_id.get(record_id)
        predicted_delta = None
        target_delta = _target_delta(record)
        row_warnings: list[str] = []

        if example is None:
            counts["records_missing_model_example"] += 1
            row_warnings.append("missing model_eval example for record_id")
        else:
            counts["records_with_model_example"] += 1
            predicted_delta_norm = example.get("predicted_delta_norm")
            if predicted_delta_norm is None:
                row_warnings.append("missing model_eval example predicted_delta_norm")
            else:
                try:
                    predicted_delta = denormalize_delta_vector(
                        [predicted_delta_norm[name] for name in CANONICAL_VARIABLE_ORDER],
                        variables_config,
                    )
                except Exception as exc:  # noqa: BLE001 - row-level diagnostics are useful here.
                    row_warnings.append(f"could not denormalize predicted_delta_norm: {exc}")

        if predicted_delta is None:
            counts["records_missing_predicted_delta"] += 1
        else:
            counts["records_with_predicted_delta"] += 1

        if target_delta is None:
            counts["records_missing_target_delta"] += 1
        else:
            counts["records_with_target_delta"] += 1

        pred_changed = _changed_set(predicted_delta, tolerances)
        target_changed = _changed_set(target_delta, tolerances)
        pred_direction = _direction_map(predicted_delta, tolerances)
        target_direction = _direction_map(target_delta, tolerances)

        if pred_changed is not None and target_changed is not None:
            counts["records_evaluated_for_changed_variables"] += 1
            tp += len(pred_changed & target_changed)
            fp += len(pred_changed - target_changed)
            fn += len(target_changed - pred_changed)
            for name in CANONICAL_VARIABLE_ORDER:
                pred_has = name in pred_changed
                target_has = name in target_changed
                if pred_has and target_has:
                    per_var[name]["tp"] += 1
                elif pred_has and not target_has:
                    per_var[name]["fp"] += 1
                elif target_has and not pred_has:
                    per_var[name]["fn"] += 1

        if pred_direction is not None and target_direction is not None:
            for name in CANONICAL_VARIABLE_ORDER:
                direction_total += 1
                per_var[name]["direction_total"] += 1
                if pred_direction[name] == target_direction[name]:
                    direction_correct += 1
                    per_var[name]["direction_correct"] += 1

        fixed_variables = _infer_fixed_variables(record.get("prompt"))
        fixed_variable_results = {}
        if fixed_variables and predicted_delta is not None:
            counts["records_evaluated_for_fixed_constraints"] += 1
            for name in sorted(fixed_variables):
                fixed_total += 1
                violates = abs(float(predicted_delta[name])) > float(tolerances[name])
                if violates:
                    fixed_violations += 1
                fixed_variable_results[name] = {
                    "predicted_delta": float(predicted_delta[name]),
                    "tolerance": float(tolerances[name]),
                    "violates": bool(violates),
                }

        records_out.append(
            {
                "record_id": record_id,
                "line_number": idx,
                "task_type": record.get("task_type"),
                "prompt": record.get("prompt"),
                "predicted_delta_physical": predicted_delta,
                "target_delta_physical": target_delta,
                "predicted_changed_variables": None if pred_changed is None else sorted(pred_changed),
                "target_changed_variables": None if target_changed is None else sorted(target_changed),
                "predicted_change_direction": pred_direction,
                "target_change_direction": target_direction,
                "fixed_variables_from_prompt": sorted(fixed_variables),
                "fixed_variable_results": fixed_variable_results,
                "warnings": row_warnings,
            }
        )

    per_variable_metrics = {}
    for name, values in per_var.items():
        block = _prf(values["tp"], values["fp"], values["fn"])
        block["change_direction_accuracy"] = _safe_rate(
            values["direction_correct"],
            values["direction_total"],
        )
        block["direction_correct"] = int(values["direction_correct"])
        block["direction_total"] = int(values["direction_total"])
        per_variable_metrics[name] = block

    metrics = _prf(tp, fp, fn)
    metrics["change_direction_accuracy"] = _safe_rate(direction_correct, direction_total)
    metrics["direction_correct"] = int(direction_correct)
    metrics["direction_total"] = int(direction_total)
    metrics["fixed_variable_violation_rate"] = _safe_rate(fixed_violations, fixed_total)
    metrics["fixed_variable_accuracy"] = _safe_rate(fixed_total - fixed_violations, fixed_total)
    metrics["fixed_variable_violations"] = int(fixed_violations)
    metrics["fixed_variable_total"] = int(fixed_total)
    metrics["per_variable"] = per_variable_metrics

    missing_examples = [
        record.get("id") or record.get("record_id") or f"line_{idx}"
        for idx, record in enumerate(data_rows, start=1)
        if (record.get("id") or record.get("record_id") or f"line_{idx}") not in examples_by_id
    ]
    if missing_examples:
        warnings.append(
            f"{len(missing_examples)} data records are missing detailed model_eval examples; "
            "rerun evaluate_cli with --max-examples at least the dataset size for full coverage"
        )

    return {
        "metadata": {
            "stage": "Stage 1A local understanding proxy",
            "model_eval_path": str(model_eval_path),
            "data_path": str(data_path),
            "variables_config_path": str(variables_config_path),
            "variable_order": CANONICAL_VARIABLE_ORDER,
            "method": (
                "Denormalize model_eval examples[*].predicted_delta_norm to physical units, "
                "threshold predicted and target deltas by per-variable tolerance, and compare "
                "changed-variable sets plus increase/decrease/unchanged directions."
            ),
        },
        "field_inspection": {
            "predicted_delta": "model_eval.examples[*].predicted_delta_norm, denormalized with variables config",
            "routed_prediction": "model_eval.examples[*].predicted_routed_setup_physical",
            "target_delta": "data JSONL target_delta when present, else target_setup - current_setup when both are present",
            "task_type": "model_eval.examples[*].task_type and data JSONL task_type",
            "record_id": "model_eval.examples[*].record_id matched to data JSONL id",
        },
        "counts": counts,
        "tolerances": {name: float(tolerances[name]) for name in CANONICAL_VARIABLE_ORDER},
        "metrics": metrics,
        "missing_model_example_record_ids": missing_examples,
        "warnings": warnings,
        "records": records_out,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_summary(result: dict, path: Path) -> None:
    metrics = result["metrics"]
    counts = result["counts"]
    lines = [
        "# Stage 1A Local Understanding Proxy",
        "",
        "## Purpose",
        "",
        (
            "This summarizes non-LLM understanding proxy metrics for the local PyTorch "
            "`profile2setup` model. The model does not emit explicit understanding JSON, "
            "so changed variables and change directions are derived from its predicted delta head."
        ),
        "",
        "## Inputs",
        "",
        f"- Model eval JSON: `{result['metadata']['model_eval_path']}`",
        f"- Data JSONL: `{result['metadata']['data_path']}`",
        f"- Variables config: `{result['metadata']['variables_config_path']}`",
        f"- Variable order: `{', '.join(result['metadata']['variable_order'])}`",
        "",
        "## Field Mapping",
        "",
    ]
    for key, value in result["field_inspection"].items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Data records: `{counts['data_records']}`",
            f"- Detailed model examples: `{counts['model_eval_examples']}`",
            f"- Records with predicted delta: `{counts['records_with_predicted_delta']}`",
            f"- Records with target delta: `{counts['records_with_target_delta']}`",
            f"- Records evaluated for changed-variable metrics: `{counts['records_evaluated_for_changed_variables']}`",
            f"- Records evaluated for fixed-constraint metrics: `{counts['records_evaluated_for_fixed_constraints']}`",
            "",
            "## Main Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| changed_variable_precision | {_fmt(metrics['precision'])} |",
            f"| changed_variable_recall | {_fmt(metrics['recall'])} |",
            f"| changed_variable_f1 | {_fmt(metrics['f1'])} |",
            f"| change_direction_accuracy | {_fmt(metrics['change_direction_accuracy'])} |",
            f"| fixed_variable_violation_rate | {_fmt(metrics['fixed_variable_violation_rate'])} |",
            f"| fixed_variable_accuracy | {_fmt(metrics['fixed_variable_accuracy'])} |",
            "",
            "## Per-Variable Metrics",
            "",
            "| Variable | Precision | Recall | F1 | Direction Accuracy |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in CANONICAL_VARIABLE_ORDER:
        block = metrics["per_variable"][name]
        lines.append(
            f"| `{name}` | {_fmt(block['precision'])} | {_fmt(block['recall'])} | "
            f"{_fmt(block['f1'])} | {_fmt(block['change_direction_accuracy'])} |"
        )

    lines.extend(["", "## Notes", ""])
    warnings = result.get("warnings") or []
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- No missing required proxy fields were found for evaluated rows.")
    if counts["records_missing_target_delta"]:
        lines.append(
            "- Records without target deltas are excluded from changed-variable and "
            "direction metrics because there is no ground-truth change to compare."
        )
    if metrics["fixed_variable_total"] == 0:
        lines.append("- No prompt-level fixed-variable constraints were detected in this subset.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = compute_proxy_metrics(
        model_eval_path=Path(args.model_eval),
        data_path=Path(args.data),
        variables_config_path=Path(args.variables_config),
    )
    _save_json(result, Path(args.out))
    write_summary(result, Path(args.summary_out))
    print(json.dumps({"out": args.out, "summary_out": args.summary_out, "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
