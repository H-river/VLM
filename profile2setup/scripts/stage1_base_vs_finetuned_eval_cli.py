"""Compare base LLM, fine-tuned LLM, and local model on Stage 1D probe labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profile2setup.evaluation.param_metrics import load_tolerances
from profile2setup.llm_api.validator import validate_all_variable_dicts, validate_llm_output
from profile2setup.schema import VARIABLE_ORDER
from profile2setup.training.normalization import denormalize_delta_vector, load_variables_config


CANONICAL_VARIABLE_ORDER = list(VARIABLE_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Stage 1D base/fine-tuned/local understanding metrics.")
    parser.add_argument("--labels", required=True, help="Probe labels JSONL")
    parser.add_argument("--base-predictions", required=True, help="Base LLM predictions JSONL")
    parser.add_argument("--finetuned-predictions", required=True, help="Fine-tuned LLM predictions JSONL")
    parser.add_argument("--local-model-eval", required=True, help="Local model eval JSON")
    parser.add_argument("--variables-config", required=True, help="Variables YAML config")
    parser.add_argument("--out", required=True, help="Output comparison JSON")
    parser.add_argument("--summary-out", required=True, help="Output comparison Markdown")
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
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(obj)
    return rows


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return float(2.0 * precision * recall / (precision + recall))


def _parse_prediction(row: dict | None) -> tuple[dict | None, bool]:
    if row is None:
        return None, False
    prediction = row.get("prediction")
    if isinstance(prediction, dict):
        return prediction, True
    raw = row.get("raw_response")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None, False
        return (parsed, True) if isinstance(parsed, dict) else (None, False)
    return None, False


def _rows_by_id(path: Path) -> tuple[dict[str, dict], list[str]]:
    if not path.exists():
        return {}, [f"predictions file not found: {path}"]
    rows = _load_jsonl(path)
    by_id: dict[str, dict] = {}
    warnings: list[str] = []
    for idx, row in enumerate(rows, start=1):
        record_id = row.get("record_id") or row.get("id")
        if not isinstance(record_id, str) or not record_id:
            warnings.append(f"{path}:{idx} missing record_id")
            continue
        by_id[record_id] = row
    return by_id, warnings


def _local_by_id(path: Path) -> tuple[dict[str, dict], list[str]]:
    if not path.exists():
        return {}, [f"local model eval file not found: {path}"]
    obj = _load_json(path)
    examples = obj.get("examples")
    if not isinstance(examples, list):
        return {}, [f"local model eval examples missing or not a list: {path}"]
    out = {}
    warnings = []
    for idx, example in enumerate(examples, start=1):
        record_id = example.get("record_id") if isinstance(example, dict) else None
        if not isinstance(record_id, str) or not record_id:
            warnings.append(f"local model eval example {idx} missing record_id")
            continue
        out[record_id] = example
    return out, warnings


def _direction_from_delta(delta: dict[str, float], tolerances: dict[str, float]) -> dict[str, str]:
    out = {}
    for name in CANONICAL_VARIABLE_ORDER:
        value = float(delta[name])
        tol = float(tolerances[name])
        if value > tol:
            out[name] = "increase"
        elif value < -tol:
            out[name] = "decrease"
        else:
            out[name] = "unchanged"
    return out


def _changed_from_direction(direction: dict[str, str] | None) -> list[str] | None:
    if direction is None:
        return None
    return [name for name in CANONICAL_VARIABLE_ORDER if direction.get(name) != "unchanged"]


def _normalize_direction_map(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    out = {}
    for name in CANONICAL_VARIABLE_ORDER:
        direction = value.get(name)
        if direction not in {"increase", "decrease", "unchanged"}:
            return None
        out[name] = direction
    return out


def _clean_delta(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        return {name: float(value[name]) for name in CANONICAL_VARIABLE_ORDER}
    except Exception:
        return None


def _llm_understanding(prediction: dict | None, tolerances: dict[str, float]) -> dict[str, Any]:
    if prediction is None:
        return {
            "changed_variables": None,
            "change_direction": None,
            "predicted_delta": None,
            "rejected": None,
            "valid": None,
            "rejection_reason": None,
        }
    understanding = prediction.get("setup_understanding")
    understanding = understanding if isinstance(understanding, dict) else {}
    direction = _normalize_direction_map(understanding.get("change_direction"))
    delta = _clean_delta(prediction.get("predicted_delta"))
    if direction is None and delta is not None:
        direction = _direction_from_delta(delta, tolerances)
    changed = understanding.get("changed_variables")
    if isinstance(changed, list) and all(isinstance(item, str) for item in changed):
        changed_variables = [name for name in CANONICAL_VARIABLE_ORDER if name in set(changed)]
    else:
        changed_variables = _changed_from_direction(direction)
    rejection_reason = prediction.get("rejection_reason")
    rejected = prediction.get("valid") is False and isinstance(rejection_reason, str) and bool(rejection_reason.strip())
    return {
        "changed_variables": changed_variables,
        "change_direction": direction,
        "predicted_delta": delta,
        "rejected": rejected,
        "valid": prediction.get("valid"),
        "rejection_reason": rejection_reason,
    }


def _local_understanding(
    example: dict | None,
    variables_config: dict,
    tolerances: dict[str, float],
) -> dict[str, Any]:
    if example is None or not isinstance(example.get("predicted_delta_norm"), dict):
        return {
            "changed_variables": None,
            "change_direction": None,
            "predicted_delta": None,
            "forced_numerical_prediction": False,
        }
    try:
        delta = denormalize_delta_vector(
            [example["predicted_delta_norm"][name] for name in CANONICAL_VARIABLE_ORDER],
            variables_config,
        )
    except Exception:
        return {
            "changed_variables": None,
            "change_direction": None,
            "predicted_delta": None,
            "forced_numerical_prediction": False,
        }
    direction = _direction_from_delta(delta, tolerances)
    return {
        "changed_variables": _changed_from_direction(direction),
        "change_direction": direction,
        "predicted_delta": delta,
        "forced_numerical_prediction": True,
    }


def _fixed_violations(
    expected_fixed: list[str],
    direction: dict[str, str] | None,
) -> tuple[int, int]:
    if direction is None:
        return 0, 0
    total = 0
    violations = 0
    for name in expected_fixed:
        if name not in CANONICAL_VARIABLE_ORDER:
            continue
        total += 1
        violations += int(direction.get(name) != "unchanged")
    return violations, total


def _empty_model_counts() -> dict[str, int]:
    return {
        "prediction_rows": 0,
        "json_valid": 0,
        "schema_valid": 0,
        "canonical": 0,
        "changed_tp": 0,
        "changed_fp": 0,
        "changed_fn": 0,
        "direction_correct": 0,
        "direction_total": 0,
        "fixed_violations": 0,
        "fixed_total": 0,
        "invalid_rejections": 0,
        "invalid_total": 0,
        "contradiction_detections": 0,
        "contradiction_total": 0,
        "invalid_forced_predictions": 0,
        "invalid_forced_total": 0,
        "ambiguous_forced_predictions": 0,
        "ambiguous_forced_total": 0,
    }


def _finalize_llm_counts(counts: dict[str, int]) -> dict[str, Any]:
    precision = _rate(counts["changed_tp"], counts["changed_tp"] + counts["changed_fp"])
    recall = _rate(counts["changed_tp"], counts["changed_tp"] + counts["changed_fn"])
    return {
        "valid_json_rate": _rate(counts["json_valid"], counts["prediction_rows"]),
        "schema_valid_rate": _rate(counts["schema_valid"], counts["prediction_rows"]),
        "canonical_variable_rate": _rate(counts["canonical"], counts["prediction_rows"]),
        "changed_variable_precision": precision,
        "changed_variable_recall": recall,
        "changed_variable_f1": _f1(precision, recall),
        "change_direction_accuracy": _rate(counts["direction_correct"], counts["direction_total"]),
        "fixed_variable_violation_rate": _rate(counts["fixed_violations"], counts["fixed_total"]),
        "invalid_rejection_accuracy": _rate(counts["invalid_rejections"], counts["invalid_total"]),
        "contradiction_detection_accuracy": _rate(
            counts["contradiction_detections"],
            counts["contradiction_total"],
        ),
        **counts,
    }


def _finalize_local_counts(counts: dict[str, int]) -> dict[str, Any]:
    precision = _rate(counts["changed_tp"], counts["changed_tp"] + counts["changed_fp"])
    recall = _rate(counts["changed_tp"], counts["changed_tp"] + counts["changed_fn"])
    return {
        "valid_json_rate": None,
        "schema_valid_rate": None,
        "canonical_variable_rate": None,
        "changed_variable_precision": precision,
        "changed_variable_recall": recall,
        "changed_variable_f1": _f1(precision, recall),
        "change_direction_accuracy": _rate(counts["direction_correct"], counts["direction_total"]),
        "fixed_variable_violation_rate": _rate(counts["fixed_violations"], counts["fixed_total"]),
        "invalid_rejection_accuracy": None,
        "contradiction_detection_accuracy": None,
        "native_rejection": "not_applicable_no_rejection_head",
        "invalid_forced_prediction_rate": _rate(
            counts["invalid_forced_predictions"],
            counts["invalid_forced_total"],
        ),
        "ambiguous_forced_prediction_rate": _rate(
            counts["ambiguous_forced_predictions"],
            counts["ambiguous_forced_total"],
        ),
        **counts,
    }


def _score_understanding(label: dict, understanding: dict[str, Any], counts: dict[str, int]) -> None:
    expected_changed = set(label.get("expected_changed_variables") or [])
    predicted_changed = understanding.get("changed_variables")
    if predicted_changed is not None:
        predicted_changed_set = set(predicted_changed)
        counts["changed_tp"] += len(expected_changed & predicted_changed_set)
        counts["changed_fp"] += len(predicted_changed_set - expected_changed)
        counts["changed_fn"] += len(expected_changed - predicted_changed_set)

    expected_direction = label.get("expected_change_direction") or {}
    predicted_direction = understanding.get("change_direction")
    if isinstance(predicted_direction, dict):
        for name in CANONICAL_VARIABLE_ORDER:
            if name in expected_direction:
                counts["direction_total"] += 1
                counts["direction_correct"] += int(predicted_direction.get(name) == expected_direction[name])

    fixed_violations, fixed_total = _fixed_violations(
        list(label.get("expected_fixed_variables") or []),
        predicted_direction,
    )
    counts["fixed_violations"] += fixed_violations
    counts["fixed_total"] += fixed_total


def _score_llm_model(labels: list[dict], rows_by_id: dict[str, dict], tolerances: dict[str, float]) -> dict[str, Any]:
    counts = _empty_model_counts()
    records = []
    for label in labels:
        record_id = label["record_id"]
        row = rows_by_id.get(record_id)
        if row is None:
            records.append({"record_id": record_id, "missing_prediction": True})
            continue
        counts["prediction_rows"] += 1
        prediction, json_valid = _parse_prediction(row)
        counts["json_valid"] += int(json_valid)
        canonical = False
        schema_valid = False
        if prediction is not None:
            try:
                validate_all_variable_dicts(prediction)
                canonical = True
            except Exception:
                canonical = False
            try:
                validate_llm_output(prediction)
                schema_valid = True
            except Exception:
                schema_valid = False
        counts["canonical"] += int(canonical)
        counts["schema_valid"] += int(schema_valid)
        understanding = _llm_understanding(prediction, tolerances)
        _score_understanding(label, understanding, counts)

        if label.get("category") == "invalid":
            counts["invalid_total"] += 1
            counts["invalid_rejections"] += int(bool(understanding.get("rejected")))
        if label.get("category") == "ambiguous_multi_intent":
            counts["contradiction_total"] += 1
            counts["contradiction_detections"] += int(bool(understanding.get("rejected")))

        records.append(
            {
                "record_id": record_id,
                "category": label.get("category"),
                "json_valid": json_valid,
                "schema_valid": schema_valid,
                "canonical": canonical,
                "understanding": understanding,
            }
        )
    return {"metrics": _finalize_llm_counts(counts), "records": records}


def _score_local_model(
    labels: list[dict],
    local_by_id: dict[str, dict],
    variables_config: dict,
    tolerances: dict[str, float],
) -> dict[str, Any]:
    counts = _empty_model_counts()
    records = []
    for label in labels:
        record_id = label["record_id"]
        understanding = _local_understanding(local_by_id.get(record_id), variables_config, tolerances)
        _score_understanding(label, understanding, counts)

        if label.get("category") == "invalid":
            counts["invalid_forced_total"] += 1
            counts["invalid_forced_predictions"] += int(bool(understanding.get("forced_numerical_prediction")))
        if label.get("category") == "ambiguous_multi_intent":
            counts["ambiguous_forced_total"] += 1
            counts["ambiguous_forced_predictions"] += int(bool(understanding.get("forced_numerical_prediction")))

        records.append(
            {
                "record_id": record_id,
                "category": label.get("category"),
                "understanding": understanding,
            }
        )
    return {"metrics": _finalize_local_counts(counts), "records": records}


def compute_comparison(
    *,
    labels_path: Path,
    base_predictions_path: Path,
    finetuned_predictions_path: Path,
    local_model_eval_path: Path,
    variables_config_path: Path,
) -> dict:
    labels = _load_jsonl(labels_path)
    variables_config = load_variables_config(variables_config_path)
    tolerances = load_tolerances(variables_config)
    base_rows, base_warnings = _rows_by_id(base_predictions_path)
    ft_rows, ft_warnings = _rows_by_id(finetuned_predictions_path)
    local_rows, local_warnings = _local_by_id(local_model_eval_path)
    return {
        "metadata": {
            "labels_path": str(labels_path),
            "base_predictions_path": str(base_predictions_path),
            "finetuned_predictions_path": str(finetuned_predictions_path),
            "local_model_eval_path": str(local_model_eval_path),
            "variables_config_path": str(variables_config_path),
            "variable_order": CANONICAL_VARIABLE_ORDER,
            "local_rejection_note": (
                "The local PyTorch model has no native rejection head; invalid rejection is not applicable."
            ),
        },
        "counts": {
            "probe_records": len(labels),
            "base_prediction_rows_loaded": len(base_rows),
            "finetuned_prediction_rows_loaded": len(ft_rows),
            "local_examples_loaded": len(local_rows),
        },
        "tolerances": {name: float(tolerances[name]) for name in CANONICAL_VARIABLE_ORDER},
        "models": {
            "base_llm": _score_llm_model(labels, base_rows, tolerances),
            "finetuned_llm": _score_llm_model(labels, ft_rows, tolerances),
            "local_pytorch": _score_local_model(labels, local_rows, variables_config, tolerances),
        },
        "warnings": base_warnings + ft_warnings + local_warnings,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_summary(result: dict, path: Path) -> None:
    base = result["models"]["base_llm"]["metrics"]
    ft = result["models"]["finetuned_llm"]["metrics"]
    local = result["models"]["local_pytorch"]["metrics"]
    lines = [
        "# Stage 1D Base vs Fine-Tuned LLM Comparison",
        "",
        "## Purpose",
        "",
        "This compares base `gpt-4o-2024-08-06`, the fine-tuned profile2setup LLM, and the local PyTorch checkpoint on the 25-record Stage 1D probe.",
        "",
        "## Inputs",
        "",
        f"- Labels: `{result['metadata']['labels_path']}`",
        f"- Base predictions: `{result['metadata']['base_predictions_path']}`",
        f"- Fine-tuned predictions: `{result['metadata']['finetuned_predictions_path']}`",
        f"- Local eval: `{result['metadata']['local_model_eval_path']}`",
        "",
        "## Loaded Rows",
        "",
        f"- Probe labels: `{result['counts']['probe_records']}`",
        f"- Base prediction rows: `{result['counts']['base_prediction_rows_loaded']}`",
        f"- Fine-tuned prediction rows: `{result['counts']['finetuned_prediction_rows_loaded']}`",
        f"- Local examples: `{result['counts']['local_examples_loaded']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Base LLM | Fine-tuned LLM | Local PyTorch |",
        "|---|---:|---:|---:|",
    ]
    for metric in [
        "valid_json_rate",
        "schema_valid_rate",
        "canonical_variable_rate",
        "changed_variable_precision",
        "changed_variable_recall",
        "changed_variable_f1",
        "change_direction_accuracy",
    ]:
        lines.append(f"| {metric} | {_fmt(base.get(metric))} | {_fmt(ft.get(metric))} | {_fmt(local.get(metric))} |")
    lines.extend(
        [
            f"| invalid_rejection_accuracy | {_fmt(base.get('invalid_rejection_accuracy'))} | {_fmt(ft.get('invalid_rejection_accuracy'))} | not applicable |",
            f"| fixed_variable_violation_rate | {_fmt(base.get('fixed_variable_violation_rate'))} | {_fmt(ft.get('fixed_variable_violation_rate'))} | {_fmt(local.get('fixed_variable_violation_rate'))} |",
            f"| contradiction_detection_accuracy | {_fmt(base.get('contradiction_detection_accuracy'))} | {_fmt(ft.get('contradiction_detection_accuracy'))} | not applicable |",
            f"| invalid_forced_prediction_rate | not applicable | not applicable | {_fmt(local.get('invalid_forced_prediction_rate'))} |",
            f"| ambiguous_forced_prediction_rate | not applicable | not applicable | {_fmt(local.get('ambiguous_forced_prediction_rate'))} |",
        ]
    )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- LLM metrics are unavailable until the corresponding prediction JSONL exists and contains rows for the 25 probe record IDs.",
            "- The local PyTorch model has no native rejection head, so invalid rejection is not applicable. Forced-prediction rates are reported separately.",
            "- No paid API calls are made by this comparison script.",
            "",
            "## Reproduction Commands",
            "",
            "Set `OPENAI_API_KEY` in your shell before running the inference commands. These commands use low image detail and temperature 0.0.",
            "",
            "Run base LLM inference:",
            "",
            "```bash",
            "/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.run_llm_api_inference_cli \\",
            "  --model gpt-4o-2024-08-06 \\",
            "  --data profile2setup/data/stage1_understanding/stage1_base_llm_probe_25.jsonl \\",
            "  --out profile2setup/results/stage1_understanding/base_llm_predictions_25.jsonl \\",
            "  --image-out-dir profile2setup/results/stage1_understanding/images/base_llm_25 \\",
            "  --image-detail low \\",
            "  --temperature 0.0",
            "```",
            "",
            "Run fine-tuned LLM inference on the same 25 records:",
            "",
            "```bash",
            "/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.run_llm_api_inference_cli \\",
            "  --model ft:gpt-4o-2024-08-06:personal:profile2setup-mm-sft:DcGtpEep \\",
            "  --data profile2setup/data/stage1_understanding/stage1_base_llm_probe_25.jsonl \\",
            "  --out profile2setup/results/stage1_understanding/finetuned_llm_predictions_25.jsonl \\",
            "  --image-out-dir profile2setup/results/stage1_understanding/images/finetuned_llm_25 \\",
            "  --image-detail low \\",
            "  --temperature 0.0",
            "```",
            "",
            "Run the standard LLM/API evaluators where possible:",
            "",
            "```bash",
            "/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.evaluate_llm_api_predictions_cli \\",
            "  --predictions profile2setup/results/stage1_understanding/base_llm_predictions_25.jsonl \\",
            "  --data profile2setup/data/stage1_understanding/stage1_base_llm_probe_25.jsonl \\",
            "  --out profile2setup/results/stage1_understanding/base_llm_eval_25.json \\",
            "  --variables-config profile2setup/configs/variables.yaml \\",
            "  --max-examples 25",
            "",
            "/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.evaluate_llm_api_predictions_cli \\",
            "  --predictions profile2setup/results/stage1_understanding/finetuned_llm_predictions_25.jsonl \\",
            "  --data profile2setup/data/stage1_understanding/stage1_base_llm_probe_25.jsonl \\",
            "  --out profile2setup/results/stage1_understanding/finetuned_llm_eval_25.json \\",
            "  --variables-config profile2setup/configs/variables.yaml \\",
            "  --max-examples 25",
            "```",
            "",
            "Re-run this comparison:",
            "",
            "```bash",
            "/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.stage1_base_vs_finetuned_eval_cli \\",
            "  --labels profile2setup/data/stage1_understanding/stage1_base_llm_probe_25_labels.jsonl \\",
            "  --base-predictions profile2setup/results/stage1_understanding/base_llm_predictions_25.jsonl \\",
            "  --finetuned-predictions profile2setup/results/stage1_understanding/finetuned_llm_predictions_25.jsonl \\",
            "  --local-model-eval profile2setup/results/stage1_understanding/local_model_eval_25.json \\",
            "  --variables-config profile2setup/configs/variables.yaml \\",
            "  --out profile2setup/results/stage1_understanding/base_vs_finetuned_llm_25.json \\",
            "  --summary-out profile2setup/results/stage1_understanding/base_vs_finetuned_llm_25.md",
            "```",
            "",
            "## Warnings",
            "",
        ]
    )
    warnings = result.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No loader warnings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = compute_comparison(
        labels_path=Path(args.labels),
        base_predictions_path=Path(args.base_predictions),
        finetuned_predictions_path=Path(args.finetuned_predictions),
        local_model_eval_path=Path(args.local_model_eval),
        variables_config_path=Path(args.variables_config),
    )
    _save_json(result, Path(args.out))
    write_summary(result, Path(args.summary_out))
    print(json.dumps({"out": args.out, "summary_out": args.summary_out, "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
