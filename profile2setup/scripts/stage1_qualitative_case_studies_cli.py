"""Build qualitative Stage 1 case studies for senior review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profile2setup.evaluation.param_metrics import load_tolerances
from profile2setup.schema import VARIABLE_ORDER
from profile2setup.training.normalization import denormalize_delta_vector, load_variables_config


CANONICAL_VARIABLE_ORDER = list(VARIABLE_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build qualitative Stage 1 case study report.")
    parser.add_argument("--data", required=True, help="Stage-1 benchmark JSONL")
    parser.add_argument("--labels", required=True, help="Stage-1 labels JSONL")
    parser.add_argument("--llm-predictions", required=True, help="Fine-tuned LLM predictions JSONL")
    parser.add_argument(
        "--fallback-llm-predictions",
        default=None,
        help="Optional fallback predictions JSONL used when --llm-predictions is missing",
    )
    parser.add_argument("--local-proxy", default=None, help="Local understanding proxy JSON, optional")
    parser.add_argument("--local-model-eval", required=True, help="Local model eval JSON")
    parser.add_argument("--variables-config", required=True, help="Variables YAML config")
    parser.add_argument("--out", required=True, help="Output Markdown path")
    parser.add_argument("--json-out", required=True, help="Output machine-readable JSON path")
    parser.add_argument("--max-cases", type=int, default=12, help="Maximum number of case studies")
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


def _rows_by_id(rows: list[dict], key: str) -> dict[str, dict]:
    return {row[key]: row for row in rows if isinstance(row.get(key), str)}


def _prediction_rows_by_id(path: Path) -> dict[str, dict]:
    return {row["record_id"]: row for row in _load_jsonl(path) if isinstance(row.get("record_id"), str)}


def _local_examples_by_id(path: Path) -> dict[str, dict]:
    obj = _load_json(path)
    examples = obj.get("examples")
    if not isinstance(examples, list):
        return {}
    return {row["record_id"]: row for row in examples if isinstance(row, dict) and isinstance(row.get("record_id"), str)}


def _proxy_records_by_id(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    obj = _load_json(path)
    records = obj.get("records")
    if not isinstance(records, list):
        return {}
    return {row["record_id"]: row for row in records if isinstance(row, dict) and isinstance(row.get("record_id"), str)}


def _parse_prediction(row: dict | None) -> dict | None:
    if row is None:
        return None
    prediction = row.get("prediction")
    if isinstance(prediction, dict):
        return prediction
    raw = row.get("raw_response")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


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


def _normalize_direction(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    out = {}
    for name in CANONICAL_VARIABLE_ORDER:
        direction = value.get(name)
        if direction not in {"increase", "decrease", "unchanged"}:
            return None
        out[name] = direction
    return out


def _clean_setup(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        return {name: float(value[name]) for name in CANONICAL_VARIABLE_ORDER}
    except Exception:
        return None


def _llm_summary(row: dict | None, tolerances: dict[str, float]) -> dict[str, Any]:
    prediction = _parse_prediction(row)
    if prediction is None:
        return {
            "available": False,
            "valid_json": row.get("valid_json") if row else None,
            "changed_variables": None,
            "change_direction": None,
            "reasoning_summary": None,
            "predicted_setup": None,
            "predicted_delta": None,
            "rejected": None,
            "rejection_reason": None,
        }
    understanding = prediction.get("setup_understanding")
    understanding = understanding if isinstance(understanding, dict) else {}
    direction = _normalize_direction(understanding.get("change_direction"))
    delta = _clean_setup(prediction.get("predicted_delta"))
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
        "available": True,
        "valid_json": row.get("valid_json") if row else None,
        "changed_variables": changed_variables,
        "change_direction": direction,
        "reasoning_summary": prediction.get("reasoning_summary"),
        "predicted_setup": _clean_setup(prediction.get("predicted_setup")),
        "predicted_delta": delta,
        "rejected": rejected,
        "rejection_reason": rejection_reason,
    }


def _local_summary(
    example: dict | None,
    proxy_record: dict | None,
    variables_config: dict,
    tolerances: dict[str, float],
) -> dict[str, Any]:
    if isinstance(proxy_record, dict) and isinstance(proxy_record.get("predicted_change_direction"), dict):
        direction = _normalize_direction(proxy_record.get("predicted_change_direction"))
        return {
            "changed_variables": proxy_record.get("predicted_changed_variables"),
            "change_direction": direction,
            "predicted_setup": example.get("predicted_routed_setup_physical") if example else None,
            "predicted_delta": proxy_record.get("predicted_delta_physical"),
        }
    if example is None or not isinstance(example.get("predicted_delta_norm"), dict):
        return {
            "changed_variables": None,
            "change_direction": None,
            "predicted_setup": None,
            "predicted_delta": None,
        }
    try:
        delta = denormalize_delta_vector(
            [example["predicted_delta_norm"][name] for name in CANONICAL_VARIABLE_ORDER],
            variables_config,
        )
    except Exception:
        delta = None
    direction = _direction_from_delta(delta, tolerances) if delta is not None else None
    return {
        "changed_variables": _changed_from_direction(direction),
        "change_direction": direction,
        "predicted_setup": example.get("predicted_routed_setup_physical"),
        "predicted_delta": delta,
    }


def _direction_accuracy(pred: dict[str, str] | None, expected: dict[str, str]) -> float | None:
    if pred is None:
        return None
    total = 0
    correct = 0
    for name in CANONICAL_VARIABLE_ORDER:
        if name in expected:
            total += 1
            correct += int(pred.get(name) == expected[name])
    if total == 0:
        return None
    return correct / total


def _changed_f1(pred: list[str] | None, expected: list[str]) -> float | None:
    if pred is None:
        return None
    pred_set = set(pred)
    exp_set = set(expected)
    tp = len(pred_set & exp_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set - pred_set)
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    if precision is None and recall is None:
        return 1.0 if not pred_set and not exp_set else None
    if precision is None or recall is None or precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _setup_error(pred: dict[str, float] | None, target: dict[str, float] | None) -> float | None:
    if pred is None or target is None:
        return None
    try:
        return sum(abs(float(pred[name]) - float(target[name])) for name in CANONICAL_VARIABLE_ORDER) / len(
            CANONICAL_VARIABLE_ORDER
        )
    except Exception:
        return None


def _image_links(row: dict | None, markdown_path: Path) -> dict[str, str]:
    if row is None or not isinstance(row.get("image_paths"), dict):
        return {}
    out = {}
    base = markdown_path.parent
    for key in ("current_profile", "target_profile", "difference_profile", "composite_profile"):
        value = row["image_paths"].get(key)
        if not isinstance(value, str):
            continue
        path = Path(value)
        if path.exists():
            try:
                out[key] = str(path.relative_to(base))
            except ValueError:
                out[key] = str(path)
        else:
            out[key] = f"missing:{value}"
    return out


def _case_quality_flags(case: dict) -> dict[str, bool]:
    label = case["label"]
    llm = case["llm"]
    local = case["local"]
    category = label["category"]

    llm_dir = case["scores"]["llm_direction_accuracy"]
    local_dir = case["scores"]["local_direction_accuracy"]
    llm_f1 = case["scores"]["llm_changed_f1"]
    local_f1 = case["scores"]["local_changed_f1"]

    if category in {"invalid", "ambiguous_multi_intent"}:
        llm_correct = bool(llm.get("rejected"))
        local_correct = False
    else:
        llm_correct = (llm_dir is not None and llm_dir >= 0.85) and (llm_f1 is not None and llm_f1 >= 0.85)
        local_correct = (local_dir is not None and local_dir >= 0.85) and (local_f1 is not None and local_f1 >= 0.85)

    llm_err = case["scores"]["llm_setup_error"]
    local_err = case["scores"]["local_setup_error"]
    return {
        "llm_correct": llm_correct,
        "local_correct": local_correct,
        "llm_better_understanding": (llm_dir or -1) > (local_dir or -1) + 0.2,
        "local_better_numerically": (
            llm_err is not None and local_err is not None and local_err < llm_err * 0.75
        ),
        "both_correct": llm_correct and local_correct,
        "both_fail": not llm_correct and not local_correct,
    }


def _add_cases(
    selected: list[dict],
    cases: list[dict],
    tag: str,
    predicate,
    count: int,
    *,
    reverse_key=None,
) -> None:
    used = {case["record_id"] for case in selected}
    candidates = [case for case in cases if case["record_id"] not in used and predicate(case)]
    if reverse_key is not None:
        candidates.sort(key=reverse_key, reverse=True)
    for case in candidates[:count]:
        case["case_type"] = tag
        selected.append(case)


def _select_cases(cases: list[dict], max_cases: int) -> list[dict]:
    selected: list[dict] = []
    _add_cases(
        selected,
        cases,
        "LLM clearly better on understanding",
        lambda c: c["flags"]["llm_better_understanding"] and c["label"]["category"] not in {"invalid", "ambiguous_multi_intent"},
        2,
        reverse_key=lambda c: (c["scores"]["llm_direction_accuracy"] or 0) - (c["scores"]["local_direction_accuracy"] or 0),
    )
    _add_cases(
        selected,
        cases,
        "Local clearly better numerically",
        lambda c: c["flags"]["local_better_numerically"] and c["label"]["category"] == "normal_edit",
        2,
        reverse_key=lambda c: (c["scores"]["llm_setup_error"] or 0) - (c["scores"]["local_setup_error"] or 0),
    )
    _add_cases(selected, cases, "Both correct", lambda c: c["flags"]["both_correct"], 2)
    _add_cases(selected, cases, "Both fail", lambda c: c["flags"]["both_fail"], 2)
    _add_cases(selected, cases, "Constraint example", lambda c: c["label"]["category"] == "constraint", 1)
    _add_cases(selected, cases, "Invalid example", lambda c: c["label"]["category"] == "invalid", 1)
    _add_cases(selected, cases, "Ambiguous example", lambda c: c["label"]["category"] == "ambiguous_multi_intent", 2)

    if len(selected) < max_cases:
        used = {case["record_id"] for case in selected}
        for case in cases:
            if case["record_id"] not in used:
                case["case_type"] = "Additional representative case"
                selected.append(case)
            if len(selected) >= max_cases:
                break
    return selected[:max_cases]


def _fmt(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _short_dict(value: dict | None) -> str:
    if value is None:
        return "not available"
    return ", ".join(f"{name}: {value.get(name)}" for name in CANONICAL_VARIABLE_ORDER)


def _interpret(case: dict) -> str:
    label = case["label"]
    scores = case["scores"]
    llm = case["llm"]
    if label["category"] in {"invalid", "ambiguous_multi_intent"}:
        if llm.get("rejected"):
            return "The LLM rejected the request as expected; the local model still produces a numerical setup because it has no rejection head."
        return "The label expects rejection or clarification, but the LLM did not reject and the local model produced a numerical prediction."
    if scores["llm_direction_accuracy"] is not None and scores["local_direction_accuracy"] is not None:
        if scores["llm_direction_accuracy"] > scores["local_direction_accuracy"]:
            return "The LLM matches the labeled direction pattern better, while the local model's inferred delta changes extra or wrong variables."
        if scores["local_direction_accuracy"] > scores["llm_direction_accuracy"]:
            return "The local delta-derived direction pattern is closer to the label than the LLM understanding JSON on this example."
    if scores["llm_setup_error"] is not None and scores["local_setup_error"] is not None:
        if scores["local_setup_error"] < scores["llm_setup_error"]:
            return "The local numerical setup is closer to ground truth even if the qualitative change labels are mixed."
        if scores["llm_setup_error"] < scores["local_setup_error"]:
            return "The LLM numerical setup is closer to ground truth on this case."
    return "Both models show mixed behavior on the available understanding and numerical signals."


def build_case_studies(
    *,
    data_path: Path,
    labels_path: Path,
    llm_predictions_path: Path,
    fallback_llm_predictions_path: Path | None,
    local_proxy_path: Path | None,
    local_model_eval_path: Path,
    variables_config_path: Path,
    markdown_path: Path,
    max_cases: int,
) -> dict:
    data = _rows_by_id(_load_jsonl(data_path), "id")
    labels = _rows_by_id(_load_jsonl(labels_path), "record_id")
    variables_config = load_variables_config(variables_config_path)
    tolerances = load_tolerances(variables_config)

    warnings = []
    prediction_source = llm_predictions_path
    if not prediction_source.exists() and fallback_llm_predictions_path is not None and fallback_llm_predictions_path.exists():
        warnings.append(
            f"Primary LLM predictions missing: {llm_predictions_path}; using fallback {fallback_llm_predictions_path}."
        )
        prediction_source = fallback_llm_predictions_path
    elif not prediction_source.exists():
        warnings.append(f"LLM predictions missing: {llm_predictions_path}.")

    llm_rows = _prediction_rows_by_id(prediction_source) if prediction_source.exists() else {}
    local_examples = _local_examples_by_id(local_model_eval_path)
    local_proxy = _proxy_records_by_id(local_proxy_path)

    all_cases = []
    for record_id, llm_row in llm_rows.items():
        label = labels.get(record_id)
        record = data.get(record_id)
        local_example = local_examples.get(record_id)
        if label is None or record is None:
            warnings.append(f"Skipping LLM prediction without matching Stage-1 label/data: {record_id}")
            continue
        llm = _llm_summary(llm_row, tolerances)
        local = _local_summary(local_example, local_proxy.get(record_id), variables_config, tolerances)
        target_setup = _clean_setup(record.get("target_setup"))
        expected_direction = label.get("expected_change_direction") or {}
        expected_changed = label.get("expected_changed_variables") or []
        case = {
            "record_id": record_id,
            "category": label.get("category"),
            "prompt": label.get("prompt"),
            "task_type": label.get("task_type"),
            "image_links": _image_links(llm_row, markdown_path),
            "label": label,
            "llm": llm,
            "local": local,
            "scores": {
                "llm_direction_accuracy": _direction_accuracy(llm.get("change_direction"), expected_direction),
                "local_direction_accuracy": _direction_accuracy(local.get("change_direction"), expected_direction),
                "llm_changed_f1": _changed_f1(llm.get("changed_variables"), expected_changed),
                "local_changed_f1": _changed_f1(local.get("changed_variables"), expected_changed),
                "llm_setup_error": _setup_error(llm.get("predicted_setup"), target_setup),
                "local_setup_error": _setup_error(local.get("predicted_setup"), target_setup),
            },
        }
        case["flags"] = _case_quality_flags(case)
        case["interpretation"] = _interpret(case)
        all_cases.append(case)

    selected = _select_cases(all_cases, max_cases=max_cases)
    return {
        "metadata": {
            "data_path": str(data_path),
            "labels_path": str(labels_path),
            "llm_predictions_path": str(llm_predictions_path),
            "llm_predictions_used": str(prediction_source) if prediction_source.exists() else None,
            "local_proxy_path": None if local_proxy_path is None else str(local_proxy_path),
            "local_model_eval_path": str(local_model_eval_path),
            "variables_config_path": str(variables_config_path),
            "variable_order": CANONICAL_VARIABLE_ORDER,
        },
        "counts": {
            "labels": len(labels),
            "llm_prediction_rows_available": len(llm_rows),
            "candidate_cases": len(all_cases),
            "selected_cases": len(selected),
        },
        "warnings": warnings,
        "cases": selected,
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Stage 1E Qualitative Case Studies",
        "",
        "## Scope",
        "",
        "This report shows representative Stage-1 understanding cases for senior review. Cases are selected from rows with available fine-tuned LLM predictions and local model predictions.",
        "",
        "## Inputs",
        "",
        f"- Stage-1 benchmark: `{report['metadata']['data_path']}`",
        f"- Stage-1 labels: `{report['metadata']['labels_path']}`",
        f"- LLM predictions requested: `{report['metadata']['llm_predictions_path']}`",
        f"- LLM predictions used: `{report['metadata']['llm_predictions_used']}`",
        f"- Local understanding proxy: `{report['metadata']['local_proxy_path']}`",
        f"- Local model eval: `{report['metadata']['local_model_eval_path']}`",
        "",
        "## Coverage",
        "",
        f"- LLM prediction rows available: `{report['counts']['llm_prediction_rows_available']}`",
        f"- Candidate cases: `{report['counts']['candidate_cases']}`",
        f"- Selected cases: `{report['counts']['selected_cases']}`",
        "",
        "## Warnings",
        "",
    ]
    if report.get("warnings"):
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- No warnings.")
    lines.extend(["", "## Case Studies", ""])

    for idx, case in enumerate(report["cases"], start=1):
        label = case["label"]
        llm = case["llm"]
        local = case["local"]
        scores = case["scores"]
        lines.extend(
            [
                f"### {idx}. {case['case_type']}",
                "",
                f"- record_id: `{case['record_id']}`",
                f"- category: `{case['category']}`",
                f"- task_type: `{case['task_type']}`",
                f"- prompt: {case['prompt']}",
                "",
            ]
        )
        if case["image_links"]:
            lines.extend(["Images:", ""])
            for key, value in case["image_links"].items():
                if value.startswith("missing:"):
                    lines.append(f"- {key}: missing image path `{value.removeprefix('missing:')}`")
                else:
                    lines.append(f"- {key}: [{value}]({value})")
            lines.append("")
        else:
            lines.extend(["Images: not available from the selected LLM prediction row.", ""])

        lines.extend(
            [
                f"- Ground truth changed variables: `{label.get('expected_changed_variables')}`",
                f"- Ground truth directions: `{_short_dict(label.get('expected_change_direction'))}`",
                f"- LLM predicted changed variables: `{llm.get('changed_variables')}`",
                f"- LLM predicted directions: `{_short_dict(llm.get('change_direction'))}`",
                f"- LLM reasoning_summary: {llm.get('reasoning_summary') or 'not available'}",
                f"- Local inferred changed variables: `{local.get('changed_variables')}`",
                f"- Local inferred directions: `{_short_dict(local.get('change_direction'))}`",
                f"- LLM setup error, mean absolute over 7 variables: `{_fmt(scores.get('llm_setup_error'))}`",
                f"- Local setup error, mean absolute over 7 variables: `{_fmt(scores.get('local_setup_error'))}`",
                f"- LLM direction accuracy: `{_fmt(scores.get('llm_direction_accuracy'))}`",
                f"- Local direction accuracy: `{_fmt(scores.get('local_direction_accuracy'))}`",
                f"- Interpretation: {case['interpretation']}",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = build_case_studies(
        data_path=Path(args.data),
        labels_path=Path(args.labels),
        llm_predictions_path=Path(args.llm_predictions),
        fallback_llm_predictions_path=Path(args.fallback_llm_predictions)
        if args.fallback_llm_predictions
        else None,
        local_proxy_path=Path(args.local_proxy) if args.local_proxy else None,
        local_model_eval_path=Path(args.local_model_eval),
        variables_config_path=Path(args.variables_config),
        markdown_path=Path(args.out),
        max_cases=args.max_cases,
    )
    _save_json(report, Path(args.json_out))
    write_markdown(report, Path(args.out))
    print(json.dumps({"out": args.out, "json_out": args.json_out, "counts": report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
