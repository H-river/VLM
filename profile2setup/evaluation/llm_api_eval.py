"""Evaluation for multimodal LLM API profile2setup predictions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from profile2setup.llm_api.validator import (
    validate_all_variable_dicts,
    validate_llm_output,
)
from profile2setup.schema import VARIABLE_ORDER, compute_delta_setup, validate_setup_dict
from profile2setup.evaluation.llm_api_visualization import (
    save_api_visualization_example,
    save_visualization_index,
    simulate_api_prediction,
)

CANONICAL_VARIABLE_ORDER = list(VARIABLE_ORDER)
LEGACY_VARIABLES = {"alignment", "alignment_x", "alignment_y"}
LEGACY_TOKEN_ALTERNATION = "|".join(
    re.escape(token) for token in sorted(LEGACY_VARIABLES, key=len, reverse=True)
)
FALLBACK_TOLERANCES = {
    "source_to_lens": 0.01,
    "lens_to_camera": 0.01,
    "focal_length": 0.005,
    "lens_x": 0.0005,
    "lens_y": 0.0005,
    "camera_x": 0.0005,
    "camera_y": 0.0005,
}
SIMULATION_POLICIES = {"target_base", "current_base", "auto"}


def _load_jsonl(path) -> list[dict]:
    jsonl_path = Path(path)
    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{jsonl_path}:{line_number} record must be a JSON object")
            records.append(record)
    return records


def _save_json(obj: Any, path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _load_variables_config(path) -> dict | None:
    if path is None:
        return None
    try:
        import yaml
    except ImportError:
        return None
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"variables config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"variables config must be a dict: {cfg_path}")
    return obj


def _load_tolerances(variables_config: dict | None) -> dict[str, float]:
    variables = (variables_config or {}).get("variables") or {}
    tolerances = {}
    for name in CANONICAL_VARIABLE_ORDER:
        spec = variables.get(name) or {}
        tolerances[name] = float(spec.get("tolerance", FALLBACK_TOLERANCES[name]))
    return tolerances


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _clean_setup(value: Any) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    if not validate_setup_dict(value):
        return None
    return {name: float(value[name]) for name in CANONICAL_VARIABLE_ORDER}


def _target_delta(record: dict) -> dict | None:
    explicit = _clean_setup(record.get("target_delta"))
    if explicit is not None:
        return explicit
    current = _clean_setup(record.get("current_setup"))
    target = _clean_setup(record.get("target_setup"))
    if current is not None and target is not None:
        return compute_delta_setup(current, target)
    return None


def _pred_delta(prediction: dict | None, data_record: dict) -> dict | None:
    if not isinstance(prediction, dict):
        return None
    explicit = _clean_setup(prediction.get("predicted_delta"))
    if explicit is not None:
        return explicit
    pred_setup = _clean_setup(prediction.get("predicted_setup"))
    current = _clean_setup(data_record.get("current_setup"))
    if pred_setup is not None and current is not None:
        return compute_delta_setup(current, pred_setup)
    return None


def _routed_setup(prediction: dict | None, data_record: dict) -> dict | None:
    if not isinstance(prediction, dict):
        return None
    pred_setup = _clean_setup(prediction.get("predicted_setup"))
    if pred_setup is not None:
        return pred_setup
    delta = _clean_setup(prediction.get("predicted_delta"))
    current = _clean_setup(data_record.get("current_setup"))
    if delta is not None and current is not None:
        return {name: float(current[name] + delta[name]) for name in CANONICAL_VARIABLE_ORDER}
    return None


def _contains_legacy(obj: Any) -> bool:
    pattern = re.compile(rf"(?<![A-Za-z0-9_])({LEGACY_TOKEN_ALTERNATION})(?![A-Za-z0-9_])")
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in LEGACY_VARIABLES:
                return True
            if _contains_legacy(value):
                return True
    elif isinstance(obj, list):
        return any(_contains_legacy(item) for item in obj)
    elif isinstance(obj, str):
        return pattern.search(obj) is not None
    return False


def _json_parse_from_prediction_row(row: dict) -> tuple[Any | None, bool]:
    if isinstance(row.get("prediction"), dict):
        return row["prediction"], True
    raw = row.get("raw_response")
    if not isinstance(raw, str) or not raw.strip():
        return None, False
    try:
        return json.loads(raw), True
    except json.JSONDecodeError:
        return None, False


def _validate_prediction(row: dict) -> tuple[dict | None, bool, bool, bool]:
    parsed, json_valid = _json_parse_from_prediction_row(row)
    if not json_valid:
        return None, False, False, False
    try:
        validate_all_variable_dicts(parsed)
        canonical = True
    except Exception:
        canonical = False
    try:
        return validate_llm_output(parsed), True, True, canonical
    except Exception:
        return None, True, False, canonical


def _changed_set(delta: dict | None, tolerances: dict[str, float]) -> set[str] | None:
    if delta is None:
        return None
    return {
        name
        for name in CANONICAL_VARIABLE_ORDER
        if abs(float(delta[name])) > float(tolerances[name])
    }


def _direction(value: float, tolerance: float) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _infer_fixed_variables(prompt: Any) -> set[str]:
    text = re.sub(r"\s+", " ", str(prompt or "").strip().lower())
    if any(phrase in text for phrase in ("keep camera fixed", "camera fixed", "do not move camera")):
        return {"camera_x", "camera_y"}
    return set()


def _metric_value(metrics: dict | None, *names: str) -> float | None:
    if not isinstance(metrics, dict):
        return None
    for name in names:
        value = metrics.get(name)
        if _is_number(value):
            return float(value)
    return None


def _compare_delta(delta: float | None, threshold: float, negative: str, positive: str) -> str:
    if delta is None:
        return "unknown"
    if delta > threshold:
        return positive
    if delta < -threshold:
        return negative
    return "approximately_unchanged"


def _compare_relative(current: float | None, target: float | None) -> str:
    if current is None or target is None:
        return "unknown"
    delta = target - current
    threshold = max(1e-12, abs(current) * 0.05)
    if delta > threshold:
        return "increases"
    if delta < -threshold:
        return "decreases"
    return "approximately_unchanged"


def _observed_profile_change(record: dict) -> dict[str, str] | None:
    current = record.get("current_metrics")
    target = record.get("target_metrics")
    if not isinstance(current, dict) or not isinstance(target, dict):
        return None

    current_x = _metric_value(current, "centroid_x_px", "centroid_x")
    target_x = _metric_value(target, "centroid_x_px", "centroid_x")
    current_y = _metric_value(current, "centroid_y_px", "centroid_y")
    target_y = _metric_value(target, "centroid_y_px", "centroid_y")
    current_sx = _metric_value(current, "sigma_x_px", "sigma_x")
    target_sx = _metric_value(target, "sigma_x_px", "sigma_x")
    current_sy = _metric_value(current, "sigma_y_px", "sigma_y")
    target_sy = _metric_value(target, "sigma_y_px", "sigma_y")
    centroid_threshold = 0.5 if current_x is not None and target_x is not None else 1e-6
    sigma_threshold = 0.3 if current_sx is not None and target_sx is not None else 1e-6

    return {
        "centroid_x": _compare_delta(
            None if current_x is None or target_x is None else target_x - current_x,
            centroid_threshold,
            "moves_left",
            "moves_right",
        ),
        "centroid_y": _compare_delta(
            None if current_y is None or target_y is None else target_y - current_y,
            centroid_threshold,
            "moves_up",
            "moves_down",
        ),
        "beam_width_x": _compare_delta(
            None if current_sx is None or target_sx is None else target_sx - current_sx,
            sigma_threshold,
            "decreases",
            "increases",
        ),
        "beam_width_y": _compare_delta(
            None if current_sy is None or target_sy is None else target_sy - current_sy,
            sigma_threshold,
            "decreases",
            "increases",
        ),
        "peak_intensity": _compare_relative(
            _metric_value(current, "peak_intensity", "peak"),
            _metric_value(target, "peak_intensity", "peak"),
        ),
        "total_intensity": _compare_relative(
            _metric_value(current, "total_intensity", "total"),
            _metric_value(target, "total_intensity", "total"),
        ),
    }


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator / denominator)


def _init_per_var() -> dict[str, list[float]]:
    return {name: [] for name in CANONICAL_VARIABLE_ORDER}


def _mae_block(errors: dict[str, list[float]]) -> tuple[float | None, dict[str, float | None]]:
    per_variable = {name: _mean(values) for name, values in errors.items()}
    all_values = [value for values in errors.values() for value in values]
    return _mean(all_values), per_variable


def _resolve_path(value: Any, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _first_path(record: dict, top_key: str, ref_key: str, repo_root: Path) -> Path | None:
    value = _resolve_path(record.get(top_key), repo_root)
    if value is not None:
        return value
    ref = record.get("profile_loss_reference")
    if isinstance(ref, dict):
        return _resolve_path(ref.get(ref_key), repo_root)
    return None


def _run_simulator_metric(
    *,
    data_record: dict,
    routed_setup: dict,
    simulation_policy: str,
    repo_root: Path,
) -> tuple[dict | None, str | None]:
    try:
        result = simulate_api_prediction(
            data_record=data_record,
            predicted_setup=routed_setup,
            simulation_policy=simulation_policy,
            repo_root=repo_root,
        )
        metrics = result["profile_metrics"]
        return {
            "normalized_profile_mse": metrics.get("normalized_mse"),
            "centroid_error_px": None
            if metrics.get("centroid_x_error_px") is None or metrics.get("centroid_y_error_px") is None
            else float(
                np.mean(
                    [
                        float(metrics["centroid_x_error_px"]),
                        float(metrics["centroid_y_error_px"]),
                    ]
                )
            ),
            "sigma_error_px": None
            if metrics.get("sigma_x_error_px") is None or metrics.get("sigma_y_error_px") is None
            else float(
                np.mean(
                    [
                        float(metrics["sigma_x_error_px"]),
                        float(metrics["sigma_y_error_px"]),
                    ]
                )
            ),
        }, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def evaluate_llm_api_predictions(
    *,
    predictions_path,
    data_path,
    out_path=None,
    variables_config_path=None,
    run_simulator: bool = False,
    simulation_policy: str = "target_base",
    save_visualizations: bool = False,
    viz_out_dir=None,
    max_viz_examples: int | None = None,
    max_examples: int | None = None,
    repo_root=None,
) -> dict[str, Any]:
    """Evaluate LLM API prediction JSONL against original profile2setup records."""
    if simulation_policy not in SIMULATION_POLICIES:
        raise ValueError(f"simulation_policy must be one of {sorted(SIMULATION_POLICIES)}")
    if save_visualizations and not run_simulator:
        raise ValueError("--save-visualizations requires simulator output; pass --run-simulator")
    if save_visualizations and viz_out_dir is None:
        raise ValueError("viz_out_dir is required when save_visualizations=True")

    variables_config = _load_variables_config(variables_config_path)
    tolerances = _load_tolerances(variables_config)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    data_records = _load_jsonl(data_path)
    predictions = _load_jsonl(predictions_path)
    data_by_id = {record.get("id"): record for record in data_records if record.get("id")}

    counts = {
        "prediction_rows": 0,
        "skipped_rows": 0,
        "evaluated_rows": 0,
        "json_valid": 0,
        "schema_valid": 0,
        "canonical_variables": 0,
        "noncanonical_output_count": 0,
        "legacy_variable_output_count": 0,
        "matched_data_records": 0,
    }

    delta_errors = _init_per_var()
    setup_errors = _init_per_var()
    routed_errors = _init_per_var()
    changed_tp = 0
    changed_fp = 0
    changed_fn = 0
    direction_correct = 0
    direction_total = 0
    fixed_correct = 0
    fixed_total = 0
    fixed_violations = 0
    observed_correct = 0
    observed_total = 0
    simulator_rows: list[dict[str, float | None]] = []
    simulator_errors: list[dict[str, Any]] = []
    visualization_examples: list[dict[str, Any]] = []
    visualization_errors: list[dict[str, Any]] = []
    viz_dir = Path(viz_out_dir) if viz_out_dir is not None else None
    examples: list[dict[str, Any]] = []

    for row in predictions:
        counts["prediction_rows"] += 1
        if row.get("status") == "skipped":
            counts["skipped_rows"] += 1
            continue
        if max_examples is not None and counts["evaluated_rows"] >= int(max_examples):
            break
        counts["evaluated_rows"] += 1

        record_id = row.get("record_id")
        data_record = data_by_id.get(record_id)
        if data_record is None and isinstance(row.get("line_number"), int):
            idx = int(row["line_number"]) - 1
            if 0 <= idx < len(data_records):
                data_record = data_records[idx]
        if data_record is None:
            continue
        counts["matched_data_records"] += 1

        parsed_json, json_valid = _json_parse_from_prediction_row(row)
        if json_valid:
            counts["json_valid"] += 1
        if _contains_legacy(parsed_json if json_valid else row.get("raw_response")):
            counts["legacy_variable_output_count"] += 1

        prediction, _, schema_valid, canonical = _validate_prediction(row)
        if json_valid:
            if canonical:
                counts["canonical_variables"] += 1
            else:
                counts["noncanonical_output_count"] += 1
        if schema_valid:
            counts["schema_valid"] += 1
        if prediction is None:
            continue

        target_setup = _clean_setup(data_record.get("target_setup"))
        target_delta = _target_delta(data_record)
        pred_setup = _clean_setup(prediction.get("predicted_setup"))
        pred_delta = _pred_delta(prediction, data_record)
        routed_setup = _routed_setup(prediction, data_record)

        if pred_delta is not None and target_delta is not None:
            for name in CANONICAL_VARIABLE_ORDER:
                delta_errors[name].append(abs(float(pred_delta[name]) - float(target_delta[name])))

        if pred_setup is not None and target_setup is not None:
            for name in CANONICAL_VARIABLE_ORDER:
                setup_errors[name].append(abs(float(pred_setup[name]) - float(target_setup[name])))

        if routed_setup is not None and target_setup is not None:
            for name in CANONICAL_VARIABLE_ORDER:
                routed_errors[name].append(abs(float(routed_setup[name]) - float(target_setup[name])))

        target_changed = _changed_set(target_delta, tolerances)
        pred_changed = _changed_set(pred_delta, tolerances)
        if target_changed is not None and pred_changed is not None:
            changed_tp += len(target_changed & pred_changed)
            changed_fp += len(pred_changed - target_changed)
            changed_fn += len(target_changed - pred_changed)
            for name in target_changed:
                target_dir = _direction(float(target_delta[name]), tolerances[name])
                pred_dir = _direction(float(pred_delta[name]), tolerances[name])
                direction_total += 1
                if target_dir != 0 and pred_dir == target_dir:
                    direction_correct += 1

        fixed_variables = _infer_fixed_variables(data_record.get("prompt"))
        if fixed_variables and pred_delta is not None:
            for name in fixed_variables:
                fixed_total += 1
                if abs(float(pred_delta[name])) <= tolerances[name]:
                    fixed_correct += 1
                else:
                    fixed_violations += 1

        target_change = _observed_profile_change(data_record)
        predicted_change = prediction.get("observed_profile_change")
        if isinstance(target_change, dict) and isinstance(predicted_change, dict):
            for key, value in target_change.items():
                if key in predicted_change:
                    observed_total += 1
                    if predicted_change.get(key) == value:
                        observed_correct += 1

        sim_metrics = None
        if run_simulator and routed_setup is not None:
            sim_metrics, sim_error = _run_simulator_metric(
                data_record=data_record,
                routed_setup=routed_setup,
                simulation_policy=simulation_policy,
                repo_root=root,
            )
            if sim_metrics is not None:
                simulator_rows.append(sim_metrics)
                if save_visualizations:
                    max_viz = None if max_viz_examples is None else int(max_viz_examples)
                    if max_viz is None or len(visualization_examples) < max_viz:
                        try:
                            sim_result = simulate_api_prediction(
                                data_record=data_record,
                                predicted_setup=routed_setup,
                                simulation_policy=simulation_policy,
                                repo_root=root,
                            )
                            visualization_examples.append(
                                save_api_visualization_example(
                                    data_record=data_record,
                                    prediction_row=row,
                                    parsed_prediction=prediction,
                                    predicted_setup=routed_setup,
                                    predicted_delta=pred_delta,
                                    simulator_result=sim_result,
                                    out_dir=viz_dir,
                                    index=len(visualization_examples),
                                    valid_json=json_valid,
                                    repo_root=root,
                                )
                            )
                        except Exception as exc:  # noqa: BLE001
                            if len(visualization_errors) < 20:
                                visualization_errors.append(
                                    {
                                        "record_id": record_id,
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                )
            elif len(simulator_errors) < 20:
                simulator_errors.append({"record_id": record_id, "error": sim_error})

        if len(examples) < 20:
            examples.append(
                {
                    "record_id": record_id,
                    "task_type": data_record.get("task_type"),
                    "valid_json": bool(json_valid),
                    "schema_valid": bool(schema_valid),
                    "predicted_delta": pred_delta,
                    "predicted_setup": pred_setup,
                    "routed_setup": routed_setup,
                    "target_delta": target_delta,
                    "target_setup": target_setup,
                    "simulator_metrics": sim_metrics,
                }
            )

    delta_mae, per_delta = _mae_block(delta_errors)
    setup_mae, per_setup = _mae_block(setup_errors)
    routed_mae, per_routed = _mae_block(routed_errors)
    precision = _rate(changed_tp, changed_tp + changed_fp)
    recall = _rate(changed_tp, changed_tp + changed_fn)
    f1 = None if precision is None or recall is None or (precision + recall) == 0 else float(
        2 * precision * recall / (precision + recall)
    )

    simulator_metric_names = ["normalized_profile_mse", "centroid_error_px", "sigma_error_px"]
    simulator_metrics = {
        name: _mean([float(row[name]) for row in simulator_rows if row.get(name) is not None])
        for name in simulator_metric_names
    }

    visualization_report = {
        "enabled": bool(save_visualizations),
        "out_dir": None if viz_dir is None else str(viz_dir),
        "num_saved": 0,
        "index_path": None,
        "errors": visualization_errors,
    }
    if save_visualizations:
        saved = save_visualization_index(viz_dir, visualization_examples)
        visualization_report.update(saved)
        visualization_report["errors"] = visualization_errors

    report = {
        "predictions_path": str(predictions_path),
        "data_path": str(data_path),
        "variables_config_path": None if variables_config_path is None else str(variables_config_path),
        "counts": counts,
        "format_metrics": {
            "valid_json_rate": _rate(counts["json_valid"], counts["evaluated_rows"]),
            "schema_valid_rate": _rate(counts["schema_valid"], counts["evaluated_rows"]),
            "canonical_variable_rate": _rate(counts["canonical_variables"], counts["evaluated_rows"]),
        },
        "understanding_metrics": {
            "changed_variable_precision": precision,
            "changed_variable_recall": recall,
            "changed_variable_f1": f1,
            "change_direction_accuracy": _rate(direction_correct, direction_total),
            "fixed_variable_accuracy": _rate(fixed_correct, fixed_total),
            "observed_profile_change_accuracy": _rate(observed_correct, observed_total),
        },
        "numerical_metrics": {
            "predicted_delta_mae": delta_mae,
            "predicted_setup_mae": setup_mae,
            "routed_setup_mae": routed_mae,
            "per_variable_delta_mae": per_delta,
            "per_variable_setup_mae": per_setup,
            "per_variable_routed_setup_mae": per_routed,
        },
        "constraint_metrics": {
            "fixed_variable_violation_rate": _rate(fixed_violations, fixed_total),
            "noncanonical_output_count": counts["noncanonical_output_count"],
            "legacy_variable_output_count": counts["legacy_variable_output_count"],
        },
        "simulator_metrics": {
            "enabled": bool(run_simulator),
            "simulation_policy": simulation_policy,
            "num_success": len(simulator_rows),
            "errors": simulator_errors,
            **simulator_metrics,
        },
        "visualization": visualization_report,
        "tolerances": tolerances,
        "examples": examples,
    }
    if out_path is not None:
        _save_json(report, out_path)
    return report
