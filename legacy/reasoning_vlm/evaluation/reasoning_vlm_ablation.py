"""Ablation runner for Level 3 profile2setup reasoning VLM modes."""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from profile2setup.evaluation.closed_loop import (
    _choose_policy_config,
    apply_predicted_setup_to_sim_config,
    compute_closed_loop_profile_metrics,
    load_base_simulator_config_from_metadata,
    resolve_metadata_paths,
    simulate_intensity_from_config,
)
from profile2setup.evaluation.profile_metrics import load_intensity
from profile2setup.inference.controller import (
    Profile2SetupController,
    assert_no_forbidden_v2_fields,
)
from profile2setup.inference.routing import (
    apply_allowed_change_mask_to_delta,
    apply_fixed_change_mask_to_delta,
    route_setup_prediction,
)
from legacy.reasoning_vlm.image_rendering import render_reasoning_images
from legacy.reasoning_vlm.intent_features import (
    build_allowed_change_mask,
    build_fixed_change_mask,
    extract_canonical_prompt,
)
from legacy.reasoning_vlm.vlm_parser import get_reasoning_command
from profile2setup.schema import VARIABLE_ORDER
from profile2setup.training.dataset import filter_records, load_jsonl
from profile2setup.training.normalization import denormalize_setup_vector


ABLATION_MODES = (
    "baseline_raw_prompt",
    "vlm_canonical_prompt",
    "vlm_canonical_prompt_plus_masks",
    "intent_features",
    "intent_features_plus_constraint_loss",
    "full_reasoned_pipeline",
)

_EPS = 1.0e-12


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if torch.is_tensor(obj):
        return _jsonable(obj.detach().cpu().numpy())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    return obj


def _save_json(obj: Any, path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2, sort_keys=True)


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "record"


def _loss_cfg(config: dict) -> dict:
    values = dict(config.get("loss") or {})
    values.update(config.get("losses") or {})
    return values


def _constraint_weight(config: dict) -> float:
    return float(_loss_cfg(config).get("constraint_weight", 0.0) or 0.0)


def _model_uses_intent(controller: Profile2SetupController) -> bool:
    return bool(getattr(controller.model, "use_intent_features", False))


def _records(data_path, task_filter=None, max_examples=None) -> list[dict]:
    records = filter_records(load_jsonl(data_path), task_filter=task_filter)
    if max_examples is not None:
        records = records[: int(max_examples)]
    return records


def _as_numpy_row(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)


def _as_mask_value(tensor: torch.Tensor) -> float:
    return float(tensor.detach().cpu().reshape(-1)[0].item())


def _empty_mode_result(mode: str, status: str = "completed") -> dict:
    return {
        "mode": mode,
        "status": status,
        "num_records_seen": 0,
        "num_predicted": 0,
        "num_prediction_skipped": 0,
        "json_validity_rate": None,
        "absolute_mae": None,
        "delta_mae": None,
        "routed_setup_mae": None,
        "constraint_violation_rate": None,
        "profile_metrics": None,
        "examples": [],
        "skipped_examples": [],
    }


def _metric_block(pred_rows: list[np.ndarray], target_rows: list[np.ndarray], names: list[str]) -> dict | None:
    if not pred_rows:
        return None
    pred = np.stack(pred_rows, axis=0)
    target = np.stack(target_rows, axis=0)
    errors = np.abs(pred - target)
    per_variable = {name: float(np.mean(errors[:, idx])) for idx, name in enumerate(names)}
    return {
        "mean": float(np.mean(errors)),
        "per_variable": per_variable,
        "num_examples": int(pred.shape[0]),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _profile_metric_summary(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    keys = [
        "normalized_mse",
        "centroid_x_error_px",
        "centroid_y_error_px",
        "sigma_x_error_px",
        "sigma_y_error_px",
    ]
    return {
        "num_simulated": int(len(rows)),
        "mean": {
            key: _mean([float(row[key]) for row in rows if row.get(key) is not None])
            for key in keys
        },
    }


def _target_label_available(sample: dict, key: str) -> bool:
    return _as_mask_value(sample[key]) > 0.0


def _constraint_mask(fixed_mask: list[float] | None, allowed_mask: list[float] | None) -> np.ndarray | None:
    if fixed_mask is None and allowed_mask is None:
        return None
    if fixed_mask is None:
        fixed = np.zeros(len(VARIABLE_ORDER), dtype=np.float64)
    else:
        fixed = np.asarray(fixed_mask, dtype=np.float64).reshape(-1)
    if allowed_mask is None:
        allowed = np.ones(len(VARIABLE_ORDER), dtype=np.float64)
    else:
        allowed = np.asarray(allowed_mask, dtype=np.float64).reshape(-1)
    if fixed.shape[0] != len(VARIABLE_ORDER) or allowed.shape[0] != len(VARIABLE_ORDER):
        raise ValueError("constraint masks must match canonical variable count")
    return np.maximum(fixed, 1.0 - allowed)


def _record_constraint_counts(
    delta_norm: np.ndarray,
    fixed_mask: list[float] | None,
    allowed_mask: list[float] | None,
    threshold: float,
) -> tuple[int, int]:
    mask = _constraint_mask(fixed_mask, allowed_mask)
    if mask is None:
        return 0, 0
    constrained = mask > 0.0
    total = int(np.sum(constrained))
    if total <= 0:
        return 0, 0
    violations = int(np.sum(np.abs(delta_norm[constrained]) > float(threshold)))
    return violations, total


def _reasoning_command_for_record(
    *,
    record: dict,
    mode: str,
    reasoning_mode: str,
    rendered_image_paths: dict | None,
    reasoning_json: str | None,
    manual_text: str | None,
) -> tuple[dict | None, str | None]:
    try:
        command = get_reasoning_command(
            mode=reasoning_mode,
            record=record,
            rendered_image_paths=rendered_image_paths,
            json_file=reasoning_json,
            manual_text=manual_text,
        )
        return command, None
    except Exception as exc:  # noqa: BLE001 - ablation reports data failures.
        return None, f"{mode} reasoning failed: {type(exc).__name__}: {exc}"


def _maybe_render_images(record: dict, render_dir: Path, mode: str, record_id: str) -> tuple[dict | None, str | None]:
    current_path = record.get("current_profile_path")
    target_path = record.get("target_profile_path")
    if not current_path or not target_path:
        return None, "missing current_profile_path or target_profile_path for rendering"
    try:
        out_dir = render_dir / mode / _safe_filename(record_id)
        return render_reasoning_images(current_path, target_path, out_dir), None
    except Exception as exc:  # noqa: BLE001
        return None, f"rendering failed: {type(exc).__name__}: {exc}"


def _is_expected_missing_context(reason: str | None) -> bool:
    if reason is None:
        return False
    expected_prefixes = (
        "missing ",
        "target_profile_path does not exist",
        "target_metadata_path does not exist",
    )
    return str(reason).startswith(expected_prefixes)


def _prepare_record(
    raw_record: dict,
    *,
    prompt: str | None,
    reasoning_command: dict | None,
    include_intent: bool,
) -> dict:
    record = copy.deepcopy(raw_record)
    if prompt:
        record["prompt"] = str(prompt)
    for key in ("reasoning_command", "vlm_reasoning", "reasoning_command_path"):
        record.pop(key, None)
    if include_intent and reasoning_command is not None:
        record["reasoning_command"] = reasoning_command
    return record


def _forward_controller(
    controller: Profile2SetupController,
    record: dict,
    *,
    fixed_mask: list[float] | None,
    allowed_mask: list[float] | None,
    apply_masks: bool,
) -> dict:
    sample = controller._build_one_sample(record)
    batch = {
        "profile": sample["profile"].unsqueeze(0).to(controller.device),
        "prompt_tokens": sample["prompt_tokens"].unsqueeze(0).to(controller.device),
        "current_setup": sample["current_setup"].unsqueeze(0).to(controller.device),
        "setup_present": sample["setup_present"].unsqueeze(0).to(controller.device),
        "intent_features": sample["intent_features"].unsqueeze(0).to(controller.device),
    }

    with torch.no_grad():
        outputs = controller.model(
            batch["profile"],
            batch["prompt_tokens"],
            batch["current_setup"],
            setup_present=batch["setup_present"],
            intent_features=batch["intent_features"],
        )
        raw_delta = outputs["delta"]
        final_delta = raw_delta
        if apply_masks:
            if allowed_mask is not None:
                final_delta = apply_allowed_change_mask_to_delta(final_delta, allowed_mask)
            if fixed_mask is not None:
                final_delta = apply_fixed_change_mask_to_delta(final_delta, fixed_mask)
        routed_outputs = dict(outputs)
        routed_outputs["delta"] = final_delta
        routed = route_setup_prediction(
            routed_outputs,
            batch["current_setup"],
            batch["setup_present"],
            prefer_absolute_when_setup_missing=True,
        )

    return {
        "sample": sample,
        "absolute_norm": _as_numpy_row(outputs["absolute"][0]),
        "raw_delta_norm": _as_numpy_row(raw_delta[0]),
        "final_delta_norm": _as_numpy_row(final_delta[0]),
        "routed_setup_norm": _as_numpy_row(routed[0]),
    }


def _simulate_for_record(
    *,
    record: dict,
    predicted_setup_physical: dict,
    simulation_policy: str,
) -> tuple[dict | None, str | None]:
    try:
        paths = resolve_metadata_paths(record)
        if not paths.get("target_profile_path"):
            return None, "missing target_profile_path"
        if not paths.get("target_metadata_path"):
            return None, "missing target_metadata_path"
        target_profile_path = Path(paths["target_profile_path"])
        target_metadata_path = Path(paths["target_metadata_path"])
        if not target_profile_path.exists():
            return None, "target_profile_path does not exist"
        if not target_metadata_path.exists():
            return None, "target_metadata_path does not exist"

        target_config = load_base_simulator_config_from_metadata(target_metadata_path)
        policy_used, base_config, warnings = _choose_policy_config(paths, target_config, simulation_policy)
        sim_config = apply_predicted_setup_to_sim_config(base_config, predicted_setup_physical)
        predicted_intensity = simulate_intensity_from_config(sim_config)
        target_intensity = load_intensity(target_profile_path)
        metrics = compute_closed_loop_profile_metrics(predicted_intensity, target_intensity)
        metrics["simulation_policy_used"] = policy_used
        if warnings:
            metrics["warnings"] = warnings
        return metrics, None
    except Exception as exc:  # noqa: BLE001 - strict handling is done by caller.
        return None, f"simulation failed: {type(exc).__name__}: {exc}"


def _mode_settings(mode: str, controller: Profile2SetupController) -> tuple[dict | None, str | None]:
    intent_supported = _model_uses_intent(controller)
    constraint_weight = _constraint_weight(controller.config)

    if mode == "baseline_raw_prompt":
        return {"reasoning": False, "canonical_prompt": False, "masks": False, "intent": False, "render": False}, None
    if mode == "vlm_canonical_prompt":
        return {"reasoning": True, "canonical_prompt": True, "masks": False, "intent": False, "render": False}, None
    if mode == "vlm_canonical_prompt_plus_masks":
        return {"reasoning": True, "canonical_prompt": True, "masks": True, "intent": False, "render": False}, None
    if mode == "intent_features":
        if not intent_supported:
            return None, "checkpoint/model config does not enable intent_features"
        return {"reasoning": True, "canonical_prompt": True, "masks": False, "intent": True, "render": False}, None
    if mode == "intent_features_plus_constraint_loss":
        if not intent_supported:
            return None, "checkpoint/model config does not enable intent_features"
        if constraint_weight <= 0.0:
            return None, "checkpoint/config has losses.constraint_weight or loss.constraint_weight <= 0"
        return {
            "reasoning": True,
            "canonical_prompt": True,
            "masks": True,
            "intent": True,
            "render": False,
        }, None
    if mode == "full_reasoned_pipeline":
        return {
            "reasoning": True,
            "canonical_prompt": True,
            "masks": True,
            "intent": intent_supported,
            "render": True,
        }, None
    raise ValueError(f"Unknown ablation mode: {mode}")


def _run_mode(
    *,
    mode: str,
    controller: Profile2SetupController,
    records: list[dict],
    reasoning_mode: str,
    reasoning_json: str | None,
    manual_text: str | None,
    render_dir: Path,
    run_simulator: bool,
    simulation_policy: str,
    strict: bool,
    violation_threshold: float,
    max_saved_examples: int,
) -> tuple[dict, dict | None]:
    settings, skip_reason = _mode_settings(mode, controller)
    if settings is None:
        result = _empty_mode_result(mode, status="skipped")
        result["skip_reason"] = skip_reason
        return result, {"mode": mode, "reason": skip_reason}

    result = _empty_mode_result(mode)
    result["num_records_seen"] = len(records)
    result["settings"] = settings
    result["constraint_weight"] = _constraint_weight(controller.config)

    json_attempted = 0
    json_valid = 0
    constraint_violations = 0
    constraint_total = 0
    absolute_pred_rows: list[np.ndarray] = []
    absolute_target_rows: list[np.ndarray] = []
    delta_pred_rows: list[np.ndarray] = []
    delta_target_rows: list[np.ndarray] = []
    routed_pred_rows: list[np.ndarray] = []
    routed_target_rows: list[np.ndarray] = []
    profile_metric_rows: list[dict] = []

    for idx, raw_record in enumerate(records):
        record_id = str(raw_record.get("id") or raw_record.get("record_id") or f"row_{idx}")
        command = None
        rendered = None
        fixed_mask = None
        allowed_mask = None
        prompt = raw_record.get("prompt")

        if settings["render"]:
            rendered, render_error = _maybe_render_images(raw_record, render_dir, mode, record_id)
            if render_error is not None:
                result["num_prediction_skipped"] += 1
                result["skipped_examples"].append({"record_id": record_id, "reason": render_error})
                if strict and not _is_expected_missing_context(render_error):
                    raise RuntimeError(f"{mode} {record_id}: {render_error}")
                continue

        if settings["reasoning"]:
            json_attempted += 1
            command, command_error = _reasoning_command_for_record(
                record=raw_record,
                mode=mode,
                reasoning_mode=reasoning_mode,
                rendered_image_paths=rendered,
                reasoning_json=reasoning_json,
                manual_text=manual_text,
            )
            if command_error is not None:
                result["num_prediction_skipped"] += 1
                result["skipped_examples"].append({"record_id": record_id, "reason": command_error})
                if strict:
                    raise RuntimeError(f"{mode} {record_id}: {command_error}")
                continue
            json_valid += 1
            prompt = extract_canonical_prompt(command) if settings["canonical_prompt"] else raw_record.get("prompt")
            fixed_mask = build_fixed_change_mask(command)
            allowed_mask = build_allowed_change_mask(command)

        model_record = _prepare_record(
            raw_record,
            prompt=prompt,
            reasoning_command=command,
            include_intent=bool(settings["intent"]),
        )

        try:
            prediction = _forward_controller(
                controller,
                model_record,
                fixed_mask=fixed_mask,
                allowed_mask=allowed_mask,
                apply_masks=bool(settings["masks"]),
            )
        except Exception as exc:  # noqa: BLE001
            reason = f"prediction failed: {type(exc).__name__}: {exc}"
            result["num_prediction_skipped"] += 1
            result["skipped_examples"].append({"record_id": record_id, "reason": reason})
            if strict:
                raise
            continue

        sample = prediction["sample"]
        result["num_predicted"] += 1

        if _target_label_available(sample, "absolute_loss_mask"):
            absolute_pred_rows.append(prediction["absolute_norm"])
            absolute_target_rows.append(_as_numpy_row(sample["target_setup"]))
            routed_pred_rows.append(prediction["routed_setup_norm"])
            routed_target_rows.append(_as_numpy_row(sample["target_setup"]))
        if _target_label_available(sample, "delta_loss_mask"):
            delta_pred_rows.append(prediction["final_delta_norm"])
            delta_target_rows.append(_as_numpy_row(sample["target_delta"]))

        violations, total = _record_constraint_counts(
            prediction["final_delta_norm"],
            fixed_mask,
            allowed_mask,
            violation_threshold,
        )
        constraint_violations += violations
        constraint_total += total

        predicted_setup_physical = denormalize_setup_vector(
            prediction["routed_setup_norm"],
            controller.variables_config,
        )

        profile_metrics = None
        if run_simulator:
            profile_metrics, sim_error = _simulate_for_record(
                record=raw_record,
                predicted_setup_physical=predicted_setup_physical,
                simulation_policy=simulation_policy,
            )
            if profile_metrics is not None:
                profile_metric_rows.append(profile_metrics)
            elif sim_error is not None:
                result["skipped_examples"].append({"record_id": record_id, "reason": sim_error})
                if strict and not _is_expected_missing_context(sim_error):
                    raise RuntimeError(f"{mode} {record_id}: {sim_error}")

        if len(result["examples"]) < int(max_saved_examples):
            example = {
                "record_id": record_id,
                "task_type": raw_record.get("task_type"),
                "prompt": raw_record.get("prompt"),
                "effective_prompt": prompt,
                "json_valid": command is not None if settings["reasoning"] else None,
                "used_intent_features": bool(settings["intent"]),
                "applied_masks": bool(settings["masks"]),
                "predicted_routed_setup_norm": {
                    name: float(prediction["routed_setup_norm"][var_idx])
                    for var_idx, name in enumerate(VARIABLE_ORDER)
                },
                "predicted_setup_physical": predicted_setup_physical,
            }
            if fixed_mask is not None:
                example["fixed_change_mask"] = {
                    name: float(fixed_mask[var_idx]) for var_idx, name in enumerate(VARIABLE_ORDER)
                }
            if allowed_mask is not None:
                example["allowed_change_mask"] = {
                    name: float(allowed_mask[var_idx]) for var_idx, name in enumerate(VARIABLE_ORDER)
                }
            if profile_metrics is not None:
                example["profile_metrics"] = profile_metrics
            result["examples"].append(example)

    result["json_validity_rate"] = None if json_attempted == 0 else float(json_valid / json_attempted)
    result["absolute_mae"] = _metric_block(absolute_pred_rows, absolute_target_rows, list(VARIABLE_ORDER))
    result["delta_mae"] = _metric_block(delta_pred_rows, delta_target_rows, list(VARIABLE_ORDER))
    result["routed_setup_mae"] = _metric_block(routed_pred_rows, routed_target_rows, list(VARIABLE_ORDER))
    result["constraint_violation_rate"] = (
        None if constraint_total == 0 else float(constraint_violations / max(constraint_total, 1))
    )
    result["constraint_violation_count"] = int(constraint_violations)
    result["constraint_position_count"] = int(constraint_total)
    result["profile_metrics"] = _profile_metric_summary(profile_metric_rows)
    return result, None


def _summary(mode_results: dict) -> dict:
    rows = []
    for mode, result in mode_results.items():
        if result.get("status") != "completed":
            continue
        profile_mean = ((result.get("profile_metrics") or {}).get("mean") or {})
        rows.append(
            {
                "mode": mode,
                "num_predicted": result.get("num_predicted"),
                "routed_setup_mae_mean": (result.get("routed_setup_mae") or {}).get("mean"),
                "delta_mae_mean": (result.get("delta_mae") or {}).get("mean"),
                "absolute_mae_mean": (result.get("absolute_mae") or {}).get("mean"),
                "constraint_violation_rate": result.get("constraint_violation_rate"),
                "json_validity_rate": result.get("json_validity_rate"),
                "normalized_profile_mse": profile_mean.get("normalized_mse"),
                "centroid_x_error_px": profile_mean.get("centroid_x_error_px"),
                "centroid_y_error_px": profile_mean.get("centroid_y_error_px"),
                "sigma_x_error_px": profile_mean.get("sigma_x_error_px"),
                "sigma_y_error_px": profile_mean.get("sigma_y_error_px"),
            }
        )
    return {
        "mode_table": rows,
        "completed_modes": [row["mode"] for row in rows],
    }


def run_reasoning_vlm_ablation(
    *,
    checkpoint_path,
    data_path,
    out_path=None,
    variables_config_path="profile2setup/configs/variables.yaml",
    config_path=None,
    device="auto",
    reasoning_mode="mock_rule_based",
    reasoning_json=None,
    manual_text=None,
    max_examples=None,
    task_filter=None,
    modes: list[str] | None = None,
    run_simulator=False,
    simulation_policy="target_base",
    render_dir=None,
    strict=True,
    violation_threshold=1.0e-8,
    max_saved_examples=8,
) -> dict:
    """Run Level 3 reasoning ablations over a profile2setup checkpoint."""
    selected_modes = list(modes or ABLATION_MODES)
    unknown_modes = sorted(set(selected_modes) - set(ABLATION_MODES))
    if unknown_modes:
        raise ValueError(f"Unknown ablation modes: {unknown_modes}")

    controller = Profile2SetupController(
        checkpoint_path=checkpoint_path,
        device=device,
        variables_config_path=variables_config_path,
        config_path=config_path,
    )
    records = _records(data_path, task_filter=task_filter, max_examples=max_examples)

    if render_dir is None:
        if out_path is not None:
            render_dir = Path(out_path).with_suffix("").parent / (Path(out_path).with_suffix("").name + "_assets")
        else:
            render_dir = Path("profile2setup/results/reasoning_vlm_ablation_assets")
    render_dir = Path(render_dir)

    mode_results = {}
    skipped_modes = []
    for mode in selected_modes:
        result, skipped = _run_mode(
            mode=mode,
            controller=controller,
            records=records,
            reasoning_mode=reasoning_mode,
            reasoning_json=reasoning_json,
            manual_text=manual_text,
            render_dir=render_dir,
            run_simulator=run_simulator,
            simulation_policy=simulation_policy,
            strict=strict,
            violation_threshold=violation_threshold,
            max_saved_examples=max_saved_examples,
        )
        mode_results[mode] = result
        if skipped is not None:
            skipped_modes.append(skipped)

    report = {
        "config": {
            "checkpoint_path": str(checkpoint_path),
            "data_path": str(data_path),
            "variables_config_path": str(variables_config_path),
            "config_path": None if config_path is None else str(config_path),
            "device": str(controller.device),
            "reasoning_mode": reasoning_mode,
            "reasoning_json": reasoning_json,
            "manual_text_provided": manual_text is not None,
            "max_examples": None if max_examples is None else int(max_examples),
            "task_filter": task_filter,
            "modes": selected_modes,
            "run_simulator": bool(run_simulator),
            "simulation_policy": simulation_policy,
            "render_dir": str(render_dir),
            "strict": bool(strict),
            "violation_threshold": float(violation_threshold),
            "model_uses_intent_features": _model_uses_intent(controller),
            "constraint_weight": _constraint_weight(controller.config),
        },
        "mode_results": mode_results,
        "skipped_modes": skipped_modes,
        "summary": _summary(mode_results),
    }
    report = _jsonable(report)
    assert_no_forbidden_v2_fields(report)
    if out_path is not None:
        _save_json(report, out_path)
    return report
