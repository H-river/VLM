"""Offline model and predictor evaluation for profile2setup v2."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from profile2setup.evaluation.param_metrics import (
    compute_physical_delta_metrics,
    compute_physical_setup_metrics,
    denormalize_setup_matrix,
    load_tolerances,
    summarize_prediction_metrics,
)
from profile2setup.inference.routing import route_setup_prediction
from profile2setup.models import build_model_from_config
from profile2setup.schema import VARIABLE_ORDER
from profile2setup.training.dataset import Profile2SetupDataset, profile2setup_collate_fn
from profile2setup.training.losses import compute_profile2setup_loss
from profile2setup.training.normalization import load_variables_config
from profile2setup.training.utils import get_device, load_yaml, move_batch_to_device, save_json


TASK_TYPES = ["absolute", "edit", "paired_no_setup"]
_FORBIDDEN_KEYS = {"alignment", "alignment_x", "alignment_y"}


def _load_checkpoint(path, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _check_no_forbidden_keys(obj, prefix: str = "") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            path = f"{prefix}.{key_str}" if prefix else key_str
            if key_str in _FORBIDDEN_KEYS:
                raise ValueError(f"Forbidden v2 result key found at {path}")
            _check_no_forbidden_keys(value, path)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _check_no_forbidden_keys(item, f"{prefix}[{idx}]" if prefix else f"[{idx}]")


def _as_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _metric_average(total: float, count: int) -> float | None:
    if count <= 0:
        return None
    return float(total / count)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _concat(rows: list[np.ndarray], width: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, width), dtype=np.float64)
    return np.concatenate(rows, axis=0).astype(np.float64, copy=False)


def _concat_mask(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 1), dtype=np.float64)
    return np.concatenate(rows, axis=0).astype(np.float64, copy=False)


def _vector_to_dict(row) -> dict:
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    return {name: float(values[idx]) for idx, name in enumerate(VARIABLE_ORDER)}


def _target_delta_or_none(row, active: bool) -> dict | None:
    if not active:
        return None
    return _vector_to_dict(row)


def _loss_from_outputs(outputs, batch, config):
    loss_cfg = dict(config.get("loss") or {})
    loss_cfg.update(config.get("losses") or {})
    return compute_profile2setup_loss(
        outputs,
        batch,
        absolute_weight=loss_cfg.get("absolute_weight", 1.0),
        delta_weight=loss_cfg.get("delta_weight", 1.0),
        change_weight=loss_cfg.get("change_weight", 0.5),
        constraint_weight=loss_cfg.get("constraint_weight", 0.0),
        fixed_change_mask=batch.get("fixed_change_mask"),
    )


def _summaries(pred, target, mask, variables_config, *, is_delta: bool) -> tuple[dict, dict]:
    normalized = summarize_prediction_metrics(pred, target, mask=mask, variable_order=VARIABLE_ORDER)
    if is_delta:
        physical = compute_physical_delta_metrics(pred, target, variables_config, mask=mask)
    else:
        physical = compute_physical_setup_metrics(pred, target, variables_config, mask=mask)
    return normalized, physical


def _task_type_counts(task_types: list[str]) -> dict:
    counter = Counter(task_types)
    return {name: int(counter.get(name, 0)) for name in TASK_TYPES}


def _task_mask(task_types: list[str], task_type: str) -> np.ndarray:
    return np.asarray([1.0 if value == task_type else 0.0 for value in task_types], dtype=np.float64).reshape(-1, 1)


def _combine_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    return (mask_a.reshape(-1, 1) * mask_b.reshape(-1, 1)).astype(np.float64)


def _build_metric_block(
    absolute_pred,
    delta_pred,
    routed_pred,
    target_setup,
    target_delta,
    absolute_mask,
    delta_mask,
    variables_config,
) -> tuple[dict, dict]:
    norm_abs, phys_abs = _summaries(absolute_pred, target_setup, absolute_mask, variables_config, is_delta=False)
    norm_delta, phys_delta = _summaries(delta_pred, target_delta, delta_mask, variables_config, is_delta=True)
    norm_routed, phys_routed = _summaries(routed_pred, target_setup, absolute_mask, variables_config, is_delta=False)
    return (
        {
            "absolute": norm_abs,
            "delta": norm_delta,
            "routed_setup": norm_routed,
        },
        {
            "absolute": phys_abs,
            "delta": phys_delta,
            "routed_setup": phys_routed,
        },
    )


def _build_per_task_metrics(collected: dict, variables_config) -> dict:
    per_task = {}
    for task_type in TASK_TYPES:
        task_mask = _task_mask(collected["task_types"], task_type)
        abs_mask = _combine_masks(collected["absolute_mask"], task_mask)
        delta_mask = _combine_masks(collected["delta_mask"], task_mask)
        normalized, physical = _build_metric_block(
            collected["absolute_pred"],
            collected["delta_pred"],
            collected["routed_pred"],
            collected["target_setup"],
            collected["target_delta"],
            abs_mask,
            delta_mask,
            variables_config,
        )
        per_task[task_type] = {
            "num_examples": int(task_mask.sum()),
            "normalized_metrics": normalized,
            "physical_metrics": physical,
        }
    return per_task


def _build_examples(collected: dict, variables_config, max_examples: int) -> list[dict]:
    if max_examples <= 0:
        return []
    count = min(max_examples, len(collected["record_ids"]))
    routed_physical = denormalize_setup_matrix(collected["routed_pred"][:count], variables_config)
    target_setup_physical = denormalize_setup_matrix(collected["target_setup"][:count], variables_config)
    examples = []
    for idx in range(count):
        delta_active = bool(float(collected["delta_mask"][idx].reshape(-1)[0]) > 0.0)
        examples.append(
            {
                "record_id": collected["record_ids"][idx],
                "task_type": collected["task_types"][idx],
                "prompt": collected["prompts"][idx],
                "setup_present": int(float(collected["setup_present"][idx].reshape(-1)[0]) > 0.0),
                "predicted_absolute_norm": _vector_to_dict(collected["absolute_pred"][idx]),
                "predicted_delta_norm": _vector_to_dict(collected["delta_pred"][idx]),
                "predicted_routed_setup_norm": _vector_to_dict(collected["routed_pred"][idx]),
                "target_setup_norm": _vector_to_dict(collected["target_setup"][idx]),
                "target_delta_norm": _target_delta_or_none(collected["target_delta"][idx], delta_active),
                "predicted_routed_setup_physical": _vector_to_dict(routed_physical[idx]),
                "target_setup_physical": _vector_to_dict(target_setup_physical[idx]),
            }
        )
    return examples


def _print_metric_table(title: str, metric_block: dict) -> None:
    print(title)
    mae = metric_block.get("mae") or {}
    units = metric_block.get("units") or {}
    for name in VARIABLE_ORDER:
        value = mae.get(name)
        unit = units.get(name)
        value_text = "none" if value is None else f"{float(value):.6g}"
        suffix = f" {unit}" if unit else ""
        print(f"  {name}: {value_text}{suffix}")


def evaluate_outputs_over_loader(
    predictor_fn: Callable[[dict], dict],
    loader,
    variables_config,
    config,
    device,
    max_examples=None,
) -> dict:
    """Evaluate model-like outputs from a predictor over a DataLoader."""
    totals = {
        "loss": 0.0,
        "absolute_loss": 0.0,
        "delta_loss": 0.0,
        "change_loss": 0.0,
        "constraint_loss": 0.0,
    }
    seen = 0
    rows = {
        "absolute_pred": [],
        "delta_pred": [],
        "routed_pred": [],
        "target_setup": [],
        "target_delta": [],
        "absolute_mask": [],
        "delta_mask": [],
        "setup_present": [],
        "record_ids": [],
        "task_types": [],
        "prompts": [],
    }

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = predictor_fn(batch)
            routed = route_setup_prediction(
                outputs,
                batch["current_setup"],
                batch["setup_present"],
                prefer_absolute_when_setup_missing=True,
            )
            loss_dict = _loss_from_outputs(outputs, batch, config)

            batch_size = int(batch["profile"].shape[0])
            seen += batch_size
            for key in totals:
                totals[key] += _as_float(loss_dict[key]) * batch_size

            rows["absolute_pred"].append(_to_numpy(outputs["absolute"]))
            rows["delta_pred"].append(_to_numpy(outputs["delta"]))
            rows["routed_pred"].append(_to_numpy(routed))
            rows["target_setup"].append(_to_numpy(batch["target_setup"]))
            rows["target_delta"].append(_to_numpy(batch["target_delta"]))
            rows["absolute_mask"].append(_to_numpy(batch["absolute_loss_mask"]))
            rows["delta_mask"].append(_to_numpy(batch["delta_loss_mask"]))
            rows["setup_present"].append(_to_numpy(batch["setup_present"]))
            rows["record_ids"].extend(batch["record_id"])
            rows["task_types"].extend(batch["task_type"])
            rows["prompts"].extend(batch["prompt"])

    collected = {
        "absolute_pred": _concat(rows["absolute_pred"], len(VARIABLE_ORDER)),
        "delta_pred": _concat(rows["delta_pred"], len(VARIABLE_ORDER)),
        "routed_pred": _concat(rows["routed_pred"], len(VARIABLE_ORDER)),
        "target_setup": _concat(rows["target_setup"], len(VARIABLE_ORDER)),
        "target_delta": _concat(rows["target_delta"], len(VARIABLE_ORDER)),
        "absolute_mask": _concat_mask(rows["absolute_mask"]),
        "delta_mask": _concat_mask(rows["delta_mask"]),
        "setup_present": _concat_mask(rows["setup_present"]),
        "record_ids": rows["record_ids"],
        "task_types": rows["task_types"],
        "prompts": rows["prompts"],
    }

    normalized_metrics, physical_metrics = _build_metric_block(
        collected["absolute_pred"],
        collected["delta_pred"],
        collected["routed_pred"],
        collected["target_setup"],
        collected["target_delta"],
        collected["absolute_mask"],
        collected["delta_mask"],
        variables_config,
    )
    max_examples = 32 if max_examples is None else int(max_examples)
    result = {
        "num_examples": int(seen),
        "task_type_counts": _task_type_counts(collected["task_types"]),
        "loss": {
            "total": _metric_average(totals["loss"], seen),
            "absolute": _metric_average(totals["absolute_loss"], seen),
            "delta": _metric_average(totals["delta_loss"], seen),
            "change": _metric_average(totals["change_loss"], seen),
            "constraint": _metric_average(totals["constraint_loss"], seen),
        },
        "normalized_metrics": normalized_metrics,
        "physical_metrics": physical_metrics,
        "per_task_type": _build_per_task_metrics(collected, variables_config),
        "examples": _build_examples(collected, variables_config, max_examples),
    }
    result["tolerances"] = load_tolerances(variables_config)
    _check_no_forbidden_keys(result)
    return result


def _build_dataset(data_path, config, vocab, variables_config_path, task_filter=None, strict=True) -> Profile2SetupDataset:
    data_cfg = config.get("data") or {}
    return Profile2SetupDataset(
        jsonl_path=data_path,
        variables_config_path=variables_config_path,
        vocab=vocab,
        input_size=data_cfg.get("input_size", 128),
        max_text_len=data_cfg.get("max_text_len", 32),
        normalize_mode=data_cfg.get("normalize_mode", "max_log"),
        task_filter=task_filter,
        limit=None,
        strict=strict,
        change_threshold=data_cfg.get("change_threshold", 1.0e-6),
    )


def _build_loader(dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=profile2setup_collate_fn,
    )


def _load_eval_config(checkpoint: dict, config_path=None) -> dict:
    if config_path is not None:
        return load_yaml(config_path)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing config; pass --config to evaluate this checkpoint")
    return config


def _validate_checkpoint_order(checkpoint: dict) -> None:
    order = checkpoint.get("variable_order")
    if order is not None and list(order) != list(VARIABLE_ORDER):
        raise ValueError(f"Checkpoint variable_order must be canonical v2 order: {VARIABLE_ORDER}; got {order}")


def evaluate_checkpoint(
    checkpoint_path,
    data_path,
    out_path=None,
    config_path=None,
    variables_config_path=None,
    batch_size=None,
    device="auto",
    max_examples=None,
    task_filter=None,
    strict=True,
) -> dict:
    """Load a Stage 5 checkpoint and evaluate it on a JSONL dataset."""
    checkpoint_path = str(checkpoint_path)
    data_path = str(data_path)
    checkpoint = _load_checkpoint(checkpoint_path, map_location="cpu")
    _validate_checkpoint_order(checkpoint)
    config = _load_eval_config(checkpoint, config_path=config_path)
    vocab = checkpoint.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError("Checkpoint is missing vocab; Stage 6 evaluation requires checkpoint['vocab']")
    variables_config_path = variables_config_path or (config.get("data") or {}).get(
        "variables_config", "profile2setup/configs/variables.yaml"
    )
    variables_config = load_variables_config(variables_config_path)
    resolved_device = get_device(device)
    dataset = _build_dataset(
        data_path,
        config,
        vocab,
        variables_config_path,
        task_filter=task_filter,
        strict=strict,
    )
    if len(dataset) == 0:
        raise RuntimeError("Evaluation dataset is empty after filtering/validation")
    batch_size = batch_size or (config.get("optimization") or {}).get("batch_size", 32)
    loader = _build_loader(dataset, int(batch_size))

    model = build_model_from_config(config, vocab_size=len(vocab)).to(resolved_device)
    model_state = checkpoint.get("model_state_dict")
    if model_state is None:
        raise ValueError("Checkpoint is missing model_state_dict")
    model.load_state_dict(model_state)
    model.eval()

    def predictor(batch):
        return model(
            batch["profile"],
            batch["prompt_tokens"],
            batch["current_setup"],
            setup_present=batch["setup_present"],
            intent_features=batch.get("intent_features"),
        )

    metrics = evaluate_outputs_over_loader(
        predictor,
        loader,
        variables_config,
        config,
        resolved_device,
        max_examples=max_examples,
    )
    result = {
        "checkpoint_path": checkpoint_path,
        "data_path": data_path,
        **metrics,
    }
    _check_no_forbidden_keys(result)
    if out_path is not None:
        save_json(result, out_path)

    print(f"checkpoint path: {checkpoint_path}")
    print(f"data path: {data_path}")
    print(f"number of examples: {result['num_examples']}")
    print(f"task type counts: {result['task_type_counts']}")
    print(f"total loss: {result['loss']['total']:.6f}")
    _print_metric_table("routed setup MAE physical", result["physical_metrics"]["routed_setup"])
    _print_metric_table("delta MAE physical", result["physical_metrics"]["delta"])
    _print_metric_table("absolute MAE physical", result["physical_metrics"]["absolute"])
    return result
