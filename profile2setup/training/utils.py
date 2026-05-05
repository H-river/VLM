"""Utility helpers for profile2setup Stage 5 training."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


TASK_TYPES = ["absolute", "edit", "paired_no_setup"]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def get_device(device_config: str) -> torch.device:
    """Resolve a training device from config."""
    device_name = str(device_config or "auto").lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("optimization.device is 'cuda', but CUDA is not available")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError("optimization.device must be one of: auto, cuda, cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move tensor values in a batch to device, leaving metadata unchanged."""
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def ensure_dir(path) -> Path:
    """Create a directory and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_yaml(path) -> dict:
    """Load a YAML file into a dict."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a dict: {yaml_path}")
    return obj


def save_json(obj, path) -> None:
    """Save a JSON-serializable object with stable formatting."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def count_parameters(model) -> int:
    """Count trainable model parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_task_types_from_records_or_dataset(obj) -> dict:
    """Count known task types from a dataset, records list, or task iterable."""
    if hasattr(obj, "records"):
        records = obj.records
    else:
        records = obj

    counter = Counter()
    for item in records:
        if isinstance(item, dict):
            task_type = item.get("task_type")
        else:
            task_type = item
        if task_type in TASK_TYPES:
            counter[task_type] += 1

    return {task_type: int(counter.get(task_type, 0)) for task_type in TASK_TYPES}


def _value_for_name(values: Any, name: str, index: int) -> float:
    if isinstance(values, dict):
        return float(values.get(name, np.nan))
    if torch.is_tensor(values):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    return float(arr.reshape(-1)[index])


def format_per_variable_table(title, variable_names, values) -> str:
    """Format per-variable scalar metrics as a compact text table."""
    lines = [str(title)]
    for idx, name in enumerate(variable_names):
        value = _value_for_name(values, name, idx)
        if np.isfinite(value):
            value_text = f"{value:.6f}"
        else:
            value_text = "nan"
        lines.append(f"  {name}: {value_text}")
    return "\n".join(lines)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    config,
    vocab,
    metrics,
    variable_order,
) -> None:
    """Save a Stage 5 training checkpoint."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "config": config,
            "vocab": vocab,
            "metrics": metrics,
            "variable_order": list(variable_order),
        },
        out_path,
        _use_new_zipfile_serialization=False,
    )


def load_checkpoint(path, map_location="cpu") -> dict:
    """Load a Stage 5 checkpoint."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=map_location)
