"""PyTorch dataset and collation utilities for profile2setup v2 Stage 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalization import (
    CANONICAL_VARIABLE_ORDER,
    get_variable_order,
    load_variables_config,
    make_zero_setup_vector,
    normalize_delta_vector,
    normalize_setup_vector,
)
from .preprocessing import make_profile_channels
from .text import SimpleTokenizer, build_vocab_from_jsonl, load_vocab

_FORBIDDEN_KEYS = {"alignment", "alignment_x", "alignment_y"}
_VALID_TASKS = {"absolute", "edit"}


def load_jsonl(path) -> list[dict]:
    """Load JSONL file into a list of dict records."""
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    records = []
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_num}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL record must be an object in {jsonl_path} at line {line_num}"
                )
            records.append(record)
    return records


def filter_records(records, task_filter=None) -> list[dict]:
    """Filter records by task type (None, absolute, edit)."""
    if task_filter is None:
        return list(records)
    if task_filter not in _VALID_TASKS:
        raise ValueError(f"task_filter must be one of {sorted(_VALID_TASKS)} or None")
    return [record for record in records if record.get("task_type") == task_filter]


def _find_forbidden_key(obj: Any, prefix: str = "") -> str | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            path = f"{prefix}.{key_str}" if prefix else key_str
            if key_str in _FORBIDDEN_KEYS:
                return path
            hit = _find_forbidden_key(value, prefix=path)
            if hit is not None:
                return hit
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            hit = _find_forbidden_key(item, prefix=path)
            if hit is not None:
                return hit
    return None


def validate_no_forbidden_fields(record) -> None:
    """Raise if forbidden legacy alignment fields exist anywhere in a record."""
    hit = _find_forbidden_key(record)
    if hit is not None:
        raise ValueError(
            f"Forbidden v2 legacy field detected at '{hit}'. "
            "alignment/alignment_x/alignment_y are not allowed; use camera_x/camera_y."
        )


def _validate_profile_path(path_value: Any, field_name: str) -> None:
    if path_value is None:
        return
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{field_name} must be a non-empty string or None")
    if not path_value.endswith(".npy"):
        raise ValueError(f"{field_name} must point to .npy intensity file, got: {path_value}")
    if not Path(path_value).exists():
        raise FileNotFoundError(f"{field_name} path does not exist: {path_value}")


def _validate_setup_like_dict(setup_like: Any, field_name: str) -> None:
    if not isinstance(setup_like, dict):
        raise ValueError(f"{field_name} must be a dict with canonical v2 variables")

    validate_no_forbidden_fields(setup_like)

    missing = [key for key in CANONICAL_VARIABLE_ORDER if key not in setup_like]
    extra = sorted(set(setup_like.keys()) - set(CANONICAL_VARIABLE_ORDER))

    if missing:
        raise ValueError(f"{field_name} missing required variables: {missing}")
    if extra:
        raise ValueError(f"{field_name} contains non-canonical variables: {extra}")
    for key in CANONICAL_VARIABLE_ORDER:
        value = setup_like[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field_name}.{key} must be numeric")


def _normalize_profile_loss_reference(value: Any) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("profile_loss_reference must be a dict")
    validate_no_forbidden_fields(value)
    return dict(value)


class Profile2SetupDataset(Dataset):
    """Dataset for profile2setup v2 JSONL records."""

    def __init__(
        self,
        jsonl_path,
        variables_config_path="profile2setup/configs/variables.yaml",
        vocab=None,
        vocab_path=None,
        input_size=128,
        max_text_len=32,
        normalize_mode="max_log",
        task_filter=None,
        limit=None,
        strict=True,
        change_threshold=1e-6,
    ):
        self.jsonl_path = str(jsonl_path)
        self.variables_config_path = str(variables_config_path)
        self.input_size = int(input_size)
        self.max_text_len = int(max_text_len)
        self.normalize_mode = normalize_mode
        self.task_filter = task_filter
        self.strict = bool(strict)
        self.change_threshold = float(change_threshold)

        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.max_text_len <= 0:
            raise ValueError("max_text_len must be positive")

        self.variables_config = load_variables_config(self.variables_config_path)
        self.variable_order = get_variable_order(self.variables_config)
        self.zero_setup = make_zero_setup_vector()

        if vocab is not None:
            vocab_data = vocab
        elif vocab_path is not None:
            vocab_data = load_vocab(vocab_path)
        else:
            vocab_data = build_vocab_from_jsonl(self.jsonl_path, min_freq=1)
        self.tokenizer = SimpleTokenizer(vocab=vocab_data, max_len=self.max_text_len)

        raw_records = load_jsonl(self.jsonl_path)
        raw_records = filter_records(raw_records, task_filter=self.task_filter)
        if limit is not None:
            raw_records = raw_records[: int(limit)]

        self.records = []
        for idx, record in enumerate(raw_records):
            try:
                normalized = self._validate_record(record, index=idx)
                self.records.append(normalized)
            except Exception as exc:
                if self.strict:
                    raise ValueError(f"Invalid record at index {idx}: {exc}") from exc

    def _validate_record(self, record: dict, index: int) -> dict:
        if not isinstance(record, dict):
            raise ValueError("record must be a dict")

        validate_no_forbidden_fields(record)

        task_type = record.get("task_type")
        if task_type not in _VALID_TASKS:
            raise ValueError(f"record.task_type must be one of {sorted(_VALID_TASKS)}, got {task_type!r}")

        prompt = record.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("record.prompt must be a string")

        record_id = record.get("id")
        if record_id is None:
            record_id = f"row_{index}"

        target_setup = record.get("target_setup")
        _validate_setup_like_dict(target_setup, "target_setup")

        current_profile_path = record.get("current_profile_path")
        target_profile_path = record.get("target_profile_path")
        _validate_profile_path(target_profile_path, "target_profile_path")

        profile_loss_reference = _normalize_profile_loss_reference(record.get("profile_loss_reference"))

        if task_type == "absolute":
            if current_profile_path is not None:
                raise ValueError("absolute record current_profile_path must be None")
            if record.get("current_setup") is not None:
                raise ValueError("absolute record current_setup must be None")
            if record.get("target_delta") is not None:
                raise ValueError("absolute record target_delta must be None")

        else:  # edit
            _validate_profile_path(current_profile_path, "current_profile_path")
            if current_profile_path is None:
                raise ValueError("edit record current_profile_path is required")
            if target_profile_path is None:
                raise ValueError("edit record target_profile_path is required")

            current_setup = record.get("current_setup")
            target_delta = record.get("target_delta")

            _validate_setup_like_dict(current_setup, "current_setup")
            _validate_setup_like_dict(target_delta, "target_delta")

        return {
            "id": str(record_id),
            "task_type": str(task_type),
            "prompt": prompt,
            "current_profile_path": current_profile_path,
            "target_profile_path": target_profile_path,
            "current_setup": record.get("current_setup"),
            "target_setup": target_setup,
            "target_delta": record.get("target_delta"),
            "profile_loss_reference": profile_loss_reference,
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        task_type = record["task_type"]

        profile_np = make_profile_channels(
            current_path=record.get("current_profile_path"),
            target_path=record.get("target_profile_path"),
            input_size=self.input_size,
            normalize_mode=self.normalize_mode,
        )

        prompt_tokens = self.tokenizer.encode(record.get("prompt", ""))

        if task_type == "absolute":
            current_setup_np = self.zero_setup.copy()
            target_setup_np = normalize_setup_vector(record["target_setup"], self.variables_config)
            target_delta_np = self.zero_setup.copy()
            change_mask_np = self.zero_setup.copy()
        elif task_type == "edit":
            current_setup_np = normalize_setup_vector(record["current_setup"], self.variables_config)
            target_setup_np = normalize_setup_vector(record["target_setup"], self.variables_config)
            target_delta_np = normalize_delta_vector(record["target_delta"], self.variables_config)
            change_mask_np = (np.abs(target_delta_np) > self.change_threshold).astype(np.float32)
        else:  # pragma: no cover
            raise RuntimeError(f"Unexpected task_type in dataset: {task_type}")

        return {
            "profile": torch.as_tensor(profile_np, dtype=torch.float32),
            "prompt_tokens": torch.as_tensor(prompt_tokens, dtype=torch.long),
            "current_setup": torch.as_tensor(current_setup_np, dtype=torch.float32),
            "target_setup": torch.as_tensor(target_setup_np, dtype=torch.float32),
            "target_delta": torch.as_tensor(target_delta_np, dtype=torch.float32),
            "change_mask": torch.as_tensor(change_mask_np, dtype=torch.float32),
            "task_type": record["task_type"],
            "record_id": record["id"],
            "prompt": record["prompt"],
            "current_profile_path": record.get("current_profile_path"),
            "target_profile_path": record.get("target_profile_path"),
            "profile_loss_reference": dict(record.get("profile_loss_reference") or {}),
        }


def profile2setup_collate_fn(batch):
    """Collate function for profile2setup Stage 3 DataLoader."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    tensor_keys = [
        "profile",
        "prompt_tokens",
        "current_setup",
        "target_setup",
        "target_delta",
        "change_mask",
    ]

    list_keys = [
        "task_type",
        "record_id",
        "prompt",
        "current_profile_path",
        "target_profile_path",
        "profile_loss_reference",
    ]

    collated = {}
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch], dim=0)

    for key in list_keys:
        collated[key] = [item[key] for item in batch]

    return collated
