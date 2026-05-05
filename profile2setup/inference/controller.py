"""Checkpoint-backed inference controller for profile2setup v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from profile2setup.inference.routing import route_setup_prediction
from profile2setup.models import build_model_from_config
from profile2setup.schema import VARIABLE_ORDER
from profile2setup.training.dataset import Profile2SetupDataset
from profile2setup.training.normalization import (
    CANONICAL_VARIABLE_ORDER,
    denormalize_delta_vector,
    denormalize_setup_vector,
    get_variable_order,
    load_variables_config,
    make_zero_setup_vector,
)
from profile2setup.training.text import SimpleTokenizer, load_vocab
from profile2setup.training.utils import get_device, load_yaml


FORBIDDEN_V2_FIELDS = {"alignment", "alignment_x", "alignment_y"}


def assert_no_forbidden_v2_fields(obj: Any, prefix: str = "") -> None:
    """Reject forbidden legacy v2 field names in nested dict/list objects."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            path = f"{prefix}.{key_str}" if prefix else key_str
            if key_str in FORBIDDEN_V2_FIELDS:
                raise ValueError(
                    f"Forbidden v2 field detected at {path}; use camera_x/camera_y only"
                )
            assert_no_forbidden_v2_fields(value, path)
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            assert_no_forbidden_v2_fields(item, path)


def _load_checkpoint(path, map_location="cpu") -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dict, got {type(checkpoint).__name__}")
    return checkpoint


def _load_config(checkpoint: dict, config_path=None) -> dict:
    if config_path is not None:
        config = load_yaml(config_path)
    else:
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise ValueError("Checkpoint is missing config; pass config_path to load this checkpoint")
    assert_no_forbidden_v2_fields(config)
    return config


def _load_checkpoint_vocab(checkpoint: dict, config: dict) -> dict:
    vocab = checkpoint.get("vocab")
    if isinstance(vocab, dict):
        return {str(key): int(value) for key, value in vocab.items()}

    vocab_path = checkpoint.get("vocab_path")
    if vocab_path is None:
        vocab_path = (config.get("data") or {}).get("vocab_path")
    if vocab_path is None:
        raise ValueError("Checkpoint is missing vocab and no usable vocab path was found")
    return load_vocab(vocab_path)


def _validate_variable_order(order: Any) -> list[str]:
    if order is None:
        order = list(VARIABLE_ORDER)
    order = list(order)
    if any(str(item) in FORBIDDEN_V2_FIELDS for item in order):
        raise ValueError("Checkpoint variable_order contains forbidden v2 fields")
    if order != list(CANONICAL_VARIABLE_ORDER):
        raise ValueError(
            "Checkpoint variable_order must match canonical profile2setup v2 order: "
            f"{CANONICAL_VARIABLE_ORDER}; got {order}"
        )
    return order


def _vector_to_dict(values) -> dict:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != len(VARIABLE_ORDER):
        raise ValueError(f"Expected vector length {len(VARIABLE_ORDER)}, got {arr.shape[0]}")
    return {name: float(arr[idx]) for idx, name in enumerate(VARIABLE_ORDER)}


def _optional_denormalized_setup(norm_values, active: bool, variables_config: dict) -> dict | None:
    if not active:
        return None
    return denormalize_setup_vector(norm_values, variables_config)


def _optional_denormalized_delta(norm_values, active: bool, variables_config: dict) -> dict | None:
    if not active:
        return None
    return denormalize_delta_vector(norm_values, variables_config)


class Profile2SetupController:
    """Load a trained checkpoint and run one-record routed setup inference."""

    def __init__(
        self,
        checkpoint_path,
        device="auto",
        variables_config_path=None,
        config_path=None,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint = _load_checkpoint(self.checkpoint_path, map_location="cpu")

        if self.checkpoint.get("model_state_dict") is None:
            raise ValueError("Checkpoint is missing model_state_dict")

        self.config = _load_config(self.checkpoint, config_path=config_path)
        self.vocab = _load_checkpoint_vocab(self.checkpoint, self.config)
        self.variable_order = _validate_variable_order(self.checkpoint.get("variable_order"))

        data_cfg = self.config.get("data") or {}
        self.variables_config_path = str(
            variables_config_path
            or data_cfg.get("variables_config")
            or "profile2setup/configs/variables.yaml"
        )
        self.variables_config = load_variables_config(self.variables_config_path)
        if get_variable_order(self.variables_config) != self.variable_order:
            raise ValueError("Variables config order does not match checkpoint variable_order")
        assert_no_forbidden_v2_fields(self.variables_config)

        self.input_size = int(data_cfg.get("input_size", 128))
        self.max_text_len = int(data_cfg.get("max_text_len", 32))
        self.normalize_mode = data_cfg.get("normalize_mode", "max_log")
        self.change_threshold = float(data_cfg.get("change_threshold", 1.0e-6))
        self.tokenizer = SimpleTokenizer(vocab=self.vocab, max_len=self.max_text_len)
        self.zero_setup = make_zero_setup_vector()

        self.device = get_device(device)
        self.model = build_model_from_config(self.config, vocab_size=len(self.vocab)).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

    def _build_one_sample(self, record: dict) -> dict:
        dataset = Profile2SetupDataset.__new__(Profile2SetupDataset)
        dataset.jsonl_path = "<in-memory-record>"
        dataset.variables_config_path = self.variables_config_path
        dataset.input_size = self.input_size
        dataset.max_text_len = self.max_text_len
        dataset.normalize_mode = self.normalize_mode
        dataset.task_filter = None
        dataset.strict = True
        dataset.change_threshold = self.change_threshold
        dataset.variables_config = self.variables_config
        dataset.variable_order = self.variable_order
        dataset.zero_setup = self.zero_setup
        dataset.tokenizer = self.tokenizer
        dataset.records = [dataset._validate_record(record, index=0)]
        return dataset[0]

    def predict_record(self, record: dict) -> dict:
        """Predict absolute, delta, and routed setup for one JSONL record."""
        if not isinstance(record, dict):
            raise ValueError("record must be a dict")
        assert_no_forbidden_v2_fields(record)

        sample = self._build_one_sample(record)
        batch = {
            "profile": sample["profile"].unsqueeze(0).to(self.device),
            "prompt_tokens": sample["prompt_tokens"].unsqueeze(0).to(self.device),
            "current_setup": sample["current_setup"].unsqueeze(0).to(self.device),
            "setup_present": sample["setup_present"].unsqueeze(0).to(self.device),
            "intent_features": sample["intent_features"].unsqueeze(0).to(self.device),
        }

        with torch.no_grad():
            outputs = self.model(
                batch["profile"],
                batch["prompt_tokens"],
                batch["current_setup"],
                setup_present=batch["setup_present"],
                intent_features=batch["intent_features"],
            )
            routed = route_setup_prediction(
                outputs,
                batch["current_setup"],
                batch["setup_present"],
                prefer_absolute_when_setup_missing=True,
            )
            change_confidence = torch.sigmoid(outputs["change_logits"])

        absolute_norm = outputs["absolute"].detach().cpu().numpy()[0]
        delta_norm = outputs["delta"].detach().cpu().numpy()[0]
        routed_norm = routed.detach().cpu().numpy()[0]
        change_conf = change_confidence.detach().cpu().numpy()[0]
        target_setup_norm = sample["target_setup"].detach().cpu().numpy()
        target_delta_norm = sample["target_delta"].detach().cpu().numpy()
        target_present = bool(float(sample["target_present"].detach().cpu().item()) > 0.0)
        delta_present = bool(float(sample["delta_loss_mask"].detach().cpu().item()) > 0.0)
        setup_present = int(float(sample["setup_present"].detach().cpu().item()) > 0.0)

        result = {
            "record_id": sample["record_id"],
            "task_type": sample["task_type"],
            "prompt": sample["prompt"],
            "setup_present": setup_present,
            "predicted_absolute_norm": _vector_to_dict(absolute_norm),
            "predicted_delta_norm": _vector_to_dict(delta_norm),
            "predicted_routed_setup_norm": _vector_to_dict(routed_norm),
            "predicted_absolute_physical": denormalize_setup_vector(absolute_norm, self.variables_config),
            "predicted_delta_physical": denormalize_delta_vector(delta_norm, self.variables_config),
            "predicted_routed_setup_physical": denormalize_setup_vector(routed_norm, self.variables_config),
            "target_setup_physical": _optional_denormalized_setup(
                target_setup_norm,
                target_present,
                self.variables_config,
            ),
            "target_delta_physical": _optional_denormalized_delta(
                target_delta_norm,
                delta_present,
                self.variables_config,
            ),
            "change_confidence": _vector_to_dict(change_conf),
        }
        assert_no_forbidden_v2_fields(result)
        return result
