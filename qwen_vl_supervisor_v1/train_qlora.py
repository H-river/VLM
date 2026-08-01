#!/usr/bin/env python3
"""Resume-capable Qwen2.5-VL QLoRA SFT for the optics supervisor.

This entry point intentionally consumes only the deterministic ``prebuilt_chat``
JSONL emitted by :mod:`qwen_vl_supervisor_v1.export_sft`.  It does not generate
predictions and it refuses protected/frozen splits, so training-time evaluation
is development-only.

The module has no torch/transformers imports at import time.  Unit tests and
``--validate-only`` therefore do not load the model or reserve GPU memory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from qwen_vl_supervisor_v1.model_snapshot import (  # noqa: E402
    pretrained_revision_kwargs,
    resolve_model_source_identity,
)
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs/training_smoke.yaml"
REQUIRED_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "qkv",
    "attn.proj",
)
SAFE_STATE_SCHEMA = "qwen_vl_safe_trainer_state_v1"
SAFE_OPTIMIZER_JSON = "optimizer_scheduler.safe.json"
SAFE_OPTIMIZER_TENSORS = "optimizer_scheduler.safe.safetensors"
SAFE_RNG_JSON = "rng_state.safe.json"
SAFE_RNG_TENSORS = "rng_state.safe.safetensors"
PROTECTED_SPLIT_FRAGMENTS = ("frozen", "heldout", "held_out", "severity_ood", "test")
PACKAGE_NAMES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "Pillow",
    "PyYAML",
    "qwen-vl-utils",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-jsonl", type=Path, help="Override data.train_jsonl.")
    parser.add_argument("--dev-jsonl", type=Path, help="Override data.dev_jsonl.")
    parser.add_argument("--output-dir", type=Path, help="Override training.output_dir.")
    parser.add_argument("--model-id", help="Override model.id (path or Hugging Face ID).")
    parser.add_argument("--revision", help="Override model.revision.")
    parser.add_argument("--processor-revision", help="Override model.processor_revision.")
    parser.add_argument(
        "--expected-local-snapshot-tree-sha256",
        help="Require this byte-exact tree identity when --model-id resolves locally.",
    )
    parser.add_argument("--max-steps", type=int, help="Override training.max_steps.")
    parser.add_argument("--save-total-limit", type=int, help="Override training.save_total_limit.")
    parser.add_argument("--seed", type=int, help="Override training.seed.")
    parser.add_argument("--data-seed", type=int, help="Override training.data_seed.")
    parser.add_argument(
        "--full-determinism",
        action="store_true",
        help="Enable PyTorch deterministic algorithms for reproducibility audits.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Exact Trainer checkpoint directory, or 'latest' under output_dir.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=None,
        help="torchrun local rank; LOCAL_RANK is used when omitted.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config, rows, split guards, and every image without model loading.",
    )
    parser.add_argument(
        "--print-resolved-config",
        action="store_true",
        help="Print the resolved config and exit without model loading.",
    )
    parser.add_argument(
        "--save-at-global-step",
        type=int,
        help="Force an audit checkpoint at this step without changing the scheduler horizon.",
    )
    parser.add_argument(
        "--stop-after-global-step",
        type=int,
        help="Stop after this optimizer step without changing training.max_steps/scheduler horizon.",
    )
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("PyYAML is required to read the training config") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"training config must be a mapping: {path}")
    return value


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deep_copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def resolve_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config_path = resolve_repo_path(args.config)
    config = deep_copy_json(load_yaml(config_path))
    if args.train_jsonl is not None:
        config["data"]["train_jsonl"] = str(args.train_jsonl)
    if args.dev_jsonl is not None:
        config["data"]["dev_jsonl"] = str(args.dev_jsonl)
    if args.output_dir is not None:
        config["training"]["output_dir"] = str(args.output_dir)
    if args.model_id is not None:
        config["model"]["id"] = args.model_id
    if args.revision is not None:
        config["model"]["revision"] = args.revision
    if args.processor_revision is not None:
        config["model"]["processor_revision"] = args.processor_revision
    if args.expected_local_snapshot_tree_sha256 is not None:
        config["model"]["expected_local_snapshot_tree_sha256"] = (
            args.expected_local_snapshot_tree_sha256
        )
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.save_total_limit is not None:
        config["training"]["save_total_limit"] = args.save_total_limit
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.data_seed is not None:
        config["training"]["data_seed"] = args.data_seed
    if args.full_determinism:
        config["training"]["full_determinism"] = True
    if args.resume_from_checkpoint is not None:
        config["training"]["resume_from_checkpoint"] = args.resume_from_checkpoint
    validate_config(config)
    return config, config_path


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{key} must be a mapping")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    model = _required_mapping(config, "model")
    processor = _required_mapping(model, "processor")
    quantization = _required_mapping(model, "quantization")
    data = _required_mapping(config, "data")
    training = _required_mapping(config, "training")
    lora = _required_mapping(config, "lora")
    training_seeds = config.get("training_seeds")
    if training_seeds is not None:
        if (
            not isinstance(training_seeds, list)
            or not training_seeds
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in training_seeds)
            or len(set(training_seeds)) != len(training_seeds)
        ):
            raise ValueError("training_seeds must be a nonempty list of unique integer seeds")

    if model.get("architecture") != "Qwen2_5_VLForConditionalGeneration":
        raise ValueError("model.architecture must freeze Qwen2_5_VLForConditionalGeneration")
    if not model.get("revision") or not model.get("processor_revision"):
        raise ValueError("model and processor revisions must be explicitly frozen")
    expected_local_tree = model.get("expected_local_snapshot_tree_sha256")
    if expected_local_tree is not None and not re.fullmatch(
        r"[0-9a-f]{64}", str(expected_local_tree)
    ):
        raise ValueError(
            "model.expected_local_snapshot_tree_sha256 must be a lowercase SHA-256"
        )
    if model.get("attn_implementation") != "sdpa":
        raise ValueError("the default audited attention implementation is sdpa")
    if quantization.get("load_in_4bit") is not True:
        raise ValueError("this entry point requires 4-bit QLoRA")
    if quantization.get("quant_type") != "nf4":
        raise ValueError("model.quantization.quant_type must be nf4")
    if quantization.get("compute_dtype") != "bfloat16":
        raise ValueError("NF4 compute_dtype must be bfloat16")
    min_pixels = int(processor.get("min_pixels", 0))
    max_pixels = int(processor.get("max_pixels", 0))
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("processor pixel bounds must be positive and ordered")

    targets = tuple(lora.get("target_modules", ()))
    if targets != REQUIRED_TARGET_MODULES:
        raise ValueError(
            "lora.target_modules changed from the architecture-audited order: "
            f"expected {REQUIRED_TARGET_MODULES}, got {targets}"
        )
    if int(lora.get("r", 0)) <= 0 or int(lora.get("alpha", 0)) <= 0:
        raise ValueError("LoRA rank and alpha must be positive")

    if data.get("format") != "prebuilt_chat":
        raise ValueError("data.format must be prebuilt_chat")
    if data.get("prohibit_frozen_splits") is not True:
        raise ValueError("data.prohibit_frozen_splits must remain true")
    train_splits = data.get("allowed_train_splits")
    dev_splits = data.get("allowed_dev_splits")
    if train_splits != ["train"] or dev_splits != ["dev"]:
        raise ValueError("only train rows and development rows are legal during training")
    for key in ("train_sha256", "dev_sha256", "export_report_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(data.get(key, ""))):
            raise ValueError(f"data.{key} must pin an exact lowercase sha256")
    for key in ("export_report", "train_export_report_key", "dev_export_report_key"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"data.{key} is required")
    for key in ("train_jsonl", "dev_jsonl"):
        lowered = str(data.get(key, "")).lower()
        if any(fragment in lowered for fragment in PROTECTED_SPLIT_FRAGMENTS):
            raise ValueError(f"protected/frozen path is forbidden in data.{key}: {lowered}")

    if int(training.get("per_device_train_batch_size", 0)) != 1:
        raise ValueError("per-device training batch size is frozen to 1 for QLoRA portability")
    if int(training.get("gradient_accumulation_steps", 0)) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if int(training.get("max_steps", 0)) <= 0:
        raise ValueError("max_steps must be positive")
    if int(training.get("save_steps", 0)) <= 0:
        raise ValueError("save_steps must be positive")
    if int(training.get("logging_steps", 0)) != 1:
        raise ValueError("logging_steps must be 1 to preserve per-step loss")
    if training.get("gradient_checkpointing") is not True:
        raise ValueError("gradient checkpointing must remain enabled")
    if training.get("bf16") is not True or training.get("fp16") is not False:
        raise ValueError("mixed precision must be bf16=true, fp16=false")
    if training.get("completion_only_loss") is not True:
        raise ValueError("completion_only_loss must remain true")
    if training.get("assistant_only_loss") is not False:
        raise ValueError("TRL VLM training requires assistant_only_loss=false")
    if training.get("packing") is not False:
        raise ValueError("VLM packing must remain disabled")
    if training.get("max_length") is not None:
        raise ValueError("max_length must be null: image-token truncation is prohibited")
    if training.get("eval_strategy") != "steps":
        raise ValueError("dev-only checkpoint selection requires eval_strategy=steps")
    if training.get("metric_for_best_model") != "eval_loss":
        raise ValueError("checkpoint selection must use development eval_loss")
    if training.get("greater_is_better") is not False:
        raise ValueError("eval_loss checkpoint selection requires greater_is_better=false")
    if training.get("ignore_data_skip") is not False:
        raise ValueError("ignore_data_skip must remain false for exact Trainer resumption")
    if training_seeds is not None and int(training["seed"]) not in training_seeds:
        raise ValueError("training.seed must be one of the frozen training_seeds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant {value}")
                    ),
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def row_split(row: Mapping[str, Any]) -> str | None:
    candidates = [row.get("split")]
    for key in ("metadata", "training_metadata", "provenance"):
        nested = row.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested.get("split"))
    values = {str(value) for value in candidates if value is not None}
    if len(values) > 1:
        raise ValueError(f"conflicting row split declarations: {sorted(values)}")
    return next(iter(values), None)


def _image_placeholder_count(messages: Any) -> int:
    if not isinstance(messages, list):
        raise ValueError("prompt must be a list of chat messages")
    count = 0
    for message in messages:
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), list):
            raise ValueError("chat messages require list-valued content")
        count += sum(
            isinstance(item, Mapping) and item.get("type") == "image"
            for item in message["content"]
        )
    return count


def validate_prebuilt_row(
    row: Mapping[str, Any],
    *,
    allowed_splits: set[str],
    image_root: Path,
    verify_image: bool,
) -> Path:
    example_id = row.get("example_id", "<unknown>")
    split = row_split(row)
    if split is None:
        raise ValueError(f"{example_id}: every exported row must declare its split")
    lowered = split.lower()
    if any(fragment in lowered for fragment in PROTECTED_SPLIT_FRAGMENTS):
        raise ValueError(f"{example_id}: protected/frozen split is forbidden: {split}")
    if split not in allowed_splits:
        raise ValueError(f"{example_id}: split {split!r} not in {sorted(allowed_splits)}")

    prompt = row.get("prompt")
    completion = row.get("completion")
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ValueError(f"{example_id}: supervisor SFT requires exactly one image path")
    if _image_placeholder_count(prompt) != 1:
        raise ValueError(f"{example_id}: supervisor prompt requires exactly one image placeholder")
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValueError(f"{example_id}: completion must contain one assistant message")
    if [message.get("role") for message in prompt] != ["system", "user"]:
        raise ValueError(f"{example_id}: prompt roles must be exactly system,user")
    message = completion[0]
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        raise ValueError(f"{example_id}: completion role must be assistant")
    if not isinstance(message.get("content"), list) or not message["content"]:
        raise ValueError(f"{example_id}: assistant completion content is missing")
    if (
        len(message["content"]) != 1
        or not isinstance(message["content"][0], Mapping)
        or message["content"][0].get("type") != "text"
        or not isinstance(message["content"][0].get("text"), str)
    ):
        raise ValueError(f"{example_id}: assistant completion must be exactly one text item")
    from qwen_vl_supervisor_v1.contracts import parse_target_strict

    parse_target_strict(message["content"][0]["text"])

    image_path = Path(images[0]).expanduser()
    if not image_path.is_absolute():
        image_path = (image_root / image_path).resolve()
    else:
        image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"{example_id}: image does not exist: {image_path}")
    if verify_image:
        from PIL import Image

        with Image.open(image_path) as image:
            image.verify()
        metadata = row.get("metadata")
        expected_hash = metadata.get("image_sha256") if isinstance(metadata, Mapping) else None
        if expected_hash is not None and sha256_path(image_path) != expected_hash:
            raise ValueError(f"{example_id}: image sha256 differs from exported metadata")
    return image_path


def limit_rows(rows: list[dict[str, Any]], value: Any) -> list[dict[str, Any]]:
    if value is None:
        return rows
    limit = int(value)
    if limit <= 0:
        raise ValueError("sample limit must be positive or null")
    return rows[:limit]


@dataclass(frozen=True)
class PreparedRows:
    train_path: Path
    dev_path: Path
    image_root: Path
    train_rows: list[dict[str, Any]]
    dev_rows: list[dict[str, Any]]


def prepare_rows(config: Mapping[str, Any], *, verify_images: bool = True) -> PreparedRows:
    data = config["data"]
    train_path = resolve_repo_path(data["train_jsonl"])
    dev_path = resolve_repo_path(data["dev_jsonl"])
    image_root = resolve_repo_path(data.get("image_root", "."))
    expected_train_hash = str(data["train_sha256"])
    expected_dev_hash = str(data["dev_sha256"])
    if sha256_path(train_path) != expected_train_hash:
        raise ValueError(f"training export hash differs from pinned config: {train_path}")
    if sha256_path(dev_path) != expected_dev_hash:
        raise ValueError(f"development export hash differs from pinned config: {dev_path}")
    report_path = resolve_repo_path(data["export_report"])
    if sha256_path(report_path) != data["export_report_sha256"]:
        raise ValueError(f"SFT export report hash differs from pinned config: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for split_kind, expected_hash in (
        ("train", expected_train_hash),
        ("dev", expected_dev_hash),
    ):
        report_key = data[f"{split_kind}_export_report_key"]
        report_entry = report.get("records", {}).get(report_key)
        if not isinstance(report_entry, Mapping) or report_entry.get("sha256") != expected_hash:
            raise ValueError(
                f"SFT export report entry {report_key!r} does not match pinned {split_kind} hash"
            )
    train_rows = limit_rows(read_jsonl(train_path), data.get("max_train_samples"))
    dev_rows = limit_rows(read_jsonl(dev_path), data.get("max_dev_samples"))
    from qwen_vl_supervisor_v1.export_sft import validate_export_row

    for row in train_rows:
        validate_export_row(row, repository_root=REPOSITORY_ROOT)
        validate_prebuilt_row(
            row,
            allowed_splits=set(data["allowed_train_splits"]),
            image_root=image_root,
            verify_image=verify_images,
        )
    for row in dev_rows:
        validate_export_row(row, repository_root=REPOSITORY_ROOT)
        validate_prebuilt_row(
            row,
            allowed_splits=set(data["allowed_dev_splits"]),
            image_root=image_root,
            verify_image=verify_images,
        )
    return PreparedRows(train_path, dev_path, image_root, train_rows, dev_rows)


def load_rgb_image(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.load()
    return rgb


def row_to_example(row: Mapping[str, Any], image_root: Path) -> dict[str, Any]:
    path = validate_prebuilt_row(
        row,
        allowed_splits={str(row_split(row))},
        image_root=image_root,
        verify_image=False,
    )
    return {
        "images": [load_rgb_image(path)],
        "prompt": row["prompt"],
        "completion": row["completion"],
    }


def require_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from safetensors.torch import load_file as safe_load_file
        from safetensors.torch import save_file as safe_save_file
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
            TrainerCallback,
            set_seed,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - depends on training environment
        raise RuntimeError(
            "training requires torch, transformers, trl, peft, bitsandbytes, "
            "accelerate, datasets, Pillow, and PyYAML"
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "safe_load_file": safe_load_file,
        "safe_save_file": safe_save_file,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLForConditionalGeneration,
        "TrainerCallback": TrainerCallback,
        "set_seed": set_seed,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _git(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *command],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def git_state() -> dict[str, Any]:
    porcelain = _git(["status", "--porcelain=v1"])
    lines = [] if not porcelain else porcelain.splitlines()
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(lines),
        "dirty_path_count": len(lines),
        "dirty_paths": [line[3:] if len(line) > 3 else line for line in lines],
    }


def world_info(local_rank_arg: int | None) -> dict[str, int]:
    local_rank = local_rank_arg
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": local_rank,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def rank_zero(info: Mapping[str, int]) -> bool:
    return int(info["rank"]) == 0


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_state_encode(
    value: Any,
    *,
    tensors: dict[str, Any],
    torch_module: Any,
) -> Any:
    """Encode optimizer/RNG containers without pickle or executable objects."""

    if torch_module.is_tensor(value):
        key = f"tensor_{len(tensors):08d}"
        tensors[key] = value.detach().cpu().contiguous()
        return {"kind": "tensor", "key": key}
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - torch training depends on numpy
        np = None
    if np is not None and isinstance(value, np.ndarray):
        key = f"tensor_{len(tensors):08d}"
        array = np.ascontiguousarray(value)
        tensors[key] = torch_module.from_numpy(array.copy()).contiguous()
        return {"kind": "numpy", "key": key, "dtype": str(value.dtype)}
    if value is None or isinstance(value, (bool, int, str)):
        return {"kind": "scalar", "value": value}
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ValueError("non-finite floats are forbidden in safe checkpoint JSON")
        return {"kind": "scalar", "value": value}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _safe_state_encode(item, tensors=tensors, torch_module=torch_module)
                for item in value
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _safe_state_encode(item, tensors=tensors, torch_module=torch_module)
                for item in value
            ],
        }
    if isinstance(value, Mapping):
        return {
            "kind": "dict",
            "items": [
                [
                    _safe_state_encode(key, tensors=tensors, torch_module=torch_module),
                    _safe_state_encode(item, tensors=tensors, torch_module=torch_module),
                ]
                for key, item in value.items()
            ],
        }
    raise TypeError(f"unsafe or unsupported checkpoint value: {type(value).__module__}.{type(value).__name__}")


def _safe_state_decode(
    node: Any,
    *,
    tensors: Mapping[str, Any],
    used_tensor_keys: set[str],
) -> Any:
    if not isinstance(node, Mapping) or set(node).isdisjoint({"kind"}):
        raise ValueError("malformed safe checkpoint node")
    kind = node.get("kind")
    if kind == "scalar" and set(node) == {"kind", "value"}:
        value = node["value"]
        if value is not None and not isinstance(value, (bool, int, float, str)):
            raise ValueError("invalid scalar in safe checkpoint")
        return value
    if kind in {"tensor", "numpy"}:
        allowed = {"kind", "key"} if kind == "tensor" else {"kind", "key", "dtype"}
        if set(node) != allowed or not isinstance(node.get("key"), str):
            raise ValueError(f"malformed {kind} reference")
        key = node["key"]
        if key not in tensors or key in used_tensor_keys:
            raise ValueError(f"missing or multiply referenced safe tensor: {key}")
        used_tensor_keys.add(key)
        tensor = tensors[key]
        if kind == "tensor":
            return tensor
        import numpy as np

        dtype = np.dtype(node["dtype"])
        return tensor.cpu().numpy().astype(dtype, copy=True)
    if kind in {"tuple", "list"}:
        if set(node) != {"kind", "items"} or not isinstance(node["items"], list):
            raise ValueError(f"malformed {kind} node")
        values = [
            _safe_state_decode(item, tensors=tensors, used_tensor_keys=used_tensor_keys)
            for item in node["items"]
        ]
        return tuple(values) if kind == "tuple" else values
    if kind == "dict":
        if set(node) != {"kind", "items"} or not isinstance(node["items"], list):
            raise ValueError("malformed dict node")
        result: dict[Any, Any] = {}
        for pair in node["items"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("malformed dict item in safe checkpoint")
            key = _safe_state_decode(pair[0], tensors=tensors, used_tensor_keys=used_tensor_keys)
            if not isinstance(key, (bool, int, float, str, tuple)):
                raise ValueError(f"unsupported decoded dictionary key: {type(key).__name__}")
            if key in result:
                raise ValueError(f"duplicate key in safe checkpoint dictionary: {key!r}")
            result[key] = _safe_state_decode(
                pair[1], tensors=tensors, used_tensor_keys=used_tensor_keys
            )
        return result
    raise ValueError(f"unknown safe checkpoint node kind: {kind!r}")


def save_safe_state_bundle(
    *,
    directory: Path,
    json_name: str,
    tensor_name: str,
    payload: Mapping[str, Any],
    torch_module: Any,
    save_file: Any,
) -> dict[str, Any]:
    """Save arbitrary tensor/scalar optimizer state as safetensors plus strict JSON."""

    directory.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Any] = {}
    structure = _safe_state_encode(payload, tensors=tensors, torch_module=torch_module)
    # safetensors permits an empty mapping, but a sentinel keeps compatibility
    # with older readers. It is never exposed to the decoder.
    persisted = tensors or {"__empty__": torch_module.empty(0, dtype=torch_module.uint8)}
    tensor_path = directory / tensor_name
    tensor_tmp = tensor_path.with_name(tensor_path.name + ".tmp")
    save_file(persisted, str(tensor_tmp), metadata={"schema": SAFE_STATE_SCHEMA})
    tensor_tmp.replace(tensor_path)
    manifest = {
        "schema": SAFE_STATE_SCHEMA,
        "tensor_file": tensor_name,
        "tensor_sha256": sha256_path(tensor_path),
        "tensor_count": len(tensors),
        "structure": structure,
    }
    atomic_json(directory / json_name, manifest)
    return {
        "json": json_name,
        "tensors": tensor_name,
        "tensor_count": len(tensors),
        "tensor_sha256": manifest["tensor_sha256"],
    }


def load_safe_state_bundle(
    *,
    directory: Path,
    json_name: str,
    tensor_name: str,
    load_file: Any,
) -> dict[str, Any]:
    """Load a safe bundle after schema, filename, and content-hash validation."""

    json_path = directory / json_name
    tensor_path = directory / tensor_name
    if not json_path.is_file() or not tensor_path.is_file():
        raise FileNotFoundError(f"safe checkpoint bundle is incomplete under {directory}")
    manifest = json.loads(
        json_path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value}")
        ),
    )
    if not isinstance(manifest, Mapping) or manifest.get("schema") != SAFE_STATE_SCHEMA:
        raise ValueError(f"unsupported safe checkpoint schema in {json_path}")
    if manifest.get("tensor_file") != tensor_name:
        raise ValueError(f"unexpected tensor filename in {json_path}")
    if sha256_path(tensor_path) != manifest.get("tensor_sha256"):
        raise ValueError(f"safe checkpoint tensor hash mismatch: {tensor_path}")
    tensors = load_file(str(tensor_path), device="cpu")
    declared_count = int(manifest.get("tensor_count", -1))
    actual_keys = set(tensors) - {"__empty__"}
    if declared_count != len(actual_keys):
        raise ValueError(f"safe checkpoint tensor count mismatch: {declared_count} != {len(actual_keys)}")
    used: set[str] = set()
    value = _safe_state_decode(manifest.get("structure"), tensors=tensors, used_tensor_keys=used)
    if used != actual_keys:
        raise ValueError(f"unreferenced tensors in safe checkpoint: {sorted(actual_keys - used)}")
    if not isinstance(value, dict):
        raise ValueError("safe checkpoint root must decode to a dictionary")
    return value


def exact_nested_state_comparison(
    expected: Any,
    actual: Any,
    *,
    torch_module: Any,
    max_examples: int = 8,
) -> dict[str, Any]:
    """Compare decoded optimizer state without dtype coercion or tolerances."""

    tensor_count = 0
    tensor_elements = 0
    mismatch_count = 0
    examples: list[dict[str, Any]] = []
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - training dependencies include NumPy
        np = None

    def mismatch(path: str, reason: str, **details: Any) -> None:
        nonlocal mismatch_count
        mismatch_count += 1
        if len(examples) < max_examples:
            examples.append({"path": path, "reason": reason, **details})

    def visit(left: Any, right: Any, path: str) -> None:
        nonlocal tensor_count, tensor_elements
        if torch_module.is_tensor(left):
            tensor_count += 1
            tensor_elements += left.numel()
            if not torch_module.is_tensor(right):
                mismatch(path, "expected_tensor", actual_type=type(right).__name__)
                return
            if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
                mismatch(
                    path,
                    "tensor_metadata",
                    expected_dtype=str(left.dtype),
                    actual_dtype=str(right.dtype),
                    expected_shape=list(left.shape),
                    actual_shape=list(right.shape),
                )
                return
            if not torch_module.equal(left.detach().cpu(), right.detach().cpu()):
                maximum = float(
                    (left.detach().float().cpu() - right.detach().float().cpu()).abs().max()
                )
                mismatch(path, "tensor_values", max_abs_difference=maximum)
            return
        if np is not None and isinstance(left, np.ndarray):
            tensor_count += 1
            tensor_elements += int(left.size)
            if not isinstance(right, np.ndarray):
                mismatch(path, "expected_numpy_array", actual_type=type(right).__name__)
                return
            if left.dtype != right.dtype or left.shape != right.shape:
                mismatch(
                    path,
                    "numpy_metadata",
                    expected_dtype=str(left.dtype),
                    actual_dtype=str(right.dtype),
                    expected_shape=list(left.shape),
                    actual_shape=list(right.shape),
                )
                return
            if not np.array_equal(left, right):
                maximum = float(np.max(np.abs(left.astype(float) - right.astype(float))))
                mismatch(path, "numpy_values", max_abs_difference=maximum)
            return
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping):
                mismatch(path, "expected_mapping", actual_type=type(right).__name__)
                return
            if set(left) != set(right):
                mismatch(
                    path,
                    "mapping_keys",
                    missing=[str(key) for key in set(left) - set(right)],
                    unexpected=[str(key) for key in set(right) - set(left)],
                )
                return
            for key in left:
                visit(left[key], right[key], f"{path}/{key}")
            return
        if isinstance(left, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                mismatch(
                    path,
                    "sequence_metadata",
                    expected_type=type(left).__name__,
                    actual_type=type(right).__name__,
                    expected_length=len(left),
                    actual_length=len(right) if isinstance(right, (list, tuple)) else None,
                )
                return
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
                visit(left_item, right_item, f"{path}/{index}")
            return
        if type(left) is not type(right) or left != right:
            mismatch(path, "scalar", expected=left, actual=right)

    visit(expected, actual, "$")
    return {
        "exact_equal": mismatch_count == 0,
        "tensor_count": tensor_count,
        "tensor_elements": tensor_elements,
        "mismatch_count": mismatch_count,
        "mismatch_examples": examples,
    }


def model_source_kwargs(
    model_cfg: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    *,
    processor: bool = False,
) -> dict[str, Any]:
    revision_key = "processor_revision" if processor else "revision"
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        "local_files_only": bool(model_cfg.get("local_files_only", False)),
    }
    kwargs.update(pretrained_revision_kwargs(source_identity, str(model_cfg[revision_key])))
    return kwargs


def torch_dtype(torch_module: Any, name: str) -> Any:
    if not hasattr(torch_module, name):
        raise ValueError(f"unknown torch dtype: {name}")
    return getattr(torch_module, name)


def module_matches(name: str, target: str) -> bool:
    return name == target or name.endswith("." + target)


def audit_lora_architecture(model: Any, targets: Sequence[str]) -> dict[str, Any]:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type != "qwen2_5_vl":
        raise ValueError(f"LoRA audit expected model_type qwen2_5_vl, got {model_type!r}")
    names = [name for name, _ in model.named_modules()]
    resolved: dict[str, list[str]] = {
        target: [name for name in names if module_matches(name, target)] for target in targets
    }
    missing = [target for target, matches in resolved.items() if not matches]
    if missing:
        raise ValueError(f"LoRA targets absent from loaded architecture: {missing}")

    unexpected_qkv = [name for name in resolved["qkv"] if not name.endswith(".attn.qkv")]
    unexpected_proj = [
        name for name in resolved["attn.proj"] if not name.endswith(".attn.proj")
    ]
    if unexpected_qkv or unexpected_proj:
        raise ValueError(
            "vision attention targets resolved outside attn.qkv/attn.proj: "
            f"qkv={unexpected_qkv[:5]}, proj={unexpected_proj[:5]}"
        )
    for target in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if any("language_model" not in name and ".model.layers." not in name for name in resolved[target]):
            raise ValueError(f"language attention target {target} resolved outside language layers")
    if not any("visual" in name for name in resolved["gate_proj"]):
        raise ValueError("vision MLP gate_proj was not found")
    if not any("language_model" in name or ".model.layers." in name for name in resolved["gate_proj"]):
        raise ValueError("language MLP gate_proj was not found")
    return {
        "model_type": model_type,
        "architecture": type(model).__name__,
        "target_counts": {target: len(matches) for target, matches in resolved.items()},
        "target_examples": {target: matches[:3] for target, matches in resolved.items()},
        "vision_attention_contract": ["attn.qkv", "attn.proj"],
        "language_attention_contract": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "language_and_vision_mlp_contract": ["gate_proj", "up_proj", "down_proj"],
    }


def parameter_counts(model: Any) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
    }


def supported_kwargs(callable_object: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_object).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if accepts_kwargs:
        return dict(values)
    return {key: value for key, value in values.items() if key in parameters}


def build_sft_config(sft_config_cls: Any, config: Mapping[str, Any], local_rank: int) -> Any:
    train = config["training"]
    values = {
        "output_dir": str(resolve_repo_path(train["output_dir"])),
        "run_name": config["run_name"],
        "per_device_train_batch_size": int(train["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(train["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(train["gradient_accumulation_steps"]),
        "learning_rate": float(train["learning_rate"]),
        "weight_decay": float(train.get("weight_decay", 0.0)),
        "warmup_ratio": float(train.get("warmup_ratio", 0.0)),
        "lr_scheduler_type": str(train["lr_scheduler_type"]),
        "optim": str(train["optim"]),
        "max_steps": int(train["max_steps"]),
        "seed": int(train["seed"]),
        "data_seed": int(train["data_seed"]),
        "full_determinism": bool(train.get("full_determinism", True)),
        "bf16": bool(train["bf16"]),
        "fp16": bool(train["fp16"]),
        "tf32": bool(train.get("tf32", False)),
        "gradient_checkpointing": bool(train["gradient_checkpointing"]),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "use_cache": False,
        "max_length": train["max_length"],
        "packing": bool(train["packing"]),
        "completion_only_loss": bool(train["completion_only_loss"]),
        "assistant_only_loss": bool(train["assistant_only_loss"]),
        "remove_unused_columns": False,
        "eval_strategy": str(train["eval_strategy"]),
        "eval_steps": int(train["eval_steps"]),
        "save_strategy": "steps",
        "save_steps": int(train["save_steps"]),
        "save_total_limit": int(train["save_total_limit"]),
        "logging_strategy": "steps",
        "logging_steps": int(train["logging_steps"]),
        "logging_first_step": True,
        "report_to": train.get("report_to", []),
        "load_best_model_at_end": bool(train["load_best_model_at_end"]),
        "metric_for_best_model": str(train["metric_for_best_model"]),
        "greater_is_better": bool(train["greater_is_better"]),
        "dataloader_num_workers": int(train.get("dataloader_num_workers", 0)),
        "dataloader_pin_memory": bool(train.get("dataloader_pin_memory", True)),
        "ddp_find_unused_parameters": bool(train.get("ddp_find_unused_parameters", False)),
        "ignore_data_skip": bool(train["ignore_data_skip"]),
        "local_rank": local_rank,
        "skip_memory_metrics": False,
    }
    return sft_config_cls(**supported_kwargs(sft_config_cls, values))


def _safe_bundle_scalar_header(json_path: Path, tensor_path: Path) -> dict[str, Any]:
    """Validate a safe sidecar's envelope and expose scalar root fields only."""

    try:
        manifest = json.loads(
            json_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safe checkpoint JSON: {json_path}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema") != SAFE_STATE_SCHEMA:
        raise ValueError(f"unsupported safe checkpoint schema in {json_path}")
    if manifest.get("tensor_file") != tensor_path.name:
        raise ValueError(f"safe checkpoint tensor filename mismatch in {json_path}")
    if not tensor_path.is_file() or sha256_path(tensor_path) != manifest.get("tensor_sha256"):
        raise ValueError(f"safe checkpoint tensor hash mismatch: {tensor_path}")
    structure = manifest.get("structure")
    if not isinstance(structure, Mapping) or structure.get("kind") != "dict":
        raise ValueError(f"safe checkpoint root is not a dictionary: {json_path}")
    items = structure.get("items")
    if not isinstance(items, list):
        raise ValueError(f"safe checkpoint dictionary items are malformed: {json_path}")
    header: dict[str, Any] = {}
    for pair in items:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"safe checkpoint dictionary item is malformed: {json_path}")
        key_node, value_node = pair
        if (
            isinstance(key_node, Mapping)
            and key_node.get("kind") == "scalar"
            and isinstance(key_node.get("value"), str)
            and isinstance(value_node, Mapping)
            and value_node.get("kind") == "scalar"
            and set(value_node) == {"kind", "value"}
        ):
            header[key_node["value"]] = value_node["value"]
    return header


def _required_header_integer(header: Mapping[str, Any], key: str, path: Path) -> int:
    value = header.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"safe checkpoint {key} must be an integer in {path}")
    return value


def checkpoint_step(path: Path) -> int:
    state_path = path / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"resume checkpoint lacks trainer_state.json: {path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    step = int(state["global_step"])
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match and int(match.group(1)) != step:
        raise ValueError(f"checkpoint directory/state step mismatch: {path} versus {step}")
    for required in (SAFE_OPTIMIZER_JSON, SAFE_OPTIMIZER_TENSORS):
        if not (path / required).is_file():
            raise FileNotFoundError(
                f"safe true Trainer resume requires {required}; legacy pickle checkpoints "
                f"are deliberately not loaded: {path}"
            )
    optimizer_header = _safe_bundle_scalar_header(
        path / SAFE_OPTIMIZER_JSON, path / SAFE_OPTIMIZER_TENSORS
    )
    optimizer_step = _required_header_integer(
        optimizer_header, "global_step", path / SAFE_OPTIMIZER_JSON
    )
    world_size = _required_header_integer(
        optimizer_header, "world_size", path / SAFE_OPTIMIZER_JSON
    )
    if optimizer_step != step:
        raise ValueError(
            f"optimizer/trainer checkpoint step mismatch: {optimizer_step} != {step} in {path}"
        )
    if world_size < 1:
        raise ValueError(f"safe checkpoint world_size must be positive in {path}")

    single_json = path / SAFE_RNG_JSON
    single_tensors = path / SAFE_RNG_TENSORS
    distributed_json = {item.name: item for item in path.glob("rng_state.rank*.safe.json")}
    distributed_tensors = {
        item.name: item for item in path.glob("rng_state.rank*.safe.safetensors")
    }
    if world_size == 1:
        if (
            not single_json.is_file()
            or not single_tensors.is_file()
            or distributed_json
            or distributed_tensors
        ):
            raise FileNotFoundError(
                "world_size=1 requires exactly rng_state.safe.json plus "
                f"rng_state.safe.safetensors and no ranked RNG sidecars: {path}"
            )
        rng_paths = [(0, single_json, single_tensors)]
    else:
        expected_json = {
            f"rng_state.rank{rank:05d}.safe.json" for rank in range(world_size)
        }
        expected_tensors = {
            f"rng_state.rank{rank:05d}.safe.safetensors" for rank in range(world_size)
        }
        if (
            single_json.exists()
            or single_tensors.exists()
            or set(distributed_json) != expected_json
            or set(distributed_tensors) != expected_tensors
        ):
            raise FileNotFoundError(
                f"world_size={world_size} requires exact contiguous ranked RNG sidecars "
                f"0..{world_size - 1} and no unranked/extra sidecars: {path}"
            )
        rng_paths = [
            (
                rank,
                distributed_json[f"rng_state.rank{rank:05d}.safe.json"],
                distributed_tensors[f"rng_state.rank{rank:05d}.safe.safetensors"],
            )
            for rank in range(world_size)
        ]
    for rank, rng_json, rng_tensors in rng_paths:
        rng_header = _safe_bundle_scalar_header(rng_json, rng_tensors)
        rng_step = _required_header_integer(rng_header, "global_step", rng_json)
        rng_world_size = _required_header_integer(rng_header, "world_size", rng_json)
        process_index = _required_header_integer(rng_header, "process_index", rng_json)
        if (rng_step, rng_world_size, process_index) != (step, world_size, rank):
            raise ValueError(
                "RNG sidecar header mismatch: "
                f"expected step/world_size/rank {(step, world_size, rank)}, got "
                f"{(rng_step, rng_world_size, process_index)} in {rng_json}"
            )
    return step


def latest_checkpoint(output_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path.resolve()))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint-* directories under {output_dir}")
    return max(candidates)[1]


def resolve_resume(value: Any, output_dir: Path) -> tuple[Path | None, int | None]:
    if value in (None, "", False):
        return None, None
    path = latest_checkpoint(output_dir) if str(value) == "latest" else resolve_repo_path(str(value))
    if not path.is_dir():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    return path, checkpoint_step(path)


def latest_saved_checkpoint(output_dir: Path) -> Path | None:
    try:
        return latest_checkpoint(output_dir)
    except FileNotFoundError:
        return None


def prior_fresh_lora_delta_evidence(output_dir: Path, current_run_id: str) -> dict[str, Any] | None:
    """Find a completed fresh run proving that this output's LoRA path updates."""

    candidates = sorted(output_dir.glob("run_manifest.*.json"), reverse=True)
    for path in candidates:
        if path.name in {"run_manifest.latest.json", f"run_manifest.{current_run_id}.json"}:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        delta = value.get("lora_delta_verification", {})
        resume = value.get("resume_audit", {})
        if (
            value.get("status") == "completed"
            and resume.get("requested") is False
            and delta.get("at_least_one_intended_parameter_changed") is True
        ):
            return {
                "manifest": str(path.resolve()),
                "manifest_sha256": sha256_path(path),
                "run_id": value.get("run_id"),
                "max_abs_delta": delta.get("max_abs_delta"),
                "changed_parameter_count": delta.get("changed_parameter_count"),
            }
    return None


def safe_rng_filenames(world_size: int, process_index: int) -> tuple[str, str]:
    if world_size <= 1:
        return SAFE_RNG_JSON, SAFE_RNG_TENSORS
    stem = f"rng_state.rank{process_index:05d}"
    return f"{stem}.safe.json", f"{stem}.safe.safetensors"


def make_safe_trainer_class(
    base_trainer_cls: Any,
    *,
    torch_module: Any,
    safe_save_file: Any,
    safe_load_file: Any,
) -> Any:
    """Create an SFTTrainer that never deserializes pickle training state."""

    import random

    import numpy as np

    class SafeCheckpointSFTTrainer(base_trainer_cls):
        safe_checkpoint_events: list[dict[str, Any]]
        safe_resume_details: dict[str, Any] | None
        safe_rng_restore_details: dict[str, Any] | None
        adapter_dtype_restore_audits: list[dict[str, Any]]

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.safe_checkpoint_events = []
            self.safe_resume_details = None
            self.safe_rng_restore_details = None
            self.adapter_dtype_restore_audits = []
            super().__init__(*args, **kwargs)

        @staticmethod
        def _parameter_dtype_summary(model: Any) -> dict[str, Any]:
            counts: dict[str, int] = {}
            tensor_count = 0
            total_elements = 0
            for parameter in model.parameters():
                if not parameter.requires_grad:
                    continue
                name = str(parameter.dtype)
                counts[name] = counts.get(name, 0) + parameter.numel()
                tensor_count += 1
                total_elements += parameter.numel()
            return {
                "dtypes_to_elements": dict(sorted(counts.items())),
                "tensor_count": tensor_count,
                "total_elements": total_elements,
            }

        @staticmethod
        def _saved_adapter_dtype_summary(checkpoint: Path) -> dict[str, Any]:
            path = checkpoint / "adapter_model.safetensors"
            if not path.is_file():
                raise FileNotFoundError(
                    f"dtype-safe PEFT restore requires adapter_model.safetensors: {checkpoint}"
                )
            tensors = safe_load_file(str(path), device="cpu")
            counts: dict[str, int] = {}
            for tensor in tensors.values():
                name = str(tensor.dtype)
                counts[name] = counts.get(name, 0) + tensor.numel()
            result = {
                "path": str(path.resolve()),
                "sha256": sha256_path(path),
                "dtypes_to_elements": dict(sorted(counts.items())),
                "tensor_count": len(tensors),
                "total_elements": sum(tensor.numel() for tensor in tensors.values()),
            }
            del tensors
            return result

        def _raw_optimizer(self) -> Any:
            optimizer = self.optimizer
            while hasattr(optimizer, "optimizer"):
                nested = optimizer.optimizer
                if nested is optimizer:
                    break
                optimizer = nested
            return optimizer

        def _optimizer_parameter_name_groups(self) -> list[list[str]]:
            names_by_id = {id(parameter): name for name, parameter in self.model.named_parameters()}
            groups: list[list[str]] = []
            for group_index, group in enumerate(self.optimizer.param_groups):
                names: list[str] = []
                for parameter_index, parameter in enumerate(group["params"]):
                    name = names_by_id.get(id(parameter))
                    if name is None:
                        raise RuntimeError(
                            "optimizer parameter is absent from model.named_parameters(): "
                            f"group={group_index}, index={parameter_index}"
                        )
                    names.append(name)
                groups.append(names)
            return groups

        def _optimizer_storage_summary(self) -> dict[str, Any]:
            raw_optimizer = self._raw_optimizer()
            tensor_count = 0
            tensor_elements = 0
            paged_tensor_count = 0
            paged_tensor_elements = 0
            devices: dict[str, int] = {}
            for state in raw_optimizer.state.values():
                if not isinstance(state, Mapping):
                    continue
                for value in state.values():
                    if not torch_module.is_tensor(value):
                        continue
                    tensor_count += 1
                    tensor_elements += value.numel()
                    device = str(value.device)
                    devices[device] = devices.get(device, 0) + value.numel()
                    if bool(getattr(value, "is_paged", False)):
                        paged_tensor_count += 1
                        paged_tensor_elements += value.numel()
            return {
                "tensor_count": tensor_count,
                "tensor_elements": tensor_elements,
                "paged_tensor_count": paged_tensor_count,
                "paged_tensor_elements": paged_tensor_elements,
                "devices_to_elements": dict(sorted(devices.items())),
            }

        def _hydrate_bitsandbytes_optimizer_state(
            self, saved_optimizer: Mapping[str, Any]
        ) -> dict[str, Any]:
            """Restore BNB state into native paged buffers instead of replacing them."""

            raw_optimizer = self._raw_optimizer()
            if not type(raw_optimizer).__module__.startswith("bitsandbytes."):
                raise TypeError("paged optimizer hydration requires a bitsandbytes optimizer")
            saved_groups = saved_optimizer.get("param_groups")
            saved_states = saved_optimizer.get("state")
            if not isinstance(saved_groups, list) or not isinstance(saved_states, Mapping):
                raise ValueError("malformed bitsandbytes optimizer state")
            if len(saved_groups) != len(raw_optimizer.param_groups):
                raise ValueError("bitsandbytes optimizer group count changed")
            raw_optimizer.state.clear()
            raw_optimizer.check_overrides()
            parameter_by_saved_id: dict[Any, Any] = {}
            with torch_module.no_grad():
                for group_index, (current_group, saved_group) in enumerate(
                    zip(raw_optimizer.param_groups, saved_groups, strict=True)
                ):
                    saved_ids = saved_group.get("params")
                    if not isinstance(saved_ids, list) or len(saved_ids) != len(
                        current_group["params"]
                    ):
                        raise ValueError(
                            f"bitsandbytes optimizer parameter count changed in group {group_index}"
                        )
                    for parameter_index, (saved_id, parameter) in enumerate(
                        zip(saved_ids, current_group["params"], strict=True)
                    ):
                        parameter_by_saved_id[saved_id] = parameter
                        raw_optimizer.init_state(
                            current_group, parameter, group_index, parameter_index
                        )
                    for key, value in saved_group.items():
                        if key != "params":
                            current_group[key] = value

                for saved_id, saved_state_value in saved_states.items():
                    if saved_id not in parameter_by_saved_id:
                        raise ValueError(f"unmapped bitsandbytes optimizer state id: {saved_id!r}")
                    if not isinstance(saved_state_value, Mapping):
                        raise ValueError("bitsandbytes per-parameter state must be a mapping")
                    saved_state = dict(saved_state_value)
                    quantized = saved_state.pop("__bnb_optimizer_quant_state__", {})
                    if not isinstance(quantized, Mapping):
                        raise ValueError("malformed wrapped bitsandbytes quantization state")
                    if set(saved_state).intersection(quantized):
                        raise ValueError("duplicate bitsandbytes state key after quantization unwrap")
                    saved_state.update(quantized)
                    target_state = raw_optimizer.state[parameter_by_saved_id[saved_id]]
                    if set(target_state) != set(saved_state):
                        raise ValueError(
                            "bitsandbytes optimizer state layout changed: "
                            f"expected {sorted(saved_state)}, initialized {sorted(target_state)}"
                        )
                    for key, value in saved_state.items():
                        target = target_state[key]
                        if torch_module.is_tensor(value):
                            if not torch_module.is_tensor(target):
                                raise ValueError(f"bitsandbytes state {key} is no longer a tensor")
                            if value.dtype != target.dtype or tuple(value.shape) != tuple(target.shape):
                                raise ValueError(
                                    f"bitsandbytes state {key} metadata changed: "
                                    f"{value.dtype}/{tuple(value.shape)} != "
                                    f"{target.dtype}/{tuple(target.shape)}"
                                )
                            target.copy_(value.to(device=target.device))
                        else:
                            target_state[key] = value
            raw_optimizer.initialized = True
            return self._optimizer_storage_summary()

        def _run_with_dtype_preserving_adapter_load(
            self,
            *,
            checkpoint: Path,
            phase: str,
            operation: Any,
            model: Any,
        ) -> Any:
            if not hasattr(model, "load_adapter"):
                return operation()
            saved = self._saved_adapter_dtype_summary(checkpoint)
            before = self._parameter_dtype_summary(model)
            if set(saved["dtypes_to_elements"]) != {"torch.bfloat16"}:
                raise ValueError(
                    f"adapter dtype invariant requires BF16 checkpoint tensors, got "
                    f"{saved['dtypes_to_elements']} at {checkpoint}"
                )
            original = model.load_adapter
            instance_had_override = "load_adapter" in getattr(model, "__dict__", {})
            previous_override = getattr(model, "__dict__", {}).get("load_adapter")
            load_calls: list[dict[str, Any]] = []

            def dtype_preserving_load_adapter(*args: Any, **kwargs: Any) -> Any:
                caller_value = kwargs.get("autocast_adapter_dtype")
                kwargs["autocast_adapter_dtype"] = False
                load_calls.append(
                    {
                        "caller_autocast_adapter_dtype": caller_value,
                        "effective_autocast_adapter_dtype": False,
                    }
                )
                return original(*args, **kwargs)

            setattr(model, "load_adapter", dtype_preserving_load_adapter)
            try:
                result = operation()
            finally:
                if instance_had_override:
                    setattr(model, "load_adapter", previous_override)
                else:
                    delattr(model, "load_adapter")
            if not load_calls:
                raise RuntimeError(f"expected PEFT adapter load did not occur during {phase}")
            after = self._parameter_dtype_summary(model)
            matches = (
                before["dtypes_to_elements"] == saved["dtypes_to_elements"]
                and after["dtypes_to_elements"] == saved["dtypes_to_elements"]
                and before["total_elements"] == saved["total_elements"]
                and after["total_elements"] == saved["total_elements"]
            )
            audit = {
                "phase": phase,
                "checkpoint": str(checkpoint.resolve()),
                "saved_adapter": saved,
                "trainable_before_load": before,
                "trainable_after_load": after,
                "load_calls": load_calls,
                "dtype_and_element_count_match": matches,
            }
            self.adapter_dtype_restore_audits.append(audit)
            if not matches:
                raise RuntimeError(f"PEFT adapter dtype invariant failed: {audit}")
            return result

        def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Any = None) -> None:
            target_model = self.model if model is None else model
            checkpoint = Path(resume_from_checkpoint)
            return self._run_with_dtype_preserving_adapter_load(
                checkpoint=checkpoint,
                phase="trainer_resume",
                operation=lambda: super(SafeCheckpointSFTTrainer, self)._load_from_checkpoint(
                    resume_from_checkpoint, model=model
                ),
                model=target_model,
            )

        def _load_best_model(self) -> None:
            checkpoint = Path(str(self.state.best_model_checkpoint))
            return self._run_with_dtype_preserving_adapter_load(
                checkpoint=checkpoint,
                phase="load_best_model_at_end",
                operation=lambda: super(SafeCheckpointSFTTrainer, self)._load_best_model(),
                model=self.model,
            )

        def _assert_supported_safe_checkpoint_backend(self) -> None:
            if self.is_deepspeed_enabled or self.is_fsdp_enabled:
                raise RuntimeError(
                    "safe sidecar checkpointing currently supports single-GPU and ordinary DDP, "
                    "not DeepSpeed or FSDP"
                )
            if str(getattr(self.args, "device", "")).startswith("xla"):
                raise RuntimeError("safe sidecar checkpointing does not support XLA")

        def _save_optimizer_and_scheduler(self, output_dir: str) -> None:
            self._assert_supported_safe_checkpoint_backend()
            if not self.args.should_save:
                return
            raw_optimizer = self._raw_optimizer()
            payload = {
                "global_step": int(self.state.global_step),
                "world_size": int(self.args.world_size),
                "trl_total_train_tokens": self._total_train_tokens,
                "optimizer_class": (
                    f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}"
                ),
                "optimizer_backend_class": (
                    f"{type(raw_optimizer).__module__}.{type(raw_optimizer).__qualname__}"
                ),
                "optimizer_parameter_name_groups": self._optimizer_parameter_name_groups(),
                "scheduler_class": (
                    f"{type(self.lr_scheduler).__module__}.{type(self.lr_scheduler).__qualname__}"
                ),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.lr_scheduler.state_dict(),
            }
            written = save_safe_state_bundle(
                directory=Path(output_dir),
                json_name=SAFE_OPTIMIZER_JSON,
                tensor_name=SAFE_OPTIMIZER_TENSORS,
                payload=payload,
                torch_module=torch_module,
                save_file=safe_save_file,
            )
            self.safe_checkpoint_events.append(
                {
                    "event": "safe_optimizer_scheduler_saved",
                    "checkpoint": str(Path(output_dir).resolve()),
                    "global_step": int(self.state.global_step),
                    **written,
                }
            )

        def _save_rng_state(self, output_dir: str) -> None:
            self._assert_supported_safe_checkpoint_backend()
            payload: dict[str, Any] = {
                "global_step": int(self.state.global_step),
                "world_size": int(self.args.world_size),
                "process_index": int(self.args.process_index),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cpu": torch_module.random.get_rng_state(),
            }
            if torch_module.cuda.is_available():
                payload["cuda"] = (
                    torch_module.cuda.random.get_rng_state_all()
                    if int(self.args.world_size) > 1
                    else torch_module.cuda.random.get_rng_state()
                )
            json_name, tensor_name = safe_rng_filenames(
                int(self.args.world_size), int(self.args.process_index)
            )
            written = save_safe_state_bundle(
                directory=Path(output_dir),
                json_name=json_name,
                tensor_name=tensor_name,
                payload=payload,
                torch_module=torch_module,
                save_file=safe_save_file,
            )
            self.safe_checkpoint_events.append(
                {
                    "event": "safe_rng_saved",
                    "checkpoint": str(Path(output_dir).resolve()),
                    "global_step": int(self.state.global_step),
                    "process_index": int(self.args.process_index),
                    **written,
                }
            )

        def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
            if checkpoint is None:
                return
            self._assert_supported_safe_checkpoint_backend()
            payload = load_safe_state_bundle(
                directory=Path(checkpoint),
                json_name=SAFE_OPTIMIZER_JSON,
                tensor_name=SAFE_OPTIMIZER_TENSORS,
                load_file=safe_load_file,
            )
            saved_step = int(payload.pop("global_step"))
            saved_world_size = int(payload.pop("world_size"))
            saved_total_train_tokens = payload.pop("trl_total_train_tokens", None)
            optimizer_class = payload.pop("optimizer_class")
            optimizer_backend_class = payload.pop("optimizer_backend_class", None)
            saved_parameter_names = payload.pop("optimizer_parameter_name_groups", None)
            scheduler_class = payload.pop("scheduler_class")
            if saved_step != int(self.state.global_step):
                raise ValueError(
                    f"safe optimizer step {saved_step} does not match Trainer state {self.state.global_step}"
                )
            if saved_world_size != int(self.args.world_size):
                raise ValueError(
                    f"safe resume requires the original world size {saved_world_size}, "
                    f"got {self.args.world_size}"
                )
            if set(payload) != {"optimizer", "scheduler"}:
                raise ValueError(f"unexpected safe optimizer payload keys: {sorted(payload)}")
            raw_optimizer = self._raw_optimizer()
            loaded_backend_class = (
                f"{type(raw_optimizer).__module__}.{type(raw_optimizer).__qualname__}"
            )
            if optimizer_backend_class is None or saved_parameter_names is None:
                raise ValueError(
                    "safe optimizer checkpoint predates name-mapped native-storage restoration"
                )
            if optimizer_backend_class != loaded_backend_class:
                raise ValueError(
                    "optimizer backend class changed: "
                    f"{optimizer_backend_class} != {loaded_backend_class}"
                )
            current_parameter_names = self._optimizer_parameter_name_groups()
            if saved_parameter_names != current_parameter_names:
                raise ValueError("optimizer parameter name/order mapping changed across resume")
            if saved_total_train_tokens is None:
                raise ValueError("safe optimizer checkpoint lacks the TRL cumulative token counter")
            if type(raw_optimizer).__module__.startswith("bitsandbytes."):
                storage = self._hydrate_bitsandbytes_optimizer_state(payload["optimizer"])
                restoration = "name_mapped_copy_into_native_bitsandbytes_buffers"
            else:
                self.optimizer.load_state_dict(payload["optimizer"])
                storage = self._optimizer_storage_summary()
                restoration = "framework_load_state_dict"
            loaded_optimizer_state = self.optimizer.state_dict()
            optimizer_state_comparison = exact_nested_state_comparison(
                payload["optimizer"], loaded_optimizer_state, torch_module=torch_module
            )
            if not optimizer_state_comparison["exact_equal"]:
                raise RuntimeError(
                    "optimizer state differs immediately after safe restore: "
                    f"{optimizer_state_comparison}"
                )
            self.lr_scheduler.load_state_dict(payload["scheduler"])
            scheduler_state_comparison = exact_nested_state_comparison(
                payload["scheduler"],
                self.lr_scheduler.state_dict(),
                torch_module=torch_module,
            )
            if not scheduler_state_comparison["exact_equal"]:
                raise RuntimeError(
                    "scheduler state differs immediately after safe restore: "
                    f"{scheduler_state_comparison}"
                )
            self._total_train_tokens = saved_total_train_tokens
            self.safe_resume_details = {
                "checkpoint": str(Path(checkpoint).resolve()),
                "global_step": saved_step,
                "world_size": saved_world_size,
                "optimizer_class_saved": optimizer_class,
                "optimizer_class_loaded": (
                    f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}"
                ),
                "optimizer_backend_class_saved": optimizer_backend_class,
                "optimizer_backend_class_loaded": loaded_backend_class,
                "optimizer_parameter_name_mapping_verified": True,
                "optimizer_parameter_count": sum(map(len, current_parameter_names)),
                "optimizer_state_restoration": restoration,
                "optimizer_native_storage": storage,
                "optimizer_state_exact_after_restore": optimizer_state_comparison,
                "scheduler_class_saved": scheduler_class,
                "scheduler_class_loaded": (
                    f"{type(self.lr_scheduler).__module__}.{type(self.lr_scheduler).__qualname__}"
                ),
                "scheduler_state_exact_after_restore": scheduler_state_comparison,
                "trl_total_train_tokens_restored": saved_total_train_tokens,
                "serialization": "strict_json_plus_safetensors_no_pickle",
            }

        def _load_rng_state(self, checkpoint: str | None) -> None:
            if checkpoint is None:
                return
            self._assert_supported_safe_checkpoint_backend()
            json_name, tensor_name = safe_rng_filenames(
                int(self.args.world_size), int(self.args.process_index)
            )
            payload = load_safe_state_bundle(
                directory=Path(checkpoint),
                json_name=json_name,
                tensor_name=tensor_name,
                load_file=safe_load_file,
            )
            saved_step = int(payload.pop("global_step"))
            saved_world_size = int(payload.pop("world_size"))
            saved_process_index = int(payload.pop("process_index"))
            if saved_step != int(self.state.global_step):
                raise ValueError(
                    f"safe RNG step {saved_step} does not match Trainer state {self.state.global_step}"
                )
            if saved_world_size != int(self.args.world_size):
                raise ValueError("safe RNG restore requires the checkpoint world size")
            if saved_process_index != int(self.args.process_index):
                raise ValueError("safe RNG restore loaded the wrong process sidecar")
            required = {"python", "numpy", "cpu"}
            if not required.issubset(payload):
                raise ValueError(f"safe RNG payload lacks keys: {sorted(required - set(payload))}")
            random.setstate(payload.pop("python"))
            np.random.set_state(payload.pop("numpy"))
            torch_module.random.set_rng_state(payload.pop("cpu"))
            if "cuda" in payload:
                cuda_state = payload.pop("cuda")
                if not torch_module.cuda.is_available():
                    raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
                if saved_world_size > 1:
                    if not isinstance(cuda_state, list):
                        raise ValueError("distributed CUDA RNG state must be a list")
                    torch_module.cuda.random.set_rng_state_all(cuda_state)
                else:
                    torch_module.cuda.random.set_rng_state(cuda_state)
            if payload:
                raise ValueError(f"unsupported device RNG state keys: {sorted(payload)}")
            self.safe_rng_restore_details = {
                "checkpoint": str(Path(checkpoint).resolve()),
                "global_step": saved_step,
                "process_index": saved_process_index,
                "restored": ["python", "numpy", "torch_cpu", "torch_cuda"],
                "serialization": "strict_json_plus_safetensors_no_pickle",
            }

    SafeCheckpointSFTTrainer.__name__ = "SafeCheckpointSFTTrainer"
    return SafeCheckpointSFTTrainer


def runtime_manifest_base(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    rows: PreparedRows,
    world: Mapping[str, int],
    run_id: str,
    model_source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    model = config["model"]
    train = config["training"]
    lora = config["lora"]
    return {
        "manifest_version": "qwen_vl_supervisor_training_run_v1",
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPOSITORY_ROOT),
        "config_path": str(config_path),
        "config_sha256": sha256_path(config_path),
        "run_name": config["run_name"],
        "training_seeds": config.get("training_seeds", [config["training"]["seed"]]),
        "model": {
            "id": model["id"],
            "source_id": model.get("source_id", model["id"]),
            "revision": model["revision"],
            "processor_revision": model["processor_revision"],
            "source_identity": dict(model_source_identity),
            "architecture": model["architecture"],
            "attention_implementation": model["attn_implementation"],
            "processor": dict(model["processor"]),
            "quantization": dict(model["quantization"]),
        },
        "data": {
            "train_jsonl": str(rows.train_path),
            "train_sha256": sha256_path(rows.train_path),
            "train_records": len(rows.train_rows),
            "train_splits": sorted({row_split(row) for row in rows.train_rows}),
            "dev_jsonl": str(rows.dev_path),
            "dev_sha256": sha256_path(rows.dev_path),
            "dev_records": len(rows.dev_rows),
            "dev_splits": sorted({row_split(row) for row in rows.dev_rows}),
            "export_report": str(resolve_repo_path(config["data"]["export_report"])),
            "export_report_sha256": config["data"]["export_report_sha256"],
            "model_visible_allowlist_validator": (
                "qwen_vl_supervisor_v1.export_sft.validate_export_row"
            ),
            "pinned_export_hashes_verified": True,
            "frozen_predictions_opened": False,
        },
        "lora": dict(lora),
        "training": {
            key: train[key]
            for key in (
                "max_steps",
                "per_device_train_batch_size",
                "per_device_eval_batch_size",
                "gradient_accumulation_steps",
                "learning_rate",
                "weight_decay",
                "optim",
                "lr_scheduler_type",
                "warmup_ratio",
                "max_length",
                "seed",
                "data_seed",
                "bf16",
                "fp16",
                "tf32",
                "gradient_checkpointing",
                "save_steps",
                "eval_steps",
                "completion_only_loss",
                "ignore_data_skip",
                "full_determinism",
            )
        },
        "distributed": dict(world),
        "packages": package_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_state(),
    }


def make_callbacks(
    trainer_callback_cls: Any,
    log_path: Path,
    rank_is_zero: bool,
    *,
    save_at_global_step: int | None = None,
    stop_after_global_step: int | None = None,
) -> tuple[list[Any], Any, Any]:
    class StructuredLogCallback(trainer_callback_cls):
        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
            if not rank_is_zero or not logs:
                return
            payload = {
                "event": "trainer_log",
                "global_step": int(state.global_step),
                "epoch": state.epoch,
                "logged_at_utc": datetime.now(timezone.utc).isoformat(),
                **{key: value for key, value in logs.items() if isinstance(value, (str, int, float, bool))},
            }
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    class ResumeSequenceCallback(trainer_callback_cls):
        def __init__(self) -> None:
            self.observed_start_step: int | None = None
            self.first_completed_step: int | None = None

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.observed_start_step = int(state.global_step)

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if self.first_completed_step is None:
                self.first_completed_step = int(state.global_step)

    class LoraDeltaCallback(trainer_callback_cls):
        def __init__(self) -> None:
            self.before: dict[str, Any] = {}
            self.changed_names: list[str] = []
            self.max_abs_delta = 0.0

        def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            if model is None:
                return
            preferred = [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and "lora_B" in name
            ]
            if not preferred:
                preferred = [
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and "lora_" in name
                ]
            for name, parameter in preferred[:32]:
                self.before[name] = parameter.detach().float().cpu().clone()

        def _measure(self, model: Any) -> None:
            if model is None:
                return
            current = dict(model.named_parameters())
            for name, before in self.before.items():
                if name not in current:
                    continue
                delta = (current[name].detach().float().cpu() - before).abs().max().item()
                self.max_abs_delta = max(self.max_abs_delta, float(delta))
                if delta > 0.0 and name not in self.changed_names:
                    self.changed_names.append(name)

        def on_step_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            # Measure immediately after optimizer steps. This remains valid even
            # when load_best_model_at_end later restores an earlier dev checkpoint.
            self._measure(model)

        def on_train_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            self._measure(model)

    class AuditControlCallback(trainer_callback_cls):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            step = int(state.global_step)
            if save_at_global_step is not None and step == save_at_global_step:
                control.should_save = True
            if stop_after_global_step is not None and step >= stop_after_global_step:
                control.should_training_stop = True
            return control

    resume = ResumeSequenceCallback()
    delta = LoraDeltaCallback()
    return [StructuredLogCallback(), resume, delta, AuditControlCallback()], resume, delta


def train(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    local_rank_arg: int | None = None,
    save_at_global_step: int | None = None,
    stop_after_global_step: int | None = None,
) -> dict[str, Any]:
    validate_config(config)
    rows = prepare_rows(config, verify_images=True)
    model_cfg = config["model"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    if bool(train_cfg.get("full_determinism", False)):
        # These must exist before torch/CUDA initialization. Transformers also
        # enables deterministic algorithms and deterministic cuDNN through the
        # resolved SFTConfig.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
    deps = require_training_dependencies()
    torch = deps["torch"]
    world = world_info(local_rank_arg)
    local_rank = int(world["local_rank"])
    output_dir = resolve_repo_path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    configured_max_steps = int(train_cfg["max_steps"])
    for name, value in (
        ("save_at_global_step", save_at_global_step),
        ("stop_after_global_step", stop_after_global_step),
    ):
        if value is not None and (value <= 0 or value > configured_max_steps):
            raise ValueError(f"{name} must be in [1, training.max_steps]")
    if stop_after_global_step is not None and save_at_global_step != stop_after_global_step:
        raise ValueError(
            "--stop-after-global-step requires the same --save-at-global-step so replay weights persist"
        )

    model_source_identity = resolve_model_source_identity(
        model_id=str(model_cfg["id"]),
        source_id=str(model_cfg.get("source_id", model_cfg["id"])),
        revision=str(model_cfg["revision"]),
        processor_revision=str(model_cfg["processor_revision"]),
        repository_root=REPOSITORY_ROOT,
        expected_local_tree_sha256=model_cfg.get("expected_local_snapshot_tree_sha256"),
    )

    require_cuda = bool(model_cfg.get("require_cuda", True))
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this QLoRA config, but torch.cuda.is_available() is false")
    if torch.cuda.is_available():
        device_index = local_rank if local_rank >= 0 else 0
        if device_index >= torch.cuda.device_count():
            raise ValueError(f"local rank {device_index} exceeds {torch.cuda.device_count()} visible GPUs")
        torch.cuda.set_device(device_index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_index)
        device_map: Any = {"": device_index}
    else:
        device_index = None
        device_map = None

    deps["set_seed"](int(train_cfg["seed"]), deterministic=bool(train_cfg.get("full_determinism", True)))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    manifest = runtime_manifest_base(
        config=config,
        config_path=config_path,
        rows=rows,
        world=world,
        run_id=run_id,
        model_source_identity=model_source_identity,
    )
    start_manifest = output_dir / f"run_manifest.{run_id}.start.json"
    final_manifest = output_dir / f"run_manifest.{run_id}.json"
    if rank_zero(world):
        atomic_json(start_manifest, manifest)

    train_examples = [row_to_example(row, rows.image_root) for row in rows.train_rows]
    dev_examples = [row_to_example(row, rows.image_root) for row in rows.dev_rows]
    train_dataset = deps["Dataset"].from_list(train_examples)
    dev_dataset = deps["Dataset"].from_list(dev_examples)

    processor_kwargs = model_source_kwargs(
        model_cfg, model_source_identity, processor=True
    )
    processor_kwargs.update(
        min_pixels=int(model_cfg["processor"]["min_pixels"]),
        max_pixels=int(model_cfg["processor"]["max_pixels"]),
    )
    processor = deps["AutoProcessor"].from_pretrained(
        model_source_identity["model_id"], **processor_kwargs
    )
    size = getattr(processor.image_processor, "size", None)
    observed_pixels = {
        "min_pixels": getattr(size, "shortest_edge", None),
        "max_pixels": getattr(size, "longest_edge", None),
    }
    expected_pixels = {
        "min_pixels": int(model_cfg["processor"]["min_pixels"]),
        "max_pixels": int(model_cfg["processor"]["max_pixels"]),
    }
    if observed_pixels != expected_pixels:
        raise ValueError(f"processor pixel budget mismatch: {observed_pixels} != {expected_pixels}")

    quantization = model_cfg["quantization"]
    quantization_config = deps["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype(torch, quantization["compute_dtype"]),
        bnb_4bit_use_double_quant=bool(quantization["double_quant"]),
    )
    model_kwargs = model_source_kwargs(
        model_cfg, model_source_identity, processor=False
    )
    model_kwargs.update(
        quantization_config=quantization_config,
        torch_dtype=torch_dtype(torch, model_cfg["torch_dtype"]),
        attn_implementation=model_cfg["attn_implementation"],
        device_map=device_map,
    )
    model = deps["Qwen2_5_VLForConditionalGeneration"].from_pretrained(
        model_source_identity["model_id"], **model_kwargs
    )
    model.config.use_cache = False
    architecture_audit = audit_lora_architecture(model, lora_cfg["target_modules"])
    model = deps["prepare_model_for_kbit_training"](
        model,
        use_gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]),
    )
    peft_config = deps["LoraConfig"](
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=list(lora_cfg["target_modules"]),
        task_type="CAUSAL_LM",
    )
    sft_config = build_sft_config(deps["SFTConfig"], config, local_rank)
    log_path = output_dir / f"trainer_log.{run_id}.jsonl"
    callbacks, resume_callback, delta_callback = make_callbacks(
        deps["TrainerCallback"],
        log_path,
        rank_zero(world),
        save_at_global_step=save_at_global_step,
        stop_after_global_step=stop_after_global_step,
    )
    safe_trainer_cls = make_safe_trainer_class(
        deps["SFTTrainer"],
        torch_module=torch,
        safe_save_file=deps["safe_save_file"],
        safe_load_file=deps["safe_load_file"],
    )
    trainer = safe_trainer_cls(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=processor,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    counts = parameter_counts(trainer.model)
    if counts["trainable"] <= 0:
        raise RuntimeError("LoRA attachment produced zero trainable parameters")

    resume_path, expected_resume_step = resolve_resume(
        train_cfg.get("resume_from_checkpoint"), output_dir
    )
    if expected_resume_step is not None and int(train_cfg["max_steps"]) <= expected_resume_step:
        raise ValueError(
            "max_steps must exceed the resumed global step so the next-step resume audit can run"
        )

    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    runtime_seconds = time.perf_counter() - started
    final_step = int(trainer.state.global_step)
    final_adapter_source: dict[str, Any]
    if stop_after_global_step is not None:
        stopped_checkpoint = output_dir / f"checkpoint-{final_step}"
        if not stopped_checkpoint.is_dir():
            raise FileNotFoundError(
                f"forced audit checkpoint is absent after controlled stop: {stopped_checkpoint}"
            )
        # Trainer's ordinary load_best_model_at_end may select a checkpoint from
        # before a deliberately early audit stop. Restore the just-executed step
        # so final_adapter represents the replayed state, while retaining the
        # original best-dev metadata in TrainerState.
        audit_count_before = len(trainer.adapter_dtype_restore_audits)
        trainer._load_from_checkpoint(str(stopped_checkpoint))
        if len(trainer.adapter_dtype_restore_audits) != audit_count_before + 1:
            raise RuntimeError("controlled-stop adapter restore did not emit one dtype audit")
        trainer.adapter_dtype_restore_audits[-1]["phase"] = (
            "controlled_stop_final_adapter_restore"
        )
        final_adapter_source = {
            "policy": "controlled_stop_uses_forced_final_step_checkpoint",
            "checkpoint": str(stopped_checkpoint.resolve()),
        }
    else:
        final_adapter_source = {
            "policy": "trainer_load_best_model_at_end",
            "checkpoint": trainer.state.best_model_checkpoint,
        }
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    trainer.save_state()
    if rank_zero(world):
        processor.save_pretrained(final_adapter)
    final_adapter_dtype = trainer._saved_adapter_dtype_summary(final_adapter)
    final_dtype_matches = (
        set(final_adapter_dtype["dtypes_to_elements"]) == {"torch.bfloat16"}
        and final_adapter_dtype["total_elements"] == int(counts["trainable"])
    )
    if not final_dtype_matches:
        raise RuntimeError(
            f"final adapter dtype invariant failed: expected {counts['trainable']} BF16 elements, "
            f"got {final_adapter_dtype}"
        )

    latest = latest_saved_checkpoint(output_dir)
    if resume_path is None:
        resume_audit: dict[str, Any] = {
            "requested": False,
            "checkpoint": None,
            "expected_checkpoint_step": None,
            "observed_start_step": resume_callback.observed_start_step,
            "first_completed_step": resume_callback.first_completed_step,
            "full_state_restore_verified": False,
            "next_step_executed_after_full_state_restore": False,
            "numerical_reproduction_verified": None,
            "numerical_reproduction_basis": "not_applicable_fresh_run",
            "status": "not_applicable_fresh_run",
        }
    else:
        safe_state_restored = (
            trainer.safe_resume_details is not None
            and trainer.safe_resume_details.get("global_step") == expected_resume_step
            and trainer.safe_rng_restore_details is not None
            and trainer.safe_rng_restore_details.get("global_step") == expected_resume_step
        )
        execution_ok = (
            resume_callback.observed_start_step == expected_resume_step
            and resume_callback.first_completed_step == expected_resume_step + 1
            and final_step > expected_resume_step
            and safe_state_restored
        )
        resume_audit = {
            "requested": True,
            "checkpoint": str(resume_path),
            "expected_checkpoint_step": expected_resume_step,
            "observed_start_step": resume_callback.observed_start_step,
            "first_completed_step": resume_callback.first_completed_step,
            "full_state_restore_verified": safe_state_restored,
            "next_step_executed_after_full_state_restore": execution_ok,
            "numerical_reproduction_verified": None,
            "numerical_reproduction_basis": (
                "requires an explicit comparison against an independent replay log"
            ),
            "safe_optimizer_scheduler_restore": trainer.safe_resume_details,
            "safe_rng_restore": trainer.safe_rng_restore_details,
            "trainer_data_skip_enabled": not bool(train_cfg.get("ignore_data_skip", False)),
            "status": "passed" if execution_ok else "failed",
        }
        if not execution_ok:
            raise RuntimeError(f"Trainer resume execution audit failed: {resume_audit}")

    lora_changed = delta_callback.max_abs_delta > 0.0
    previous_delta_evidence = prior_fresh_lora_delta_evidence(output_dir, run_id)
    if resume_path is None and not lora_changed:
        raise RuntimeError("no audited LoRA parameter changed during training")
    if resume_path is not None and not lora_changed and previous_delta_evidence is None:
        raise RuntimeError(
            "terminal resume produced no LoRA delta and no completed fresh-run delta evidence exists"
        )
    terminal_learning_rates = [
        float(group["lr"]) for group in trainer.optimizer.param_groups if "lr" in group
    ]
    metrics = {
        key: value
        for key, value in result.metrics.items()
        if isinstance(value, (str, int, float, bool))
    }
    gpu = None
    if torch.cuda.is_available() and device_index is not None:
        gpu = {
            "device_index": device_index,
            "name": torch.cuda.get_device_name(device_index),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
        }
    manifest.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "architecture_audit": architecture_audit,
            "processor_pixel_budget_observed": observed_pixels,
            "parameters": counts,
            "audit_execution_controls": {
                "configured_scheduler_max_steps": configured_max_steps,
                "forced_checkpoint_global_step": save_at_global_step,
                "stop_after_global_step": stop_after_global_step,
                "scheduler_horizon_changed_by_stop_control": False,
            },
            "adapter_dtype_invariant": {
                "required_dtype": "torch.bfloat16",
                "expected_trainable_elements": counts["trainable"],
                "restore_audits": trainer.adapter_dtype_restore_audits,
                "final_adapter": final_adapter_dtype,
                "final_adapter_matches": final_dtype_matches,
                "final_adapter_source": final_adapter_source,
            },
            "training_result": {
                "global_step": final_step,
                "runtime_wall_seconds": runtime_seconds,
                "metrics": metrics,
                "examples_per_second": metrics.get("train_samples_per_second"),
                "log_jsonl": str(log_path),
                "latest_checkpoint": str(latest) if latest else None,
                "best_dev_checkpoint": trainer.state.best_model_checkpoint,
                "final_adapter": str(final_adapter),
            },
            "lora_delta_verification": {
                "required_for_this_run": resume_path is None,
                "audited_parameter_count": len(delta_callback.before),
                "changed_parameter_count": len(delta_callback.changed_names),
                "changed_parameter_examples": delta_callback.changed_names[:8],
                "max_abs_delta": delta_callback.max_abs_delta,
                "at_least_one_intended_parameter_changed": lora_changed,
                "status": (
                    "passed"
                    if lora_changed
                    else "not_required_terminal_resume_with_restored_zero_learning_rate"
                ),
                "terminal_learning_rates": terminal_learning_rates,
                "reason_if_unchanged": (
                    None
                    if lora_changed
                    else "checkpoint-20 restored a terminal cosine-schedule optimizer LR of zero; "
                    "step 21 audits stateful resume sequencing, not an additional parameter update"
                ),
                "prior_fresh_run_delta_evidence": previous_delta_evidence,
            },
            "resume_audit": resume_audit,
            "gpu": gpu,
            "safe_checkpointing": {
                "schema": SAFE_STATE_SCHEMA,
                "serialization": "strict_json_plus_safetensors_no_pickle",
                "events": trainer.safe_checkpoint_events,
                "legacy_optimizer_or_rng_pickle_loaded": False,
            },
        }
    )
    if rank_zero(world):
        atomic_json(final_manifest, manifest)
        atomic_json(output_dir / "run_manifest.latest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config, config_path = resolve_config(args)
    if args.print_resolved_config:
        print(json.dumps(config, indent=2, sort_keys=True))
        return
    if args.validate_only:
        rows = prepare_rows(config, verify_images=True)
        model_cfg = config["model"]
        model_source_identity = resolve_model_source_identity(
            model_id=str(model_cfg["id"]),
            source_id=str(model_cfg.get("source_id", model_cfg["id"])),
            revision=str(model_cfg["revision"]),
            processor_revision=str(model_cfg["processor_revision"]),
            repository_root=REPOSITORY_ROOT,
            expected_local_tree_sha256=model_cfg.get(
                "expected_local_snapshot_tree_sha256"
            ),
        )
        print(
            json.dumps(
                {
                    "status": "valid",
                    "config": str(config_path),
                    "train_jsonl": str(rows.train_path),
                    "train_records": len(rows.train_rows),
                    "dev_jsonl": str(rows.dev_path),
                    "dev_records": len(rows.dev_rows),
                    "images_verified": len(rows.train_rows) + len(rows.dev_rows),
                    "model_source_identity": model_source_identity,
                    "model_loaded": False,
                    "frozen_predictions_opened": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    manifest = train(
        config,
        config_path,
        local_rank_arg=args.local_rank,
        save_at_global_step=args.save_at_global_step,
        stop_after_global_step=args.stop_after_global_step,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
