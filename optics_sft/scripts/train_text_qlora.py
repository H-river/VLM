#!/usr/bin/env python3
"""Train a text-only QLoRA adapter on optics text SFT rows."""

from __future__ import annotations

import os
import sys

# TRL loads UTF-8 jinja chat templates at import time; Windows GBK default breaks this.
if sys.platform == "win32" and not sys.flags.utf8_mode:
    print("[train] Re-launching Python with UTF-8 mode (-X utf8)...", flush=True)
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen2.5 text model with 4-bit QLoRA on text SFT data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("optics_sft/configs/qwen25_3b_text_qlora_inverse.yaml"),
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-samples", type=int, default=2)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install PyYAML to read configs.") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            rows.append(row)
    return rows


from optics_sft.physics.text_target import apply_training_messages, normalize_training_target
from optics_sft.utils.paths import resolve_config_path


def resolve_model_reference(model_name: str, *, local_files_only: bool) -> str:
    candidate = Path(model_name).expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    if candidate.is_dir() and (candidate / "config.json").exists():
        return str(candidate)
    if local_files_only:
        raise FileNotFoundError(
            "Local model directory not found: "
            f"{candidate}\n"
            f"Download with: huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir {candidate}"
        )
    return model_name


def row_to_sft_example(row: dict[str, Any], *, training_target: str) -> dict[str, Any]:
    if training_target == "full":
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Text row {row.get('sample_id')} is missing messages")
        return {"messages": messages}
    return {"messages": apply_training_messages(row, training_target)}  # type: ignore[arg-type]


def require_training_imports() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependencies. Install torch, datasets, transformers, peft, bitsandbytes, trl, and PyYAML."
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def dtype_from_name(torch_module: Any, name: str) -> Any:
    if not hasattr(torch_module, name):
        raise ValueError(f"Unsupported torch dtype in config: {name}")
    return getattr(torch_module, name)


def build_sft_config(
    sft_config_cls: Any,
    train_cfg: dict[str, Any],
    output_dir: str,
    smoke_test: bool,
    has_eval_dataset: bool,
) -> Any:
    configured_max_steps = int(train_cfg.get("max_steps", -1))
    max_steps = min(configured_max_steps, 1) if smoke_test and configured_max_steps > 0 else configured_max_steps
    if smoke_test and configured_max_steps <= 0:
        max_steps = 1
    eval_strategy = (
        "no" if smoke_test else "steps" if has_eval_dataset and train_cfg.get("eval_steps") else "no"
    )
    kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(train_cfg["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": 1 if smoke_test else int(train_cfg["gradient_accumulation_steps"]),
        "learning_rate": float(train_cfg["learning_rate"]),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.0)),
        "num_train_epochs": float(train_cfg["num_train_epochs"]),
        "max_steps": max_steps,
        "max_length": train_cfg.get("max_length", 4096),
        "bf16": bool(train_cfg.get("bf16", False)),
        "fp16": bool(train_cfg.get("fp16", False)),
        "eval_steps": int(train_cfg["eval_steps"]),
        "save_strategy": "no" if smoke_test else "steps",
        "save_steps": int(train_cfg["save_steps"]),
        "logging_steps": 1 if smoke_test else int(train_cfg["logging_steps"]),
        "save_total_limit": int(train_cfg.get("save_total_limit", 3)),
        "report_to": train_cfg.get("report_to", []),
        "remove_unused_columns": False,
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", False)),
        "optim": train_cfg.get("optim", "adamw_torch"),
        "packing": False,
        "completion_only_loss": bool(train_cfg.get("completion_only_loss", True)),
        "assistant_only_loss": bool(train_cfg.get("assistant_only_loss", False)),
    }
    if smoke_test or not has_eval_dataset:
        kwargs.pop("eval_steps", None)

    params = inspect.signature(sft_config_cls).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_strategy
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = eval_strategy
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    supported_kwargs = kwargs if accepts_var_kwargs else {key: value for key, value in kwargs.items() if key in params}
    return sft_config_cls(**supported_kwargs)


def build_trainer(trainer_cls: Any, tokenizer: Any, **kwargs: Any) -> Any:
    params = inspect.signature(trainer_cls).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    supported_kwargs = kwargs if accepts_var_kwargs else {key: value for key, value in kwargs.items() if key in params}
    return trainer_cls(**supported_kwargs)


def train(config: dict[str, Any], smoke_test: bool = False, smoke_max_samples: int = 2) -> None:
    deps = require_training_imports()
    model_cfg = config["model"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    output_dir_path = resolve_config_path(str(train_cfg.get("output_dir", output_cfg["output_dir"])))
    if smoke_test:
        output_dir_path = output_dir_path.parent / f"{output_dir_path.name}_smoke"
    output_dir = str(output_dir_path)
    print(f"Training will save adapter to: {output_dir}", flush=True)

    train_rows = read_jsonl(resolve_config_path(data_cfg["train_jsonl"]))
    val_path = resolve_config_path(data_cfg["val_jsonl"])
    val_rows = read_jsonl(val_path) if val_path.exists() else []
    training_target = normalize_training_target(
        data_cfg.get("training_target"),
        compact_target=bool(data_cfg.get("compact_target", False)),
    )
    model_cfg["name"] = resolve_model_reference(
        str(model_cfg["name"]),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    if smoke_test:
        train_rows = train_rows[: min(smoke_max_samples, 2)]
        val_rows = val_rows[:1]
        print(
            f"[smoke] loaded {len(train_rows)} train rows and {len(val_rows)} val rows "
            f"(training_target={training_target})"
        )
    else:
        print(
            f"Loaded {len(train_rows)} train rows and {len(val_rows)} val rows "
            f"(training_target={training_target})"
        )

    train_dataset = deps["Dataset"].from_list(
        [row_to_sft_example(row, training_target=training_target) for row in train_rows]
    )
    eval_dataset = (
        deps["Dataset"].from_list(
            [row_to_sft_example(row, training_target=training_target) for row in val_rows]
        )
        if val_rows
        else None
    )

    local_files_only = bool(model_cfg.get("local_files_only", False))
    tokenizer = deps["AutoTokenizer"].from_pretrained(
        model_cfg["name"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = deps["BitsAndBytesConfig"](
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype_from_name(
            deps["torch"], str(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        ),
        bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
    )
    model = deps["AutoModelForCausalLM"].from_pretrained(
        model_cfg["name"],
        quantization_config=quantization_config,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=local_files_only,
        device_map="auto",
    )
    model = deps["prepare_model_for_kbit_training"](model)
    peft_config = deps["LoraConfig"](
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        task_type="CAUSAL_LM",
    )
    sft_config = build_sft_config(
        deps["SFTConfig"],
        train_cfg,
        output_dir,
        smoke_test,
        eval_dataset is not None,
    )
    trainer = build_trainer(
        deps["SFTTrainer"],
        tokenizer,
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Saved text QLoRA adapter to {output_dir}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    train(config, smoke_test=args.smoke_test, smoke_max_samples=args.smoke_max_samples)


if __name__ == "__main__":
    main()
