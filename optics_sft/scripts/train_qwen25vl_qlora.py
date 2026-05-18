#!/usr/bin/env python3
"""Train a Qwen2.5-VL adapter for optics SFT using 4-bit QLoRA.

The script keeps the first executable path intentionally small:

- load YAML config
- load JSONL rows
- load current/target images with PIL
- build examples containing `images`, `prompt`, `completion`, and TRL-style
  multimodal `messages`
- load Qwen2.5-VL processor/model with 4-bit quantization
- attach LoRA through TRL's `SFTTrainer`
- save the trained adapter

Use `--smoke-test` to cap the run to 1-2 samples and one optimizer step.
Model files are not downloaded unless this script is actually executed.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Qwen2.5-VL-3B with 4-bit QLoRA on optics SFT data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("optics_sft/configs/qwen25vl_3b_qlora.yaml"),
        help="Path to the QLoRA YAML config.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use at most two train samples and run a one-step training smoke test.",
    )
    parser.add_argument(
        "--smoke-max-samples",
        type=int,
        default=2,
        help="Maximum number of training samples to load when --smoke-test is set.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install PyYAML to read configs.") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            rows.append(row)
    return rows


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    return rows[: max(0, limit)]


def resolve_image_path(image_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else image_root / path


def load_rgb_image(path: Path) -> Any:
    from PIL import Image

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    image = Image.open(path).convert("RGB")
    image.load()
    return image


def build_prompt(metadata: dict[str, Any]) -> str:
    return (
        "You are controlling an optical beam setup. Compare the current beam "
        "image with the target beam image. Use the metadata below and return "
        "only strict JSON containing these top-level keys: task, diagnosis, "
        "control_plan, confidence. The control_plan must contain numeric "
        "lens_x_delta_mm, lens_y_delta_mm, camera_x_delta_mm, and "
        "camera_y_delta_mm values.\n\n"
        f"metadata: {json.dumps(metadata, sort_keys=True)}"
    )


def build_messages(prompt: str, completion: str) -> list[dict[str, Any]]:
    return build_messages_with_images(prompt, completion, None)


def build_messages_with_images(
    prompt: str, completion: str | None, images: list[Any] | None
) -> list[dict[str, Any]]:
    image_items: list[dict[str, Any]]
    if images is None:
        image_items = [{"type": "image"}, {"type": "image"}]
    else:
        image_items = [{"type": "image", "image": image} for image in images]

    messages = [
        {
            "role": "user",
            "content": [
                *image_items,
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if completion is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": completion}],
            }
        )
    return messages


def build_placeholder_messages(prompt: str, completion: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": completion}],
        },
    ]


def row_to_sft_example(row: dict[str, Any], image_root: Path) -> dict[str, Any]:
    current_path = resolve_image_path(image_root, row["current_image_path"])
    target_path = resolve_image_path(image_root, row["target_image_path"])
    current_image = load_rgb_image(current_path)
    target_image = load_rgb_image(target_path)
    prompt = build_prompt(row.get("metadata", {}))
    completion = json.dumps(row["label"], sort_keys=True)

    return {
        "images": [current_image, target_image],
        "prompt": prompt,
        "completion": completion,
        "messages": build_placeholder_messages(prompt, completion),
    }


def require_training_imports() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependencies. Install torch, datasets, Pillow, "
            "transformers, peft, bitsandbytes, trl, and PyYAML before training."
        ) from exc

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLForConditionalGeneration,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def examples_to_dataset(examples: Iterable[dict[str, Any]], dataset_cls: Any) -> Any:
    examples = list(examples)
    if not examples:
        raise ValueError("No SFT examples were loaded.")
    return dataset_cls.from_list(examples)


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

    # max_length=None is important for VLM SFT: length truncation can remove
    # image tokens and corrupt multimodal samples.
    # Smoke tests use full-sequence loss because TRL treats this as a language
    # modeling dataset. Completion-only loss needs a prompt-completion dataset
    # conversion that TRL recognizes.
    kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(train_cfg["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": 1
        if smoke_test
        else int(train_cfg["gradient_accumulation_steps"]),
        "learning_rate": float(train_cfg["learning_rate"]),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.0)),
        "num_train_epochs": float(train_cfg["num_train_epochs"]),
        "max_steps": max_steps,
        "max_length": train_cfg.get("max_length"),
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
        "completion_only_loss": bool(train_cfg.get("completion_only_loss", False)),
        "assistant_only_loss": bool(train_cfg.get("assistant_only_loss", False)),
    }
    if smoke_test or not has_eval_dataset:
        kwargs.pop("eval_steps", None)

    params = inspect.signature(sft_config_cls).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_strategy
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = eval_strategy

    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    supported_kwargs = kwargs if accepts_var_kwargs else {key: value for key, value in kwargs.items() if key in params}
    return sft_config_cls(**supported_kwargs)


def build_trainer(trainer_cls: Any, processor: Any, **kwargs: Any) -> Any:
    params = inspect.signature(trainer_cls).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = processor
    elif "tokenizer" in params:
        kwargs["tokenizer"] = processor

    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    supported_kwargs = kwargs if accepts_var_kwargs else {key: value for key, value in kwargs.items() if key in params}
    return trainer_cls(**supported_kwargs)


def train(config: dict[str, Any], smoke_test: bool = False, smoke_max_samples: int = 2) -> None:
    deps = require_training_imports()

    model_cfg = config["model"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]

    image_root = Path(data_cfg["image_root"])
    train_rows = read_jsonl(Path(data_cfg["train_jsonl"]))
    val_path = Path(data_cfg["val_jsonl"])
    val_rows = read_jsonl(val_path) if val_path.exists() else []

    if smoke_test:
        train_rows = limit_rows(train_rows, min(smoke_max_samples, 2))
        val_rows = limit_rows(val_rows, 1)
        print(f"[smoke] loaded {len(train_rows)} train rows and {len(val_rows)} val rows")
    else:
        print(f"Loaded {len(train_rows)} train rows and {len(val_rows)} val rows")

    train_examples = [row_to_sft_example(row, image_root) for row in train_rows]
    val_examples = [row_to_sft_example(row, image_root) for row in val_rows]
    train_dataset = examples_to_dataset(train_examples, deps["Dataset"])
    eval_dataset = examples_to_dataset(val_examples, deps["Dataset"]) if val_examples else None

    local_files_only = bool(model_cfg.get("local_files_only", False))
    processor = deps["AutoProcessor"].from_pretrained(
        model_cfg["name"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=local_files_only,
    )
    quantization_config = deps["BitsAndBytesConfig"](
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype_from_name(
            deps["torch"], str(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        ),
        bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
    )
    model = deps["Qwen2_5_VLForConditionalGeneration"].from_pretrained(
        model_cfg["name"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=local_files_only,
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
        str(output_cfg["output_dir"]),
        smoke_test=smoke_test,
        has_eval_dataset=eval_dataset is not None,
    )

    trainer = build_trainer(
        deps["SFTTrainer"],
        processor,
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_cfg["output_dir"]))
    print(f"Saved adapter to {output_cfg['output_dir']}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    train(config, smoke_test=args.smoke_test, smoke_max_samples=args.smoke_max_samples)


if __name__ == "__main__":
    main()
