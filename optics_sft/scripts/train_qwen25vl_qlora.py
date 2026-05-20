#!/usr/bin/env python3
"""Train a Qwen2.5-VL adapter for optics SFT using 4-bit QLoRA.

The script keeps the first executable path intentionally small:

- load YAML config
- load JSONL rows
- load current/target images with PIL
- build examples containing `images`, conversational `prompt`, and
  assistant-only `completion`
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
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optics_sft.physics.prompt_builder import build_physics_prompt, expected_image_slots


PROMPT_MODES = {"image_only", "setup_only", "metadata_assisted"}
LABEL_MODES = {"continuous_control", "direction_classification"}
DATASET_FORMATS = {"legacy_pair", "physics_mixed"}
SETUP_METADATA_KEYS = (
    "wavelength_nm",
    "beam_waist_mm",
    "lens_focal_length_mm",
    "source_to_lens_mm",
    "lens_to_camera_mm",
)
ZERO_THRESHOLD_MM = 0.005
SMALL_THRESHOLD_MM = 0.025
MEDIUM_THRESHOLD_MM = 0.060


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


def filtered_metadata(metadata: dict[str, Any], prompt_mode: str) -> dict[str, Any]:
    if prompt_mode == "metadata_assisted":
        return metadata
    if prompt_mode == "setup_only":
        return {key: metadata[key] for key in SETUP_METADATA_KEYS if key in metadata}
    if prompt_mode == "image_only":
        return {}
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}. Expected one of {sorted(PROMPT_MODES)}")


def direction_label(delta_mm: float) -> str:
    if abs(delta_mm) < ZERO_THRESHOLD_MM:
        return "zero"
    return "positive" if delta_mm > 0 else "negative"


def magnitude_class(delta_mm: float) -> str:
    abs_delta = abs(delta_mm)
    if abs_delta < ZERO_THRESHOLD_MM:
        return "zero"
    if abs_delta < SMALL_THRESHOLD_MM:
        return "small"
    if abs_delta < MEDIUM_THRESHOLD_MM:
        return "medium"
    return "large"


def relative_position_from_direction(direction: str, axis: str) -> str:
    if direction == "zero":
        return "center"
    if axis == "x":
        return "left" if direction == "positive" else "right"
    if axis == "y":
        return "below" if direction == "negative" else "above"
    raise ValueError(f"Unsupported axis: {axis}")


def direction_classification_target(label: dict[str, Any]) -> dict[str, Any]:
    plan = label.get("control_plan")
    if not isinstance(plan, dict):
        raise ValueError("Direction classification labels require label.control_plan.")

    lens_x_delta = float(plan["lens_x_delta_mm"])
    lens_y_delta = float(plan["lens_y_delta_mm"])
    lens_x_direction = direction_label(lens_x_delta)
    lens_y_direction = direction_label(lens_y_delta)
    return {
        "task": "beam_alignment_direction_classification",
        "visual_diagnosis": {
            "current_relative_to_target_x": relative_position_from_direction(lens_x_direction, "x"),
            "current_relative_to_target_y": relative_position_from_direction(lens_y_direction, "y"),
        },
        "control_intent": {
            "lens_x_direction": lens_x_direction,
            "lens_x_magnitude_class": magnitude_class(lens_x_delta),
            "lens_y_direction": lens_y_direction,
            "lens_y_magnitude_class": magnitude_class(lens_y_delta),
        },
    }


def target_for_label_mode(row: dict[str, Any], label_mode: str) -> dict[str, Any]:
    if label_mode == "continuous_control":
        return row["label"]
    if label_mode == "direction_classification":
        return direction_classification_target(row["label"])
    raise ValueError(f"Unsupported label_mode: {label_mode}. Expected one of {sorted(LABEL_MODES)}")


def build_prompt(
    metadata: dict[str, Any],
    prompt_mode: str = "metadata_assisted",
    label_mode: str = "continuous_control",
) -> str:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}. Expected one of {sorted(PROMPT_MODES)}")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unsupported label_mode: {label_mode}. Expected one of {sorted(LABEL_MODES)}")

    base = (
        "You are controlling an optical beam setup. Compare the current beam "
        "image with the target beam image. "
    )
    direction_schema = (
        "Return only strict JSON with task, visual_diagnosis, and control_intent. "
        "visual_diagnosis must classify current_relative_to_target_x as left, center, or right, "
        "and current_relative_to_target_y as above, center, or below. "
        "control_intent must classify lens_x_direction and lens_y_direction as negative, zero, or positive, "
        "and lens_x_magnitude_class and lens_y_magnitude_class as zero, small, medium, or large."
    )

    if prompt_mode == "image_only":
        if label_mode == "direction_classification":
            return (
                base
                + "Use only the two images to classify the visual offset and lens correction intent. "
                + direction_schema
            )
        return (
            base
            + "Use only the two images to infer the needed optical alignment correction. "
            + "Return only strict JSON matching the training response format for this task."
        )

    prompt_metadata = filtered_metadata(metadata, prompt_mode)
    metadata_label = "setup metadata" if prompt_mode == "setup_only" else "metadata"
    if label_mode == "direction_classification":
        return (
            base
            + f"Use the {metadata_label} below to classify the visual offset and lens correction intent. "
            + direction_schema
            + f"\n\n{metadata_label}: {json.dumps(prompt_metadata, sort_keys=True)}"
        )
    return (
        base
        + f"Use the {metadata_label} below and return only strict JSON containing these top-level keys: "
        + "task, diagnosis, control_plan, confidence. The control_plan must contain numeric "
        + "lens_x_delta_mm, lens_y_delta_mm, camera_x_delta_mm, and camera_y_delta_mm values.\n\n"
        + f"{metadata_label}: {json.dumps(prompt_metadata, sort_keys=True)}"
    )


def build_prompt_messages(prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_completion_messages(completion: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": completion}],
        }
    ]


def build_variable_image_prompt_messages(prompt: str, image_count: int) -> list[dict[str, Any]]:
    if image_count <= 0:
        raise ValueError("Physics SFT examples require at least one image slot.")
    return [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(image_count)]
            + [{"type": "text", "text": prompt}],
        }
    ]


def row_to_sft_example(
    row: dict[str, Any],
    image_root: Path,
    prompt_mode: str,
    label_mode: str,
) -> dict[str, Any]:
    current_path = resolve_image_path(image_root, row["current_image_path"])
    target_path = resolve_image_path(image_root, row["target_image_path"])
    current_image = load_rgb_image(current_path)
    target_image = load_rgb_image(target_path)
    prompt = build_prompt(row.get("metadata", {}), prompt_mode, label_mode)
    completion = json.dumps(target_for_label_mode(row, label_mode), sort_keys=True)

    return {
        "images": [current_image, target_image],
        "prompt": build_prompt_messages(prompt),
        "completion": build_completion_messages(completion),
    }


def physics_row_to_sft_example(row: dict[str, Any], image_root: Path) -> dict[str, Any]:
    sample_type = row["sample_type"]
    if not isinstance(sample_type, str):
        raise ValueError("Physics rows require string sample_type.")

    prompt = build_physics_prompt(row)
    image_slots = expected_image_slots(row)
    images = [
        load_rgb_image(resolve_image_path(image_root, slot["path"]))
        for slot in image_slots
    ]
    completion = json.dumps(row["target"], sort_keys=True)

    return {
        "images": images,
        "prompt": build_variable_image_prompt_messages(prompt, len(images)),
        "completion": build_completion_messages(completion),
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
    # Prompt and completion are kept as separate conversational fields so TRL's
    # VLM collator can mask the prompt when completion_only_loss=True.
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
    output_dir = str(train_cfg.get("output_dir", output_cfg["output_dir"]))
    if smoke_test:
        output_dir = f"{output_dir}_smoke"

    image_root = Path(data_cfg["image_root"])
    dataset_format = str(data_cfg.get("dataset_format", "legacy_pair"))
    if dataset_format not in DATASET_FORMATS:
        raise ValueError(f"Unsupported dataset_format: {dataset_format}. Expected one of {sorted(DATASET_FORMATS)}")

    prompt_mode = str(data_cfg.get("prompt_mode", "metadata_assisted"))
    label_mode = str(data_cfg.get("label_mode", "continuous_control"))
    if dataset_format == "legacy_pair":
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}. Expected one of {sorted(PROMPT_MODES)}")
        if label_mode not in LABEL_MODES:
            raise ValueError(f"Unsupported label_mode: {label_mode}. Expected one of {sorted(LABEL_MODES)}")

    train_rows = read_jsonl(Path(data_cfg["train_jsonl"]))
    val_path = Path(data_cfg["val_jsonl"])
    val_rows = read_jsonl(val_path) if val_path.exists() else []

    if smoke_test:
        train_rows = limit_rows(train_rows, min(smoke_max_samples, 2))
        val_rows = limit_rows(val_rows, 1)
        print(f"[smoke] loaded {len(train_rows)} train rows and {len(val_rows)} val rows ({dataset_format})")
    else:
        print(f"Loaded {len(train_rows)} train rows and {len(val_rows)} val rows ({dataset_format})")

    if dataset_format == "legacy_pair":
        train_examples = [row_to_sft_example(row, image_root, prompt_mode, label_mode) for row in train_rows]
        val_examples = [row_to_sft_example(row, image_root, prompt_mode, label_mode) for row in val_rows]
    else:
        train_examples = [physics_row_to_sft_example(row, image_root) for row in train_rows]
        val_examples = [physics_row_to_sft_example(row, image_root) for row in val_rows]

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
        output_dir,
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
    trainer.save_model(output_dir)
    print(f"Saved adapter to {output_dir}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    train(config, smoke_test=args.smoke_test, smoke_max_samples=args.smoke_max_samples)


if __name__ == "__main__":
    main()
