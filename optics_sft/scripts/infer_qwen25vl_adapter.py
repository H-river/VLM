#!/usr/bin/env python3
"""Run one optics control-plan inference with a trained Qwen2.5-VL adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_qwen25vl_qlora import build_prompt, load_rgb_image, load_yaml, resolve_image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run optics control-plan inference with a Qwen2.5-VL adapter."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("optics_sft/configs/qwen25vl_3b_qlora.yaml"),
        help="Path to the QLoRA YAML config.",
    )
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="Optional local base model path. Overrides the base model path from config.",
    )
    parser.add_argument("--current-image", type=Path, default=None)
    parser.add_argument("--target-image", type=Path, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument(
        "--sample-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL dataset file. When set, image paths and metadata are read from one row.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata JSON must contain an object: {path}")
    return metadata


def resolve_base_model(args: argparse.Namespace, config: dict[str, Any]) -> str:
    if args.base_model is not None:
        return str(args.base_model)

    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict) and model_cfg.get("name"):
        return str(model_cfg["name"])
    if config.get("model_name"):
        return str(config["model_name"])
    raise ValueError("Base model path was not provided and was not found in config.")


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter-dir: {args.adapter_dir}")
    if args.sample_jsonl is not None:
        if not args.sample_jsonl.exists():
            raise FileNotFoundError(f"Missing sample-jsonl: {args.sample_jsonl}")
        return

    missing = [
        name
        for name in ("current_image", "target_image", "metadata_json")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "Provide either --sample-jsonl or all of --current-image, --target-image, and --metadata-json. "
            f"Missing: {', '.join(missing)}"
        )
    for name in ("current_image", "target_image", "metadata_json"):
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(f"Missing {name.replace('_', '-')}: {path}")


def load_sample_from_jsonl(path: Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise ValueError("--sample-index must be non-negative")
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if line_number == index:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Sample row must be an object: {path}:{line_number + 1}")
                return row
    raise IndexError(f"Sample index {index} is out of range for {path}")


def resolve_row_image_path(row_path: str, image_root: Path | None) -> Path:
    path = Path(row_path)
    if path.is_absolute():
        return path
    if image_root is None:
        raise ValueError("Relative sample image paths require --image-root or config data.image_root.")
    return resolve_image_path(image_root, row_path)


def resolve_inputs(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    if args.sample_jsonl is None:
        return args.current_image, args.target_image, load_metadata(args.metadata_json)

    row = load_sample_from_jsonl(args.sample_jsonl, args.sample_index)
    config_data = config.get("data", {})
    image_root = args.image_root or (
        Path(config_data["image_root"])
        if isinstance(config_data, dict) and config_data.get("image_root")
        else None
    )
    current_image = resolve_row_image_path(row["current_image_path"], image_root)
    target_image = resolve_row_image_path(row["target_image_path"], image_root)
    metadata = load_metadata(args.metadata_json) if args.metadata_json else row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Sample metadata must be an object")
    return current_image, target_image, metadata


def require_inference_imports() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Missing inference dependencies. Install torch, Pillow, "
            "transformers, peft, bitsandbytes, and PyYAML."
        ) from exc

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def dtype_from_name(torch_module: Any, name: str) -> Any:
    if not hasattr(torch_module, name):
        raise ValueError(f"Unsupported torch dtype in config: {name}")
    return getattr(torch_module, name)


def user_messages(prompt: str) -> list[dict[str, Any]]:
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


def first_json_block(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        direct_error = str(exc)
    else:
        if isinstance(parsed, dict):
            return parsed, None
        return None, "Generated text is valid JSON but not a JSON object."

    block = first_json_block(text)
    if block is None:
        return None, direct_error
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "Extracted JSON is not an object."
    return parsed, None


def generation_kwargs(config: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    generation_cfg = config.get("generation", {})
    do_sample = bool(generation_cfg.get("do_sample", False))
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs["temperature"] = float(generation_cfg.get("temperature", 0.7))
    return kwargs


def generate_prediction(
    args: argparse.Namespace,
    config: dict[str, Any],
    current_image_path: Path,
    target_image_path: Path,
    metadata: dict[str, Any],
    base_model_path: str,
) -> dict[str, Any]:
    current_image = load_rgb_image(current_image_path)
    target_image = load_rgb_image(target_image_path)
    prompt = build_prompt(metadata)

    deps = require_inference_imports()
    model_cfg = config["model"]
    generation_cfg = config.get("generation", {})

    processor = deps["AutoProcessor"].from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    quantization_config = deps["BitsAndBytesConfig"](
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype_from_name(
            deps["torch"], str(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        ),
        bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
    )
    base_model = deps["AutoModelForImageTextToText"].from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = deps["PeftModel"].from_pretrained(
        base_model,
        args.adapter_dir,
        local_files_only=True,
    )
    model.eval()

    messages = user_messages(prompt)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[current_image, target_image],
        return_tensors="pt",
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    max_new_tokens = args.max_new_tokens or int(generation_cfg.get("max_new_tokens", 256))
    with deps["torch"].no_grad():
        generated = model.generate(**inputs, **generation_kwargs(config, max_new_tokens))
    prompt_len = inputs["input_ids"].shape[-1]
    generated_only = generated[:, prompt_len:]
    raw_text = processor.batch_decode(generated_only, skip_special_tokens=True)[0].strip()
    parsed_json, parse_error = extract_json_object(raw_text)

    result = {
        "raw_text": raw_text,
        "base_model": base_model_path,
        "adapter_dir": str(args.adapter_dir),
        "current_image": str(current_image_path),
        "target_image": str(target_image_path),
        "metadata": metadata,
    }
    if parsed_json is not None:
        result["parsed_json"] = parsed_json
    else:
        result["parse_error"] = parse_error or "Failed to parse generated JSON."
    return result


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    config = load_yaml(args.config)
    base_model_path = resolve_base_model(args, config)
    current_image_path, target_image_path, metadata = resolve_inputs(args, config)
    result = generate_prediction(
        args,
        config,
        current_image_path,
        target_image_path,
        metadata,
        base_model_path,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote inference result to {args.output_json}")


if __name__ == "__main__":
    main()
