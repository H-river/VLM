#!/usr/bin/env python3
"""Batch inference for physics_mixed SFT rows with a Qwen2.5-VL LoRA adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optics_sft.physics.prompt_builder import build_physics_prompt, expected_image_slots
from optics_sft.scripts.train_qwen25vl_qlora import (
    dtype_from_name,
    load_rgb_image,
    load_yaml,
    resolve_image_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch inference for physics_mixed rows with a Qwen2.5-VL adapter."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("optics_sft/configs/qwen25vl_3b_qlora_physics_mixed.yaml"),
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    if max_samples is not None and max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
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
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


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


def first_json_object_text(text: str) -> str | None:
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
        return None, "Generated text is valid JSON but not an object."

    block = first_json_object_text(text)
    if block is None:
        return None, direct_error
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "Extracted JSON is not an object."
    return parsed, None


def build_user_messages(prompt: str, image_count: int) -> list[dict[str, Any]]:
    if image_count <= 0:
        raise ValueError("Physics inference rows require at least one image slot.")
    return [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(image_count)]
            + [{"type": "text", "text": prompt}],
        }
    ]


def generation_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    generation_cfg = config.get("generation", {})
    if not isinstance(generation_cfg, dict):
        generation_cfg = {}

    do_sample = bool(generation_cfg.get("do_sample", False))
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation_cfg.get("max_new_tokens", 512)),
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs["temperature"] = float(generation_cfg.get("temperature", 0.7))
    return kwargs


def resolve_slot_image(slot: dict[str, str], image_root: Path) -> Path:
    return resolve_image_path(image_root, slot["path"])


def row_prompt_and_images(row: dict[str, Any], image_root: Path) -> tuple[str, list[Any], list[dict[str, str]]]:
    prompt = build_physics_prompt(row)
    slots = expected_image_slots(row)
    images = [load_rgb_image(resolve_slot_image(slot, image_root)) for slot in slots]
    return prompt, images, slots


def load_model_and_processor(config: dict[str, Any], adapter_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    deps = require_inference_imports()
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("Config model section must be an object.")
    model_name = str(model_cfg["name"])
    local_files_only = bool(model_cfg.get("local_files_only", True))
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))

    processor = deps["AutoProcessor"].from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    quantization_config = deps["BitsAndBytesConfig"](
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype_from_name(
            deps["torch"],
            str(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        ),
        bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
    )
    base_model = deps["AutoModelForImageTextToText"].from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model = deps["PeftModel"].from_pretrained(
        base_model,
        adapter_path,
        local_files_only=True,
    )
    model.eval()
    return processor, model, deps


def predict_row(
    row: dict[str, Any],
    sample_index: int,
    image_root: Path,
    config: dict[str, Any],
    processor: Any,
    model: Any,
    deps: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    prompt, images, image_slots = row_prompt_and_images(row, image_root)
    messages = build_user_messages(prompt, len(images))
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with deps["torch"].no_grad():
        generated = model.generate(
            **inputs,
            **generation_kwargs(config),
        )
    prompt_len = inputs["input_ids"].shape[-1]
    raw_prediction_text = processor.batch_decode(
        generated[:, prompt_len:],
        skip_special_tokens=True,
    )[0].strip()
    parsed_json, parse_error = extract_json_object(raw_prediction_text)
    if parse_error:
        errors.append(parse_error)

    return {
        "sample_index": sample_index,
        "sample_id": row.get("sample_id"),
        "sample_type": row.get("sample_type"),
        "image_slots": image_slots,
        "raw_prediction_text": raw_prediction_text,
        "parsed_json": parsed_json,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Config does not exist: {args.config}")
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {args.input_jsonl}")
    if not args.image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {args.image_root}")

    config = load_yaml(args.config)
    rows = read_jsonl(args.input_jsonl, args.max_samples)
    processor, model, deps = load_model_and_processor(config, args.adapter_path)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for sample_index, row in enumerate(rows):
            try:
                prediction = predict_row(
                    row=row,
                    sample_index=sample_index,
                    image_root=args.image_root,
                    config=config,
                    processor=processor,
                    model=model,
                    deps=deps,
                )
            except Exception as exc:  # Keep batch output inspectable on row-level failures.
                prediction = {
                    "sample_index": sample_index,
                    "sample_id": row.get("sample_id"),
                    "sample_type": row.get("sample_type"),
                    "raw_prediction_text": None,
                    "parsed_json": None,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            f.write(json.dumps(prediction, sort_keys=True) + "\n")
            print(f"[{sample_index + 1}/{len(rows)}] wrote prediction for {row.get('sample_id')}")

    print(f"Wrote {len(rows)} predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
