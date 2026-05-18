#!/usr/bin/env python3
"""Batch inference for optics SFT validation rows with a Qwen2.5-VL LoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from infer_qwen25vl_adapter import (
    extract_json_object,
    generation_kwargs,
    require_inference_imports,
    user_messages,
)
from train_qwen25vl_qlora import build_prompt, load_rgb_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch optics control-plan inference with a Qwen2.5-VL adapter."
    )
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--sample-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
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
            if limit is not None and len(rows) >= limit:
                break
    return rows


def resolve_sample_image(path_value: str, image_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    return image_root / path


def load_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    deps = require_inference_imports()
    processor = deps["AutoProcessor"].from_pretrained(
        str(args.base_model),
        local_files_only=True,
        trust_remote_code=True,
    )
    quantization_config = deps["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=deps["torch"].bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = deps["AutoModelForImageTextToText"].from_pretrained(
        str(args.base_model),
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
    return processor, model, deps


def predict_row(
    row: dict[str, Any],
    sample_index: int,
    args: argparse.Namespace,
    processor: Any,
    model: Any,
    deps: dict[str, Any],
) -> dict[str, Any]:
    current_image_path = resolve_sample_image(row["current_image_path"], args.image_root)
    target_image_path = resolve_sample_image(row["target_image_path"], args.image_root)
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    prompt = build_prompt(metadata)
    text = processor.apply_chat_template(
        user_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[load_rgb_image(current_image_path), load_rgb_image(target_image_path)],
        return_tensors="pt",
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    gen_config = {
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        }
    }
    with deps["torch"].no_grad():
        generated = model.generate(
            **inputs,
            **generation_kwargs(gen_config, args.max_new_tokens),
        )
    prompt_len = inputs["input_ids"].shape[-1]
    raw_text = processor.batch_decode(
        generated[:, prompt_len:],
        skip_special_tokens=True,
    )[0].strip()
    parsed_json, parse_error = extract_json_object(raw_text)

    return {
        "sample_index": sample_index,
        "current_image": str(current_image_path),
        "target_image": str(target_image_path),
        "metadata": metadata,
        "raw_text": raw_text,
        "parsed_json": parsed_json,
        "parse_error": parse_error,
    }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter-dir: {args.adapter_dir}")
    if not args.sample_jsonl.exists():
        raise FileNotFoundError(f"Missing sample-jsonl: {args.sample_jsonl}")
    if not args.image_root.exists():
        raise FileNotFoundError(f"Missing image-root: {args.image_root}")

    rows = read_jsonl(args.sample_jsonl, args.limit)
    processor, model, deps = load_model(args)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for sample_index, row in enumerate(rows):
            prediction = predict_row(row, sample_index, args, processor, model, deps)
            f.write(json.dumps(prediction, sort_keys=True) + "\n")
            print(f"[{sample_index + 1}/{len(rows)}] wrote prediction")

    print(f"Wrote {len(rows)} predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
