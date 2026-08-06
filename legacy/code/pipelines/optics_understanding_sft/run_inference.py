#!/usr/bin/env python3
"""Deterministic base-model or adapter inference for canonical pilot records."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

from .core import load_yaml, read_jsonl, stable_json_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--task-type")
    parser.add_argument(
        "--source-task-type",
        help="Filter derived records by their original task family when present.",
    )
    parser.add_argument(
        "--smoke-mixed",
        action="store_true",
        help="Select one text-only and one visual row after other filters.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def require_inference_imports() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Run inference in the optics_qlora Conda environment") from exc
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def dtype_from_name(torch_module: Any, name: str) -> Any:
    if not hasattr(torch_module, name):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return getattr(torch_module, name)


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
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as direct_error:
        block = first_json_object_text(stripped)
        if block is None:
            return None, str(direct_error)
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError as block_error:
            return None, str(block_error)
    if not isinstance(parsed, dict):
        return None, "generated JSON is not an object"
    return parsed, None


def load_image(path: Path) -> Any:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.load()
    return image


def load_model(config: Mapping[str, Any], adapter_path: Path | None) -> tuple[Any, Any, dict[str, Any]]:
    deps = require_inference_imports()
    model_cfg = config["model"]
    processor = deps["AutoProcessor"].from_pretrained(
        model_cfg["name"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )
    quantization = deps["BitsAndBytesConfig"](
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=dtype_from_name(
            deps["torch"], str(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        ),
        bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
    )
    model = deps["AutoModelForImageTextToText"].from_pretrained(
        model_cfg["name"],
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )
    if adapter_path is not None:
        if not adapter_path.exists():
            raise FileNotFoundError(adapter_path)
        model = deps["PeftModel"].from_pretrained(model, adapter_path, local_files_only=True)
    model.eval()
    return processor, model, deps


def generation_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = config.get("generation", {})
    do_sample = bool(cfg.get("do_sample", False))
    result: dict[str, Any] = {
        "max_new_tokens": int(cfg.get("max_new_tokens", 384)),
        "do_sample": do_sample,
    }
    if do_sample:
        result["temperature"] = float(cfg.get("temperature", 0.7))
    return result


def user_messages(prompt: str, image_count: int) -> list[dict[str, Any]]:
    content = [{"type": "image"} for _ in range(image_count)]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def predict_record(
    record: Mapping[str, Any],
    *,
    image_root: Path,
    processor: Any,
    model: Any,
    deps: Mapping[str, Any],
    config: Mapping[str, Any],
    model_kind: str,
) -> dict[str, Any]:
    image_paths = [str(value) for value in record["prompt_inputs"].get("images", [])]
    images = [load_image(image_root / value) for value in image_paths]
    messages = user_messages(str(record["prompt"]), len(images))
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs: dict[str, Any] = {"text": [rendered], "return_tensors": "pt"}
    if images:
        processor_kwargs["images"] = images
    inputs = processor(**processor_kwargs)
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    torch = deps["torch"]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs(config))
    elapsed = time.perf_counter() - started
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    generated_tokens = int(generated.shape[-1] - prompt_tokens)
    raw = processor.batch_decode(generated[:, prompt_tokens:], skip_special_tokens=True)[0].strip()
    parsed, parse_error = extract_json_object(raw)
    peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    return {
        "example_id": record["example_id"],
        "group_id": record["group_id"],
        "task_type": record["task_type"],
        "modality": record["modality"],
        "model_kind": model_kind,
        "prompt_hash": stable_json_hash(record["prompt"]),
        "image_paths": image_paths,
        "raw_prediction_text": raw,
        "parsed_json": parsed,
        "parse_error": parse_error,
        "input_tokens": prompt_tokens,
        "output_tokens": generated_tokens,
        "latency_seconds": elapsed,
        "peak_cuda_memory_bytes": peak_memory,
    }


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["example_id"] for row in read_jsonl(path)}


def write_run_manifest(
    output_jsonl: Path,
    *,
    config: Mapping[str, Any],
    rows: list[dict[str, Any]],
    input_jsonl: Path,
    image_root: Path,
    adapter_path: Path | None,
) -> None:
    manifest = {
        "model_kind": "adapter" if adapter_path is not None else "base",
        "model_name": config["model"]["name"],
        "adapter_path": str(adapter_path.resolve()) if adapter_path is not None else None,
        "generation": generation_kwargs(config),
        "config_hash": stable_json_hash(config),
        "input_jsonl": str(input_jsonl.resolve()),
        "input_records_hash": stable_json_hash(rows),
        "image_root": str(image_root.resolve()),
        "record_count": len(rows),
        "example_ids": [row["example_id"] for row in rows],
    }
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.with_suffix(".run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    rows = read_jsonl(args.input_jsonl)
    if args.task_type:
        rows = [row for row in rows if row.get("task_type") == args.task_type]
    if args.source_task_type:
        rows = [
            row for row in rows if row.get("source_task_type") == args.source_task_type
        ]
    if args.smoke_mixed:
        text_row = next((row for row in rows if row.get("modality") == "text"), None)
        visual_row = next((row for row in rows if row.get("modality") == "visual"), None)
        if text_row is None or visual_row is None:
            raise ValueError("--smoke-mixed requires at least one text and one visual record")
        rows = [text_row, visual_row]
    if args.max_samples is not None:
        rows = rows[: max(0, args.max_samples)]
    write_run_manifest(
        args.output_jsonl,
        config=config,
        rows=rows,
        input_jsonl=args.input_jsonl,
        image_root=args.image_root,
        adapter_path=args.adapter_path,
    )
    completed = existing_ids(args.output_jsonl) if args.resume else set()
    if args.output_jsonl.exists() and not args.resume:
        args.output_jsonl.unlink()
    pending = [row for row in rows if row["example_id"] not in completed]
    print(f"Loaded {len(rows)} records; {len(pending)} pending", flush=True)
    if not pending:
        return
    processor, model, deps = load_model(config, args.adapter_path)
    model_kind = "adapter" if args.adapter_path is not None else "base"
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("a", encoding="utf-8") as stream:
        for index, record in enumerate(pending, start=1):
            try:
                result = predict_record(
                    record,
                    image_root=args.image_root,
                    processor=processor,
                    model=model,
                    deps=deps,
                    config=config,
                    model_kind=model_kind,
                )
            except Exception as exc:
                result = {
                    "example_id": record["example_id"],
                    "group_id": record["group_id"],
                    "task_type": record["task_type"],
                    "modality": record["modality"],
                    "model_kind": model_kind,
                    "raw_prediction_text": "",
                    "parsed_json": None,
                    "parse_error": f"{type(exc).__name__}: {exc}",
                }
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
            print(
                f"[{index}/{len(pending)}] {record['example_id']} "
                f"json={'ok' if result.get('parsed_json') is not None else 'invalid'} "
                f"latency={result.get('latency_seconds', 0.0):.2f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
