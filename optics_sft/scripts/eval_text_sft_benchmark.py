#!/usr/bin/env python3
"""Run a text SFT benchmark on base and/or fine-tuned models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optics_sft.eval.control_metrics import summarize_benchmark
from optics_sft.physics.text_prompt_builder import build_text_prompt
from optics_sft.physics.text_target import normalize_training_target
from optics_sft.utils.paths import resolve_adapter_path, resolve_config_path


def resolve_model_reference(model_arg: Path, *, local_files_only: bool) -> str:
    candidate = model_arg.expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    if candidate.is_dir() and (candidate / "config.json").exists():
        return str(candidate)
    if local_files_only:
        raise FileNotFoundError(
            "Local model directory not found: "
            f"{candidate}\n"
            "Download it first, for example:\n"
            f"  huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir {candidate}\n"
            "Or drop --local-files-only and use --model-name Qwen/Qwen2.5-3B-Instruct"
        )
    return str(model_arg)


def resolve_optional_path(path_arg: Path | None) -> Path | None:
    if path_arg is None:
        return None
    return resolve_config_path(path_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate text SFT rows with a causal LM.")
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--model-name", type=Path, required=True, help="Base model path or HF id.")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--save-predictions-jsonl",
        type=Path,
        default=None,
        help="Optional path to write per-sample generated_text for sign diagnostics.",
    )
    parser.add_argument(
        "--training-target",
        choices=("full", "compact", "control_plan_only"),
        default=None,
        help="User prompt template for inference (default: row target_format or compact).",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def require_inference_imports() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Missing inference dependencies. Install torch, transformers, peft, and bitsandbytes."
        ) from exc
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def prompt_messages_for_row(row: dict[str, Any], training_target: str | None) -> list[dict[str, str]]:
    explicit = training_target or row.get("target_format")
    if explicit is None:
        explicit = "compact"
    mode = normalize_training_target(
        explicit,
        compact_target=explicit == "compact" or row.get("target_format") == "compact",
    )
    return [{"role": "user", "content": build_text_prompt(row, training_target=mode)}]


def generate_predictions(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    adapter_path: Path | None,
    max_new_tokens: int,
    local_files_only: bool,
    training_target: str | None,
) -> list[dict[str, Any]]:
    deps = require_inference_imports()
    tokenizer = deps["AutoTokenizer"].from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = deps["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=deps["torch"].bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = deps["AutoModelForCausalLM"].from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
        local_files_only=local_files_only,
        device_map="auto",
    )
    if adapter_path is not None:
        model = deps["PeftModel"].from_pretrained(model, str(adapter_path))
    model.eval()

    predictions: list[dict[str, Any]] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id", f"row_{index}")
        print(f"[eval {index}/{total}] generating {sample_id}", flush=True)
        prompt_messages = prompt_messages_for_row(row, training_target)
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predictions.append(
            {
                "sample_id": row.get("sample_id"),
                "generated_text": generated_text,
            }
        )
    return predictions


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.eval_jsonl)
    if args.max_samples is not None:
        rows = rows[: max(0, args.max_samples)]

    model_name = resolve_model_reference(args.model_name, local_files_only=args.local_files_only)
    adapter_path = resolve_adapter_path(args.adapter_path)

    predictions = generate_predictions(
        rows,
        model_name=model_name,
        adapter_path=adapter_path,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_files_only,
        training_target=args.training_target,
    )
    model_label = "sft" if adapter_path is not None else "base"
    report = summarize_benchmark(predictions, rows, model_name=model_label)
    report["eval_jsonl"] = str(args.eval_jsonl)
    report["model_path"] = model_name
    report["adapter_path"] = str(adapter_path) if adapter_path else None
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.save_predictions_jsonl is not None:
        args.save_predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.save_predictions_jsonl.open("w", encoding="utf-8") as handle:
            for pred in predictions:
                handle.write(json.dumps(pred, sort_keys=True) + "\n")
        print(f"Wrote predictions to {args.save_predictions_jsonl}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
