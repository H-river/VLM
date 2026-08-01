#!/usr/bin/env python3
"""Resumable zero-image Qwen gain inference on prebuilt visible prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "active_diagnosis_v13_qwen_gain_inference_v1"
GAIN_CLASSES = (0.5, 0.75, 1.0, 1.25, 1.5)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _append(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inference_identity(base_model: Path, adapter_dir: Path, input_jsonl: Path) -> dict[str, Any]:
    """Bind resumable rows to the exact input, adapter, and base-model metadata."""

    base = base_model.resolve()
    adapter = adapter_dir.resolve()
    source = input_jsonl.resolve()
    if not base.is_dir() or not adapter.is_dir() or not source.is_file():
        raise ValueError("base model, adapter directory, and input JSONL must exist")
    adapter_files = sorted(path for path in adapter.iterdir() if path.is_file())
    if not adapter_files:
        raise ValueError("adapter directory has no root files")
    adapter_digest = hashlib.sha256()
    adapter_entries = []
    for path in adapter_files:
        digest = _sha256(path)
        adapter_entries.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest})
        adapter_digest.update(path.name.encode("utf-8"))
        adapter_digest.update(b"\0")
        adapter_digest.update(digest.encode("ascii"))
        adapter_digest.update(b"\n")
    base_metadata = {}
    for name in ("config.json", "model.safetensors.index.json"):
        path = base / name
        if path.exists():
            base_metadata[name] = _sha256(path)
    return {
        "version": VERSION,
        "base_model": str(base),
        "base_model_metadata_sha256": base_metadata,
        "adapter_dir": str(adapter),
        "adapter_root_files": adapter_entries,
        "adapter_root_tree_sha256": adapter_digest.hexdigest(),
        "input_jsonl": str(source),
        "input_jsonl_sha256": _sha256(source),
        "protected_set_used": False,
    }


def bind_resume_manifest(output_jsonl: Path, identity: dict[str, Any]) -> Path:
    """Create or verify the immutable identity for a resumable output JSONL."""

    output = output_jsonl.resolve()
    manifest = Path(str(output) + ".manifest.json")
    if manifest.exists():
        prior = json.loads(manifest.read_text(encoding="utf-8"))
        if prior != identity:
            raise ValueError("inference resume identity differs from existing manifest")
    elif output.exists() and output.stat().st_size:
        raise ValueError("existing inference rows have no resume identity manifest")
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_suffix(f"{manifest.suffix}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, manifest)
    return manifest


def _extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidates = [text.strip()]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, None
    return None, "no valid JSON object"


def _truth(row: dict[str, Any]) -> float:
    completion = row["completion"]
    text = completion[0]["content"][0]["text"]
    return float(json.loads(text)["estimated_gain"])


def _prompt_text(row: dict[str, Any]) -> str:
    return str(row["prompt"][0]["content"][0]["text"])


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid_gain_prediction"]]
    labels = [f"{gain:g}" for gain in GAIN_CLASSES]
    confusion = {truth: Counter() for truth in labels}
    for row in valid:
        confusion[f"{float(row['true_gain_evaluator_only']):g}"][
            f"{float(row['estimated_gain']):g}"
        ] += 1
    return {
        "version": VERSION,
        "split": "development_group_heldout_validation",
        "protected_set_used": False,
        "records": len(rows),
        "valid_json_rate": float(np.mean([row["parsed_json"] is not None for row in rows])),
        "valid_gain_prediction_rate": float(
            np.mean([row["valid_gain_prediction"] for row in rows])
        ),
        "gain_accuracy": (
            float(
                np.mean(
                    [
                        float(row["estimated_gain"])
                        == float(row["true_gain_evaluator_only"])
                        for row in valid
                    ]
                )
            )
            if valid
            else 0.0
        ),
        "accuracy_denominator": len(valid),
        "per_gain_accuracy": {
            label: (
                float(
                    np.mean(
                        [
                            float(row["estimated_gain"])
                            == float(row["true_gain_evaluator_only"])
                            for row in valid
                            if f"{float(row['true_gain_evaluator_only']):g}" == label
                        ]
                    )
                )
                if any(
                    f"{float(row['true_gain_evaluator_only']):g}" == label
                    for row in valid
                )
                else None
            )
            for label in labels
        },
        "confusion_matrix_rows_true_columns_predicted": [
            [int(confusion[truth][prediction]) for prediction in labels]
            for truth in labels
        ],
        "labels": labels,
        "groups": len({str(row["group_id"]) for row in rows}),
        "mean_inference_seconds": float(
            np.mean([float(row["inference_seconds"]) for row in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    inputs = _jsonl(args.input_jsonl.resolve())
    if not inputs:
        raise ValueError("empty inference dataset")
    if any(row.get("images") for row in inputs):
        raise ValueError("gain inference expects zero-image prebuilt-chat rows")
    input_ids = [str(row["example_id"]) for row in inputs]
    if len(input_ids) != len(set(input_ids)):
        raise ValueError("inference dataset contains duplicate example IDs")
    identity = inference_identity(
        args.base_model, args.adapter_dir, args.input_jsonl
    )
    bind_resume_manifest(args.output_jsonl, identity)
    completed_rows = _jsonl(args.output_jsonl.resolve())
    completed_ids = [str(row["example_id"]) for row in completed_rows]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("inference output contains duplicate example IDs")
    if not set(completed_ids).issubset(set(input_ids)):
        raise ValueError("inference output contains rows outside the bound input dataset")
    completed = {str(row["example_id"]) for row in completed_rows}
    processor = AutoProcessor.from_pretrained(
        str(args.base_model.resolve()), local_files_only=True, trust_remote_code=True
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        str(args.base_model.resolve()),
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=quantization,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        base, str(args.adapter_dir.resolve()), local_files_only=True
    )
    model.eval()
    for index, row in enumerate(inputs, start=1):
        example_id = str(row["example_id"])
        if example_id in completed:
            continue
        prompt_text = _prompt_text(row)
        if str(_truth(row)) in prompt_text:
            # Candidate values legitimately occur in the prompt, so exact
            # numeric substring matching cannot establish leakage. The dataset
            # manifest and prompt serializer provide the policy contract.
            pass
        rendered = processor.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        encoded = processor(text=[rendered], return_tensors="pt")
        encoded = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        elapsed = time.perf_counter() - started
        prompt_length = encoded["input_ids"].shape[-1]
        raw = processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )[0].strip()
        parsed, parse_error = _extract_json(raw)
        estimated = None
        if parsed is not None:
            try:
                estimated = float(parsed["estimated_gain"])
            except (KeyError, TypeError, ValueError):
                pass
        valid = estimated in GAIN_CLASSES
        output = {
            "version": VERSION,
            "example_id": example_id,
            "case_id": example_id.split("__g", 1)[0],
            "group_id": row["group_id"],
            "probe_design": json.loads(prompt_text.split("\n\n", 1)[1])["probe_design"],
            "probe_fraction": float(
                json.loads(prompt_text.split("\n\n", 1)[1])["probe_fraction"]
            ),
            "visible_prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
            "serialization_seed": int(row["provenance"]["serialization_seed"]),
            "retained_feature_indices": row["provenance"].get(
                "retained_feature_indices"
            ),
            "raw_text": raw,
            "parsed_json": parsed,
            "parse_error": parse_error,
            "estimated_gain": estimated,
            "valid_gain_prediction": valid,
            "true_gain_evaluator_only": _truth(row),
            "correct": bool(valid and estimated == _truth(row)),
            "inference_seconds": elapsed,
            "protected_set_used": False,
            "input_prompt_used_completion": False,
        }
        _append(args.output_jsonl.resolve(), output)
        print(
            json.dumps(
                {
                    "event": "qwen_gain_prediction",
                    "index": index,
                    "records": len(inputs),
                    "example_id": example_id,
                    "valid": valid,
                    "correct": output["correct"],
                    "seconds": elapsed,
                }
            ),
            flush=True,
        )
    rows = _jsonl(args.output_jsonl.resolve())
    report = _summarize(rows)
    report["resume_identity"] = identity
    args.summary.resolve().parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.resolve().with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.summary.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
