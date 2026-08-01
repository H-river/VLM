#!/usr/bin/env python3
"""Deterministic, resumable development generation for supervisor QLoRA adapters.

This entry point has no model imports at module import time.  It requires the
caller to name both the development ``prebuilt_chat`` JSONL and adapter; there
is deliberately no default data path and every row must declare split ``dev``.
The stored prediction is the complete decoded continuation after slicing only
the chat-template prompt tokens.  In particular, this module never searches
for, extracts, repairs, or normalises a JSON substring.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# PyTorch deterministic CUDA matmul requires this to be present before the
# runtime imports torch or initializes CUDA.  This is a reproducibility setting,
# not a decoding repair or model change.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from qwen_vl_supervisor_v1.model_snapshot import (  # noqa: E402
    compact_source_identity,
    pretrained_revision_kwargs,
    resolve_model_source_identity,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs/training_smoke.yaml"
EXPECTED_ARCHITECTURE = "Qwen2_5_VLForConditionalGeneration"
EXPECTED_SOURCE_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
EXPECTED_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
EXPECTED_MIN_PIXELS = 56 * 56
EXPECTED_MAX_PIXELS = 224 * 224
PROTECTED_PATH_FRAGMENTS = ("frozen", "heldout", "held_out", "severity_ood")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Explicit dev prebuilt_chat JSONL.")
    parser.add_argument("--adapter", type=Path, required=True, help="Saved PEFT adapter directory.")
    parser.add_argument("--output", type=Path, required=True, help="Resumable prediction JSONL.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--expected-local-snapshot-tree-sha256",
        help=(
            "Override model.expected_local_snapshot_tree_sha256 without changing the "
            "byte-pinned source config."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Generation report JSON (default: <output>.report.json).",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Root for relative image paths in the supplied JSONL.",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Continuation budget; values below 64 are rejected for the canonical JSON contract.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=None,
        help="CUDA-local rank; LOCAL_RANK is used when omitted.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_fingerprint(path: Path) -> dict[str, str]:
    """Hash the inference-relevant adapter files, excluding optimizer state."""

    candidates = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "generation_config.json",
    )
    hashes = {
        name: sha256_path(path / name)
        for name in candidates
        if (path / name).is_file()
    }
    if "adapter_config.json" not in hashes or not any(
        name in hashes for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError(
            f"adapter must contain adapter_config.json and adapter_model.safetensors/bin: {path}"
        )
    return hashes


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-specific dependency failure
        raise RuntimeError("PyYAML is required to read the smoke configuration") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"generation config must be a mapping: {path}")
    return value


@dataclass(frozen=True)
class GenerationConfig:
    model_id: str
    source_id: str
    revision: str
    processor_revision: str
    expected_local_snapshot_tree_sha256: str | None
    trust_remote_code: bool
    local_files_only: bool
    min_pixels: int
    max_pixels: int
    double_quant: bool


def generation_config(raw: Mapping[str, Any]) -> GenerationConfig:
    model = raw.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("config.model must be a mapping")
    processor = model.get("processor")
    quantization = model.get("quantization")
    if not isinstance(processor, Mapping) or not isinstance(quantization, Mapping):
        raise ValueError("config.model.processor and quantization must be mappings")
    if model.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError(f"generation requires {EXPECTED_ARCHITECTURE}")
    source_id = str(model.get("source_id", ""))
    if source_id != EXPECTED_SOURCE_ID:
        raise ValueError(f"generation requires source_id {EXPECTED_SOURCE_ID!r}")
    revision = str(model.get("revision", ""))
    processor_revision = str(model.get("processor_revision", ""))
    if revision != EXPECTED_REVISION or processor_revision != EXPECTED_REVISION:
        raise ValueError("model and processor revisions differ from the frozen smoke revision")
    min_pixels = int(processor.get("min_pixels", 0))
    max_pixels = int(processor.get("max_pixels", 0))
    if (min_pixels, max_pixels) != (EXPECTED_MIN_PIXELS, EXPECTED_MAX_PIXELS):
        raise ValueError(
            "smoke generation pixel budget changed: "
            f"expected {(EXPECTED_MIN_PIXELS, EXPECTED_MAX_PIXELS)}, "
            f"got {(min_pixels, max_pixels)}"
        )
    if quantization.get("load_in_4bit") is not True:
        raise ValueError("generation requires 4-bit base-model loading")
    if quantization.get("quant_type") != "nf4":
        raise ValueError("generation requires NF4 quantization")
    if quantization.get("compute_dtype") != "bfloat16":
        raise ValueError("generation requires bfloat16 NF4 compute")
    if model.get("torch_dtype") != "bfloat16" or model.get("attn_implementation") != "sdpa":
        raise ValueError("generation requires torch_dtype=bfloat16 and SDPA attention")
    model_id = str(model.get("id", ""))
    if not model_id:
        raise ValueError("config.model.id must name the local or cached 3B checkpoint")
    expected_local_tree = model.get("expected_local_snapshot_tree_sha256")
    if expected_local_tree is not None and not HEX_SHA256.fullmatch(str(expected_local_tree)):
        raise ValueError(
            "model.expected_local_snapshot_tree_sha256 must be a lowercase SHA-256"
        )
    return GenerationConfig(
        model_id=model_id,
        source_id=source_id,
        revision=revision,
        processor_revision=processor_revision,
        expected_local_snapshot_tree_sha256=(
            str(expected_local_tree) if expected_local_tree is not None else None
        ),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        local_files_only=bool(model.get("local_files_only", True)),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        double_quant=bool(quantization.get("double_quant", True)),
    )


def read_jsonl(path: Path, *, kind: str) -> list[dict[str, Any]]:
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
                raise ValueError(f"{kind} {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{kind} {path}:{line_number}: expected one JSON object")
            rows.append(row)
    if not rows and kind == "data":
        raise ValueError(f"development data is empty: {path}")
    return rows


def _image_slot_count(messages: Any) -> int:
    if not isinstance(messages, list):
        raise ValueError("prompt must be a list of messages")
    count = 0
    for message in messages:
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), list):
            raise ValueError("each prompt message must have list-valued content")
        for content in message["content"]:
            if not isinstance(content, Mapping) or content.get("type") not in {"image", "text"}:
                raise ValueError("prompt content supports only image and text items")
            count += content.get("type") == "image"
    return count


def resolve_image(path_text: str, image_root: Path) -> Path:
    path = Path(path_text).expanduser()
    return path.resolve() if path.is_absolute() else (image_root / path).resolve()


def expected_image_sha256(row: Mapping[str, Any], *, sample_id: str) -> str:
    """Return the immutable image digest exported with one prebuilt-chat row."""

    metadata = row.get("metadata")
    expected = metadata.get("image_sha256") if isinstance(metadata, Mapping) else None
    if not isinstance(expected, str) or not HEX_SHA256.fullmatch(expected):
        raise ValueError(
            f"{sample_id}: metadata.image_sha256 must be a lowercase SHA-256"
        )
    return expected


def verify_current_image_sha256(row: Mapping[str, Any], image_path: Path) -> str:
    """Hash the actual current-image bytes and enforce the exported digest."""

    sample_id = row.get("example_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("image row example_id must be a non-empty string")
    expected = expected_image_sha256(row, sample_id=sample_id)
    actual = sha256_path(image_path)
    if actual != expected:
        raise ValueError(
            f"{sample_id}: current image SHA-256 mismatch for {image_path}: "
            f"expected {expected}, got {actual}"
        )
    return actual


def validate_dev_rows(
    rows: Sequence[Mapping[str, Any]], *, image_root: Path, verify_images: bool = True
) -> list[Path]:
    sample_ids: set[str] = set()
    image_paths: list[Path] = []
    for index, row in enumerate(rows):
        sample_id = row.get("example_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"data row {index}: example_id must be a non-empty string")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate development example_id: {sample_id}")
        sample_ids.add(sample_id)
        if row.get("split") != "dev":
            raise ValueError(f"{sample_id}: generation accepts only explicit split='dev'")
        if _image_slot_count(row.get("prompt")) != 1:
            raise ValueError(f"{sample_id}: prompt must contain exactly one current-image slot")
        roles = [message.get("role") for message in row["prompt"]]
        if roles != ["system", "user"]:
            raise ValueError(f"{sample_id}: generation prompt roles must be system,user")
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise ValueError(f"{sample_id}: exactly one image path is required")
        image_path = resolve_image(images[0], image_root)
        if not image_path.is_file():
            raise FileNotFoundError(f"{sample_id}: image does not exist: {image_path}")
        verify_current_image_sha256(row, image_path)
        if verify_images:
            from PIL import Image

            with Image.open(image_path) as image:
                image.verify()
        image_paths.append(image_path)
    return image_paths


def guard_dev_data_path(path: Path) -> None:
    lowered = path.as_posix().lower()
    found = [fragment for fragment in PROTECTED_PATH_FRAGMENTS if fragment in lowered]
    if found:
        raise ValueError(f"protected/frozen data path is forbidden for generation: {found}")


def decode_generated_continuation(
    processor: Any, generated_ids: Any, prompt_token_count: int
) -> tuple[str, Any]:
    """Decode exactly the tokens after the rendered prompt, without text cleanup."""

    if prompt_token_count <= 0:
        raise ValueError("prompt_token_count must be positive")
    continuation_ids = generated_ids[:, prompt_token_count:]
    decoded = processor.batch_decode(
        continuation_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(decoded[0], str):
        raise ValueError("batch-1 decode must return exactly one string")
    return decoded[0], continuation_ids


def prediction_context(
    *,
    data_sha256: str,
    adapter_hashes: Mapping[str, str],
    config_sha256: str,
    model_source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "data_sha256": data_sha256,
        "adapter_hashes": dict(adapter_hashes),
        "config_sha256": config_sha256,
        "model_revision": EXPECTED_REVISION,
        "model_source_identity": compact_source_identity(model_source_identity),
        "pixel_budget": {
            "min_pixels": EXPECTED_MIN_PIXELS,
            "max_pixels": EXPECTED_MAX_PIXELS,
        },
    }


def completed_predictions(
    path: Path,
    *,
    expected_ids: set[str],
    seed: int,
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = read_jsonl(path, kind="predictions")
    seen: set[str] = set()
    for index, record in enumerate(records):
        sample_id = record.get("sample_id")
        if sample_id not in expected_ids:
            raise ValueError(f"prediction row {index} is not in the supplied dev data: {sample_id!r}")
        if sample_id in seen:
            raise ValueError(f"duplicate resumable prediction: {sample_id}")
        seen.add(sample_id)
        if record.get("seed") != seed:
            raise ValueError(f"prediction {sample_id}: seed differs from requested resume seed")
        if not isinstance(record.get("prediction"), str):
            raise ValueError(f"prediction {sample_id}: prediction must remain a raw string")
        if record.get("run_context") != context:
            raise ValueError(f"prediction {sample_id}: data/adapter/config context changed")
    return records


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def require_runtime() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from PIL import Image
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
            set_seed,
        )
    except ImportError as exc:  # pragma: no cover - runtime environment specific
        raise RuntimeError(
            "generation requires torch, transformers, peft, bitsandbytes, Pillow, and PyYAML"
        ) from exc
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "Image": Image,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLForConditionalGeneration,
        "set_seed": set_seed,
    }


def local_device(torch: Any, local_rank_arg: int | None) -> tuple[int, dict[str, int]]:
    world = {
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(
            local_rank_arg
            if local_rank_arg is not None
            else os.environ.get("LOCAL_RANK", "-1")
        ),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
    if world["world_size"] != 1:
        raise ValueError(
            "this batch-1 resumable writer requires WORLD_SIZE=1; launch one process per output file"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit Qwen2.5-VL generation requires a CUDA device")
    device_index = world["local_rank"] if world["local_rank"] >= 0 else 0
    if device_index >= torch.cuda.device_count():
        raise ValueError(
            f"LOCAL_RANK device {device_index} exceeds {torch.cuda.device_count()} visible GPUs"
        )
    torch.cuda.set_device(device_index)
    return device_index, world


def load_rgb_image(image_cls: Any, path: Path) -> Any:
    with image_cls.open(path) as image:
        loaded = image.convert("RGB")
        loaded.load()
    return loaded


def observed_pixel_budget(processor: Any) -> dict[str, Any]:
    size = getattr(processor.image_processor, "size", None)
    return {
        "min_pixels": getattr(size, "shortest_edge", None),
        "max_pixels": getattr(size, "longest_edge", None),
    }


def load_model_stack(
    deps: Mapping[str, Any],
    cfg: GenerationConfig,
    adapter: Path,
    device_index: int,
    model_source_identity: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    torch = deps["torch"]
    processor_kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "local_files_only": cfg.local_files_only,
        "min_pixels": cfg.min_pixels,
        "max_pixels": cfg.max_pixels,
    }
    processor_kwargs.update(
        pretrained_revision_kwargs(model_source_identity, cfg.processor_revision)
    )
    processor = deps["AutoProcessor"].from_pretrained(
        model_source_identity["model_id"],
        **processor_kwargs,
    )
    observed = observed_pixel_budget(processor)
    expected = {"min_pixels": cfg.min_pixels, "max_pixels": cfg.max_pixels}
    if observed != expected:
        raise ValueError(f"processor pixel budget mismatch: {observed} != {expected}")
    quantization = deps["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=cfg.double_quant,
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "local_files_only": cfg.local_files_only,
        "quantization_config": quantization,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "device_map": {"": device_index},
    }
    model_kwargs.update(pretrained_revision_kwargs(model_source_identity, cfg.revision))
    base_model = deps["Qwen2_5_VLForConditionalGeneration"].from_pretrained(
        model_source_identity["model_id"],
        **model_kwargs,
    )
    model = deps["PeftModel"].from_pretrained(
        base_model,
        str(adapter),
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    model.config.use_cache = True
    return processor, model, observed


def generate_one(
    *,
    row: Mapping[str, Any],
    image_path: Path,
    processor: Any,
    model: Any,
    deps: Mapping[str, Any],
    device_index: int,
    seed: int,
    max_new_tokens: int,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    # Re-hash immediately before image loading.  This intentionally duplicates
    # the up-front validation so file drift during a long/resumed run is caught
    # before this sample reaches processor/model inference.
    verify_current_image_sha256(row, image_path)
    torch = deps["torch"]
    rendered = processor.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True,
    )
    image = load_rgb_image(deps["Image"], image_path)
    inputs = processor(text=[rendered], images=[image], return_tensors="pt")
    device = torch.device("cuda", device_index)
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    torch.cuda.reset_peak_memory_stats(device_index)
    torch.cuda.synchronize(device_index)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    torch.cuda.synchronize(device_index)
    latency = time.perf_counter() - started
    prediction, continuation_ids = decode_generated_continuation(
        processor, generated, prompt_tokens
    )
    output_tokens = int(continuation_ids.shape[-1])
    eos = model.generation_config.eos_token_id
    eos_ids = {int(value) for value in eos} if isinstance(eos, (list, tuple)) else {int(eos)}
    last_token = int(continuation_ids[0, -1].item()) if output_tokens else None
    return {
        "sample_id": row["example_id"],
        "prediction": prediction,
        "seed": seed,
        "latency_seconds": latency,
        "telemetry": {
            "batch_size": 1,
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "max_new_tokens": max_new_tokens,
            "finish_reason": "eos" if last_token in eos_ids else "length_or_other",
            "cuda_device_index": device_index,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
        },
        "run_context": dict(context),
    }


def report_path(args: argparse.Namespace) -> Path:
    if args.report is not None:
        return resolve_repo_path(args.report)
    output = resolve_repo_path(args.output)
    return output.with_suffix(output.suffix + ".report.json")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_new_tokens < 64:
        raise ValueError("--max-new-tokens must be at least 64 for the canonical supervisor JSON")
    data_path = resolve_repo_path(args.data)
    adapter_path = resolve_repo_path(args.adapter)
    output_path = resolve_repo_path(args.output)
    config_path = resolve_repo_path(args.config)
    image_root = resolve_repo_path(args.image_root)
    guard_dev_data_path(data_path)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter_path}")
    if output_path.resolve() == data_path.resolve():
        raise ValueError("prediction output must not overwrite the supplied development data")
    raw_config = load_yaml(config_path)
    if args.expected_local_snapshot_tree_sha256 is not None:
        raw_config["model"]["expected_local_snapshot_tree_sha256"] = (
            args.expected_local_snapshot_tree_sha256
        )
    cfg = generation_config(raw_config)
    model_source_identity = resolve_model_source_identity(
        model_id=cfg.model_id,
        source_id=cfg.source_id,
        revision=cfg.revision,
        processor_revision=cfg.processor_revision,
        repository_root=REPOSITORY_ROOT,
        expected_local_tree_sha256=cfg.expected_local_snapshot_tree_sha256,
    )
    adapter_hashes = adapter_fingerprint(adapter_path)
    rows = read_jsonl(data_path, kind="data")
    image_paths = validate_dev_rows(rows, image_root=image_root, verify_images=True)
    data_hash = sha256_path(data_path)
    config_hash = sha256_path(config_path)
    context = prediction_context(
        data_sha256=data_hash,
        adapter_hashes=adapter_hashes,
        config_sha256=config_hash,
        model_source_identity=model_source_identity,
    )
    existing = completed_predictions(
        output_path,
        expected_ids={str(row["example_id"]) for row in rows},
        seed=args.seed,
        context=context,
    )
    completed_ids = {str(row["sample_id"]) for row in existing}
    pending = [
        (row, path)
        for row, path in zip(rows, image_paths, strict=True)
        if row["example_id"] not in completed_ids
    ]
    started_at = utc_now()
    report: dict[str, Any] = {
        "schema_version": "qwen_vl_supervisor_generation_v1.0.0",
        "status": "completed" if not pending else "starting",
        "started_at_utc": started_at,
        "completed_at_utc": started_at if not pending else None,
        "data": {
            "path": str(data_path),
            "sha256": data_hash,
            "split": "dev",
            "records": len(rows),
            "frozen_or_protected_predictions_opened": False,
        },
        "image_integrity": {
            "policy": "metadata_image_sha256_matches_actual_bytes_before_run_and_each_inference",
            "verified_records_before_model_load": len(rows),
        },
        "adapter": {"path": str(adapter_path), "hashes": adapter_hashes},
        "config": {
            "path": str(config_path),
            "sha256": config_hash,
            "expected_local_snapshot_tree_sha256_override": (
                args.expected_local_snapshot_tree_sha256
            ),
        },
        "model": {
            "id": cfg.model_id,
            "source_id": cfg.source_id,
            "revision": cfg.revision,
            "processor_revision": cfg.processor_revision,
            "source_identity": model_source_identity,
            "architecture": EXPECTED_ARCHITECTURE,
            "quantization": "NF4_4bit",
            "compute_dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "processor_pixel_budget": {
                "min_pixels": cfg.min_pixels,
                "max_pixels": cfg.max_pixels,
            },
        },
        "decoding": {
            "batch_size": 1,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "text_policy": "decode_all_tokens_after_exact_chat_template_prompt_length",
            "json_extraction_or_repair": False,
        },
        "seed": args.seed,
        "output": str(output_path),
        "resume": {
            "records_found": len(existing),
            "records_pending_at_start": len(pending),
        },
        "progress": {"completed_records": len(existing), "expected_records": len(rows)},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(
                ("torch", "transformers", "peft", "bitsandbytes", "Pillow", "PyYAML")
            ),
        },
    }
    destination_report = report_path(args)
    atomic_json(destination_report, report)
    if not pending:
        return report

    deps = require_runtime()
    device_index, world = local_device(deps["torch"], args.local_rank)
    report["distributed"] = world
    report["cuda"] = {
        "device_index": device_index,
        "name": deps["torch"].cuda.get_device_name(device_index),
    }
    deps["set_seed"](args.seed, deterministic=True)
    try:
        processor, model, observed = load_model_stack(
            deps, cfg, adapter_path, device_index, model_source_identity
        )
        report["model"]["processor_pixel_budget_observed"] = observed
        report["status"] = "running"
        atomic_json(destination_report, report)
        generated_now: list[dict[str, Any]] = []
        for row, image_path in pending:
            result = generate_one(
                row=row,
                image_path=image_path,
                processor=processor,
                model=model,
                deps=deps,
                device_index=device_index,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                context=context,
            )
            append_jsonl(output_path, result)
            generated_now.append(result)
            report["progress"]["completed_records"] = len(existing) + len(generated_now)
            atomic_json(destination_report, report)
        latencies = [float(row["latency_seconds"]) for row in generated_now]
        all_records = completed_predictions(
            output_path,
            expected_ids={str(row["example_id"]) for row in rows},
            seed=args.seed,
            context=context,
        )
        if len(all_records) != len(rows):
            raise RuntimeError("generation ended without one prediction per supplied dev row")
        report["status"] = "completed"
        report["completed_at_utc"] = utc_now()
        report["progress"] = {
            "completed_records": len(all_records),
            "expected_records": len(rows),
            "generated_this_invocation": len(generated_now),
        }
        report["telemetry"] = {
            "generation_latency_seconds_total_this_invocation": sum(latencies),
            "generation_latency_seconds_mean_this_invocation": (
                sum(latencies) / len(latencies) if latencies else None
            ),
            "gpu_peak_allocated_bytes": int(
                deps["torch"].cuda.max_memory_allocated(device_index)
            ),
            "gpu_peak_reserved_bytes": int(
                deps["torch"].cuda.max_memory_reserved(device_index)
            ),
        }
    except Exception as exc:
        report["status"] = "failed"
        report["failed_at_utc"] = utc_now()
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        atomic_json(destination_report, report)
        raise
    atomic_json(destination_report, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
