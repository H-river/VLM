#!/usr/bin/env python3
"""Strict whole-string generation for the independent plan-selector adapter."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from qwen_vl_supervisor_v1 import generate as qwen_generation
from qwen_vl_supervisor_v1.model_snapshot import pretrained_revision_kwargs, resolve_model_source_identity

from .selector_contract import SelectorDecision, parse_selector_output
from .training import REPOSITORY_ROOT, validate_candidate_config


ADAPTER_NAME = "qwen_reasoning_plan_selector"


@dataclass(frozen=True)
class SelectorGeneration:
    raw_text: str
    parsed: SelectorDecision | None
    valid_json: bool
    error: str | None
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {"raw_text": self.raw_text, "parsed": None if self.parsed is None else self.parsed.to_dict(), "valid_json": self.valid_json, "error": self.error, "latency_seconds": self.latency_seconds}


def parse_generated_text(text: str, *, latency_seconds: float = 0.0) -> SelectorGeneration:
    try:
        parsed = parse_selector_output(text)
    except ValueError as exc:
        return SelectorGeneration(text, None, False, str(exc), float(latency_seconds))
    return SelectorGeneration(text, parsed, True, None, float(latency_seconds))


class QwenSelectorBackend:
    def __init__(self, *, processor: Any, model: Any, deps: Mapping[str, Any], device_index: int, adapter_hashes: Mapping[str, str]) -> None:
        self.processor, self.model, self.deps = processor, model, dict(deps)
        self.device_index, self.adapter_hashes = int(device_index), dict(adapter_hashes)

    @classmethod
    def load(cls, *, config: Mapping[str, Any], adapter_path: Path, local_rank: int | None = None) -> "QwenSelectorBackend":
        validate_candidate_config(config)
        adapter_path = adapter_path.resolve()
        if not adapter_path.is_dir():
            raise FileNotFoundError(adapter_path)
        adapter_hashes = qwen_generation.adapter_fingerprint(adapter_path)
        cfg = qwen_generation.generation_config(config)
        deps = qwen_generation.require_runtime()
        device_index, _ = qwen_generation.local_device(deps["torch"], local_rank)
        identity = resolve_model_source_identity(model_id=cfg.model_id, source_id=cfg.source_id, revision=cfg.revision, processor_revision=cfg.processor_revision, repository_root=REPOSITORY_ROOT, expected_local_tree_sha256=cfg.expected_local_snapshot_tree_sha256)
        processor_kwargs = {"trust_remote_code": cfg.trust_remote_code, "local_files_only": cfg.local_files_only, "min_pixels": cfg.min_pixels, "max_pixels": cfg.max_pixels}
        processor_kwargs.update(pretrained_revision_kwargs(identity, cfg.processor_revision))
        processor = deps["AutoProcessor"].from_pretrained(identity["model_id"], **processor_kwargs)
        observed = qwen_generation.observed_pixel_budget(processor)
        if observed != {"min_pixels": cfg.min_pixels, "max_pixels": cfg.max_pixels}:
            raise ValueError("processor pixel budget mismatch")
        torch = deps["torch"]
        quantization = deps["BitsAndBytesConfig"](load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=cfg.double_quant)
        model_kwargs = {"trust_remote_code": cfg.trust_remote_code, "local_files_only": cfg.local_files_only, "quantization_config": quantization, "torch_dtype": torch.bfloat16, "attn_implementation": "sdpa", "device_map": {"": device_index}}
        model_kwargs.update(pretrained_revision_kwargs(identity, cfg.revision))
        base_model = deps["Qwen2_5_VLForConditionalGeneration"].from_pretrained(identity["model_id"], **model_kwargs)
        model = deps["PeftModel"].from_pretrained(base_model, str(adapter_path), adapter_name=ADAPTER_NAME, is_trainable=False, local_files_only=True)
        model.set_adapter(ADAPTER_NAME); model.eval(); model.config.use_cache = True
        return cls(processor=processor, model=model, deps=deps, device_index=device_index, adapter_hashes=adapter_hashes)

    def generate(self, prompt: list[dict[str, Any]], image_path: Path, *, seed: int, max_new_tokens: int = 192) -> SelectorGeneration:
        if max_new_tokens != 192:
            raise ValueError("max_new_tokens differs from preregistration")
        started = time.perf_counter()
        torch = self.deps["torch"]
        self.deps["set_seed"](int(seed), deterministic=True)
        rendered = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        image = qwen_generation.load_rgb_image(self.deps["Image"], image_path.resolve())
        inputs = self.processor(text=[rendered], images=[image], return_tensors="pt")
        device = torch.device("cuda", self.device_index)
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True)
        decoded, _ = qwen_generation.decode_generated_continuation(self.processor, generated, prompt_tokens)
        return parse_generated_text(decoded, latency_seconds=time.perf_counter() - started)


__all__ = ["ADAPTER_NAME", "QwenSelectorBackend", "SelectorGeneration", "parse_generated_text"]
