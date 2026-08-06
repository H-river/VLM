"""Independent Qwen adapter loading and strict candidate meta generation."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_vl_supervisor_v1 import generate as qwen_generation
from qwen_vl_supervisor_v1.model_snapshot import (
    pretrained_revision_kwargs,
    resolve_model_source_identity,
)

from .contracts import MetaContractError, MetaControllerDecision, parse_meta_output
from .training import (
    PACKAGE_ROOT,
    REPOSITORY_ROOT,
    _guard_candidate_path,
    validate_candidate_config,
    validate_prebuilt_row,
)


ADAPTER_NAME = "qwen_h1_meta_v0"


@dataclass(frozen=True)
class MetaGeneration:
    raw_text: str
    parsed: MetaControllerDecision | None
    valid_json: bool
    error_code: str | None
    error_message: str | None
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "parsed": None if self.parsed is None else self.parsed.to_dict(),
            "valid_json": self.valid_json,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "latency_seconds": self.latency_seconds,
        }


def parse_generated_text(text: str, *, latency_seconds: float = 0.0) -> MetaGeneration:
    try:
        parsed = parse_meta_output(text)
    except MetaContractError as exc:
        return MetaGeneration(
            raw_text=text,
            parsed=None,
            valid_json=False,
            error_code=exc.code,
            error_message=str(exc),
            latency_seconds=float(latency_seconds),
        )
    return MetaGeneration(
        raw_text=text,
        parsed=parsed,
        valid_json=True,
        error_code=None,
        error_message=None,
        latency_seconds=float(latency_seconds),
    )


class IndependentMetaAdapter:
    """Small testable shell around a raw-text generation backend.

    The backend receives ``(prompt, image_path, seed, max_new_tokens)`` and
    returns the unmodified decoded continuation.  This shell never extracts or
    repairs a JSON substring.
    """

    def __init__(
        self,
        backend: Callable[[list[dict[str, Any]], Path, int, int], str],
        *,
        adapter_name: str = ADAPTER_NAME,
    ) -> None:
        if adapter_name != ADAPTER_NAME:
            raise ValueError(f"independent adapter name must remain {ADAPTER_NAME!r}")
        self.backend = backend
        self.adapter_name = adapter_name

    def generate_row(
        self,
        row: Mapping[str, Any],
        *,
        image_root: Path = REPOSITORY_ROOT,
        seed: int,
        max_new_tokens: int = 256,
    ) -> MetaGeneration:
        if max_new_tokens != 256:
            raise ValueError("candidate generation max_new_tokens is preregistered to 256")
        image_path = validate_prebuilt_row(
            row,
            allowed_splits={"dev"},
            image_root=image_root,
            verify_image=True,
        )
        started = time.perf_counter()
        raw = self.backend(list(row["prompt"]), image_path, int(seed), max_new_tokens)
        elapsed = time.perf_counter() - started
        if not isinstance(raw, str):
            raise TypeError("generation backend must return the raw decoded string")
        return parse_generated_text(raw, latency_seconds=elapsed)


class QwenMetaBackend:
    """Lazily loaded Qwen2.5-VL base plus only the independent meta adapter."""

    def __init__(
        self,
        *,
        processor: Any,
        model: Any,
        deps: Mapping[str, Any],
        device_index: int,
        adapter_hashes: Mapping[str, str],
    ) -> None:
        self.processor = processor
        self.model = model
        self.deps = dict(deps)
        self.device_index = int(device_index)
        self.adapter_hashes = dict(adapter_hashes)

    @classmethod
    def load(
        cls,
        *,
        config: Mapping[str, Any],
        adapter_path: Path,
        local_rank: int | None = None,
    ) -> "QwenMetaBackend":
        """Load no supervisor adapter and perform no data/model generation."""

        validate_candidate_config(config)
        adapter_path = _guard_candidate_path(adapter_path, role="meta_adapter")
        adapter_hashes = qwen_generation.adapter_fingerprint(adapter_path)
        cfg = qwen_generation.generation_config(config)
        deps = qwen_generation.require_runtime()
        device_index, _ = qwen_generation.local_device(deps["torch"], local_rank)
        identity = resolve_model_source_identity(
            model_id=cfg.model_id,
            source_id=cfg.source_id,
            revision=cfg.revision,
            processor_revision=cfg.processor_revision,
            repository_root=REPOSITORY_ROOT,
            expected_local_tree_sha256=cfg.expected_local_snapshot_tree_sha256,
        )
        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": cfg.trust_remote_code,
            "local_files_only": cfg.local_files_only,
            "min_pixels": cfg.min_pixels,
            "max_pixels": cfg.max_pixels,
        }
        processor_kwargs.update(
            pretrained_revision_kwargs(identity, cfg.processor_revision)
        )
        processor = deps["AutoProcessor"].from_pretrained(
            identity["model_id"], **processor_kwargs
        )
        observed = qwen_generation.observed_pixel_budget(processor)
        expected = {"min_pixels": cfg.min_pixels, "max_pixels": cfg.max_pixels}
        if observed != expected:
            raise ValueError(f"processor pixel budget mismatch: {observed} != {expected}")

        torch = deps["torch"]
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
        model_kwargs.update(pretrained_revision_kwargs(identity, cfg.revision))
        base_model = deps["Qwen2_5_VLForConditionalGeneration"].from_pretrained(
            identity["model_id"], **model_kwargs
        )
        model = deps["PeftModel"].from_pretrained(
            base_model,
            str(adapter_path),
            adapter_name=ADAPTER_NAME,
            is_trainable=False,
            local_files_only=True,
        )
        model.set_adapter(ADAPTER_NAME)
        model.eval()
        model.config.use_cache = True
        return cls(
            processor=processor,
            model=model,
            deps=deps,
            device_index=device_index,
            adapter_hashes=adapter_hashes,
        )

    def __call__(
        self,
        prompt: list[dict[str, Any]],
        image_path: Path,
        seed: int,
        max_new_tokens: int,
    ) -> str:
        image_path = _guard_candidate_path(image_path, role="generation_image")
        torch = self.deps["torch"]
        self.deps["set_seed"](int(seed), deterministic=True)
        rendered = self.processor.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        image = qwen_generation.load_rgb_image(self.deps["Image"], image_path)
        inputs = self.processor(text=[rendered], images=[image], return_tensors="pt")
        device = torch.device("cuda", self.device_index)
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        decoded, _ = qwen_generation.decode_generated_continuation(
            self.processor, generated, prompt_tokens
        )
        return decoded


def load_independent_adapter(
    *,
    config: Mapping[str, Any],
    adapter_path: Path,
    local_rank: int | None = None,
) -> IndependentMetaAdapter:
    backend = QwenMetaBackend.load(
        config=config,
        adapter_path=adapter_path,
        local_rank=local_rank,
    )
    return IndependentMetaAdapter(backend)


__all__ = [
    "ADAPTER_NAME",
    "IndependentMetaAdapter",
    "MetaGeneration",
    "QwenMetaBackend",
    "load_independent_adapter",
    "parse_generated_text",
]
