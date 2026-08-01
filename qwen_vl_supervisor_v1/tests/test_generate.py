import json
from pathlib import Path

import pytest
from PIL import Image

from qwen_vl_supervisor_v1.generate import (
    EXPECTED_MAX_PIXELS,
    EXPECTED_MIN_PIXELS,
    EXPECTED_REVISION,
    completed_predictions,
    decode_generated_continuation,
    generate_one,
    generation_config,
    parse_args,
    prediction_context,
    sha256_path,
    validate_dev_rows,
)


def _config():
    return {
        "model": {
            "id": "/models/Qwen2.5-VL-3B-Instruct",
            "source_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "revision": EXPECTED_REVISION,
            "processor_revision": EXPECTED_REVISION,
            "architecture": "Qwen2_5_VLForConditionalGeneration",
            "trust_remote_code": False,
            "local_files_only": True,
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "processor": {
                "min_pixels": EXPECTED_MIN_PIXELS,
                "max_pixels": EXPECTED_MAX_PIXELS,
            },
            "quantization": {
                "load_in_4bit": True,
                "quant_type": "nf4",
                "compute_dtype": "bfloat16",
                "double_quant": True,
            },
        }
    }


def _row(image_name="beam.png", split="dev", image_sha256=None):
    return {
        "example_id": "sample-1",
        "split": split,
        "images": [image_name],
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": "contract"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "structured state"},
                ],
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"diagnosis":"nominal"}'}],
            }
        ],
        "metadata": {"image_sha256": image_sha256},
    }


def test_cli_requires_explicit_data_and_adapter():
    with pytest.raises(SystemExit):
        parse_args(["--output", "predictions.jsonl"])


def test_config_and_rows_are_frozen_to_smoke_dev_contract(tmp_path: Path):
    config = generation_config(_config())
    assert (config.min_pixels, config.max_pixels) == (56 * 56, 224 * 224)

    Image.new("L", (128, 128), color=1).save(tmp_path / "beam.png")
    digest = sha256_path(tmp_path / "beam.png")
    assert validate_dev_rows(
        [_row(image_sha256=digest)], image_root=tmp_path
    ) == [tmp_path / "beam.png"]
    with pytest.raises(ValueError, match="only explicit split='dev'"):
        validate_dev_rows(
            [_row(split="frozen_iid", image_sha256=digest)], image_root=tmp_path
        )


def test_prediction_context_pins_local_snapshot_tree() -> None:
    context = prediction_context(
        data_sha256="data",
        adapter_hashes={"adapter_model.safetensors": "adapter"},
        config_sha256="config",
        model_source_identity={
            "kind": "local_snapshot",
            "expected_tree_sha256": None,
            "expected_tree_sha256_verified": None,
            "fingerprint": {
                "schema": "qwen_vl_local_snapshot_tree_v1",
                "tree_sha256": "a" * 64,
                "file_count": 14,
                "total_bytes": 7_520_919_614,
            },
        },
    )
    assert context["model_source_identity"]["tree_sha256"] == "a" * 64


def test_image_bytes_are_rehashed_immediately_before_generation(tmp_path: Path):
    image_path = tmp_path / "beam.png"
    Image.new("L", (128, 128), color=1).save(image_path)
    row = _row(image_sha256=sha256_path(image_path))
    assert validate_dev_rows([row], image_root=tmp_path) == [image_path]

    Image.new("L", (128, 128), color=2).save(image_path)
    with pytest.raises(ValueError, match="current image SHA-256 mismatch"):
        generate_one(
            row=row,
            image_path=image_path,
            processor=None,
            model=None,
            deps={},
            device_index=0,
            seed=1,
            max_new_tokens=64,
            context={},
        )


class _FakeGenerated:
    def __init__(self):
        self.slice = None

    def __getitem__(self, value):
        self.slice = value
        return "continuation-token-marker"


class _FakeProcessor:
    def __init__(self, text):
        self.text = text
        self.arguments = None

    def batch_decode(self, ids, **kwargs):
        self.arguments = (ids, kwargs)
        return [self.text]


def test_decode_slices_only_prompt_tokens_and_preserves_whole_invalid_string():
    raw = 'prefix {"diagnosis":"nominal"} trailing\n'
    generated = _FakeGenerated()
    processor = _FakeProcessor(raw)
    prediction, token_marker = decode_generated_continuation(processor, generated, 17)

    assert generated.slice == (slice(None), slice(17, None))
    assert token_marker == "continuation-token-marker"
    assert prediction == raw
    assert processor.arguments == (
        "continuation-token-marker",
        {"skip_special_tokens": True, "clean_up_tokenization_spaces": False},
    )


def test_resume_rejects_context_change_and_accepts_raw_invalid_prediction(tmp_path: Path):
    output = tmp_path / "predictions.jsonl"
    context = {
        "data_sha256": "data",
        "adapter_hashes": {"adapter_model.safetensors": "adapter"},
        "config_sha256": "config",
    }
    record = {
        "sample_id": "sample-1",
        "prediction": "definitely not JSON",
        "seed": 7,
        "latency_seconds": 0.1,
        "telemetry": {"batch_size": 1},
        "run_context": context,
    }
    output.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert completed_predictions(
        output, expected_ids={"sample-1"}, seed=7, context=context
    ) == [record]
    with pytest.raises(ValueError, match="context changed"):
        completed_predictions(
            output,
            expected_ids={"sample-1"},
            seed=7,
            context={**context, "data_sha256": "different"},
        )
