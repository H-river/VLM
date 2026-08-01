import json

import pytest

from active_diagnosis_v13.infer_qwen_gain import (
    _extract_json,
    bind_resume_manifest,
    inference_identity,
)
from active_diagnosis_v13.audit_qwen_environment import weight_files
from optics_sft.scripts.train_qwen25vl_qlora import (
    prebuilt_chat_row_to_sft_example,
    prebuilt_smoke_rows,
)


def test_extract_qwen_gain_json_from_fenced_noise() -> None:
    parsed, error = _extract_json(
        'answer follows: {"estimated_gain": 0.75, "confidence": 0.8}'
    )
    assert error is None
    assert parsed == {"estimated_gain": 0.75, "confidence": 0.8}


def test_invalid_qwen_output_is_rejected() -> None:
    parsed, error = _extract_json("gain is probably high")
    assert parsed is None
    assert error


def test_zero_image_prebuilt_chat_is_supported(tmp_path) -> None:
    row = {
        "example_id": "case__g0.75",
        "images": [],
        "prompt": [
            {"role": "user", "content": [{"type": "text", "text": "estimate"}]}
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"estimated_gain": 0.75}'}],
            }
        ],
    }
    example = prebuilt_chat_row_to_sft_example(row, tmp_path)
    assert example["images"] == []
    assert example["prompt"] == row["prompt"]
    assert prebuilt_smoke_rows([row], 1) == [row]


def test_qwen_preflight_excludes_weight_metadata_sidecars(tmp_path) -> None:
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"weights")
    metadata = tmp_path / "model-00001-of-00001.safetensors.metadata"
    metadata.write_text("metadata")
    assert weight_files([shard, metadata]) == [shard]


def test_qwen_inference_resume_is_bound_to_adapter_and_input(tmp_path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text("{}")
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    source = tmp_path / "validation.jsonl"
    source.write_text(json.dumps({"example_id": "case"}) + "\n")
    output = tmp_path / "predictions.jsonl"
    identity = inference_identity(base, adapter, source)
    manifest = bind_resume_manifest(output, identity)
    assert json.loads(manifest.read_text()) == identity
    assert bind_resume_manifest(output, identity) == manifest

    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    changed = inference_identity(base, adapter, source)
    with pytest.raises(ValueError, match="resume identity differs"):
        bind_resume_manifest(output, changed)


def test_qwen_inference_refuses_unbound_existing_rows(tmp_path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text('{"example_id":"stale"}\n')
    with pytest.raises(ValueError, match="no resume identity"):
        bind_resume_manifest(output, {"version": "test"})
