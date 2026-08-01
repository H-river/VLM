import json
from pathlib import Path

import pytest

from qwen_vl_supervisor_v1.train_qlora import (
    REQUIRED_TARGET_MODULES,
    SAFE_OPTIMIZER_JSON,
    SAFE_OPTIMIZER_TENSORS,
    SAFE_RNG_JSON,
    SAFE_RNG_TENSORS,
    audit_lora_architecture,
    checkpoint_step,
    exact_nested_state_comparison,
    load_safe_state_bundle,
    load_yaml,
    prepare_rows,
    save_safe_state_bundle,
    validate_config,
    validate_prebuilt_row,
)


MODULE_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_and_export_validate_without_model_load() -> None:
    config = load_yaml(MODULE_ROOT / "configs/training_smoke.yaml")
    validate_config(config)
    rows = prepare_rows(config, verify_images=True)
    assert len(rows.train_rows) == 36
    assert len(rows.dev_rows) == 12
    assert {row["split"] for row in rows.train_rows} == {"train"}
    assert {row["split"] for row in rows.dev_rows} == {"dev"}


def test_server_config_freezes_three_training_seeds() -> None:
    config = load_yaml(MODULE_ROOT / "configs/training_server.yaml")
    validate_config(config)
    assert config["training_seeds"] == [2026080101, 2026080102, 2026080103]
    assert config["model"]["expected_local_snapshot_tree_sha256"] == (
        "2e1bd29589b91134a667572080bec76a5fb1446c49acddfa7f13049314bf3175"
    )


def test_frozen_row_is_rejected_before_image_open() -> None:
    row = json.loads((MODULE_ROOT / "sft/sft_smoke_train.jsonl").read_text().splitlines()[0])
    row["split"] = "frozen_iid"
    with pytest.raises(ValueError, match="protected/frozen split"):
        validate_prebuilt_row(
            row,
            allowed_splits={"train"},
            image_root=MODULE_ROOT.parent,
            verify_image=False,
        )


def test_architecture_audit_requires_exact_vision_attention_projection() -> None:
    names = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.self_attn.v_proj",
        "model.language_model.layers.0.self_attn.o_proj",
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.mlp.up_proj",
        "model.language_model.layers.0.mlp.down_proj",
        "model.visual.blocks.0.mlp.gate_proj",
        "model.visual.blocks.0.mlp.up_proj",
        "model.visual.blocks.0.mlp.down_proj",
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.attn.proj",
        # This module must not be selected by the exact `attn.proj` target.
        "model.visual.patch_embed.proj",
    ]

    class FakeModel:
        config = type("Config", (), {"model_type": "qwen2_5_vl"})()

        def named_modules(self):
            return [(name, object()) for name in names]

    audit = audit_lora_architecture(FakeModel(), REQUIRED_TARGET_MODULES)
    assert audit["target_counts"]["attn.proj"] == 1
    assert audit["target_examples"]["attn.proj"] == ["model.visual.blocks.0.attn.proj"]


def test_resume_checkpoint_requires_full_trainer_state(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safe_torch = pytest.importorskip("safetensors.torch")
    checkpoint = tmp_path / "checkpoint-20"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step":20}\n')
    # Legacy pickle files are intentionally insufficient and never loaded.
    (checkpoint / "optimizer.pt").touch()
    (checkpoint / "scheduler.pt").touch()
    (checkpoint / "rng_state.pth").touch()
    with pytest.raises(FileNotFoundError, match="legacy pickle checkpoints"):
        checkpoint_step(checkpoint)
    save_safe_state_bundle(
        directory=checkpoint,
        json_name=SAFE_OPTIMIZER_JSON,
        tensor_name=SAFE_OPTIMIZER_TENSORS,
        payload={"global_step": 20, "world_size": 1},
        torch_module=torch,
        save_file=safe_torch.save_file,
    )
    save_safe_state_bundle(
        directory=checkpoint,
        json_name=SAFE_RNG_JSON,
        tensor_name=SAFE_RNG_TENSORS,
        payload={"global_step": 20, "world_size": 1, "process_index": 0},
        torch_module=torch,
        save_file=safe_torch.save_file,
    )
    assert checkpoint_step(checkpoint) == 20


def test_distributed_checkpoint_requires_exact_contiguous_rng_ranks(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safe_torch = pytest.importorskip("safetensors.torch")
    checkpoint = tmp_path / "checkpoint-20"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step":20}\n')
    save_safe_state_bundle(
        directory=checkpoint,
        json_name=SAFE_OPTIMIZER_JSON,
        tensor_name=SAFE_OPTIMIZER_TENSORS,
        payload={"global_step": 20, "world_size": 4},
        torch_module=torch,
        save_file=safe_torch.save_file,
    )

    def save_rank(rank: int) -> None:
        save_safe_state_bundle(
            directory=checkpoint,
            json_name=f"rng_state.rank{rank:05d}.safe.json",
            tensor_name=f"rng_state.rank{rank:05d}.safe.safetensors",
            payload={"global_step": 20, "world_size": 4, "process_index": rank},
            torch_module=torch,
            save_file=safe_torch.save_file,
        )

    save_rank(0)
    with pytest.raises(FileNotFoundError, match="exact contiguous ranked RNG sidecars"):
        checkpoint_step(checkpoint)
    for rank in range(1, 4):
        save_rank(rank)
    assert checkpoint_step(checkpoint) == 20
    save_rank(4)
    with pytest.raises(FileNotFoundError, match="no unranked/extra sidecars"):
        checkpoint_step(checkpoint)


def test_safe_state_roundtrip_uses_json_and_safetensors_only(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safe_torch = pytest.importorskip("safetensors.torch")
    payload = {
        "global_step": 20,
        "optimizer": {
            "state": {0: {"step": torch.tensor(20), "exp_avg": torch.tensor([0.25, -0.5])}},
            "param_groups": [{"params": [0], "lr": 0.0002, "betas": (0.9, 0.999)}],
        },
        "scheduler": {"last_epoch": 20, "_last_lr": [0.0001]},
    }
    save_safe_state_bundle(
        directory=tmp_path,
        json_name=SAFE_OPTIMIZER_JSON,
        tensor_name=SAFE_OPTIMIZER_TENSORS,
        payload=payload,
        torch_module=torch,
        save_file=safe_torch.save_file,
    )
    restored = load_safe_state_bundle(
        directory=tmp_path,
        json_name=SAFE_OPTIMIZER_JSON,
        tensor_name=SAFE_OPTIMIZER_TENSORS,
        load_file=safe_torch.load_file,
    )
    assert restored["global_step"] == 20
    assert restored["optimizer"]["param_groups"][0]["betas"] == (0.9, 0.999)
    assert torch.equal(
        restored["optimizer"]["state"][0]["exp_avg"],
        payload["optimizer"]["state"][0]["exp_avg"],
    )
    assert not list(tmp_path.glob("*.pt"))
    assert not list(tmp_path.glob("*.pth"))


def test_exact_nested_state_comparison_rejects_value_and_dtype_drift() -> None:
    torch = pytest.importorskip("torch")
    expected = {"state": [torch.tensor([1.0, 2.0], dtype=torch.float32)], "step": 2}
    equal = exact_nested_state_comparison(expected, expected, torch_module=torch)
    assert equal["exact_equal"] is True
    value_drift = exact_nested_state_comparison(
        expected,
        {"state": [torch.tensor([1.0, 2.5], dtype=torch.float32)], "step": 2},
        torch_module=torch,
    )
    assert value_drift["exact_equal"] is False
    dtype_drift = exact_nested_state_comparison(
        expected,
        {"state": [torch.tensor([1.0, 2.0], dtype=torch.float64)], "step": 2},
        torch_module=torch,
    )
    assert dtype_drift["exact_equal"] is False
    numpy = pytest.importorskip("numpy")
    arrays = {"rng": numpy.array([1, 2, 3], dtype=numpy.uint32)}
    assert exact_nested_state_comparison(arrays, arrays, torch_module=torch)["exact_equal"] is True
