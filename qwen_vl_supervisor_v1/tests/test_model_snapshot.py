import json
import shutil
from pathlib import Path

import pytest

from qwen_vl_supervisor_v1.model_snapshot import (
    compact_source_identity,
    fingerprint_snapshot_tree,
    pretrained_revision_kwargs,
    resolve_model_source_identity,
)


REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def _fake_snapshot(root: Path) -> Path:
    root.mkdir(parents=True)
    for name in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (root / name).write_text("{}\n", encoding="utf-8")
    (root / "weights.safetensors").write_bytes(b"fake-weight-bytes")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layer.weight": "weights.safetensors"}}) + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("model card\n", encoding="utf-8")
    (root / ".cache").mkdir()
    (root / ".cache" / "volatile.json").write_text("first\n", encoding="utf-8")
    return root


def test_snapshot_fingerprint_is_path_independent_and_excludes_cache(tmp_path: Path) -> None:
    first = _fake_snapshot(tmp_path / "first")
    second = tmp_path / "second"
    shutil.copytree(first, second)
    (second / ".cache" / "volatile.json").write_text("different\n", encoding="utf-8")

    first_identity = fingerprint_snapshot_tree(first)
    second_identity = fingerprint_snapshot_tree(second)
    assert first_identity["tree_sha256"] == second_identity["tree_sha256"]
    assert first_identity["referenced_weight_shards"] == ["weights.safetensors"]
    assert ".cache/volatile.json" not in {
        entry["path"] for entry in first_identity["files"]
    }

    (second / "weights.safetensors").write_bytes(b"mutated-weight-bytes")
    assert (
        fingerprint_snapshot_tree(second)["tree_sha256"]
        != first_identity["tree_sha256"]
    )


def test_snapshot_requires_every_indexed_weight_shard(tmp_path: Path) -> None:
    root = _fake_snapshot(tmp_path / "snapshot")
    (root / "weights.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="referenced weight shard is absent"):
        fingerprint_snapshot_tree(root)


def test_local_expected_hash_is_enforced_and_revision_kwargs_are_omitted(
    tmp_path: Path,
) -> None:
    root = _fake_snapshot(tmp_path / "snapshot")
    expected = fingerprint_snapshot_tree(root)["tree_sha256"]
    identity = resolve_model_source_identity(
        model_id=str(root),
        source_id="Qwen/Qwen2.5-VL-3B-Instruct",
        revision=REVISION,
        processor_revision=REVISION,
        repository_root=tmp_path,
        expected_local_tree_sha256=expected,
    )
    assert identity["expected_tree_sha256_verified"] is True
    assert pretrained_revision_kwargs(identity, REVISION) == {}
    assert compact_source_identity(identity)["tree_sha256"] == expected

    with pytest.raises(ValueError, match="snapshot tree SHA-256 mismatch"):
        resolve_model_source_identity(
            model_id=str(root),
            source_id="Qwen/Qwen2.5-VL-3B-Instruct",
            revision=REVISION,
            processor_revision=REVISION,
            repository_root=tmp_path,
            expected_local_tree_sha256="0" * 64,
        )


def test_remote_identity_keeps_revision_enforcement(tmp_path: Path) -> None:
    identity = resolve_model_source_identity(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        source_id="Qwen/Qwen2.5-VL-3B-Instruct",
        revision=REVISION,
        processor_revision=REVISION,
        repository_root=tmp_path,
        expected_local_tree_sha256="0" * 64,
    )
    assert identity["kind"] == "huggingface_revision"
    assert pretrained_revision_kwargs(identity, REVISION) == {"revision": REVISION}
    assert compact_source_identity(identity)["revision"] == REVISION
