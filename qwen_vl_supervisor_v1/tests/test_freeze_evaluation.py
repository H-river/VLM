import hashlib
import json
from pathlib import Path

import pytest

from qwen_vl_supervisor_v1.scripts.freeze_evaluation import (
    FreezeError,
    _hash_path,
    _load_verified_preparation,
    _manifest_file_entries,
    _validate_baseline_selection_for_freeze,
    _validate_dev_selection_for_freeze,
    _verified_manifest_image,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_image_digest_is_computed_from_actual_bytes(tmp_path: Path):
    image = tmp_path / "image.png"
    image.write_bytes(b"actual deterministic PNG stand-in bytes")
    expected = _sha256(image)

    entry = _verified_manifest_image(
        tmp_path,
        split="frozen_iid",
        sample_id="sample-1",
        assets={
            "current_image_path": "image.png",
            "current_image_sha256": expected,
        },
    )
    assert entry["sha256"] == expected
    assert entry["manifest_current_image_sha256"] == expected
    assert entry["path"] == "image.png"

    image.write_bytes(b"drifted bytes")
    with pytest.raises(FreezeError, match="image-byte SHA-256 mismatch"):
        _verified_manifest_image(
            tmp_path,
            split="frozen_iid",
            sample_id="sample-1",
            assets={
                "current_image_path": "image.png",
                "current_image_sha256": expected,
            },
        )


def test_manifest_images_enter_later_verification_registry():
    manifests = {
        "frozen_iid": {
            "path": "manifest.jsonl",
            "sha256": hashlib.sha256(b"manifest").hexdigest(),
            "images": [
                {
                    "sample_id": "sample-1",
                    "path": "image.png",
                    "hash_algorithm": "sha256_file_v1",
                    "sha256": hashlib.sha256(b"image").hexdigest(),
                }
            ],
        }
    }
    entries = _manifest_file_entries(manifests)
    assert [entry["role"] for entry in entries] == [
        "manifest:frozen_iid",
        "manifest_image:frozen_iid:sample-1",
    ]


def test_manifest_image_path_may_not_traverse_a_symlink(tmp_path: Path):
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    image = real_directory / "image.png"
    image.write_bytes(b"image bytes")
    (tmp_path / "linked").symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(FreezeError, match="contains a symbolic link"):
        _verified_manifest_image(
            tmp_path,
            split="frozen_iid",
            sample_id="sample-1",
            assets={
                "current_image_path": "linked/image.png",
                "current_image_sha256": _sha256(image),
            },
        )


def test_preparation_artifact_sha_and_locked_file_drift_are_enforced(tmp_path: Path):
    locked = tmp_path / "locked.txt"
    locked.write_text("before server training\n", encoding="utf-8")
    locked_hash = _sha256(locked)
    artifact = tmp_path / "protocol-freeze.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": (
                    "qwen_vl_supervisor_evaluation_protocol_freeze_v1.0.0"
                ),
                "protocol_commitment": {
                    "sealed_before_server_training": True,
                    "required_by_later_final_freeze": True,
                    "formal_evaluation_executed_by_this_utility": False,
                    "prediction_files_opened_by_this_utility": False,
                },
                "files_to_verify_before_final_freeze": [
                    {
                        "role": "locked_input",
                        "path": "locked.txt",
                        "hash_algorithm": "sha256_file_v1",
                        "sha256": locked_hash,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_hash = _sha256(artifact)

    _, observed, _, files = _load_verified_preparation(
        tmp_path,
        artifact,
        artifact_hash,
    )
    assert observed == artifact_hash
    assert files[0]["sha256"] == locked_hash

    locked.write_text("changed after preparation\n", encoding="utf-8")
    with pytest.raises(FreezeError, match="frozen file drift"):
        _load_verified_preparation(tmp_path, artifact, artifact_hash)


def test_final_freeze_structurally_validates_selection_before_linking(tmp_path: Path):
    with pytest.raises(FreezeError, match="failed strict validation"):
        _validate_dev_selection_for_freeze(
            tmp_path,
            artifact={},
            config={},
            preparation={},
            training_config=tmp_path / "training.yaml",
            training_algorithm="sha256_file_v1",
            training_hash=hashlib.sha256(b"training").hexdigest(),
            training_seeds=[1],
            final_checkpoint_root=tmp_path,
        )


def test_final_freeze_rejects_unvalidated_baseline_selection(tmp_path: Path):
    with pytest.raises(FreezeError, match="failed exact validation"):
        _validate_baseline_selection_for_freeze(
            tmp_path,
            artifact={},
            config={},
        )


def test_selected_checkpoint_must_hash_and_resolve_beneath_final_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from qwen_vl_supervisor_v1 import select_dev_checkpoints

    monkeypatch.setattr(select_dev_checkpoints, "validate_selection_artifact", lambda _: None)
    training = tmp_path / "training.yaml"
    dev_sft = tmp_path / "dev.jsonl"
    dev_manifest = tmp_path / "manifest_dev.jsonl"
    selector_file = tmp_path / "selector.py"
    schema_file = tmp_path / "schema.json"
    for path, value in (
        (training, b"training"),
        (dev_sft, b"dev sft"),
        (dev_manifest, b"dev manifest"),
        (selector_file, b"selector"),
        (schema_file, b"schema"),
    ):
        path.write_bytes(value)

    final_root = tmp_path / "final-root"
    selected_path = final_root / "seed_1" / "checkpoint-25"
    selected_path.mkdir(parents=True)
    (selected_path / "adapter.safetensors").write_bytes(b"adapter")
    selected_algorithm, selected_hash = _hash_path(selected_path)

    def entry(path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(tmp_path).as_posix(),
            "hash_algorithm": "sha256_file_v1",
            "sha256": _sha256(path),
        }

    artifact = {
        "training_seeds": [1],
        "protected_or_frozen_data_used": False,
        "frozen_predictions_opened": False,
        "coverage": {"complete": True},
        "training_config": entry(training),
        "dev_data": {
            "sft": entry(dev_sft),
            "source_manifest": entry(dev_manifest),
        },
        "selector": {
            "implementation": entry(selector_file),
            "json_schema": entry(schema_file),
        },
        "seed_results": [
            {
                "training_seed": 1,
                "selected": {
                    "optimizer_step": 25,
                    "selection_rank": 1,
                    "checkpoint": {
                        "path": selected_path.relative_to(tmp_path).as_posix(),
                        "hash_algorithm": selected_algorithm,
                        "sha256": selected_hash,
                    },
                },
            }
        ],
    }
    config = {
        "immutable_artifacts": {
            "dev_checkpoint_selector": {
                "path": selector_file.relative_to(tmp_path).as_posix(),
                "sha256": _sha256(selector_file),
            },
            "dev_selection_artifact_schema": {
                "path": schema_file.relative_to(tmp_path).as_posix(),
                "sha256": _sha256(schema_file),
            },
        }
    }
    preparation = {
        "training_data_identities": {"dev_jsonl": entry(dev_sft)},
        "manifests": {
            "dev": {
                "path": dev_manifest.relative_to(tmp_path).as_posix(),
                "sha256": _sha256(dev_manifest),
            }
        },
    }
    links = _validate_dev_selection_for_freeze(
        tmp_path,
        artifact=artifact,
        config=config,
        preparation=preparation,
        training_config=training,
        training_algorithm="sha256_file_v1",
        training_hash=_sha256(training),
        training_seeds=[1],
        final_checkpoint_root=final_root,
    )
    assert links == [
        {
            "training_seed": 1,
            "optimizer_step": 25,
            "selection_rank": 1,
            "selected_checkpoint_path": "final-root/seed_1/checkpoint-25",
            "relative_path_within_final_checkpoint_root": "seed_1/checkpoint-25",
            "hash_algorithm": selected_algorithm,
            "sha256": selected_hash,
        }
    ]

    outside = tmp_path / "outside-checkpoint"
    outside.mkdir()
    (outside / "adapter.safetensors").write_bytes(b"adapter")
    outside_algorithm, outside_hash = _hash_path(outside)
    artifact["seed_results"][0]["selected"]["checkpoint"] = {
        "path": "outside-checkpoint",
        "hash_algorithm": outside_algorithm,
        "sha256": outside_hash,
    }
    with pytest.raises(FreezeError, match="escapes final checkpoint root"):
        _validate_dev_selection_for_freeze(
            tmp_path,
            artifact=artifact,
            config=config,
            preparation=preparation,
            training_config=training,
            training_algorithm="sha256_file_v1",
            training_hash=_sha256(training),
            training_seeds=[1],
            final_checkpoint_root=final_root,
        )
