from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from qwen_vl_supervisor_v1.evaluate_offline import evaluate_files
from qwen_vl_supervisor_v1.select_dev_checkpoints import (
    RANKING_RULE,
    SelectionError,
    build_selection_artifact,
    validate_selection_artifact,
)
from qwen_vl_supervisor_v1.train_qlora import (
    SAFE_OPTIMIZER_JSON,
    SAFE_OPTIMIZER_TENSORS,
    SAFE_RNG_JSON,
    SAFE_RNG_TENSORS,
    SAFE_STATE_SCHEMA,
)


MODULE_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision(diagnosis: str) -> str:
    mapping = {
        "nominal": ("standard", "execute"),
        "sensor_saturation": ("lower_exposure_reacquire", "reacquire"),
    }
    policy, action = mapping[diagnosis]
    return json.dumps(
        {
            "diagnosis": diagnosis,
            "measurement_policy": policy,
            "supervisor_action": action,
        },
        separators=(",", ":"),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_safe_header_bundle(
    checkpoint: Path,
    *,
    json_name: str,
    tensor_name: str,
    header: dict[str, int],
) -> None:
    """Write the dependency-light envelope consumed by checkpoint_step.

    Selector tests validate checkpoint metadata rather than tensor loading, so
    the tensor payload may be opaque bytes as long as its declared digest is
    exact. Tensor round trips remain covered in the torch-dependent trainer
    tests.
    """

    tensor_path = checkpoint / tensor_name
    tensor_path.write_bytes(b"safe checkpoint selector fixture\n")
    items = [
        [
            {"kind": "scalar", "value": key},
            {"kind": "scalar", "value": value},
        ]
        for key, value in header.items()
    ]
    _write_json(
        checkpoint / json_name,
        {
            "schema": SAFE_STATE_SCHEMA,
            "tensor_file": tensor_name,
            "tensor_sha256": _sha256(tensor_path),
            "structure": {"kind": "dict", "items": items},
        },
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    config = root / "config.yaml"
    dev_sft = root / "sft_dev.jsonl"
    dev_manifest = root / "manifest_dev.jsonl"
    training_root = root / "training"
    evidence_root = root / "evidence"

    manifest_rows = [
        {
            "sample_id": "nominal-1",
            "split": "dev",
            "target": {
                "diagnosis": "nominal",
                "measurement_policy": "standard",
                "supervisor_action": "execute",
            },
            "provenance": {"anomaly_family": "sensor_saturation"},
        },
        {
            "sample_id": "saturation-1",
            "split": "dev",
            "target": {
                "diagnosis": "sensor_saturation",
                "measurement_policy": "lower_exposure_reacquire",
                "supervisor_action": "reacquire",
            },
            "provenance": {"anomaly_family": "sensor_saturation"},
        },
    ]
    dev_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    sft_rows = [
        {"example_id": row["sample_id"], "split": "dev"} for row in manifest_rows
    ]
    dev_sft.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sft_rows),
        encoding="utf-8",
    )
    config.write_text(
        "\n".join(
            [
                "training_seeds: [101, 202]",
                "data:",
                "  train_sha256: '" + "1" * 64 + "'",
                f"  dev_jsonl: {dev_sft.name}",
                f"  dev_sha256: {_sha256(dev_sft)}",
                "training:",
                "  max_steps: 2",
                "  save_steps: 1",
                "  save_total_limit: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_hash = _sha256(config)

    for seed in (101, 202):
        seed_dir = training_root / f"seed_{seed}"
        _write_json(
            seed_dir / "run_manifest.latest.json",
            {
                "status": "completed",
                "config_sha256": config_hash,
                "training": {"seed": seed, "data_seed": seed, "max_steps": 2},
                "training_result": {"global_step": 2},
                "data": {
                    "train_sha256": "1" * 64,
                    "dev_sha256": _sha256(dev_sft),
                    "frozen_predictions_opened": False,
                },
                "safe_checkpointing": {"legacy_optimizer_or_rng_pickle_loaded": False},
            },
        )
        for step in (1, 2):
            checkpoint = seed_dir / f"checkpoint-{step}"
            checkpoint.mkdir(parents=True)
            _write_json(checkpoint / "trainer_state.json", {"global_step": step})
            _write_json(checkpoint / "adapter_config.json", {"r": 8})
            (checkpoint / "adapter_model.safetensors").write_bytes(
                f"adapter:{seed}:{step}".encode()
            )
            _write_safe_header_bundle(
                checkpoint,
                json_name=SAFE_OPTIMIZER_JSON,
                tensor_name=SAFE_OPTIMIZER_TENSORS,
                header={"global_step": step, "world_size": 1},
            )
            _write_safe_header_bundle(
                checkpoint,
                json_name=SAFE_RNG_JSON,
                tensor_name=SAFE_RNG_TENSORS,
                header={"global_step": step, "world_size": 1, "process_index": 0},
            )

            evidence = evidence_root / f"seed_{seed}" / f"checkpoint-{step}"
            evidence.mkdir(parents=True)
            predictions = evidence / "predictions_dev.jsonl"
            prediction_rows = []
            for row in manifest_rows:
                diagnosis = row["target"]["diagnosis"]
                # Later checkpoint is strictly better; ranking must beat the
                # lower-step tie-break rather than selecting step 1.
                if step == 1 and row["sample_id"] == "saturation-1":
                    diagnosis = "nominal"
                prediction_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "prediction": _decision(diagnosis),
                        "seed": seed,
                    }
                )
            predictions.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
                encoding="utf-8",
            )
            generation_report = {
                "status": "completed",
                "seed": seed,
                "data": {
                    "split": "dev",
                    "sha256": _sha256(dev_sft),
                    "records": 2,
                    "frozen_or_protected_predictions_opened": False,
                },
                "adapter": {
                    "path": str(checkpoint.resolve()),
                    "hashes": {
                        "adapter_config.json": _sha256(checkpoint / "adapter_config.json"),
                        "adapter_model.safetensors": _sha256(
                            checkpoint / "adapter_model.safetensors"
                        ),
                    },
                },
                "progress": {"completed_records": 2, "expected_records": 2},
                "decoding": {"json_extraction_or_repair": False},
                "image_integrity": {"verified_records_before_model_load": 2},
                "config": {"sha256": config_hash},
                "output": str(predictions.resolve()),
            }
            _write_json(evidence / "predictions_dev.jsonl.report.json", generation_report)
            _write_json(
                evidence / "offline_dev.json",
                evaluate_files(dev_manifest, predictions, expected_seeds=[seed]),
            )

    return {
        "repository_root": root,
        "training_config": config,
        "training_root": training_root,
        "evidence_root": evidence_root,
        "dev_sft": dev_sft,
        "dev_manifest": dev_manifest,
    }


def test_selector_proves_complete_coverage_and_ranks_every_seed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    artifact = build_selection_artifact(**paths)

    assert artifact["ranking_rule"] == list(RANKING_RULE)
    assert artifact["coverage"] == {
        "expected_training_seeds": 2,
        "observed_training_seeds": 2,
        "expected_checkpoints_total": 4,
        "observed_checkpoints_total": 4,
        "complete": True,
    }
    assert [result["selected"]["optimizer_step"] for result in artifact["seed_results"]] == [
        2,
        2,
    ]
    assert all(
        result["observed_checkpoint_steps"] == [1, 2]
        and result["complete_checkpoint_coverage"]
        for result in artifact["seed_results"]
    )
    validate_selection_artifact(artifact)

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (MODULE_ROOT / "schema/dev_selection_artifact.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(artifact)


def test_selector_rejects_missing_checkpoint_and_protected_claim(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    missing = paths["training_root"] / "seed_101/checkpoint-1"
    missing.rename(paths["training_root"] / "seed_101/checkpoint-999")
    with pytest.raises(SelectionError, match="checkpoint coverage mismatch"):
        build_selection_artifact(**paths)

    paths = _fixture(tmp_path / "second")
    artifact = build_selection_artifact(**paths)
    tampered = copy.deepcopy(artifact)
    tampered["protected_or_frozen_data_used"] = True
    with pytest.raises(SelectionError, match="protected/frozen"):
        validate_selection_artifact(tampered)


def test_selector_recomputes_and_rejects_tampered_offline_report(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    report_path = paths["evidence_root"] / "seed_101/checkpoint-1/offline_dev.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["per_seed"][0]["joint_exact_accuracy"] = 1.0
    _write_json(report_path, report)

    with pytest.raises(SelectionError, match="not the exact current reducer result"):
        build_selection_artifact(**paths)
