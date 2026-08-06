from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pytest

from qwen_h1_meta_v0_candidate import training
from qwen_h1_meta_v0_candidate.data_pipeline import build_prebuilt_chat_row
from qwen_h1_meta_v0_candidate.generate_training_config import build_training_config


def _write_image(path: Path) -> str:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (1024, 1024), color=127).save(path)
    return training.sha256_path(path)


def _row(
    *,
    split: str,
    record_id: str,
    image: Path,
    image_sha256: str,
    valid_input,
    valid_output,
) -> dict:
    manifest = {
        "record_id": record_id,
        "split": split,
        "model_visible_input": copy.deepcopy(valid_input),
        "oracle_output": copy.deepcopy(valid_output),
        "image": {"storage_path": str(image), "sha256": image_sha256},
        "identity": {"setup_hash": "a" * 64},
        "oracle_audit": {"selected_configuration_id": "synthetic_config"},
    }
    return build_prebuilt_chat_row(
        manifest_row=manifest,
        system_prompt=training.SYSTEM_PROMPT,
        prompt_contract=training.PROMPT_CONTRACT,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _pass_audit() -> dict:
    return {
        "version": training.OFFICIAL_DATA_AUDIT_VERSION,
        "candidate_only": True,
        "audited_splits": ["train", "dev"],
        "record_count": 72,
        "known_identity_registry_count": 298,
        "identity_overlap": {
            "known_overlap_count": 0,
            "cross_split_overlap_count": 0,
        },
        "exact_visible_collision": {"conflicting_collision_group_count": 0},
        "rounded_4dp_visible_collision": {
            "conflicting_collision_group_count": 0
        },
        "near_visible_collision": {"conflicting_near_pair_rate": 0.0},
        "visible_only_classifier": {"macro_f1": 0.75},
        "visible_plus_hidden_setup_classifier": {"macro_f1": 0.75},
        "hidden_setup_macro_f1_gain": 0.0,
        "thresholds": {
            "exact_conflict_count_max": 0,
            "rounded_4dp_conflict_count_max": 0,
            "near_conflict_rate_max": 0.20,
            "hidden_setup_macro_f1_gain_max": 0.15,
            "visible_classifier_macro_f1_min": 0.50,
        },
        "gate": "PASS",
        "red_reasons": [],
        "sft_export_permitted": True,
        "diagnostics": {},
    }


def _write_official_bundle(
    root: Path,
    *,
    valid_input,
    valid_output,
    train_count: int = 48,
    dev_count: int = 24,
    audit: dict | None = None,
) -> tuple[Path, Path, Path]:
    image = root / "data/images/current.png"
    image_hash = _write_image(image)
    train_rows = [
        _row(
            split="train",
            record_id=f"candidate_train_{index:04d}",
            image=image,
            image_sha256=image_hash,
            valid_input=valid_input,
            valid_output=valid_output,
        )
        for index in range(train_count)
    ]
    dev_rows = [
        _row(
            split="dev",
            record_id=f"candidate_dev_{index:04d}",
            image=image,
            image_sha256=image_hash,
            valid_input=valid_input,
            valid_output=valid_output,
        )
        for index in range(dev_count)
    ]
    train_path = root / "prebuilt_chat_train.jsonl"
    dev_path = root / "prebuilt_chat_dev.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(dev_path, dev_rows)
    audit_path = root / "reports/data_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit or _pass_audit(), sort_keys=True), encoding="utf-8"
    )
    report_path = root / "sft_export_report.json"
    report_path.write_text(
        json.dumps(
            {
                "version": training.OFFICIAL_SFT_EXPORT_REPORT_VERSION,
                "candidate_only": True,
                "information_gate": "PASS",
                "data_audit": {
                    "path": str(audit_path),
                    "sha256": training.sha256_path(audit_path),
                },
                "records": {
                    "train": {
                        "path": str(train_path),
                        "records": 48,
                        "sha256": training.sha256_path(train_path),
                    },
                    "dev": {
                        "path": str(dev_path),
                        "records": 24,
                        "sha256": training.sha256_path(dev_path),
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return train_path, dev_path, report_path


def test_data_export_to_candidate_prepare_rows_roundtrip(
    valid_input, valid_output
) -> None:
    # The inherited engine intentionally rejects any data path containing the
    # substring "test". Pytest's tmp_path always contains that substring, so
    # exercise the unmodified production guard in a recoverable candidate-local
    # directory whose name is itself legal.
    with tempfile.TemporaryDirectory(
        prefix=".candidate_fixture_", dir=training.PACKAGE_ROOT
    ) as directory:
        fixture_root = Path(directory)
        train_path, dev_path, report_path = _write_official_bundle(
            fixture_root,
            valid_input=valid_input,
            valid_output=valid_output,
        )
        output_dir = fixture_root / "artifacts/training/seed_2026080201"
        config = build_training_config(
            train_jsonl=train_path,
            dev_jsonl=dev_path,
            export_report=report_path,
            output_dir=output_dir,
            seed=2026080201,
        )
        training.validate_candidate_config(config)
        prepared = training.prepare_rows(config, verify_images=True)
        assert len(prepared.train_rows) == 48
        assert len(prepared.dev_rows) == 24
        assert prepared.train_rows[0]["example_id"] == "candidate_train_0000"
        assert prepared.dev_rows[0]["example_id"] == "candidate_dev_0000"
        assert config["schema_version"] == "qwen_h1_meta_training_v0"
        assert config["training_seeds"] == [2026080201, 2026080202, 2026080203]
        assert config["training"]["max_steps"] == 200


def test_red_audit_cannot_build_training_config(valid_input, valid_output) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".candidate_red_fixture_", dir=training.PACKAGE_ROOT
    ) as directory:
        audit = _pass_audit()
        audit.update(
            {
                "gate": "RED_STOP",
                "red_reasons": ["visible_classifier_macro_f1_below_0.50"],
                "sft_export_permitted": False,
                "visible_only_classifier": {"macro_f1": 0.01},
            }
        )
        train_path, dev_path, report_path = _write_official_bundle(
            Path(directory),
            valid_input=valid_input,
            valid_output=valid_output,
            audit=audit,
        )
        with pytest.raises(ValueError, match="unqualified PASS"):
            build_training_config(
                train_jsonl=train_path,
                dev_jsonl=dev_path,
                export_report=report_path,
                output_dir=Path(directory) / "artifacts/training/seed_2026080201",
                seed=2026080201,
            )


def test_wrong_train_cardinality_cannot_build_config(valid_input, valid_output) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".candidate_count_fixture_", dir=training.PACKAGE_ROOT
    ) as directory:
        train_path, dev_path, report_path = _write_official_bundle(
            Path(directory),
            valid_input=valid_input,
            valid_output=valid_output,
            train_count=47,
        )
        with pytest.raises(ValueError, match="exactly 48 rows"):
            build_training_config(
                train_jsonl=train_path,
                dev_jsonl=dev_path,
                export_report=report_path,
                output_dir=Path(directory) / "artifacts/training/seed_2026080201",
                seed=2026080201,
            )


def test_candidate_validator_rejects_prompt_or_supervisor_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, valid_input, valid_output
) -> None:
    monkeypatch.setattr(training, "PACKAGE_ROOT", tmp_path)
    image = tmp_path / "image.png"
    image_hash = _write_image(image)
    row = _row(
        split="dev",
        record_id="candidate_dev_0002",
        image=image,
        image_sha256=image_hash,
        valid_input=valid_input,
        valid_output=valid_output,
    )
    training.validate_prebuilt_row(
        row,
        allowed_splits={"dev"},
        image_root=tmp_path,
        verify_image=True,
    )
    wrong_prompt = copy.deepcopy(row)
    wrong_prompt["prompt"][0]["content"][0]["text"] = "different system"
    with pytest.raises(ValueError, match="system prompt"):
        training.validate_prebuilt_row(
            wrong_prompt,
            allowed_splits={"dev"},
            image_root=tmp_path,
            verify_image=False,
        )
    supervisor_target = copy.deepcopy(row)
    supervisor_target["completion"][0]["content"][0]["text"] = (
        '{"diagnosis":"nominal","measurement_policy":"standard",'
        '"supervisor_action":"execute"}'
    )
    with pytest.raises(Exception):
        training.validate_prebuilt_row(
            supervisor_target,
            allowed_splits={"dev"},
            image_root=tmp_path,
            verify_image=False,
        )


def test_training_engine_injection_is_scoped_and_restored() -> None:
    original_prepare = training.base.prepare_rows
    original_validator = training.base.validate_config
    with training.candidate_training_engine():
        assert training.base.prepare_rows is training.prepare_rows
        assert training.base.validate_config is training.validate_candidate_config
    assert training.base.prepare_rows is original_prepare
    assert training.base.validate_config is original_validator
