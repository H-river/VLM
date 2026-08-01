from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from qwen_vl_supervisor_v1.contracts import TARGET_KEYS, parse_target_strict
from qwen_vl_supervisor_v1.export_sft import export, user_text, validate_export_row
from qwen_vl_supervisor_v1.validate_manifest import (
    ManifestValidationError,
    load_manifest,
    validate_manifest_files,
    validate_record,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_target_parser_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_target_strict(
            '{"diagnosis":"nominal","diagnosis":"sensor_saturation",'
            '"measurement_policy":"standard","supervisor_action":"execute"}'
        )
MODULE_ROOT = REPOSITORY_ROOT / "qwen_vl_supervisor_v1"
MANIFEST_ROOT = MODULE_ROOT / "manifests"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifests() -> list[Path]:
    return [
        MANIFEST_ROOT / "manifest_train.jsonl",
        MANIFEST_ROOT / "manifest_dev.jsonl",
        MANIFEST_ROOT / "manifest_frozen_iid.jsonl",
        MANIFEST_ROOT / "manifest_frozen_ood.jsonl",
    ]


def test_schema_is_valid_json_and_closes_every_primary_object():
    schema = json.loads((MODULE_ROOT / "schema/manifest.schema.json").read_text())
    assert schema["$id"] == "qwen_vl_supervisor_manifest_v1.0.0"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["model_input"]["additionalProperties"] is False
    assert schema["properties"]["target"]["additionalProperties"] is False
    assert schema["properties"]["provenance"]["additionalProperties"] is False


def test_full_manifests_pass_cross_split_validation_and_expected_counts():
    report = validate_manifest_files(_manifests(), repository_root=REPOSITORY_ROOT)
    assert report["status"] == "pass"
    assert report["records"] == 264
    assert report["split_counts"] == {
        "dev": 36,
        "frozen_iid": 72,
        "frozen_ood": 60,
        "train": 96,
    }
    for key in (
        "setup_overlap_across_splits",
        "pair_overlap_across_splits",
        "episode_overlap_across_splits",
        "augmented_base_overlap_across_splits",
        "image_hash_overlap_across_splits",
        "protected_rows_in_train_or_dev",
        "fixed_pixel_reflection_rows",
    ):
        assert report[key] == 0


def test_legal_class_counts_and_no_continue_stop_labels():
    expected = {
        "manifest_train.jsonl": {"nominal": 48, "sensor_saturation": 18, "secondary_reflection": 30},
        "manifest_dev.jsonl": {"nominal": 18, "sensor_saturation": 6, "secondary_reflection": 12},
        "manifest_frozen_iid.jsonl": {"nominal": 36, "sensor_saturation": 6, "secondary_reflection": 30},
        "manifest_frozen_ood.jsonl": {"nominal": 30, "sensor_saturation": 30},
    }
    for path in _manifests():
        rows = load_manifest(path)
        assert dict(Counter(row["target"]["diagnosis"] for row in rows)) == expected[path.name]
        assert not {row["target"]["supervisor_action"] for row in rows} & {"continue", "stop"}
        if path.name in {"manifest_train.jsonl", "manifest_dev.jsonl"}:
            assert not {row["provenance"]["source_split"] for row in rows} & {"iid_heldout", "severity_ood"}
        reflection = [row for row in rows if row["provenance"]["anomaly_family"] == "secondary_reflection"]
        assert all(row["provenance"]["source_cohort"].startswith("width_relative_") for row in reflection)


def test_export_is_byte_identical_for_same_seed_and_balanced(tmp_path: Path):
    kwargs = {
        "repository_root": REPOSITORY_ROOT,
        "train_manifest": MANIFEST_ROOT / "manifest_train.jsonl",
        "dev_manifest": MANIFEST_ROOT / "manifest_dev.jsonl",
        "seed": 2026080101,
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    report_a = export(output_dir=first, **kwargs)
    report_b = export(output_dir=second, **kwargs)
    for relative in (
        "sft_train.jsonl", "sft_dev.jsonl", "sft_smoke_train.jsonl",
        "sft_smoke_dev.jsonl", "sample_qwen_vl.jsonl", "export_report.json",
        "rendered_examples/nominal.md", "rendered_examples/sensor_saturation.md",
        "rendered_examples/secondary_reflection.md",
        "manifest_views/manifest_smoke_train_view.jsonl",
        "manifest_views/manifest_smoke_dev_view.jsonl",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
    assert report_a == report_b
    smoke_train = [json.loads(line) for line in (first / "sft_smoke_train.jsonl").read_text().splitlines()]
    smoke_dev = [json.loads(line) for line in (first / "sft_smoke_dev.jsonl").read_text().splitlines()]
    for rows, per_class in ((smoke_train, 12), (smoke_dev, 4)):
        counts = Counter(parse_target_strict(row["completion"][0]["content"][0]["text"])["diagnosis"] for row in rows)
        assert counts == {"nominal": per_class, "sensor_saturation": per_class, "secondary_reflection": per_class}


def test_export_has_one_image_roundtrips_numbers_and_never_leaks_target_to_user():
    manifest_by_id = {
        row["sample_id"]: row
        for path in (MANIFEST_ROOT / "manifest_train.jsonl", MANIFEST_ROOT / "manifest_dev.jsonl")
        for row in load_manifest(path)
    }
    for export_path in (MODULE_ROOT / "sft/sft_train.jsonl", MODULE_ROOT / "sft/sft_dev.jsonl"):
        for line in export_path.read_text().splitlines():
            row = json.loads(line)
            validate_export_row(row, repository_root=REPOSITORY_ROOT)
            original = manifest_by_id[row["example_id"]]
            user = next(
                item["text"]
                for message in row["prompt"] if message["role"] == "user"
                for item in message["content"] if item["type"] == "text"
            )
            state = json.loads(user.split("\n", 1)[1])
            assert state["current_metrics"] == original["model_input"]["current_metrics"]
            assert state["goal_metrics"] == original["model_input"]["goal_metrics"]
            target = {key: original["target"][key] for key in TARGET_KEYS}
            assert parse_target_strict(row["completion"][0]["content"][0]["text"]) == target
            assert target["diagnosis"] not in user
            assert original["provenance"]["source_sample_id"] not in user
            assert original["assets"]["current_image_path"] not in user


def test_no_truncation_contract_keeps_image_state_goal_and_target():
    row = json.loads((MODULE_ROOT / "sft/sample_qwen_vl.jsonl").read_text().splitlines()[0])
    assert sum(
        item.get("type") == "image"
        for message in row["prompt"]
        for item in message["content"]
    ) == 1
    user = next(item["text"] for message in row["prompt"] if message["role"] == "user" for item in message["content"] if item["type"] == "text")
    state = json.loads(user.split("\n", 1)[1])
    assert state["current_metrics"]
    assert state["goal_metrics"]
    assert parse_target_strict(row["completion"][0]["content"][0]["text"])


def test_validator_fails_missing_invalid_nonfinite_future_and_leakage():
    original = load_manifest(MANIFEST_ROOT / "manifest_train.jsonl")[0]
    cases = []

    missing = copy.deepcopy(original)
    del missing["model_input"]["goal_metrics"]
    cases.append(missing)

    invalid_enum = copy.deepcopy(original)
    invalid_enum["target"]["diagnosis"] = "fixed_pixel_reflection"
    cases.append(invalid_enum)

    nonfinite = copy.deepcopy(original)
    nonfinite["model_input"]["current_metrics"]["centroid_x"] = float("nan")
    cases.append(nonfinite)

    future = copy.deepcopy(original)
    future["step_index"] = 1
    future["model_input"]["remaining_step_budget"] = 7
    future["model_input"]["recent_history"] = [{
        "observation_index": 1,
        "observed_metrics": copy.deepcopy(original["model_input"]["current_metrics"]),
        "previous_high_level_action": "execute",
    }]
    cases.append(future)

    leakage = copy.deepcopy(original)
    leakage["model_input"]["anomaly_label"] = "sensor_saturation"
    cases.append(leakage)

    for index, case in enumerate(cases):
        with pytest.raises(ManifestValidationError):
            validate_record(case, repository_root=REPOSITORY_ROOT, context=f"case-{index}")


def test_validator_fails_duplicate_ids_broken_pairs_and_cross_split_hashes(tmp_path: Path):
    rows = load_manifest(MANIFEST_ROOT / "manifest_train.jsonl")[:2]
    duplicate = copy.deepcopy(rows)
    duplicate[1]["sample_id"] = duplicate[0]["sample_id"]
    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_text("".join(json.dumps(row) + "\n" for row in duplicate))
    with pytest.raises(ManifestValidationError, match="duplicate sample IDs"):
        validate_manifest_files([duplicate_path], repository_root=REPOSITORY_ROOT)

    # A single member cannot constitute a counterfactual pair.
    broken_path = tmp_path / "broken.jsonl"
    broken_path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ManifestValidationError, match="broken pair"):
        validate_manifest_files([broken_path], repository_root=REPOSITORY_ROOT)

    # Copy a valid pair to a second split with new IDs but identical setup/image hashes.
    pair_id = rows[0]["counterfactual_pair_id"]
    pair = [row for row in load_manifest(MANIFEST_ROOT / "manifest_train.jsonl") if row["counterfactual_pair_id"] == pair_id]
    assert len(pair) == 2
    first_path = tmp_path / "one.jsonl"
    second_path = tmp_path / "two.jsonl"
    first_path.write_text("".join(json.dumps(row) + "\n" for row in pair))
    moved = copy.deepcopy(pair)
    for index, row in enumerate(moved):
        row["split"] = "dev"
        row["sample_id"] = f"qvlsup1_{index + 1:024x}"
    second_path.write_text("".join(json.dumps(row) + "\n" for row in moved))
    with pytest.raises(ManifestValidationError, match="setup overlap"):
        validate_manifest_files([first_path, second_path], repository_root=REPOSITORY_ROOT)
