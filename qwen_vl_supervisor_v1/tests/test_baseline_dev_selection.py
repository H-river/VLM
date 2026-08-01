from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from qwen_vl_supervisor_v1.evaluate_offline import evaluate_records
from qwen_vl_supervisor_v1.validate_baseline_dev_selection import (
    A_CANDIDATES,
    A_COMPLEXITY_RANK,
    A_SELECTION_RULE,
    B_ARBITRATION,
    B_SELECTION_RULE,
    FILE_HASH_ALGORITHM,
    OUTPUT_MAPPING,
    REFERENCE_SEED,
    SCHEMA_VERSION,
    BaselineSelectionError,
    _b_prediction_rows,
    _b_rank_key,
    _canonical_sha256,
    _report_metrics,
    validate_baseline_selection_artifact,
)


MODULE_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "hash_algorithm": FILE_HASH_ALGORITHM,
        "sha256": _sha256(path),
    }


def _decision(diagnosis: str) -> str:
    return json.dumps(
        OUTPUT_MAPPING[diagnosis],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _coverage(report: dict[str, Any], size: int) -> dict[str, Any]:
    row = report["per_seed"][0]
    return {
        "manifest_records": size,
        "supplied_predictions": row["supplied_prediction_count"],
        "missing_predictions": row["missing_prediction_count"],
        "coverage_rate": row["coverage_rate"],
        "expected_seed_enforced": report["seed_enforcement"]
        == {"enabled": True, "expected_seeds": [REFERENCE_SEED]},
    }


def _fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = tmp_path
    evidence = root / "selection_evidence"
    train_manifest = root / "manifests/manifest_train.jsonl"
    dev_manifest = root / "manifests/manifest_dev.jsonl"
    train_rows = [
        {
            "sample_id": "train-nominal",
            "split": "train",
            "target": dict(OUTPUT_MAPPING["nominal"]),
        }
    ]
    dev_rows = [
        {
            "sample_id": "dev-nominal",
            "split": "dev",
            "target": dict(OUTPUT_MAPPING["nominal"]),
        },
        {
            "sample_id": "dev-saturation",
            "split": "dev",
            "target": dict(OUTPUT_MAPPING["sensor_saturation"]),
        },
        {
            "sample_id": "dev-reflection",
            "split": "dev",
            "target": dict(OUTPUT_MAPPING["secondary_reflection"]),
        },
    ]
    _write_jsonl(train_manifest, train_rows)
    _write_jsonl(dev_manifest, dev_rows)

    validator_path = root / "validation/validate_baseline_dev_selection.py"
    schema_path = root / "validation/baseline_dev_selection_artifact.schema.json"
    validator_path.parent.mkdir(parents=True, exist_ok=True)
    validator_path.write_text("# pinned validator fixture\n", encoding="utf-8")
    schema_path.write_text("{}\n", encoding="utf-8")

    specialist_paths = {
        "saturation_multimodal_small_model": root / "models/saturation.pt",
        "width_relative_reflection_selected_diagnostic": root / "models/reflection.joblib",
    }
    for name, path in specialist_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((name + "\n").encode("utf-8"))

    config: dict[str, Any] = {
        "manifests": {
            "train": {"path": train_manifest.relative_to(root).as_posix()},
            "dev": {"path": dev_manifest.relative_to(root).as_posix()},
        },
        "comparison_matrix": {
            "A": {"candidates": list(A_CANDIDATES)},
            "B": {
                "fixed_specialists": {
                    name: {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
                    for name, path in specialist_paths.items()
                }
            },
        },
        "immutable_artifacts": {
            "baseline_dev_selection_validator": {
                "path": validator_path.relative_to(root).as_posix(),
                "sha256": _sha256(validator_path),
            },
            "baseline_dev_selection_artifact_schema": {
                "path": schema_path.relative_to(root).as_posix(),
                "sha256": _sha256(schema_path),
            },
        },
    }

    manifest_ids = [row["sample_id"] for row in dev_rows]
    candidate_diagnoses = {
        "deterministic_frozen_rules": ["nominal", "nominal", "nominal"],
        "standardized_multinomial_logistic_regression": [
            "nominal",
            "sensor_saturation",
            "secondary_reflection",
        ],
        "standardized_one_hidden_layer_mlp": [
            "nominal",
            "sensor_saturation",
            "secondary_reflection",
        ],
    }
    candidates: list[dict[str, Any]] = []
    for name in A_CANDIDATES:
        candidate_root = evidence / "arm_A" / name
        model = candidate_root / "model.bin"
        training_report = candidate_root / "training_report.json"
        predictions = candidate_root / "predictions_dev.jsonl"
        report_path = candidate_root / "offline_dev.json"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes((name + " model\n").encode("utf-8"))
        _write_json(
            training_report,
            {
                "candidate": name,
                "training_split": "train",
                "selection_split": "dev",
                "protected_or_frozen_data_used": False,
            },
        )
        prediction_rows = [
            {
                "sample_id": sample_id,
                "prediction": _decision(diagnosis),
                "seed": REFERENCE_SEED,
            }
            for sample_id, diagnosis in zip(manifest_ids, candidate_diagnoses[name], strict=True)
        ]
        _write_jsonl(predictions, prediction_rows)
        report = evaluate_records(dev_rows, prediction_rows, expected_seeds=[REFERENCE_SEED])
        _write_json(report_path, report)
        candidates.append(
            {
                "candidate_name": name,
                "model_complexity_rank": A_COMPLEXITY_RANK[name],
                "selection_rank": 0,
                "model_artifact": _entry(root, model),
                "training_report": _entry(root, training_report),
                "predictions": _entry(root, predictions),
                "offline_report": _entry(root, report_path),
                "dev_coverage": _coverage(report, len(dev_rows)),
                "ranking_metrics": _report_metrics(report),
                "eligible": True,
                "protected_or_frozen_data_used": False,
                "frozen_predictions_opened": False,
            }
        )
    candidates.sort(
        key=lambda row: (
            -row["ranking_metrics"]["joint_exact_accuracy"],
            -row["ranking_metrics"]["diagnosis_macro_f1"],
            -row["ranking_metrics"]["valid_json_rate"],
            row["model_complexity_rank"],
            row["candidate_name"],
        )
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["selection_rank"] = rank
    a_winner = candidates[0]

    scores_path = evidence / "arm_B/scores_dev.jsonl"
    scores = [
        {
            "sample_id": "dev-nominal",
            "sensor_saturation_score": 0.1,
            "secondary_reflection_score": 0.1,
        },
        {
            "sample_id": "dev-saturation",
            "sensor_saturation_score": 0.9,
            "secondary_reflection_score": 0.2,
        },
        {
            "sample_id": "dev-reflection",
            "sensor_saturation_score": 0.2,
            "secondary_reflection_score": 0.9,
        },
    ]
    _write_jsonl(scores_path, scores)
    threshold_grid_path = evidence / "arm_B/threshold_grid_dev.jsonl"
    grid: list[dict[str, Any]] = []
    for saturation_threshold in (0.4, 0.6):
        for reflection_threshold in (0.4, 0.6):
            rows = _b_prediction_rows(
                scores,
                saturation_threshold=saturation_threshold,
                reflection_threshold=reflection_threshold,
            )
            report = evaluate_records(dev_rows, rows, expected_seeds=[REFERENCE_SEED])
            grid.append(
                {
                    "sensor_saturation_threshold": saturation_threshold,
                    "secondary_reflection_threshold": reflection_threshold,
                    "selection_rank": 0,
                    "ranking_metrics": _report_metrics(report),
                }
            )
    grid.sort(key=_b_rank_key)
    for rank, row in enumerate(grid, start=1):
        row["selection_rank"] = rank
    _write_jsonl(threshold_grid_path, grid)
    b_winner = grid[0]
    b_predictions = _b_prediction_rows(
        scores,
        saturation_threshold=b_winner["sensor_saturation_threshold"],
        reflection_threshold=b_winner["secondary_reflection_threshold"],
    )
    b_predictions_path = evidence / "arm_B/predictions_dev.jsonl"
    b_report_path = evidence / "arm_B/offline_dev.json"
    _write_jsonl(b_predictions_path, b_predictions)
    b_report = evaluate_records(dev_rows, b_predictions, expected_seeds=[REFERENCE_SEED])
    _write_json(b_report_path, b_report)

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "dev_only_A_and_B_baseline_selection",
        "selection_split": "dev",
        "train_data": {
            "source_manifest": _entry(root, train_manifest),
            "record_count": len(train_rows),
            "sample_id_set_sha256": _canonical_sha256(["train-nominal"]),
        },
        "dev_data": {
            "source_manifest": _entry(root, dev_manifest),
            "record_count": len(dev_rows),
            "sample_id_set_sha256": _canonical_sha256(sorted(manifest_ids)),
        },
        "arm_A": {
            "selection_rule": list(A_SELECTION_RULE),
            "candidates": candidates,
            "selected": {
                "candidate_name": a_winner["candidate_name"],
                "selection_rank": 1,
                "model_artifact": a_winner["model_artifact"],
                "ranking_metrics": a_winner["ranking_metrics"],
                "output_mapping": OUTPUT_MAPPING,
            },
            "protected_or_frozen_data_used": False,
            "frozen_predictions_opened": False,
        },
        "arm_B": {
            "fixed_specialists": {
                name: _entry(root, path) for name, path in specialist_paths.items()
            },
            "score_evidence": _entry(root, scores_path),
            "threshold_grid": _entry(root, threshold_grid_path),
            "grid_definition": {
                "sensor_saturation_thresholds": [0.4, 0.6],
                "secondary_reflection_thresholds": [0.4, 0.6],
                "cartesian_product_complete": True,
            },
            "selection_rule": list(B_SELECTION_RULE),
            "predictions": _entry(root, b_predictions_path),
            "offline_report": _entry(root, b_report_path),
            "dev_coverage": _coverage(b_report, len(dev_rows)),
            "selected": {
                "thresholds": {
                    "sensor_saturation": b_winner["sensor_saturation_threshold"],
                    "secondary_reflection": b_winner["secondary_reflection_threshold"],
                },
                "selection_rank": 1,
                "ranking_metrics": b_winner["ranking_metrics"],
                "arbitration": B_ARBITRATION,
                "output_mapping": OUTPUT_MAPPING,
            },
            "protected_or_frozen_data_used": False,
            "frozen_predictions_opened": False,
        },
        "protected_or_frozen_data_used": False,
        "frozen_predictions_opened": False,
        "timestamp_in_artifact": False,
        "validator": {
            "implementation": _entry(root, validator_path),
            "json_schema": _entry(root, schema_path),
            "offline_reports_recomputed_from_raw_predictions": True,
            "B_grid_recomputed_from_raw_scores": True,
        },
    }
    return artifact, config


def test_valid_artifact_recomputes_both_arms_and_matches_json_schema(tmp_path: Path) -> None:
    artifact, config = _fixture(tmp_path)
    result = validate_baseline_selection_artifact(
        artifact,
        repository_root=tmp_path,
        evaluation_config=config,
    )

    assert result["verification"] == "passed_exact_raw_evidence_recomputation"
    assert result["arm_A"]["selected_candidate"] == "standardized_multinomial_logistic_regression"
    assert result["arm_B"]["selected_thresholds"] == {
        "sensor_saturation": 0.4,
        "secondary_reflection": 0.4,
    }
    assert result["protected_or_frozen_data_used"] is False
    assert result["frozen_predictions_opened"] is False

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (MODULE_ROOT / "schema/baseline_dev_selection_artifact.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(artifact)


def test_rejects_tampered_a_winner_and_rank_order(tmp_path: Path) -> None:
    artifact, config = _fixture(tmp_path)
    tampered = copy.deepcopy(artifact)
    tampered["arm_A"]["selected"]["candidate_name"] = "deterministic_frozen_rules"
    with pytest.raises(BaselineSelectionError, match="rank-one candidate"):
        validate_baseline_selection_artifact(
            tampered,
            repository_root=tmp_path,
            evaluation_config=config,
        )

    tampered = copy.deepcopy(artifact)
    tampered["arm_A"]["candidates"][0]["selection_rank"] = 2
    with pytest.raises(BaselineSelectionError, match="selection ranks"):
        validate_baseline_selection_artifact(
            tampered,
            repository_root=tmp_path,
            evaluation_config=config,
        )


def test_rejects_tampered_b_grid_metrics_and_selected_threshold(tmp_path: Path) -> None:
    artifact, config = _fixture(tmp_path)
    grid_path = tmp_path / artifact["arm_B"]["threshold_grid"]["path"]
    rows = [json.loads(line) for line in grid_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["ranking_metrics"]["joint_exact_accuracy"] = 0.0
    _write_jsonl(grid_path, rows)
    artifact["arm_B"]["threshold_grid"] = _entry(tmp_path, grid_path)
    with pytest.raises(BaselineSelectionError, match="raw-score recomputation"):
        validate_baseline_selection_artifact(
            artifact,
            repository_root=tmp_path,
            evaluation_config=config,
        )

    artifact, config = _fixture(tmp_path / "second")
    artifact["arm_B"]["selected"]["thresholds"]["sensor_saturation"] = 0.6
    with pytest.raises(BaselineSelectionError, match="not rank one"):
        validate_baseline_selection_artifact(
            artifact,
            repository_root=tmp_path / "second",
            evaluation_config=config,
        )


def test_rejects_frozen_claim_or_protected_evidence_path(tmp_path: Path) -> None:
    artifact, config = _fixture(tmp_path)
    artifact["frozen_predictions_opened"] = True
    with pytest.raises(BaselineSelectionError, match="protection/timestamp"):
        validate_baseline_selection_artifact(
            artifact,
            repository_root=tmp_path,
            evaluation_config=config,
        )

    artifact, config = _fixture(tmp_path / "second")
    old_path = tmp_path / "second" / artifact["arm_B"]["score_evidence"]["path"]
    protected_path = tmp_path / "second/selection_evidence/arm_B/frozen_scores.jsonl"
    old_path.rename(protected_path)
    artifact["arm_B"]["score_evidence"] = _entry(tmp_path / "second", protected_path)
    with pytest.raises(BaselineSelectionError, match="protected/frozen path"):
        validate_baseline_selection_artifact(
            artifact,
            repository_root=tmp_path / "second",
            evaluation_config=config,
        )
