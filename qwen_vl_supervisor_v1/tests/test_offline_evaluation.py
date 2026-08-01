import json
from pathlib import Path

import pytest

from qwen_vl_supervisor_v1.evaluate_offline import (
    EvaluationError,
    PredictionFormatError,
    evaluate_files,
    evaluate_records,
    main,
    parse_supervisor_json,
)


def _decision(diagnosis="nominal", policy="standard", action="execute"):
    return json.dumps(
        {
            "diagnosis": diagnosis,
            "measurement_policy": policy,
            "supervisor_action": action,
        },
        separators=(",", ":"),
    )


def _manifest():
    return [
        {
            "sample_id": "nominal-1",
            "target": {
                "diagnosis": "nominal",
                "measurement_policy": "standard",
                "supervisor_action": "execute",
            },
            "provenance": {
                "anomaly_family": "nominal",
                "width_quartile": "Q2",
                "boundary_status": False,
                "severity_bucket": "none",
            },
        },
        {
            "sample_id": "saturation-1",
            "target": {
                "diagnosis": "sensor_saturation",
                "measurement_policy": "lower_exposure_reacquire",
                "supervisor_action": "reacquire",
            },
            "provenance": {
                "anomaly_family": "sensor_saturation",
                "width_quartile": "Q3",
                "boundary_status": True,
                "severity_bucket": "high",
            },
        },
        {
            "sample_id": "reflection-1",
            "target": {
                "diagnosis": "secondary_reflection",
                "measurement_policy": "primary_spot",
                "supervisor_action": "switch_measurement",
            },
            "provenance": {
                "anomaly_family": "width_relative_reflection",
                "width_quartile": "Q1",
                "boundary_status": "boundary",
                "severity_bucket": "medium",
            },
        },
        {
            "sample_id": "masked-action",
            "target": {
                "diagnosis": "nominal",
                "measurement_policy": "standard",
                # This sentinel deliberately is not a frozen enum.  Masking
                # must prevent the evaluator from inspecting or scoring it.
                "supervisor_action": "BLINDED",
                "field_mask": {
                    "diagnosis": True,
                    "measurement_policy": True,
                    "supervisor_action": False,
                },
            },
            "provenance": {
                "anomaly_family": "nominal",
                "boundary_status": False,
                "severity_bucket": "none",
            },
        },
    ]


def test_strict_whole_string_parser_accepts_only_canonical_contract():
    parsed = parse_supervisor_json("  \n" + _decision() + "\t")
    assert parsed["supervisor_action"] == "execute"

    invalid = [
        "```json\n" + _decision() + "\n```",
        _decision() + " trailing",
        "{}",
        '["nominal", "standard", "execute"]',
        '{"diagnosis":"nominal","measurement_policy":"standard",'
        '"supervisor_action":"execute","confidence":1}',
        '{"diagnosis":"nominal","diagnosis":"sensor_saturation",'
        '"measurement_policy":"standard","supervisor_action":"execute"}',
        '{"diagnosis":"unknown","measurement_policy":"standard",'
        '"supervisor_action":"execute"}',
        '{"diagnosis":"nominal","measurement_policy":"standard",'
        '"supervisor_action":NaN}',
    ]
    for text in invalid:
        with pytest.raises(PredictionFormatError):
            parse_supervisor_json(text)


def test_metrics_include_invalid_json_and_exclude_masked_targets():
    predictions = [
        {"sample_id": "nominal-1", "prediction": _decision(), "seed": 7},
        {
            "sample_id": "saturation-1",
            "prediction": _decision("nominal", "lower_exposure_reacquire", "reacquire"),
            "seed": 7,
        },
        {"sample_id": "reflection-1", "prediction": "not json", "seed": 7},
        {"sample_id": "masked-action", "prediction": _decision(), "seed": 7},
    ]
    report = evaluate_records(_manifest(), predictions)
    seed = report["per_seed"][0]

    assert seed["coverage_rate"] == 1.0
    assert seed["valid_json_rate"] == 0.75
    assert seed["diagnosis_balanced_accuracy"] == pytest.approx(1 / 3)
    assert seed["diagnosis_macro_f1"] == pytest.approx(0.8 / 3)
    assert seed["measurement_policy_accuracy"] == 0.75
    assert seed["supervisor_action_scored_count"] == 3
    assert seed["supervisor_action_macro_f1"] == pytest.approx(2 / 3)
    assert seed["joint_exact_accuracy"] == 0.5

    diagnosis_cm = seed["confusion_matrices"]["diagnosis"]
    invalid_index = diagnosis_cm["labels"].index("<invalid>")
    reflection_index = diagnosis_cm["labels"].index("secondary_reflection")
    assert diagnosis_cm["matrix"][reflection_index][invalid_index] == 1

    assert set(seed["slices"]["anomaly_family"]) == {
        "nominal",
        "sensor_saturation",
        "width_relative_reflection",
    }
    assert set(seed["slices"]["reflection_width_quartile"]) == {"Q1"}
    assert set(seed["slices"]["boundary_status"]) == {"boundary", "non_boundary"}
    assert set(seed["slices"]["severity_bucket"]) == {"high", "medium", "none"}


def test_per_seed_and_aggregate_statistics():
    predictions = []
    for seed in (11, 22):
        for row in _manifest():
            target = row["target"]
            action = target["supervisor_action"]
            if action == "BLINDED":
                action = "execute"
            prediction = _decision(target["diagnosis"], target["measurement_policy"], action)
            if seed == 22 and row["sample_id"] == "reflection-1":
                prediction = "invalid"
            predictions.append({"sample_id": row["sample_id"], "prediction": prediction, "seed": seed})

    report = evaluate_records(_manifest(), predictions)
    assert [item["seed"] for item in report["per_seed"]] == [11, 22]
    stats = report["aggregate"]["metrics"]["valid_json_rate"]
    assert stats == {
        "n": 2,
        "mean": 0.875,
        "std_population": 0.125,
        "min": 0.75,
        "max": 1.0,
    }
    slices = report["aggregate"]["slices"]
    assert set(slices) == {
        "anomaly_family",
        "reflection_width_quartile",
        "boundary_status",
        "severity_bucket",
    }
    q1 = slices["reflection_width_quartile"]["Q1"]
    assert q1["seeds_present"] == [11, 22]
    assert q1["metrics"]["valid_json_rate"] == {
        "n": 2,
        "mean": 0.5,
        "std_population": 0.5,
        "min": 0.0,
        "max": 1.0,
    }
    assert "boundary" in slices["boundary_status"]
    reflection_cm = q1["summed_confusion_matrices"]["diagnosis"]
    truth_index = reflection_cm["labels"].index("secondary_reflection")
    invalid_index = reflection_cm["labels"].index("<invalid>")
    assert reflection_cm["summed_matrix"][truth_index][invalid_index] == 1


def test_exact_expected_seed_set_rejects_missing_extra_and_seedless_predictions():
    rows = _manifest()
    one_seed = [
        {"sample_id": row["sample_id"], "prediction": _decision(), "seed": 11}
        for row in rows
    ]
    with pytest.raises(EvaluationError, match=r"missing=\[22\].*extra=\[\]"):
        evaluate_records(rows, one_seed, expected_seeds=[11, 22])

    extra = one_seed + [
        {"sample_id": row["sample_id"], "prediction": _decision(), "seed": 33}
        for row in rows
    ]
    with pytest.raises(EvaluationError, match=r"missing=\[22\].*extra=\[33\]"):
        evaluate_records(rows, extra, expected_seeds=[11, 22])

    seedless = [
        {"sample_id": row["sample_id"], "prediction": _decision()}
        for row in rows
    ]
    with pytest.raises(EvaluationError, match=r"extra=\[None\]"):
        evaluate_records(rows, seedless, expected_seeds=[11])

    # No expected set is the intentionally permissive development smoke mode.
    assert evaluate_records(rows, seedless)["seed_enforcement"] == {
        "enabled": False,
        "expected_seeds": None,
    }


def test_missing_rows_are_invalid_and_remain_in_metric_denominators():
    predictions = [
        {"sample_id": "nominal-1", "prediction": _decision(), "seed": 3},
        {
            "sample_id": "saturation-1",
            "prediction": _decision("sensor_saturation", "lower_exposure_reacquire", "reacquire"),
            "seed": 3,
        },
    ]

    seed = evaluate_records(_manifest(), predictions)["per_seed"][0]
    assert seed["manifest_count"] == 4
    assert seed["supplied_prediction_count"] == 2
    assert seed["coverage_rate"] == 0.5
    assert seed["missing_prediction_count"] == 2
    assert seed["num_predictions"] == 4
    assert seed["valid_json_count"] == 2
    assert seed["valid_json_rate"] == 0.5
    assert seed["joint_scored_count"] == 4
    assert seed["joint_exact_accuracy"] == 0.5
    assert {item["sample_id"] for item in seed["invalid_predictions"]} == {
        "reflection-1",
        "masked-action",
    }
    assert {item["error"] for item in seed["invalid_predictions"]} == {"missing_prediction"}


def test_cli_reads_jsonl_joins_by_sample_id_and_writes_report(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "report.json"
    manifest_path.write_text(
        "\n".join(json.dumps(row) for row in _manifest()) + "\n", encoding="utf-8"
    )
    predictions_path.write_text(
        "\n".join(
            json.dumps({"sample_id": row["sample_id"], "prediction": _decision()})
            for row in _manifest()
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(
        [
            "--manifest",
            str(manifest_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
        ]
    ) == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["manifest_count"] == 4
    assert written["per_seed"][0]["seed"] is None
    assert written["per_seed"][0]["valid_json_rate"] == 1.0
    assert evaluate_files(manifest_path, predictions_path) == written


def test_cli_loads_exact_seed_schedule_from_evaluation_yaml(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "report.json"
    config_path = tmp_path / "evaluation.yaml"
    manifest_path.write_text(
        "\n".join(json.dumps(row) for row in _manifest()) + "\n", encoding="utf-8"
    )
    predictions_path.write_text(
        "\n".join(
            json.dumps(
                {"sample_id": row["sample_id"], "prediction": _decision(), "seed": seed}
            )
            for seed in (11, 22)
            for row in _manifest()
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        "finalization:\n  training_seeds: [11, 22]\n", encoding="utf-8"
    )

    assert main(
        [
            "--manifest",
            str(manifest_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
            "--evaluation-config",
            str(config_path),
        ]
    ) == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["seed_enforcement"] == {
        "enabled": True,
        "expected_seeds": [11, 22],
    }
    assert [row["seed"] for row in written["per_seed"]] == [11, 22]


def test_unknown_and_duplicate_prediction_ids_fail_loudly():
    with pytest.raises(EvaluationError, match="unknown sample_id"):
        evaluate_records(_manifest(), [{"sample_id": "missing", "prediction": _decision()}])

    duplicate = [
        {"sample_id": "nominal-1", "prediction": _decision()},
        {"sample_id": "nominal-1", "prediction": _decision()},
    ]
    with pytest.raises(EvaluationError, match="duplicate prediction"):
        evaluate_records(_manifest(), duplicate)
