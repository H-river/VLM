from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from qwen_h1_meta_v0_candidate import training
from qwen_h1_meta_v0_candidate.baselines import (
    FrequencyMetaBaseline,
    MLPConfig,
    NumericFeatureEncoder,
    NumericMLPMetaBaseline,
    RandomMetaBaseline,
    RuleMetaBaseline,
)
from qwen_h1_meta_v0_candidate.contracts import parse_meta_output
from qwen_h1_meta_v0_candidate.data_pipeline import build_prebuilt_chat_row
from qwen_h1_meta_v0_candidate.evaluate_offline import (
    OfflineEvaluationError,
    evaluate_meta_predictions,
)
from qwen_h1_meta_v0_candidate.inference import (
    IndependentMetaAdapter,
    parse_generated_text,
)
from qwen_h1_meta_v0_candidate.generate_dev import generate_dev_rows


def _canonical(value: dict) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _alternative(valid_output: dict) -> dict:
    value = copy.deepcopy(valid_output)
    value.update(
        {
            "objective_profile": "width_priority",
            "step_scale": "fine",
            "risk_mode": "conservative",
            "confidence": "medium",
            "reason_codes": ["width_error_dominant"],
        }
    )
    value["directional_prior"] = {
        key: "decrease" for key in value["directional_prior"]
    }
    return parse_meta_output(value).to_dict()


def test_strict_generation_wrapper_preserves_invalid_raw_text(valid_output) -> None:
    valid = parse_generated_text(_canonical(valid_output))
    assert valid.valid_json is True
    assert valid.parsed is not None
    invalid = parse_generated_text("prefix " + _canonical(valid_output))
    assert invalid.valid_json is False
    assert invalid.raw_text.startswith("prefix ")
    assert invalid.error_code == "invalid_json"


def test_independent_adapter_uses_candidate_row_and_strict_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, valid_input, valid_output
) -> None:
    from PIL import Image

    monkeypatch.setattr(training, "PACKAGE_ROOT", tmp_path)
    image = tmp_path / "current.png"
    Image.new("L", (1024, 1024), 127).save(image)
    image_hash = training.sha256_path(image)
    row = build_prebuilt_chat_row(
        manifest_row={
            "record_id": "candidate_dev_generation_0001",
            "split": "dev",
            "model_visible_input": valid_input,
            "oracle_output": valid_output,
            "image": {"storage_path": str(image), "sha256": image_hash},
            "identity": {"setup_hash": "b" * 64},
            "oracle_audit": {"selected_configuration_id": "synthetic"},
        },
        system_prompt=training.SYSTEM_PROMPT,
        prompt_contract=training.PROMPT_CONTRACT,
    )
    calls = []

    def backend(prompt, image_path, seed, max_new_tokens):
        calls.append((prompt, image_path, seed, max_new_tokens))
        return _canonical(valid_output)

    generated = IndependentMetaAdapter(backend).generate_row(
        row, image_root=tmp_path, seed=2026080201
    )
    assert generated.valid_json is True
    assert generated.parsed.to_dict() == valid_output
    assert calls[0][2:] == (2026080201, 256)
    records = generate_dev_rows(
        rows=[row],
        adapter=IndependentMetaAdapter(backend),
        image_root=tmp_path,
        seed=2026080201,
        run_metadata={"adapter_hashes": {"adapter_config.json": "a" * 64}},
        expected_count=1,
    )
    assert records[0]["prediction"] == _canonical(valid_output)
    assert records[0]["generation_metadata"]["adapter_hashes"]
    assert records[0]["latency_seconds"] >= 0.0


def test_rule_frequency_and_random_baselines(valid_input, valid_output) -> None:
    rule = RuleMetaBaseline()
    invalid_measurement = copy.deepcopy(valid_input)
    invalid_measurement["measurement_validity"]["state"] = "invalid"
    assert rule.predict(invalid_measurement)["decision"] == "reobserve"
    exhausted = copy.deepcopy(valid_input)
    exhausted["remaining_budget"]["control_steps"] = 0
    assert rule.predict(exhausted)["decision"] == "stop"

    guided = copy.deepcopy(valid_input)
    guided["normalized_signed_error"]["centroid_x_px"] = 3.0
    guided["current_beam_state"]["centroid_x_px"] = 515.0
    guided["target_beam_state"]["centroid_x_px"] = 512.0
    for history in guided["history"]:
        history["executed_action_mm"]["lens_x_delta_mm"] = 0.05
        history["measured_beam_delta"]["centroid_x_px"] = 0.5
    predicted = rule.predict(guided)
    assert predicted["decision"] == "run_guided_h1"
    assert predicted["objective_profile"] == "centroid_priority"
    parse_meta_output(predicted)

    alternate = _alternative(valid_output)
    frequency = FrequencyMetaBaseline().fit([valid_output, valid_output, alternate])
    assert frequency.predict(valid_input) == valid_output
    random = RandomMetaBaseline(seed=17).fit([valid_output, alternate])
    first = random.predict(valid_input, sample_id="sample-a")
    second = random.predict(valid_input, sample_id="sample-a")
    assert first == second
    parse_meta_output(first)


def test_numeric_mlp_is_deterministic_and_handles_unseen_dev_configuration(
    tmp_path: Path, valid_input, valid_output
) -> None:
    positive = copy.deepcopy(valid_input)
    positive["normalized_signed_error"]["centroid_x_px"] = 4.0
    negative = copy.deepcopy(valid_input)
    negative["normalized_signed_error"]["centroid_x_px"] = -4.0
    target_a = copy.deepcopy(valid_output)
    target_b = _alternative(valid_output)
    unseen = copy.deepcopy(target_b)
    unseen["mask_profile"] = "camera_only"
    unseen["reason_codes"] = ["recent_response_mismatch"]
    parse_meta_output(unseen)

    train_inputs = [copy.deepcopy(positive) for _ in range(8)] + [
        copy.deepcopy(negative) for _ in range(8)
    ]
    train_targets = [target_a] * 8 + [target_b] * 8
    dev_inputs = [copy.deepcopy(positive), copy.deepcopy(negative), copy.deepcopy(positive)]
    dev_targets = [target_a, target_b, unseen]
    config = MLPConfig(
        hidden_units=8,
        epochs=20,
        batch_size=4,
        learning_rate=0.02,
        seed=11,
    )
    first = NumericMLPMetaBaseline(config).fit(
        train_inputs,
        train_targets,
        dev_inputs=dev_inputs,
        dev_targets=dev_targets,
    )
    second = NumericMLPMetaBaseline(config).fit(
        train_inputs,
        train_targets,
        dev_inputs=dev_inputs,
        dev_targets=dev_targets,
    )
    assert first.unseen_dev_configuration_count == 1
    assert first.best_epoch == second.best_epoch
    assert np.isclose(first.best_dev_loss, second.best_dev_loss)
    assert first.predict(dev_inputs) == second.predict(dev_inputs)
    assert "sha256" in first.state_dict()["actuator_semantics_encoding"]
    assert NumericFeatureEncoder().transform_one(valid_input).ndim == 1
    state_path = tmp_path / "numeric_mlp.json"
    first.save(state_path)
    restored = NumericMLPMetaBaseline.load(state_path)
    assert restored.predict(dev_inputs) == first.predict(dev_inputs)


def test_offline_evaluator_reports_all_seed_fields_and_regret(valid_output) -> None:
    alternative = _alternative(valid_output)
    manifest = [
        {
            "record_id": f"dev-{index:02d}",
            "split": "dev",
            "oracle_output": valid_output if index % 2 == 0 else alternative,
        }
        for index in range(24)
    ]
    predictions = []
    for seed in (2026080201, 2026080202, 2026080203):
        for index, row in enumerate(manifest):
            target = row["oracle_output"]
            predictions.append(
                {
                    "sample_id": row["record_id"],
                    "seed": seed,
                    "prediction": (
                        "not-json"
                        if seed == 2026080202 and index == 1
                        else _canonical(target)
                    ),
                    "latency_seconds": 0.1,
                }
            )

    def regret(sample_id, target, predicted, record):
        del sample_id, record
        return 0.0 if predicted == target else 1.0

    metadata = {
        seed: {
            "best_dev_loss": 1.0,
            "final_dev_loss": 1.2,
            "wall_time_seconds": 10.0,
            "peak_memory_bytes": 1234,
        }
        for seed in (2026080201, 2026080202, 2026080203)
    }
    report = evaluate_meta_predictions(
        manifest_records=manifest,
        prediction_records=predictions,
        expected_seeds=(2026080201, 2026080202, 2026080203),
        regret_hook=regret,
        seed_metadata=metadata,
    )
    perfect = report["per_seed"][0]
    assert perfect["valid_json_rate"] == 1.0
    assert perfect["full_configuration_exact_match"] == 1.0
    assert perfect["compiled_guidance_validity_rate"] == 1.0
    assert perfect["configuration_regret"]["mean"] == 0.0
    assert perfect["training_run"]["peak_memory_bytes"] == 1234
    assert report["per_seed"][1]["invalid_or_unknown_field_rate"] == 1 / 24
    assert report["per_seed"][2]["prediction_coverage_rate"] == 1.0
    assert report["manifest_split"] == "dev"
    assert report["manifest_record_count"] == 24
    assert report["prediction_grid_complete"] is True
    with pytest.raises(OfflineEvaluationError, match="one unique row for every"):
        evaluate_meta_predictions(
            manifest_records=manifest,
            prediction_records=predictions[:-1],
            expected_seeds=(2026080201, 2026080202, 2026080203),
        )
    for key in (
        "objective_profile_accuracy",
        "mask_profile_accuracy",
        "step_scale_accuracy",
        "risk_mode_accuracy",
        "direction_accuracy",
        "direction_macro_f1",
        "decision_macro_f1",
    ):
        assert key in perfect
