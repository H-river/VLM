from __future__ import annotations

import copy
import json

import pytest

from qwen_h1_meta_v0_candidate.contracts import (
    MetaContractError,
    parse_meta_input,
    parse_meta_output,
    verify_protocol_freeze,
)


def test_protocol_freeze_and_valid_round_trip(valid_input, valid_output) -> None:
    verified = verify_protocol_freeze()
    assert "input_schema.json" in verified
    assert parse_meta_input(json.dumps(valid_input)).to_dict() == valid_input
    assert parse_meta_output(json.dumps(valid_output)).to_dict() == valid_output


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (
            '{"schema_version":"qwen_h1_meta_v0",'
            '"schema_version":"qwen_h1_meta_v0"}',
            "duplicate_json_key",
        ),
        ('{"x":NaN}', "non_finite_number"),
        ('{"x":Infinity}', "non_finite_number"),
        ('[]', "invalid_json_top_level"),
    ],
)
def test_output_rejects_malformed_json(payload: str, code: str) -> None:
    with pytest.raises(MetaContractError) as caught:
        parse_meta_output(payload)
    assert caught.value.code == code


@pytest.mark.parametrize(
    "payload",
    [
        "model says: {}",
        "{} trailing text",
        "```json\n{}\n```",
        "{}{}",
    ],
)
def test_output_rejects_every_text_prefix_suffix(payload: str) -> None:
    with pytest.raises(MetaContractError) as caught:
        parse_meta_output(payload)
    assert caught.value.code == "invalid_json"


def test_output_rejects_extra_and_direct_action_injection(valid_output) -> None:
    extra = copy.deepcopy(valid_output)
    extra["rationale"] = "free form"
    with pytest.raises(MetaContractError) as caught:
        parse_meta_output(extra)
    assert caught.value.code == "schema_validation"

    injected = copy.deepcopy(valid_output)
    injected["selected_action"] = {"lens_x_delta_mm": 0.05}
    with pytest.raises(MetaContractError) as caught:
        parse_meta_output(injected)
    assert caught.value.code == "direct_continuous_action_injection"


def test_output_rejects_numeric_direction_as_continuous_injection(valid_output) -> None:
    injected = copy.deepcopy(valid_output)
    injected["directional_prior"]["lens_x_delta_mm"] = 0.05
    with pytest.raises(MetaContractError) as caught:
        parse_meta_output(injected)
    assert caught.value.code == "schema_validation"


@pytest.mark.parametrize("field", ["mu", "sigma", "bounds", "covariance"])
def test_output_classifies_continuous_distribution_injection_as_security_event(
    valid_output, field: str
) -> None:
    injected = copy.deepcopy(valid_output)
    injected[field] = [0.01, 0.02]
    with pytest.raises(MetaContractError) as caught:
        parse_meta_output(injected)
    assert caught.value.code == "direct_continuous_action_injection"


def test_mapping_nonfinite_is_rejected_before_schema(valid_output) -> None:
    invalid = copy.deepcopy(valid_output)
    invalid["confidence"] = float("inf")
    with pytest.raises(MetaContractError) as caught:
        parse_meta_output(invalid)
    assert caught.value.code == "non_finite_number"


def test_input_rejects_hidden_field_and_extra_field(valid_input) -> None:
    leaked = copy.deepcopy(valid_input)
    leaked["setup_context"] = {"power_w": 1.0}
    with pytest.raises(MetaContractError) as caught:
        parse_meta_input(leaked)
    assert caught.value.code == "forbidden_visible_field"

    extra = copy.deepcopy(valid_input)
    extra["comment"] = "not in schema"
    with pytest.raises(MetaContractError) as caught:
        parse_meta_input(extra)
    assert caught.value.code == "schema_validation"


def test_input_rejects_noncanonical_image_and_actuator_semantics(valid_input) -> None:
    wrong_image = copy.deepcopy(valid_input)
    wrong_image["current_beam_image"]["width_px"] = 128
    wrong_image["current_beam_image"]["height_px"] = 128
    with pytest.raises(MetaContractError) as caught:
        parse_meta_input(wrong_image)
    assert caught.value.code == "invalid_image_role"

    duplicate = copy.deepcopy(valid_input)
    duplicate["actuator_semantics"][1]["action_id"] = "lens_x_delta_mm"
    with pytest.raises(MetaContractError) as caught:
        parse_meta_input(duplicate)
    assert caught.value.code == "duplicate_actuator_semantics"

    wrong_bounds = copy.deepcopy(valid_input)
    wrong_bounds["actuator_semantics"][0]["legal_per_step_bounds_mm"] = [-0.1, 0.1]
    with pytest.raises(MetaContractError) as caught:
        parse_meta_input(wrong_bounds)
    assert caught.value.code == "invalid_actuator_semantics"


def test_input_rejects_history_and_uncertainty_inconsistency(valid_input) -> None:
    padding = copy.deepcopy(valid_input)
    padding["history"][0]["valid"] = False
    with pytest.raises(MetaContractError) as caught:
        parse_meta_input(padding)
    assert caught.value.code == "invalid_history_padding"

    nonzero_padding = copy.deepcopy(valid_input)
    nonzero_padding["history"][0]["valid"] = False
    nonzero_padding["history"][0]["padding_reason"] = "no_history"
    nonzero_padding["history"][0]["executed_action_mm"]["lens_x_delta_mm"] = 0.01
    with pytest.raises(MetaContractError) as caught:
        parse_meta_input(nonzero_padding)
    assert caught.value.code == "nonzero_history_padding"

    summary = copy.deepcopy(valid_input)
    summary["forward_uncertainty"]["mean"] = 0.2
    with pytest.raises(MetaContractError) as caught:
        parse_meta_input(summary)
    assert caught.value.code == "inconsistent_uncertainty_summary"
