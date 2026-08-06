from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from Qwen_orchestration.runtime.dispatcher import validate_decision
from Qwen_orchestration.runtime.errors import ContractError, SpecialistError
from Qwen_orchestration.v12.adapter import (
    ACTION_FIELDS,
    POSITION_FIELDS,
    V12Adapter,
    canonical_four_vector,
)
from Qwen_orchestration.v12.dispatcher import validate_v12_decision


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROWS = (
    REPO_ROOT
    / "runs/overnight_v12_semantics_20260731_002709/data/"
    "corrected_128_16_16_v2/transitions/test.jsonl"
)


@pytest.fixture(scope="module")
def row() -> dict:
    return json.loads(TEST_ROWS.open(encoding="utf-8").readline())


@pytest.fixture(scope="module")
def adapter() -> V12Adapter:
    return V12Adapter()


def quantity(values: list[float], unit: str = "mm") -> dict:
    return {
        "values": dict(zip(("lens_x", "lens_y", "camera_x", "camera_y"), values)),
        "unit": unit,
    }


def state_decision(row: dict, *, route: str = "predict_forward_from_state_v12") -> dict:
    task = (
        "direction_prediction_v12"
        if "direction" in route
        else "forward_prediction_v12"
    )
    return {
        "schema_version": "qwen_orchestration_decision_v12_v1",
        "status": "ready",
        "task_type": task,
        "route_name": route,
        "arguments": {
            "setup_context": row["setup_context"],
            "actuator_position": quantity(
                [row["positions_mm"][field] for field in POSITION_FIELDS]
            ),
            "current_beam_state": row["metrics"],
            "continuous_action": quantity(
                [row["action_mm"][field] for field in ACTION_FIELDS]
            ),
        },
        "image_roles": {},
        "missing_fields": [],
        "reason": None,
    }


def inverse_decision(row: dict) -> dict:
    value = state_decision(row)
    value["task_type"] = "inverse_control_v12"
    value["route_name"] = "inverse_control_from_states_v12_h1"
    value["arguments"].pop("continuous_action")
    value["arguments"]["target_beam_state"] = row["next_metrics"]
    return value


def test_action_order_signs_and_scientific_notation() -> None:
    result = canonical_four_vector(
        quantity([3e-2, -2.5e-2, 1e-2, -1.5e-2]),
        name="continuous_action",
        output_fields=ACTION_FIELDS,
    )
    assert list(result) == list(ACTION_FIELDS)
    assert list(result.values()) == [0.03, -0.025, 0.01, -0.015]


@pytest.mark.parametrize("unit", ["um", "µm", "μm"])
def test_micrometre_units_equal_mm(unit: str) -> None:
    microns = canonical_four_vector(
        quantity([30.0, -20.0, 10.0, -5.0], unit),
        name="continuous_action",
        output_fields=ACTION_FIELDS,
    )
    millimetres = canonical_four_vector(
        quantity([0.03, -0.02, 0.01, -0.005], "mm"),
        name="continuous_action",
        output_fields=ACTION_FIELDS,
    )
    assert microns == millimetres


def test_canonical_unit_is_mm() -> None:
    assert canonical_four_vector(
        quantity([0.01, 0, 0, 0], "canonical"),
        name="continuous_action",
        output_fields=ACTION_FIELDS,
    )["lens_x_delta_mm"] == 0.01


def test_missing_unit_is_never_guessed() -> None:
    with pytest.raises(ContractError, match="exactly values and unit"):
        canonical_four_vector(
            {"values": quantity([0, 0, 0, 0])["values"]},
            name="continuous_action",
            output_fields=ACTION_FIELDS,
        )


def test_noop_is_exact_and_has_no_bias(adapter: V12Adapter, row: dict) -> None:
    result = adapter.forward(
        setup_context=row["setup_context"],
        actuator_position=quantity(
            [row["positions_mm"][field] for field in POSITION_FIELDS]
        ),
        current_beam_state=row["metrics"],
        continuous_action=quantity([0.0, 0.0, 0.0, 0.0]),
        run_id="unit-noop",
        route="predict_forward_from_state_v12",
    )
    assert result["mean_normalized_metric_delta"] == [0.0] * 5
    assert list(result["decoded_physical_metric_delta"].values()) == [0.0] * 5
    assert result["predicted_next_beam_state"] == row["metrics"]


@pytest.mark.parametrize(
    "values",
    [
        [0.05, 0.0, 0.0, 0.0],
        [-0.05, 0.05, -0.02, 0.02],
        [0.0123, -0.0045, 0.0067, -0.0089],
    ],
)
def test_single_multi_and_non_grid_actions_are_v12(
    adapter: V12Adapter, row: dict, values: list[float]
) -> None:
    action = adapter.canonicalize_action(
        quantity(values),
        adapter.canonicalize_position(
            quantity([row["positions_mm"][field] for field in POSITION_FIELDS])
        ),
    )
    assert np.allclose(list(action.values()), values)


def test_action_at_bound_is_legal(adapter: V12Adapter) -> None:
    position = adapter.canonicalize_position(quantity([0, 0, 0, 0]))
    assert adapter.canonicalize_action(
        quantity([0.05, -0.05, 0.02, -0.02]), position
    )


def test_action_over_bound_is_rejected(adapter: V12Adapter) -> None:
    position = adapter.canonicalize_position(quantity([0, 0, 0, 0]))
    with pytest.raises(ContractError, match="per-step bounds"):
        adapter.canonicalize_action(quantity([0.050001, 0, 0, 0]), position)


def test_post_action_absolute_position_is_rejected(adapter: V12Adapter) -> None:
    position = adapter.canonicalize_position(quantity([2.99, 0, 0, 0]))
    with pytest.raises(ContractError, match="absolute limits"):
        adapter.canonicalize_action(quantity([0.02, 0, 0, 0]), position)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nan_and_inf_are_rejected(adapter: V12Adapter, invalid: float) -> None:
    with pytest.raises(ContractError, match="finite"):
        adapter.canonicalize_state(
            {
                "centroid_x_px": invalid,
                "centroid_y_px": 0,
                "sigma_x_px": 1,
                "sigma_y_px": 1,
                "peak_intensity": 1,
            },
            "current_beam_state",
        )


def test_missing_current_state_fails_registry(row: dict) -> None:
    decision = state_decision(row)
    decision["arguments"].pop("current_beam_state")
    with pytest.raises(ContractError, match="argument groups differ"):
        validate_v12_decision(decision, {})


def test_missing_target_state_fails_registry(row: dict) -> None:
    decision = inverse_decision(row)
    decision["arguments"].pop("target_beam_state")
    with pytest.raises(ContractError, match="argument groups differ"):
        validate_v12_decision(decision, {})


def test_image_roles_cannot_be_swapped_or_aliased(row: dict, tmp_path: Path) -> None:
    image = tmp_path / "beam.png"
    image.write_bytes(b"not decoded during validation")
    decision = inverse_decision(row)
    decision["route_name"] = "inverse_control_from_images_v12_h1"
    decision["arguments"] = {
        key: value
        for key, value in decision["arguments"].items()
        if key in {"setup_context", "actuator_position"}
    }
    decision["arguments"]["image_calibration"] = {
        "linear_intensity_low": 0,
        "linear_intensity_high": 1,
        "gamma": 1,
        "source_sensor_resolution_px": [1024, 1024],
    }
    decision["image_roles"] = {"current_beam": "image_0", "target_beam": "image_0"}
    with pytest.raises(ContractError, match="cannot fill multiple"):
        validate_v12_decision(decision, {"image_0": image})
    decision["image_roles"] = {"current_beam": "image_0", "desired_beam": "image_1"}
    with pytest.raises(ContractError, match="image_roles"):
        validate_v12_decision(decision, {"image_0": image, "image_1": image})


def test_malformed_json_is_not_a_decision() -> None:
    with pytest.raises(json.JSONDecodeError):
        json.loads('{"status":"ready",')


def test_checkpoint_hash_mismatch_stops_loading(tmp_path: Path) -> None:
    source = json.loads(
        (REPO_ROOT / "Qwen_orchestration/v12/runtime_config.json").read_text()
    )
    source["v12_checkpoint"]["sha256"] = "0" * 64
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(SpecialistError, match="hash mismatch"):
        V12Adapter(path)


def test_v12_schema_forbids_legacy_81_route(row: dict) -> None:
    decision = state_decision(row)
    decision["route_name"] = "predict_forward_from_state_v1"
    with pytest.raises(ContractError, match="schema violation"):
        validate_v12_decision(decision, {})


def test_old_route_contract_remains_valid() -> None:
    old = {
        "schema_version": "qwen_orchestration_decision_v1",
        "status": "unsupported",
        "task_type": None,
        "route_name": None,
        "arguments": {},
        "image_roles": {},
        "missing_fields": [],
        "clarification_question": None,
    }
    assert validate_decision(old, {}) == {}
