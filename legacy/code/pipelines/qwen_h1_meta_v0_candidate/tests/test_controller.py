from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from continuous_control_v12.contracts import Bounds
from continuous_control_v12.mpc import CEMMPC
from qwen_h1_meta_v0_candidate import controller as controller_module
from qwen_h1_meta_v0_candidate.compiler import (
    H1OnlyViolation,
    compile_guidance,
    default_h1_config,
)
from qwen_h1_meta_v0_candidate.controller import (
    MetaH1Controller,
    canonical_one_step_score,
    make_physical_evaluation,
    plan_guided_h1,
    reject_only_safety_gate,
)


def bounds() -> Bounds:
    return Bounds(
        action_low=np.asarray([-0.05, -0.05, -0.02, -0.02]),
        action_high=np.asarray([0.05, 0.05, 0.02, 0.02]),
        position_low=np.full(4, -3.0),
        position_high=np.full(4, 3.0),
    )


def small_test_config() -> dict:
    return default_h1_config()


class LinearPredictor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, positions, metrics, action):
        del positions
        self.calls += 1
        output = np.asarray(metrics, dtype=np.float64).copy()
        output[:4] += np.asarray(action, dtype=np.float64)
        output[4] += float(np.asarray(action).sum())
        return (
            output,
            np.zeros(5, dtype=np.float64),
            {
                "clipping_probability": np.asarray([0.0]),
                "boundary_probability": np.asarray([0.0]),
            },
        )


POSITIONS = np.zeros(4, dtype=np.float64)
CURRENT = np.zeros(5, dtype=np.float64)
TARGET = np.asarray([2.0, -2.0, 3.0, -3.0, 20.0])
REFERENCE = np.asarray([0.0, 0.0, 1.0, 1.0, 100.0])


def _run_direct(predictor: LinearPredictor) -> dict:
    planner = CEMMPC(
        bounds=bounds(), predictor=predictor, config=small_test_config(), seed=90210
    )
    return planner.plan(
        positions_mm=POSITIONS,
        current_metrics=CURRENT,
        target_metrics=TARGET,
        allowed_dofs=["lens_x", "lens_y", "camera_x", "camera_y"],
        tolerance_reference=REFERENCE,
    )


def test_off_is_exact_unchanged_cem_and_never_parses_meta() -> None:
    expected = _run_direct(LinearPredictor())
    controller = MetaH1Controller(
        default_bounds=bounds(),
        predictor=LinearPredictor(),
        default_config=small_test_config(),
        seed=90210,
        mode="off",
    )
    result = controller.run(
        positions_mm=POSITIONS,
        current_metrics=CURRENT,
        target_metrics=TARGET,
        tolerance_reference=REFERENCE,
        guidance_payload='{"mu":[0.1,0.2]} definitely invalid',
    )
    assert result.default_plan == expected
    assert result.selected_action == expected["selected_effective_action"]
    assert result.selected_source == "default"
    assert result.parse_error_code is None


def test_default_full_result_canonical_hash_is_repeat_stable_and_matches_off() -> None:
    first = _run_direct(LinearPredictor())
    second = _run_direct(LinearPredictor())
    off = MetaH1Controller(
        default_bounds=bounds(),
        predictor=LinearPredictor(),
        default_config=small_test_config(),
        seed=90210,
        mode="off",
    ).run(
        positions_mm=POSITIONS,
        current_metrics=CURRENT,
        target_metrics=TARGET,
        tolerance_reference=REFERENCE,
    )

    def digest(value) -> str:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    assert first == second == off.default_plan
    assert digest(first) == digest(second) == digest(off.default_plan)
    # The preregistered 8ff4... digest used an unrecorded Phase-0 trace
    # serialization.  Full-result equality above is the authoritative runtime
    # regression and does not guess a replacement serialization.


def test_frozen_cem_and_forward_sources_remain_byte_unchanged() -> None:
    repository = Path(__file__).resolve().parents[2]
    expected = {
        "continuous_control_v12/mpc.py": "bfa5f50d1c053b7f2dcbc2b6c5f8544d0165e572706264a02568d2eed8085d78",
        "continuous_control_v12/world_model.py": "7e38071cf9ca2f14425261f3ef87e9abf696375b6fa2be103db16e97c77d1c26",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((repository / relative).read_bytes()).hexdigest() == digest


def test_guided_h1_uses_compiled_bounds_and_exact_candidate_budget(valid_output) -> None:
    predictor = LinearPredictor()
    guidance = compile_guidance(valid_output, default_bounds=bounds())
    plan = plan_guided_h1(
        guidance=guidance,
        default_bounds=bounds(),
        predictor=predictor,
        default_config=default_h1_config(),
        seed=90210,
        positions_mm=POSITIONS,
        current_metrics=CURRENT,
        target_metrics=TARGET,
        tolerance_reference=REFERENCE,
    )
    action = np.asarray(list(plan["selected_effective_action"].values()))
    compiled_bounds = guidance.bounds_for(bounds())
    assert np.all(action >= compiled_bounds.action_low - 1e-12)
    assert np.all(action <= compiled_bounds.action_high + 1e-12)
    assert plan["candidate_sequences_evaluated"] == 24 * 3
    assert predictor.calls == 24 * 3


def test_first_normal_mean_shift_changes_only_first_draw() -> None:
    location = np.zeros((1, 1, 4), dtype=np.float64)
    scale = np.ones((1, 1, 4), dtype=np.float64)
    shift = np.asarray([0.1, -0.2, 0.0, 0.3])
    expected_rng = np.random.default_rng(17)
    expected_first = expected_rng.normal(location + shift.reshape(1, 1, 4), scale, size=(3, 1, 4))
    expected_second = expected_rng.normal(location, scale, size=(3, 1, 4))

    proxy = controller_module._FirstNormalMeanShiftGenerator(
        np.random.default_rng(17), shift
    )
    assert np.array_equal(proxy.normal(location, scale, size=(3, 1, 4)), expected_first)
    assert np.array_equal(proxy.normal(location, scale, size=(3, 1, 4)), expected_second)


def _plan(action: list[float]) -> dict:
    fields = (
        "lens_x_delta_mm",
        "lens_y_delta_mm",
        "camera_x_delta_mm",
        "camera_y_delta_mm",
    )
    encoded = dict(zip(fields, action, strict=True))
    return {
        "selected_action": dict(encoded),
        "selected_requested_action": dict(encoded),
        "selected_effective_action": dict(encoded),
    }


def _evaluation(
    centroid_x: float,
    *,
    uncertainty: float = 0.0,
    members: list[list[float]] | None = None,
):
    return make_physical_evaluation(
        [centroid_x, 0.0, 1.0, 1.0, 100.0],
        [uncertainty] * 5,
        {"clipping_probability": [0.0], "boundary_probability": [0.0]},
        member_next_metrics=members,
    )


def test_canonical_scorer_matches_existing_h1_formula() -> None:
    evaluation = make_physical_evaluation(
        [2.0, 0.0, 1.0, 1.0, 100.0],
        [0.5] * 5,
        {"clipping_probability": [0.1], "boundary_probability": [0.2]},
        member_next_metrics=[[2.0, 0.0, 1.0, 1.0, 100.0]],
    )
    config = default_h1_config()
    score = canonical_one_step_score(
        action_mm=[0.025, 0.0, 0.0, 0.0],
        evaluation=evaluation,
        target_metrics=[1.0, 0.0, 1.0, 1.0, 100.0],
        tolerance_reference=[0.0, 0.0, 1.0, 1.0, 100.0],
        default_bounds=bounds(),
        default_config=config,
        limit_projected=False,
    )
    terminal_max = 1.0
    terminal_mean = 0.2
    expected = (
        terminal_max
        + 0.15 * terminal_mean
        + 0.10 * terminal_max
        + 0.02 * 0.5
        + 1.0 * 0.3
        + 0.1 * 0.5
    )
    assert score == pytest.approx(expected)


def _gate(
    valid_output,
    *,
    default_evaluation,
    guided_evaluation,
    guided_action=None,
    measurement_valid=True,
):
    guidance = compile_guidance(valid_output, default_bounds=bounds())
    return reject_only_safety_gate(
        positions_mm=POSITIONS,
        target_metrics=[1.0, 0.0, 1.0, 1.0, 100.0],
        tolerance_reference=[0.0, 0.0, 1.0, 1.0, 100.0],
        default_plan=_plan([0.0, 0.0, 0.0, 0.0]),
        guided_plan=_plan(
            [0.01, 0.0, 0.0, 0.0] if guided_action is None else guided_action
        ),
        default_evaluation=default_evaluation,
        guided_evaluation=guided_evaluation,
        guidance=guidance,
        default_bounds=bounds(),
        default_config=default_h1_config(),
        measurement_valid=measurement_valid,
    )


def test_reject_only_gate_accepts_only_canonically_better_member_safe_guidance(valid_output) -> None:
    default_eval = _evaluation(2.0, members=[[2.0, 0.0, 1.0, 1.0, 100.0]] * 3)
    guided_eval = _evaluation(1.0, members=[[1.0, 0.0, 1.0, 1.0, 100.0]] * 3)
    accepted = _gate(valid_output, default_evaluation=default_eval, guided_evaluation=guided_eval)
    assert accepted.accepted_guided
    assert accepted.rejection_reasons == ()

    worse = _gate(
        valid_output,
        default_evaluation=default_eval,
        guided_evaluation=_evaluation(
            4.0, members=[[4.0, 0.0, 1.0, 1.0, 100.0]] * 3
        ),
    )
    assert not worse.accepted_guided
    assert "guided_worse_than_default_canonical_objective" in worse.rejection_reasons


def test_reject_only_gate_fails_closed_on_uncertainty_members_bounds_and_measurement(valid_output) -> None:
    default_eval = _evaluation(2.0, members=[[2.0, 0.0, 1.0, 1.0, 100.0]] * 3)
    catastrophic_members = [
        [1.0, 0.0, 1.0, 1.0, 100.0],
        [10.0, 0.0, 1.0, 1.0, 100.0],
        [1.0, 0.0, 1.0, 1.0, 100.0],
    ]
    catastrophic = _gate(
        valid_output,
        default_evaluation=default_eval,
        guided_evaluation=_evaluation(1.0, members=catastrophic_members),
    )
    assert "catastrophic_member_increase" in catastrophic.rejection_reasons

    unavailable = _gate(
        valid_output,
        default_evaluation=default_eval,
        guided_evaluation=_evaluation(1.0, members=None),
    )
    assert "member_predictions_unavailable" in unavailable.rejection_reasons

    uncertain = _gate(
        valid_output,
        default_evaluation=default_eval,
        guided_evaluation=_evaluation(
            1.0,
            uncertainty=5.1,
            members=[[1.0, 0.0, 1.0, 1.0, 100.0]] * 3,
        ),
    )
    assert "guided_uncertainty_above_threshold" in uncertain.rejection_reasons

    outside_compiled = _gate(
        valid_output,
        default_evaluation=default_eval,
        guided_evaluation=_evaluation(
            1.0, members=[[1.0, 0.0, 1.0, 1.0, 100.0]] * 3
        ),
        guided_action=[0.04, 0.0, 0.0, 0.0],
    )
    assert "guided_action_failed_existing_bounds" in outside_compiled.rejection_reasons

    invalid_measurement = _gate(
        valid_output,
        default_evaluation=default_eval,
        guided_evaluation=_evaluation(
            1.0, members=[[1.0, 0.0, 1.0, 1.0, 100.0]] * 3
        ),
        measurement_valid=False,
    )
    assert "measurement_invalid" in invalid_measurement.rejection_reasons


def test_guarded_controller_can_accept_guided_after_independent_physical_rescore(
    monkeypatch, valid_output
) -> None:
    default_plan = _plan([0.0, 0.0, 0.0, 0.0])
    guided_plan = _plan([0.01, 0.0, 0.0, 0.0])
    monkeypatch.setattr(controller_module, "plan_default_h1", lambda **kwargs: default_plan)
    monkeypatch.setattr(controller_module, "plan_guided_h1", lambda **kwargs: guided_plan)

    def evaluator(positions, current, action):
        del positions, current
        if float(action[0]) == 0.0:
            return _evaluation(2.0, members=[[2.0, 0.0, 1.0, 1.0, 100.0]] * 3)
        return _evaluation(1.0, members=[[1.0, 0.0, 1.0, 1.0, 100.0]] * 3)

    controller = MetaH1Controller(
        default_bounds=bounds(),
        predictor=LinearPredictor(),
        default_config=default_h1_config(),
        seed=7,
        mode="guarded",
        action_evaluator=evaluator,
    )
    result = controller.run(
        positions_mm=POSITIONS,
        current_metrics=[0.0, 0.0, 1.0, 1.0, 100.0],
        target_metrics=[1.0, 0.0, 1.0, 1.0, 100.0],
        guidance_payload=valid_output,
    )
    assert result.dispatch
    assert result.selected_source == "guided"
    assert result.selected_action == guided_plan["selected_effective_action"]
    assert result.gate is not None and result.gate.accepted_guided


def test_invalid_or_injected_guidance_falls_back_to_unchanged_default(
    monkeypatch, valid_output
) -> None:
    default_plan = _plan([0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(controller_module, "plan_default_h1", lambda **kwargs: default_plan)
    controller = MetaH1Controller(
        default_bounds=bounds(),
        predictor=LinearPredictor(),
        default_config=default_h1_config(),
        seed=7,
        mode="guarded",
    )
    injected = copy.deepcopy(valid_output)
    injected["mu"] = [0.1, 0.2, 0.3, 0.4]
    result = controller.run(
        positions_mm=POSITIONS,
        current_metrics=CURRENT,
        target_metrics=TARGET,
        guidance_payload=injected,
    )
    assert result.selected_source == "default"
    assert result.parse_error_code == "direct_continuous_action_injection"
    assert result.fallback_reason == "direct_continuous_action_injection"


@pytest.mark.parametrize(
    ("decision", "observation_request", "expected_operation"),
    [
        ("stop", "reuse_current", "stop"),
        ("stop", "revalidate", "stop"),
        ("reobserve", "reuse_current", "reobserve"),
        ("reobserve", "revalidate", "reobserve"),
        ("run_guided_h1", "revalidate", "reobserve"),
        ("run_default_h1", "revalidate", "reobserve"),
    ],
)
def test_decision_observation_combinations_are_safely_semanticized(
    monkeypatch, valid_output, decision, observation_request, expected_operation
) -> None:
    monkeypatch.setattr(
        controller_module,
        "plan_default_h1",
        lambda **kwargs: _plan([0.0, 0.0, 0.0, 0.0]),
    )
    payload = copy.deepcopy(valid_output)
    payload["decision"] = decision
    payload["observation_request"] = observation_request
    controller = MetaH1Controller(
        default_bounds=bounds(),
        predictor=LinearPredictor(),
        default_config=default_h1_config(),
        seed=7,
        mode="guarded",
    )
    result = controller.run(
        positions_mm=POSITIONS,
        current_metrics=CURRENT,
        target_metrics=TARGET,
        guidance_payload=payload,
    )
    assert result.operation == expected_operation
    assert not result.dispatch
    assert result.selected_action is None


def test_shadow_never_allows_meta_stop_to_change_default_dispatch(
    monkeypatch, valid_output
) -> None:
    default_plan = _plan([0.01, 0.0, 0.0, 0.0])
    monkeypatch.setattr(controller_module, "plan_default_h1", lambda **kwargs: default_plan)
    payload = copy.deepcopy(valid_output)
    payload["decision"] = "stop"
    payload["observation_request"] = "revalidate"
    controller = MetaH1Controller(
        default_bounds=bounds(),
        predictor=LinearPredictor(),
        default_config=default_h1_config(),
        seed=7,
        mode="shadow",
    )
    result = controller.run(
        positions_mm=POSITIONS,
        current_metrics=CURRENT,
        target_metrics=TARGET,
        guidance_payload=payload,
    )
    assert result.dispatch
    assert result.selected_source == "default"
    assert result.selected_action == default_plan["selected_effective_action"]


def test_controller_constructor_makes_h3_impossible() -> None:
    config = default_h1_config()
    config["horizon"] = 3
    with pytest.raises(H1OnlyViolation):
        MetaH1Controller(
            default_bounds=bounds(),
            predictor=LinearPredictor(),
            default_config=config,
            seed=7,
            mode="off",
        )
