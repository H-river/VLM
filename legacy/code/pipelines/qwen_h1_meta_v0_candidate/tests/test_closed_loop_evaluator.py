from __future__ import annotations

import copy

import numpy as np
import pytest

from continuous_control_v12.contracts import Bounds, apply_action
from qwen_h1_meta_v0_candidate.compiler import default_h1_config
from qwen_h1_meta_v0_candidate.controller import make_physical_evaluation
from qwen_h1_meta_v0_candidate.evaluate_closed_loop import (
    BOOTSTRAP_SAMPLES,
    CallablePolicy,
    ClosedLoopEvaluationError,
    LiveImagePolicy,
    MetaPolicyResult,
    MethodSpec,
    PairedClosedLoopEvaluator,
    SimulationObservation,
    TracePolicy,
    UnavailablePolicy,
    aggregate_method_episodes,
    merge_closed_loop_reports,
    normalize_candidate_eval_manifest,
    paired_comparison,
    runtime_input_fingerprint,
    sensor_image_fingerprint,
    setup_bootstrap_interval,
    standard_method_specs,
)


class SyntheticBackend:
    def __init__(self) -> None:
        self.bounds = Bounds(
            action_low=np.asarray([-0.05, -0.05, -0.02, -0.02]),
            action_high=np.asarray([0.05, 0.05, 0.02, 0.02]),
            position_low=np.full(4, -3.0),
            position_high=np.full(4, 3.0),
        )
        self.identity = {
            "backend": "candidate_test_synthetic",
            "forward_checkpoint_sha256": "synthetic",
            "simulator_config_sha256": "synthetic",
        }
        self.initial_image_calls = 0

    @staticmethod
    def _next(metrics, action):
        output = np.asarray(metrics, dtype=np.float64).copy()
        output[0] += 20.0 * float(action[0])
        output[1] += 20.0 * float(action[1])
        return output

    def predictor(self, record):
        del record

        def predict(positions, metrics, action):
            del positions
            return (
                self._next(metrics, action),
                np.full(5, 0.1, dtype=np.float64),
                {
                    "clipping_probability": np.asarray([0.0]),
                    "boundary_probability": np.asarray([0.0]),
                },
            )

        return predict

    def action_evaluator(self, record):
        del record

        def evaluate(positions, metrics, action):
            del positions
            predicted = self._next(metrics, action)
            return make_physical_evaluation(
                predicted,
                [0.1] * 5,
                {
                    "clipping_probability": [0.0],
                    "boundary_probability": [0.0],
                },
                member_next_metrics=np.stack([predicted, predicted, predicted]),
            )

        return evaluate

    def initial_sensor_image(self, record, positions_mm):
        del record, positions_mm
        self.initial_image_calls += 1
        return np.full((8, 8), 0.5, dtype=np.float64)

    def simulate(self, record, positions_mm, action_mm):
        del record
        next_positions = apply_action(positions_mm, action_mm, self.bounds)
        metrics = np.asarray(
            [20.0 * next_positions[0], 20.0 * next_positions[1], 10.0, 10.0, 100.0]
        )
        return SimulationObservation(
            positions_mm=tuple(float(value) for value in next_positions),
            metrics=tuple(float(value) for value in metrics),
            simulator_valid=True,
            clipping_fraction=0.0,
            camera_boundary_indicator=False,
            raw_auxiliary={"simulator_valid": True},
            sensor_image_normalized=np.full(
                (8, 8), 0.5 + 0.1 * float(next_positions[0]), dtype=np.float64
            ),
        )


def synthetic_record(valid_input, *, record_id="synthetic_eval_0000", setup="setup-a"):
    runtime = copy.deepcopy(valid_input)
    runtime["current_beam_state"] = {
        "centroid_x_px": 0.0,
        "centroid_y_px": 0.0,
        "sigma_x_px": 10.0,
        "sigma_y_px": 10.0,
        "peak_intensity": 100.0,
    }
    runtime["target_beam_state"] = {
        "centroid_x_px": 2.0,
        "centroid_y_px": 0.0,
        "sigma_x_px": 10.0,
        "sigma_y_px": 10.0,
        "peak_intensity": 100.0,
    }
    runtime["normalized_signed_error"] = {
        "centroid_x_px": 2.0,
        "centroid_y_px": 0.0,
        "sigma_x_px": 0.0,
        "sigma_y_px": 0.0,
        "peak_intensity": 0.0,
    }
    return {
        "schema_version": "qwen_h1_meta_v0_candidate_record_v1",
        "record_id": record_id,
        "split": "candidate_eval",
        "image": {"storage_path": "/must/not/be/opened.png"},
        "model_visible_input": runtime,
        "oracle_output": None,
        "evaluator_only": {
            "setup_id": setup,
            "episode_id": record_id,
            "target_counterfactual_id": "target-a",
            "setup_index": 0,
            "target_index": 0,
            "setup_context": {},
            "simulator_fixed": {"simulator_semantics_version": "synthetic"},
        },
    }


def default_output(valid_output):
    output = copy.deepcopy(valid_output)
    output["decision"] = "run_default_h1"
    output["observation_request"] = "reuse_current"
    return output


def test_manifest_is_candidate_eval_only_and_never_resolves_image(valid_input) -> None:
    record = synthetic_record(valid_input)
    normalized = normalize_candidate_eval_manifest(
        [record], require_preregistered_cardinality=False
    )
    assert normalized[0]["image"]["storage_path"] == "/must/not/be/opened.png"

    wrong = copy.deepcopy(record)
    wrong["split"] = "dev"
    with pytest.raises(ClosedLoopEvaluationError, match="candidate_eval only"):
        normalize_candidate_eval_manifest(
            [wrong], require_preregistered_cardinality=False
        )
    preregistered_name = copy.deepcopy(record)
    preregistered_name["record_id"] = "qh1meta_eval_0000__target-00-a"
    with pytest.raises(ClosedLoopEvaluationError, match="12 setups x 3 targets"):
        normalize_candidate_eval_manifest(
            [preregistered_name], require_preregistered_cardinality=True
        )


def test_trace_policy_requires_exact_episode_step_and_reports_missing(valid_output) -> None:
    trace = TracePolicy(
        [
            {
                "episode_id": "episode-a",
                "step": 0,
                "prediction": default_output(valid_output),
                "latency_seconds": 0.25,
                "configuration_regret": 0.0,
            }
        ]
    )
    from qwen_h1_meta_v0_candidate.evaluate_closed_loop import PolicyContext

    context = PolicyContext("m", "episode-a", "setup-a", 0, 1, {}, {})
    assert trace.decide(context).latency_seconds == 0.25
    missing = trace.decide(
        PolicyContext("m", "episode-a", "setup-a", 1, 1, {}, {})
    )
    assert not missing.available
    assert missing.unavailable_reason == "missing_per_step_prediction_trace"


def test_dynamic_qwen_trace_is_bound_to_current_state_and_sensor_image(
    valid_input, valid_output
) -> None:
    image = np.full((8, 8), 0.5, dtype=np.float64)
    trace = TracePolicy(
        [
            {
                "episode_id": "episode-a",
                "step": 0,
                "prediction": default_output(valid_output),
                "runtime_input_sha256": runtime_input_fingerprint(valid_input),
                "sensor_image_fingerprint_sha256": sensor_image_fingerprint(image),
                "latency_seconds": 0.5,
            }
        ],
        require_dynamic_image_binding=True,
    )
    from qwen_h1_meta_v0_candidate.evaluate_closed_loop import PolicyContext

    context = PolicyContext(
        "qwen_guided_h1_seed_2026080201",
        "episode-a",
        "setup-a",
        0,
        1,
        valid_input,
        {},
        image,
    )
    assert trace.decide(context).available
    changed = copy.deepcopy(valid_input)
    changed["current_beam_state"]["centroid_x_px"] += 1.0
    mismatch = trace.decide(
        PolicyContext(
            context.method_name,
            context.episode_id,
            context.setup_id,
            context.step,
            context.planner_seed,
            changed,
            {},
            image,
        )
    )
    assert not mismatch.available
    assert mismatch.unavailable_reason == "dynamic_trace_runtime_state_mismatch"


def test_standard_registry_is_exact_h1_and_dual_budget_is_48x3() -> None:
    methods = standard_method_specs()
    assert tuple(method.name for method in methods) == (
        "default_h1",
        "dual_budget_default_h1",
        "rule_guided_h1",
        "metrics_mlp_meta_h1",
        "qwen_guided_h1_seed_2026080201",
        "qwen_guided_h1_seed_2026080202",
        "qwen_guided_h1_seed_2026080203",
        "oracle_guided_h1",
        "random_or_frequency_meta_h1",
        "shadow_qwen_h1",
    )
    assert all(method.resolved_config()["horizon"] == 1 for method in methods)
    dual = next(method for method in methods if method.name == "dual_budget_default_h1")
    assert dual.resolved_config()["population"] == 48
    assert dual.resolved_config()["cem_iterations"] == 3


def test_standard_registry_rejects_static_trace_as_qwen(valid_output) -> None:
    static_trace = TracePolicy(
        [
            {
                "episode_id": "episode-a",
                "step": 0,
                "prediction": default_output(valid_output),
            }
        ]
    )
    with pytest.raises(ClosedLoopEvaluationError, match="live image-aware"):
        standard_method_specs(
            {"qwen_guided_h1_seed_2026080201": static_trace}
        )


def test_small_paired_run_uses_matched_seeds_and_reports_all_metrics(
    valid_input, valid_output
) -> None:
    static_default = CallablePolicy(
        lambda context: MetaPolicyResult(
            payload=default_output(valid_output),
            latency_seconds=0.01,
            configuration_regret=0.0,
        )
    )
    dual_config = default_h1_config()
    dual_config["population"] = 48
    methods = (
        MethodSpec("default_h1", "off"),
        MethodSpec(
            "dual_budget_default_h1",
            "off",
            planner_config=dual_config,
            planner_profile="dual_budget_default_h1",
        ),
        MethodSpec("static_meta_h1", "guarded", policy=static_default),
    )
    backend = SyntheticBackend()
    report = PairedClosedLoopEvaluator(
        backend=backend,
        methods=methods,
        bootstrap_samples=32,
    ).evaluate(
        [synthetic_record(valid_input)],
        require_preregistered_cardinality=False,
    )
    assert report["candidate_only"] is True
    assert report["scientific_conclusion"] is False
    assert set(report["methods"]) == {
        "default_h1",
        "dual_budget_default_h1",
        "static_meta_h1",
    }
    default_episode = report["episodes"]["default_h1"][0]
    meta_episode = report["episodes"]["static_meta_h1"][0]
    assert default_episode["trace"][0]["planner_seed"] == meta_episode["trace"][0][
        "planner_seed"
    ]
    assert default_episode["candidate_evaluation_count"] % 72 == 0
    assert report["episodes"]["dual_budget_default_h1"][0][
        "candidate_evaluation_count"
    ] % 144 == 0
    assert meta_episode["configuration_regret"]["value"] == 0.0
    for field in (
        "strict_all_five_success",
        "steps_to_success",
        "final_normalized_error",
        "error_step_auc",
        "actual_canonical_objective_reduction",
    ):
        assert field in report["methods"]["default_h1"]["performance"]
    for field in (
        "action_norm",
        "overshoot_frequency",
        "out_of_domain_proposal_frequency",
        "safety_rejection_rate",
        "invalid_json_rate",
        "override_rate",
        "fallback_rate",
        "reobserve_rate",
        "stop_rate",
        "guided_acceptance_rate",
        "configuration_regret",
        "ensemble_uncertainty",
        "policy_latency_seconds",
        "candidate_evaluation_count",
        "wall_clock_compute_seconds",
    ):
        assert field in report["methods"]["static_meta_h1"]["audit"]
    assert report["paired_vs_default"]["static_meta_h1"][
        "paired_win_loss_tie"
    ]["value"] is not None
    assert backend.initial_image_calls == 0


def test_live_image_policy_gets_fresh_current_and_transition_images(
    valid_input, valid_output
) -> None:
    observed = []

    def live(context):
        observed.append(
            (
                context.step,
                None
                if context.sensor_image_normalized is None
                else context.sensor_image_normalized.copy(),
            )
        )
        return MetaPolicyResult(
            payload=default_output(valid_output),
            latency_seconds=0.01,
            configuration_regret=0.0,
        )

    backend = SyntheticBackend()
    record = synthetic_record(valid_input)
    record["model_visible_input"]["target_beam_state"]["centroid_x_px"] = 4.0
    record["model_visible_input"]["normalized_signed_error"]["centroid_x_px"] = 4.0
    report = PairedClosedLoopEvaluator(
        backend=backend,
        methods=(
            MethodSpec("default_h1", "off"),
            MethodSpec("live_qwen_h1", "guarded", policy=LiveImagePolicy(live)),
        ),
        bootstrap_samples=8,
    ).evaluate(
        [record],
        require_preregistered_cardinality=False,
    )
    assert backend.initial_image_calls == 1
    assert len(observed) >= 2 and all(image is not None for _, image in observed)
    assert not np.array_equal(observed[0][1], observed[1][1])
    provenance = report["methods"]["live_qwen_h1"]["policy_provenance"]
    assert provenance["requires_sensor_image"]
    assert provenance["live_image_generation"]


def test_missing_dynamic_trace_never_masquerades_as_method_performance(
    valid_input,
) -> None:
    report = PairedClosedLoopEvaluator(
        backend=SyntheticBackend(),
        methods=(
            MethodSpec("default_h1", "off"),
            MethodSpec(
                "missing_meta_h1",
                "guarded",
                policy=UnavailablePolicy("synthetic missing trace"),
            ),
        ),
        bootstrap_samples=16,
    ).evaluate(
        [synthetic_record(valid_input)],
        require_preregistered_cardinality=False,
    )
    summary = report["methods"]["missing_meta_h1"]
    assert not summary["method_input_complete"]
    assert summary["performance"]["strict_all_five_success"]["value"] is None
    assert "unavailable" in summary["performance"]["strict_all_five_success"]["reason"]
    paired = report["paired_vs_default"]["missing_meta_h1"]
    assert paired["paired_win_loss_tie"]["value"] is None
    assert report["episodes"]["missing_meta_h1"][0]["fallback_count"] >= 1


def test_supervisor_recovery_skips_meta_policy_and_reports_explicit_limitation(
    valid_input, valid_output
) -> None:
    calls = []

    def should_not_run(context):
        calls.append(context)
        return default_output(valid_output)

    record = synthetic_record(valid_input)
    record["model_visible_input"]["measurement_validity"] = {
        "state": "requires_recovery",
        "supervisor_diagnosis": "sensor_saturation",
        "measurement_policy": "lower_exposure_reacquire",
    }
    report = PairedClosedLoopEvaluator(
        backend=SyntheticBackend(),
        methods=(
            MethodSpec("default_h1", "off"),
            MethodSpec("meta_h1", "guarded", policy=CallablePolicy(should_not_run)),
        ),
        bootstrap_samples=8,
    ).evaluate([record], require_preregistered_cardinality=False)
    assert calls == []
    episode = report["episodes"]["meta_h1"][0]
    assert episode["reobserve_count"] == 1
    assert episode["termination_reason"] == "reobserve_requested_no_recovery_backend"


def test_setup_bootstrap_is_deterministic_and_uses_setup_means() -> None:
    values = {"setup-a": [1.0, 1.0, 1.0], "setup-b": [-1.0, -1.0, -1.0]}
    first = setup_bootstrap_interval(values, seed=7, samples=64)
    second = setup_bootstrap_interval(values, seed=7, samples=64)
    assert first == second
    assert first["value"]["point_estimate"] == 0.0
    assert first["value"]["unit"] == "setup"


def test_method_subset_reports_can_merge_after_default_reproduction(
    valid_input, valid_output
) -> None:
    policy = CallablePolicy(
        lambda context: MetaPolicyResult(
            payload=default_output(valid_output), latency_seconds=0.01
        )
    )
    record = synthetic_record(valid_input)
    first = PairedClosedLoopEvaluator(
        backend=SyntheticBackend(),
        methods=(
            MethodSpec("default_h1", "off"),
            MethodSpec("method_a", "guarded", policy=policy),
        ),
        bootstrap_samples=8,
    ).evaluate([record], require_preregistered_cardinality=False)
    second = PairedClosedLoopEvaluator(
        backend=SyntheticBackend(),
        methods=(
            MethodSpec("default_h1", "off"),
            MethodSpec("method_b", "guarded", policy=policy),
        ),
        bootstrap_samples=8,
    ).evaluate([record], require_preregistered_cardinality=False)
    merged = merge_closed_loop_reports([first, second])
    assert merged["method_order"] == ["default_h1", "method_a", "method_b"]
    assert merged["merged_partial_report_count"] == 2


def test_paired_wlt_and_null_reason_are_explicit() -> None:
    baseline = [
        {
            "record_id": "a",
            "setup_id": "s",
            "method_input_complete": True,
            "strict_all_five_success": False,
            "steps_to_success": None,
            "final_normalized_error": 2.0,
            "actual_canonical_objective_reduction": 0.0,
        }
    ]
    better = [
        {
            **baseline[0],
            "method_input_complete": True,
            "strict_all_five_success": True,
            "steps_to_success": 2,
            "final_normalized_error": 0.8,
            "actual_canonical_objective_reduction": 1.0,
        }
    ]
    compared = paired_comparison(baseline, better, bootstrap_samples=16)
    assert compared["paired_win_loss_tie"]["value"]["wins"] == 1

    incomplete = [{**better[0], "method_input_complete": False}]
    missing = paired_comparison(baseline, incomplete, bootstrap_samples=16)
    assert missing["paired_win_loss_tie"]["value"] is None
    assert missing["paired_win_loss_tie"]["reason"]


def test_h3_is_impossible_in_method_and_evaluator() -> None:
    config = default_h1_config()
    config["horizon"] = 3
    with pytest.raises(Exception):
        MethodSpec("bad_h3", "off", planner_config=config)
    with pytest.raises(ValueError, match="H1-only"):
        MethodSpec("another_h3", "off")
