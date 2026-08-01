from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from control_rebuild_v4 import audit_completion
from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v4.closed_loop import (
    NumericalClosedLoopControllerV4,
    target_reached,
)
from control_rebuild_v4.evaluate_orchestrated_system import (
    RecordingRuntime,
    physical_forward_direction_metrics,
)
from control_rebuild_v4.inverse_data import (
    derived_inverse_pairs,
    paired_error_indices,
    physically_valid_states,
    state_mapping,
)
from control_rebuild_v4.models import anchored_forward_residual_model
from control_rebuild_v4.orchestrated_runtime import (
    ROUTE_ARTIFACTS,
    ROUTE_BACKENDS,
    OrchestratedSpecialistRuntimeV4,
    _action_index,
)
from control_rebuild_v4.selection_gates import (
    gate_aware_selection_key,
    gate_margin_summary,
)
from control_rebuild_v4.seed_measurement_prediction_cache import validate_cache
from control_rebuild_v4.summarize_results import failure_focused_validation_block
from control_rebuild_v4.visual_runtime import VisualInversePipelineV4
from Qwen_orchestration.runtime.errors import ContractError


def synthetic_group() -> dict:
    current = np.asarray([511.5, 511.5, 30.0, 28.0, 100.0])
    candidates = []
    for index, action in enumerate(ACTION_GRID):
        state = current.copy()
        state[0] += 0.25 * float(action["lens_x_delta_mm"]) / 0.05
        state[1] += 0.25 * float(action["lens_y_delta_mm"]) / 0.05
        state[2] += 0.4 * float(action["camera_x_delta_mm"]) / 0.02
        state[3] += 0.4 * float(action["camera_y_delta_mm"]) / 0.02
        state[4] += float(index - 40) * 0.01
        candidates.append(
            {
                "action": action,
                "next_state": state_mapping(state),
                "change": state_mapping(state - current),
            }
        )
    return {
        "group_id": "synthetic",
        "setup": {},
        "current_beam_state": state_mapping(current),
        "candidates": candidates,
    }


def test_gate_aware_checkpoint_selection_prioritizes_registered_gates() -> None:
    four_passed = gate_margin_summary(
        {"overall": 0.02, "boundary": 0.01, "nonlinear": 0.01, "iid": 0.01}
    )
    three_passed = gate_margin_summary(
        {"overall": 0.20, "boundary": 0.20, "nonlinear": -0.001, "iid": 0.20}
    )
    assert gate_aware_selection_key(four_passed, 0.4) > gate_aware_selection_key(
        three_passed,
        0.9,
    )

    weaker_worst_margin = gate_margin_summary(
        {"overall": 0.02, "boundary": 0.001, "nonlinear": 0.03, "iid": 0.02}
    )
    assert gate_aware_selection_key(four_passed, 0.4) > gate_aware_selection_key(
        weaker_worst_margin,
        0.9,
    )


def test_measurement_prediction_cache_requires_exact_finite_records(tmp_path) -> None:
    path = tmp_path / "measurement_predictions_train.npz"
    keys = [["state-1", "clean"], ["state-1", "noise"]]
    np.savez(
        path,
        keys_json=json.dumps(keys),
        predictions=np.ones((2, 5), dtype=np.float32),
    )
    evidence = validate_cache(path, keys)
    assert evidence["key_count"] == 2
    assert evidence["prediction_shape"] == [2, 5]
    assert evidence["finite"] is True

    np.savez(
        path,
        keys_json=json.dumps(keys),
        predictions=np.full((2, 5), np.nan, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="non-finite"):
        validate_cache(path, keys)


def test_derived_pairs_include_reachable_and_guaranteed_infeasible() -> None:
    pairs = derived_inverse_pairs([synthetic_group()])
    assert len(pairs) == 6
    assert sum(bool(pair["matching_indices"]) for pair in pairs) == 4
    assert sum(not pair["matching_indices"] for pair in pairs) == 2
    assert all(
        pair["selected_index"] is None for pair in pairs if not pair["matching_indices"]
    )


def test_anchored_network_is_exactly_zero_for_zero_action() -> None:
    import torch

    torch.manual_seed(3)
    model = anchored_forward_residual_model(torch, 47)
    features = torch.randn(2, 81, 47)
    zero = features[:, 40, :].clone()
    output = model(features, zero)
    assert torch.equal(output[:, 40, :], torch.zeros(2, 5))


def test_physical_state_floor() -> None:
    values = np.asarray([[1.0, 2.0, -3.0, 0.0, -1.0]])
    result = physically_valid_states(values)
    assert np.all(result[:, 2:4] >= 0.25)
    assert np.all(result[:, 4] > 0.0)


def test_measurement_errors_preserve_paired_condition() -> None:
    bank_conditions = np.asarray([0, 0, 1, 1, 2, 2])
    requested = np.asarray([2, 0, 1, 2, 1, 0])
    indices = paired_error_indices(bank_conditions, requested, np.random.default_rng(9))
    assert np.array_equal(bank_conditions[indices], requested)


def test_closed_loop_replans_and_stops_on_true_target() -> None:
    desired = np.asarray([2.0, 3.0, 4.0, 5.0, 10.0], dtype=np.float32)

    class Forward:
        def predict_states(self, rows):
            return np.repeat(desired[None, None, :], len(rows) * 81, axis=0).reshape(
                len(rows), 81, 5
            )

    class Inverse:
        def score_requests(self, setups, current, desired_state, candidates):
            return {
                "selected_indices": np.asarray([0]),
                "selected_actions": [ACTION_GRID[0]],
                "predicted_statuses": ["unique"],
                "scores": np.zeros((1, 81), dtype=np.float32),
            }

    controller = NumericalClosedLoopControllerV4(Forward(), Inverse())

    def executor(setup, action):
        return dict(setup), desired.copy()

    result = controller.run(
        {},
        np.asarray([0.0, 0.0, 4.0, 5.0, 10.0], dtype=np.float32),
        desired,
        executor,
        max_steps=3,
    )
    assert result["final_target_reached"]
    assert result["executed_steps"] == 1
    assert result["stop_reason"] == "target_reached"
    assert target_reached(desired, desired)


def test_orchestration_overlay_preserves_fixed_action_grid() -> None:
    assert _action_index(ACTION_GRID[0]) == 0
    assert _action_index(ACTION_GRID[40]) == 40
    assert _action_index(ACTION_GRID[80]) == 80
    invalid = dict(ACTION_GRID[40])
    invalid["camera_x_delta_mm"] = 0.01
    with pytest.raises(ContractError, match="81-action grid"):
        _action_index(invalid)


def test_orchestration_overlay_preserves_nonready_contracts() -> None:
    runtime = OrchestratedSpecialistRuntimeV4.__new__(OrchestratedSpecialistRuntimeV4)
    clarification = runtime.dispatch(
        {
            "schema_version": "qwen_orchestration_decision_v1",
            "status": "needs_clarification",
            "task_type": "inverse_control",
            "route_name": None,
            "arguments": {},
            "image_roles": {},
            "missing_fields": ["desired_beam_state"],
            "clarification_question": "What desired beam state should be reached?",
        }
    )
    assert clarification == {
        "status": "needs_clarification",
        "missing_fields": ["desired_beam_state"],
        "clarification_question": "What desired beam state should be reached?",
        "executed": False,
    }
    unsupported = runtime.dispatch(
        {
            "schema_version": "qwen_orchestration_decision_v1",
            "status": "unsupported",
            "task_type": None,
            "route_name": None,
            "arguments": {},
            "image_roles": {},
            "missing_fields": [],
            "clarification_question": None,
        }
    )
    assert unsupported["status"] == "unsupported"
    assert unsupported["executed"] is False
    assert unsupported["supported_task_types"] == [
        "beam_profile_measurement",
        "direction_prediction",
        "forward_prediction",
        "inverse_control",
    ]


def test_direct_image_uses_analytic_fallback_outside_learned_gamma(
    tmp_path,
) -> None:
    from PIL import Image

    image_path = tmp_path / "out_of_range_gamma.png"
    Image.fromarray(np.full((16, 16), 255, dtype=np.uint8)).save(image_path)
    pipeline = VisualInversePipelineV4.__new__(VisualInversePipelineV4)
    measured = pipeline.measure_image(
        image_path,
        {
            "gamma": 0.5,
            "linear_intensity_high": 10.0,
            "linear_intensity_low": 2.0,
            "source_sensor_resolution_px": [16, 16],
        },
    )
    assert measured["measurement_source"] == "calibrated_analytic_moments"
    assert measured["v3_beam_state"] is None
    assert measured["beam_state"][4] == pytest.approx(10.0)
    assert measured["simulator_at_inference"] is False


def test_candidate_manifest_rejects_modified_artifact(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"verified payload")
    valid_entry = {
        "path": str(artifact),
        "size": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    artifact_keys = (
        "measurement_v3",
        "measurement_calibrator_v4",
        "forward_v4",
        "inverse_v4",
        "visual_scorer_v4",
        "direction_v1",
    )
    entries = {key: dict(valid_entry) for key in artifact_keys}
    entries["forward_v4"]["sha256"] = "0" * 64
    manifest = {
        "manifest_version": "qwen_to_specialists_v4_candidate_manifest",
        "complete": True,
        "seed": 20260726,
        "action_grid_size": 81,
        "simulator_at_inference": False,
        "direct_measurement_policy": {
            "learned_gamma_min": 0.75,
            "learned_gamma_max": 1.05,
        },
        "frozen_qwen": {
            "registry": dict(valid_entry),
            "decision_schema": dict(valid_entry),
        },
        "artifacts": entries,
        "routes": {
            route: {
                "backend": ROUTE_BACKENDS[route],
                "artifacts": list(ROUTE_ARTIFACTS[route]),
            }
            for route in ROUTE_BACKENDS
        },
    }
    manifest_path = tmp_path / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError, match="digest mismatch: forward_v4"):
        OrchestratedSpecialistRuntimeV4.from_manifest(
            None,
            manifest_path,
            None,
        )


def test_completion_audit_requires_exact_manifest_pins_and_routes(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    measurement_v3_run = tmp_path / "measurement_v3"
    measurement_v4_run = tmp_path / "measurement_v4"
    run_dir.mkdir()
    measurement_v3_run.mkdir()
    measurement_v4_run.mkdir()
    paths = {
        "measurement_v3": measurement_v3_run / "measurement_v3.pt",
        "measurement_calibrator_v4": (
            measurement_v4_run / "measurement_calibrator_v4.pt"
        ),
        "forward_v4": run_dir / "forward_physics_residual_v4.pt",
        "inverse_v4": run_dir / "inverse_control_v4.pt",
        "visual_scorer_v4": run_dir / "visual_sensor_scorer_v4_integrated.pt",
        "direction_v1": tmp_path / "direction.pkl",
    }
    registry = tmp_path / "registry.yaml"
    schema = tmp_path / "schema.json"
    for index, path in enumerate([*paths.values(), registry, schema]):
        path.write_bytes(f"artifact-{index}".encode())

    def pin(path):
        return {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    monkeypatch.setattr(
        audit_completion,
        "DEFAULT_MEASUREMENT_V3_RUN",
        measurement_v3_run,
    )
    monkeypatch.setattr(
        audit_completion,
        "DIRECTION_ARTIFACT",
        paths["direction_v1"],
    )
    monkeypatch.setattr(audit_completion, "QWEN_REGISTRY", registry)
    monkeypatch.setattr(audit_completion, "QWEN_SCHEMA", schema)
    manifest = {
        "artifacts": {key: pin(path) for key, path in paths.items()},
        "frozen_qwen": {
            "registry": pin(registry),
            "decision_schema": pin(schema),
        },
        "routes": {
            route: {
                "backend": ROUTE_BACKENDS[route],
                "artifacts": list(ROUTE_ARTIFACTS[route]),
            }
            for route in ROUTE_BACKENDS
        },
    }
    assert (
        audit_completion.candidate_manifest_contract_failures(
            manifest,
            run_dir,
            measurement_v4_run,
        )
        == []
    )

    missing_artifact = json.loads(json.dumps(manifest))
    del missing_artifact["artifacts"]["direction_v1"]
    failures = audit_completion.candidate_manifest_contract_failures(
        missing_artifact,
        run_dir,
        measurement_v4_run,
    )
    assert "artifacts: exact six-key set differs" in failures
    assert "artifacts.direction_v1: pin is not an object" in failures

    wrong_route = json.loads(json.dumps(manifest))
    wrong_route["routes"]["predict_forward_from_state_v1"]["backend"] = "wrong"
    assert (
        "routes: exact route, backend, or artifact mapping differs"
        in audit_completion.candidate_manifest_contract_failures(
            wrong_route,
            run_dir,
            measurement_v4_run,
        )
    )


def test_completion_audit_requires_real_success_and_failure_examples() -> None:
    examples = {
        "definitions": {
            task: f"{task} metric definition"
            for task in ("forward", "inverse", "measurement", "visual_inverse")
        },
        "tasks": {
            "forward": {
                "success": {
                    "source_split": "test_iid",
                    "group_id": "forward-success",
                    "strict_all_five_success": True,
                },
                "failure": {
                    "source_split": "test_ood_physics",
                    "group_id": "forward-failure",
                    "strict_all_five_success": False,
                },
            },
            "inverse": {
                "success": {
                    "source_split": "test_iid",
                    "request_id": "inverse-success",
                    "physical_target_reached": True,
                },
                "failure": {
                    "source_split": "test_ood_physics",
                    "request_id": "inverse-failure",
                    "physical_target_reached": False,
                },
            },
            "measurement": {
                "success": {
                    "source_split": "test_visual_stress",
                    "state_id": "measurement-success",
                    "strict_all_five_success": True,
                },
                "failure": {
                    "source_split": "test_visual_stress",
                    "state_id": "measurement-failure",
                    "strict_all_five_success": False,
                },
            },
            "visual_inverse": {
                "success": {
                    "source_split": "test_visual_stress",
                    "group_id": "visual-success",
                    "physical_target_reached": True,
                },
                "failure": {
                    "source_split": "test_visual_stress",
                    "group_id": "visual-failure",
                    "physical_target_reached": False,
                },
            },
        },
    }
    expected_splits = {"test_iid", "test_ood_physics", "test_visual_stress"}
    assert (
        audit_completion.real_example_contract_failures(examples, expected_splits) == []
    )

    examples["tasks"]["visual_inverse"]["failure"] = None
    examples["tasks"]["forward"]["success"]["strict_all_five_success"] = False
    failures = audit_completion.real_example_contract_failures(
        examples,
        expected_splits,
    )
    assert "visual_inverse.failure: record is missing" in failures
    assert "forward.success: metric outcome differs" in failures


def test_completion_audit_checksum_contract_has_exact_unique_coverage(
    tmp_path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"value": 1}\n', encoding="utf-8")
    second.write_text('{"value": 2}\n', encoding="utf-8")

    def checksum_line(path):
        return (
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(tmp_path)}"
        )

    checksum_path = tmp_path / "checksums.sha256"
    checksum_path.write_text(
        f"{checksum_line(first)}\n{checksum_line(second)}\n",
        encoding="utf-8",
    )
    evidence = audit_completion.checksum_contract_evidence(tmp_path)
    assert evidence["failures"] == []
    assert evidence["line_count"] == 2
    assert evidence["unique_target_count"] == 2
    assert evidence["expected_file_count"] == 2

    checksum_path.write_text(
        f"{checksum_line(first)}\n{checksum_line(first)}\n",
        encoding="utf-8",
    )
    evidence = audit_completion.checksum_contract_evidence(tmp_path)
    assert evidence["duplicate_targets"] == ["first.json"]
    assert evidence["missing_targets"] == ["second.json"]


def test_completion_audit_requires_complete_unique_system_details(
    tmp_path,
) -> None:
    base = {
        "executed": True,
        "dispatch_error": None,
    }
    rows = [
        {
            **base,
            "example_id": "direction-1",
            "target_status": "ready",
            "target_route": "predict_direction_from_state_v1",
            "predicted_status": "ready",
            "predicted_route": "predict_direction_from_state_v1",
            "physical_metric": "all_five_directions_exact",
            "physical_success": True,
            "correctly_routed_specialist_physical_success": True,
        },
        {
            **base,
            "example_id": "forward-1",
            "target_status": "ready",
            "target_route": "predict_forward_from_image_v1",
            "predicted_status": "ready",
            "predicted_route": "predict_forward_from_image_v1",
            "physical_metric": "all_five_numerical_changes_within_tolerance",
            "physical_success": False,
            "correctly_routed_specialist_physical_success": True,
        },
        {
            **base,
            "example_id": "clarification-1",
            "target_status": "needs_clarification",
            "target_route": None,
            "predicted_status": "needs_clarification",
            "predicted_route": None,
            "executed": False,
        },
        {
            **base,
            "example_id": "unsupported-invalid-prediction-1",
            "target_status": "unsupported",
            "target_route": None,
            "predicted_status": "needs_attention",
            "predicted_route": None,
            "executed": False,
            "dispatch_error": "ContractError: schema violation at status",
        },
    ]
    report = {
        "record_count": 4,
        "task_ready_counts": {
            "direction_prediction": 1,
            "forward_prediction": 1,
        },
        "metrics": {
            "target_ready_count": 2,
            "successful_valid_execution_rate": 1.0,
            "direction_physical_count": 1,
            "forward_physical_count": 1,
        },
    }
    path = tmp_path / "details.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    evidence = audit_completion.orchestrated_details_contract_evidence(
        path,
        report,
    )
    assert evidence["failures"] == []
    assert evidence["unique_example_id_count"] == 4
    assert evidence["invalid_predicted_status_count"] == 1
    assert evidence["predicted_status_counts"]["<invalid>"] == 1

    rows[1]["example_id"] = "direction-1"
    del rows[0]["physical_success"]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    failures = audit_completion.orchestrated_details_contract_evidence(
        path,
        report,
    )["failures"]
    assert any("duplicate details example IDs" in value for value in failures)
    assert "details row 0: direction outcomes are missing" in failures

    rows[1]["example_id"] = "forward-1"
    rows[0]["physical_success"] = True
    rows[3]["dispatch_error"] = None
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    failures = audit_completion.orchestrated_details_contract_evidence(
        path,
        report,
    )["failures"]
    assert (
        "details row 3: invalid predicted status lacks a dispatch error"
        in failures
    )

    rows[3]["dispatch_error"] = "ContractError: schema violation at status"
    rows[3]["executed"] = True
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    failures = audit_completion.orchestrated_details_contract_evidence(
        path,
        report,
    )["failures"]
    assert "details row 3: invalid predicted status was executed" in failures


def test_completion_audit_requires_all_registered_validation_gates() -> None:
    gate_count = audit_completion.EXPECTED_VALIDATION_GATE_COUNT
    decision = {
        "decision_rule_version": ("control_rebuild_v4_preregistered_validation_gates"),
        "thresholds": dict(audit_completion.VALIDATION_THRESHOLDS),
        "gates": [
            {
                "label": f"gate-{index}",
                "comparison": ">=",
                "observed": 1.0,
                "threshold": 0.5,
                "passed": True,
            }
            for index in range(gate_count)
        ],
        "gate_count": gate_count,
        "passed_gate_count": gate_count,
        "failed_gates": [],
    }
    assert audit_completion.validation_decision_contract_failures(decision) == []

    decision["gates"] = []
    decision["gate_count"] = 0
    decision["passed_gate_count"] = 0
    failures = audit_completion.validation_decision_contract_failures(decision)
    assert "gate record count differs" in failures
    assert "declared gate count differs" in failures
    assert "declared passed-gate count differs" in failures


def test_failure_focused_validation_is_reported_separately(tmp_path) -> None:
    forward = {
        "validation": {
            "validation_expanded": {
                "groups": 450,
                "forward": {
                    "count": 36450,
                    "strict_all_five_success": 0.4,
                    "mae_in_tolerance_units": 0.7,
                },
                "inverse_selection": {
                    "request_count": 1350,
                    "target_success": 0.5,
                },
                "forward_cost_only": {
                    "pair_count": 2700,
                    "reachable_pair_count": 1800,
                    "target_success_feasible": 0.6,
                },
            }
        }
    }
    inverse_metrics = {
        "pair_count": 2700,
        "reachable_pair_count": 1800,
        "target_success_feasible": 0.6,
        "status_accuracy": 0.7,
    }
    inverse = {
        "validation": {
            "metrics_at_selected_alpha": {
                "expanded_clean": inverse_metrics,
                "expanded_measurement_augmented": {
                    **inverse_metrics,
                    "target_success_feasible": 0.55,
                },
            }
        }
    }
    (tmp_path / "forward_physics_residual_v4_summary.json").write_text(
        json.dumps(forward),
        encoding="utf-8",
    )
    (tmp_path / "inverse_control_v4_summary.json").write_text(
        json.dumps(inverse),
        encoding="utf-8",
    )
    comparison = {
        "scope": "same 450 selection-validation groups",
        "comparison": {
            "all": {
                "forward_strict_all_five": {
                    "v3": 0.3,
                    "v4": 0.4,
                    "absolute_delta": 0.1,
                }
            },
            "by_category": {},
        },
    }
    (tmp_path / "v3_v4_selection_validation_comparison.json").write_text(
        json.dumps(comparison),
        encoding="utf-8",
    )
    result = failure_focused_validation_block(tmp_path)
    assert result is not None
    assert result["group_count"] == 450
    assert result["scope"].startswith("450-group")
    assert result["forward"]["transition_count"] == 36450
    assert result["forward_only_target_retrieval"]["pair_count"] == 2700
    assert result["forward_only_target_retrieval"]["request_count"] == 1800
    assert result["forward_only_target_retrieval"]["target_success"] == pytest.approx(
        0.6
    )
    assert (
        result["forward_only_target_retrieval"]["metric_source"]
        == "exact_six_request_physical_retrieval"
    )
    assert result["inverse_clean"]["reachable_pair_count"] == 1800
    assert result["inverse_paired_measurement_errors"][
        "target_success_feasible"
    ] == pytest.approx(0.55)
    assert result["same_distribution_v3_v4"]["comparison"]["all"][
        "forward_strict_all_five"
    ]["absolute_delta"] == pytest.approx(0.1)
    assert result["held_out_used_for_training_or_selection"] == 0


def test_combined_forward_metric_uses_physical_truth() -> None:
    action = dict(ACTION_GRID[0])
    current = {
        "centroid_x_px": 0.0,
        "centroid_y_px": 0.0,
        "sigma_x_px": 10.0,
        "sigma_y_px": 10.0,
        "peak_intensity": 100.0,
    }
    true_change = {
        "centroid_x_px": 0.5,
        "centroid_y_px": -0.5,
        "sigma_x_px": 0.5,
        "sigma_y_px": -0.5,
        "peak_intensity": 1.0,
    }
    true_directions = {
        "centroid_x": "no_change",
        "centroid_y": "no_change",
        "width_x": "no_change",
        "width_y": "no_change",
        "peak_intensity": "no_change",
    }
    correct_result = {
        "change": true_change,
        "directions": true_directions,
    }
    wrong_forward = {
        **correct_result,
        "change": {**true_change, "centroid_x_px": 3.0},
    }

    class FakeRuntime:
        def dispatch(self, decision, available_images=None):
            return {
                "executed": True,
                "result": decision["mock_result"],
            }

    target_direction = {
        "status": "ready",
        "task_type": "direction_prediction",
        "route_name": "predict_direction_from_state_v1",
        "arguments": {"action": action},
        "mock_result": correct_result,
    }
    predicted_direction = {"mock_result": correct_result}
    target_forward = {
        "status": "ready",
        "task_type": "forward_prediction",
        "route_name": "predict_forward_from_state_v1",
        "arguments": {"action": action},
        "mock_result": correct_result,
    }
    predicted_forward = {"mock_result": wrong_forward}
    canonical = [
        {
            "example_id": "direction",
            "group_id": "group",
            "target_decision": target_direction,
        },
        {
            "example_id": "forward",
            "group_id": "group",
            "target_decision": target_forward,
        },
    ]
    predictions = [
        {"example_id": "direction", "parsed_json": predicted_direction},
        {"example_id": "forward", "parsed_json": predicted_forward},
    ]
    private = [
        {
            "group_id": "group",
            "setup": {},
            "current_beam_state": current,
        }
    ]
    recording = RecordingRuntime(FakeRuntime())
    for target, predicted in (
        (target_direction, predicted_direction),
        (target_forward, predicted_forward),
    ):
        recording.dispatch(target)
        recording.dispatch(predicted)
    metrics, details = physical_forward_direction_metrics(
        canonical,
        predictions,
        private,
        recording,
        truth_function=lambda private_row, selected_action: {
            "change": true_change,
            "directions": true_directions,
        },
    )
    assert metrics["private_forward_ground_truth_simulator_calls"] == 1
    assert metrics["end_to_end_direction_physical_all_five_exact"] == 1.0
    assert all(
        field["accuracy"] == 1.0
        for field in metrics[
            "end_to_end_direction_physical_per_field"
        ].values()
    )
    assert metrics["end_to_end_forward_physical_strict_all_five_success"] == 0.0
    assert metrics["correctly_routed_forward_physical_strict_all_five_success"] == 1.0
    assert details["forward"]["physical_success"] is False
