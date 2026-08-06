from __future__ import annotations

import json

from control_rebuild_v4.assess_quickcheck import quickcheck_decision
from control_rebuild_v4.assess_validation import assess_validation


def write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def passing_fixture(tmp_path) -> dict:
    metric = lambda delta: {
        "v3": 0.30,
        "v4": 0.30 + delta,
        "absolute_delta": delta,
    }
    comparison = {
        "complete": True,
        "models": {"v4": {"all": {"group_count": 450}}},
        "comparison": {
            "all": {
                "forward_strict_all_five": metric(0.02),
                "forward_only_target_success_feasible": metric(0.02),
                "inverse_target_success_feasible": metric(0.02),
                "inverse_status_accuracy": metric(0.0),
            },
            "by_category": {
                category: {
                    "forward_strict_all_five": metric(0.01),
                    "inverse_target_success_feasible": metric(0.01),
                }
                for category in ("ood_boundary", "high_nonlinearity")
            },
        },
    }
    controlled = {
        "complete": True,
        "split_results": {
            "val": {
                "numerical": {
                    "forward": {"strict_all_five_success": 0.32},
                    "inverse": {
                        "learned_residual_corrected": {"target_success_feasible": 0.40}
                    },
                },
                "measurement": {"all_conditions": {"strict_all_five_success": 0.68}},
                "visual_inverse": {
                    "model_measurement": {"physical_target_success_feasible": 0.45}
                },
            }
        },
    }
    inverse = {
        "validation": {
            "metrics_at_selected_alpha": {
                "expanded_clean": {"target_success_feasible": 0.50},
                "expanded_measurement_augmented": {"target_success_feasible": 0.40},
            }
        }
    }
    closed_loop = {
        "complete": True,
        "metrics": {"reached_by_step": {"3": {"rate": 0.50}}},
    }
    overlay = {
        "complete": True,
        "routes_passed": 7,
        "simulator_inference_calls": 0,
        "direct_measurement_validation": {"strict_all_five_success": 0.75},
    }
    system = {
        "complete": True,
        "metrics": {
            "successful_valid_execution_rate": 0.98,
            "physical_loss_versus_correctly_routed_specialist_by_task": {
                task: 0.01
                for task in ("direction", "forward", "measurement", "inverse")
            },
            "simulator_calls_during_inference_count": 0,
        },
    }
    files = {
        "v3_v4_selection_validation_comparison.json": comparison,
        "controlled_validation.json": controlled,
        "inverse_control_v4_summary.json": inverse,
        "closed_loop_val.json": closed_loop,
        "orchestrated_runtime_validation.json": overlay,
        "orchestrated_system_validation.json": system,
    }
    for name, value in files.items():
        write_json(tmp_path / name, value)
    return comparison


def test_validation_gate_passes_only_when_all_preregistered_checks_pass(
    tmp_path,
) -> None:
    comparison = passing_fixture(tmp_path)
    passed = assess_validation(tmp_path)
    assert passed["proceed_to_heldout"] is True
    assert passed["passed_gate_count"] == passed["gate_count"]
    assert passed["held_out_evaluation_opened"] is False

    comparison["comparison"]["all"]["forward_strict_all_five"]["absolute_delta"] = 0.0
    write_json(
        tmp_path / "v3_v4_selection_validation_comparison.json",
        comparison,
    )
    failed = assess_validation(tmp_path)
    assert failed["proceed_to_heldout"] is False
    assert failed["failed_gates"] == ["same-distribution forward all-five improvement"]


def test_validation_gate_accepts_registered_quickcheck_group_count(tmp_path) -> None:
    comparison = passing_fixture(tmp_path)
    comparison["models"]["v4"]["all"]["group_count"] = 300
    write_json(
        tmp_path / "v3_v4_selection_validation_comparison.json",
        comparison,
    )
    result = assess_validation(tmp_path, expected_comparison_groups=300)
    assert result["proceed_to_heldout"] is True
    assert result["gate_count"] == 26
    assert (
        result["gates"][1]["label"]
        == "comparison uses all 300 difficult validation groups"
    )


def test_quickcheck_decision_never_authorizes_heldout() -> None:
    result = quickcheck_decision(
        {
            "proceed_to_heldout": True,
            "gate_count": 26,
            "passed_gate_count": 26,
            "failed_gates": [],
            "held_out_evaluation_opened": False,
            "complete": True,
        },
        unique_train_groups=2000,
        unique_validation_groups=300,
    )
    assert "proceed_to_heldout" not in result
    assert result["proceed_to_server_training"] is True
    assert result["held_out_evaluation_opened"] is False
    assert result["training_repeat"] == 1
