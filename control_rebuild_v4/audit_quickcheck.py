#!/usr/bin/env python3
"""Audit the validation-only 12-hour quickcheck without requiring it to pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Qwen_orchestration.scripts.verify_frozen_baseline import (
    DEFAULT_MANIFEST as QWEN_FREEZE_MANIFEST,
)
from Qwen_orchestration.scripts.verify_frozen_baseline import verify as verify_qwen
from control_rebuild_v4.assess_validation import THRESHOLDS
from control_rebuild_v4.audit_completion import (
    Audit,
    candidate_manifest_contract_failures,
    checksum_contract_evidence,
    orchestrated_details_contract_evidence,
    rates_in_range,
    real_example_contract_failures,
)
from control_rebuild_v4.evaluate_closed_loop import aggregate as aggregate_closed_loop
from control_rebuild_v4.validate_orchestrated_runtime import ROUTES

DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/control_rebuild_v4_quickcheck"
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_quickcheck_12h"
DEFAULT_MEASUREMENT_RUN = REPO_ROOT.parent / "VLM_runs/measurement_rebuild_v4_one_seed"
DEFAULT_MEASUREMENT_DATA = REPO_ROOT.parent / "VLM_data/measurement_rebuild_v3"
EXPECTED_TRAIN_GROUPS = 2000
EXPECTED_VAL_GROUPS = 300
EXPECTED_CHECKSUM_FILES = EXPECTED_TRAIN_GROUPS + EXPECTED_VAL_GROUPS + 3
EXPECTED_IMAGE_CONDITIONS = [
    "clean",
    "noise",
    "blur",
    "dim_noise",
    "saturation",
    "crop_boundary",
    "gamma_shift",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--measurement-v4-run",
        type=Path,
        default=DEFAULT_MEASUREMENT_RUN,
    )
    parser.add_argument(
        "--measurement-data",
        type=Path,
        default=DEFAULT_MEASUREMENT_DATA,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    measurement_run = args.measurement_v4_run.resolve()
    measurement_data = args.measurement_data.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else run_dir / "quickcheck_audit.json"
    )
    audit = Audit()
    qwen_failures = verify_qwen(QWEN_FREEZE_MANIFEST)
    audit.require(
        "frozen Qwen baseline integrity",
        not qwen_failures,
        {"failures": qwen_failures},
    )

    required = [
        data_dir / "manifest.json",
        data_dir / "checksums.sha256",
        run_dir / "forward_physics_residual_v4.pt",
        run_dir / "forward_physics_residual_v4_summary.json",
        run_dir / "inverse_control_v4.pt",
        run_dir / "inverse_control_v4_summary.json",
        run_dir / "measurement_prediction_cache_provenance.json",
        run_dir / "v3_v4_selection_validation_comparison.json",
        run_dir / "visual_sensor_scorer_v4_integrated.pt",
        run_dir / "visual_sensor_scorer_v4_integrated_summary.json",
        run_dir / "candidate_overlay_manifest.json",
        run_dir / "controlled_validation.json",
        run_dir / "closed_loop_val.json",
        run_dir / "orchestrated_runtime_validation.json",
        run_dir / "orchestrated_system_validation.json",
        run_dir / "orchestrated_system_validation.details.jsonl",
        run_dir / "quickcheck_decision.json",
        run_dir / "validation_performance_summary.json",
        run_dir / "VALIDATION_REPORT.md",
        run_dir / "validation_examples.json",
        run_dir / "VALIDATION_EXAMPLES.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    audit.require("all quickcheck artifacts exist", not missing, {"missing": missing})
    if missing:
        result = {
            "audit_version": "control_rebuild_v4_quickcheck_audit",
            "complete": False,
            "checks": audit.checks,
            "failure_count": len(audit.failures),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(1)

    manifest = read_json(data_dir / "manifest.json")
    verification = manifest.get("verification", {})
    splits = verification.get("splits", {})
    train = splits.get("train", {})
    val = splits.get("val", {})
    audit.require(
        "quickcheck dataset has exact unique split sizes",
        verification.get("complete") is True
        and train.get("groups") == EXPECTED_TRAIN_GROUPS
        and train.get("transitions") == EXPECTED_TRAIN_GROUPS * 81
        and val.get("groups") == EXPECTED_VAL_GROUPS
        and val.get("transitions") == EXPECTED_VAL_GROUPS * 81
        and sum(train.get("category_counts", {}).values()) == EXPECTED_TRAIN_GROUPS
        and sum(val.get("category_counts", {}).values()) == EXPECTED_VAL_GROUPS
        and set(train.get("category_counts", {}))
        == {"iid_expanded", "ood_boundary", "high_nonlinearity"}
        and set(val.get("category_counts", {}))
        == {"iid_expanded", "ood_boundary", "high_nonlinearity"},
        {"train": train, "val": val},
    )
    audit.require(
        "quickcheck dataset preserves split and physics invariants",
        verification.get("cross_split_group_overlap") == 0
        and verification.get("candidate_count_per_group") == 81
        and verification.get("finite_numerical_values") is True
        and verification.get("action_grid_order_exact") is True
        and verification.get("held_out_v2_test_files_opened") is False,
        verification,
    )
    checksum = checksum_contract_evidence(data_dir)
    checksum_manifest = manifest.get("checksum_contract", {})
    audit.require(
        "quickcheck checksum coverage is exact",
        not checksum["failures"]
        and checksum["line_count"] == EXPECTED_CHECKSUM_FILES
        and checksum["unique_target_count"] == EXPECTED_CHECKSUM_FILES
        and checksum["expected_file_count"] == EXPECTED_CHECKSUM_FILES
        and checksum_manifest.get("file_count") == EXPECTED_CHECKSUM_FILES
        and checksum_manifest.get("checksum_manifest_sha256")
        == checksum["checksum_manifest_sha256"],
        checksum,
    )

    import torch

    forward_artifact = torch.load(
        run_dir / "forward_physics_residual_v4.pt",
        map_location="cpu",
        weights_only=False,
    )
    forward_summary = read_json(run_dir / "forward_physics_residual_v4_summary.json")
    inverse_artifact = torch.load(
        run_dir / "inverse_control_v4.pt",
        map_location="cpu",
        weights_only=False,
    )
    inverse_summary = read_json(run_dir / "inverse_control_v4_summary.json")
    inverse_training = inverse_summary.get("training", {})
    forward_validation = forward_summary.get("validation", {})
    forward_selection_gates = forward_validation.get(
        "checkpoint_selection_gates",
        {},
    )
    expected_forward_selection_gates = {
        "same_distribution_overall_forward_delta",
        "same_distribution_overall_forward_retrieval_delta",
        "ood_boundary_forward_delta",
        "high_nonlinearity_forward_delta",
        "old_iid_forward_floor",
    }
    audit.require(
        "forward quickcheck uses 2,000 unique difficult groups without repetition",
        forward_artifact.get("zero_action_exact") is True
        and forward_artifact.get("checkpoint_selection")
        == "validation_only_gate_count_then_worst_margin_then_composite"
        and forward_summary.get("v4_unique_train_groups") == EXPECTED_TRAIN_GROUPS
        and forward_summary.get("v4_train_repeat") == 1
        and forward_summary.get("unique_train_groups") == 4500
        and forward_summary.get("train_groups") == 4500
        and forward_summary.get("train_transitions") == 364500
        and forward_summary.get("epochs") == 36
        and forward_summary.get("parameter_count") == 983813
        and forward_summary.get("best_selection_key")
        == forward_validation.get("checkpoint_selection_key")
        and forward_selection_gates.get("gate_count") == 5
        and set(forward_selection_gates.get("margins", {}))
        == expected_forward_selection_gates
        and forward_summary.get("v3_selection_validation_reference", {})
        .get("all", {})
        .get("groups")
        == EXPECTED_VAL_GROUPS
        and forward_summary.get("held_out_test_used") is False,
        {
            "v4_unique_train_groups": forward_summary.get("v4_unique_train_groups"),
            "repeat": forward_summary.get("v4_train_repeat"),
            "unique_train_groups": forward_summary.get("unique_train_groups"),
            "effective_train_groups": forward_summary.get("train_groups"),
            "train_transitions": forward_summary.get("train_transitions"),
            "epochs": forward_summary.get("epochs"),
            "parameter_count": forward_summary.get("parameter_count"),
            "checkpoint_selection": forward_artifact.get("checkpoint_selection"),
            "best_selection_key": forward_summary.get("best_selection_key"),
            "selection_gates": forward_selection_gates,
        },
    )
    inverse_validation = inverse_summary.get("validation", {})
    inverse_selection_gates = inverse_validation.get(
        "checkpoint_selection_gates",
        {},
    )
    expected_inverse_selection_gates = {
        "same_distribution_overall_inverse_delta",
        "same_distribution_inverse_status_delta",
        "ood_boundary_inverse_delta",
        "high_nonlinearity_inverse_delta",
        "old_iid_inverse_floor",
        "paired_error_retention_floor",
    }
    audit.require(
        "inverse quickcheck uses 12,000 unique derived requests without repetition",
        inverse_artifact.get("v4_train_repeat") == 1
        and inverse_artifact.get("checkpoint_selection")
        == "validation_only_gate_count_then_worst_margin_then_composite"
        and inverse_training.get("v2_pairs") == 60000
        and inverse_training.get("v4_unique_derived_pairs") == 12000
        and inverse_training.get("v4_derived_pairs") == 12000
        and inverse_training.get("total_unique_pairs") == 72000
        and inverse_training.get("total_pairs") == 72000
        and inverse_summary.get("epochs") == 20
        and inverse_summary.get("parameter_count") == 385284
        and inverse_summary.get("validation_pair_counts")
        == {"iid": 900, "expanded": 1800}
        and inverse_summary.get("best_selection_key")
        == inverse_validation.get("checkpoint_selection_key")
        and inverse_selection_gates.get("gate_count") == 6
        and set(inverse_selection_gates.get("margins", {}))
        == expected_inverse_selection_gates
        and inverse_summary.get("v3_selection_validation_reference", {})
        .get("all", {})
        .get("group_count")
        == EXPECTED_VAL_GROUPS
        and inverse_summary.get("held_out_test_used") is False,
        {
            "training": inverse_training,
            "checkpoint_selection": inverse_artifact.get("checkpoint_selection"),
            "best_selection_key": inverse_summary.get("best_selection_key"),
            "selection_gates": inverse_selection_gates,
            "epochs": inverse_summary.get("epochs"),
            "parameter_count": inverse_summary.get("parameter_count"),
            "validation_pair_counts": inverse_summary.get("validation_pair_counts"),
        },
    )
    cache_provenance = read_json(
        run_dir / "measurement_prediction_cache_provenance.json"
    )
    cache_failures = []
    for split, expected_keys in (("train", 70000), ("val", 8400)):
        block = cache_provenance.get("splits", {}).get(split, {})
        source = block.get("source", {})
        destination = block.get("destination", {})
        source_path = Path(source.get("path", ""))
        destination_path = Path(destination.get("path", ""))
        for label, pin, path in (
            ("source", source, source_path),
            ("destination", destination, destination_path),
        ):
            if not path.is_file():
                cache_failures.append(f"{split}.{label}: file is missing")
                continue
            if pin.get("sha256") != sha256(path):
                cache_failures.append(f"{split}.{label}: digest differs")
            if pin.get("size") != path.stat().st_size:
                cache_failures.append(f"{split}.{label}: size differs")
            if pin.get("key_count") != expected_keys:
                cache_failures.append(f"{split}.{label}: key count differs")
            if pin.get("prediction_shape") != [expected_keys, 5]:
                cache_failures.append(f"{split}.{label}: shape differs")
            if pin.get("finite") is not True:
                cache_failures.append(f"{split}.{label}: finite flag differs")
        if source.get("sha256") != destination.get("sha256"):
            cache_failures.append(f"{split}: source and destination digests differ")
    audit.require(
        "visual training reuses the exact frozen train and validation measurements",
        cache_provenance.get("complete") is True
        and cache_provenance.get("conditions") == EXPECTED_IMAGE_CONDITIONS
        and cache_provenance.get("calibration_applied_in_cache") is False
        and cache_provenance.get("held_out_test_used") is False
        and not cache_failures,
        {"failures": cache_failures, "splits": cache_provenance.get("splits")},
    )
    visual_summary = read_json(
        run_dir / "visual_sensor_scorer_v4_integrated_summary.json"
    )
    visual_artifact = torch.load(
        run_dir / "visual_sensor_scorer_v4_integrated.pt",
        map_location="cpu",
        weights_only=False,
    )
    measurement_config = read_json(measurement_data / "config.json")
    with (measurement_data / "states/train.jsonl").open(
        "r",
        encoding="utf-8",
    ) as stream:
        first_measurement_state = json.loads(next(stream))
    image_calibration = first_measurement_state.get("image_calibration", {})
    visual_validation = visual_summary.get("validation", {})
    visual_selection_gates = visual_validation.get(
        "checkpoint_selection_gates",
        {},
    )
    visual_selected_metrics = visual_validation.get("selected_alpha", {})
    expected_visual_margin = (
        float(visual_selected_metrics.get("target_success_feasible", float("-inf")))
        - THRESHOLDS["visual_inverse_floor"]
    )
    audit.require(
        "visual quickcheck keeps the complete image data and resolution",
        visual_summary.get("train_groups") == 2500
        and visual_summary.get("train_pairs") == 52500
        and visual_summary.get("val_groups") == 300
        and visual_summary.get("val_pairs") == 6300
        and visual_summary.get("epochs") == 16
        and visual_summary.get("parameter_count") == 385284
        and visual_summary.get("conditions") == EXPECTED_IMAGE_CONDITIONS
        and visual_summary.get("held_out_test_used") is False
        and measurement_config.get("stored_resolution_px") == 512
        and measurement_config.get("conditions") == EXPECTED_IMAGE_CONDITIONS
        and image_calibration.get("source_sensor_resolution_px") == [1024, 1024]
        and image_calibration.get("stored_resolution_px") == [512, 512]
        and visual_artifact.get("coordinate_frame") == "camera_sensor_array"
        and len(visual_artifact.get("action_grid", [])) == 81
        and visual_artifact.get("checkpoint_selection")
        == "validation_only_physical_target_floor_then_margin_then_composite"
        and visual_summary.get("best_selection_key")
        == visual_validation.get("checkpoint_selection_key")
        and visual_selection_gates.get("gate_count") == 1
        and set(visual_selection_gates.get("margins", {}))
        == {"visual_inverse_physical_target_floor"}
        and abs(
            float(
                visual_selection_gates.get("margins", {}).get(
                    "visual_inverse_physical_target_floor",
                    float("inf"),
                )
            )
            - expected_visual_margin
        )
        <= 1e-12
        and Path(visual_artifact.get("forward_artifact", "")).resolve()
        == (run_dir / "forward_physics_residual_v4.pt").resolve()
        and Path(visual_artifact.get("inverse_initialization", "")).resolve()
        == (run_dir / "inverse_control_v4.pt").resolve(),
        {
            "train_groups": visual_summary.get("train_groups"),
            "train_pairs": visual_summary.get("train_pairs"),
            "val_groups": visual_summary.get("val_groups"),
            "val_pairs": visual_summary.get("val_pairs"),
            "epochs": visual_summary.get("epochs"),
            "parameter_count": visual_summary.get("parameter_count"),
            "conditions": visual_summary.get("conditions"),
            "stored_resolution_px": measurement_config.get("stored_resolution_px"),
            "source_sensor_resolution_px": image_calibration.get(
                "source_sensor_resolution_px"
            ),
            "artifact_stored_resolution_px": image_calibration.get(
                "stored_resolution_px"
            ),
            "checkpoint_selection": visual_artifact.get("checkpoint_selection"),
            "best_selection_key": visual_summary.get("best_selection_key"),
            "selection_gates": visual_selection_gates,
        },
    )

    comparison = read_json(run_dir / "v3_v4_selection_validation_comparison.json")
    audit.require(
        "v3 and v4 use the same 300 quickcheck validation groups",
        comparison.get("complete") is True
        and comparison.get("models", {}).get("v3", {}).get("all", {}).get("group_count")
        == EXPECTED_VAL_GROUPS
        and comparison.get("models", {})
        .get("v4", {})
        .get("all", {})
        .get("transition_count")
        == EXPECTED_VAL_GROUPS * 81
        and comparison.get("category_counts") == val.get("category_counts")
        and comparison.get("held_out_used_for_training_or_selection") == 0,
        {
            "category_counts": comparison.get("category_counts"),
            "v3": comparison.get("models", {}).get("v3", {}).get("all", {}),
            "v4": comparison.get("models", {}).get("v4", {}).get("all", {}),
        },
    )

    candidate = read_json(run_dir / "candidate_overlay_manifest.json")
    candidate_failures = candidate_manifest_contract_failures(
        candidate,
        run_dir,
        measurement_run,
    )
    audit.require(
        "quickcheck candidate manifest pins exact artifacts and routes",
        candidate.get("complete") is True
        and not candidate_failures
        and candidate.get("held_out_test_used_for_training_or_selection") == 0,
        {"failures": candidate_failures},
    )

    controlled = read_json(run_dir / "controlled_validation.json")
    loop = read_json(run_dir / "closed_loop_val.json")
    overlay = read_json(run_dir / "orchestrated_runtime_validation.json")
    system = read_json(run_dir / "orchestrated_system_validation.json")
    system_metrics = system.get("metrics", {})
    controlled_val = controlled.get("split_results", {}).get("val", {})
    controlled_numerical = controlled_val.get("numerical", {})
    controlled_inverse = controlled_numerical.get("inverse", {})
    controlled_measurement = controlled_val.get("measurement", {})
    controlled_visual = controlled_val.get("visual_inverse", {})
    numerical_inverse_blocks = [
        controlled_inverse.get(name, {})
        for name in (
            "learned_residual_corrected",
            "forward_cost_only",
            "true_candidate_oracle",
        )
    ]
    visual_metric_blocks = [
        controlled_visual.get(name, {})
        for name in ("oracle_measurement", "model_measurement")
    ]
    audit.require(
        "controlled specialist validation is complete and test-isolated",
        controlled.get("complete") is True
        and controlled.get("splits_requested") == ["val"]
        and controlled.get("conditions") == EXPECTED_IMAGE_CONDITIONS
        and set(controlled.get("split_results", {})) == {"val"}
        and controlled.get("held_out_examples_used_for_training_or_selection") == 0
        and controlled.get("held_out_state_examples_evaluated") == 0
        and controlled_numerical.get("group_count") == 300
        and controlled_numerical.get("forward", {}).get("count") == 24300
        and all(block.get("pair_count") == 900 for block in numerical_inverse_blocks)
        and all(
            block.get("reachable_pair_count") == 600
            for block in numerical_inverse_blocks
        )
        and controlled_measurement.get("state_count") == 1200
        and controlled_measurement.get("view_count") == 8400
        and controlled_measurement.get("all_conditions", {}).get("count") == 8400
        and set(controlled_measurement.get("by_condition", {}))
        == set(EXPECTED_IMAGE_CONDITIONS)
        and all(
            block.get("count") == 1200
            for block in controlled_measurement.get("by_condition", {}).values()
        )
        and set(controlled_visual.get("by_condition", {}))
        == set(EXPECTED_IMAGE_CONDITIONS)
        and all(block.get("request_count") == 6300 for block in visual_metric_blocks)
        and all(
            block.get("reachable_request_count") == 4200
            for block in visual_metric_blocks
        )
        and sum(controlled_visual.get("request_status_counts", {}).values()) == 6300,
        {
            "complete": controlled.get("complete"),
            "splits": sorted(controlled.get("split_results", {})),
            "numerical_group_count": controlled_numerical.get("group_count"),
            "forward_transition_count": controlled_numerical.get("forward", {}).get(
                "count"
            ),
            "inverse_pair_counts": [
                block.get("pair_count") for block in numerical_inverse_blocks
            ],
            "measurement_state_count": controlled_measurement.get("state_count"),
            "measurement_view_count": controlled_measurement.get("view_count"),
            "visual_request_counts": [
                block.get("request_count") for block in visual_metric_blocks
            ],
        },
    )
    loop_records = loop.get("records", [])
    try:
        recomputed_loop_metrics = aggregate_closed_loop(loop_records, 3)
    except (KeyError, TypeError, ValueError):
        recomputed_loop_metrics = None
    loop_request_ids = [
        str(record.get("request_id"))
        for record in loop_records
        if isinstance(record, dict)
    ]
    audit.require(
        "quickcheck closed loop contains 50 three-step validation requests",
        loop.get("complete") is True
        and loop.get("split") == "val"
        and isinstance(loop_records, list)
        and len(loop_records) == 50
        and len(set(loop_request_ids)) == 50
        and all(identifier.startswith("val:") for identifier in loop_request_ids)
        and all(
            isinstance(record, dict)
            and 0 <= int(record.get("source_target_index", -1)) < 81
            and int(record.get("source_target_index", -1)) != 40
            and 0 <= int(record.get("executed_steps", -1)) <= 3
            and len(record.get("trace", [])) <= 3
            for record in loop_records
        )
        and loop.get("metrics", {}).get("request_count") == 50
        and loop.get("metrics", {}).get("max_steps") == 3
        and loop.get("metrics") == recomputed_loop_metrics
        and loop.get("held_out_test_used_for_training_or_selection") == 0
        and loop.get("held_out_requests_evaluated") == 0,
        {
            "metrics": loop.get("metrics", {}),
            "recomputed_metrics": recomputed_loop_metrics,
            "record_count": (
                len(loop_records) if isinstance(loop_records, list) else None
            ),
            "unique_request_count": len(set(loop_request_ids)),
        },
    )
    overlay_routes = overlay.get("route_results", [])
    overlay_route_names = [
        str(row.get("route_name")) for row in overlay_routes if isinstance(row, dict)
    ]
    direct_measurement = overlay.get("direct_measurement_validation", {})
    audit.require(
        "all routes and direct images execute without an inference simulator",
        overlay.get("complete") is True
        and overlay.get("routes_expected") == 7
        and overlay.get("routes_passed") == 7
        and overlay.get("all_routes_passed") is True
        and overlay.get("simulator_inference_calls") == 0
        and len(overlay_routes) == 7
        and set(overlay_route_names) == set(ROUTES)
        and len(set(overlay_route_names)) == 7
        and all(
            row.get("passed") is True and row.get("simulator_at_inference") is False
            for row in overlay_routes
        )
        and direct_measurement.get("count") == 150
        and sum(direct_measurement.get("measurement_source_counts", {}).values())
        == 150,
        {
            "routes_passed": overlay.get("routes_passed"),
            "route_names": overlay_route_names,
            "direct_count": direct_measurement.get("count"),
            "measurement_source_counts": direct_measurement.get(
                "measurement_source_counts"
            ),
            "simulator_inference_calls": overlay.get("simulator_inference_calls"),
        },
    )
    expected_task_ready_counts = {
        "beam_profile_measurement": 150,
        "direction_prediction": 300,
        "forward_prediction": 300,
        "inverse_control": 300,
    }
    audit.require(
        "all 1,600 saved Qwen decisions execute through the quickcheck overlay",
        system.get("complete") is True
        and system.get("record_count") == 1600
        and system.get("task_ready_counts") == expected_task_ready_counts
        and system.get("max_per_category") is None
        and system_metrics.get("target_ready_count") == 1050
        and system_metrics.get("direction_physical_count") == 300
        and system_metrics.get("forward_physical_count") == 300
        and system_metrics.get("simulator_calls_during_inference_count") == 0
        and system_metrics.get("private_simulator_scoring_calls") == 600
        and system_metrics.get("private_forward_ground_truth_simulator_calls") == 150
        and system_metrics.get("private_simulator_scoring_calls_total") == 750,
        {
            "record_count": system.get("record_count"),
            "task_ready_counts": system.get("task_ready_counts"),
            "target_ready_count": system_metrics.get("target_ready_count"),
            "direction_physical_count": system_metrics.get("direction_physical_count"),
            "forward_physical_count": system_metrics.get("forward_physical_count"),
            "private_scoring_calls": system_metrics.get(
                "private_simulator_scoring_calls_total"
            ),
        },
    )
    details = orchestrated_details_contract_evidence(
        run_dir / "orchestrated_system_validation.details.jsonl",
        system,
    )
    audit.require(
        "combined-system details reconcile all 1,600 Qwen records",
        not details["failures"]
        and details["record_count"] == 1600
        and details["unique_example_id_count"] == 1600,
        details,
    )

    decision = read_json(run_dir / "quickcheck_decision.json")
    gates = decision.get("gates", [])
    passed_labels = [row.get("label") for row in gates if row.get("passed") is True]
    failed_labels = [row.get("label") for row in gates if row.get("passed") is not True]
    expected_authorization = not failed_labels
    audit.require(
        "quickcheck applies all 26 unchanged bars without held-out authorization",
        decision.get("complete") is True
        and decision.get("decision_rule_version")
        == "control_rebuild_v4_quickcheck_same_registered_performance_bars"
        and decision.get("source_gate_rule")
        == "control_rebuild_v4_preregistered_validation_gates"
        and decision.get("thresholds") == THRESHOLDS
        and decision.get("unique_train_groups") == EXPECTED_TRAIN_GROUPS
        and decision.get("unique_validation_groups") == EXPECTED_VAL_GROUPS
        and decision.get("training_repeat") == 1
        and len(gates) == 26
        and len({row.get("label") for row in gates}) == 26
        and decision.get("gate_count") == 26
        and decision.get("passed_gate_count") == len(passed_labels)
        and decision.get("failed_gates") == failed_labels
        and decision.get("proceed_to_server_training") is expected_authorization
        and decision.get("held_out_evaluation_opened") is False
        and "proceed_to_heldout" not in decision,
        {
            "gate_count": len(gates),
            "passed_gate_count": len(passed_labels),
            "failed_gates": failed_labels,
            "proceed_to_server_training": decision.get("proceed_to_server_training"),
        },
    )

    performance_summary = read_json(run_dir / "validation_performance_summary.json")
    reported_decision = performance_summary.get("validation_decision", {})
    failure_focused = performance_summary.get(
        "failure_focused_selection_validation",
        {},
    )
    reported_retrieval = failure_focused.get(
        "forward_only_target_retrieval",
        {},
    )
    reported_inverse_clean = failure_focused.get("inverse_clean", {})
    reported_inverse_noisy = failure_focused.get(
        "inverse_paired_measurement_errors",
        {},
    )
    report_text = (run_dir / "VALIDATION_REPORT.md").read_text(encoding="utf-8")
    audit.require(
        "quickcheck report uses exact validation denominators and decision scope",
        performance_summary.get("phase") == "validation"
        and performance_summary.get("evidence_scope", {}).get(
            "held_out_used_for_training_or_selection"
        )
        == 0
        and reported_decision.get("authorization_target") == "server_training"
        and reported_decision.get("authorized") is expected_authorization
        and reported_decision.get("proceed_to_heldout") is False
        and reported_decision.get("proceed_to_server_training")
        is expected_authorization
        and failure_focused.get("group_count") == EXPECTED_VAL_GROUPS
        and failure_focused.get("category_counts") == val.get("category_counts")
        and failure_focused.get("forward", {}).get("transition_count")
        == EXPECTED_VAL_GROUPS * 81
        and reported_retrieval.get("metric_source")
        == "exact_six_request_physical_retrieval"
        and reported_retrieval.get("pair_count") == EXPECTED_VAL_GROUPS * 6
        and reported_retrieval.get("request_count") == EXPECTED_VAL_GROUPS * 4
        and reported_inverse_clean.get("pair_count") == EXPECTED_VAL_GROUPS * 6
        and reported_inverse_clean.get("reachable_pair_count")
        == EXPECTED_VAL_GROUPS * 4
        and reported_inverse_noisy.get("pair_count") == EXPECTED_VAL_GROUPS * 6
        and reported_inverse_noisy.get("reachable_pair_count")
        == EXPECTED_VAL_GROUPS * 4
        and failure_focused.get("held_out_used_for_training_or_selection") == 0
        and f"The {EXPECTED_VAL_GROUPS} setups contain" in report_text
        and "The 450 setups contain" not in report_text
        and "No held-out test was opened." in report_text,
        {
            "phase": performance_summary.get("phase"),
            "decision": reported_decision,
            "group_count": failure_focused.get("group_count"),
            "category_counts": failure_focused.get("category_counts"),
            "forward": failure_focused.get("forward"),
            "forward_retrieval": reported_retrieval,
            "inverse_clean": reported_inverse_clean,
            "inverse_paired_measurement_errors": reported_inverse_noisy,
        },
    )

    examples = read_json(run_dir / "validation_examples.json")
    example_failures = real_example_contract_failures(examples, {"val"})
    audit.require(
        "quickcheck reports eight real validation examples",
        examples.get("phase") == "validation" and not example_failures,
        {"failures": example_failures},
    )
    held_out_outputs = [
        run_dir / "controlled_evaluation.json",
        run_dir / "final_performance_summary.json",
        run_dir / "FINAL_REPORT.md",
        run_dir / "final_evaluation.complete",
    ]
    opened = [str(path) for path in held_out_outputs if path.exists()]
    audit.require(
        "quickcheck opened no held-out evaluation artifact",
        not opened,
        {"unexpected": opened},
    )
    rate_failures = [
        *rates_in_range(controlled, "controlled_validation"),
        *rates_in_range(loop, "closed_loop_val"),
        *rates_in_range(system, "orchestrated_system_validation"),
    ]
    audit.require(
        "all quickcheck primary rates are finite and inside zero to one",
        not rate_failures,
        {"failures": rate_failures},
    )

    stage_dir = run_dir / "quickcheck_stages"
    required_markers = [
        "verify_frozen_qwen.complete",
        "seed_measurement_prediction_cache.complete",
        "finalize_numerical_dataset.complete",
        "train_forward_v4.complete",
        "train_inverse_v4.complete",
        "compare_v3_v4_selection_validation.complete",
        "train_visual_integrated_v4.complete",
        "candidate_overlay_manifest.complete",
        "controlled_validation.complete",
        "closed_loop_val.complete",
        "orchestrated_runtime_validation.complete",
        "orchestrated_system_validation.complete",
        "assess_quickcheck.complete",
        "summarize_quickcheck.complete",
    ]
    missing_markers = [
        name for name in required_markers if not (stage_dir / name).is_file()
    ]
    audit.require(
        "all restart-safe quickcheck stages completed",
        not missing_markers,
        {"missing": missing_markers},
    )

    result = {
        "audit_version": "control_rebuild_v4_quickcheck_audit",
        "complete": not audit.failures,
        "performance_bar_passed": bool(decision.get("proceed_to_server_training")),
        "checks": audit.checks,
        "check_count": len(audit.checks),
        "failure_count": len(audit.failures),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
