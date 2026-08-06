#!/usr/bin/env python3
"""Build 60 candidate-only matched initial-state groups from legal source cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    Bounds,
    metrics_dict,
    metrics_vector,
    position_dict,
    position_vector,
    setup_hash,
    stable_seed,
    tolerance_vector,
)
from continuous_control_v12.simulator import simulate_state
from vlm_optics_benchmark.visual_anomalies import canonical_patch

from .core import (
    FIXED_CONTROLLER_CONFIG,
    PLAN_NAMES,
    apply_sensor_family,
    PLAN_BANK_REVISION,
    build_plan_bank,
    canonical_hash,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_SUITE = REPOSITORY_ROOT / "supervisor_v1_1_candidate/suites/full_suite.json"
DEFAULT_SOURCE_TRAIN = REPOSITORY_ROOT / "supervisor_v1_1_candidate/suites/train_suite.json"
DEFAULT_SOURCE_DEV = REPOSITORY_ROOT / "supervisor_v1_1_candidate/suites/dev_suite.json"
DEFAULT_V12_CONFIG = REPOSITORY_ROOT / "continuous_control_v12/config_v12_semantics_v2.json"
DEFAULT_BASE_CONFIG = REPOSITORY_ROOT / "optical_sim/configs/base_config.yaml"
DEFAULT_FORWARD_CHECKPOINT = REPOSITORY_ROOT / "runs/overnight_v12_semantics_20260731_002709/models/lc_128g_v2/continuous_forward_v12_128g.pt"
DEFAULT_LOCKED_H1 = REPOSITORY_ROOT / "runs/v12_mpc_h1_h3_diagnosis_20260731_111829/configs/locked_primary_config.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts/qwen_reasoning_plan_selector"
ROOT_SEED = 2026080401


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _selected_cases(
    full_suite: Mapping[str, Any], train_ids: set[str], dev_ids: set[str],
    *, include_supplemental: bool = False,
) -> list[tuple[dict[str, Any], str]]:
    by_stratum: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"train": [], "dev": []}
    )
    for raw in full_suite["cases"]:
        case = dict(raw)
        case_id = str(case["case_id"])
        source_split = "train" if case_id in train_ids else "dev" if case_id in dev_ids else None
        if source_split is None:
            raise ValueError(f"source case lacks candidate split: {case_id}")
        by_stratum[str(case["stratum"])][source_split].append(case)
    selected: list[tuple[dict[str, Any], str]] = []
    train_candidates: list[dict[str, Any]] = []
    confirmation_candidates: list[dict[str, Any]] = []
    supplemental_candidates: list[dict[str, Any]] = []
    for stratum in sorted(by_stratum):
        source_train = sorted(by_stratum[stratum]["train"], key=lambda row: row["case_id"])
        source_dev = sorted(by_stratum[stratum]["dev"], key=lambda row: row["case_id"])
        if len(source_train) < 3 or len(source_dev) < 1:
            raise ValueError(f"insufficient candidate-only source cases in {stratum}")
        train_candidates.extend(source_train[:3])
        confirmation_candidates.append(source_dev[0])
        supplemental_candidates.extend(source_train[3:6])
    train_candidates = sorted(train_candidates, key=lambda row: stable_seed(ROOT_SEED, row["case_id"], "split"))
    for index, case in enumerate(train_candidates):
        selected.append((case, "candidate_train" if index < 7 else "candidate_dev"))
    selected.extend((case, "candidate_confirmation") for case in confirmation_candidates)
    if include_supplemental:
        supplemental_candidates = sorted(
            supplemental_candidates,
            key=lambda row: stable_seed(ROOT_SEED, row["case_id"], "supplemental_split"),
        )
        selected.extend(
            (case, "candidate_train" if index < 7 else "candidate_dev")
            for index, case in enumerate(supplemental_candidates)
        )
    expected = 21 if include_supplemental else 12
    if len(selected) != expected:
        raise RuntimeError(f"exactly {expected} base states are required")
    return selected


def _shape_target(
    case: Mapping[str, Any],
    capture: Mapping[str, Any],
    *,
    bounds: Bounds,
    base_config: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select a physical target by simulator-only coverage scoring, never plan outcome."""

    initial_position = position_vector(case["initial_positions_mm"])
    initial_metrics = metrics_vector(capture["metrics"])
    action_high = bounds.action_high
    patterns = []
    lens_scales = (2.0, 3.0, 4.0)
    for axis in (0, 1):
        outward = 1.0 if initial_position[axis] >= 0.0 else -1.0
        for scale in lens_scales:
            for sign in (outward, -outward):
                delta = np.zeros(4, dtype=np.float64)
                delta[axis] = sign * scale * action_high[axis]
                camera_axis = axis + 2
                delta[camera_axis] = sign * min(4.0 * action_high[camera_axis], scale * action_high[axis])
                patterns.append(delta)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            delta = np.asarray(
                [3.0 * sx * action_high[0], 3.0 * sy * action_high[1], 3.0 * sx * action_high[2], 3.0 * sy * action_high[3]],
                dtype=np.float64,
            )
            patterns.append(delta)
    candidates = []
    for index, delta in enumerate(patterns):
        goal = np.clip(initial_position + delta, bounds.position_low, bounds.position_high)
        target_capture = simulate_state(
            case["setup_context"],
            position_dict(goal),
            case["simulator_fixed"],
            str(base_config.resolve()),
            bounds,
        )
        target = metrics_vector(target_capture["metrics"])
        components = np.abs(target - initial_metrics) / tolerance_vector(initial_metrics)
        centroid = float(components[:2].max())
        shape = float(components[2:].max())
        distance = float(components.max())
        score = shape - 0.20 * centroid
        candidates.append(
            {
                "index": index,
                "goal": goal,
                "target": target,
                "normalized_components": components,
                "centroid_error": centroid,
                "shape_error": shape,
                "distance": distance,
                "coverage_score": score,
            }
        )
    eligible = [row for row in candidates if row["distance"] >= 1.25]
    if not eligible:
        raise RuntimeError(f"no nontrivial physical target for {case['case_id']}")
    selected = max(
        eligible,
        key=lambda row: (row["coverage_score"], row["shape_error"], row["distance"], -row["index"]),
    )
    return (
        np.asarray(selected["goal"], dtype=np.float64),
        np.asarray(selected["target"], dtype=np.float64),
        {
            "selection": "simulator_only_shape_coverage_no_plan_rollouts",
            "candidate_count": len(candidates),
            "selected_index": int(selected["index"]),
            "selected_normalized_components": selected["normalized_components"].tolist(),
            "selected_centroid_error": float(selected["centroid_error"]),
            "selected_shape_error": float(selected["shape_error"]),
            "selected_distance": float(selected["distance"]),
            "selected_coverage_score": float(selected["coverage_score"]),
        },
    )


def _bounds_payload(bounds: Bounds) -> dict[str, list[float]]:
    return {
        "action_low": bounds.action_low.tolist(),
        "action_high": bounds.action_high.tolist(),
        "position_low": bounds.position_low.tolist(),
        "position_high": bounds.position_high.tolist(),
    }


def _boundary_bounds(case: Mapping[str, Any], global_bounds: Bounds) -> tuple[Bounds, dict[str, Any]]:
    current = position_vector(case["initial_positions_mm"])
    goal = position_vector(case["q_goal_mm"])
    delta = goal - current
    axis_order = np.argsort(np.abs(delta) / global_bounds.action_high)
    axis = int(axis_order[0])
    low = global_bounds.position_low.copy()
    high = global_bounds.position_high.copy()
    near_margin = 0.10 * global_bounds.action_high[axis]
    if delta[axis] >= 0.0:
        low[axis] = current[axis] - near_margin
    else:
        high[axis] = current[axis] + near_margin
    local = Bounds(
        action_low=global_bounds.action_low.copy(),
        action_high=global_bounds.action_high.copy(),
        position_low=low,
        position_high=high,
    )
    local.validate()
    if np.any(current < low) or np.any(current > high) or np.any(goal < low) or np.any(goal > high):
        raise RuntimeError("boundary intervention made initial or goal position illegal")
    return local, {
        "changed_axis": ACTION_FIELDS[axis],
        "near_boundary_side": "lower" if delta[axis] >= 0.0 else "upper",
        "initial_margin_mm": near_margin,
        "q_goal_delta_mm": delta.tolist(),
        "same_initial_state_and_target_as_nominal": True,
    }


def _save_sensor_png(image: np.ndarray, path: Path) -> None:
    patch = canonical_patch(np.asarray(image, dtype=np.float64), size=256)
    quantized = np.rint(np.clip(patch, 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(quantized, mode="L").save(path)


def build(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    output = args.output.resolve()
    existing_rollout = output / "rollout_results.jsonl"
    if existing_rollout.exists() and not args.include_supplemental:
        raise FileExistsError("refusing to replace an existing formal rollout")
    if existing_rollout.exists() and args.include_supplemental:
        prior_rows = sum(1 for line in existing_rollout.open(encoding="utf-8") if line.strip())
        prior_config = json.loads((output / "rollout_config.json").read_text())
        prior_confirmation = output / "split_manifests/confirmation_frozen.json"
        if prior_rows != 900:
            raise RuntimeError("supplemental generation requires the complete 900-episode revision1 run")
        if prior_config.get("plan_bank_revision") != PLAN_BANK_REVISION or prior_config.get("supplemental_sampling_triggered"):
            raise RuntimeError("supplemental generation may run exactly once on revision1")
        if sha256_path(prior_confirmation) != "003458efeb2b5e4e81553a825b03ba93070b5aa7aedbea9b65f31c1ae9bdb867":
            raise RuntimeError("confirmation manifest differs before supplemental generation")
    output.mkdir(parents=True, exist_ok=True)
    full_suite = json.loads(args.source_suite.resolve().read_text())
    train_suite = json.loads(args.source_train.resolve().read_text())
    dev_suite = json.loads(args.source_dev.resolve().read_text())
    train_ids = {str(row["case_id"]) for row in train_suite["cases"]}
    dev_ids = {str(row["case_id"]) for row in dev_suite["cases"]}
    v12_config = json.loads(args.v12_config.resolve().read_text())
    global_bounds = Bounds.from_config(v12_config)
    selected = _selected_cases(
        full_suite, train_ids, dev_ids,
        include_supplemental=bool(args.include_supplemental),
    )
    groups: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    source_registry = []
    for base_index, (case, split) in enumerate(selected):
        base_state_id = f"qrps_base_{base_index:03d}"
        capture = simulate_state(
            case["setup_context"],
            case["initial_positions_mm"],
            case["simulator_fixed"],
            str(args.base_config.resolve()),
            global_bounds,
        )
        simulated_initial = metrics_vector(capture["metrics"])
        serialized_initial = metrics_vector(case["initial_metrics"])
        replay_error = float(np.max(np.abs(simulated_initial - serialized_initial)))
        if replay_error > 1e-6 * max(1.0, float(np.max(np.abs(serialized_initial)))):
            raise RuntimeError(f"source initial replay mismatch for {case['case_id']}: {replay_error}")
        cache_path = output / "initial_captures" / f"{base_state_id}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            intensity=np.asarray(capture["intensity"], dtype=np.float32),
            metrics=simulated_initial,
        )
        shape_goal, shape_target, target_audit = _shape_target(
            case, capture, bounds=global_bounds, base_config=args.base_config
        )
        local_bounds, boundary_audit = _boundary_bounds(case, global_bounds)
        anomaly_seed = stable_seed(ROOT_SEED, base_state_id, "anomaly")
        rng = np.random.default_rng(anomaly_seed)
        reflection = {
            "k": float(rng.choice((1.75, 2.25, 2.75))),
            "amplitude": float(rng.choice((0.30, 0.40, 0.50))),
            "width_ratio": float(rng.choice((0.85, 1.0, 1.15))),
            "angle_radians": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
        saturation = {
            "clip_level_fraction_of_peak": float(rng.choice((0.35, 0.50, 0.65)))
        }
        common = {
            "base_state_id": base_state_id,
            "source_case_id": str(case["case_id"]),
            "source_candidate_split": "candidate_train" if str(case["case_id"]) in train_ids else "candidate_dev",
            "split": split,
            "setup_hash": setup_hash(case["setup_context"], case["simulator_fixed"]),
            "setup_context": case["setup_context"],
            "simulator_fixed": case["simulator_fixed"],
            "initial_positions_mm": case["initial_positions_mm"],
            "initial_clean_metrics": metrics_dict(simulated_initial),
            "tolerance_reference_metrics": metrics_dict(simulated_initial),
            "stratum": str(case["stratum"]),
            "regime": str(case["regime"]),
            "initial_capture_cache": str(cache_path.relative_to(output)),
            "initial_capture_cache_sha256": sha256_path(cache_path),
            "frozen_or_protected": False,
        }
        variants = [
            ("nominal", "nominal", {}, metrics_vector(case["target_metrics"]), global_bounds, "base"),
            ("reflection", "width_relative_reflection", reflection, metrics_vector(case["target_metrics"]), global_bounds, "image_family"),
            ("saturation", "sensor_saturation", saturation, metrics_vector(case["target_metrics"]), global_bounds, "image_family"),
            ("target_swap", "nominal", {}, shape_target, global_bounds, "target"),
            ("boundary_swap", "nominal", {}, metrics_vector(case["target_metrics"]), local_bounds, "legal_boundary"),
        ]
        variant_ids = {}
        for variant_name, family, anomaly, target, bounds, intervention_type in variants:
            group_id = f"{base_state_id}__{variant_name}"
            variant_ids[variant_name] = group_id
            sensor_image, sensor_metadata = apply_sensor_family(
                np.asarray(capture["intensity"]), family, anomaly
            )
            image_path = output / "images" / f"{group_id}.png"
            _save_sensor_png(sensor_image, image_path)
            group = {
                **common,
                "group_id": group_id,
                "variant": variant_name,
                "visual_family": family,
                "anomaly": anomaly,
                "intervention_type": intervention_type,
                "target_metrics": metrics_dict(target),
                "q_goal_mm_evaluator_only": (
                    position_dict(shape_goal) if variant_name == "target_swap" else case["q_goal_mm"]
                ),
                "q_goal_visible_to_selector": False,
                "bounds": _bounds_payload(bounds),
                "initial_sensor_image": str(image_path.relative_to(output)),
                "initial_sensor_image_sha256": sha256_path(image_path),
                "sensor_generation_evidence": sensor_metadata,
                "target_generation_audit": target_audit if variant_name == "target_swap" else {
                    "selection": "source_candidate_physical_q_goal_replay",
                    "source_exact_q_goal_replay_max_abs_metric_error": float(case["exact_q_goal_replay_max_abs_metric_error"]),
                },
                "boundary_intervention_audit": boundary_audit if variant_name == "boundary_swap" else None,
            }
            groups.append(group)
        interventions.extend(
            [
                {
                    "pair_id": f"{base_state_id}__target",
                    "type": "same_image_and_state_change_target",
                    "left_group_id": variant_ids["nominal"],
                    "right_group_id": variant_ids["target_swap"],
                    "physical_consistency": "target regenerated by corrected simulator at a legal q_goal",
                },
                {
                    "pair_id": f"{base_state_id}__boundary",
                    "type": "same_image_and_target_change_legal_boundary",
                    "left_group_id": variant_ids["nominal"],
                    "right_group_id": variant_ids["boundary_swap"],
                    "physical_consistency": "identical state and target; legal boundary changed and probes must be regenerated",
                },
                {
                    "pair_id": f"{base_state_id}__reflection",
                    "type": "same_physical_state_and_target_change_image_family",
                    "left_group_id": variant_ids["nominal"],
                    "right_group_id": variant_ids["reflection"],
                    "physical_consistency": "width-relative reflection regenerated from the state image",
                },
                {
                    "pair_id": f"{base_state_id}__saturation",
                    "type": "same_physical_state_and_target_change_image_family",
                    "left_group_id": variant_ids["nominal"],
                    "right_group_id": variant_ids["saturation"],
                    "physical_consistency": "sensor clipping regenerated from the state image",
                },
            ]
        )
        source_registry.append(
            {
                "base_state_id": base_state_id,
                "source_case_id": case["case_id"],
                "source_candidate_split": common["source_candidate_split"],
                "assigned_split": split,
                "setup_hash": common["setup_hash"],
                "stratum": case["stratum"],
                "regime": case["regime"],
                "initial_replay_max_abs_error": replay_error,
            }
        )
    expected_groups = 105 if args.include_supplemental else 60
    if len(groups) != expected_groups:
        raise RuntimeError(f"expected {expected_groups} groups, produced {len(groups)}")
    if any(bool(row["frozen_or_protected"]) for row in groups):
        raise RuntimeError("protected group entered candidate manifest")
    split_setups: dict[str, set[str]] = defaultdict(set)
    for row in groups:
        split_setups[str(row["split"])].add(str(row["setup_hash"]))
    split_names = sorted(split_setups)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            if split_setups[left] & split_setups[right]:
                raise RuntimeError("setup leakage across candidate splits")
    groups_path = output / "split_manifests/groups.jsonl"
    interventions_path = output / "matched_interventions.jsonl"
    write_jsonl(groups_path, groups)
    write_jsonl(interventions_path, interventions)
    split_manifest = {
        split: {
            "group_ids": [row["group_id"] for row in groups if row["split"] == split],
            "base_state_ids": sorted({row["base_state_id"] for row in groups if row["split"] == split}),
            "setup_hashes": sorted(split_setups[split]),
        }
        for split in split_names
    }
    atomic_json(output / "split_manifests/splits.json", split_manifest)
    confirmation_manifest = split_manifest["candidate_confirmation"]
    atomic_json(output / "split_manifests/confirmation_frozen.json", confirmation_manifest)
    confirmation_hash = sha256_path(output / "split_manifests/confirmation_frozen.json")
    rollout_config = {
        "candidate_only": True,
        "formal_frozen_evaluation_enabled": False,
        "protected_split_accessed": False,
        "root_seed": ROOT_SEED,
        "groups": len(groups),
        "supplemental_sampling_triggered": bool(args.include_supplemental),
        "plans": list(PLAN_NAMES),
        "plan_bank_revision": PLAN_BANK_REVISION,
        "cem_seeds_per_group_plan": 3,
        "minimum_formal_episodes": len(groups) * len(PLAN_NAMES) * 3,
        "fixed_controller_config": FIXED_CONTROLLER_CONFIG,
        "fixed_controller_config_hash": canonical_hash(FIXED_CONTROLLER_CONFIG),
        "gain_policy": "checkpoint-native action semantics; no per-plan scale",
        "action_bounds": _bounds_payload(global_bounds),
        "common_random_numbers": "same three CEM seeds for every plan within group",
        "ranking": [
            "strict_all_five_success_rate_desc",
            "terminal_normalized_error_asc",
            "control_steps_asc",
            "boundary_risk_cost_asc",
        ],
        "ambiguous_rule": "top plans tied in success and within 0.05 normalized error are ambiguous",
        "source_files": {
            "source_suite": str(args.source_suite.resolve()),
            "source_suite_sha256": sha256_path(args.source_suite.resolve()),
            "source_train": str(args.source_train.resolve()),
            "source_train_sha256": sha256_path(args.source_train.resolve()),
            "source_dev": str(args.source_dev.resolve()),
            "source_dev_sha256": sha256_path(args.source_dev.resolve()),
            "v12_config": str(args.v12_config.resolve()),
            "v12_config_sha256": sha256_path(args.v12_config.resolve()),
            "base_simulator_config": str(args.base_config.resolve()),
            "base_simulator_config_sha256": sha256_path(args.base_config.resolve()),
            "learned_h1_checkpoint": str(args.forward_checkpoint.resolve()),
            "learned_h1_checkpoint_sha256": sha256_path(args.forward_checkpoint.resolve()),
            "locked_h1_protocol": str(args.locked_h1.resolve()),
            "locked_h1_protocol_sha256": sha256_path(args.locked_h1.resolve()),
        },
        "confirmation_manifest_sha256": confirmation_hash,
        "generation_elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "rollout_config.json", rollout_config)
    plan_contracts = {
        "plan_bank_revision": PLAN_BANK_REVISION,
        "plan_bank": {name: spec.to_dict() for name, spec in build_plan_bank().items()},
        "fixed_controller_config_hash_by_plan": {
            name: canonical_hash(FIXED_CONTROLLER_CONFIG) for name in PLAN_NAMES
        },
        "allowed_plan_varying_fields": [
            "measurement_source",
            "phase_one_metric_weights",
            "phase_one_transition",
            "actuator_policy",
        ],
        "forbidden_qwen_outputs": ["gain", "action_bound", "continuous_action", "cem_budget", "horizon"],
        "primary_spot_implementation": "observed-image-only peak detection plus weighted two-Gaussian EM deblending",
        "recover_then_full": {
            "available": False,
            "reason": "simulator has no causal exposure state; a string-only reacquire is prohibited",
        },
    }
    atomic_json(output / "plan_contracts.json", plan_contracts)
    manifest = {
        "status": "CANDIDATE ONLY - FROZEN/PROTECTED EVALUATION DISABLED",
        "git_commit": os.popen(f"git -C {REPOSITORY_ROOT} rev-parse HEAD").read().strip(),
        "git_branch": os.popen(f"git -C {REPOSITORY_ROOT} branch --show-current").read().strip(),
        "working_tree_was_dirty": True,
        "existing_user_changes_preserved": True,
        "commit_or_push_performed": False,
        "source_registry": source_registry,
        "artifact_files": {
            "groups": str(groups_path.relative_to(output)),
            "groups_sha256": sha256_path(groups_path),
            "interventions": str(interventions_path.relative_to(output)),
            "interventions_sha256": sha256_path(interventions_path),
            "confirmation_manifest_sha256": confirmation_hash,
        },
        "rollout_config_hash": canonical_hash(rollout_config),
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"groups": len(groups), "output": str(output), "elapsed_seconds": time.perf_counter() - started}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", type=Path, default=DEFAULT_SOURCE_SUITE)
    parser.add_argument("--source-train", type=Path, default=DEFAULT_SOURCE_TRAIN)
    parser.add_argument("--source-dev", type=Path, default=DEFAULT_SOURCE_DEV)
    parser.add_argument("--v12-config", type=Path, default=DEFAULT_V12_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--forward-checkpoint", type=Path, default=DEFAULT_FORWARD_CHECKPOINT)
    parser.add_argument("--locked-h1", type=Path, default=DEFAULT_LOCKED_H1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-supplemental", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
