#!/usr/bin/env python3
"""Run final-only physical probe-coupling interventions after Qwen training.

Each intervention changes an optical setup parameter, regenerates the sensor
capture, physically reachable target, and Learned-H1 probes, while retaining
the source pair's dominant normalized error component.  All plans then run
with the same 13 CEM seeds as the paired original confirmation group.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from continuous_control_v12.contracts import Bounds, metrics_dict, metrics_vector, position_vector, setup_hash, stable_seed, tolerance_vector
from continuous_control_v12.simulator import simulate_state
from continuous_control_v12.world_model import load_forward_ensemble

from .core import FIXED_CONTROLLER_CONFIG, PLAN_NAMES, FixedGainPlanController, PlanCEM, apply_sensor_family, canonical_hash, measure_capture
from .generate_groups import _save_sensor_png
from .run_rollouts import DEFAULT_BASE_CONFIG, DEFAULT_CHECKPOINT, SEED_ROOT
from .export_sft import _visible_state
from .protocol import SYSTEM_PROMPT, USER_PREFIX


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/qwen_reasoning_plan_selector"
SEED_INDICES = tuple(range(13))
FACTORS = (0.97, 1.03, 0.94, 1.06)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bounds_from(group: dict) -> Bounds:
    value = group["bounds"]
    return Bounds(action_low=np.asarray(value["action_low"], dtype=np.float64), action_high=np.asarray(value["action_high"], dtype=np.float64), position_low=np.asarray(value["position_low"], dtype=np.float64), position_high=np.asarray(value["position_high"], dtype=np.float64))


def components(initial: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(initial - target) / tolerance_vector(initial)


def probe_vector(audit: dict) -> np.ndarray:
    return np.asarray([row["sensitivity"] for row in audit["rows"]], dtype=np.float64)


def main() -> None:
    started = time.perf_counter()
    for seed in (2026080401, 2026080402, 2026080403):
        manifest = json.loads((ARTIFACT / f"training/seed_{seed}/run_manifest.latest.json").read_text())
        if manifest.get("status") != "completed":
            raise RuntimeError("probe-coupling confirmation may run only after all Qwen training seeds complete")
    groups = read_jsonl(ARTIFACT / "split_manifests/groups.jsonl")
    originals = [row for row in groups if row["split"] == "candidate_confirmation" and row["variant"] == "nominal"]
    if len(originals) != 3:
        raise RuntimeError("expected three nominal confirmation source groups")
    rollout_rows = read_jsonl(ARTIFACT / "rollout_results.jsonl")
    original_direct = {(row["group_id"], int(row["seed_index"])): row for row in rollout_rows if row["plan_name"] == "direct_all_five"}
    model = load_forward_ensemble(DEFAULT_CHECKPOINT, device_name="cuda")
    controller = FixedGainPlanController(model=model, base_config_path=str(DEFAULT_BASE_CONFIG))
    intervention_groups = []
    pair_rows = []
    initial_captures: dict[str, dict] = {}
    for source in originals:
        source_bounds = bounds_from(source)
        source_initial = metrics_vector(source["initial_clean_metrics"])
        source_target = metrics_vector(source["target_metrics"])
        source_dominant = int(np.argmax(components(source_initial, source_target)))
        source_probe = probe_vector(original_direct[(source["group_id"], 0)]["initial_probe_audit"])
        candidates = []
        for factor in FACTORS:
            setup = copy.deepcopy(source["setup_context"])
            setup["lens_focal_length_mm"] = float(setup["lens_focal_length_mm"] * factor)
            initial_capture = simulate_state(setup, source["initial_positions_mm"], source["simulator_fixed"], str(DEFAULT_BASE_CONFIG), source_bounds)
            goal_capture = simulate_state(setup, source["q_goal_mm_evaluator_only"], source["simulator_fixed"], str(DEFAULT_BASE_CONFIG), source_bounds)
            initial = metrics_vector(initial_capture["metrics"])
            target = metrics_vector(goal_capture["metrics"])
            dominant = int(np.argmax(components(initial, target)))
            if dominant != source_dominant or not bool(initial_capture["auxiliary"]["simulator_valid"]) or not bool(goal_capture["auxiliary"]["simulator_valid"]):
                continue
            measured, _, _ = measure_capture(initial_capture, family="nominal", anomaly={}, source="standard")
            planner = PlanCEM(model=model, setup_context=setup, bounds=source_bounds, config=FIXED_CONTROLLER_CONFIG, seed=int(stable_seed(SEED_ROOT, source["group_id"], 0) % (2**32)))
            probe = planner.probe_actuators(position_vector(source["initial_positions_mm"]), measured)
            relative_change = float(np.linalg.norm(probe_vector(probe) - source_probe) / max(np.linalg.norm(source_probe), 1e-12))
            candidates.append((relative_change, factor, setup, initial_capture, target, probe))
        if not candidates:
            raise RuntimeError(f"no physically valid dominant-error-preserving coupling intervention for {source['group_id']}")
        relative_change, factor, setup, capture, target, probe = max(candidates, key=lambda row: (row[0], -FACTORS.index(row[1])))
        if relative_change < 0.05:
            raise RuntimeError(f"physical probe coupling changed by less than 5% for {source['group_id']}: {relative_change}")
        group_id = f"{source['group_id']}__probe_coupling_swap"
        image, sensor_evidence = apply_sensor_family(np.asarray(capture["intensity"]), "nominal", {})
        image_path = ARTIFACT / "probe_coupling_confirmation/images" / f"{group_id}.png"
        _save_sensor_png(image, image_path)
        cache_path = ARTIFACT / "probe_coupling_confirmation/initial_captures" / f"{group_id}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, intensity=np.asarray(capture["intensity"], dtype=np.float32), metrics=metrics_vector(capture["metrics"]))
        group = {**copy.deepcopy(source), "group_id": group_id, "base_state_id": f"{source['base_state_id']}__probe_coupling_swap", "split": "candidate_probe_coupling_confirmation", "setup_context": setup, "setup_hash": setup_hash(setup, source["simulator_fixed"]), "initial_clean_metrics": metrics_dict(metrics_vector(capture["metrics"])), "tolerance_reference_metrics": metrics_dict(metrics_vector(capture["metrics"])), "target_metrics": metrics_dict(target), "visual_family": "nominal", "variant": "probe_coupling_swap", "anomaly": {}, "intervention_type": "physical_probe_coupling", "initial_capture_cache": str(cache_path.relative_to(ARTIFACT)), "initial_capture_cache_sha256": sha256(cache_path), "initial_sensor_image": str(image_path.relative_to(ARTIFACT)), "initial_sensor_image_sha256": sha256(image_path), "sensor_generation_evidence": sensor_evidence, "frozen_or_protected": False}
        intervention_groups.append(group)
        initial_captures[group_id] = {"metrics": metrics_vector(capture["metrics"]), "intensity": np.asarray(capture["intensity"], dtype=np.float32)}
        pair_rows.append({"pair_id": f"{source['base_state_id']}__physical_probe_coupling", "type": "same_dominant_error_change_physical_probe_coupling", "left_group_id": source["group_id"], "right_group_id": group_id, "changed_setup_field": "lens_focal_length_mm", "factor": factor, "dominant_error_component": source_dominant, "right_dominant_error_component": int(np.argmax(components(metrics_vector(capture["metrics"]), target))), "probe_sensitivity_relative_l2_change": relative_change, "right_probe_audit": probe, "target_regenerated_at_same_physical_q_goal": True, "selection_used_no_plan_outcomes": True})
    manifest_path = ARTIFACT / "probe_coupling_confirmation/groups.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in intervention_groups:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    pairs_path = ARTIFACT / "probe_coupling_confirmation/pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as stream:
        for row in pair_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    output = ARTIFACT / "probe_coupling_confirmation/rollout_results.jsonl"
    completed = {(row["group_id"], row["plan_name"], int(row["seed_index"])) for row in read_jsonl(output)} if output.exists() else set()
    executed, durations = 0, []
    with output.open("a" if output.exists() else "w", encoding="utf-8", buffering=1) as stream:
        for group in intervention_groups:
            for plan in PLAN_NAMES:
                for seed_index in SEED_INDICES:
                    key = (group["group_id"], plan, seed_index)
                    if key in completed:
                        continue
                    cem_seed = int(stable_seed(SEED_ROOT, group["group_id"].split("__probe_coupling_swap")[0], seed_index) % (2**32))
                    one = time.perf_counter()
                    result = controller.run_episode(group, plan_name=plan, cem_seed=cem_seed, initial_capture=initial_captures[group["group_id"]])
                    elapsed = time.perf_counter() - one
                    row = {"schema_version": "qwen_reasoning_plan_probe_coupling_confirmation_v1", "candidate_only": True, "formal_frozen_evaluation_enabled": False, "final_confirmation_only": True, "paired_common_random_number_seed": True, "seed_index": seed_index, "episode_wall_seconds": elapsed, **result}
                    if row["fixed_controller_config_hash"] != canonical_hash(FIXED_CONTROLLER_CONFIG):
                        raise RuntimeError("controller config drift")
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
                    completed.add(key); executed += 1; durations.append(elapsed)
    status = {"candidate_only": True, "groups": len(intervention_groups), "pairs": len(pair_rows), "plans": list(PLAN_NAMES), "seeds_per_group_plan": len(SEED_INDICES), "expected_episodes": len(intervention_groups) * len(PLAN_NAMES) * len(SEED_INDICES), "completed_episodes": len(completed), "executed_this_invocation": executed, "wall_seconds": time.perf_counter() - started, "mean_episode_seconds": float(np.mean(durations)) if durations else 0.0, "output_sha256": sha256(output), "groups_sha256": sha256(manifest_path), "pairs_sha256": sha256(pairs_path), "fixed_controller_config_hash": canonical_hash(FIXED_CONTROLLER_CONFIG), "frozen_or_protected_enabled": False}
    completed_rows = read_jsonl(output)
    representatives = {row["group_id"]: row for row in completed_rows if row["plan_name"] == "direct_all_five" and int(row["seed_index"]) == 0}
    inputs_path = ARTIFACT / "probe_coupling_confirmation/inputs.jsonl"
    with inputs_path.open("w", encoding="utf-8") as stream:
        for group in sorted(intervention_groups, key=lambda row: row["group_id"]):
            visible = _visible_state(group, representatives[group["group_id"]])
            prompt = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PREFIX + json.dumps(visible, sort_keys=True, separators=(",", ":"), allow_nan=False)}]}]
            image_repo_relative = (ARTIFACT / group["initial_sensor_image"]).relative_to(ROOT).as_posix()
            stream.write(json.dumps({"group_id": group["group_id"], "images": [image_repo_relative], "image_sha256": group["initial_sensor_image_sha256"], "prompt": prompt, "candidate_only": True}, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    status["inputs_sha256"] = sha256(inputs_path)
    (ARTIFACT / "probe_coupling_confirmation/status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
