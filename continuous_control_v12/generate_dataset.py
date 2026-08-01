#!/usr/bin/env python3
"""Generate versioned continuous v12 transitions; full generation is opt-in."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    OUTPUT_FIELDS,
    POSITION_FIELDS,
    Bounds,
    action_dict,
    apply_action,
    context_hash,
    position_dict,
    position_vector,
    project_action,
    setup_hash,
    split_hash,
    stable_seed,
)
from continuous_control_v12.sampling import sample_continuous_actions
from continuous_control_v12.schema import sha256_file, validate_dataset
from continuous_control_v12.simulator import (
    REGIMES,
    default_simulator_fixed,
    is_corrected_semantics,
    sample_group_setup,
    simulate_state,
)

DEFAULT_CONFIG = Path(__file__).with_name("config_v12.json")
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_data/continuous_control_v12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Generate only the deterministic tiny smoke configuration.",
    )
    parser.add_argument(
        "--store-images",
        action="store_true",
        help="Store compressed intensity arrays and populate image references.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a version/config-matched incomplete group-wise generation.",
    )
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )


def save_capture(
    output_dir: Path,
    split: str,
    state_id: str,
    capture: dict[str, Any],
    store_images: bool,
    cache: dict[str, str | None],
) -> str | None:
    if state_id in cache:
        return cache[state_id]
    if not store_images:
        cache[state_id] = None
        return None
    relative = Path("images") / split / f"{state_id}.npz"
    path = output_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if "intensity" not in existing.files:
                raise ValueError(f"incomplete existing capture during resume: {path}")
        cache[state_id] = str(relative)
        return str(relative)
    if capture.get("power_semantics") == "source_integrated_optical_power_w":
        np.savez_compressed(
            path,
            image_raw=capture["image_raw"],
            image_normalized=capture["image_normalized"],
            intensity=capture["intensity"],
            valid_region_mask=capture["valid_region_mask"],
        )
    else:
        np.savez_compressed(path, intensity=capture["intensity"])
    cache[state_id] = str(relative)
    return str(relative)


def transition_row(
    *,
    config: dict[str, Any],
    group_id: str,
    split: str,
    transition_id: str,
    setup_context: dict[str, float],
    simulator_fixed: dict[str, Any],
    positions: np.ndarray,
    current: dict[str, Any],
    current_ref: str | None,
    action: np.ndarray,
    requested_action: np.ndarray,
    next_positions: np.ndarray,
    next_capture: dict[str, Any],
    next_ref: str | None,
    sampling: dict[str, Any],
    oracle_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    boundary = bool(
        next_capture["auxiliary"]["camera_boundary_indicator"]
        or next_capture["auxiliary"]["actuator_limit_indicator"]
    )
    row = {
        "schema_version": config["schema_version"],
        "transition_id": transition_id,
        "group_id": group_id,
        "split": split,
        "simulator_seed": stable_seed(config["seed"], transition_id, "simulator"),
        "generator_version": config["generator_version"],
        "setup_context": setup_context,
        "simulator_fixed": simulator_fixed,
        "positions_mm": position_dict(positions),
        "image_ref": current_ref,
        "metrics": current["metrics"],
        "action_mm": action_dict(action),
        "next_positions_mm": position_dict(next_positions),
        "next_image_ref": next_ref,
        "next_metrics": next_capture["metrics"],
        "flags": {
            "terminated": False,
            "boundary": boundary,
            "invalid_simulation": not bool(
                next_capture["auxiliary"]["simulator_valid"]
            ),
        },
        "auxiliary": next_capture["auxiliary"],
        "sampling": sampling,
        "setup_hash": setup_hash(setup_context, simulator_fixed),
        "context_hash": context_hash(
            setup_context,
            simulator_fixed,
            position_dict(positions),
            current["metrics"],
        ),
    }
    if str(config["schema_version"]) == "v12.1.0":
        row.update(
            {
                "simulator_semantics_version": simulator_fixed[
                    "simulator_semantics_version"
                ],
                "sampling_method": simulator_fixed[
                    "sensor_sampling_method"
                ],
                "requested_action_mm": action_dict(requested_action),
                "requested_next_positions_mm": position_dict(
                    positions + requested_action
                ),
                "metrics_lab_frame": current["metrics_lab_frame"],
                "next_metrics_lab_frame": next_capture[
                    "metrics_lab_frame"
                ],
                "metrics_sensor_frame": current["metrics_sensor_frame"],
                "next_metrics_sensor_frame": next_capture[
                    "metrics_sensor_frame"
                ],
                "coordinate_semantics": current["coordinate_semantics"],
                "next_coordinate_semantics": next_capture[
                    "coordinate_semantics"
                ],
                "sampling_metadata": current["sampling_metadata"],
                "next_sampling_metadata": next_capture[
                    "sampling_metadata"
                ],
                "power_semantics": current["power_semantics"],
                "intensity_normalization": current[
                    "intensity_normalization"
                ],
            }
        )
    if oracle_metadata is not None:
        row["oracle_metadata"] = oracle_metadata
    return row


def generate_group(
    *,
    config: dict[str, Any],
    output_dir: Path,
    split: str,
    group_index: int,
    regime: str,
    bounds: Bounds,
    simulator_fixed: dict[str, Any],
    base_config_path: str,
    probe_count: int,
    trajectories: int,
    trajectory_steps: int,
    store_images: bool,
) -> list[dict[str, Any]]:
    group_id = f"v12_{split}_{group_index:06d}"
    image_cache: dict[str, str | None] = {}
    minimum_fraction = float(
        simulator_fixed.get("minimum_initial_captured_power_fraction", 0.0)
    )
    signal_quality_policy_active = (
        "minimum_initial_captured_power_fraction" in simulator_fixed
    )
    maximum_attempts = int(
        simulator_fixed.get("maximum_setup_resample_attempts", 1)
    )
    for setup_attempt in range(maximum_attempts):
        setup_context, initial_positions_dict = sample_group_setup(
            regime,
            group_id,
            int(config["seed"]),
            simulator_fixed,
            bounds,
            setup_attempt=setup_attempt,
        )
        initial_positions = position_vector(initial_positions_dict)
        initial = simulate_state(
            setup_context,
            initial_positions_dict,
            simulator_fixed,
            base_config_path,
            bounds,
        )
        initial_fraction = float(
            initial["auxiliary"].get("captured_power_w", 0.0)
            / max(
                float(
                    initial["auxiliary"].get(
                        "source_integrated_power_w",
                        setup_context["power_w"],
                    )
                ),
                1e-30,
            )
        )
        if initial_fraction >= minimum_fraction:
            break
    else:
        raise RuntimeError(
            f"{group_id}: no setup met minimum initial captured-power "
            f"fraction {minimum_fraction} in {maximum_attempts} attempts"
        )
    initial_ref = save_capture(
        output_dir,
        split,
        f"{group_id}_initial",
        initial,
        store_images,
        image_cache,
    )
    rows: list[dict[str, Any]] = []
    probes = sample_continuous_actions(
        probe_count,
        stable_seed(config["seed"], group_id, "probes"),
        bounds,
        config.get("sampling"),
    )
    for probe_index, probe in enumerate(probes):
        requested = np.asarray(probe["action"], dtype=np.float64)
        action = project_action(initial_positions, requested, bounds)
        next_positions = apply_action(initial_positions, action, bounds)
        capture = simulate_state(
            setup_context,
            position_dict(next_positions),
            simulator_fixed,
            base_config_path,
            bounds,
        )
        state_id = f"{group_id}_probe_{probe_index:03d}_next"
        next_ref = save_capture(
            output_dir, split, state_id, capture, store_images, image_cache
        )
        sampling = dict(probe["sampling"])
        if signal_quality_policy_active:
            sampling["setup_resample_attempt"] = setup_attempt
            sampling["initial_captured_power_fraction"] = initial_fraction
        if not np.allclose(requested, action, atol=1e-12, rtol=0.0):
            sampling["absolute_limit_projection"] = True
            sampling["requested_action_mm"] = action_dict(requested)
        rows.append(
            transition_row(
                config=config,
                group_id=group_id,
                split=split,
                transition_id=f"{group_id}_probe_{probe_index:03d}",
                setup_context=setup_context,
                simulator_fixed=simulator_fixed,
                positions=initial_positions,
                current=initial,
                current_ref=initial_ref,
                action=action,
                requested_action=requested,
                next_positions=next_positions,
                next_capture=capture,
                next_ref=next_ref,
                sampling=sampling,
            )
        )

    for trajectory_index in range(trajectories):
        rng = np.random.default_rng(
            stable_seed(config["seed"], group_id, trajectory_index, "trajectory")
        )
        target_delta = rng.uniform(-0.9, 0.9, size=4) * bounds.action_high * max(
            trajectory_steps, 2
        )
        q_star = np.clip(
            initial_positions + target_delta,
            bounds.position_low,
            bounds.position_high,
        )
        current_positions = initial_positions.copy()
        current = initial
        current_ref = initial_ref
        for step in range(trajectory_steps):
            requested = q_star - current_positions
            action = project_action(current_positions, requested, bounds)
            next_positions = apply_action(current_positions, action, bounds)
            capture = simulate_state(
                setup_context,
                position_dict(next_positions),
                simulator_fixed,
                base_config_path,
                bounds,
            )
            state_id = (
                f"{group_id}_trajectory_{trajectory_index:02d}_step_{step:02d}_next"
            )
            next_ref = save_capture(
                output_dir, split, state_id, capture, store_images, image_cache
            )
            trajectory_sampling = {
                "kind": "closed_loop_trajectory",
                "trajectory_id": f"{group_id}_trajectory_{trajectory_index:02d}",
                "step": step,
            }
            if signal_quality_policy_active:
                trajectory_sampling.update(
                    {
                        "setup_resample_attempt": setup_attempt,
                        "initial_captured_power_fraction": initial_fraction,
                    }
                )
            row = transition_row(
                config=config,
                group_id=group_id,
                split=split,
                transition_id=(
                    f"{group_id}_trajectory_{trajectory_index:02d}_step_{step:02d}"
                ),
                setup_context=setup_context,
                simulator_fixed=simulator_fixed,
                positions=current_positions,
                current=current,
                current_ref=current_ref,
                action=action,
                requested_action=requested,
                next_positions=next_positions,
                next_capture=capture,
                next_ref=next_ref,
                sampling=trajectory_sampling,
                oracle_metadata={
                    "q_star_mm": position_dict(q_star),
                    "visibility": "supervision_only_not_deployed_input",
                },
            )
            row["flags"]["terminated"] = bool(
                np.allclose(next_positions, q_star, atol=1e-10, rtol=0.0)
            )
            rows.append(row)
            current_positions = next_positions
            current = capture
            current_ref = next_ref
    return rows


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    if "physics_structured_rebuild_v10" in output_dir.parts:
        raise ValueError("v12 refuses every v10 output path")
    state_path = output_dir / "generation_state.json"
    if output_dir.exists():
        if not args.resume:
            raise RuntimeError(
                f"refusing to overwrite existing dataset: {output_dir}"
            )
        if (output_dir / "manifest.json").exists():
            raise RuntimeError(
                f"dataset already has a manifest; refusing resume: {output_dir}"
            )
        if not state_path.exists():
            raise RuntimeError(
                f"resume requires generation_state.json: {output_dir}"
            )
        prior_state = json.loads(state_path.read_text(encoding="utf-8"))
        if prior_state["config_sha256"] != sha256_file(config_path):
            raise RuntimeError("resume config hash differs from incomplete run")
        if prior_state["schema_version"] != config["schema_version"]:
            raise RuntimeError("resume schema version differs")
    else:
        output_dir.mkdir(parents=True)
        prior_state = {
            "version": "continuous_control_v12_generation_state",
            "schema_version": config["schema_version"],
            "config_sha256": sha256_file(config_path),
            "complete": False,
            "completed_groups": [],
        }
        state_path.write_text(
            json.dumps(prior_state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    bounds = Bounds.from_config(config)
    bounds.validate()
    base_config_path = str((REPO_ROOT / config["simulator"]["base_config"]).resolve())
    if args.smoke:
        group_counts = config["smoke"]["group_counts"]
        probe_count = int(config["smoke"]["continuous_probes"])
        trajectories = int(config["smoke"]["trajectories"])
        trajectory_steps = int(config["smoke"]["trajectory_steps"])
        simulator_fixed = default_simulator_fixed(
            base_config_path,
            grid_size=int(config["smoke"]["simulator_grid_size"]),
            grid_extent_mm=float(
                config["smoke"].get(
                    "simulator_grid_extent_mm",
                    config["simulator"].get("data_grid_extent_mm", 30.0),
                )
            ),
            sensor_resolution=list(config["smoke"]["sensor_resolution"]),
            semantics=config["simulator"].get("semantics"),
        )
        scale = "smoke"
        store_images = True
    else:
        group_counts = config["group_counts"]
        probe_count = int(config["per_group"]["continuous_probes"])
        trajectories = int(config["per_group"]["trajectories"])
        trajectory_steps = int(config["per_group"]["trajectory_steps"])
        simulator_fixed = default_simulator_fixed(
            base_config_path,
            grid_size=config["simulator"].get("data_grid_size"),
            grid_extent_mm=config["simulator"].get("data_grid_extent_mm"),
            sensor_resolution=config["simulator"].get(
                "data_sensor_resolution"
            ),
            semantics=config["simulator"].get("semantics"),
        )
        scale = "full"
        store_images = bool(args.store_images)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    coverage_regimes: Counter[str] = Counter()
    coverage_sampling: Counter[str] = Counter()
    group_regimes: dict[str, str] = {}
    action_values = []
    position_values = []
    auxiliary_available: defaultdict[str, int] = defaultdict(int)
    for split_index, split in enumerate(("train", "development", "test")):
        rows = []
        for group_index in range(int(group_counts[split])):
            regime = REGIMES[
                stable_seed(config["seed"], split, group_index, "regime")
                % len(REGIMES)
            ]
            group_id = f"v12_{split}_{group_index:06d}"
            group_regimes[group_id] = regime
            coverage_regimes[regime] += 1
            partial_path = (
                output_dir
                / "partial_groups"
                / split
                / f"{group_id}.jsonl"
            )
            if partial_path.exists():
                group_rows = [
                    json.loads(line)
                    for line in partial_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ]
                if not group_rows or {
                    str(row["group_id"]) for row in group_rows
                } != {group_id}:
                    raise ValueError(f"invalid partial group: {partial_path}")
            else:
                group_rows = generate_group(
                    config=config,
                    output_dir=output_dir,
                    split=split,
                    group_index=group_index,
                    regime=regime,
                    bounds=bounds,
                    simulator_fixed=simulator_fixed,
                    base_config_path=base_config_path,
                    probe_count=probe_count,
                    trajectories=trajectories,
                    trajectory_steps=trajectory_steps,
                    store_images=store_images,
                )
                write_jsonl(partial_path, group_rows)
            completed = set(prior_state.get("completed_groups", []))
            if group_id not in completed:
                completed.add(group_id)
                prior_state["completed_groups"] = sorted(completed)
                prior_state["last_completed_group"] = group_id
                prior_state["updated_unix_time"] = time.time()
                state_path.write_text(
                    json.dumps(prior_state, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            rows.extend(group_rows)
            for row in group_rows:
                coverage_sampling[row["sampling"]["kind"]] += 1
                action_values.append([row["action_mm"][field] for field in ACTION_FIELDS])
                position_values.extend(
                    [
                        [row["positions_mm"][field] for field in POSITION_FIELDS],
                        [
                            row["next_positions_mm"][field]
                            for field in POSITION_FIELDS
                        ],
                    ]
                )
                for key, value in row["auxiliary"].items():
                    if value is not None:
                        auxiliary_available[key] += 1
        split_rows[split] = rows
        write_jsonl(output_dir / "transitions" / f"{split}.jsonl", rows)

    split_summary = {}
    split_hashes = {}
    for split, rows in split_rows.items():
        groups = sorted({str(row["group_id"]) for row in rows})
        path = output_dir / "transitions" / f"{split}.jsonl"
        split_hashes[split] = split_hash(groups)
        split_summary[split] = {
            "groups": len(groups),
            "transitions": len(rows),
            "jsonl": str(path.relative_to(output_dir)),
            "jsonl_sha256": sha256_file(path),
        }
    actions = np.asarray(action_values, dtype=np.float64)
    positions = np.asarray(position_values, dtype=np.float64)
    corrected = is_corrected_semantics(simulator_fixed)
    manifest = {
        "version": config["version"],
        "schema_version": config["schema_version"],
        "complete": True,
        "scale": scale,
        "units": {
            "actions": "mm",
            "absolute_positions": "mm",
            "wavelength": "nm",
            "pixel_size": "um",
            "centroids_and_sigmas": "px",
            "peak_intensity": (
                "raw_irradiance_w_per_m2"
                if corrected
                else "simulator_intensity_units"
            ),
        },
        "output_order": list(OUTPUT_FIELDS),
        "tolerances": config["output_tolerances"],
        "feature_definitions": {
            "setup_context": (
                "eight causal numerical setup fields; power_w is source "
                "integrated optical power and causally scales absolute intensity"
                if corrected
                else "eight causal/fixed numerical setup fields; power is retained "
                "although the current simulator source normalization ignores it"
            ),
            "absolute_positions": list(POSITION_FIELDS),
            "current_metrics": list(OUTPUT_FIELDS),
            "continuous_action": list(ACTION_FIELDS),
            "image_conditioning": "optional; references null when images not stored",
            "coordinate_convention": config["simulator"][
                "position_coordinate_convention"
            ],
            "metrics_lab_frame": "metrics and next_metrics",
            "image_sensor_frame": "stored raw and normalized sensor arrays",
            "camera_pose_lab": "positions_mm.camera_x_mm/camera_y_mm",
            "sensor_pixel_pitch": "setup_context.pixel_size_um",
            "sensor_origin": simulator_fixed.get(
                "sensor_origin_convention", "legacy_implicit"
            ),
            "axis_convention": simulator_fixed.get(
                "axis_convention", "legacy_axis_0_plus_y_axis_1_plus_x"
            ),
            "lab_to_sensor_transform": simulator_fixed.get(
                "lab_to_sensor_transform", "legacy_documented_out_of_schema"
            ),
            "target_control_frame": simulator_fixed.get(
                "target_control_frame",
                "legacy_implicit_lab_frame_pseudo_pixels",
            ),
            "phase": "fixed zero and unavailable to deployed input",
        },
        "generator": {
            "version": config["generator_version"],
            "repository_commit": git_commit(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "simulator": "optical_sim",
            "simulator_fixed": simulator_fixed,
        },
        "seeds": {"root": int(config["seed"])},
        "split_hashes": split_hashes,
        "split_summary": split_summary,
        "coverage": {
            "regime_group_counts": dict(sorted(coverage_regimes.items())),
            "sampling_counts": dict(sorted(coverage_sampling.items())),
            "action_min_mm": actions.min(axis=0).tolist(),
            "action_max_mm": actions.max(axis=0).tolist(),
            "position_min_mm": positions.min(axis=0).tolist(),
            "position_max_mm": positions.max(axis=0).tolist(),
            "auxiliary_non_null_counts": dict(sorted(auxiliary_available.items())),
            "images_stored": store_images,
        },
        "per_step_action_bounds_mm": config["per_step_action_bounds_mm"],
        "absolute_position_limits_mm": config["absolute_position_limits_mm"],
        "absolute_limit_source": config["absolute_limit_source"],
        "group_regimes": group_regimes,
        "q_star_deployed_input": False,
        "simulator_semantics_version": simulator_fixed.get(
            "simulator_semantics_version", "legacy_v1_implicit"
        ),
        "sensor_sampling_method": simulator_fixed.get(
            "sensor_sampling_method",
            "legacy_left_searchsorted_integer_lookup",
        ),
        "power_semantics": simulator_fixed.get(
            "power_semantics", "legacy_declared_but_ignored"
        ),
        "intensity_normalization": simulator_fixed.get(
            "intensity_normalization",
            "legacy_unnormalized_simulator_units",
        ),
        "power_affects_current_simulator": corrected,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = validate_dataset(output_dir, config)
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prior_state["complete"] = True
    prior_state["manifest"] = "manifest.json"
    prior_state["updated_unix_time"] = time.time()
    state_path.write_text(
        json.dumps(prior_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(output_dir / "manifest.json"),
                "validation": validation,
                "scale": scale,
                "split_summary": split_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
