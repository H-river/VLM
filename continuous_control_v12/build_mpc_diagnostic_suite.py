#!/usr/bin/env python3
"""Build fresh group-disjoint, simulator-verified v12 MPC diagnostic targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
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
    metrics_vector,
    normalized_distance,
    position_dict,
    position_vector,
    setup_hash,
    stable_seed,
)
from continuous_control_v12.schema import read_jsonl
from continuous_control_v12.simulator import (
    CORRECTED_SEMANTICS_VERSION,
    default_simulator_fixed,
    sample_group_setup,
    simulate_state,
)

STRATA = (
    "one_step_reachable_interior",
    "multi_step_reachable_interior",
    "reachable_boundary_or_clipping",
)
INTERIOR_REGIMES = (
    "ordinary",
    "focusing",
    "high_offset_interaction",
)
BOUNDARY_REGIMES = ("camera_boundary", "clipping")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_locked_sources(config: dict[str, Any]) -> None:
    checks = (
        ("corrected_v12_config", "corrected_v12_config_sha256"),
        ("base_simulator_config", "base_simulator_config_sha256"),
        ("checkpoint", "checkpoint_sha256"),
    )
    for path_key, hash_key in checks:
        path = Path(config[path_key]).resolve()
        actual = _sha256(path)
        if actual != str(config[hash_key]):
            raise ValueError(
                f"locked source hash mismatch for {path}: "
                f"{actual} != {config[hash_key]}"
            )
    manifest = (
        Path(config["training_dataset"]).resolve() / "manifest.json"
    )
    actual = _sha256(manifest)
    if actual != str(config["training_dataset_manifest_sha256"]):
        raise ValueError("locked training manifest hash mismatch")


def _prior_ids_and_hashes(
    data_dir: Path,
    excluded_suites: list[Path],
) -> tuple[set[str], set[str]]:
    group_ids: set[str] = set()
    setup_hashes: set[str] = set()
    for split in ("train", "development", "test"):
        for row in read_jsonl(data_dir / "transitions" / f"{split}.jsonl"):
            group_ids.add(str(row["group_id"]))
            setup_hashes.add(str(row["setup_hash"]))
    for suite_path in excluded_suites:
        suite = json.loads(suite_path.resolve().read_text(encoding="utf-8"))
        for case in suite["cases"]:
            group_ids.add(str(case["group_id"]))
            setup_hashes.add(str(case["setup_hash"]))
    return group_ids, setup_hashes


def _training_action_distribution(
    data_dir: Path,
    quantile_config: dict[str, Any],
) -> dict[str, Any]:
    rows = read_jsonl(data_dir / "transitions/train.jsonl")
    actions = np.asarray(
        [
            [float(row["action_mm"][field]) for field in ACTION_FIELDS]
            for row in rows
        ],
        dtype=np.float64,
    )
    q_low = float(quantile_config["central_quantile_low"])
    q_high = float(quantile_config["central_quantile_high"])
    tail_low = float(quantile_config["tail_quantile_low"])
    tail_high = float(quantile_config["tail_quantile_high"])
    return {
        "rows": int(len(actions)),
        "axis_order": list(ACTION_FIELDS),
        "central_quantiles": [q_low, q_high],
        "tail_quantiles": [tail_low, tail_high],
        "per_axis": {
            field: {
                "minimum": float(actions[:, index].min()),
                "q01": float(np.quantile(actions[:, index], tail_low)),
                "q05": float(np.quantile(actions[:, index], q_low)),
                "median": float(np.median(actions[:, index])),
                "q95": float(np.quantile(actions[:, index], q_high)),
                "q99": float(np.quantile(actions[:, index], tail_high)),
                "maximum": float(actions[:, index].max()),
            }
            for index, field in enumerate(ACTION_FIELDS)
        },
    }


def _goal_delta(
    *,
    stratum: str,
    stratum_index: int,
    target_attempt: int,
    rng: np.random.Generator,
    bounds: Bounds,
    regime: str,
) -> tuple[np.ndarray, int]:
    high = bounds.action_high
    directions = rng.choice((-1.0, 1.0), size=4)
    if stratum == "one_step_reachable_interior":
        scales = (0.12, 0.32, 0.68, 0.92)
        scale = scales[(stratum_index + target_attempt) % len(scales)]
        active_count = 2 + (stratum_index % 3)
        active = rng.choice(4, size=active_count, replace=False)
        delta = np.zeros(4, dtype=np.float64)
        delta[active] = (
            directions[active]
            * high[active]
            * scale
            * rng.uniform(0.72, 1.0, size=active_count)
        )
        return delta, 1

    steps = 2 + (stratum_index % 3)
    delta = np.zeros(4, dtype=np.float64)
    if stratum == "reachable_boundary_or_clipping":
        anchor_choices = (0, 1) if regime == "camera_boundary" else (0, 1, 2, 3)
        anchor = int(anchor_choices[stratum_index % len(anchor_choices)])
    else:
        anchor = int((stratum_index + target_attempt) % 4)
    delta[anchor] = (
        directions[anchor]
        * high[anchor]
        * (steps - rng.uniform(0.08, 0.32))
    )
    other_axes = [axis for axis in range(4) if axis != anchor]
    active_others = rng.choice(
        other_axes,
        size=1 + (stratum_index % min(3, len(other_axes))),
        replace=False,
    )
    for axis in np.atleast_1d(active_others):
        delta[int(axis)] = (
            directions[int(axis)]
            * high[int(axis)]
            * rng.uniform(0.25, max(0.35, steps - 0.55))
        )
    required = int(
        math.ceil(float(np.max(np.abs(delta) / bounds.action_high)) - 1e-12)
    )
    return delta, required


def _boundary_condition(capture: dict[str, Any], regime: str) -> bool:
    auxiliary = capture["auxiliary"]
    if regime == "camera_boundary":
        return bool(auxiliary["camera_boundary_indicator"]) or float(
            auxiliary["distance_to_camera_boundary_px"]
        ) <= 64.0
    if regime == "clipping":
        return float(auxiliary["clipping_fraction"]) > 0.01
    return (
        not bool(auxiliary["camera_boundary_indicator"])
        and float(auxiliary["clipping_fraction"]) < 0.05
    )


def _signal_fraction(capture: dict[str, Any]) -> float:
    auxiliary = capture["auxiliary"]
    return float(auxiliary["captured_power_w"]) / max(
        float(auxiliary["source_integrated_power_w"]), 1e-30
    )


def _serializable_auxiliary(capture: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            bool(value)
            if isinstance(value, (bool, np.bool_))
            else None
            if value is None
            else float(value)
            if isinstance(value, (int, float, np.integer, np.floating))
            else value
        )
        for key, value in capture["auxiliary"].items()
    }


def _build_case(
    *,
    locked: dict[str, Any],
    v12_config: dict[str, Any],
    bounds: Bounds,
    simulator_fixed: dict[str, Any],
    base_config_path: str,
    suite_label: str,
    stratum: str,
    stratum_index: int,
    excluded_group_ids: set[str],
    excluded_setup_hashes: set[str],
) -> dict[str, Any]:
    root_seed = int(locked["root_seed"])
    group_id = (
        f"v12_mpcdiag_{suite_label}_{STRATA.index(stratum):02d}_"
        f"{stratum_index:04d}"
    )
    if group_id in excluded_group_ids:
        raise ValueError(f"diagnostic group ID collision: {group_id}")
    if stratum == "reachable_boundary_or_clipping":
        regime = BOUNDARY_REGIMES[stratum_index % len(BOUNDARY_REGIMES)]
    else:
        regime = INTERIOR_REGIMES[
            stratum_index % len(INTERIOR_REGIMES)
        ]
    suite_config = locked["suite"]
    minimum_signal = float(
        suite_config["minimum_initial_captured_power_fraction"]
    )
    maximum_setup_attempts = int(suite_config["maximum_setup_attempts"])
    maximum_target_attempts = int(suite_config["maximum_target_attempts"])
    minimum_distance = float(suite_config["minimum_initial_distance"])
    for setup_attempt in range(maximum_setup_attempts):
        setup_context, initial_positions = sample_group_setup(
            regime,
            group_id,
            root_seed,
            simulator_fixed,
            bounds,
            setup_attempt=setup_attempt,
        )
        current_setup_hash = setup_hash(setup_context, simulator_fixed)
        if current_setup_hash in excluded_setup_hashes:
            continue
        initial_capture = simulate_state(
            setup_context,
            initial_positions,
            simulator_fixed,
            base_config_path,
            bounds,
        )
        if (
            not bool(initial_capture["auxiliary"]["simulator_valid"])
            or _signal_fraction(initial_capture) < minimum_signal
        ):
            continue
        if stratum == "reachable_boundary_or_clipping":
            if not _boundary_condition(initial_capture, regime):
                continue
        elif not _boundary_condition(initial_capture, regime):
            continue
        initial_vector = position_vector(initial_positions)
        for target_attempt in range(maximum_target_attempts):
            target_rng = np.random.default_rng(
                stable_seed(
                    root_seed,
                    group_id,
                    stratum,
                    "target",
                    target_attempt,
                )
            )
            delta, required_steps = _goal_delta(
                stratum=stratum,
                stratum_index=stratum_index,
                target_attempt=target_attempt,
                rng=target_rng,
                bounds=bounds,
                regime=regime,
            )
            goal_vector = initial_vector + delta
            if np.any(goal_vector < bounds.position_low) or np.any(
                goal_vector > bounds.position_high
            ):
                continue
            q_goal = position_dict(goal_vector)
            goal_capture = simulate_state(
                setup_context,
                q_goal,
                simulator_fixed,
                base_config_path,
                bounds,
            )
            if (
                not bool(goal_capture["auxiliary"]["simulator_valid"])
                or _signal_fraction(goal_capture) < minimum_signal
                or not _boundary_condition(goal_capture, regime)
            ):
                continue
            initial_distance = normalized_distance(
                initial_capture["metrics"],
                goal_capture["metrics"],
                initial_capture["metrics"],
            )
            if initial_distance <= minimum_distance:
                continue
            replay = simulate_state(
                setup_context,
                q_goal,
                simulator_fixed,
                base_config_path,
                bounds,
            )
            replay_error = np.abs(
                metrics_vector(replay["metrics"])
                - metrics_vector(goal_capture["metrics"])
            )
            if not np.array_equal(
                metrics_vector(replay["metrics"]),
                metrics_vector(goal_capture["metrics"]),
            ):
                raise RuntimeError(
                    f"non-deterministic q_goal replay for {group_id}: "
                    f"{replay_error.tolist()}"
                )
            if (
                stratum == "one_step_reachable_interior"
                and required_steps != 1
            ):
                raise RuntimeError(
                    "one-step target construction is inconsistent"
                )
            if (
                stratum != "one_step_reachable_interior"
                and required_steps not in (2, 3, 4)
            ):
                continue
            return {
                "case_id": group_id,
                "group_id": group_id,
                "group_seed": int(
                    stable_seed(root_seed, group_id, "diagnostic_group")
                ),
                "stratum": stratum,
                "regime": regime,
                "setup_hash": current_setup_hash,
                "setup_resample_attempt": setup_attempt,
                "target_attempt": target_attempt,
                "setup_context": setup_context,
                "simulator_fixed": simulator_fixed,
                "initial_positions_mm": initial_positions,
                "initial_metrics": initial_capture["metrics"],
                "initial_auxiliary": _serializable_auxiliary(
                    initial_capture
                ),
                "target_metrics": goal_capture["metrics"],
                "goal_auxiliary": _serializable_auxiliary(goal_capture),
                "q_goal_mm": q_goal,
                "q_goal_visibility": "evaluator_only_never_planner_input",
                "generating_delta_mm": {
                    field: float(delta[index])
                    for index, field in enumerate(ACTION_FIELDS)
                },
                "minimum_bounded_position_steps": required_steps,
                "initial_normalized_distance": float(initial_distance),
                "exact_q_goal_replay_max_abs_metric_error": float(
                    replay_error.max()
                ),
                "known_reachability": (
                    "verified_by_q_goal_simulator_replay"
                ),
            }
    raise RuntimeError(f"no diagnostic target found for {group_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int)
    parser.add_argument("--suite-label", required=True)
    parser.add_argument(
        "--exclude-suite",
        action="append",
        type=Path,
        default=[],
    )
    args = parser.parse_args()
    started = time.perf_counter()
    config_path = args.config.resolve()
    locked = json.loads(config_path.read_text(encoding="utf-8"))
    _verify_locked_sources(locked)
    v12_config_path = Path(locked["corrected_v12_config"]).resolve()
    v12_config = json.loads(v12_config_path.read_text(encoding="utf-8"))
    if (
        v12_config["simulator"]["semantics"]["simulator_semantics_version"]
        != CORRECTED_SEMANTICS_VERSION
    ):
        raise ValueError("diagnostic suite requires corrected v12 semantics")
    groups = int(locked["suite"]["groups"] if args.groups is None else args.groups)
    if groups < 3 or groups % len(STRATA):
        raise ValueError("diagnostic group count must be divisible by three")
    data_dir = Path(locked["training_dataset"]).resolve()
    excluded_ids, excluded_hashes = _prior_ids_and_hashes(
        data_dir,
        [path.resolve() for path in args.exclude_suite],
    )
    bounds = Bounds.from_config(v12_config)
    simulator = v12_config["simulator"]
    simulator_fixed = default_simulator_fixed(
        str(Path(locked["base_simulator_config"]).resolve()),
        grid_size=int(simulator["data_grid_size"]),
        grid_extent_mm=float(simulator["data_grid_extent_mm"]),
        sensor_resolution=[
            int(value) for value in simulator["data_sensor_resolution"]
        ],
        semantics=simulator["semantics"],
    )
    cases = []
    per_stratum = groups // len(STRATA)
    for stratum in STRATA:
        for index in range(per_stratum):
            case = _build_case(
                locked=locked,
                v12_config=v12_config,
                bounds=bounds,
                simulator_fixed=simulator_fixed,
                base_config_path=str(
                    Path(locked["base_simulator_config"]).resolve()
                ),
                suite_label=args.suite_label,
                stratum=stratum,
                stratum_index=index,
                excluded_group_ids=excluded_ids,
                excluded_setup_hashes=excluded_hashes,
            )
            cases.append(case)
            excluded_ids.add(str(case["group_id"]))
            excluded_hashes.add(str(case["setup_hash"]))
            print(
                json.dumps(
                    {
                        "case_id": case["case_id"],
                        "stratum": case["stratum"],
                        "regime": case["regime"],
                        "initial_normalized_distance": (
                            case["initial_normalized_distance"]
                        ),
                        "minimum_bounded_position_steps": (
                            case["minimum_bounded_position_steps"]
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    distances = np.asarray(
        [float(case["initial_normalized_distance"]) for case in cases]
    )
    low_cut, high_cut = np.quantile(distances, [1.0 / 3.0, 2.0 / 3.0])
    for case in cases:
        value = float(case["initial_normalized_distance"])
        case["initial_distance_band"] = (
            "low"
            if value <= low_cut
            else "medium"
            if value <= high_cut
            else "high"
        )
    group_ids = [str(case["group_id"]) for case in cases]
    setup_hashes = [str(case["setup_hash"]) for case in cases]
    manifest = {
        "version": "v12_mpc_h1_h3_evaluation_suite_v1",
        "schema_version": str(locked["schema_version"]),
        "simulator_semantics_version": CORRECTED_SEMANTICS_VERSION,
        "suite_label": args.suite_label,
        "locked_config": str(config_path),
        "locked_config_sha256": _sha256(config_path),
        "source_training_dataset": str(data_dir),
        "source_checkpoint": str(Path(locked["checkpoint"]).resolve()),
        "root_seed": int(locked["root_seed"]),
        "cases": cases,
        "training_action_distribution": _training_action_distribution(
            data_dir, locked["action_distribution"]
        ),
        "validation": {
            "groups": len(cases),
            "unique_group_ids": len(set(group_ids)),
            "unique_setup_hashes": len(set(setup_hashes)),
            "prior_group_id_overlap": len(set(group_ids) & excluded_ids)
            - len(cases),
            "prior_setup_hash_overlap": len(set(setup_hashes) & excluded_hashes)
            - len(cases),
            "q_goal_deployed_input": False,
            "initial_strict_successes": int(
                sum(
                    float(case["initial_normalized_distance"]) <= 1.0
                    for case in cases
                )
            ),
            "q_goal_replay_max_abs_metric_error": float(
                max(
                    case["exact_q_goal_replay_max_abs_metric_error"]
                    for case in cases
                )
            ),
            "stratum_counts": {
                stratum: int(
                    sum(case["stratum"] == stratum for case in cases)
                )
                for stratum in STRATA
            },
            "distance_band_counts": {
                band: int(
                    sum(
                        case["initial_distance_band"] == band
                        for case in cases
                    )
                )
                for band in ("low", "medium", "high")
            },
            "minimum_initial_distance": float(distances.min()),
            "maximum_initial_distance": float(distances.max()),
            "boundary_stratum_definition": (
                "camera boundary indicator true or centroid within 64 px of "
                "an image edge; clipping regime requires aperture clipping "
                "fraction > 0.01"
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    # Compute overlaps against the original exclusion sets before this build.
    original_ids, original_hashes = _prior_ids_and_hashes(
        data_dir,
        [path.resolve() for path in args.exclude_suite],
    )
    manifest["validation"]["prior_group_id_overlap"] = len(
        set(group_ids) & original_ids
    )
    manifest["validation"]["prior_setup_hash_overlap"] = len(
        set(setup_hashes) & original_hashes
    )
    if any(
        (
            manifest["validation"]["unique_group_ids"] != len(cases),
            manifest["validation"]["unique_setup_hashes"] != len(cases),
            manifest["validation"]["prior_group_id_overlap"] != 0,
            manifest["validation"]["prior_setup_hash_overlap"] != 0,
            manifest["validation"]["initial_strict_successes"] != 0,
            manifest["validation"]["q_goal_replay_max_abs_metric_error"] != 0.0,
        )
    ):
        raise RuntimeError(
            f"diagnostic suite validation failed: {manifest['validation']}"
        )
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite diagnostic suite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                "validation": manifest["validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
