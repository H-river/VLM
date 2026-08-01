#!/usr/bin/env python3
"""Generate simulator-derived reachable and mismatched v12 target profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuous_control_v12.contracts import (
    Bounds,
    apply_action,
    position_dict,
    position_vector,
    project_action,
    stable_seed,
)
from continuous_control_v12.reachability import (
    discrete_81_oracle,
    estimate_continuous_oracle,
    oracle_gap,
)
from continuous_control_v12.schema import read_jsonl, validate_json_schema
from continuous_control_v12.simulator import simulate_state

DEFAULT_CONFIG = Path(__file__).with_name("config_v12.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify-candidate-infeasible",
        action="store_true",
        help="Run the expensive multi-start continuous oracle on mismatched targets.",
    )
    parser.add_argument("--oracle-multistarts", type=int)
    parser.add_argument("--oracle-max-iterations", type=int)
    parser.add_argument(
        "--split",
        choices=("train", "development", "test"),
        help="Restrict target construction to one group-disjoint split.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        help="Use only the first N deterministic groups from the selected split(s).",
    )
    parser.add_argument(
        "--group-id",
        action="append",
        dest="group_ids",
        help="Select an exact group ID; repeat for a fixed stratified target set.",
    )
    return parser.parse_args()


def first_group_rows(
    data_dir: Path, selected_split: str | None = None
) -> list[dict[str, Any]]:
    rows = []
    splits = (
        (selected_split,)
        if selected_split is not None
        else ("train", "development", "test")
    )
    for split in splits:
        seen: set[str] = set()
        for row in read_jsonl(data_dir / "transitions" / f"{split}.jsonl"):
            group_id = str(row["group_id"])
            if group_id not in seen:
                seen.add(group_id)
                rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    data_dir = args.data_dir.resolve()
    if "physics_structured_rebuild_v10" in data_dir.parts:
        raise ValueError("v12 target generation refuses every v10 data path")
    output = (
        args.output.resolve()
        if args.output is not None
        else data_dir / "targets.jsonl"
    )
    if output.exists():
        raise RuntimeError(f"refusing to overwrite targets: {output}")
    bounds = Bounds.from_config(config)
    dataset_manifest = json.loads(
        (data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    store_target_images = bool(
        dataset_manifest["coverage"].get("images_stored", False)
    )
    base_config_path = str(
        (REPO_ROOT / config["simulator"]["base_config"]).resolve()
    )
    groups = first_group_rows(data_dir, args.split)
    if args.group_ids:
        if args.max_groups is not None:
            raise ValueError("--group-id cannot be combined with --max-groups")
        by_id = {str(row["group_id"]): row for row in groups}
        missing = [group for group in args.group_ids if group not in by_id]
        if missing:
            raise ValueError(f"requested target groups not found: {missing}")
        groups = [by_id[group] for group in args.group_ids]
    if args.max_groups is not None:
        if args.max_groups < 2:
            raise ValueError("--max-groups must be at least 2")
        groups = groups[: int(args.max_groups)]
    if len(groups) < 2:
        raise ValueError("target construction requires at least two groups")
    generated: list[list[dict[str, Any]]] = []
    for row in groups:
        group_id = str(row["group_id"])
        current_positions = position_vector(row["positions_mm"])
        current_metrics = row["metrics"]
        rng = np.random.default_rng(
            stable_seed(config["seed"], group_id, "targets")
        )
        one_action = rng.uniform(
            bounds.action_low * 0.9, bounds.action_high * 0.9
        )
        one_action = project_action(current_positions, one_action, bounds)
        one_q = apply_action(current_positions, one_action, bounds)
        one_capture = simulate_state(
            row["setup_context"],
            position_dict(one_q),
            row["simulator_fixed"],
            base_config_path,
            bounds,
        )
        multi_delta = rng.uniform(1.5, 3.2, size=4) * bounds.action_high
        multi_delta *= rng.choice((-1.0, 1.0), size=4)
        axis = int(rng.integers(0, 4))
        multi_delta[axis] = (
            rng.choice((-1.0, 1.0))
            * bounds.action_high[axis]
            * rng.uniform(2.1, 3.2)
        )
        multi_q = np.clip(
            current_positions + multi_delta,
            bounds.position_low,
            bounds.position_high,
        )
        multi_capture = simulate_state(
            row["setup_context"],
            position_dict(multi_q),
            row["simulator_fixed"],
            base_config_path,
            bounds,
        )
        one_image_ref = None
        multi_image_ref = None
        if store_target_images:
            image_dir = data_dir / "target_images"
            image_dir.mkdir(parents=True, exist_ok=True)
            one_image_ref = f"target_images/{group_id}_one_step.npz"
            multi_image_ref = f"target_images/{group_id}_multi_step.npz"
            np.savez_compressed(
                data_dir / one_image_ref, intensity=one_capture["intensity"]
            )
            np.savez_compressed(
                data_dir / multi_image_ref, intensity=multi_capture["intensity"]
            )
        generated.append(
            [
                {
                    "schema_version": config["schema_version"],
                    "target_id": f"{group_id}_one_step",
                    "group_id": group_id,
                    "category": "one_step_reachable",
                    "target_metrics": one_capture["metrics"],
                    "target_image_ref": one_image_ref,
                    "tolerance_reference_metrics": current_metrics,
                    "oracle_metadata": {
                        "q_star_mm": position_dict(one_q),
                        "generating_action_mm": {
                            field: float(one_action[index])
                            for index, field in enumerate(
                                (
                                    "lens_x_delta_mm",
                                    "lens_y_delta_mm",
                                    "camera_x_delta_mm",
                                    "camera_y_delta_mm",
                                )
                            )
                        },
                        "visibility": "supervision_only_not_deployed_input",
                    },
                    "estimated_reachability": {
                        "label": "clearly_reachable",
                        "best_distance": 0.0,
                        "witness": "generating_q_star_simulator_replay",
                    },
                    "discrete_81_oracle": None,
                },
                {
                    "schema_version": config["schema_version"],
                    "target_id": f"{group_id}_multi_step",
                    "group_id": group_id,
                    "category": "multi_step_reachable",
                    "target_metrics": multi_capture["metrics"],
                    "target_image_ref": multi_image_ref,
                    "tolerance_reference_metrics": current_metrics,
                    "oracle_metadata": {
                        "q_star_mm": position_dict(multi_q),
                        "minimum_bounded_position_steps_lower_bound": int(
                            np.ceil(
                                np.max(
                                    np.abs(multi_q - current_positions)
                                    / bounds.action_high
                                )
                            )
                        ),
                        "visibility": "supervision_only_not_deployed_input",
                    },
                    "estimated_reachability": {
                        "label": "clearly_reachable",
                        "best_distance": 0.0,
                        "witness": "generating_q_star_simulator_replay",
                    },
                    "discrete_81_oracle": None,
                },
            ]
        )
    targets: list[dict[str, Any]] = []
    for index, group_targets in enumerate(generated):
        row = groups[index]
        targets.extend(group_targets)
        source = generated[(index + 1) % len(generated)][1]
        candidate = {
            "schema_version": config["schema_version"],
            "target_id": f"{row['group_id']}_mismatched_candidate",
            "group_id": str(row["group_id"]),
            "category": "ambiguous_boundary",
            "target_metrics": source["target_metrics"],
            "target_image_ref": source["target_image_ref"],
            "tolerance_reference_metrics": row["metrics"],
            "oracle_metadata": {
                "mismatched_profile_source_group_id": source["group_id"],
                "construction": "simulator_derived_profile_from_different_setup",
                "visibility": "audit_only_not_deployed_input",
            },
            "estimated_reachability": None,
            "discrete_81_oracle": None,
        }
        discrete = discrete_81_oracle(
            setup_context=row["setup_context"],
            simulator_fixed=row["simulator_fixed"],
            current_positions_mm=row["positions_mm"],
            current_metrics=row["metrics"],
            target_metrics=candidate["target_metrics"],
            base_config_path=base_config_path,
            bounds=bounds,
        )
        candidate["discrete_81_oracle"] = discrete
        if args.verify_candidate_infeasible:
            oracle_config = config["oracle"]
            estimate = estimate_continuous_oracle(
                setup_context=row["setup_context"],
                simulator_fixed=row["simulator_fixed"],
                current_positions_mm=row["positions_mm"],
                current_metrics=row["metrics"],
                target_metrics=candidate["target_metrics"],
                base_config_path=base_config_path,
                bounds=bounds,
                seed=stable_seed(config["seed"], row["group_id"], "oracle"),
                multistarts=int(
                    args.oracle_multistarts or oracle_config["multistarts"]
                ),
                max_iterations=int(
                    args.oracle_max_iterations or oracle_config["max_iterations"]
                ),
                reachable_below=float(
                    oracle_config["clearly_reachable_below"]
                ),
                infeasible_above=float(
                    oracle_config["candidate_infeasible_above"]
                ),
                agreement_tolerance=float(
                    oracle_config["agreement_tolerance"]
                ),
            )
            candidate["estimated_reachability"] = estimate
            candidate["oracle_gap"] = oracle_gap(estimate, discrete)
            candidate["category"] = (
                "candidate_infeasible"
                if estimate["label"] == "candidate_infeasible"
                else "estimated_reachable"
                if estimate["label"] == "clearly_reachable"
                else "ambiguous_boundary"
            )
        targets.append(candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    for target in targets:
        validate_json_schema(target, "target_v12.schema.json")
        if target["target_image_ref"] is not None and not (
            data_dir / target["target_image_ref"]
        ).is_file():
            raise ValueError(
                f"missing target image: {target['target_image_ref']}"
            )
    with output.open("w", encoding="utf-8") as stream:
        for target in targets:
            stream.write(
                json.dumps(target, sort_keys=True, separators=(",", ":")) + "\n"
            )
    target_manifest = {
        "version": "continuous_control_v12_targets",
        "schema_version": config["schema_version"],
        "targets": len(targets),
        "category_counts": {
            category: sum(row["category"] == category for row in targets)
            for category in sorted({row["category"] for row in targets})
        },
        "jsonl": str(output),
        "jsonl_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "target_images_stored": store_target_images,
        "q_star_location": "oracle_metadata_only",
        "q_star_deployed_input": False,
        "candidate_infeasible_requires_repeated_optimizer_agreement": True,
        "candidate_oracle_verified": bool(args.verify_candidate_infeasible),
        "selected_split": args.split,
        "max_groups": args.max_groups,
        "selected_group_ids": [str(row["group_id"]) for row in groups],
    }
    manifest_path = output.with_name("targets_manifest.json")
    manifest_path.write_text(
        json.dumps(target_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": str(manifest_path),
                "targets": len(targets),
                "candidate_oracle_verified": bool(
                    args.verify_candidate_infeasible
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
