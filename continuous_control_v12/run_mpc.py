#!/usr/bin/env python3
"""Run oracle-simulator or learned-model v12 MPC on one generated target."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuous_control_v12.contracts import Bounds, stable_seed
from continuous_control_v12.mpc import (
    CEMMPC,
    learned_predictor,
    run_closed_loop,
    simulator_predictor,
)
from continuous_control_v12.schema import read_jsonl
from continuous_control_v12.world_model import load_forward_ensemble

DEFAULT_CONFIG = Path(__file__).with_name("config_v12.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("oracle", "learned"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--targets",
        type=Path,
        help="Versioned target JSONL; defaults to DATA_DIR/targets.jsonl.",
    )
    parser.add_argument("--target-id")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--population", type=int)
    parser.add_argument("--elites", type=int)
    parser.add_argument("--cem-iterations", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    data_dir = args.data_dir.resolve()
    if "physics_structured_rebuild_v10" in data_dir.parts:
        raise ValueError("v12 MPC refuses every v10 data path")
    targets_path = (
        args.targets.resolve()
        if args.targets is not None
        else data_dir / "targets.jsonl"
    )
    targets = read_jsonl(targets_path)
    dataset_manifest = json.loads(
        (data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    target = (
        next(row for row in targets if row["target_id"] == args.target_id)
        if args.target_id
        else next(
            row for row in targets if row["category"] == "one_step_reachable"
        )
    )
    group_id = str(target["group_id"])
    initial = None
    for split in ("train", "development", "test"):
        for row in read_jsonl(data_dir / "transitions" / f"{split}.jsonl"):
            if row["group_id"] == group_id:
                initial = row
                break
        if initial is not None:
            break
    if initial is None:
        raise ValueError(f"target group not found: {group_id}")
    bounds = Bounds.from_config(config)
    base_config_path = str(
        (REPO_ROOT / config["simulator"]["base_config"]).resolve()
    )
    mpc_config = dict(config["mpc"])
    if args.smoke:
        mpc_config.update(
            {
                "horizon": int(config["smoke"]["mpc_horizon"]),
                "population": int(config["smoke"]["mpc_population"]),
                "elites": int(config["smoke"]["mpc_elites"]),
                "cem_iterations": int(config["smoke"]["mpc_iterations"]),
            }
        )
    for argument, key in (
        (args.horizon, "horizon"),
        (args.population, "population"),
        (args.elites, "elites"),
        (args.cem_iterations, "cem_iterations"),
    ):
        if argument is not None:
            if argument < 1:
                raise ValueError(f"--{key.replace('_', '-')} must be positive")
            mpc_config[key] = int(argument)
    if args.mode == "oracle":
        predictor = simulator_predictor(
            setup_context=initial["setup_context"],
            simulator_fixed=initial["simulator_fixed"],
            base_config_path=base_config_path,
            bounds=bounds,
        )
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for learned mode")
        model = load_forward_ensemble(args.checkpoint, device_name="cpu")
        predictor = learned_predictor(model, initial["setup_context"])
    planner_seed = stable_seed(
        config["seed"], target["target_id"], "matched_mpc"
    )
    planner = CEMMPC(
        bounds=bounds,
        predictor=predictor,
        config=mpc_config,
        seed=planner_seed,
    )
    episode = run_closed_loop(
        planner=planner,
        setup_context=initial["setup_context"],
        simulator_fixed=initial["simulator_fixed"],
        initial_positions_mm=initial["positions_mm"],
        initial_metrics=initial["metrics"],
        target_metrics=target["target_metrics"],
        allowed_dofs=["lens_x", "lens_y", "camera_x", "camera_y"],
        base_config_path=base_config_path,
        bounds=bounds,
        max_steps=int(args.max_steps),
    )
    result = {
        "version": "continuous_mpc_v12",
        "mode": args.mode,
        "group_id": group_id,
        "target_id": target["target_id"],
        "target_category": target["category"],
        "targets_path": str(targets_path),
        "planner_seed": planner_seed,
        "mpc_config": mpc_config,
        "regime": dataset_manifest["group_regimes"][group_id],
        "q_star_used_by_controller": False,
        "episode": episode,
        "scientific_claim": "smoke_only_no_comparison_to_v9"
        if args.smoke
        else (
            "matched_diagnostic_requires_group_level_aggregation"
            if any(
                value is not None
                for value in (
                    args.horizon,
                    args.population,
                    args.elites,
                    args.cem_iterations,
                )
            )
            else "requires_group_level_evaluation"
        ),
    }
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise RuntimeError(f"refusing to overwrite MPC output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
