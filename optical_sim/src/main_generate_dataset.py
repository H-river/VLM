#!/usr/bin/env python3
"""
main_generate_dataset.py
Entry point: generate a dataset of simulated beam profiles.

Usage examples:
    # Single run from base config
    python -m src.main_generate_dataset --mode single --config configs/base_config.yaml

    # Sweep
    python -m src.main_generate_dataset --mode sweep --config configs/sweep_config.yaml

    # Random sampling
    python -m src.main_generate_dataset --mode random --config configs/random_config.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

from .optical_elements import OpticalSetup, setup_from_dict
from .experiment_generator import (
    load_yaml,
    generate_sweep_experiments,
    generate_random_experiments,
)
from .simulator import run_simulation
from .metrics import compute_metrics, BeamMetrics
from .io_utils import save_run, save_summary


def _run_one(run_name: str,
             setup: OpticalSetup,
             output_dir: str,
             metadata_format: str = "legacy") -> dict:
    """Simulate, measure, and save a single run. Returns the metadata dict."""
    result = run_simulation(setup)
    metrics = compute_metrics(
        result["intensity"], result["sensor_X"], result["sensor_Y"]
    )
    meta = save_run(
        output_dir,
        run_name,
        setup,
        result["intensity"],
        metrics,
        metadata_format=metadata_format,
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Optical simulation dataset generator")
    parser.add_argument("--mode", choices=["single", "sweep", "random"],
                        default="single", help="Generation mode")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", default=None,
                        help="Output directory (overrides config)")
    args = parser.parse_args()

    # --- build experiment list ---
    experiments: List[Tuple[str, OpticalSetup]] = []
    metadata_format = "v2" if Path(args.config).name.endswith("_v2.yaml") else "legacy"

    if args.mode == "single":
        cfg = load_yaml(args.config)
        experiments.append(("single_run", setup_from_dict(cfg)))
        output_dir = args.output or "outputs/single"

    elif args.mode == "sweep":
        experiments = generate_sweep_experiments(args.config)
        scfg = load_yaml(args.config)
        output_dir = args.output or scfg.get("output_dir", "outputs/sweep")

    elif args.mode == "random":
        experiments = generate_random_experiments(args.config)
        rcfg = load_yaml(args.config)
        output_dir = args.output or rcfg.get("output_dir", "outputs/random")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"▶ Mode={args.mode}  |  {len(experiments)} experiment(s)  |  output → {output_dir}")

    # --- run all experiments ---
    records = []
    t0 = time.time()
    for i, (name, setup) in enumerate(experiments):
        t1 = time.time()
        meta = _run_one(name, setup, output_dir, metadata_format=metadata_format)
        elapsed = time.time() - t1
        records.append(meta)
        print(f"  [{i + 1}/{len(experiments)}] {name}  ({elapsed:.2f}s)")

    # --- global summary ---
    save_summary(output_dir, records)
    total = time.time() - t0
    print(f"✓ Done — {len(records)} runs in {total:.1f}s.  Summary → {output_dir}/summary.jsonl")


if __name__ == "__main__":
    main()
