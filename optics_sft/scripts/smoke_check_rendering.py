#!/usr/bin/env python3
"""Smoke check for rendering simulator intensities to RGB PNGs."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optical_sim.src.experiment_generator import load_yaml
from optical_sim.src.optical_elements import setup_from_dict
from optics_sft.physics.rendering import random_render_params, save_intensity_png
from optics_sft.physics.sim_adapter import simulate_and_measure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one simulated beam at three difficulty levels.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../VLM_data/physics_sft_smoke/images"),
        help="Directory where smoke PNGs will be written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml("optical_sim/configs/base_config.yaml")
    setup = setup_from_dict(config)
    result = simulate_and_measure(setup)
    intensity = result["intensity"]

    rng = random.Random(args.seed)
    written: list[Path] = []
    for difficulty in ("clean", "medium", "hard"):
        options = random_render_params(rng, difficulty=difficulty)
        path = args.output_dir / f"beam_{difficulty}.png"
        written.append(save_intensity_png(intensity, path, options))

    for path in written:
        print(f"wrote {path}")
    print("rendering smoke check: OK")


if __name__ == "__main__":
    main()
