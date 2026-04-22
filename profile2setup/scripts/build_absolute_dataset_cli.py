"""CLI for building profile2setup absolute dataset."""

from __future__ import annotations

import argparse

from profile2setup.data_prep.build_absolute_dataset import build_absolute_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build profile2setup absolute dataset")
    parser.add_argument("--sim-dir", required=True, help="Simulator output directory")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument(
        "--prompts",
        default="profile2setup/configs/prompts.yaml",
        help="Prompts YAML path",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict", dest="strict", action="store_true")
    strict_group.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)

    args = parser.parse_args()

    build_absolute_dataset(
        sim_dir=args.sim_dir,
        out_path=args.out,
        prompts_path=args.prompts,
        strict=args.strict,
        limit=args.limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
