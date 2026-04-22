"""CLI for building profile2setup edit dataset."""

from __future__ import annotations

import argparse

from profile2setup.data_prep.build_edit_dataset import build_edit_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build profile2setup edit dataset")
    parser.add_argument("--sim-dir", required=True, help="Simulator output directory")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--num-pairs", type=int, required=True, help="Number of edit pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--prompts",
        default="profile2setup/configs/prompts.yaml",
        help="Prompts YAML path",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional sample limit before pairing",
    )
    parser.add_argument(
        "--allow-self-pairs",
        action="store_true",
        help="Allow pairing a sample with itself",
    )

    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict", dest="strict", action="store_true")
    strict_group.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)

    args = parser.parse_args()

    build_edit_dataset(
        sim_dir=args.sim_dir,
        out_path=args.out,
        num_pairs=args.num_pairs,
        prompts_path=args.prompts,
        strict=args.strict,
        seed=args.seed,
        limit_samples=args.limit_samples,
        allow_self_pairs=args.allow_self_pairs,
    )


if __name__ == "__main__":
    main()
