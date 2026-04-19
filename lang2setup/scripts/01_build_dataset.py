#!/usr/bin/env python3
"""
01_build_dataset.py
CLI script: generate the text-supervision JSONL from simulation outputs.

Usage:
    python -m lang2setup.scripts.01_build_dataset \
        --sim-dir ../outputs/random_5k \
        --output  data/lang2setup_all.jsonl
"""
import argparse
from pathlib import Path

from lang2setup.data_prep.build_dataset import build_dataset
from lang2setup.data_prep.split import split_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-dir", required=True,
                        help="Path to simulation output directory (e.g. optical_sim/outputs/random_5k)")
    parser.add_argument("--output", default="lang2setup/data/lang2setup_all.jsonl")
    parser.add_argument("--split-output", default="lang2setup/data/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Step 1: Build full dataset
    build_dataset(args.sim_dir, args.output, seed=args.seed)

    # Step 2: Split
    split_dataset(args.output, args.split_output)


if __name__ == "__main__":
    main()
