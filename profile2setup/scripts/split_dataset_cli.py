"""CLI for splitting profile2setup JSONL datasets."""

from __future__ import annotations

import argparse

from profile2setup.data_prep.split import split_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Split profile2setup JSONL dataset")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--out-dir", required=True, help="Output directory for splits")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_jsonl(
        input_path=args.input,
        out_dir=args.out_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
