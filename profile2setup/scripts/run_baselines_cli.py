"""CLI for offline profile2setup baseline evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from profile2setup.evaluation.baselines import (
    MeanAbsoluteBaseline,
    MeanDeltaBaseline,
    NearestNeighborProfileBaseline,
    ZeroDeltaMeanAbsoluteBaseline,
)
from profile2setup.evaluation.evaluate_model import (
    _check_no_forbidden_keys,
    evaluate_outputs_over_loader,
)
from profile2setup.training.dataset import Profile2SetupDataset, profile2setup_collate_fn
from profile2setup.training.normalization import load_variables_config
from profile2setup.training.text import build_vocab_from_jsonl_files, load_vocab, save_vocab
from profile2setup.training.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate profile2setup v2 baselines")
    parser.add_argument("--train", required=True, help="Training JSONL for baseline fitting")
    parser.add_argument("--test", required=True, help="Test JSONL for baseline evaluation")
    parser.add_argument(
        "--variables-config",
        default="profile2setup/configs/variables.yaml",
        help="Variables YAML config",
    )
    parser.add_argument("--vocab", default=None, help="Optional vocab JSON")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--max-text-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="profile2setup/results/baselines.json")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--include-nearest-neighbor", action="store_true")
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict", dest="strict", action="store_true")
    strict_group.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def _build_vocab(args) -> dict:
    if args.vocab is not None and Path(args.vocab).exists():
        return load_vocab(args.vocab)
    vocab = build_vocab_from_jsonl_files([args.train, args.test], min_freq=1)
    if args.vocab is not None:
        save_vocab(vocab, args.vocab)
    return vocab


def _build_dataset(path, args, vocab):
    return Profile2SetupDataset(
        jsonl_path=path,
        variables_config_path=args.variables_config,
        vocab=vocab,
        input_size=args.input_size,
        max_text_len=args.max_text_len,
        normalize_mode="max_log",
        task_filter=None,
        limit=None,
        strict=args.strict,
        change_threshold=1.0e-6,
    )


def main() -> None:
    args = parse_args()
    vocab = _build_vocab(args)
    variables_config = load_variables_config(args.variables_config)
    train_dataset = _build_dataset(args.train, args, vocab)
    test_dataset = _build_dataset(args.test, args, vocab)
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=profile2setup_collate_fn,
    )
    config = {
        "loss": {
            "absolute_weight": 1.0,
            "delta_weight": 1.0,
            "change_weight": 0.5,
        },
        "data": {
            "input_size": args.input_size,
            "max_text_len": args.max_text_len,
            "normalize_mode": "max_log",
            "change_threshold": 1.0e-6,
        },
    }

    baselines = [
        MeanAbsoluteBaseline(variables_config_path=args.variables_config),
        MeanDeltaBaseline(variables_config_path=args.variables_config),
        ZeroDeltaMeanAbsoluteBaseline(variables_config_path=args.variables_config),
    ]
    if args.include_nearest_neighbor:
        baselines.append(NearestNeighborProfileBaseline(variables_config_path=args.variables_config))

    results = {
        "data": {
            "train": args.train,
            "test": args.test,
        },
        "baselines": {},
    }
    device = torch.device("cpu")
    for baseline in baselines:
        baseline.fit(train_dataset)
        metrics = evaluate_outputs_over_loader(
            baseline.predict_batch,
            loader,
            variables_config,
            config,
            device,
            max_examples=args.max_examples,
        )
        results["baselines"][baseline.name()] = metrics
        routed_mae = metrics["physical_metrics"]["routed_setup"]["mae"]
        print(f"{baseline.name()} routed setup physical MAE: {routed_mae}")

    _check_no_forbidden_keys(results)
    save_json(results, args.out)
    print(f"saved result JSON: {args.out}")


if __name__ == "__main__":
    main()
