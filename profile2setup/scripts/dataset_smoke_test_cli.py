"""Stage 3 smoke test CLI for profile2setup dataset preprocessing/loading."""

from __future__ import annotations

import argparse

try:
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "dataset_smoke_test_cli requires PyTorch. Install torch to run Stage 3 dataset loading."
    ) from exc

from profile2setup.training.dataset import Profile2SetupDataset, profile2setup_collate_fn
from profile2setup.training.text import build_vocab_from_jsonl, save_vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="profile2setup Stage 3 dataset smoke test")
    parser.add_argument("--jsonl", required=True, help="Path to input JSONL dataset")
    parser.add_argument(
        "--variables-config",
        default="profile2setup/configs/variables.yaml",
        help="Path to variables YAML config",
    )
    parser.add_argument(
        "--vocab-out",
        default="profile2setup/data/vocab.json",
        help="Where to save built vocab JSON",
    )
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--max-text-len", type=int, default=32)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument(
        "--task-filter",
        choices=["absolute", "edit", "current_only", "current-only", "paired_no_setup", "paired-no-setup"],
        default=None,
    )

    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict", dest="strict", action="store_true")
    strict_group.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vocab = build_vocab_from_jsonl(args.jsonl, min_freq=1)
    save_vocab(vocab, args.vocab_out)

    dataset = Profile2SetupDataset(
        jsonl_path=args.jsonl,
        variables_config_path=args.variables_config,
        vocab=vocab,
        input_size=args.input_size,
        max_text_len=args.max_text_len,
        normalize_mode="max_log",
        task_filter=args.task_filter,
        limit=args.limit,
        strict=args.strict,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after filtering/validation")

    probe_count = min(len(dataset), max(1, int(args.limit)))
    for idx in range(probe_count):
        _ = dataset[idx]

    batch_size = min(4, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=profile2setup_collate_fn,
    )
    batch = next(iter(loader))

    assert tuple(batch["profile"].shape) == (batch_size, 4, args.input_size, args.input_size)
    assert tuple(batch["prompt_tokens"].shape) == (batch_size, args.max_text_len)
    assert tuple(batch["current_setup"].shape) == (batch_size, 7)
    assert tuple(batch["target_setup"].shape) == (batch_size, 7)
    assert tuple(batch["target_delta"].shape) == (batch_size, 7)
    assert tuple(batch["change_mask"].shape) == (batch_size, 7)
    assert tuple(batch["setup_present"].shape) == (batch_size, 1)
    assert tuple(batch["absolute_loss_mask"].shape) == (batch_size, 1)
    assert tuple(batch["delta_loss_mask"].shape) == (batch_size, 1)
    assert tuple(batch["change_loss_mask"].shape) == (batch_size, 1)

    print(f"dataset length: {len(dataset)}")
    print(f"vocab size: {len(vocab)}")
    print(f"profile batch shape: {tuple(batch['profile'].shape)}")
    print(f"prompt token batch shape: {tuple(batch['prompt_tokens'].shape)}")
    print(f"current setup batch shape: {tuple(batch['current_setup'].shape)}")
    print(f"target setup batch shape: {tuple(batch['target_setup'].shape)}")
    print(f"target delta batch shape: {tuple(batch['target_delta'].shape)}")
    print(f"change mask batch shape: {tuple(batch['change_mask'].shape)}")
    print(f"setup present shape: {tuple(batch['setup_present'].shape)}")
    print(f"absolute loss mask shape: {tuple(batch['absolute_loss_mask'].shape)}")
    print(f"delta loss mask shape: {tuple(batch['delta_loss_mask'].shape)}")
    print(f"change loss mask shape: {tuple(batch['change_loss_mask'].shape)}")
    print(f"task types: {batch['task_type']}")
    print(f"record ids: {batch['record_id']}")


if __name__ == "__main__":
    main()
