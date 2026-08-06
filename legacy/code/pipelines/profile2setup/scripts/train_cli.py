"""Command-line entry point for profile2setup Stage 5 training."""

from __future__ import annotations

import argparse

from profile2setup.training.train import train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train profile2setup v2")
    parser.add_argument(
        "--config",
        default="profile2setup/configs/train.yaml",
        help="Path to training YAML config",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run tiny training")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--train-path", default=None, help="Override data.train_path")
    parser.add_argument("--val-path", default=None, help="Override data.val_path")
    parser.add_argument("--epochs", type=int, default=None, help="Override optimization.epochs")
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=None,
        help="When resuming, train this many more epochs after the checkpoint epoch",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override optimization.batch_size")
    parser.add_argument("--run-name", default=None, help="Override experiment.name")
    parser.add_argument("--resume-checkpoint", default=None, help="Continue training from a checkpoint")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.run_name is not None:
        overrides.setdefault("experiment", {})["name"] = args.run_name
    if args.device is not None:
        overrides.setdefault("optimization", {})["device"] = args.device
    if args.epochs is not None:
        overrides.setdefault("optimization", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("optimization", {})["batch_size"] = args.batch_size
    if args.train_path is not None:
        overrides.setdefault("data", {})["train_path"] = args.train_path
    if args.val_path is not None:
        overrides.setdefault("data", {})["val_path"] = args.val_path
    return overrides


def main() -> None:
    args = parse_args()
    result = train_from_config(
        args.config,
        overrides=build_overrides(args),
        smoke_test=args.smoke_test,
        resume_checkpoint_path=args.resume_checkpoint,
        additional_epochs=args.additional_epochs,
    )
    print(f"best validation loss: {result['best_val_loss']:.6f}")
    print(f"best checkpoint path: {result['best_checkpoint_path']}")


if __name__ == "__main__":
    main()
