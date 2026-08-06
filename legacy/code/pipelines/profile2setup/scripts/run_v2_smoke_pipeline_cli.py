"""Run a small end-to-end profile2setup v2 smoke workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


SMOKE_RUN_NAME = "profile2setup_v2_smoke_pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the profile2setup v2 smoke pipeline")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--config", default="profile2setup/configs/train.yaml")
    parser.add_argument("--variables-config", default="profile2setup/configs/variables.yaml")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-closed-loop", action="store_true")
    parser.add_argument("--max-closed-loop-examples", type=int, default=5)
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict", dest="strict", action="store_true")
    strict_group.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def _load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping: {cfg_path}")
    return cfg


def _checkpoint_path(config_path: str, *, skip_training: bool) -> Path:
    config = _load_config(config_path)
    checkpoint_cfg = config.get("checkpoint") or {}
    experiment_cfg = config.get("experiment") or {}
    save_dir = Path(checkpoint_cfg.get("save_dir", "profile2setup/checkpoints"))
    run_name = experiment_cfg.get("name", "profile2setup_v2") if skip_training else SMOKE_RUN_NAME
    return save_dir / str(run_name) / "best.pt"


def _run_step(name: str, command: list[str], *, strict: bool) -> bool:
    print(f"\n== {name} ==")
    print(" ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode == 0:
        print(f"{name}: passed")
        return True

    message = f"{name}: failed with exit code {completed.returncode}"
    if strict:
        print(message)
        sys.exit(completed.returncode)
    print(f"{message}; continuing because --no-strict is set")
    return False


def _strict_flag(strict: bool) -> str:
    return "--strict" if strict else "--no-strict"


def main() -> None:
    args = parse_args()
    Path("profile2setup/results").mkdir(parents=True, exist_ok=True)

    python = sys.executable
    strict_flag = _strict_flag(args.strict)
    checkpoint = _checkpoint_path(args.config, skip_training=args.skip_training)

    steps: list[tuple[str, list[str]]] = [
        (
            "dataset smoke test",
            [
                python,
                "-m",
                "profile2setup.scripts.dataset_smoke_test_cli",
                "--jsonl",
                args.train_jsonl,
                "--variables-config",
                args.variables_config,
                "--input-size",
                "128",
                "--max-text-len",
                "32",
                "--limit",
                "4",
                strict_flag,
            ],
        ),
        (
            "model smoke test",
            [
                python,
                "-m",
                "profile2setup.scripts.model_smoke_test_cli",
                "--vocab-size",
                "100",
                "--batch-size",
                "2",
                "--input-size",
                "128",
                "--text-len",
                "32",
                "--config",
                args.config,
            ],
        ),
    ]

    if not args.skip_training:
        steps.append(
            (
                "training smoke test",
                [
                    python,
                    "-m",
                    "profile2setup.scripts.train_cli",
                    "--config",
                    args.config,
                    "--smoke-test",
                    "--train-path",
                    args.train_jsonl,
                    "--val-path",
                    args.val_jsonl,
                    "--run-name",
                    SMOKE_RUN_NAME,
                ],
            )
        )

    if not args.skip_eval:
        steps.extend(
            [
                (
                    "baseline evaluation smoke",
                    [
                        python,
                        "-m",
                        "profile2setup.scripts.run_baselines_cli",
                        "--train",
                        args.train_jsonl,
                        "--test",
                        args.test_jsonl,
                        "--variables-config",
                        args.variables_config,
                        "--out",
                        "profile2setup/results/baselines.json",
                        "--max-examples",
                        "16",
                        strict_flag,
                    ],
                ),
                (
                    "model offline evaluation smoke",
                    [
                        python,
                        "-m",
                        "profile2setup.scripts.evaluate_cli",
                        "--checkpoint",
                        str(checkpoint),
                        "--data",
                        args.test_jsonl,
                        "--out",
                        "profile2setup/results/model_eval.json",
                        "--config",
                        args.config,
                        "--variables-config",
                        args.variables_config,
                        "--batch-size",
                        "4",
                        "--max-examples",
                        "16",
                        strict_flag,
                    ],
                ),
            ]
        )

    if not args.skip_closed_loop:
        steps.append(
            (
                "closed-loop smoke evaluation",
                [
                    python,
                    "-m",
                    "profile2setup.scripts.closed_loop_eval_cli",
                    "--checkpoint",
                    str(checkpoint),
                    "--data",
                    args.test_jsonl,
                    "--out",
                    "profile2setup/results/closed_loop_smoke.json",
                    "--variables-config",
                    args.variables_config,
                    "--config",
                    args.config,
                    "--simulation-policy",
                    "target_base",
                    "--max-examples",
                    str(args.max_closed_loop_examples),
                    strict_flag,
                ],
            )
        )

    steps.append(
        (
            "v2 integrity check",
            [
                python,
                "-m",
                "profile2setup.scripts.check_v2_integrity_cli",
                "--root",
                "profile2setup",
                "--data-dir",
                str(Path(args.train_jsonl).parent.parent if Path(args.train_jsonl).parent.name in {"train", "val", "test"} else Path("profile2setup/data")),
                "--results-dir",
                "profile2setup/results",
                strict_flag,
            ],
        )
    )

    failures = 0
    for name, command in steps:
        if not _run_step(name, command, strict=args.strict):
            failures += 1

    if failures:
        print(f"\nprofile2setup v2 smoke pipeline completed with {failures} failed step(s)")
        return
    print("\nprofile2setup v2 smoke pipeline passed")


if __name__ == "__main__":
    main()
