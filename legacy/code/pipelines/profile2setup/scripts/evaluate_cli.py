"""CLI for offline profile2setup checkpoint evaluation."""

from __future__ import annotations

import argparse

from profile2setup.evaluation.evaluate_model import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a profile2setup v2 checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to Stage 5 checkpoint")
    parser.add_argument("--data", required=True, help="Path to evaluation JSONL")
    parser.add_argument("--out", default="profile2setup/results/model_eval.json", help="Output JSON path")
    parser.add_argument("--config", default=None, help="Optional config override")
    parser.add_argument(
        "--variables-config",
        default="profile2setup/configs/variables.yaml",
        help="Variables YAML config",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--task-filter",
        choices=["absolute", "edit", "paired_no_setup", "paired-no-setup"],
        default=None,
    )
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict", dest="strict", action="store_true")
    strict_group.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        out_path=args.out,
        config_path=args.config,
        variables_config_path=args.variables_config,
        batch_size=args.batch_size,
        device=args.device,
        max_examples=args.max_examples,
        task_filter=args.task_filter,
        strict=args.strict,
    )
    print(f"saved result JSON: {args.out}")
    print(f"final routed setup physical MAE: {result['physical_metrics']['routed_setup']['mae']}")


if __name__ == "__main__":
    main()
