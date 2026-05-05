"""CLI for profile2setup closed-loop simulator evaluation."""

from __future__ import annotations

import argparse

from profile2setup.evaluation.closed_loop import run_closed_loop_evaluation
from profile2setup.schema import VARIABLE_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run profile2setup v2 closed-loop simulation evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to trained profile2setup checkpoint")
    parser.add_argument("--data", required=True, help="Path to evaluation JSONL")
    parser.add_argument("--out", default="profile2setup/results/closed_loop.json", help="Output JSON path")
    parser.add_argument(
        "--variables-config",
        default="profile2setup/configs/variables.yaml",
        help="Variables YAML config",
    )
    parser.add_argument("--config", default=None, help="Optional config override")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--task-filter",
        choices=["absolute", "edit", "paired_no_setup", "paired-no-setup"],
        default=None,
    )
    parser.add_argument(
        "--simulation-policy",
        choices=["target_base", "current_base", "auto"],
        default="target_base",
    )
    parser.add_argument("--save-predicted-profiles-dir", default=None)
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict", dest="strict", action="store_true")
    strict_group.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def _fmt(value) -> str:
    if value is None:
        return "none"
    return f"{float(value):.6g}"


def main() -> None:
    args = parse_args()
    result = run_closed_loop_evaluation(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        out_path=args.out,
        variables_config_path=args.variables_config,
        config_path=args.config,
        device=args.device,
        max_examples=args.max_examples,
        task_filter=args.task_filter,
        strict=args.strict,
        simulation_policy=args.simulation_policy,
        save_predicted_profiles_dir=args.save_predicted_profiles_dir,
    )

    aggregate = result.get("aggregate") or {}
    profile_mean = aggregate.get("profile_metrics_mean") or {}
    param_mean = aggregate.get("parameter_error_mean") or {}
    centroid_x = profile_mean.get("centroid_x_error_px")
    centroid_y = profile_mean.get("centroid_y_error_px")
    sigma_x = profile_mean.get("sigma_x_error_px")
    sigma_y = profile_mean.get("sigma_y_error_px")

    print(f"saved result JSON: {args.out}")
    print(f"number seen: {result['num_records_seen']}")
    print(f"number evaluated: {result['num_profile_evaluated']}")
    print(f"number skipped: {result['num_skipped']}")
    print(f"mean normalized MSE: {_fmt(profile_mean.get('normalized_mse'))}")
    print(f"mean centroid error px: x={_fmt(centroid_x)} y={_fmt(centroid_y)}")
    print(f"mean sigma error px: x={_fmt(sigma_x)} y={_fmt(sigma_y)}")
    print("mean parameter error:")
    for name in VARIABLE_ORDER:
        print(f"  {name}: {_fmt(param_mean.get(name))}")


if __name__ == "__main__":
    main()
