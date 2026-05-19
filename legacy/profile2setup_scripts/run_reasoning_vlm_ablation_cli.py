"""CLI for Level 3 reasoning VLM ablation evaluation."""

from __future__ import annotations

import argparse
import json

from legacy.reasoning_vlm.evaluation.reasoning_vlm_ablation import (
    ABLATION_MODES,
    run_reasoning_vlm_ablation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run profile2setup Level 3 reasoning VLM ablations")
    parser.add_argument("--checkpoint", required=True, help="Path to trained profile2setup checkpoint")
    parser.add_argument("--data", required=True, help="Evaluation JSONL path")
    parser.add_argument("--out", default="profile2setup/results/reasoning_vlm_ablation.json")
    parser.add_argument(
        "--variables-config",
        default="profile2setup/configs/variables.yaml",
        help="Variables YAML config",
    )
    parser.add_argument("--config", default=None, help="Optional checkpoint config override")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--reasoning-mode",
        choices=["mock_rule_based", "json_file", "manual_text"],
        default="mock_rule_based",
    )
    parser.add_argument("--reasoning-json", default=None, help="Reasoning JSON for mode=json_file")
    parser.add_argument("--manual-text", default=None, help="Reasoning text for mode=manual_text")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--task-filter",
        choices=["absolute", "edit", "paired_no_setup", "paired-no-setup"],
        default=None,
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(ABLATION_MODES),
        default=None,
        help="Optional subset of ablation modes",
    )
    parser.add_argument("--run-simulator", action="store_true")
    parser.add_argument(
        "--simulation-policy",
        choices=["target_base", "current_base", "auto"],
        default="target_base",
    )
    parser.add_argument("--render-dir", default=None)
    parser.add_argument("--violation-threshold", type=float, default=1.0e-8)
    parser.add_argument("--max-saved-examples", type=int, default=8)
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict", dest="strict", action="store_true")
    strict_group.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_reasoning_vlm_ablation(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        out_path=args.out,
        variables_config_path=args.variables_config,
        config_path=args.config,
        device=args.device,
        reasoning_mode=args.reasoning_mode,
        reasoning_json=args.reasoning_json,
        manual_text=args.manual_text,
        max_examples=args.max_examples,
        task_filter=args.task_filter,
        modes=args.modes,
        run_simulator=args.run_simulator,
        simulation_policy=args.simulation_policy,
        render_dir=args.render_dir,
        strict=args.strict,
        violation_threshold=args.violation_threshold,
        max_saved_examples=args.max_saved_examples,
    )
    print(json.dumps(
        {
            "out": args.out,
            "completed_modes": result["summary"]["completed_modes"],
            "skipped_modes": result["skipped_modes"],
            "num_modes": len(result["mode_results"]),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
