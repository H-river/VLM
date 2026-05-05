"""Evaluate multimodal LLM API predictions for profile2setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from profile2setup.evaluation.llm_api_eval import evaluate_llm_api_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate profile2setup LLM API prediction JSONL.")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL from run_llm_api_inference_cli")
    parser.add_argument("--data", required=True, help="Original profile2setup test JSONL")
    parser.add_argument("--out", required=True, help="Output JSON evaluation report")
    parser.add_argument("--variables-config", default=None, help="Variables YAML with tolerances")
    parser.add_argument("--run-simulator", action="store_true", help="Run optional simulator profile metrics")
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save GT-vs-predicted simulator comparison PNG/JSON files",
    )
    parser.add_argument(
        "--viz-out-dir",
        default="profile2setup/results/llm_api_eval/viz",
        help="Directory for visualization PNG/JSON outputs",
    )
    parser.add_argument("--max-viz-examples", type=int, default=20, help="Maximum visualized examples to save")
    parser.add_argument(
        "--simulation-policy",
        choices=("target_base", "current_base", "auto"),
        default="target_base",
    )
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum non-skipped prediction rows to evaluate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_llm_api_predictions(
        predictions_path=Path(args.predictions),
        data_path=Path(args.data),
        out_path=Path(args.out),
        variables_config_path=Path(args.variables_config) if args.variables_config else None,
        run_simulator=args.run_simulator,
        simulation_policy=args.simulation_policy,
        save_visualizations=args.save_visualizations,
        viz_out_dir=Path(args.viz_out_dir),
        max_viz_examples=args.max_viz_examples,
        max_examples=args.max_examples,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
