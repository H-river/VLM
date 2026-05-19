"""CLI wrapper for physics-understanding diagnostic evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legacy.physics_understanding.evaluation.physics_understanding_eval import evaluate_physics_understanding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate physics-understanding probe predictions.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL")
    parser.add_argument("--probes", required=True, help="Probe JSONL")
    parser.add_argument("--out", required=True, help="Output JSON report")
    parser.add_argument("--variables-config", required=True, help="Variables YAML config")
    parser.add_argument(
        "--markdown-out",
        default=None,
        help="Output Markdown report. Defaults to the --out path with .md suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_physics_understanding(
        predictions_path=Path(args.predictions),
        probes_path=Path(args.probes),
        out_path=Path(args.out),
        variables_config_path=Path(args.variables_config),
        markdown_out_path=Path(args.markdown_out) if args.markdown_out else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
