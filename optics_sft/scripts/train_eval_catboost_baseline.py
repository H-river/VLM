#!/usr/bin/env python3
"""Train a simple CatBoost baseline and emit the same metrics as eval_text_sft_benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optics_sft.baselines.catboost_inverse import (
    load_models,
    predict_control_plans,
    save_models,
    train_models,
)
from optics_sft.eval.control_metrics import summarize_benchmark
from optics_sft.utils.paths import resolve_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/eval CatBoost inverse-control baseline on text SFT JSONL rows."
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("../VLM_runs/catboost_text_inverse_v1"),
        help="Directory to save/load CatBoost .cbm models.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training and load models from --model-dir.",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    train_path = resolve_config_path(args.train_jsonl)
    eval_path = resolve_config_path(args.eval_jsonl)
    model_dir = resolve_config_path(args.model_dir)
    output_report = resolve_config_path(args.output_report)

    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)

    extra_params: dict[str, Any] = {}
    if args.iterations is not None:
        extra_params["iterations"] = args.iterations
    if args.depth is not None:
        extra_params["depth"] = args.depth

    if args.predict_only:
        models, feature_cols = load_models(model_dir)
        print(f"Loaded CatBoost models from {model_dir}")
    else:
        print(
            f"Training CatBoost on {len(train_rows)} rows "
            f"(val monitor: {len(eval_rows)} rows from eval-jsonl)..."
        )
        models, feature_cols = train_models(
            train_rows,
            val_rows=eval_rows,
            params=extra_params or None,
        )
        save_models(models, model_dir, feature_cols=feature_cols, params=extra_params or None)
        print(f"Saved models to {model_dir}")

    predictions = predict_control_plans(models, eval_rows, feature_cols=feature_cols)
    report = summarize_benchmark(predictions, eval_rows, model_name="catboost")
    report["train_jsonl"] = str(train_path)
    report["eval_jsonl"] = str(eval_path)
    report["model_dir"] = str(model_dir)
    report["predict_only"] = args.predict_only
    report["num_features"] = len(feature_cols)

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
