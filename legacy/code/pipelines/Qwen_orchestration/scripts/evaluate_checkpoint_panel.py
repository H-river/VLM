#!/usr/bin/env python3
"""Run diagnostic gates, full validation, and frozen checkpoint selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Qwen_orchestration.scripts.select_checkpoint import assess, select, selection_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--qwen-jsonl", type=Path, required=True)
    parser.add_argument("--canonical-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--full-candidates",
        type=int,
        default=2,
        help="Number of strongest diagnostic passers to run on full validation.",
    )
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    return int(path.name.removeprefix("checkpoint-"))


def run_evaluation(
    args: argparse.Namespace, checkpoint: Path, subset: str
) -> Path:
    stem = f"{checkpoint.name}_{subset}"
    output = args.results_dir / f"{stem}.jsonl"
    summary = args.results_dir / f"{stem}.summary.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "Qwen_orchestration/scripts/evaluate_qwen.py"),
        "--config",
        str(args.config),
        "--input-jsonl",
        str(args.qwen_jsonl),
        "--canonical-jsonl",
        str(args.canonical_jsonl),
        "--image-root",
        str(args.image_root),
        "--output-jsonl",
        str(output),
        "--summary-json",
        str(summary),
        "--adapter-path",
        str(checkpoint),
        "--stage",
        args.stage,
        "--subset",
        subset,
        "--batch-size",
        str(args.batch_size),
        "--resume",
    ]
    subprocess.run(command, check=True)
    return summary


def write_selection(
    args: argparse.Namespace,
    diagnostic: list[dict[str, Any]],
    full_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    full_assessments, selected = select(full_reports, args.stage)
    report = {
        "stage": args.stage,
        "policy": (
            "diagnostic gates first; full validation for the strongest configured "
            "number of diagnostic passers; all full gates; lexicographic metrics; "
            "earliest checkpoint tie-break"
        ),
        "diagnostic_candidates": diagnostic,
        "full_validation_candidates": full_assessments,
        "selected_checkpoint": selected,
    }
    path = args.results_dir / "selection.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    checkpoints = sorted(
        (
            path
            for path in args.run_dir.glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
        ),
        key=checkpoint_step,
    )
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints under {args.run_dir}")
    if args.full_candidates < 1:
        raise ValueError("--full-candidates must be positive")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_assessments = []
    passing_checkpoints = []
    for checkpoint in checkpoints:
        summary_path = run_evaluation(args, checkpoint, "diagnostic")
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        assessment = assess(report, args.stage)
        diagnostic_assessments.append(assessment)
        if assessment["passed"]:
            passing_checkpoints.append(checkpoint)
    if not passing_checkpoints:
        report = write_selection(args, diagnostic_assessments, [])
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(2)
    diagnostic_by_step = {
        item["checkpoint_step"]: item for item in diagnostic_assessments
    }
    passing_checkpoints.sort(
        key=lambda checkpoint: selection_key(
            diagnostic_by_step[checkpoint_step(checkpoint)], args.stage
        ),
        reverse=True,
    )
    full_reports = []
    for checkpoint in passing_checkpoints[: args.full_candidates]:
        summary_path = run_evaluation(args, checkpoint, "all")
        full_reports.append(json.loads(summary_path.read_text(encoding="utf-8")))
    report = write_selection(args, diagnostic_assessments, full_reports)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["selected_checkpoint"] is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
