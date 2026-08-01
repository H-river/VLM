#!/usr/bin/env python3
"""Resume generation, verify it, and conditionally train all specialists."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--worker-affinities",
        default="0;8;1,9",
        help="Semicolon-separated CPU sets forwarded to the dataset builder.",
    )
    parser.add_argument("--objective-start-epoch", type=float, required=True)
    parser.add_argument("--training-deadline-hours", type=float, default=12.0)
    return parser.parse_args()


def write_state(path: Path, **values: Any) -> None:
    current: dict[str, Any] = {}
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(values)
    current["updated_at_epoch"] = time.time()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(command: list[str]) -> None:
    print(f"running: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("--workers must be between 1 and 4")
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "pipeline_state.json"
    python = sys.executable
    deadline = (
        float(args.objective_start_epoch)
        + float(args.training_deadline_hours) * 3600.0
    )
    write_state(
        state_path,
        stage="dataset_generation",
        objective_start_epoch=float(args.objective_start_epoch),
        training_deadline_epoch=deadline,
        data_dir=str(data_dir),
        run_dir=str(run_dir),
        generator_workers=args.workers,
        worker_affinities=args.worker_affinities,
        process_id=os.getpid(),
    )
    try:
        run(
            [
                python,
                "specialist_rebuild_v2/build_dataset.py",
                "--output-dir",
                str(data_dir),
                "--workers",
                str(args.workers),
                "--worker-affinities",
                args.worker_affinities,
            ]
        )
        dataset_finished = time.time()
        write_state(
            state_path,
            stage="dataset_verification",
            dataset_finished_at_epoch=dataset_finished,
            implementation_plus_dataset_seconds=(
                dataset_finished - float(args.objective_start_epoch)
            ),
        )
        run(
            [
                python,
                "specialist_rebuild_v2/verify_dataset.py",
                str(data_dir),
            ]
        )
        verified = time.time()
        within_deadline = dataset_finished <= deadline
        write_state(
            state_path,
            stage="training" if within_deadline else "complete_without_training",
            dataset_verified_at_epoch=verified,
            implementation_plus_verified_dataset_seconds=(
                verified - float(args.objective_start_epoch)
            ),
            training_condition_met=within_deadline,
            training_condition_basis="dataset_finished_at_epoch",
        )
        if within_deadline:
            training_started = time.time()
            write_state(
                state_path,
                stage="training",
                training_started_at_epoch=training_started,
            )
            run(
                [
                    python,
                    "specialist_rebuild_v2/train_models.py",
                    str(data_dir),
                    str(run_dir),
                ]
            )
            write_state(
                state_path,
                stage="artifact_audit",
                training_finished_at_epoch=time.time(),
            )
            run(
                [
                    python,
                    "specialist_rebuild_v2/audit_artifacts.py",
                    str(data_dir),
                    str(run_dir),
                ]
            )
            write_state(
                state_path,
                stage="complete",
                artifact_audit_finished_at_epoch=time.time(),
            )
        print(state_path.read_text(encoding="utf-8"), flush=True)
    except BaseException as error:
        write_state(
            state_path,
            stage="failed",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise


if __name__ == "__main__":
    main()
