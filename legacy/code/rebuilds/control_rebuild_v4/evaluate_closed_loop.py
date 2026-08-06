#!/usr/bin/env python3
"""Run a resumable, bounded closed-loop evaluation on one declared split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import read_jsonl
from control_rebuild_v3.train_forward import configure
from control_rebuild_v4.closed_loop import (
    NumericalClosedLoopControllerV4,
    OpticalSimulatorExecutor,
)
from control_rebuild_v4.forward_runtime import load_forward_runtime_v4
from control_rebuild_v4.inverse_runtime import load_inverse_runtime_v4
from specialist_rebuild_v2.common import raw_state_array, stable_token

DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--split",
        choices=("val", "test_iid", "test_ood_physics"),
        default="val",
    )
    parser.add_argument("--max-requests", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def target_index(group_id: str) -> int:
    index = int(stable_token(group_id, "closed_loop_target")[:8], 16) % 81
    return (index + 1) % 81 if index == 40 else index


def aggregate(records: list[dict[str, Any]], max_steps: int) -> dict[str, Any]:
    count = len(records)
    reached_at = []
    for record in records:
        step = None
        for row in record["trace"]:
            if row["true_target_reached"]:
                step = int(row["step"])
                break
        reached_at.append(step)
    return {
        "request_count": count,
        "max_steps": max_steps,
        "initial_target_reached": int(
            sum(record["initial_target_reached"] for record in records)
        ),
        "reached_by_step": {
            str(step): {
                "count": int(
                    sum(
                        reached is not None and reached <= step
                        for reached in reached_at
                    )
                ),
                "rate": (
                    0.0
                    if count == 0
                    else float(
                        np.mean(
                            [
                                reached is not None and reached <= step
                                for reached in reached_at
                            ]
                        )
                    )
                ),
            }
            for step in range(1, max_steps + 1)
        },
        "final_physical_success": (
            0.0
            if count == 0
            else float(np.mean([record["final_target_reached"] for record in records]))
        ),
        "mean_executed_steps": (
            0.0
            if count == 0
            else float(np.mean([record["executed_steps"] for record in records]))
        ),
        "zero_action_stalls": int(
            sum(record["stop_reason"] == "zero_action_stall" for record in records)
        ),
        "predicted_infeasible_requests": int(
            sum(
                any(
                    row["predicted_status"] == "infeasible_within_limits"
                    for row in record["trace"]
                )
                for record in records
            )
        ),
    }


def main() -> None:
    args = parse_args()
    if args.max_requests < 1:
        raise ValueError("--max-requests must be positive")
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else run_dir / f"closed_loop_{args.split}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if bool(existing.get("complete")):
            raise RuntimeError(f"refusing to overwrite completed evaluation: {output}")
        records = list(existing.get("records", []))
    completed_ids = {str(row["request_id"]) for row in records}

    torch, device = configure(20260726, args.device)
    forward_path = run_dir / "forward_physics_residual_v4.pt"
    inverse_path = run_dir / "inverse_control_v4.pt"
    forward, _ = load_forward_runtime_v4(forward_path, torch, device)
    inverse, _ = load_inverse_runtime_v4(inverse_path, torch, device)
    controller = NumericalClosedLoopControllerV4(forward, inverse)
    executor = OpticalSimulatorExecutor(REPO_ROOT)
    rows = read_jsonl(data_dir / "grids" / f"{args.split}.jsonl")
    rows = rows[: args.max_requests]
    started = time.perf_counter()
    for request_number, row in enumerate(rows):
        request_id = f"{args.split}:{row['group_id']}"
        if request_id in completed_ids:
            continue
        index = target_index(str(row["group_id"]))
        record = controller.run(
            row["setup"],
            raw_state_array(row["current_beam_state"]),
            raw_state_array(row["candidates"][index]["next_state"]),
            executor,
            request_id=request_id,
            max_steps=args.max_steps,
        )
        record["source_target_index"] = index
        records.append(record)
        partial = {
            "evaluation_version": "closed_loop_control_v4_once",
            "split": args.split,
            "records": records,
            "metrics": aggregate(records, args.max_steps),
            "complete": False,
        }
        output.write_text(
            json.dumps(partial, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "completed": len(records),
                    "total": len(rows),
                    "latest_success": record["final_target_reached"],
                }
            ),
            flush=True,
        )
    result = {
        "evaluation_version": "closed_loop_control_v4_once",
        "split": args.split,
        "artifacts": {
            "forward": str(forward_path),
            "inverse": str(inverse_path),
        },
        "records": records,
        "metrics": aggregate(records, args.max_steps),
        "held_out_test_used_for_training_or_selection": 0,
        "held_out_requests_evaluated": (0 if args.split == "val" else len(records)),
        "complete": True,
        "seconds": time.perf_counter() - started,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
