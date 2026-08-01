#!/usr/bin/env python3
"""Execute every frozen correct route and record integration acceptance evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Qwen_orchestration.scripts.evaluate_v12_e2e import (
    image_bindings,
    specialist_success,
    validate_freeze,
)
from Qwen_orchestration.v12.adapter import V12Adapter
from Qwen_orchestration.v12.dispatcher import V12OrchestrationRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = run_dir / "integration_smoke_summary.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite smoke result: {output}")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    cases = validate_freeze(run_dir, config)
    ready = [case for case in cases if case["ground_truth"]["decision"]["status"] == "ready"]
    adapter = V12Adapter()
    runtime = V12OrchestrationRuntime(adapter)
    started = time.perf_counter()
    rows = []
    for case in ready:
        error = None
        outcome = None
        try:
            outcome = runtime.dispatch(
                case["ground_truth"]["decision"],
                image_bindings(case),
                run_id=f"{run_dir.name}:{case['case_id']}:integration-smoke",
                execution_context=case["execution_context"],
            )
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        episode = (
            outcome["result"].get("episode")
            if outcome is not None and isinstance(outcome.get("result"), dict)
            else None
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "route": case["route"],
                "modality": case["modality"],
                "executed": bool(outcome and outcome.get("executed")),
                "backend_v12": bool(outcome and outcome.get("executed_backend") == "v12"),
                "finite": bool(outcome is not None and finite(outcome)),
                "specialist_success": specialist_success(case, outcome),
                "inverse_steps": None if episode is None else int(episode["steps"]),
                "inverse_illegal_actions": None if episode is None else int(episode["illegal_actions"]),
                "planner_backend": (
                    None
                    if outcome is None
                    else outcome["result"].get("planner_backend")
                ),
                "error": error,
            }
        )
    counts = Counter(row["route"] for row in rows)
    inverse_rows = [row for row in rows if "inverse" in row["route"]]
    acceptance = {
        "all_registered_routes_exercised": len(counts) == 7 and all(value > 0 for value in counts.values()),
        "all_calls_executed_by_v12": all(row["executed"] and row["backend_v12"] for row in rows),
        "no_crash_or_nonfinite": all(row["error"] is None and row["finite"] for row in rows),
        "h1_only": bool(inverse_rows) and all(row["planner_backend"] == "learned_h1_cem" for row in inverse_rows),
        "inverse_boundaries_legal": bool(inverse_rows) and all(row["inverse_illegal_actions"] == 0 for row in inverse_rows),
        "nontrivial_inverse_exercised": any((row["inverse_steps"] or 0) > 0 for row in inverse_rows),
    }
    summary = {
        "version": "qwen_v12_integration_smoke_v1",
        "run_id": run_dir.name,
        "case_count": len(rows),
        "route_counts": dict(sorted(counts.items())),
        "specialist_strict_success_count": sum(row["specialist_success"] for row in rows),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256": adapter.checkpoint_hash,
        "manifest_sha256": config["manifest_sha256"],
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
        "cases": rows,
    }
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if not summary["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
