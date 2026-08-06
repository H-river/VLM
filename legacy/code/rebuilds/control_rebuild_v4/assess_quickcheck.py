#!/usr/bin/env python3
"""Apply the registered performance bars to a validation-only server trial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v4.assess_validation import assess_validation

DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_quickcheck_12h"
DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/control_rebuild_v4_quickcheck"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--expected-train-groups", type=int, default=2000)
    parser.add_argument("--expected-val-groups", type=int, default=300)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def quickcheck_decision(
    validation_result: dict,
    *,
    unique_train_groups: int,
    unique_validation_groups: int,
) -> dict:
    result = dict(validation_result)
    authorized = bool(result.pop("proceed_to_heldout"))
    result.update(
        {
            "decision_rule_version": (
                "control_rebuild_v4_quickcheck_same_registered_performance_bars"
            ),
            "source_gate_rule": ("control_rebuild_v4_preregistered_validation_gates"),
            "scope": (
                "validation-only 12-hour laptop trial; no held-out test was "
                "opened; passing authorizes full unique-data server training"
            ),
            "unique_train_groups": int(unique_train_groups),
            "unique_validation_groups": int(unique_validation_groups),
            "training_repeat": 1,
            "proceed_to_server_training": authorized,
            "held_out_evaluation_opened": False,
            "complete": True,
        }
    )
    return result


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    data_dir = args.data_dir.resolve()
    config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    configured = config["group_counts"]
    if int(configured["train"]) != int(args.expected_train_groups):
        raise ValueError("quickcheck training-group count differs")
    if int(configured["val"]) != int(args.expected_val_groups):
        raise ValueError("quickcheck validation-group count differs")
    result = quickcheck_decision(
        assess_validation(
            run_dir,
            expected_comparison_groups=int(args.expected_val_groups),
        ),
        unique_train_groups=int(args.expected_train_groups),
        unique_validation_groups=int(args.expected_val_groups),
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else run_dir / "quickcheck_decision.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
