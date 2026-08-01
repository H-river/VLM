#!/usr/bin/env python3
"""Apply the preregistered candidate-coverage proposal-seed replication rule."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

VERSION = "active_diagnosis_v13_boundary_candidate_replication_decision_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rule = json.loads(args.rule.resolve().read_text())
    audit = json.loads(args.audit.resolve().read_text())
    if rule.get("protected_set_used") is not False or audit.get(
        "protected_set_used"
    ) is not False:
        raise ValueError("replication decision requires development-only inputs")
    if int(audit["episodes"]) != 20:
        raise ValueError("replication decision requires the complete 20-episode audit")
    if int(audit["budgets"][-1]) != int(rule["decision_budget"]):
        raise ValueError("candidate audit does not reach the registered decision budget")
    pairing = audit["maximum_budget_pairing"]
    union_ids = sorted(
        set(pairing["both_success"])
        | set(pairing["uniform_only_success"])
        | set(pairing["conditioned_only_success"])
    )
    threshold = int(rule["launch_seed2_if_at_least"])
    report = {
        "version": VERSION,
        "protected_set_used": False,
        "metric": rule["metric"],
        "decision_budget": int(rule["decision_budget"]),
        "observed_proposal_union_successes": len(union_ids),
        "observed_proposal_union_success_ids": union_ids,
        "launch_threshold": threshold,
        "launch_seed2": len(union_ids) >= threshold,
        "seed2_root_seed": int(rule["seed2_root_seed"]),
        "seed2_maximum_budget": int(rule["seed2_maximum_budget"]),
        "interpretation_guard": rule["interpretation_guard"],
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
