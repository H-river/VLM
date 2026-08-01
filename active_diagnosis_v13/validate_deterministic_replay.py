#!/usr/bin/env python3
"""Verify that a small replay is byte-equivalent after removing timing/IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


VERSION = "active_diagnosis_v13_deterministic_replay_validation_v1"
NON_BEHAVIORAL_KEYS = {
    "record_id",
    "policy_name",
    "wall_runtime_seconds",
    # Added to the schema after the reference process loaded. Its value is
    # separately fixed by config and does not alter the episode projection.
    "planner_root_seed",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _stable(item)
            for key, item in sorted(value.items())
            if key not in NON_BEHAVIORAL_KEYS
        }
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = {_key(row): row for row in _jsonl(args.reference.resolve())}
    replay = {_key(row): row for row in _jsonl(args.replay.resolve())}
    if len(replay) != 15:
        raise ValueError(f"expected 15 replay records, found {len(replay)}")
    missing = sorted(set(replay) - set(reference))
    mismatches = []
    matched = []
    for key, row in replay.items():
        if key not in reference:
            continue
        left, right = _stable(reference[key]), _stable(row)
        if left != right:
            mismatches.append(
                {
                    "case_id": key[0],
                    "gain": key[1],
                    "reference_sha256": _digest(left),
                    "replay_sha256": _digest(right),
                }
            )
        else:
            matched.append({"case_id": key[0], "gain": key[1], "sha256": _digest(left)})
    report = {
        "version": VERSION,
        "reference": str(args.reference.resolve()),
        "replay": str(args.replay.resolve()),
        "replay_records": len(replay),
        "exact_stable_matches": len(matched),
        "missing_reference_keys": missing,
        "mismatches": mismatches,
        "non_behavioral_keys_excluded": sorted(NON_BEHAVIORAL_KEYS),
        "passes": not missing and not mismatches and len(matched) == 15,
        "matched_digests": matched,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
