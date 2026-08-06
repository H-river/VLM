#!/usr/bin/env python3
"""Validate v10 grouped schemas without loading locked-test labels by default."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_structured_rebuild_v10.contracts import (
    REGIMES,
    STATE_FIELDS,
    context_hash,
    setup_hash,
    sha256_file,
    validate_action_order,
)

DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/physics_structured_rebuild_v10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--include-locked-after-freeze", action="store_true")
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
    return False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_split(path: Path, data_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    setup_hashes: set[str] = set()
    context_hashes: set[str] = set()
    regimes = Counter()
    requested_cardinality = Counter()
    for row in rows:
        if not finite(row):
            raise ValueError(f"{row.get('group_id')}: non-finite value")
        if row["regime"] not in REGIMES:
            raise ValueError(f"{row['group_id']}: unknown regime")
        validate_action_order(row["candidates"])
        if row["setup_hash"] != setup_hash(row["setup"]):
            raise ValueError(f"{row['group_id']}: setup hash mismatch")
        if row["context_hash"] != context_hash(
            row["setup"], row["current_beam_state"]
        ):
            raise ValueError(f"{row['group_id']}: context hash mismatch")
        if row["setup_hash"] in setup_hashes or row["context_hash"] in context_hashes:
            raise ValueError(f"{row['group_id']}: duplicate group hash")
        setup_hashes.add(row["setup_hash"])
        context_hashes.add(row["context_hash"])
        zero = row["candidates"][40]["next_state"]
        if any(
            abs(float(zero[field]) - float(row["current_beam_state"][field]))
            > 2e-8
            for field in STATE_FIELDS
        ):
            raise ValueError(f"{row['group_id']}: zero action does not reconstruct current state")
        for request in row["visual_requests"]:
            for field in ("current_image", "target_image"):
                image_path = data_dir / request[field]
                if not image_path.is_file():
                    raise ValueError(f"{row['group_id']}: missing {image_path}")
        regimes[row["regime"]] += 1
        requested = row["candidates"][int(row["natural_requested_action_index"])]
        requested_cardinality[requested["auxiliary"]["action_cardinality"]] += 1
    return {
        "groups": len(rows),
        "transitions": 81 * len(rows),
        "setup_hashes": setup_hashes,
        "context_hashes": context_hashes,
        "regime_counts": dict(sorted(regimes.items())),
        "natural_request_cardinality_counts": dict(sorted(requested_cardinality.items())),
        "jsonl_sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    selected_splits = ["train", "development"]
    if args.include_locked_after_freeze:
        selected_splits.append("locked_test")
    reports = {
        split: validate_split(data_dir / "grids" / f"{split}.jsonl", data_dir)
        for split in selected_splits
    }
    for split, report in reports.items():
        expected = manifest["split_summary"][split]
        if report["groups"] != int(expected["groups"]):
            raise ValueError(f"{split}: group count differs from manifest")
        if report["jsonl_sha256"] != expected["jsonl_sha256"]:
            raise ValueError(f"{split}: JSONL hash differs from manifest")
    for left_index, left in enumerate(selected_splits):
        for right in selected_splits[left_index + 1 :]:
            if reports[left]["setup_hashes"] & reports[right]["setup_hashes"]:
                raise ValueError(f"{left}/{right}: setup overlap")
            if reports[left]["context_hashes"] & reports[right]["context_hashes"]:
                raise ValueError(f"{left}/{right}: context overlap")
    serializable = {
        split: {
            key: value
            for key, value in report.items()
            if key not in {"setup_hashes", "context_hashes"}
        }
        for split, report in reports.items()
    }
    output = data_dir / (
        "validation_after_freeze.json"
        if args.include_locked_after_freeze
        else "validation_pre_freeze.json"
    )
    result = {
        "version": "physics_structured_rebuild_v10_dataset_validation",
        "splits_opened": selected_splits,
        "locked_test_labels_opened": bool(args.include_locked_after_freeze),
        "reports": serializable,
        "cross_split_setup_hash_overlap": 0,
        "cross_split_context_hash_overlap": 0,
        "complete": True,
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != result:
            raise RuntimeError(f"refusing to overwrite differing validation: {output}")
    else:
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

