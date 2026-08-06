#!/usr/bin/env python3
"""Verify the completed v4 numerical dataset and write stable checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import ACTION_GRID, read_jsonl
from control_rebuild_v4.build_numerical_dataset import category_for
from specialist_rebuild_v2.common import ACTION_FIELDS, STATE_FIELDS

DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/control_rebuild_v4_numerical"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--expected-groups-per-split",
        type=int,
        help="Test-only override for a deliberately truncated smoke dataset.",
    )
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_group(row: dict[str, Any]) -> None:
    setup = row.get("setup", {})
    if not setup or any(not math.isfinite(float(value)) for value in setup.values()):
        raise ValueError(f"{row.get('group_id')}: setup contains non-finite values")
    current = row.get("current_beam_state", {})
    if set(current) != set(STATE_FIELDS) or any(
        not math.isfinite(float(current[field])) for field in STATE_FIELDS
    ):
        raise ValueError(
            f"{row.get('group_id')}: current state violates the five-state contract"
        )
    candidates = row.get("candidates", [])
    if len(candidates) != 81:
        raise ValueError(f"{row.get('group_id')}: expected 81 candidates")
    for index, (candidate, expected_action) in enumerate(
        zip(candidates, ACTION_GRID, strict=True)
    ):
        action = candidate["action"]
        if any(
            float(action[field]) != float(expected_action[field])
            for field in ACTION_FIELDS
        ):
            raise ValueError(f"{row['group_id']}: action order mismatch at {index}")
        for section in ("next_state", "change"):
            if set(candidate[section]) != set(STATE_FIELDS):
                raise ValueError(
                    f"{row['group_id']}: {section} fields differ at {index}"
                )
            if any(
                not math.isfinite(float(candidate[section][field]))
                for field in STATE_FIELDS
            ):
                raise ValueError(f"{row['group_id']}: non-finite {section} at {index}")
        for field in STATE_FIELDS:
            reconstructed = float(current[field]) + float(candidate["change"][field])
            if not math.isclose(
                reconstructed,
                float(candidate["next_state"][field]),
                rel_tol=0.0,
                abs_tol=2e-8,
            ):
                raise ValueError(
                    f"{row['group_id']}: change does not reconstruct "
                    f"next_state at {index}.{field}"
                )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_ids: set[str] = set()
    verification = {}
    seed = int(config["seed"])
    fractions = config["category_fractions"]
    dataset_version = str(config["version"])
    for split, expected_count in config["group_counts"].items():
        if args.expected_groups_per_split is not None:
            expected_count = args.expected_groups_per_split
        rows = read_jsonl(data_dir / "grids" / f"{split}.jsonl")
        if len(rows) != int(expected_count):
            raise ValueError(
                f"{split}: expected {expected_count} groups, got {len(rows)}"
            )
        ids = [str(row["group_id"]) for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{split}: duplicate group ids")
        expected_ids = {
            f"v4_{split}_{index:06d}" for index in range(int(expected_count))
        }
        actual_ids = set(ids)
        if actual_ids != expected_ids:
            raise ValueError(
                f"{split}: group id set differs: "
                f"missing={sorted(expected_ids - actual_ids)[:5]}, "
                f"extra={sorted(actual_ids - expected_ids)[:5]}"
            )
        overlap = all_ids.intersection(ids)
        if overlap:
            raise ValueError(f"{split}: cross-split id overlap: {sorted(overlap)[:5]}")
        all_ids.update(ids)
        for row in rows:
            if str(row["split"]) != split:
                raise ValueError(f"{row['group_id']}: row split does not match file")
            if str(row.get("dataset_version")) != dataset_version:
                raise ValueError(
                    f"{row['group_id']}: dataset version does not match config"
                )
            expected_category = category_for(
                seed, split, str(row["group_id"]), fractions
            )
            if str(row.get("source_category")) != expected_category:
                raise ValueError(
                    f"{row['group_id']}: expected category {expected_category}"
                )
            verify_group(row)
        verification[split] = {
            "groups": len(rows),
            "transitions": len(rows) * 81,
            "category_counts": dict(
                sorted(Counter(str(row["source_category"]) for row in rows).items())
            ),
        }

    checksum_paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.name not in {"checksums.sha256", "manifest.json"}
    )
    lines = [f"{digest(path)}  {path.relative_to(data_dir)}" for path in checksum_paths]
    checksum_text = "\n".join(lines) + "\n"
    (data_dir / "checksums.sha256").write_text(checksum_text, encoding="utf-8")
    manifest.update(
        {
            "verification": {
                "complete": True,
                "splits": verification,
                "cross_split_group_overlap": 0,
                "candidate_count_per_group": 81,
                "finite_numerical_values": True,
                "action_grid_order_exact": True,
                "held_out_v2_test_files_opened": False,
            },
            "checksum_contract": {
                "excludes": ["manifest.json", "checksums.sha256"],
                "file_count": len(checksum_paths),
                "checksum_manifest_sha256": hashlib.sha256(
                    checksum_text.encode()
                ).hexdigest(),
            },
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verified": True,
                "splits": verification,
                "checksum_file_count": len(checksum_paths),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
