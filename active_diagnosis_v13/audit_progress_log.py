#!/usr/bin/env python3
"""Audit timestamp cadence and required fields in the 12-hour progress log."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

VERSION = "active_diagnosis_v13_progress_log_audit_v1"
HEADING = re.compile(r"^## (?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\s+—\s+(?P<title>.+)$")
REQUIRED_FIELDS = (
    "Current branch:",
    "Completed artifacts:",
    "Running jobs:",
    "Measured results:",
    "Next queued task 1:",
    "Next queued task 2:",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sections(text: str) -> list[dict[str, Any]]:
    output = []
    current = None
    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            if current is not None:
                output.append(current)
            current = {
                "timestamp": datetime.fromisoformat(match.group("timestamp")),
                "timestamp_text": match.group("timestamp"),
                "title": match.group("title"),
                "body": [],
            }
        elif current is not None:
            current["body"].append(line)
    if current is not None:
        output.append(current)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-log", type=Path, required=True)
    parser.add_argument("--expected-start", required=True)
    parser.add_argument("--maximum-gap-minutes", type=float, default=30.0)
    parser.add_argument("--require-end-at-or-after")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sections = _sections(args.progress_log.resolve().read_text())
    if not sections:
        raise ValueError("progress log has no timestamped sections")
    expected_start = datetime.fromisoformat(args.expected_start)
    if expected_start.utcoffset() is None:
        raise ValueError("expected start must contain an explicit UTC offset")
    timestamps = [section["timestamp"] for section in sections]
    gaps = [
        (timestamps[index] - timestamps[index - 1]).total_seconds()
        for index in range(1, len(timestamps))
    ]
    ordering_errors = [
        index for index, gap in enumerate(gaps, start=1) if gap <= 0.0
    ]
    field_errors = []
    entries = []
    for section in sections:
        body = "\n".join(section["body"])
        missing = [field for field in REQUIRED_FIELDS if field not in body]
        if missing:
            field_errors.append(
                {"timestamp": section["timestamp_text"], "missing_fields": missing}
            )
        entries.append(
            {
                "timestamp": section["timestamp_text"],
                "title": section["title"],
                "required_fields_complete": not missing,
            }
        )
    maximum_gap = max(gaps, default=0.0)
    cadence_limit = float(args.maximum_gap_minutes) * 60.0
    required_end = (
        None
        if args.require_end_at_or_after is None
        else datetime.fromisoformat(args.require_end_at_or_after)
    )
    if required_end is not None and required_end.utcoffset() is None:
        raise ValueError("required end must contain an explicit UTC offset")
    checks = {
        "first_timestamp_matches_expected_start": timestamps[0] == expected_start,
        "timestamps_strictly_increasing": not ordering_errors,
        "maximum_gap_at_most_limit": maximum_gap <= cadence_limit,
        "all_required_fields_present": not field_errors,
        "required_end_reached": required_end is None or timestamps[-1] >= required_end,
    }
    report = {
        "version": VERSION,
        "protected_set_used": False,
        "progress_log": str(args.progress_log.resolve()),
        "entries": entries,
        "entry_count": len(entries),
        "expected_start": expected_start.isoformat(),
        "latest_timestamp": timestamps[-1].isoformat(),
        "elapsed_seconds": (timestamps[-1] - timestamps[0]).total_seconds(),
        "maximum_gap_seconds": maximum_gap,
        "maximum_gap_minutes_allowed": float(args.maximum_gap_minutes),
        "required_end_at_or_after": (
            None if required_end is None else required_end.isoformat()
        ),
        "ordering_error_indices": ordering_errors,
        "required_field_errors": field_errors,
        "checks": checks,
        "passes": all(checks.values()),
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
