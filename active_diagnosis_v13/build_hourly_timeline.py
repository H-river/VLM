#!/usr/bin/env python3
"""Build a machine-readable twelve-hour evidence timeline from the progress log."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from active_diagnosis_v13.audit_progress_log import _sections

VERSION = "active_diagnosis_v13_hourly_timeline_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _field(body: list[str], label: str) -> str | None:
    prefix = f"- {label}:"
    for line in body:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-log", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--hours", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = datetime.fromisoformat(args.start)
    if start.utcoffset() is None or args.hours <= 0:
        raise ValueError("start needs an explicit UTC offset and hours must be positive")
    sections = _sections(args.progress_log.resolve().read_text())
    timeline = []
    for hour in range(args.hours):
        lower = start + timedelta(hours=hour)
        upper = lower + timedelta(hours=1)
        entries = [
            section
            for section in sections
            if lower <= section["timestamp"] < upper
        ]
        timeline.append(
            {
                "hour": hour + 1,
                "label": f"T+{hour}–T+{hour + 1}",
                "start": lower.isoformat(),
                "end": upper.isoformat(),
                "entry_count": len(entries),
                "evidence": [
                    {
                        "timestamp": section["timestamp_text"],
                        "title": section["title"],
                        "completed_artifacts": _field(
                            section["body"], "Completed artifacts"
                        ),
                        "measured_results": _field(
                            section["body"], "Measured results"
                        ),
                    }
                    for section in entries
                ],
            }
        )
    report = {
        "version": VERSION,
        "protected_set_used": False,
        "start": start.isoformat(),
        "hours": args.hours,
        "timeline": timeline,
        "hours_with_evidence": sum(row["entry_count"] > 0 for row in timeline),
        "all_hours_have_evidence": all(row["entry_count"] > 0 for row in timeline),
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_hours_have_evidence"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
