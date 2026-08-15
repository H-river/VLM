#!/usr/bin/env python3
"""Verify that the compact result table matches its tracked source reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "results/published_tables/confirmed_results.csv"


def main() -> int:
    with TABLE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    sources: dict[Path, str] = {}
    failures = []
    for row in rows:
        source = ROOT / row["source_file"]
        text = sources.setdefault(source, source.read_text(encoding="utf-8"))
        if row["source_literal"] not in text:
            failures.append(row["result_id"])
        if row["evidence_level"] != "CONFIRMED":
            failures.append(f"{row['result_id']}:non_confirmed_label")
    result = {
        "component": "confirmed_results_table",
        "status": "PASS" if not failures else "FAIL",
        "rows": len(rows),
        "source_files": sorted(str(path.relative_to(ROOT)) for path in sources),
        "failures": failures,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

