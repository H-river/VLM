#!/usr/bin/env python3
"""Parse and integrity-check every machine-readable development artifact."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any


VERSION = "active_diagnosis_v13_development_integrity_v1"
CASE_PATTERN = re.compile(r"v12_mpcdiag_primary_\d+_(\d{4})")


def _walk(value: Any, location: str, errors: list[str]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"nonfinite numeric value at {location}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _walk(child, f"{location}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk(child, f"{location}[{index}]", errors)


def _protected_case_tokens(value: Any) -> set[str]:
    matches: set[str] = set()
    if isinstance(value, str):
        for suffix in CASE_PATTERN.findall(value):
            if int(suffix) >= 10:
                matches.add(suffix)
    elif isinstance(value, dict):
        for child in value.values():
            matches.update(_protected_case_tokens(child))
    elif isinstance(value, list):
        for child in value:
            matches.update(_protected_case_tokens(child))
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.development_dir.resolve()
    output = args.output.resolve()
    if not root.is_dir() or output.parent != root:
        raise ValueError("output must be directly inside the development directory")
    json_paths = sorted(path for path in root.rglob("*.json") if path != output)
    jsonl_paths = sorted(root.rglob("*.jsonl"))
    temporary_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and ".tmp." in path.name
    )
    errors: list[str] = []
    protected_tokens: set[str] = set()
    record_id_files = 0
    jsonl_rows = 0
    total_bytes = 0
    for path in [*json_paths, *jsonl_paths]:
        total_bytes += path.stat().st_size
        relative = str(path.relative_to(root))
        values: list[Any]
        try:
            if path.suffix == ".jsonl":
                values = [
                    json.loads(line)
                    for line in path.read_text().splitlines()
                    if line.strip()
                ]
                jsonl_rows += len(values)
            else:
                values = [json.loads(path.read_text())]
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"parse failure {relative}: {exc}")
            continue
        for index, value in enumerate(values):
            _walk(value, f"{relative}[{index}]", errors)
            protected_tokens.update(_protected_case_tokens(value))
        record_ids = [
            str(value["record_id"])
            for value in values
            if isinstance(value, dict) and "record_id" in value
        ]
        if record_ids:
            record_id_files += 1
            if len(record_ids) != len(set(record_ids)):
                errors.append(f"duplicate record_id within {relative}")
    if protected_tokens:
        errors.append(f"protected case suffixes present: {sorted(protected_tokens)}")
    if temporary_paths:
        errors.append(
            "temporary write files present: "
            + ", ".join(str(path.relative_to(root)) for path in temporary_paths)
        )
    report = {
        "version": VERSION,
        "role": "development_artifact_integrity_no_protected_access",
        "split": "development_only",
        "protected_set_used": False,
        "development_dir": str(root),
        "checks": {
            "all_json_and_jsonl_parse": not any(
                error.startswith("parse failure") for error in errors
            ),
            "all_numeric_values_finite": not any(
                error.startswith("nonfinite") for error in errors
            ),
            "record_ids_unique_within_each_file": not any(
                error.startswith("duplicate record_id") for error in errors
            ),
            "protected_case_suffixes_absent": not protected_tokens,
            "temporary_write_files_absent": not temporary_paths,
        },
        "inventory": {
            "json_files_excluding_this_output": len(json_paths),
            "jsonl_files": len(jsonl_paths),
            "jsonl_rows": jsonl_rows,
            "record_id_files": record_id_files,
            "total_machine_readable_bytes_excluding_this_output": total_bytes,
        },
        "errors": errors,
        "passes": not errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
