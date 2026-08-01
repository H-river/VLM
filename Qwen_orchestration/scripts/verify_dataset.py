#!/usr/bin/env python3
"""Verify the generated orchestration dataset without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../VLM_data/qwen_orchestration/v1"),
    )
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    root = parse_args().data_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "audit_report.json").read_text(encoding="utf-8"))
    if manifest["dataset_version"] != "qwen_orchestration_sft_v1":
        raise RuntimeError("unexpected dataset version")
    if not audit["passed"] or audit["total_records"] != 15400:
        raise RuntimeError("dataset audit is not a passing 15,400-record audit")
    expected: dict[str, str] = {}
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected_digest, separator, relative = line.partition("  ")
        if not separator or relative in expected:
            raise RuntimeError(f"invalid checksum line: {line}")
        expected[relative] = expected_digest
    for relative, expected_digest in expected.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing dataset file: {relative}")
        actual = digest(path)
        if actual != expected_digest:
            raise RuntimeError(f"dataset checksum mismatch: {relative}")
    print(
        "Dataset verification passed: "
        f"{audit['total_records']} records, "
        f"{audit['unique_physical_groups']} groups, "
        f"{len(expected)} checksummed files."
    )


if __name__ == "__main__":
    main()
