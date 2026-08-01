#!/usr/bin/env python3
"""Snapshot the exact research source bytes used at the formal gate."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


VERSION = "active_diagnosis_v13_source_snapshot_v1"
IGNORED_PARTS = {"__pycache__", ".pytest_cache"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inventory(repo_root: Path, sources: list[Path]) -> tuple[list[dict[str, Any]], str]:
    root = repo_root.resolve()
    files: set[Path] = set()
    for source in sources:
        resolved = source.resolve()
        candidates = resolved.rglob("*") if resolved.is_dir() else (resolved,)
        for candidate in candidates:
            if candidate.is_file() and not IGNORED_PARTS.intersection(candidate.parts):
                try:
                    candidate.relative_to(root)
                except ValueError as error:
                    raise ValueError(f"source is outside repository: {candidate}") from error
                files.add(candidate)
    entries = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(files, key=lambda value: str(value.relative_to(root)))
    ]
    aggregate = hashlib.sha256()
    for entry in entries:
        aggregate.update(str(entry["path"]).encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(entry["sha256"]).encode("ascii"))
        aggregate.update(b"\n")
    return entries, aggregate.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--protected-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    entries, aggregate = source_inventory(root, [path.resolve() for path in args.source])
    gate = args.gate_dir.resolve()
    protected = args.protected_dir.resolve()
    report = {
        "version": VERSION,
        "captured_at": dt.datetime.now().astimezone().isoformat(),
        "repository_root": str(root),
        "git_branch": _git(root, "branch", "--show-current"),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "source_file_count": len(entries),
        "source_tree_sha256": aggregate,
        "sources": entries,
        "formal_gate_artifacts_present": any(
            (gate / name).exists()
            for name in (
                "gate_a_diagnosis.json",
                "frozen_decision.json",
                "branch_a_resolution.json",
            )
        ),
        "protected_directory_present": protected.exists(),
        "protected_directory_traversed": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
