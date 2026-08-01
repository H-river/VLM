#!/usr/bin/env python3
"""Verify the Qwen orchestration frozen baseline without modifying artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ORCHESTRATION_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ORCHESTRATION_ROOT.parent
DEFAULT_MANIFEST = ORCHESTRATION_ROOT / "freeze" / "baseline_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        item_hash = sha256_file(item)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        total_size += size
    return {
        "file_count": file_count,
        "total_size": total_size,
        "tree_sha256": digest.hexdigest(),
    }


def resolve(path_value: str) -> Path:
    return (WORKSPACE_ROOT / path_value).resolve()


def verify(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for entry in manifest["files"]:
        path = resolve(entry["path"])
        if not path.is_file():
            failures.append(f"missing file: {entry['path']}")
            continue
        actual_size = path.stat().st_size
        if actual_size != entry["size"]:
            failures.append(
                f"file size mismatch: {entry['path']} expected={entry['size']} actual={actual_size}"
            )
            continue
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            failures.append(
                f"file digest mismatch: {entry['path']} expected={entry['sha256']} actual={actual_hash}"
            )
    for entry in manifest["directories"]:
        path = resolve(entry["path"])
        if not path.is_dir():
            failures.append(f"missing directory: {entry['path']}")
            continue
        actual = directory_fingerprint(path)
        for key in ("file_count", "total_size", "tree_sha256"):
            if actual[key] != entry[key]:
                failures.append(
                    f"directory {key} mismatch: {entry['path']} "
                    f"expected={entry[key]} actual={actual[key]}"
                )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    failures = verify(args.manifest)
    if failures:
        print(f"Frozen baseline verification FAILED ({len(failures)} issue(s))")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(
        "Frozen baseline verification passed: "
        f"{len(manifest['files'])} files and {len(manifest['directories'])} directories "
        f"for {manifest['freeze_id']}"
    )


if __name__ == "__main__":
    main()
