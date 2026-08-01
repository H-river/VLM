#!/usr/bin/env python3
"""Create the formal Gate-A diagnosis once, and never before its timestamp."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


VERSION = "active_diagnosis_v13_formal_gate_freeze_once_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_timestamp(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("formal gate timestamp must include a UTC offset")
    return parsed


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--not-before", required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    config = args.config.resolve()
    not_before = _parse_timestamp(args.not_before)
    now = dt.datetime.now(dt.timezone.utc)
    if now < not_before.astimezone(dt.timezone.utc):
        remaining = (not_before.astimezone(dt.timezone.utc) - now).total_seconds()
        raise ValueError(f"formal Gate A is locked for another {remaining:.1f} seconds")
    diagnosis = root / "gate_a_diagnosis.json"
    freeze = root / "frozen_decision.json"
    manifest_path = root / "formal_gate_freeze_manifest.json"
    if diagnosis.exists() or freeze.exists():
        raise ValueError("formal Gate A artifacts already exist; refusing to regenerate")
    command = [
        sys.executable,
        "-m",
        "active_diagnosis_v13.run_gate_a",
        "aggregate",
        "--config",
        str(config),
        "--output-dir",
        str(root),
    ]
    started = time.time()
    manifest: dict[str, Any] = {
        "version": VERSION,
        "status": "running",
        "not_before": not_before.isoformat(),
        "started_at": dt.datetime.now().astimezone().isoformat(),
        "started_unix": started,
        "command": command,
        "config": str(config),
        "config_sha256": _sha256(config),
        "protected_set_used": False,
    }
    _atomic_json(manifest_path, manifest)
    log_path = root / "formal_gate_freeze.log"
    with log_path.open("a", buffering=1) as stream:
        result = subprocess.run(
            command,
            cwd=Path.cwd(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    errors = []
    if result.returncode != 0:
        errors.append(f"aggregate exited {result.returncode}")
    if not diagnosis.exists() or not freeze.exists():
        errors.append("formal Gate A artifacts are incomplete")
    manifest.update(
        {
            "status": "complete" if not errors else "failed",
            "returncode": result.returncode,
            "finished_at": dt.datetime.now().astimezone().isoformat(),
            "elapsed_seconds": time.time() - started,
            "log": str(log_path),
            "errors": errors,
        }
    )
    if diagnosis.exists():
        manifest["gate_a_diagnosis_sha256"] = _sha256(diagnosis)
    if freeze.exists():
        manifest["frozen_decision_sha256"] = _sha256(freeze)
    _atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
