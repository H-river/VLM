#!/usr/bin/env python3
"""Run the single post-freeze protected direct/oracle/probe evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


VERSION = "active_diagnosis_v13_protected_once_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate_protected_rows(rows: list[dict[str, Any]], mode: str) -> list[str]:
    """Validate the exact registered 18-case by 5-gain protected support."""

    errors: list[str] = []
    if len(rows) != 90:
        return [f"{mode} has {len(rows)} rows"]
    record_ids = [str(row["record_id"]) for row in rows]
    if len(set(record_ids)) != len(record_ids):
        errors.append(f"{mode} has duplicate record IDs")
    episode_keys = [
        (str(row["case_id"]), float(row["evaluator_only_true_gain"])) for row in rows
    ]
    if len(set(episode_keys)) != 90:
        errors.append(f"{mode} does not contain 90 unique case-gain episodes")
    cases = {
        (str(row["stratum"]), int(str(row["case_id"]).rsplit("_", 1)[1]))
        for row in rows
    }
    strata = {stratum for stratum, _ in cases}
    expected_suffixes = set(range(10, 16))
    if len(strata) != 3 or len(cases) != 18:
        errors.append(f"{mode} does not contain 18 cases across three strata")
    for stratum in sorted(strata):
        suffixes = {suffix for name, suffix in cases if name == stratum}
        if suffixes != expected_suffixes:
            errors.append(
                f"{mode} stratum {stratum} suffixes are {sorted(suffixes)}, "
                f"expected {sorted(expected_suffixes)}"
            )
    return errors


def _run(command: list[str], log_path: Path, output: Path) -> dict[str, Any]:
    started = time.time()
    with log_path.open("a", buffering=1) as stream:
        stream.write(json.dumps({"event": "launch", "command": command}) + "\n")
        process = subprocess.run(
            command,
            cwd=Path.cwd(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        stream.write(
            json.dumps(
                {
                    "event": "exit",
                    "returncode": process.returncode,
                    "elapsed_seconds": time.time() - started,
                }
            )
            + "\n"
        )
    return {
        "output": str(output),
        "log": str(log_path),
        "returncode": process.returncode,
        "records": len(_rows(output)),
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--protected-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    development = args.development_dir.resolve()
    protected = args.protected_dir.resolve()
    config = args.config.resolve()
    freeze_path = args.freeze.resolve()
    if not freeze_path.exists():
        raise ValueError("protected evaluation is forbidden before the freeze artifact exists")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze.get("frozen") or freeze.get("protected_set_used_for_selection"):
        raise ValueError("invalid development-only freeze artifact")
    selected = freeze["selected_probe"]
    model_path = Path(selected["classifier_bundle"]).resolve()
    if _sha256(model_path) != str(selected["classifier_bundle_sha256"]):
        raise ValueError("frozen protected probe classifier hash mismatch")
    if freeze.get("config_sha256") != _sha256(config):
        raise ValueError("protected config does not match the frozen decision")
    freeze_sha256 = _sha256(freeze_path)
    manifest_path = protected / "protected_once_manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            raise ValueError("protected evaluation already completed; refusing a second test")
        if prior.get("selection_source_sha256") != freeze_sha256:
            raise ValueError("partial protected run uses a different frozen decision")
    protected.mkdir(parents=True, exist_ok=True)
    log_dir = protected / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for mode in ("direct", "oracle_known", "probe_replan"):
        command = [
            sys.executable,
            "-m",
            "active_diagnosis_v13.run_gate_a",
            "control",
            "--config",
            str(config),
            "--output-dir",
            str(protected),
            "--split",
            "protected",
            "--freeze",
            str(freeze_path),
            "--mode",
            mode,
        ]
        if mode == "probe_replan":
            command.extend(
                [
                    "--probe-model",
                    str(model_path),
                    "--probe-design",
                    str(selected["selected_design"]),
                    "--probe-fraction",
                    f"{float(selected['selected_fraction']):g}",
                ]
            )
        jobs.append(
            {
                "mode": mode,
                "command": command,
                "log": log_dir / f"{mode}.log",
                "output": protected / "control" / f"{mode}.jsonl",
            }
        )
    manifest: dict[str, Any] = {
        "version": VERSION,
        "status": "running",
        "selection_source": str(freeze_path),
        "selection_source_sha256": freeze_sha256,
        "config_sha256": _sha256(config),
        "selected_primary_branch": freeze["selected_primary_branch"],
        "selected_probe": selected,
        "protected_set_used_for_selection": False,
        "jobs": [],
    }
    _atomic_json(manifest_path, manifest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(_run, job["command"], job["log"], job["output"]): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            result = {"mode": job["mode"], **future.result()}
            manifest["jobs"].append(result)
            _atomic_json(manifest_path, manifest)
            print(json.dumps({"event": "protected_mode_complete", **result}), flush=True)
    errors = []
    for job in jobs:
        rows = _rows(job["output"])
        errors.extend(validate_protected_rows(rows, str(job["mode"])))
    manifest["errors"] = errors
    manifest["status"] = "complete" if not errors else "failed"
    manifest["finished_unix"] = time.time()
    _atomic_json(manifest_path, manifest)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
