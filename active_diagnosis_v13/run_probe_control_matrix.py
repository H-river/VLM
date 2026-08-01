#!/usr/bin/env python3
"""Run every non-selected probe-design control cell with bounded concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from active_diagnosis_v13.summarize_probe_controls import output_name


VERSION = "active_diagnosis_v13_probe_control_matrix_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _run_cell(
    *,
    root: Path,
    config: Path,
    design: str,
    fraction: float,
    model_path: Path,
    log_dir: Path,
) -> dict[str, Any]:
    name = output_name(design, fraction)
    output = root / "control" / f"{name}.jsonl"
    command = [
        sys.executable,
        "-m",
        "active_diagnosis_v13.run_gate_a",
        "control",
        "--config",
        str(config),
        "--output-dir",
        str(root),
        "--mode",
        "probe_replan",
        "--probe-model",
        str(model_path),
        "--probe-design",
        design,
        "--probe-fraction",
        f"{fraction:g}",
        "--output-name",
        name,
    ]
    started = time.time()
    log_path = log_dir / f"{name}.log"
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
    records = sum(1 for line in output.read_text().splitlines() if line) if output.exists() else 0
    return {
        "design": design,
        "fraction": fraction,
        "output_name": name,
        "model_path": str(model_path),
        "log_path": str(log_path),
        "output_path": str(output),
        "records": records,
        "returncode": process.returncode,
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--include-selected", action="store_true")
    args = parser.parse_args()
    if args.max_workers < 1:
        raise ValueError("max-workers must be positive")
    root = args.gate_dir.resolve()
    config = args.config.resolve()
    summary = json.loads(
        (root / "probes" / "probe_summary.json").read_text(encoding="utf-8")
    )
    selected_key = (
        str(summary["selected"]["selected_design"]),
        float(summary["selected"]["selected_fraction"]),
    )
    cells = []
    for cell in summary["designs"]:
        key = (str(cell["design"]), float(cell["fraction"]))
        if key == selected_key and not args.include_selected:
            continue
        cells.append(
            {
                "design": key[0],
                "fraction": key[1],
                "model_path": Path(str(cell["classifier_bundle"])).resolve(),
            }
        )
    matrix_dir = root / "probe_control_matrix"
    log_dir = matrix_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "selected_probe_skipped": not args.include_selected,
        "selected_probe": summary["selected"],
        "max_workers": args.max_workers,
        "started_unix": time.time(),
        "status": "running",
        "queued_cells": [
            {"design": cell["design"], "fraction": cell["fraction"]}
            for cell in cells
        ],
        "completed": [],
    }
    manifest_path = matrix_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                _run_cell,
                root=root,
                config=config,
                design=cell["design"],
                fraction=cell["fraction"],
                model_path=cell["model_path"],
                log_dir=log_dir,
            ): cell
            for cell in cells
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            manifest["completed"].append(result)
            if int(result["returncode"]) != 0 or int(result["records"]) != 150:
                failures.append(result)
            _atomic_json(manifest_path, manifest)
            print(json.dumps({"event": "cell_complete", **result}), flush=True)
    manifest["finished_unix"] = time.time()
    manifest["status"] = "failed" if failures else "complete"
    manifest["failures"] = failures
    _atomic_json(manifest_path, manifest)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
