#!/usr/bin/env python3
"""Evaluate one selected Stage-2 checkpoint once on all sealed test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("test_iid", "test_ood_language", "test_visual_stress")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the same sealed evaluation after interruption.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection_json.read_text(encoding="utf-8"))
    selected = selection.get("selected_checkpoint")
    if selection.get("stage") != "stage2" or not selected or not selected["passed"]:
        raise RuntimeError("sealed tests require a fully promoted Stage-2 selection")
    adapter = Path(selected["adapter_path"]).resolve()
    adapter_weights = adapter / "adapter_model.safetensors"
    identity = {
        "stage": "stage2",
        "adapter_path": str(adapter),
        "adapter_sha256": sha256(adapter_weights),
        "selection_sha256": sha256(args.selection_json),
        "config_sha256": sha256(args.config),
        "dataset_manifest_sha256": sha256(args.data_dir / "manifest.json"),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.results_dir / "sealed_evaluation_started.json"
    if ledger_path.exists():
        previous = json.loads(ledger_path.read_text(encoding="utf-8"))
        previous_identity = {
            key: previous[key] for key in identity
        }
        if previous_identity != identity:
            raise RuntimeError(
                "sealed test ledger belongs to a different checkpoint or protocol"
            )
        if not args.resume:
            raise RuntimeError(
                "sealed evaluation already started; use --resume only to finish "
                "this identical recorded run"
            )
    else:
        ledger = {
            **identity,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy": (
                "No checkpoint, prompt, threshold, registry, or repair selection "
                "may use these test outputs."
            ),
        }
        ledger_path.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    split_reports: dict[str, Any] = {}
    for split in SPLITS:
        generated = args.results_dir / f"{split}_stage2.jsonl"
        summary = args.results_dir / f"{split}_stage2.summary.json"
        generate_command = [
            sys.executable,
            str(REPO_ROOT / "Qwen_orchestration/scripts/evaluate_qwen.py"),
            "--config",
            str(args.config),
            "--input-jsonl",
            str(args.data_dir / f"exports/qwen/{split}_stage2.jsonl"),
            "--canonical-jsonl",
            str(args.data_dir / f"canonical/{split}.jsonl"),
            "--image-root",
            str(args.data_dir),
            "--output-jsonl",
            str(generated),
            "--summary-json",
            str(summary),
            "--adapter-path",
            str(adapter),
            "--stage",
            "stage2",
            "--subset",
            "all",
            "--batch-size",
            str(args.batch_size),
            "--resume",
        ]
        generation_code = run(generate_command)
        if generation_code != 0:
            raise RuntimeError(f"{split} generation failed with code {generation_code}")
        end_to_end = args.results_dir / f"{split}_end_to_end.json"
        details = args.results_dir / f"{split}_end_to_end.details.jsonl"
        end_to_end_code = run(
            [
                sys.executable,
                str(REPO_ROOT / "Qwen_orchestration/scripts/evaluate_end_to_end.py"),
                "--predictions-jsonl",
                str(generated),
                "--canonical-jsonl",
                str(args.data_dir / f"canonical/{split}.jsonl"),
                "--image-root",
                str(args.data_dir),
                "--private-source-jsonl",
                str(args.data_dir / f"private/source_cases/{split}.jsonl"),
                "--output-json",
                str(end_to_end),
                "--details-jsonl",
                str(details),
            ]
        )
        split_reports[split] = {
            "orchestration": json.loads(summary.read_text(encoding="utf-8"))[
                "metrics"
            ],
            "end_to_end": json.loads(end_to_end.read_text(encoding="utf-8"))[
                "metrics"
            ],
            "end_to_end_gate_exit_code": end_to_end_code,
        }

    report = {
        **identity,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "splits": split_reports,
    }
    (args.results_dir / "sealed_evaluation_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
