#!/usr/bin/env python3
"""Build resumable forward labels for existing Qwen orchestration train cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v4.evaluate_orchestrated_system import (
    simulator_forward_truth,
)

DEFAULT_QWEN = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_OUTPUT = (
    REPO_ROOT.parent / "VLM_data/physics_structured_rebuild_v9/qwen_adaptation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-every", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    qwen_data = args.qwen_data.resolve()
    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    source_path = qwen_data / "private/source_cases/train.jsonl"
    rows = read_jsonl(source_path)
    if len(rows) != 1000:
        raise ValueError("expected 1,000 Qwen orchestration train source cases")

    generated = 0
    resumed = 0
    for index, row in enumerate(rows):
        group_id = str(row["group_id"])
        shard = shard_dir / f"{index:04d}_{group_id}.json"
        if shard.exists():
            cached = json.loads(shard.read_text(encoding="utf-8"))
            if cached.get("group_id") != group_id:
                raise ValueError(f"cached group differs: {shard}")
            resumed += 1
        else:
            truth = simulator_forward_truth(row, row["action"])
            record = {
                "version": "qwen_forward_adaptation_v9",
                "split": "train",
                "group_id": group_id,
                "setup": row["setup"],
                "current_beam_state": row["current_beam_state"],
                "action": row["action"],
                "images": row["images"],
                "image_calibration": row["image_calibration"],
                "truth_change": truth["change"],
                "truth_directions": truth["directions"],
                "source": row["source"],
            }
            write_atomic(shard, record)
            generated += 1
        completed = index + 1
        if completed % int(args.report_every) == 0 or completed == len(rows):
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(rows),
                        "generated": generated,
                        "resumed": resumed,
                        "seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    output = output_dir / "train.jsonl"
    manifest_path = output_dir / "manifest.json"
    if output.exists() or manifest_path.exists():
        raise RuntimeError("refusing to overwrite consolidated adaptation data")
    shard_paths = sorted(shard_dir.glob("*.json"))
    if len(shard_paths) != len(rows):
        raise ValueError("adaptation shard count differs")
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for path in shard_paths:
            stream.write(path.read_text(encoding="utf-8").strip() + "\n")
    temporary.replace(output)
    manifest = {
        "version": "qwen_forward_adaptation_v9",
        "split": "train",
        "count": len(rows),
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "output": str(output),
        "output_sha256": sha256(output),
        "simulator_calls": generated,
        "resumed_shards": resumed,
        "held_out_validation_files_opened": [],
        "held_out_test_files_opened": [],
        "seconds": time.perf_counter() - started,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
