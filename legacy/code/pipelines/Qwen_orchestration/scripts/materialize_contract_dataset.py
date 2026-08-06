#!/usr/bin/env python3
"""Create checksum-audited Qwen rows carrying the shared system contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Qwen_orchestration.runtime.prompt_contract import (
    DECISION_SYSTEM_CONTRACT,
    apply_decision_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def materialize(source: Path, output: Path) -> int:
    rows = read_jsonl(source)
    seen: set[str] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            example_id = row["example_id"]
            if example_id in seen:
                raise RuntimeError(f"duplicate example_id: {example_id}")
            seen.add(example_id)
            transformed = dict(row)
            transformed["prompt"] = apply_decision_contract(row["prompt"])
            if transformed["prompt"][0]["content"][0]["text"] != DECISION_SYSTEM_CONTRACT:
                raise RuntimeError(f"contract insertion failed: {example_id}")
            stream.write(
                json.dumps(
                    transformed,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    return len(rows)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    sources = {
        split: source_root / "exports/qwen" / f"{split}_stage2.jsonl"
        for split in ("train", "val")
    }
    outputs = {
        split: output_root / f"{split}_stage2_contract.jsonl"
        for split in ("train", "val")
    }
    counts = {
        split: materialize(sources[split], outputs[split])
        for split in ("train", "val")
    }
    manifest = {
        "format": "qwen_orchestration_contract_dataset_v1",
        "source_root": str(source_root),
        "contract_sha256": hashlib.sha256(
            DECISION_SYSTEM_CONTRACT.encode("utf-8")
        ).hexdigest(),
        "counts": counts,
        "source_sha256": {
            split: sha256(path) for split, path in sources.items()
        },
        "output_sha256": {
            split: sha256(path) for split, path in outputs.items()
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
