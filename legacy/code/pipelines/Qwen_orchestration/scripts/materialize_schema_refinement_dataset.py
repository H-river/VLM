#!/usr/bin/env python3
"""Create a deterministic, checksum-audited Stage-2 schema-refinement set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


READY_ROUTES = (
    "measure_beam_profile_v1",
    "predict_direction_from_state_v1",
    "predict_direction_from_image_v1",
    "predict_forward_from_state_v1",
    "predict_forward_from_image_v1",
    "select_inverse_action_from_states_v1",
    "select_inverse_action_from_images_v1",
)
TARGET_COUNTS = {
    "needs_clarification": 1200,
    "unsupported": 200,
    "measure_beam_profile_v1": 86,
    "predict_direction_from_state_v1": 86,
    "predict_direction_from_image_v1": 86,
    "predict_forward_from_state_v1": 86,
    "predict_forward_from_image_v1": 86,
    "select_inverse_action_from_states_v1": 85,
    "select_inverse_action_from_images_v1": 85,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(seed: int, example_id: str, purpose: str) -> str:
    return hashlib.sha256(
        f"{seed}:{purpose}:{example_id}".encode("utf-8")
    ).hexdigest()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    qwen_path = source_root / "exports/qwen/train_stage2.jsonl"
    canonical_path = source_root / "canonical/train.jsonl"
    qwen_rows = read_jsonl(qwen_path)
    canonical_rows = read_jsonl(canonical_path)
    category_by_id = {
        row["example_id"]: row["category"] for row in canonical_rows
    }
    if len(category_by_id) != len(canonical_rows):
        raise RuntimeError("canonical training example IDs are not unique")
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in qwen_rows:
        example_id = row["example_id"]
        if example_id in seen:
            raise RuntimeError(f"duplicate Qwen example ID: {example_id}")
        seen.add(example_id)
        try:
            category = category_by_id[example_id]
        except KeyError as error:
            raise RuntimeError(
                f"Qwen example missing from canonical data: {example_id}"
            ) from error
        by_category[category].append(row)

    selected: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    for category, count in TARGET_COUNTS.items():
        ordered = sorted(
            by_category[category],
            key=lambda row: rank(args.seed, row["example_id"], category),
        )
        if len(ordered) < count:
            raise RuntimeError(
                f"category {category} has {len(ordered)} rows, needs {count}"
            )
        selected.extend(ordered[:count])
        selected_counts[category] = count
    if len(selected) != 2000:
        raise RuntimeError(f"selected {len(selected)} rows, expected 2000")
    if len({row["example_id"] for row in selected}) != len(selected):
        raise RuntimeError("selected refinement rows are not unique")
    selected.sort(
        key=lambda row: rank(args.seed, row["example_id"], "output-order")
    )

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "train_stage2_schema_refinement.jsonl"
    with output_path.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    manifest = {
        "format": "qwen_orchestration_schema_refinement_dataset_v1",
        "seed": args.seed,
        "source_root": str(source_root),
        "selection_policy": (
            "unique records; deterministic SHA-256 rank within category; "
            "1200 clarification, 600 balanced ready-route, 200 unsupported"
        ),
        "count": len(selected),
        "category_counts": selected_counts,
        "source_sha256": {
            "qwen_train": sha256(qwen_path),
            "canonical_train": sha256(canonical_path),
        },
        "output_sha256": sha256(output_path),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
