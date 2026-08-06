#!/usr/bin/env python3
"""Build paired full-visual and image-withheld evaluation panels.

The two panels retain identical example IDs and targets.  The image-withheld
variant changes only the pixel channel and explicitly marks that change in the
prompt, allowing the existing evaluator to compare like-for-like cases.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .core import read_jsonl, stable_json_hash, write_jsonl


SUPPORTED_TASKS = {
    "causal_effects",
    "forward_prediction",
    "diagnosis",
    "counterfactual_reasoning",
    "visual_state_classification",
    "visual_pair_direction_extraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def image_withheld_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a prompt-identical ablation except for removal of image inputs."""

    result = copy.deepcopy(record)
    inputs = result["prompt_inputs"]
    image_paths = list(inputs.get("images", []))
    if not image_paths:
        raise ValueError(f"Visual record has no images: {record['example_id']}")

    old_format = inputs.get("observation_format")
    new_format = "images withheld for controlled ablation; no pixel observations supplied"
    inputs["images"] = []
    inputs["observation_format"] = new_format
    result["modality"] = "text"
    if old_format is not None:
        result["prompt"] = str(result["prompt"]).replace(str(old_format), new_format)
    marker = (
        "\n\nControlled ablation: the image channel has been withheld. "
        "Use only the remaining prompt-visible information."
    )
    contract_marker = "\n\nReturn only strict JSON"
    if contract_marker in result["prompt"]:
        result["prompt"] = result["prompt"].replace(
            contract_marker, marker + contract_marker, 1
        )
    else:
        result["prompt"] += marker
    result["provenance"]["ablation"] = "images_withheld"
    result["provenance"]["withheld_image_count"] = len(image_paths)
    return result


def build_panels(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    visual = [row for row in rows if row.get("modality") == "visual"]
    unsupported = sorted({row.get("task_type") for row in visual} - SUPPORTED_TASKS)
    if unsupported:
        raise ValueError(f"Unexpected visual task types: {unsupported}")
    if len({row["example_id"] for row in visual}) != len(visual):
        raise ValueError("Visual example IDs are not unique")
    withheld = [image_withheld_record(row) for row in visual]
    return visual, withheld


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.source_jsonl)
    visual, withheld = build_panels(source_rows)
    if not visual:
        raise ValueError("Source contains no visual records")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "full_visual.jsonl"
    withheld_path = args.output_dir / "images_withheld.jsonl"
    write_jsonl(full_path, visual)
    write_jsonl(withheld_path, withheld)

    task_counts = Counter(row["task_type"] for row in visual)
    manifest = {
        "source_jsonl": str(args.source_jsonl.resolve()),
        "record_count": len(visual),
        "task_counts": dict(sorted(task_counts.items())),
        "example_ids_identical": [row["example_id"] for row in visual]
        == [row["example_id"] for row in withheld],
        "targets_identical": [row["target"] for row in visual]
        == [row["target"] for row in withheld],
        "full_visual_hash": stable_json_hash(visual),
        "images_withheld_hash": stable_json_hash(withheld),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
