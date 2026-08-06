#!/usr/bin/env python3
"""Build a compact perturbed-image orchestration probe for the current checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .build_visual_tool_orchestration_v10_2 import derive
from .core import read_jsonl, stable_json_hash, write_jsonl
from .visual_state_tool_v10_1 import load_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--image-root", type=Path, nargs="+", required=True)
    parser.add_argument("--state-calibration", type=Path, required=True)
    parser.add_argument("--pair-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--dataset-version", default="v10_15_robust_probe")
    parser.add_argument("--dataset-name", default="visual_robust_orchestration_probe_v10_15")
    parser.add_argument("--omit-optional-difference-reference", action="store_true")
    args = parser.parse_args()
    if len(args.image_root) not in {1, len(args.records_jsonl)}:
        parser.error("provide one shared image root or one root per records JSONL")
    image_roots = (
        args.image_root * len(args.records_jsonl)
        if len(args.image_root) == 1
        else args.image_root
    )
    rng = random.Random(args.seed)
    buckets: dict[tuple[str, str], list[tuple[dict, Path]]] = defaultdict(list)
    for records_path, image_root in zip(args.records_jsonl, image_roots):
        for row in read_jsonl(records_path):
            perturbation = str(row["provenance"]["perturbation"])
            buckets[(perturbation, str(row["task_type"]))].append((row, image_root))
    selected = []
    for key in sorted(buckets):
        rows = buckets[key]
        rng.shuffle(rows)
        selected.extend(rows[:5])
    state_calibration = load_calibration(args.state_calibration)
    pair_calibration = load_calibration(args.pair_calibration)
    derived = [
        output
        for source, image_root in selected
        for output in derive(
            source,
            split="dev",
            image_root=image_root,
            state_calibration=state_calibration,
            pair_calibration=pair_calibration,
            dataset_version=args.dataset_version,
            include_optional_reference=not args.omit_optional_difference_reference,
        )
    ]
    write_jsonl(args.output_dir / "canonical" / "dev.jsonl", derived)
    manifest = {
        "dataset": args.dataset_name,
        "seed": args.seed,
        "source_count": len(selected),
        "record_count": len(derived),
        "source_counts": dict(
            sorted(
                Counter(
                    f"{row['provenance']['perturbation']}:{row['task_type']}"
                    for row, _image_root in selected
                ).items()
            )
        ),
        "canonical_hash": stable_json_hash(derived),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
