#!/usr/bin/env python3
"""Build full-sensor, train-only records for deterministic visual calibration."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .build_expanded_visual_eval_v10_5 import label_counts, missing_classes, selected_sources
from .build_visual_evidence_v10 import build_split
from .core import read_jsonl, stable_json_hash, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--master-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-version", default="visual_tool_calibration_v10_6")
    parser.add_argument("--size-px", type=int, default=384)
    parser.add_argument("--sensor-crop-px", type=int, default=1024)
    parser.add_argument("--raw", action="store_true", help="Render images without colored aids")
    args = parser.parse_args()

    sources = selected_sources(read_jsonl(args.records_jsonl))
    selection_path = args.output_dir / "source_selection.jsonl"
    write_jsonl(selection_path, sources)
    generated = build_split(
        selection_path,
        args.master_jsonl,
        args.output_dir,
        "train",
        args.size_px,
        args.dataset_version,
        args.sensor_crop_px,
        not args.raw,
        True,
        augment_train=False,
    )
    canonical_path = args.output_dir / "canonical" / "train.jsonl"
    write_jsonl(canonical_path, generated)
    counts = label_counts(generated)
    absent = missing_classes(counts)
    audit = {
        "passed": len(generated) == 3 * len(sources) and not absent["state"] and not absent["pair"],
        "training_only": True,
        "source_records": len(sources),
        "source_groups": len({row["group_id"] for row in sources}),
        "source_task_counts": dict(sorted(Counter(row["task_type"] for row in sources).items())),
        "generated_records": len(generated),
        "generated_groups": len({row["group_id"] for row in generated}),
        "label_counts": counts,
        "missing_target_classes": absent,
        "canonical_hash": stable_json_hash(generated),
        "rendering": {
            "size_px": args.size_px,
            "sensor_crop_px": args.sensor_crop_px,
            "full_sensor": args.sensor_crop_px == 1024,
            "instrumented_overlay": not args.raw,
            "shared_pair_normalization": True,
            "signed_difference": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_report.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("visual calibration dataset audit failed")


if __name__ == "__main__":
    main()
