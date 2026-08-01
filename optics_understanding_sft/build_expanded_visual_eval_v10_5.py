#!/usr/bin/env python3
"""Build a broad, dev-only visual evaluation panel from every eligible task row.

Unlike the original visual-evidence builder, this script deliberately ignores the
source row's modality flag.  It replays two simulator states for every validation
record whose task semantics define an observable pair, then emits two state
classification records and one paired-direction record.  No training examples
or calibrator parameters are produced here.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .build_visual_evidence_v10 import build_split
from .core import make_qwen_record, read_jsonl, stable_json_hash, write_jsonl


REPLAY_PAIRS = {
    "causal_effects": ("before", "after"),
    "forward_prediction": ("before", "after"),
    "diagnosis": ("baseline", "observed"),
    "counterfactual_reasoning": ("scenario_a_after", "scenario_b_after"),
}

STATE_FIELDS = (
    "centroid_horizontal_region",
    "centroid_vertical_region",
    "sigma_x_band",
    "sigma_y_band",
)
PAIR_FIELDS = ("centroid_x", "centroid_y", "sigma_x", "sigma_y", "peak_intensity")
STATE_CLASSES = {
    "centroid_horizontal_region": {"left_of_center", "centered", "right_of_center"},
    "centroid_vertical_region": {"above_center", "centered", "below_center"},
    "sigma_x_band": {"narrow", "medium", "wide"},
    "sigma_y_band": {"narrow", "medium", "wide"},
}
PAIR_CLASSES = {field: {"decrease", "no_change", "increase"} for field in PAIR_FIELDS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--master-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-version", default="visual_eval_expanded_v10_5")
    parser.add_argument("--size-px", type=int, default=384)
    parser.add_argument("--sensor-crop-px", type=int, default=512)
    parser.add_argument("--raw", action="store_true", help="Render images without colored aids")
    return parser.parse_args()


def selected_sources(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return visual-shaped source rows with explicit simulator replay names."""

    selected: list[dict[str, Any]] = []
    for original in rows:
        replay_pair = REPLAY_PAIRS.get(str(original["task_type"]))
        if replay_pair is None:
            continue
        row = copy.deepcopy(dict(original))
        row["modality"] = "visual"
        row.setdefault("prompt_inputs", {})["images"] = [
            f"images/synthetic/{row['example_id']}_{name}.png" for name in replay_pair
        ]
        selected.append(row)
    return selected


def label_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    state: dict[str, Counter[str]] = defaultdict(Counter)
    pair: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        answer = row["target"]["answer"]
        if row["task_type"] == "visual_state_classification":
            for field in STATE_FIELDS:
                state[field][str(answer[field])] += 1
        elif row["task_type"] == "visual_pair_direction_extraction":
            values = answer["observed_direction_set"]
            for field in PAIR_FIELDS:
                pair[field][str(values[field])] += 1
    return {
        "state": {field: dict(sorted(counts.items())) for field, counts in sorted(state.items())},
        "pair": {field: dict(sorted(counts.items())) for field, counts in sorted(pair.items())},
    }


def missing_classes(counts: Mapping[str, Any]) -> dict[str, dict[str, list[str]]]:
    missing: dict[str, dict[str, list[str]]] = {"state": {}, "pair": {}}
    for kind, expected in (("state", STATE_CLASSES), ("pair", PAIR_CLASSES)):
        for field, values in expected.items():
            absent = sorted(values - set(counts[kind].get(field, {})))
            if absent:
                missing[kind][field] = absent
    return missing


def main() -> None:
    args = parse_args()
    sources = selected_sources(read_jsonl(args.records_jsonl))
    selection_path = args.output_dir / "source_selection.jsonl"
    write_jsonl(selection_path, sources)

    generated = build_split(
        selection_path,
        args.master_jsonl,
        args.output_dir,
        "dev",
        args.size_px,
        args.dataset_version,
        args.sensor_crop_px,
        not args.raw,
        True,
    )
    canonical_path = args.output_dir / "canonical" / "dev.jsonl"
    write_jsonl(canonical_path, generated)
    write_jsonl(
        args.output_dir / "exports" / "qwen" / "dev.jsonl",
        (make_qwen_record(row, include_target=True) for row in generated),
    )

    counts = label_counts(generated)
    absent = missing_classes(counts)
    example_ids = [str(row["example_id"]) for row in generated]
    image_paths = [
        str(image)
        for row in generated
        for image in row["prompt_inputs"].get("images", [])
    ]
    missing_images = sorted(
        path for path in set(image_paths) if not (args.output_dir / path).is_file()
    )
    source_task_counts = Counter(str(row["task_type"]) for row in sources)
    expected_records = len(sources) * 3
    audit = {
        "passed": (
            len(generated) == expected_records
            and len(example_ids) == len(set(example_ids))
            and not absent["state"]
            and not absent["pair"]
            and not missing_images
        ),
        "source_records": len(sources),
        "source_groups": len({str(row["group_id"]) for row in sources}),
        "source_task_counts": dict(sorted(source_task_counts.items())),
        "generated_records": len(generated),
        "expected_generated_records": expected_records,
        "generated_groups": len({str(row["group_id"]) for row in generated}),
        "duplicate_example_ids": len(example_ids) - len(set(example_ids)),
        "referenced_image_count": len(set(image_paths)),
        "missing_images": missing_images,
        "label_counts": counts,
        "missing_target_classes": absent,
        "class_coverage_complete": not absent["state"] and not absent["pair"],
    }
    manifest = {
        "dataset": args.dataset_version,
        "purpose": "independent_dev_only_visual_evaluation",
        "training_allowed": False,
        "source_records": str(args.records_jsonl.resolve()),
        "source_master": str(args.master_jsonl.resolve()),
        "canonical_hash": stable_json_hash(generated),
        "rendering": {
            "size_px": args.size_px,
            "sensor_crop_px": args.sensor_crop_px,
            "instrumented_overlay": not args.raw,
            "shared_pair_normalization": True,
            "signed_difference": True,
        },
        "audit": audit,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_report.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("expanded visual evaluation audit failed")


if __name__ == "__main__":
    main()
