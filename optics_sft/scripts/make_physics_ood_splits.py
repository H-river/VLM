#!/usr/bin/env python3
"""Create ID/OOD JSONL splits for physics-aware SFT rows."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optics_sft.physics.prompt_builder import assert_prompt_inputs_safe


OOD_PARAMETERS = (
    "lens_focal_length_mm",
    "lens_to_camera_mm",
    "source_to_lens_mm",
    "beam_waist_mm",
    "pixel_size_um",
)
OUTPUT_SPLITS = ("train", "val", "test_id", "test_ood")
OLD_SPLIT_TAGS = {
    "train",
    "val",
    "test",
    "test_id",
    "test_ood",
    "id",
    "ood",
}
VAL_FRACTION = 0.10
TEST_ID_FRACTION = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create physics-aware ID/OOD dataset splits.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ood-parameter", choices=OOD_PARAMETERS, required=True)
    parser.add_argument("--train-min", type=float, required=True)
    parser.add_argument("--train-max", type=float, required=True)
    parser.add_argument("--ood-min", type=float, required=True)
    parser.add_argument("--ood-max", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def prompt_inputs(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("prompt_inputs")
    if not isinstance(value, Mapping):
        raise ValueError(f"Row {row.get('sample_id', '<missing-id>')} has no object prompt_inputs")
    return value


def safe_setup_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = prompt_inputs(row).get("safe_setup_metadata")
    if not isinstance(value, Mapping):
        raise ValueError(f"Row {row.get('sample_id', '<missing-id>')} has no object safe_setup_metadata")
    return value


def collect_parameter_values(metadata: Mapping[str, Any], parameter: str) -> list[float]:
    values: list[float] = []
    candidate = metadata.get(parameter)
    if is_number(candidate):
        values.append(float(candidate))

    for nested in metadata.values():
        if isinstance(nested, Mapping):
            values.extend(collect_parameter_values(nested, parameter))
    return values


def in_range(value: float, low: float, high: float) -> bool:
    return low <= value <= high


def classify_values(
    values: list[float],
    *,
    train_min: float,
    train_max: float,
    ood_min: float,
    ood_max: float,
) -> str:
    if not values:
        return "missing_parameter"
    if any(in_range(value, ood_min, ood_max) for value in values):
        return "ood"
    if all(in_range(value, train_min, train_max) for value in values):
        return "id"
    return "out_of_range"


def split_id_rows(rows: list[dict[str, Any]], rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    total = len(shuffled)
    val_count = int(round(total * VAL_FRACTION))
    test_id_count = int(round(total * TEST_ID_FRACTION))
    val_rows = shuffled[:val_count]
    test_id_rows = shuffled[val_count : val_count + test_id_count]
    train_rows = shuffled[val_count + test_id_count :]
    return {
        "train": train_rows,
        "val": val_rows,
        "test_id": test_id_rows,
    }


def cleaned_tags(row: Mapping[str, Any]) -> list[str]:
    tags = row.get("split_tags")
    if not isinstance(tags, list):
        return []
    cleaned: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        if tag in OLD_SPLIT_TAGS or tag.startswith("ood:"):
            continue
        if tag not in cleaned:
            cleaned.append(tag)
    return cleaned


def with_split_tags(row: dict[str, Any], split: str, ood_parameter: str) -> dict[str, Any]:
    copied = copy.deepcopy(row)
    tags = cleaned_tags(copied)
    tags.append(split)
    if split == "test_ood":
        tags.extend(["ood", f"ood:{ood_parameter}"])
    else:
        tags.append("id")
    copied["split_tags"] = tags
    return copied


def rows_by_sample_type(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("sample_type", "unknown")) for row in rows))


def audit_prompt_inputs(rows: Iterable[dict[str, Any]]) -> None:
    failures: list[str] = []
    for row in rows:
        try:
            assert_prompt_inputs_safe(row)
        except ValueError as exc:
            failures.append(f"{row.get('sample_id', '<missing-id>')}: {exc}")
    if failures:
        joined = "\n".join(failures[:10])
        raise ValueError(f"Prompt leakage audit failed:\n{joined}")


def validate_ranges(args: argparse.Namespace) -> None:
    if args.train_min > args.train_max:
        raise ValueError("--train-min must be <= --train-max")
    if args.ood_min > args.ood_max:
        raise ValueError("--ood-min must be <= --ood-max")


def main() -> None:
    args = parse_args()
    validate_ranges(args)
    rng = random.Random(args.seed)
    rows = read_jsonl(args.input_jsonl)
    audit_prompt_inputs(rows)

    id_candidates: list[dict[str, Any]] = []
    ood_candidates: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    parameter_values: dict[str, list[float]] = {}

    for row in rows:
        values = collect_parameter_values(safe_setup_metadata(row), args.ood_parameter)
        parameter_values[str(row.get("sample_id", ""))] = values
        classification = classify_values(
            values,
            train_min=args.train_min,
            train_max=args.train_max,
            ood_min=args.ood_min,
            ood_max=args.ood_max,
        )
        if classification == "id":
            id_candidates.append(row)
        elif classification == "ood":
            ood_candidates.append(row)
        else:
            ignored.append({**row, "_ood_split_ignore_reason": classification})

    split_rows = split_id_rows(id_candidates, rng)
    rng.shuffle(ood_candidates)
    split_rows["test_ood"] = ood_candidates

    output_rows = {
        split: [
            with_split_tags(row, split, args.ood_parameter)
            for row in split_rows.get(split, [])
        ]
        for split in OUTPUT_SPLITS
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in OUTPUT_SPLITS:
        write_jsonl(args.output_dir / f"{split}.jsonl", output_rows[split])

    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(args.output_dir),
        "ood_parameter": args.ood_parameter,
        "seed": args.seed,
        "ranges": {
            "train": {"min": args.train_min, "max": args.train_max},
            "ood": {"min": args.ood_min, "max": args.ood_max},
        },
        "split_policy": {
            "id_candidate_rule": "all extracted parameter values must be within train range",
            "ood_candidate_rule": "any extracted parameter value within OOD range is assigned to test_ood",
            "id_split_fractions": {
                "train": 1.0 - VAL_FRACTION - TEST_ID_FRACTION,
                "val": VAL_FRACTION,
                "test_id": TEST_ID_FRACTION,
            },
            "image_files_moved": False,
        },
        "counts": {split: len(output_rows[split]) for split in OUTPUT_SPLITS},
        "counts_by_sample_type": {
            split: rows_by_sample_type(output_rows[split])
            for split in OUTPUT_SPLITS
        },
        "input_count": len(rows),
        "id_candidate_count": len(id_candidates),
        "ood_candidate_count": len(ood_candidates),
        "ignored_count": len(ignored),
        "ignored_by_reason": dict(
            Counter(str(row.get("_ood_split_ignore_reason", "unknown")) for row in ignored)
        ),
        "parameter_value_examples": {
            sample_id: values
            for sample_id, values in list(parameter_values.items())[:10]
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote OOD splits to {args.output_dir}")
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
