#!/usr/bin/env python3
"""Score dedicated LLM direction predictions on canonical all-field records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from optics_understanding_sft.core import read_jsonl, write_jsonl
from optics_understanding_sft.direction_inverse_v1.train_direction_small import (
    CLASSES,
    CLASS_TO_INDEX,
    FIELDS,
    macro_f1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--small-summary", type=Path)
    return parser.parse_args()


def parse_directions(prediction: Mapping[str, Any]) -> dict[str, str] | None:
    parsed = prediction.get("parsed_json")
    if not isinstance(parsed, Mapping):
        return None
    answer = parsed.get("answer")
    if not isinstance(answer, Mapping):
        return None
    values = answer.get("directions")
    if not isinstance(values, Mapping):
        return None
    if not all(values.get(field) in CLASSES for field in FIELDS):
        return None
    return {field: str(values[field]) for field in FIELDS}


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.records)
    predictions = {row["example_id"]: row for row in read_jsonl(args.predictions)}
    missing = [row["example_id"] for row in records if row["example_id"] not in predictions]
    if missing:
        # Partial checkpoint panels are valid only when the prediction manifest
        # deliberately selected a prefix. Evaluate exactly that frozen subset.
        records = [row for row in records if row["example_id"] in predictions]
    target = np.asarray(
        [
            [CLASS_TO_INDEX[row["target"]["answer"]["directions"][field]] for field in FIELDS]
            for row in records
        ],
        dtype=np.int64,
    )
    parsed_rows = []
    predicted_values = []
    valid_count = 0
    for record in records:
        prediction = predictions[record["example_id"]]
        values = parse_directions(prediction)
        if values is None:
            values = {field: "no_change" for field in FIELDS}
        else:
            valid_count += 1
        predicted_values.append([CLASS_TO_INDEX[values[field]] for field in FIELDS])
        parsed_rows.append(
            {
                "example_id": record["example_id"],
                "group_id": record["group_id"],
                "target": record["target"]["answer"]["directions"],
                "prediction": values,
                "schema_valid": parse_directions(prediction) is not None,
                "raw_prediction_text": prediction.get("raw_prediction_text", ""),
            }
        )
    predicted = np.asarray(predicted_values, dtype=np.int64)
    metrics = macro_f1(target, predicted)
    collapse = {
        field: {
            "majority_label": CLASSES[Counter(predicted[:, index].tolist()).most_common(1)[0][0]],
            "majority_fraction": Counter(predicted[:, index].tolist()).most_common(1)[0][1] / len(records),
        }
        for index, field in enumerate(FIELDS)
    }
    summary: dict[str, Any] = {
        "split": args.split,
        "count": len(records),
        "schema_valid_rate": valid_count / len(records) if records else 0.0,
        "metrics": metrics,
        "collapse": collapse,
    }
    if args.small_summary and args.small_summary.exists():
        small = json.loads(args.small_summary.read_text(encoding="utf-8"))
        split_key = args.split if args.split in small else None
        if split_key:
            small_metrics = small[split_key]["shared_mlp"]
            summary["small_mlp_comparison"] = {
                "small_mlp_macro_f1": small_metrics["equal_field_macro_f1"],
                "llm_macro_f1": metrics["equal_field_macro_f1"],
                "llm_minus_small": metrics["equal_field_macro_f1"]
                - small_metrics["equal_field_macro_f1"],
            }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_jsonl(args.output_dir / "details.jsonl", parsed_rows)
    lines = [
        f"# Dedicated direction LLM: {args.split}",
        "",
        f"- Records: {len(records)}",
        f"- Schema-valid: {summary['schema_valid_rate']:.3f}",
        f"- Equal-field macro-F1: {metrics['equal_field_macro_f1']:.3f}",
        f"- All-five exact: {metrics['joint_exact']:.3f}",
    ]
    if "small_mlp_comparison" in summary:
        comparison = summary["small_mlp_comparison"]
        lines.append(
            f"- Difference from small MLP: {comparison['llm_minus_small']:+.3f}"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
