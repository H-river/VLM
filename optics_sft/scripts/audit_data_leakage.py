#!/usr/bin/env python3
"""Audit optics SFT rows and predictions for simple leakage signals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTROL_KEYS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
FORBIDDEN_METADATA_KEYS = {
    "control_plan",
    "label",
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
}
DIRECT_FEATURE_KEYS = {
    "centroid_error_px",
    "current_centroid_px",
    "target_centroid_px",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit optics SFT data and predictions for leakage signals.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
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


def sorted_top_level_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    return sorted(keys)


def sorted_metadata_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            keys.update(metadata.keys())
    return sorted(keys)


def metadata_key_hits(rows: list[dict[str, Any]], keys: set[str]) -> dict[str, int]:
    counts = {key: 0 for key in sorted(keys)}
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            continue
        for key in keys:
            if key in metadata:
                counts[key] += 1
    return counts


def image_filenames(rows: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for row in rows:
        for key in ("current_image_path", "target_image_path"):
            value = row.get(key)
            if isinstance(value, str):
                names.add(Path(value).name)
    return names


def sample_ids(rows: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        value = row.get("current_image_path")
        if isinstance(value, str):
            stem = Path(value).stem
            if stem.startswith("current_"):
                ids.add(stem.removeprefix("current_"))
    return ids


def row_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def control_plan_from_label(row: dict[str, Any]) -> dict[str, Any] | None:
    label = row.get("label")
    if isinstance(label, dict) and isinstance(label.get("control_plan"), dict):
        return label["control_plan"]
    return None


def control_plan_from_prediction(row: dict[str, Any]) -> dict[str, Any] | None:
    parsed = row.get("parsed_json")
    if isinstance(parsed, dict) and isinstance(parsed.get("control_plan"), dict):
        return parsed["control_plan"]
    return None


def exact_control_plan_match(pred_plan: dict[str, Any] | None, label_plan: dict[str, Any] | None) -> bool:
    if pred_plan is None or label_plan is None:
        return False
    return all(pred_plan.get(key) == label_plan.get(key) for key in CONTROL_KEYS)


def exact_prediction_match(prediction: dict[str, Any], label_row: dict[str, Any]) -> bool:
    parsed = prediction.get("parsed_json")
    label = label_row.get("label")
    return isinstance(parsed, dict) and isinstance(label, dict) and parsed == label


def expected_plan_from_centroid(metadata: dict[str, Any]) -> dict[str, float] | None:
    centroid = metadata.get("centroid_error_px")
    if not isinstance(centroid, dict):
        return None
    x = centroid.get("x")
    y = centroid.get("y")
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return None
    if not isinstance(y, (int, float)) or isinstance(y, bool):
        return None
    return {
        "lens_x_delta_mm": round(-0.002 * float(x), 5),
        "lens_y_delta_mm": round(-0.002 * float(y), 5),
        "camera_x_delta_mm": 0.0,
        "camera_y_delta_mm": 0.0,
    }


def audit_predictions(
    val_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    compared = min(len(val_rows), len(prediction_rows))
    exact_label_matches = 0
    exact_plan_matches = 0
    suspicious_examples: list[dict[str, Any]] = []

    for index, (label_row, prediction_row) in enumerate(zip(val_rows, prediction_rows)):
        label_plan = control_plan_from_label(label_row)
        pred_plan = control_plan_from_prediction(prediction_row)
        label_exact = exact_prediction_match(prediction_row, label_row)
        plan_exact = exact_control_plan_match(pred_plan, label_plan)
        if label_exact:
            exact_label_matches += 1
        if plan_exact:
            exact_plan_matches += 1
        if plan_exact and len(suspicious_examples) < 10:
            metadata = label_row.get("metadata") if isinstance(label_row.get("metadata"), dict) else {}
            suspicious_examples.append(
                {
                    "sample_index": prediction_row.get("sample_index", index),
                    "current_image_path": label_row.get("current_image_path"),
                    "target_image_path": label_row.get("target_image_path"),
                    "label_control_plan": label_plan,
                    "predicted_control_plan": pred_plan,
                    "metadata_centroid_error_px": metadata.get("centroid_error_px"),
                    "control_plan_from_metadata_centroid_rule": expected_plan_from_centroid(metadata),
                    "parsed_json_exactly_equals_label": label_exact,
                }
            )

    return {
        "prediction_rows": len(prediction_rows),
        "label_rows_compared": compared,
        "exact_parsed_json_equals_label_count": exact_label_matches,
        "exact_parsed_json_equals_label_rate": exact_label_matches / compared if compared else None,
        "exact_control_plan_match_count": exact_plan_matches,
        "exact_control_plan_match_rate": exact_plan_matches / compared if compared else None,
        "suspicious_exact_control_plan_examples": suspicious_examples,
    }


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(args.train_jsonl)
    val_rows = read_jsonl(args.val_jsonl)
    prediction_rows = read_jsonl(args.predictions_jsonl)

    train_image_names = image_filenames(train_rows)
    val_image_names = image_filenames(val_rows)
    train_sample_ids = sample_ids(train_rows)
    val_sample_ids = sample_ids(val_rows)
    train_hashes = {row_hash(row) for row in train_rows}
    val_hashes = {row_hash(row) for row in val_rows}

    report = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "predictions_rows": len(prediction_rows),
        "train_top_level_keys": sorted_top_level_keys(train_rows),
        "val_top_level_keys": sorted_top_level_keys(val_rows),
        "train_metadata_keys": sorted_metadata_keys(train_rows),
        "val_metadata_keys": sorted_metadata_keys(val_rows),
        "forbidden_answer_like_metadata_key_counts": {
            "train": metadata_key_hits(train_rows, FORBIDDEN_METADATA_KEYS),
            "val": metadata_key_hits(val_rows, FORBIDDEN_METADATA_KEYS),
        },
        "direct_feature_metadata_key_counts": {
            "train": metadata_key_hits(train_rows, DIRECT_FEATURE_KEYS),
            "val": metadata_key_hits(val_rows, DIRECT_FEATURE_KEYS),
        },
        "train_val_image_filename_overlap_count": len(train_image_names & val_image_names),
        "train_val_image_filename_overlap_examples": sorted(train_image_names & val_image_names)[:20],
        "train_val_sample_id_overlap_count": len(train_sample_ids & val_sample_ids),
        "train_val_sample_id_overlap_examples": sorted(train_sample_ids & val_sample_ids)[:20],
        "exact_train_val_row_duplicate_count": len(train_hashes & val_hashes),
        "image_root": str(args.image_root) if args.image_root is not None else None,
        "prediction_label_match_audit": audit_predictions(val_rows, prediction_rows),
    }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
