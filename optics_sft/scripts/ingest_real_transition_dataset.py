#!/usr/bin/env python3
"""Ingest measured before/action/after rows as forward-transition physics SFT data."""

from __future__ import annotations

import argparse
import csv
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

from optics_sft.physics.metadata_policy import (
    SAFE_PROMPT_METADATA_KEYS,
    assert_no_prompt_leakage,
    filter_safe_prompt_metadata,
)


ACTION_KEYS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
STATE_FIELDS = (
    "centroid_x_m",
    "centroid_y_m",
    "sigma_x_m",
    "sigma_y_m",
    "centroid_x_px",
    "centroid_y_px",
    "sigma_x_px",
    "sigma_y_px",
    "peak_intensity",
)
SPLITS = ("train", "val", "test")
DATASET_NAME = "physics_sft_real_transitions_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest real measured transition rows.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-csv", type=Path)
    input_group.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("../VLM_data/physics_sft_real_transitions_v1"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print converted rows without writing output files.",
    )
    return parser.parse_args()


def parse_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def set_dotted_value(row: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current: dict[str, Any] = row
    for part in parts[:-1]:
        existing = current.get(part)
        if not isinstance(existing, dict):
            existing = {}
            current[part] = existing
        current = existing
    current[parts[-1]] = value


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:
                    continue
                parsed_value = parse_scalar(value)
                if parsed_value is None:
                    continue
                set_dotted_value(parsed, key.strip(), parsed_value)
            rows.append(parsed)
    return rows


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
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


def read_input_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    if args.input_csv is not None:
        return read_csv_rows(args.input_csv), str(args.input_csv)
    if args.input_jsonl is not None:
        return read_jsonl_rows(args.input_jsonl), str(args.input_jsonl)
    raise ValueError("Expected --input-csv or --input-jsonl")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def number_value(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if is_number(value):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def nested_mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = row.get(key)
    return value if isinstance(value, Mapping) else None


def first_string(row: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def required_image_path(row: Mapping[str, Any], key: str) -> str:
    images = nested_mapping(row, "images")
    value = first_string(row, (key, f"prompt_inputs.images.{key}"))
    if value is None and images is not None:
        image_value = images.get(key)
        if isinstance(image_value, str) and image_value:
            value = image_value
    prompt_inputs = nested_mapping(row, "prompt_inputs")
    if value is None and prompt_inputs is not None:
        prompt_images = nested_mapping(prompt_inputs, "images")
        if prompt_images is not None:
            image_value = prompt_images.get(key)
            if isinstance(image_value, str) and image_value:
                value = image_value
    if value is None:
        raise ValueError(f"Missing required image field: {key}")
    return value


def normalize_image_path(path_text: str, image_root: Path) -> str:
    path = Path(path_text)
    if path.is_absolute():
        try:
            return path.relative_to(image_root).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def action_from_row(row: Mapping[str, Any]) -> dict[str, float]:
    action: dict[str, Any] = {}
    nested = nested_mapping(row, "action")
    prompt_inputs = nested_mapping(row, "prompt_inputs")
    prompt_action = nested_mapping(prompt_inputs, "action") if prompt_inputs is not None else None
    for key in ACTION_KEYS:
        value = None
        for source in (nested, prompt_action, row):
            if isinstance(source, Mapping) and key in source:
                value = source[key]
                break
        if value is None and f"action.{key}" in row:
            value = row[f"action.{key}"]
        if not is_number(value):
            raise ValueError(f"Missing or non-numeric action.{key}")
        action[key] = float(value)
    return action


def safe_metadata_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    direct = nested_mapping(row, "safe_setup_metadata")
    prompt_inputs = nested_mapping(row, "prompt_inputs")
    prompt_metadata = nested_mapping(prompt_inputs, "safe_setup_metadata") if prompt_inputs is not None else None

    for source in (direct, prompt_metadata):
        if isinstance(source, Mapping):
            metadata.update(dict(source))

    for key in SAFE_PROMPT_METADATA_KEYS:
        if key in row:
            metadata[key] = row[key]
        dotted = f"safe_setup_metadata.{key}"
        if dotted in row:
            metadata[key] = row[dotted]

    metadata = filter_safe_prompt_metadata(metadata)
    if not metadata:
        raise ValueError("Missing safe setup metadata")
    return metadata


def state_from_row(row: Mapping[str, Any], prefix: str) -> dict[str, float]:
    state: dict[str, float] = {}
    nested = nested_mapping(row, f"{prefix}_state")
    private_eval = nested_mapping(row, "private_eval")
    private_state = nested_mapping(private_eval, f"{prefix}_state") if private_eval is not None else None
    for key in STATE_FIELDS:
        value = None
        for source in (nested, private_state):
            if isinstance(source, Mapping) and key in source:
                value = source[key]
                break
        if value is None:
            for flat_key in (
                f"{prefix}_state.{key}",
                f"{prefix}.{key}",
                f"{prefix}_{key}",
            ):
                if flat_key in row:
                    value = row[flat_key]
                    break
        if is_number(value):
            state[key] = float(value)
    return state


def predicted_change(before_state: Mapping[str, Any], after_state: Mapping[str, Any]) -> dict[str, Any]:
    change: dict[str, Any] = {}
    if is_number(before_state.get("centroid_x_px")) and is_number(after_state.get("centroid_x_px")):
        change.setdefault("centroid_shift_px", {})["x"] = (
            float(after_state["centroid_x_px"]) - float(before_state["centroid_x_px"])
        )
    if is_number(before_state.get("centroid_y_px")) and is_number(after_state.get("centroid_y_px")):
        change.setdefault("centroid_shift_px", {})["y"] = (
            float(after_state["centroid_y_px"]) - float(before_state["centroid_y_px"])
        )
    if is_number(before_state.get("sigma_x_px")) and is_number(after_state.get("sigma_x_px")):
        change.setdefault("sigma_change_px", {})["x"] = (
            float(after_state["sigma_x_px"]) - float(before_state["sigma_x_px"])
        )
    if is_number(before_state.get("sigma_y_px")) and is_number(after_state.get("sigma_y_px")):
        change.setdefault("sigma_change_px", {})["y"] = (
            float(after_state["sigma_y_px"]) - float(before_state["sigma_y_px"])
        )
    if is_number(before_state.get("peak_intensity")) and is_number(after_state.get("peak_intensity")):
        change["peak_intensity_change"] = (
            float(after_state["peak_intensity"]) - float(before_state["peak_intensity"])
        )
    return change


def split_assignments(num_rows: int, val_ratio: float, test_ratio: float, seed: int) -> dict[int, str]:
    if num_rows <= 0:
        return {}
    if val_ratio < 0.0 or test_ratio < 0.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("--val-ratio and --test-ratio must be non-negative and sum to less than 1")
    indices = list(range(num_rows))
    random.Random(seed).shuffle(indices)
    val_count = int(round(num_rows * val_ratio))
    test_count = int(round(num_rows * test_ratio))
    val_indices = set(indices[:val_count])
    test_indices = set(indices[val_count : val_count + test_count])
    return {
        index: "val" if index in val_indices else "test" if index in test_indices else "train"
        for index in range(num_rows)
    }


def convert_row(row: Mapping[str, Any], index: int, split: str, image_root: Path) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or row.get("id") or f"real_fwd_{index:06d}")
    before_path = normalize_image_path(required_image_path(row, "before_image_path"), image_root)
    after_path = normalize_image_path(required_image_path(row, "after_image_path"), image_root)
    action = action_from_row(row)
    safe_metadata = safe_metadata_from_row(row)
    before_state = state_from_row(row, "before")
    after_state = state_from_row(row, "after")
    metrics_missing = not before_state or not after_state

    prompt_inputs = {
        "images": {
            "before_image_path": before_path,
            "after_image_path": after_path,
        },
        "safe_setup_metadata": safe_metadata,
        "action": action,
    }
    assert_no_prompt_leakage(prompt_inputs)

    target: dict[str, Any] = {
        "task": "measured_forward_transition",
        "metrics_missing": metrics_missing,
    }
    if after_state:
        target["predicted_after_state"] = copy.deepcopy(after_state)
    if before_state and after_state:
        change = predicted_change(before_state, after_state)
        if change:
            target["predicted_change"] = change

    source_metadata = {
        key: value
        for key, value in row.items()
        if key not in {"prompt_inputs", "target", "private_eval", "images", "action", "safe_setup_metadata"}
        and not str(key).startswith(("before_state", "after_state", "before_", "after_", "safe_setup_metadata."))
        and key not in ACTION_KEYS
    }

    return {
        "sample_id": sample_id,
        "sample_type": "forward_transition",
        "prompt_inputs": prompt_inputs,
        "target": target,
        "private_eval": {
            "before_state": before_state,
            "after_state": after_state,
            "metrics_missing": metrics_missing,
            "after_image_path": after_path,
            "source_metadata": source_metadata,
        },
        "split_tags": [split, "forward_transition", DATASET_NAME, "real_measured"],
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def rows_by_sample_type(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("sample_type", "unknown")) for row in rows))


def main() -> None:
    args = parse_args()
    raw_rows, input_path = read_input_rows(args)
    assignments = split_assignments(len(raw_rows), args.val_ratio, args.test_ratio, args.seed)
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    errors: list[str] = []

    for index, raw_row in enumerate(raw_rows):
        split = assignments.get(index, "train")
        try:
            converted = convert_row(raw_row, index, split, args.image_root)
        except Exception as exc:
            errors.append(f"row {index}: {type(exc).__name__}: {exc}")
            continue
        rows_by_split[split].append(converted)

    if errors:
        joined = "\n".join(errors[:10])
        raise ValueError(f"Failed to convert {len(errors)} rows:\n{joined}")

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    manifest = {
        "dataset": DATASET_NAME,
        "input_path": input_path,
        "image_root": str(args.image_root),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "counts": {split: len(rows_by_split[split]) for split in SPLITS},
        "counts_by_sample_type": {
            split: rows_by_sample_type(rows_by_split[split]) for split in SPLITS
        },
        "metrics_missing_count": sum(1 for row in all_rows if row["private_eval"]["metrics_missing"]),
        "metrics_present_count": sum(1 for row in all_rows if not row["private_eval"]["metrics_missing"]),
        "image_files_moved": False,
    }

    if args.dry_run:
        print(json.dumps({"manifest": manifest, "rows": all_rows[:5]}, indent=2, sort_keys=True))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(args.output_dir / f"{split}.jsonl", rows_by_split[split])
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote real transition dataset to {args.output_dir}")
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
