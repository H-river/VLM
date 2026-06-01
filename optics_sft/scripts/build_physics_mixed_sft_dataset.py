#!/usr/bin/env python3
"""Merge physics task JSONLs into one mixed-task SFT dataset."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optics_sft.physics.prompt_builder import assert_prompt_inputs_safe


SPLITS = ("train", "val", "test")
WEIGHTS_BY_TYPE_ARGS = {
    "forward_transition": "transition_weight",
    "inverse_control": "inverse_weight",
    "counterfactual_pair": "counterfactual_weight",
    "trajectory": "trajectory_weight",
}
INPUT_ARGS_BY_TYPE = {
    "forward_transition": "transition_jsonl",
    "inverse_control": "inverse_jsonl",
    "counterfactual_pair": "counterfactual_jsonl",
    "trajectory": "trajectory_jsonl",
}
SCHEMA_PATH = ROOT / "optics_sft" / "data_schema" / "physics_sft_sample_schema.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge physics task JSONLs into train/val/test mixed-task SFT files."
    )
    parser.add_argument("--transition-jsonl", type=Path)
    parser.add_argument("--inverse-jsonl", type=Path)
    parser.add_argument("--counterfactual-jsonl", type=Path)
    parser.add_argument("--trajectory-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("../VLM_data/physics_sft_mixed_v1"))
    parser.add_argument(
        "--image-root-strategy",
        choices=("copy", "symlink", "keep_relative"),
        default="keep_relative",
    )
    parser.add_argument("--transition-weight", type=float, default=0.50)
    parser.add_argument("--inverse-weight", type=float, default=0.30)
    parser.add_argument("--counterfactual-weight", type=float, default=0.15)
    parser.add_argument("--trajectory-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
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


def source_split_from_path(path: Path) -> str | None:
    stem = path.stem.lower()
    for split in SPLITS:
        if stem == split or stem.endswith(f"_{split}") or stem.endswith(f"-{split}"):
            return split
    return None


def row_split(row: dict[str, Any], source_path: Path) -> str:
    tags = row.get("split_tags", [])
    if isinstance(tags, list):
        for split in SPLITS:
            if split in tags:
                return split
    source_split = source_split_from_path(source_path)
    if source_split is not None:
        return source_split
    return "train"


def image_root_for_jsonl(path: Path) -> Path:
    manifest_path = path.parent / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        image_root = manifest.get("image_root")
        if isinstance(image_root, str) and image_root:
            candidate = Path(image_root)
            return candidate if candidate.is_absolute() else path.parent / candidate
    return path.parent / "images"


def source_name(path: Path, sample_type: str) -> str:
    parent_name = path.parent.name
    if parent_name:
        return parent_name
    return sample_type


def collect_image_paths(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, str) and key.endswith("_image_path"):
                paths.append((path, value))
            else:
                paths.extend(collect_image_paths(value, path))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(collect_image_paths(value, path))
    return paths


def set_nested_dict_value(obj: dict[str, Any], dotted_path: str, value: str) -> None:
    parts = dotted_path.split(".")
    current: Any = obj
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def materialize_images(
    row: dict[str, Any],
    *,
    input_image_root: Path,
    output_image_root: Path,
    source_prefix: str,
    strategy: str,
) -> dict[str, Any]:
    copied = copy.deepcopy(row)
    output_image_root.mkdir(parents=True, exist_ok=True)
    images = copied.get("prompt_inputs", {}).get("images", {})
    for key_path, image_path in collect_image_paths(images):
        source_path = Path(image_path)
        if not source_path.is_absolute():
            source_path = input_image_root / source_path
        if not source_path.exists():
            raise FileNotFoundError(f"Image referenced by {copied.get('sample_id')} does not exist: {source_path}")
        if strategy == "keep_relative":
            relative_path = Path(os.path.relpath(source_path.resolve(), output_image_root.resolve()))
            set_nested_dict_value(images, key_path, relative_path.as_posix())
            continue

        relative_path = Path(source_prefix) / Path(image_path)
        dest_path = output_image_root / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if strategy == "copy":
            shutil.copy2(source_path, dest_path)
        elif strategy == "symlink":
            if dest_path.exists() or dest_path.is_symlink():
                dest_path.unlink()
            dest_path.symlink_to(source_path.resolve())
        else:
            raise ValueError(f"Unsupported image root strategy: {strategy}")
        set_nested_dict_value(images, key_path, relative_path.as_posix())
    return copied


def normalize_weights(args: argparse.Namespace) -> dict[str, float]:
    weights = {
        sample_type: float(getattr(args, arg_name))
        for sample_type, arg_name in WEIGHTS_BY_TYPE_ARGS.items()
    }
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("Sampling weights must be non-negative")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("At least one sampling weight must be positive")
    return {key: value / total for key, value in weights.items()}


def split_limit(args: argparse.Namespace, split: str) -> int | None:
    value = getattr(args, f"max_{split}_samples")
    if value is not None and value < 0:
        raise ValueError(f"--max-{split}-samples must be non-negative")
    return value


def weighted_sample_split(
    rows_by_type: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
    limit: int | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    available_total = sum(len(rows) for rows in rows_by_type.values())
    if available_total == 0:
        return []
    target_total = available_total if limit is None else min(limit, available_total)

    pools = {sample_type: list(rows) for sample_type, rows in rows_by_type.items()}
    for rows in pools.values():
        rng.shuffle(rows)

    selected_counts = {sample_type: 0 for sample_type in weights}
    raw_allocations = {sample_type: weights[sample_type] * target_total for sample_type in weights}
    for sample_type, raw in raw_allocations.items():
        selected_counts[sample_type] = min(len(pools.get(sample_type, [])), int(raw))

    while sum(selected_counts.values()) < target_total:
        candidates = [
            sample_type
            for sample_type in weights
            if selected_counts[sample_type] < len(pools.get(sample_type, []))
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda sample_type: (
                raw_allocations[sample_type] - selected_counts[sample_type],
                weights[sample_type],
            ),
            reverse=True,
        )
        selected_counts[candidates[0]] += 1

    selected: list[dict[str, Any]] = []
    for sample_type, count in selected_counts.items():
        selected.extend(pools.get(sample_type, [])[:count])
    rng.shuffle(selected)
    return selected


def load_schema_validator() -> tuple[Any | None, str]:
    try:
        import jsonschema
    except ImportError:
        return None, "jsonschema_not_installed"

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return validator, "enabled"


def validate_schema(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    validator, status = load_schema_validator()
    if validator is None:
        return {"status": status, "error_count": 0, "examples": []}

    examples: list[dict[str, str]] = []
    error_count = 0
    for row in rows:
        row_errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
        error_count += len(row_errors)
        for error in row_errors[: max(0, 3 - len(examples))]:
            examples.append(
                {
                    "sample_id": str(row.get("sample_id", "")),
                    "sample_type": str(row.get("sample_type", "")),
                    "path": ".".join(str(part) for part in error.path),
                    "message": error.message,
                }
            )
        if len(examples) >= 3:
            continue
    return {"status": status, "error_count": error_count, "examples": examples}


def audit_leakage(rows: Iterable[dict[str, Any]]) -> None:
    failures: list[str] = []
    for row in rows:
        try:
            assert_prompt_inputs_safe(row)
        except ValueError as exc:
            failures.append(f"{row.get('sample_id', '<missing-id>')}: {exc}")
    if failures:
        joined = "\n".join(failures[:10])
        raise ValueError(f"Prompt leakage audit failed:\n{joined}")


def rows_by_sample_type(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("sample_type", "unknown")) for row in rows))


def load_inputs(args: argparse.Namespace) -> dict[str, dict[str, list[dict[str, Any]]]]:
    loaded: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: defaultdict(list) for split in SPLITS
    }
    any_input = False
    for sample_type, arg_name in INPUT_ARGS_BY_TYPE.items():
        jsonl_path = getattr(args, arg_name)
        if jsonl_path is None:
            continue
        any_input = True
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Input JSONL does not exist: {jsonl_path}")

        rows = read_jsonl(jsonl_path)
        input_image_root = image_root_for_jsonl(jsonl_path)
        source_prefix = source_name(jsonl_path, sample_type)
        for row in rows:
            if row.get("sample_type") != sample_type:
                raise ValueError(
                    f"{jsonl_path} contains sample_type {row.get('sample_type')!r}, expected {sample_type!r}"
                )
            split = row_split(row, jsonl_path)
            materialized = materialize_images(
                row,
                input_image_root=input_image_root,
                output_image_root=args.output_dir / "images",
                source_prefix=source_prefix,
                strategy=args.image_root_strategy,
            )
            loaded[split][sample_type].append(materialized)

    if not any_input:
        raise ValueError("At least one input JSONL must be provided.")
    return {split: dict(rows_by_type) for split, rows_by_type in loaded.items()}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = normalize_weights(args)
    loaded = load_inputs(args)

    output_rows: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        output_rows[split] = weighted_sample_split(
            loaded[split],
            weights,
            split_limit(args, split),
            rng,
        )
        audit_leakage(output_rows[split])
        write_jsonl(args.output_dir / f"{split}.jsonl", output_rows[split])

    all_rows = [row for rows in output_rows.values() for row in rows]
    schema_validation = validate_schema(all_rows)
    manifest = {
        "dataset": "physics_sft_mixed_v1",
        "output_dir": str(args.output_dir),
        "image_root": "images",
        "image_root_strategy": args.image_root_strategy,
        "seed": args.seed,
        "weights": weights,
        "counts": {split: len(rows) for split, rows in output_rows.items()},
        "counts_by_sample_type": {
            split: rows_by_sample_type(rows)
            for split, rows in output_rows.items()
        },
        "input_counts_by_sample_type": {
            split: {sample_type: len(rows) for sample_type, rows in loaded[split].items()}
            for split in SPLITS
        },
        "schema_validation": schema_validation,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote mixed dataset to {args.output_dir}")
    print(json.dumps(manifest["counts"], sort_keys=True))
    if schema_validation["status"] == "enabled" and schema_validation["error_count"]:
        print(
            "[warn] schema validation reported "
            f"{schema_validation['error_count']} errors; see manifest.json for examples"
        )
    elif schema_validation["status"] == "jsonschema_not_installed":
        print("[warn] jsonschema is not installed; skipped schema validation")


if __name__ == "__main__":
    main()
