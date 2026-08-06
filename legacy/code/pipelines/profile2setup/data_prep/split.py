"""Deterministic JSONL dataset splitting utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def split_jsonl(
    input_path,
    out_dir,
    train_frac=0.8,
    val_frac=0.1,
    test_frac=0.1,
    seed=42,
):
    """Split a JSONL dataset into train/val/test deterministically."""
    total_frac = float(train_frac) + float(val_frac) + float(test_frac)
    if abs(total_frac - 1.0) > 1e-9:
        raise ValueError(
            f"Fractions must sum to 1.0, got {total_frac} "
            f"(train={train_frac}, val={val_frac}, test={test_frac})"
        )

    input_path = Path(input_path)
    out_dir = Path(out_dir)

    records = _read_jsonl(input_path)

    ids = [rec.get("id") for rec in records]
    if any(not rid for rid in ids):
        raise ValueError("All records must contain non-empty 'id'")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate record ids detected in input dataset")

    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * float(train_frac))
    n_val = int(n * float(val_frac))
    n_test = n - n_train - n_val

    train_records = records[:n_train]
    val_records = records[n_train:n_train + n_val]
    test_records = records[n_train + n_val:]

    # extra guard against overlap
    train_ids = {rec["id"] for rec in train_records}
    val_ids = {rec["id"] for rec in val_records}
    test_ids = {rec["id"] for rec in test_records}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Duplicate ids across splits after splitting")

    _write_jsonl(out_dir / "train.jsonl", train_records)
    _write_jsonl(out_dir / "val.jsonl", val_records)
    _write_jsonl(out_dir / "test.jsonl", test_records)

    print(
        "split summary: "
        f"total={n}, train={len(train_records)}, val={len(val_records)}, test={len(test_records)}"
    )

    return {
        "total": n,
        "train": len(train_records),
        "val": len(val_records),
        "test": len(test_records),
        "out_dir": str(out_dir),
    }
