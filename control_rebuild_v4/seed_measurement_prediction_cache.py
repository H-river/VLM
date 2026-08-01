#!/usr/bin/env python3
"""Validate and copy frozen train/validation measurement predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_quickcheck_12h"
DEFAULT_MEASUREMENT_DATA = REPO_ROOT.parent / "VLM_data/measurement_rebuild_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--measurement-data",
        type=Path,
        default=DEFAULT_MEASUREMENT_DATA,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_keys(
    state_path: Path,
    conditions: list[str],
) -> list[list[str]]:
    keys = []
    with state_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            keys.extend([str(row["state_id"]), condition] for condition in conditions)
    return keys


def validate_cache(
    path: Path,
    keys_expected: list[list[str]],
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != {"keys_json", "predictions"}:
            raise ValueError(f"{path}: cache members differ")
        keys = json.loads(str(loaded["keys_json"].item()))
        values = np.asarray(loaded["predictions"])
    if keys != keys_expected:
        raise ValueError(f"{path}: state and condition keys differ")
    if values.shape != (len(keys_expected), 5):
        raise ValueError(f"{path}: prediction shape differs")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{path}: predictions are not floating point")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: predictions contain non-finite values")
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256(path),
        "key_count": len(keys),
        "prediction_shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite": True,
    }


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    measurement_data = args.measurement_data.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((measurement_data / "config.json").read_text(encoding="utf-8"))
    conditions = [str(value) for value in config["conditions"]]
    split_evidence = {}
    for split in ("train", "val"):
        keys = expected_keys(
            measurement_data / "states" / f"{split}.jsonl",
            conditions,
        )
        source = source_dir / f"measurement_predictions_{split}.npz"
        destination = output_dir / source.name
        source_evidence = validate_cache(source, keys)
        if (
            not destination.is_file()
            or sha256(destination) != source_evidence["sha256"]
        ):
            temporary = destination.with_suffix(".npz.tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        destination_evidence = validate_cache(destination, keys)
        if destination_evidence["sha256"] != source_evidence["sha256"]:
            raise ValueError(f"{split}: copied cache digest differs")
        split_evidence[split] = {
            "source": source_evidence,
            "destination": destination_evidence,
        }
    result = {
        "version": "control_rebuild_v4_frozen_measurement_cache_seed",
        "measurement_data": str(measurement_data),
        "conditions": conditions,
        "splits": split_evidence,
        "calibration_applied_in_cache": False,
        "calibration_application": (
            "train_visual_scorer applies the selected v4 calibrator after loading"
        ),
        "held_out_test_used": False,
        "complete": True,
    }
    output = output_dir / "measurement_prediction_cache_provenance.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
