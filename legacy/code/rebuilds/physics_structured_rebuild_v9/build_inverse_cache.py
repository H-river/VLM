#!/usr/bin/env python3
"""Build numerical-inverse requests using the frozen strongest v7 forward."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import read_jsonl
from control_rebuild_v3.train_forward import configure
from control_rebuild_v4.inverse_data import derived_inverse_pairs
from control_rebuild_v4.train_inverse import prepare_requests
from control_rebuild_v5.train_inverse import selected_truth
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)

DEFAULT_OLD_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_DIFFICULT_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v4_quickcheck"
)
DEFAULT_ADDITIONAL_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v5_numerical"
)
DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7.pt"
)
DEFAULT_ERROR_BANK = (
    REPO_ROOT.parent
    / "VLM_runs/control_rebuild_v4_one_seed/measurement_error_bank_v4.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent / "VLM_data/physics_structured_inverse_v9"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-data", type=Path, default=DEFAULT_OLD_DATA)
    parser.add_argument(
        "--difficult-data",
        type=Path,
        default=DEFAULT_DIFFICULT_DATA,
    )
    parser.add_argument(
        "--additional-data",
        type=Path,
        default=DEFAULT_ADDITIONAL_DATA,
    )
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--error-bank", type=Path, default=DEFAULT_ERROR_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_arrays(
    arrays: dict[str, Any],
    pairs: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    return {
        "contexts": np.asarray(arrays["contexts"], dtype=np.float32),
        "desired": np.asarray(
            arrays["desired_observed"],
            dtype=np.float32,
        ),
        "candidate_states": np.asarray(
            arrays["candidate_bank"][arrays["candidate_indices"]],
            dtype=np.float32,
        ),
        "positives": np.asarray(arrays["positives"], dtype=np.bool_),
        "statuses": np.asarray(arrays["statuses"], dtype=np.int64),
        "selected_truth": selected_truth(pairs),
        "noisy": np.asarray(arrays["noisy"], dtype=np.bool_),
    }


def save_block(
    path: Path,
    arrays: dict[str, Any],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    output = block_arrays(arrays, pairs)
    np.savez(path, **output)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "request_count": int(len(output["contexts"])),
        "reachable_count": int(output["positives"].any(axis=1).sum()),
        "measurement_augmented_count": int(output["noisy"].sum()),
        "candidate_count": int(output["candidate_states"].shape[1]),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        raise RuntimeError(f"refusing to overwrite inverse cache: {metadata_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    old_train_rows = read_jsonl(
        args.old_data.resolve() / "grids/train.jsonl"
    )
    old_val_rows = read_jsonl(args.old_data.resolve() / "grids/val.jsonl")
    difficult_train_rows = read_jsonl(
        args.difficult_data.resolve() / "grids/train.jsonl"
    )
    difficult_val_rows = read_jsonl(
        args.difficult_data.resolve() / "grids/val.jsonl"
    )
    additional_rows = read_jsonl(
        args.additional_data.resolve() / "grids/train.jsonl"
    )
    old_train_pairs = read_jsonl(
        args.old_data.resolve() / "inverse/train.jsonl"
    )
    old_val_pairs = read_jsonl(args.old_data.resolve() / "inverse/val.jsonl")
    difficult_train_pairs = derived_inverse_pairs(difficult_train_rows)
    difficult_val_pairs = derived_inverse_pairs(difficult_val_rows)
    additional_pairs = derived_inverse_pairs(additional_rows)
    train_rows = [
        *old_train_rows,
        *difficult_train_rows,
        *additional_rows,
    ]
    train_pairs = [
        *old_train_pairs,
        *difficult_train_pairs,
        *additional_pairs,
    ]
    loaded_bank = np.load(args.error_bank.resolve(), allow_pickle=False)
    error_bank = np.asarray(
        loaded_bank["normalized_errors"],
        dtype=np.float32,
    )
    error_conditions = np.asarray(
        loaded_bank["condition_indices"],
        dtype=np.int64,
    )
    torch, device = configure(int(args.seed), args.device)
    forward, _ = load_forward_direction_runtime_v7(
        args.forward_artifact.resolve(),
        torch,
        device,
    )
    blocks = {
        "train": (
            train_pairs,
            prepare_requests(
                train_rows,
                train_pairs,
                forward,
                error_bank,
                error_conditions,
                seed=int(args.seed) + 701,
                noisy_fraction=0.5,
            ),
        ),
        "iid_clean": (
            old_val_pairs,
            prepare_requests(
                old_val_rows,
                old_val_pairs,
                forward,
                error_bank,
                error_conditions,
                seed=int(args.seed) + 702,
                noisy_fraction=0.0,
            ),
        ),
        "iid_measurement_augmented": (
            old_val_pairs,
            prepare_requests(
                old_val_rows,
                old_val_pairs,
                forward,
                error_bank,
                error_conditions,
                seed=int(args.seed) + 702,
                noisy_fraction=1.0,
            ),
        ),
        "difficult_clean": (
            difficult_val_pairs,
            prepare_requests(
                difficult_val_rows,
                difficult_val_pairs,
                forward,
                error_bank,
                error_conditions,
                seed=int(args.seed) + 703,
                noisy_fraction=0.0,
            ),
        ),
        "difficult_measurement_augmented": (
            difficult_val_pairs,
            prepare_requests(
                difficult_val_rows,
                difficult_val_pairs,
                forward,
                error_bank,
                error_conditions,
                seed=int(args.seed) + 703,
                noisy_fraction=1.0,
            ),
        ),
    }
    reports = {}
    for name, (pairs, arrays) in blocks.items():
        path = output_dir / f"{name}.npz"
        reports[name] = save_block(path, arrays, pairs)
        print(json.dumps({name: reports[name]}, sort_keys=True), flush=True)
    metadata = {
        "version": "physics_structured_inverse_v9_cache",
        "complete": True,
        "seed": int(args.seed),
        "blocks": reports,
        "training_sources": {
            "old_groups": len(old_train_rows),
            "difficult_groups": len(difficult_train_rows),
            "additional_v5_groups": len(additional_rows),
            "total_groups": len(train_rows),
            "total_unique_pairs": len(train_pairs),
        },
        "forward_artifact": str(args.forward_artifact.resolve()),
        "forward_artifact_sha256": sha256(args.forward_artifact.resolve()),
        "measurement_error_bank": str(args.error_bank.resolve()),
        "measurement_error_bank_sha256": sha256(args.error_bank.resolve()),
        "targeted_partial_shards_used": False,
        "held_out_files_opened": [],
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
