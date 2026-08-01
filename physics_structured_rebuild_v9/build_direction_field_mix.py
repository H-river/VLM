#!/usr/bin/env python3
"""Build a lightweight field-selected direction candidate and system cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from specialist_rebuild_v2.common import DIRECTION_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-tree", type=Path, required=True)
    parser.add_argument("--secondary-tree", type=Path, required=True)
    parser.add_argument("--primary-cache", type=Path, required=True)
    parser.add_argument("--secondary-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--secondary-fields",
        nargs="+",
        default=[
            "centroid_x",
            "centroid_y",
            "width_x",
            "width_y",
        ],
    )
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"artifact is not a mapping: {path}")
    return value


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    tree_path = output_dir / "direction_field_mix_v9.pkl"
    cache_path = output_dir / "direction_validation_cache.npz"
    summary_path = output_dir / "direction_field_mix_summary.json"
    if any(path.exists() for path in (tree_path, cache_path, summary_path)):
        raise RuntimeError(f"refusing to overwrite output: {output_dir}")

    fields = list(DIRECTION_FIELDS)
    secondary_fields = set(args.secondary_fields)
    unknown = secondary_fields.difference(fields)
    if unknown:
        raise ValueError(f"unknown direction fields: {sorted(unknown)}")

    primary_path = args.primary_tree.resolve()
    secondary_path = args.secondary_tree.resolve()
    primary = load_pickle(primary_path)
    secondary = load_pickle(secondary_path)
    if primary.get("feature_mode") != secondary.get("feature_mode"):
        raise ValueError("source tree feature modes differ")
    if int(primary.get("input_dim", -1)) != int(
        secondary.get("input_dim", -2)
    ):
        raise ValueError("source tree input dimensions differ")
    if list(primary.get("direction_fields", [])) != fields:
        raise ValueError("primary direction fields differ")
    if list(secondary.get("direction_fields", [])) != fields:
        raise ValueError("secondary direction fields differ")

    primary_models = list(primary["models"])
    secondary_models = list(secondary["models"])
    selected_models = [
        (
            secondary_models[index]
            if field in secondary_fields
            else primary_models[index]
        )
        for index, field in enumerate(fields)
    ]

    primary_cache_path = args.primary_cache.resolve()
    secondary_cache_path = args.secondary_cache.resolve()
    primary_cache = np.load(primary_cache_path, allow_pickle=False)
    secondary_cache = np.load(secondary_cache_path, allow_pickle=False)
    for key in ("changes", "truth", "routes", "example_ids"):
        if not np.array_equal(primary_cache[key], secondary_cache[key]):
            raise ValueError(f"source system caches differ for {key}")
    probabilities = np.asarray(
        primary_cache["tree_probabilities"],
        dtype=np.float32,
    ).copy()
    secondary_probability = np.asarray(
        secondary_cache["tree_probabilities"],
        dtype=np.float32,
    )
    for index, field in enumerate(fields):
        if field in secondary_fields:
            probabilities[:, index] = secondary_probability[:, index]

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "version": "direction_field_mix_v9_one_seed",
        "model": "balanced_field_selected_direction_tree_v9",
        "seed": int(args.seed),
        "input_dim": int(primary["input_dim"]),
        "direction_fields": fields,
        "classes": list(primary["classes"]),
        "models": selected_models,
        "feature_mode": str(primary["feature_mode"]),
        "forward_artifact": str(primary["forward_artifact"]),
        "forward_artifact_sha256": str(
            primary["forward_artifact_sha256"]
        ),
        "residual_forward_tree_artifact": primary.get(
            "residual_forward_tree_artifact"
        ),
        "residual_forward_tree_artifact_sha256": primary.get(
            "residual_forward_tree_artifact_sha256"
        ),
        "field_sources": {
            field: (
                "secondary" if field in secondary_fields else "primary"
            )
            for field in fields
        },
        "primary_tree": str(primary_path),
        "primary_tree_sha256": sha256(primary_path),
        "secondary_tree": str(secondary_path),
        "secondary_tree_sha256": sha256(secondary_path),
        "held_out_test_used": False,
    }
    with tree_path.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(
        cache_path,
        changes=np.asarray(primary_cache["changes"], dtype=np.float32),
        tree_probabilities=probabilities,
        truth=np.asarray(primary_cache["truth"], dtype=np.int64),
        routes=np.asarray(primary_cache["routes"], dtype=np.str_),
        example_ids=np.asarray(primary_cache["example_ids"], dtype=np.str_),
    )
    summary = {
        "version": artifact["version"],
        "artifact": str(tree_path),
        "artifact_sha256": sha256(tree_path),
        "system_cache": str(cache_path),
        "system_cache_sha256": sha256(cache_path),
        "field_sources": artifact["field_sources"],
        "source_contract": {
            "primary_tree": str(primary_path),
            "secondary_tree": str(secondary_path),
            "primary_cache": str(primary_cache_path),
            "secondary_cache": str(secondary_cache_path),
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
