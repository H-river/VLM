#!/usr/bin/env python3
"""Evaluate a frozen learned inverse selector on unopened validation caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_RUN / "inverse_selector_leakage_free_hgb_v9.pkl",
    )
    parser.add_argument(
        "--system-cache",
        type=Path,
        default=DEFAULT_RUN / "inverse_enumerator_ensemble_search.npz",
    )
    parser.add_argument(
        "--protected-cache",
        type=Path,
        default=DEFAULT_RUN / "inverse_selector_multifeature_inputs.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_RUN
            / "inverse_selector_leakage_free_hgb_v9_validation.json"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_arrays(
    arrays: Any,
    prefix: str,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    separator = "" if not prefix else f"{prefix}_"
    features = np.column_stack(
        [
            np.asarray(
                arrays[f"{separator}feature_{name}"],
                dtype=np.float32,
            )
            for name in feature_names
        ]
    )
    primary = np.asarray(
        arrays[f"{separator}primary_success"],
        dtype=np.bool_,
    )
    secondary = np.asarray(
        arrays[f"{separator}secondary_success"],
        dtype=np.bool_,
    )
    return features, primary, secondary


def metrics(
    classifier: Any,
    features: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
) -> dict[str, Any]:
    choose_secondary = (
        np.asarray(classifier.predict(features)).astype(np.int64) == 1
    )
    selected = np.where(choose_secondary, secondary, primary)
    return {
        "count": int(len(selected)),
        "primary_success_count": int(primary.sum()),
        "secondary_success_count": int(secondary.sum()),
        "oracle_union_success_count": int((primary | secondary).sum()),
        "selected_success_count": int(selected.sum()),
        "selected_success_rate": float(selected.mean()),
        "secondary_selected_count": int(choose_secondary.sum()),
        "exclusive_selector_accuracy": float(
            np.mean(
                choose_secondary[primary ^ secondary]
                == secondary[primary ^ secondary]
            )
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    artifact_path = args.artifact.resolve()
    with artifact_path.open("rb") as stream:
        artifact = pickle.load(stream)
    if artifact.get("model") != "v8_ranker_dual_forward_hgb_selector_v9":
        raise ValueError("unexpected inverse selector artifact")
    feature_names = [str(name) for name in artifact["feature_names"]]
    classifier = artifact["classifier"]
    system_path = args.system_cache.resolve()
    protected_path = args.protected_cache.resolve()
    system = np.load(system_path, allow_pickle=False)
    protected = np.load(protected_path, allow_pickle=False)
    blocks = {
        "system_state_inverse": block_arrays(system, "", feature_names),
        "iid_clean": block_arrays(protected, "iid_clean", feature_names),
        "difficult_clean": block_arrays(
            protected,
            "difficult_clean",
            feature_names,
        ),
    }
    result = {
        "version": "inverse_selector_leakage_free_validation_v9_one_seed",
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "feature_names": feature_names,
        "blocks": {
            name: metrics(classifier, *values)
            for name, values in blocks.items()
        },
        "source_contract": {
            "system_cache": str(system_path),
            "system_cache_sha256": sha256(system_path),
            "protected_cache": str(protected_path),
            "protected_cache_sha256": sha256(protected_path),
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
