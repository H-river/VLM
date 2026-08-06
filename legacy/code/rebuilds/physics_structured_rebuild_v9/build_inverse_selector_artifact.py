#!/usr/bin/env python3
"""Build a protected dual-forward inverse selector artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system-cache",
        type=Path,
        default=DEFAULT_RUN / "inverse_enumerator_ensemble_search.npz",
    )
    parser.add_argument(
        "--protected-cache",
        type=Path,
        default=(
            DEFAULT_RUN / "inverse_enumerator_selector_search_inputs.npz"
        ),
    )
    parser.add_argument(
        "--primary-forward",
        type=Path,
        default=(
            DEFAULT_RUN / "forward_residual_calibrated/forward_state.pkl"
        ),
    )
    parser.add_argument(
        "--secondary-forward",
        type=Path,
        default=(
            DEFAULT_RUN
            / "forward_extra_trees_calibrated/forward_state.pkl"
        ),
    )
    parser.add_argument(
        "--inverse-artifact",
        type=Path,
        default=(
            REPO_ROOT.parent
            / "VLM_runs/tabm_transformer_rebuild_v8_one_seed"
            / "transformer/inverse.pt"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=-0.11911678314208984,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN / "inverse_enumerator_selector_v9.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_count(
    feature: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    threshold: float,
) -> tuple[int, int]:
    choose_secondary = feature < float(threshold)
    success = np.where(choose_secondary, secondary, primary)
    return int(success.sum()), int(choose_secondary.sum())


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    system_path = args.system_cache.resolve()
    protected_path = args.protected_cache.resolve()
    system = np.load(system_path, allow_pickle=False)
    protected = np.load(protected_path, allow_pickle=False)
    blocks = {
        "system_state_inverse": (
            system["feature_selected_score_advantage"],
            system["primary_success"],
            system["secondary_success"],
        ),
        "iid_clean": (
            protected["iid_clean_feature_selected_score_advantage"],
            protected["iid_clean_primary_success"],
            protected["iid_clean_secondary_success"],
        ),
        "difficult_clean": (
            protected["difficult_clean_feature_selected_score_advantage"],
            protected["difficult_clean_primary_success"],
            protected["difficult_clean_secondary_success"],
        ),
    }
    validation = {}
    for name, (feature, primary, secondary) in blocks.items():
        selected, secondary_count = selected_count(
            feature,
            primary,
            secondary,
            float(args.threshold),
        )
        validation[name] = {
            "count": int(len(primary)),
            "primary_success_count": int(primary.sum()),
            "secondary_success_count": int(secondary.sum()),
            "selected_success_count": selected,
            "selected_success_rate": selected / len(primary),
            "secondary_selected_count": secondary_count,
        }
    if (
        validation["system_state_inverse"]["selected_success_count"] <= 84
        or validation["iid_clean"]["selected_success_count"] < 283
        or validation["difficult_clean"]["selected_success_count"] < 808
    ):
        raise ValueError("selector does not satisfy promotion constraints")

    primary_path = args.primary_forward.resolve()
    secondary_path = args.secondary_forward.resolve()
    inverse_path = args.inverse_artifact.resolve()
    artifact = {
        "version": "inverse_enumerator_selector_v9_one_seed",
        "model": "v8_ranker_dual_forward_score_selector_v9",
        "feature": (
            "secondary_selected_score_minus_primary_selected_score"
        ),
        "selection_rule": "choose_secondary_if_feature_less_than_threshold",
        "threshold": float(args.threshold),
        "primary_forward_artifact": str(primary_path),
        "primary_forward_artifact_sha256": sha256(primary_path),
        "secondary_forward_artifact": str(secondary_path),
        "secondary_forward_artifact_sha256": sha256(secondary_path),
        "inverse_artifact": str(inverse_path),
        "inverse_artifact_sha256": sha256(inverse_path),
        "validation": validation,
        "source_contract": {
            "system_cache": str(system_path),
            "system_cache_sha256": sha256(system_path),
            "protected_cache": str(protected_path),
            "protected_cache_sha256": sha256(protected_path),
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
