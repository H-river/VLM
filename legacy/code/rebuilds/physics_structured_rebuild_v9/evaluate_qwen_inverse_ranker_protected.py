#!/usr/bin/env python3
"""Compare base and natural-adapted inverse rankers on protected clean blocks."""

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

from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.inverse_runtime_v8 import (
    load_inverse_runtime_v8,
)
from physics_structured_rebuild_v9.inverse_lightgbm_runtime import (
    load_inverse_lightgbm_runtime_v9,
)
from physics_structured_rebuild_v9.inverse_direct_policy_runtime import (
    load_direct_inverse_policy_runtime_v9,
)
from physics_structured_rebuild_v9.inverse_success_runtime import (
    load_inverse_success_ranker_runtime_v9,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/tabm_transformer_inverse_v8"
DEFAULT_BASE = (
    REPO_ROOT.parent
    / "VLM_runs/tabm_transformer_rebuild_v8_one_seed"
    / "transformer/inverse.pt"
)
DEFAULT_ADAPTED = DEFAULT_RUN / "qwen_inverse_ranker_adaptation/inverse.pt"
DEFAULT_OUTPUT = (
    DEFAULT_RUN / "qwen_inverse_ranker_protected_validation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-artifact", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapted-artifact", type=Path, default=DEFAULT_ADAPTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=("iid_clean", "difficult_clean"),
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(
    positives: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    index = np.arange(len(selected))
    success = positives[index, selected]
    feasible = positives.any(axis=1)
    return {
        "count": len(selected),
        "feasible_count": int(feasible.sum()),
        "success_all_count": int(success.sum()),
        "success_all": float(success.mean()),
        "success_feasible_count": int(success[feasible].sum()),
        "success_feasible": float(success[feasible].mean()),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    torch, device = configure(int(args.seed), args.device)
    base_path = args.base_artifact.resolve()
    adapted_path = args.adapted_artifact.resolve()
    base, _ = load_inverse_runtime_v8(base_path, torch, device)
    if adapted_path.suffix == ".pkl":
        with adapted_path.open("rb") as stream:
            adapted_metadata = pickle.load(stream)
        adapted, _ = (
            load_inverse_success_ranker_runtime_v9(
                adapted_path,
                torch,
                device,
            )
            if adapted_metadata.get("model")
            == "group_balanced_inverse_success_v9"
            else load_inverse_lightgbm_runtime_v9(
                adapted_path,
                torch,
                device,
            )
        )
    else:
        adapted_metadata = torch.load(
            adapted_path,
            map_location="cpu",
            weights_only=False,
        )
        adapted, _ = (
            load_direct_inverse_policy_runtime_v9(
                adapted_path,
                torch,
                device,
            )
            if adapted_metadata.get("model") == "direct_inverse_policy_v9"
            else load_inverse_runtime_v8(adapted_path, torch, device)
        )
    block_metrics = {}
    for block_name in args.blocks:
        path = args.data_dir.resolve() / f"{block_name}.npz"
        arrays = np.load(path, allow_pickle=False)
        inputs = (
            np.asarray(arrays["contexts"], dtype=np.float32),
            np.asarray(arrays["desired"], dtype=np.float32),
            np.asarray(arrays["candidate_states"], dtype=np.float32),
        )
        positives = np.asarray(arrays["positives"], dtype=np.bool_)
        base_result = base.score_feature_arrays(*inputs)
        adapted_result = adapted.score_feature_arrays(*inputs)
        block_metrics[str(block_name)] = {
            "base": metrics(
                positives,
                np.asarray(base_result["selected_indices"], dtype=np.int64),
            ),
            "adapted": metrics(
                positives,
                np.asarray(
                    adapted_result["selected_indices"],
                    dtype=np.int64,
                ),
            ),
        }
        print(
            json.dumps(
                {
                    "block": str(block_name),
                    **block_metrics[str(block_name)],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    promotion_passed = all(
        values["adapted"]["success_all_count"]
        >= values["base"]["success_all_count"]
        for values in block_metrics.values()
    )
    report = {
        "version": "qwen_inverse_ranker_protected_validation_v9_one_seed",
        "blocks": block_metrics,
        "artifacts": {
            "base": {"path": str(base_path), "sha256": sha256(base_path)},
            "adapted": {
                "path": str(adapted_path),
                "sha256": sha256(adapted_path),
            },
        },
        "source_contract": {
            "protected_data_dir": str(args.data_dir.resolve()),
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
        "promotion_passed": bool(promotion_passed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
