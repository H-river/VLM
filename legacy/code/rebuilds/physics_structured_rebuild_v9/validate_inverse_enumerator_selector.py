#!/usr/bin/env python3
"""Validate a fixed two-enumerator selector on frozen inverse arrays."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.forward_runtime import (
    load_residual_forward_runtime_v9,
)
from physics_structured_rebuild_v9.grouped_forward_tree_runtime import (
    load_grouped_forward_tree_runtime_v9,
)
from physics_structured_rebuild_v9.full_basis_forward_surface import (
    load_full_basis_forward_surface_runtime_v9,
)
from physics_structured_rebuild_v9.inverse_lightgbm_runtime import (
    load_inverse_lightgbm_runtime_v9,
)
from physics_structured_rebuild_v9.inverse_runtime_v8 import (
    load_inverse_runtime_v8,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_EXTRA_TREES_FORWARD_STATE,
    DEFAULT_RESIDUAL_FORWARD_STATE,
    DEFAULT_TRANSFORMER_INVERSE_V8,
)
from specialist_rebuild_v2.common import SETUP_FIELDS, STATE_FIELDS

DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/tabm_transformer_inverse_v8"
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "inverse_enumerator_selector_protected_validation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--primary-forward",
        type=Path,
        default=DEFAULT_RESIDUAL_FORWARD_STATE,
    )
    parser.add_argument(
        "--secondary-forward",
        type=Path,
        default=DEFAULT_EXTRA_TREES_FORWARD_STATE,
    )
    parser.add_argument(
        "--secondary-kind",
        choices=("residual", "grouped_tree", "full_basis"),
        default="residual",
    )
    parser.add_argument(
        "--inverse-artifact",
        type=Path,
        default=DEFAULT_TRANSFORMER_INVERSE_V8,
    )
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=("iid_clean", "difficult_clean"),
    )
    parser.add_argument(
        "--selected-score-threshold",
        type=float,
        default=-0.09585162997245789,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache",
        type=Path,
        help="Optional NPZ output containing block selector features.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def rows_from_context(contexts: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for index, context in enumerate(contexts):
        setup = {
            field: float(context[position])
            for position, field in enumerate(SETUP_FIELDS)
        }
        current_values = np.asarray(context[12:17], dtype=np.float64).copy()
        current_values[-1] = math.expm1(float(current_values[-1]))
        current = {
            field: float(current_values[position])
            for position, field in enumerate(STATE_FIELDS)
        }
        rows.append(
            {
                "group_id": f"inverse_selector_validation_{index}",
                "setup": setup,
                "current_beam_state": current,
            }
        )
    return rows


def selected_values(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return matrix[np.arange(len(indices)), indices]


def margins(scores: np.ndarray) -> np.ndarray:
    ordered = np.partition(scores, kth=-2, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def metric(
    positives: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    success = positives[np.arange(len(selected)), selected]
    feasible = positives.any(axis=1)
    return {
        "count": int(len(success)),
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
    cache_output = (
        output.with_suffix(".npz")
        if args.cache is None
        else args.cache.resolve()
    )
    if cache_output.exists():
        raise RuntimeError(f"refusing to overwrite cache: {cache_output}")
    torch, device = configure(int(args.seed), args.device)
    primary, _ = load_residual_forward_runtime_v9(
        args.primary_forward.resolve(),
        torch,
        device,
    )
    secondary, _ = (
        load_full_basis_forward_surface_runtime_v9(
            args.secondary_forward.resolve(),
            torch,
            device,
        )
        if args.secondary_kind == "full_basis"
        else load_grouped_forward_tree_runtime_v9(
            args.secondary_forward.resolve(),
            torch,
            device,
        )
        if args.secondary_kind == "grouped_tree"
        else load_residual_forward_runtime_v9(
            args.secondary_forward.resolve(),
            torch,
            device,
        )
    )
    inverse_path = args.inverse_artifact.resolve()
    inverse, _ = (
        load_inverse_lightgbm_runtime_v9(inverse_path, torch, device)
        if inverse_path.suffix == ".pkl"
        else load_inverse_runtime_v8(inverse_path, torch, device)
    )

    block_metrics = {}
    cache_arrays: dict[str, np.ndarray] = {}
    for block in args.blocks:
        path = args.data_dir.resolve() / f"{block}.npz"
        arrays = np.load(path, allow_pickle=False)
        contexts = np.asarray(arrays["contexts"], dtype=np.float32)
        desired = np.asarray(arrays["desired"], dtype=np.float32)
        positives = np.asarray(arrays["positives"], dtype=np.bool_)
        rows = rows_from_context(contexts)
        primary_states = primary.predict_states(rows)
        secondary_states = secondary.predict_states(rows)
        primary_result = inverse.score_feature_arrays(
            contexts,
            desired,
            primary_states,
        )
        secondary_result = inverse.score_feature_arrays(
            contexts,
            desired,
            secondary_states,
        )
        primary_index = np.asarray(
            primary_result["selected_indices"],
            dtype=np.int64,
        )
        secondary_index = np.asarray(
            secondary_result["selected_indices"],
            dtype=np.int64,
        )
        primary_scores = np.asarray(
            primary_result["scores"],
            dtype=np.float32,
        )
        secondary_scores = np.asarray(
            secondary_result["scores"],
            dtype=np.float32,
        )
        primary_costs = np.asarray(
            primary_result["base_costs"],
            dtype=np.float32,
        )
        secondary_costs = np.asarray(
            secondary_result["base_costs"],
            dtype=np.float32,
        )
        diagnostic_features = {
            "selected_score_advantage": (
            selected_values(secondary_scores, secondary_index)
            - selected_values(primary_scores, primary_index)
            ),
            "selected_base_cost_advantage": (
                selected_values(primary_costs, primary_index)
                - selected_values(secondary_costs, secondary_index)
            ),
            "score_margin_advantage": (
                margins(secondary_scores) - margins(primary_scores)
            ),
        }
        score_advantage = diagnostic_features[
            "selected_score_advantage"
        ]
        choose_secondary = (
            score_advantage < float(args.selected_score_threshold)
        )
        selected = np.where(
            choose_secondary,
            secondary_index,
            primary_index,
        )
        block_metrics[str(block)] = {
            "primary": metric(positives, primary_index),
            "secondary": metric(positives, secondary_index),
            "selector": metric(positives, selected),
            "selector_secondary_count": int(choose_secondary.sum()),
            "action_disagreement_count": int(
                np.sum(primary_index != secondary_index)
            ),
        }
        for name, values in diagnostic_features.items():
            cache_arrays[f"{block}_feature_{name}"] = values
        cache_arrays[f"{block}_primary_success"] = positives[
            np.arange(len(primary_index)),
            primary_index,
        ]
        cache_arrays[f"{block}_secondary_success"] = positives[
            np.arange(len(secondary_index)),
            secondary_index,
        ]
        cache_arrays[f"{block}_feasible"] = positives.any(axis=1)
        print(
            json.dumps(
                {
                    "block": str(block),
                    **block_metrics[str(block)],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        "version": "inverse_enumerator_selector_protected_v9_one_seed",
        "rule": {
            "feature": "secondary_selected_score_minus_primary_selected_score",
            "secondary_if": "feature_less_than_threshold",
            "threshold": float(args.selected_score_threshold),
        },
        "blocks": block_metrics,
        "source_contract": {
            "data_dir": str(args.data_dir.resolve()),
            "primary_forward": str(args.primary_forward.resolve()),
            "secondary_forward": str(args.secondary_forward.resolve()),
            "secondary_forward_kind": str(args.secondary_kind),
            "inverse_artifact": str(args.inverse_artifact.resolve()),
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_output, **cache_arrays)
    summary["cache"] = str(cache_output)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
