#!/usr/bin/env python3
"""Build leakage-free training features for the dual-forward inverse selector."""

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
from physics_structured_rebuild_v9.inverse_runtime_v8 import (
    load_inverse_runtime_v8,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_EXTRA_TREES_FORWARD_STATE,
    DEFAULT_RESIDUAL_FORWARD_STATE,
    DEFAULT_TRANSFORMER_INVERSE_V8,
)
from specialist_rebuild_v2.common import SETUP_FIELDS, STATE_FIELDS

DEFAULT_DATA = (
    REPO_ROOT.parent / "VLM_data/tabm_transformer_inverse_v8/train.npz"
)
DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
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
        "--inverse-artifact",
        type=Path,
        default=DEFAULT_TRANSFORMER_INVERSE_V8,
    )
    parser.add_argument("--sample-count", type=int, default=12_000)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN / "inverse_selector_training_features.npz",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_RUN / "inverse_selector_training_features.json",
    )
    return parser.parse_args()


def rows_from_context(contexts: np.ndarray, indices: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for local_index, context in enumerate(contexts):
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
                "group_id": f"inverse_selector_train_{int(indices[local_index])}",
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


def stratified_indices(
    statuses: np.ndarray,
    noisy: np.ndarray,
    feasible: np.ndarray,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count <= 0 or sample_count % 4:
        raise ValueError("sample-count must be positive and divisible by four")
    rng = np.random.default_rng(seed)
    per_stratum = sample_count // 4
    selected_parts = []
    stratum_parts = []
    for noise_value in (False, True):
        for status_value in (0, 1):
            candidates = np.flatnonzero(
                feasible
                & (noisy == noise_value)
                & (statuses == status_value)
            )
            if len(candidates) < per_stratum:
                raise ValueError("not enough rows in a requested stratum")
            chosen = rng.choice(candidates, size=per_stratum, replace=False)
            selected_parts.append(chosen)
            stratum_parts.append(
                np.full(
                    per_stratum,
                    int(noise_value) * 2 + int(status_value),
                    dtype=np.int8,
                )
            )
    selected = np.concatenate(selected_parts)
    strata = np.concatenate(stratum_parts)
    order = rng.permutation(len(selected))
    return selected[order], strata[order]


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    summary_path = args.summary.resolve()
    for path in (output, summary_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite output: {path}")

    arrays = np.load(args.data.resolve(), allow_pickle=False)
    contexts_all = np.asarray(arrays["contexts"], dtype=np.float32)
    desired_all = np.asarray(arrays["desired"], dtype=np.float32)
    positives_all = np.asarray(arrays["positives"], dtype=np.bool_)
    statuses_all = np.asarray(arrays["statuses"], dtype=np.int8)
    noisy_all = np.asarray(arrays["noisy"], dtype=np.bool_)
    feasible_all = positives_all.any(axis=1)
    indices, strata = stratified_indices(
        statuses_all,
        noisy_all,
        feasible_all,
        int(args.sample_count),
        int(args.seed),
    )
    contexts = contexts_all[indices]
    desired = desired_all[indices]
    positives = positives_all[indices]

    feature_names = (
        "selected_score_advantage",
        "selected_base_cost_advantage",
        "score_margin_advantage",
    )
    features = np.empty((len(indices), len(feature_names)), dtype=np.float32)
    primary_success = np.empty(len(indices), dtype=np.bool_)
    secondary_success = np.empty(len(indices), dtype=np.bool_)
    primary_index = np.empty(len(indices), dtype=np.int16)
    secondary_index = np.empty(len(indices), dtype=np.int16)

    torch, device = configure(int(args.seed), args.device)
    primary, _ = load_residual_forward_runtime_v9(
        args.primary_forward.resolve(),
        torch,
        device,
    )
    secondary, _ = load_residual_forward_runtime_v9(
        args.secondary_forward.resolve(),
        torch,
        device,
    )
    inverse, _ = load_inverse_runtime_v8(
        args.inverse_artifact.resolve(),
        torch,
        device,
    )

    for start in range(0, len(indices), int(args.chunk_size)):
        stop = min(start + int(args.chunk_size), len(indices))
        rows = rows_from_context(contexts[start:stop], indices[start:stop])
        primary_states = primary.predict_states(rows)
        secondary_states = secondary.predict_states(rows)
        primary_result = inverse.score_feature_arrays(
            contexts[start:stop],
            desired[start:stop],
            primary_states,
        )
        secondary_result = inverse.score_feature_arrays(
            contexts[start:stop],
            desired[start:stop],
            secondary_states,
        )
        p_index = np.asarray(primary_result["selected_indices"], dtype=np.int64)
        s_index = np.asarray(secondary_result["selected_indices"], dtype=np.int64)
        p_scores = np.asarray(primary_result["scores"], dtype=np.float32)
        s_scores = np.asarray(secondary_result["scores"], dtype=np.float32)
        p_costs = np.asarray(primary_result["base_costs"], dtype=np.float32)
        s_costs = np.asarray(secondary_result["base_costs"], dtype=np.float32)
        features[start:stop, 0] = (
            selected_values(s_scores, s_index)
            - selected_values(p_scores, p_index)
        )
        features[start:stop, 1] = (
            selected_values(p_costs, p_index)
            - selected_values(s_costs, s_index)
        )
        features[start:stop, 2] = margins(s_scores) - margins(p_scores)
        local = np.arange(stop - start)
        primary_success[start:stop] = positives[start:stop][local, p_index]
        secondary_success[start:stop] = positives[start:stop][local, s_index]
        primary_index[start:stop] = p_index
        secondary_index[start:stop] = s_index
        print(
            json.dumps(
                {
                    "processed": stop,
                    "count": len(indices),
                    "primary_success": int(primary_success[:stop].sum()),
                    "secondary_success": int(secondary_success[:stop].sum()),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=features,
        feature_names=np.asarray(feature_names),
        primary_success=primary_success,
        secondary_success=secondary_success,
        primary_index=primary_index,
        secondary_index=secondary_index,
        source_indices=indices,
        strata=strata,
        noisy=noisy_all[indices],
        statuses=statuses_all[indices],
    )
    exclusive = primary_success ^ secondary_success
    summary = {
        "version": "inverse_selector_training_features_v9_one_seed",
        "sample_count": len(indices),
        "seed": int(args.seed),
        "sampling": (
            "equal feasible rows from clean/already, clean/change, "
            "noisy/already, and noisy/change training strata"
        ),
        "feature_names": list(feature_names),
        "primary_success_count": int(primary_success.sum()),
        "secondary_success_count": int(secondary_success.sum()),
        "exclusive_success_count": int(exclusive.sum()),
        "primary_only_success_count": int(
            np.sum(primary_success & ~secondary_success)
        ),
        "secondary_only_success_count": int(
            np.sum(~primary_success & secondary_success)
        ),
        "action_disagreement_count": int(
            np.sum(primary_index != secondary_index)
        ),
        "source_contract": {
            "training_data": str(args.data.resolve()),
            "protected_validation_files_opened": [],
            "system_validation_files_opened": [],
        },
        "cache": str(output),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
