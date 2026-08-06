#!/usr/bin/env python3
"""Build training diagnostics for natural versus grouped inverse enumerators."""

from __future__ import annotations

import argparse
import hashlib
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
from physics_structured_rebuild_v9.inverse_runtime_v8 import (
    load_inverse_runtime_v8,
)
from physics_structured_rebuild_v9.inverse_selector_runtime import (
    margins,
    selected_values,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_COMBINED_NATURAL_INVERSE_V9,
    DEFAULT_NATURAL_GRID_FORWARD_STATE,
)
from specialist_rebuild_v2.common import SETUP_FIELDS, STATE_FIELDS

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_DATA = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9"
    / "combined_natural_inverse_adaptation/train.npz"
)
DEFAULT_GROUPED = DEFAULT_RUN / "grouped_forward_tree_protected_v9.pkl"
DEFAULT_OUTPUT = DEFAULT_RUN / "grouped_inverse_selector_cache_v9.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--primary-forward",
        type=Path,
        default=DEFAULT_NATURAL_GRID_FORWARD_STATE,
    )
    parser.add_argument("--grouped-forward", type=Path, default=DEFAULT_GROUPED)
    parser.add_argument(
        "--secondary-kind",
        choices=("grouped_tree", "full_basis"),
        default="grouped_tree",
    )
    parser.add_argument(
        "--inverse-artifact",
        type=Path,
        default=DEFAULT_COMBINED_NATURAL_INVERSE_V9,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_from_contexts(
    contexts: np.ndarray,
    group_ids: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for index, context in enumerate(contexts):
        setup = {
            field: float(context[position])
            for position, field in enumerate(SETUP_FIELDS)
        }
        current_values = np.asarray(context[12:17], dtype=np.float64).copy()
        current_values[-1] = math.expm1(float(current_values[-1]))
        rows.append(
            {
                "group_id": str(group_ids[index]),
                "setup": setup,
                "current_beam_state": {
                    field: float(current_values[position])
                    for position, field in enumerate(STATE_FIELDS)
                },
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    with np.load(args.data.resolve(), allow_pickle=False) as data:
        group_ids = np.asarray(data["group_ids"], dtype=np.str_)
        contexts = np.asarray(data["contexts"], dtype=np.float32)
        desired = np.asarray(data["desired"], dtype=np.float32)
        positives = np.asarray(data["positives"], dtype=np.bool_)
    count = len(group_ids)
    features = np.empty((count, 3), dtype=np.float32)
    primary_success = np.empty(count, dtype=np.bool_)
    grouped_success = np.empty(count, dtype=np.bool_)
    primary_index = np.empty(count, dtype=np.int16)
    grouped_index = np.empty(count, dtype=np.int16)
    torch, device = configure(int(args.seed), args.device)
    primary, _ = load_residual_forward_runtime_v9(
        args.primary_forward.resolve(),
        torch,
        device,
    )
    grouped, _ = (
        load_full_basis_forward_surface_runtime_v9(
            args.grouped_forward.resolve(),
            torch,
            device,
        )
        if args.secondary_kind == "full_basis"
        else load_grouped_forward_tree_runtime_v9(
            args.grouped_forward.resolve(),
            torch,
            device,
        )
    )
    inverse, _ = load_inverse_runtime_v8(
        args.inverse_artifact.resolve(),
        torch,
        device,
    )
    for start in range(0, count, int(args.chunk_size)):
        stop = min(start + int(args.chunk_size), count)
        rows = rows_from_contexts(
            contexts[start:stop],
            group_ids[start:stop],
        )
        primary_states = primary.predict_states(rows)
        grouped_states = grouped.predict_states(rows)
        primary_result = inverse.score_feature_arrays(
            contexts[start:stop],
            desired[start:stop],
            primary_states,
        )
        grouped_result = inverse.score_feature_arrays(
            contexts[start:stop],
            desired[start:stop],
            grouped_states,
        )
        p_index = np.asarray(
            primary_result["selected_indices"],
            dtype=np.int64,
        )
        g_index = np.asarray(
            grouped_result["selected_indices"],
            dtype=np.int64,
        )
        p_scores = np.asarray(primary_result["scores"], dtype=np.float32)
        g_scores = np.asarray(grouped_result["scores"], dtype=np.float32)
        p_costs = np.asarray(primary_result["base_costs"], dtype=np.float32)
        g_costs = np.asarray(grouped_result["base_costs"], dtype=np.float32)
        features[start:stop, 0] = (
            selected_values(g_scores, g_index)
            - selected_values(p_scores, p_index)
        )
        features[start:stop, 1] = (
            selected_values(p_costs, p_index)
            - selected_values(g_costs, g_index)
        )
        features[start:stop, 2] = margins(g_scores) - margins(p_scores)
        local = np.arange(stop - start)
        primary_success[start:stop] = positives[start:stop][local, p_index]
        grouped_success[start:stop] = positives[start:stop][local, g_index]
        primary_index[start:stop] = p_index
        grouped_index[start:stop] = g_index
        print(
            json.dumps(
                {
                    "processed": stop,
                    "count": count,
                    "primary_success": int(primary_success[:stop].sum()),
                    "grouped_success": int(grouped_success[:stop].sum()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_names = np.asarray(
        [
            "selected_score_advantage",
            "selected_base_cost_advantage",
            "score_margin_advantage",
        ],
        dtype=np.str_,
    )
    np.savez_compressed(
        output,
        group_ids=group_ids,
        features=features,
        feature_names=feature_names,
        primary_success=primary_success,
        grouped_success=grouped_success,
        primary_index=primary_index,
        grouped_index=grouped_index,
    )
    report = {
        "version": "grouped_inverse_selector_cache_v9_one_seed",
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "count": int(count),
        "primary_success_count": int(primary_success.sum()),
        "grouped_success_count": int(grouped_success.sum()),
        "oracle_union_success_count": int(
            (primary_success | grouped_success).sum()
        ),
        "primary_only_success_count": int(
            (primary_success & ~grouped_success).sum()
        ),
        "grouped_only_success_count": int(
            (~primary_success & grouped_success).sum()
        ),
        "action_disagreement_count": int(
            (primary_index != grouped_index).sum()
        ),
        "source_contract": {
            "training_data": str(args.data.resolve()),
            "secondary_forward_kind": str(args.secondary_kind),
            "system_validation_used": False,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
