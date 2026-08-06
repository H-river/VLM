#!/usr/bin/env python3
"""Compare two forward enumerators under the frozen v8 inverse ranker."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Qwen_orchestration.scripts.evaluate_end_to_end import (
    private_inverse_target_reached,
    read_jsonl,
)
from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.forward_runtime import (
    load_residual_forward_runtime_v9,
)
from physics_structured_rebuild_v9.inverse_runtime_v8 import (
    load_inverse_runtime_v8,
)
from physics_structured_rebuild_v9.inverse_lightgbm_runtime import (
    load_inverse_lightgbm_runtime_v9,
)
from physics_structured_rebuild_v9.grouped_forward_tree_runtime import (
    load_grouped_forward_tree_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_EXTRA_TREES_FORWARD_STATE,
    DEFAULT_RESIDUAL_FORWARD_STATE,
    DEFAULT_TRANSFORMER_INVERSE_V8,
)
from specialist_rebuild_v2.common import raw_state_array

DEFAULT_QWEN_DATA = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "inverse_enumerator_ensemble_diagnostic.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN_DATA)
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
        choices=("residual", "grouped_tree"),
        default="residual",
    )
    parser.add_argument(
        "--inverse-artifact",
        type=Path,
        default=DEFAULT_TRANSFORMER_INVERSE_V8,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache",
        type=Path,
        help="Optional NPZ output containing selector features and success masks.",
    )
    parser.add_argument(
        "--selector-artifact",
        type=Path,
        help=(
            "Optional frozen HGB selector. The artifact is evaluated only; "
            "the validation data are never used to fit or tune it."
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def selected_values(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return matrix[np.arange(len(indices)), indices]


def margins(scores: np.ndarray) -> np.ndarray:
    ordered = np.partition(scores, kth=-2, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def threshold_search(
    feature: np.ndarray,
    primary_success: np.ndarray,
    secondary_success: np.ndarray,
) -> dict[str, Any]:
    unique = np.unique(feature)
    thresholds = np.concatenate(
        [
            np.asarray([-np.inf], dtype=np.float64),
            (unique[:-1] + unique[1:]) / 2.0,
            np.asarray([np.inf], dtype=np.float64),
        ]
    )
    best = None
    for direction in ("secondary_if_greater", "secondary_if_less"):
        for threshold in thresholds:
            choose_secondary = (
                feature > threshold
                if direction == "secondary_if_greater"
                else feature < threshold
            )
            success = np.where(
                choose_secondary,
                secondary_success,
                primary_success,
            )
            candidate = {
                "direction": direction,
                "threshold": float(threshold),
                "secondary_count": int(choose_secondary.sum()),
                "success_count": int(success.sum()),
                "success_rate": float(success.mean()),
            }
            key = (
                candidate["success_count"],
                -candidate["secondary_count"],
            )
            if best is None or key > best[0]:
                best = (key, candidate)
    assert best is not None
    return best[1]


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
    qwen_data = args.qwen_data.resolve()
    canonical = [
        row
        for row in read_jsonl(qwen_data / "canonical/val.jsonl")
        if row["target_decision"].get("route_name")
        == "select_inverse_action_from_states_v1"
    ]
    if len(canonical) != 150:
        raise ValueError("expected 150 numerical-state inverse requests")
    private_by_group = {
        str(row["group_id"]): row
        for row in read_jsonl(qwen_data / "private/source_cases/val.jsonl")
    }
    rows = [
        {
            "group_id": str(row["example_id"]),
            "setup": row["target_decision"]["arguments"]["setup"],
            "current_beam_state": row["target_decision"]["arguments"][
                "current_beam_state"
            ],
        }
        for row in canonical
    ]
    setups = [row["target_decision"]["arguments"]["setup"] for row in canonical]
    current = np.stack(
        [
            raw_state_array(
                row["target_decision"]["arguments"]["current_beam_state"]
            )
            for row in canonical
        ]
    )
    desired = np.stack(
        [
            raw_state_array(
                row["target_decision"]["arguments"]["desired_beam_state"]
            )
            for row in canonical
        ]
    )

    torch, device = configure(int(args.seed), args.device)
    primary, _ = load_residual_forward_runtime_v9(
        args.primary_forward.resolve(),
        torch,
        device,
    )
    secondary, _ = (
        load_grouped_forward_tree_runtime_v9(
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
    if inverse_path.suffix == ".pkl":
        inverse, _ = load_inverse_lightgbm_runtime_v9(
            inverse_path,
            torch,
            device,
        )
    else:
        inverse, _ = load_inverse_runtime_v8(
            inverse_path,
            torch,
            device,
        )
    primary_states = primary.predict_states(rows)
    secondary_states = secondary.predict_states(rows)
    primary_result = inverse.score_requests(
        setups,
        current,
        desired,
        primary_states,
    )
    secondary_result = inverse.score_requests(
        setups,
        current,
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

    primary_success = []
    secondary_success = []
    for index, row in enumerate(canonical):
        private = private_by_group[str(row["group_id"])]
        primary_success.append(
            private_inverse_target_reached(
                private["setup"],
                primary_result["selected_actions"][index],
                private["desired_beam_state"],
            )
        )
        secondary_success.append(
            private_inverse_target_reached(
                private["setup"],
                secondary_result["selected_actions"][index],
                private["desired_beam_state"],
            )
        )
    primary_success_array = np.asarray(primary_success, dtype=np.bool_)
    secondary_success_array = np.asarray(secondary_success, dtype=np.bool_)
    disagreement = primary_index != secondary_index
    primary_scores = np.asarray(primary_result["scores"], dtype=np.float32)
    secondary_scores = np.asarray(secondary_result["scores"], dtype=np.float32)
    primary_costs = np.asarray(
        primary_result["base_costs"],
        dtype=np.float32,
    )
    secondary_costs = np.asarray(
        secondary_result["base_costs"],
        dtype=np.float32,
    )
    diagnostic_features = {
        "selected_base_cost_advantage": (
            selected_values(primary_costs, primary_index)
            - selected_values(secondary_costs, secondary_index)
        ),
        "selected_score_advantage": (
            selected_values(secondary_scores, secondary_index)
            - selected_values(primary_scores, primary_index)
        ),
        "score_margin_advantage": (
            margins(secondary_scores) - margins(primary_scores)
        ),
    }
    learned_selector = None
    selector_probability = None
    selector_choice = None
    if args.selector_artifact is not None:
        selector_path = args.selector_artifact.resolve()
        with selector_path.open("rb") as stream:
            selector = pickle.load(stream)
        if selector.get("model") != "grouped_forward_inverse_hgb_selector_v9":
            raise ValueError("unexpected learned inverse selector")
        expected_paths = {
            "primary_forward_artifact": args.primary_forward.resolve(),
            "secondary_forward_artifact": args.secondary_forward.resolve(),
            "inverse_artifact": args.inverse_artifact.resolve(),
        }
        for key, expected in expected_paths.items():
            if Path(str(selector[key])).resolve() != expected:
                raise ValueError(f"learned selector {key} differs")
        feature_names = [str(name) for name in selector["feature_names"]]
        selector_features = np.column_stack(
            [diagnostic_features[name] for name in feature_names]
        ).astype(np.float32)
        selector_probability = np.asarray(
            selector["classifier"].predict_proba(selector_features)[:, 1],
            dtype=np.float32,
        )
        selector_choice = selector_probability >= float(selector["threshold"])
        selector_success = np.where(
            selector_choice,
            secondary_success_array,
            primary_success_array,
        )
        learned_selector = {
            "artifact": str(selector_path),
            "threshold": float(selector["threshold"]),
            "feature_names": feature_names,
            "secondary_count": int(selector_choice.sum()),
            "success_count": int(selector_success.sum()),
            "success_rate": float(selector_success.mean()),
        }
    rules = {
        name: threshold_search(
            values,
            primary_success_array,
            secondary_success_array,
        )
        for name, values in diagnostic_features.items()
    }
    summary = {
        "version": "inverse_enumerator_ensemble_diagnostic_v9_one_seed",
        "count": len(canonical),
        "primary": {
            "success_count": int(primary_success_array.sum()),
            "success_rate": float(primary_success_array.mean()),
        },
        "secondary": {
            "success_count": int(secondary_success_array.sum()),
            "success_rate": float(secondary_success_array.mean()),
        },
        "disagreement": {
            "action_count": int(disagreement.sum()),
            "both_succeed": int(
                np.sum(primary_success_array & secondary_success_array)
            ),
            "primary_only_succeeds": int(
                np.sum(primary_success_array & ~secondary_success_array)
            ),
            "secondary_only_succeeds": int(
                np.sum(~primary_success_array & secondary_success_array)
            ),
            "neither_succeeds": int(
                np.sum(~primary_success_array & ~secondary_success_array)
            ),
            "oracle_union_success_count": int(
                np.sum(primary_success_array | secondary_success_array)
            ),
            "oracle_union_success_rate": float(
                np.mean(primary_success_array | secondary_success_array)
            ),
        },
        "single_feature_selector_search": rules,
        "learned_selector": learned_selector,
        "source_contract": {
            "qwen_validation": str(
                (qwen_data / "canonical/val.jsonl").resolve()
            ),
            "primary_forward": str(args.primary_forward.resolve()),
            "secondary_forward": str(args.secondary_forward.resolve()),
            "inverse_artifact": str(args.inverse_artifact.resolve()),
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_output,
        primary_success=primary_success_array,
        secondary_success=secondary_success_array,
        primary_index=primary_index,
        secondary_index=secondary_index,
        **{
            f"feature_{name}": values
            for name, values in diagnostic_features.items()
        },
        **(
            {
                "learned_selector_probability": selector_probability,
                "learned_selector_secondary": selector_choice,
            }
            if selector_probability is not None
            else {}
        ),
    )
    summary["cache"] = str(cache_output)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
