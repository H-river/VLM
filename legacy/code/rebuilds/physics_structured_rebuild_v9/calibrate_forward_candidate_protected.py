#!/usr/bin/env python3
"""Calibrate five forward blends using only protected specialist validation."""

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

from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import load_grid_arrays
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from physics_structured_rebuild_v9.calibrate_forward_system import (
    BLENDS,
    tree_residuals,
)
from physics_structured_rebuild_v9.calibrate_system_direction import sha256
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
)
from specialist_rebuild_v2.common import ACTION_FIELDS, STATE_FIELDS, fixed_action_grid

DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        type=Path,
        default=DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
    )
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--tree-artifact", type=Path, required=True)
    parser.add_argument(
        "--old-validation",
        type=Path,
        default=REPO_ROOT.parent
        / "VLM_data/specialist_rebuild_v2/grids/val.jsonl",
    )
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=REPO_ROOT.parent
        / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--split-confirmation",
        action="store_true",
        help=(
            "Choose blends on even-numbered protected groups and reserve "
            "odd-numbered groups for untouched confirmation."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def high_mask(group_count: int) -> np.ndarray:
    return np.tile(
        np.asarray(
            [
                sum(
                    abs(float(action[field])) > 0.0
                    for field in ACTION_FIELDS
                )
                >= 3
                for action in fixed_action_grid()
            ],
            dtype=np.bool_,
        ),
        group_count,
    )


def exact_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    high: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[int, int]:
    if mask is None:
        mask = np.ones(len(prediction), dtype=np.bool_)
    exact = np.all(np.abs(prediction - target) <= 1.0, axis=1)
    return int(exact[mask].sum()), int(exact[mask & high].sum())


def prediction(
    prior: np.ndarray,
    residual: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            prior[:, field]
            + float(BLENDS[int(selected[field])]) * residual[:, field]
            for field in range(len(STATE_FIELDS))
        ],
        axis=1,
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")

    torch, device = configure(int(args.seed), args.device)
    current_path = args.current.resolve()
    current, _ = load_forward_selector_ensemble_runtime_v9(
        current_path,
        torch,
        device,
    )
    forward_path = args.forward_artifact.resolve()
    base, _ = load_forward_direction_runtime_v7(
        forward_path,
        torch,
        device,
    )
    tree_path = args.tree_artifact.resolve()
    with tree_path.open("rb") as stream:
        tree = pickle.load(stream)
    if tree.get("model") not in {
        "v7_stacked_five_head_hist_gradient_boosting_forward_v9",
        "v7_stacked_five_head_extra_trees_forward_v9",
        "v7_stacked_five_head_lightgbm_forward_v9",
    }:
        raise ValueError("unexpected residual-forward tree artifact")
    models = list(tree["models"])

    blocks = {}
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        rows = read_jsonl(path)
        arrays = load_grid_arrays(path, include_legacy_features=False)
        prior = base.predict_changes(rows).reshape(-1, 5)
        residual = tree_residuals(models, arrays.features, prior)
        current_prediction = current.predict_changes(rows).reshape(-1, 5)
        high = high_mask(arrays.group_count)
        group_index = np.repeat(
            np.arange(arrays.group_count, dtype=np.int64),
            len(current_prediction) // arrays.group_count,
        )
        if len(group_index) != len(current_prediction):
            raise ValueError("protected forward group shape differs")
        masks = {
            "full": np.ones(len(current_prediction), dtype=np.bool_),
            "calibration": (
                group_index % 2 == 0
                if args.split_confirmation
                else np.ones(len(current_prediction), dtype=np.bool_)
            ),
            "confirmation": (
                group_index % 2 == 1
                if args.split_confirmation
                else np.ones(len(current_prediction), dtype=np.bool_)
            ),
        }
        current_counts = {
            split: exact_counts(
                current_prediction,
                arrays.normalized_changes,
                high,
                mask,
            )
            for split, mask in masks.items()
        }
        blocks[name] = {
            "prior": prior,
            "residual": residual,
            "target": arrays.normalized_changes,
            "high": high,
            "current_counts": current_counts,
            "masks": masks,
        }
    selection_split = (
        "calibration" if args.split_confirmation else "full"
    )

    def evaluate(
        selected: np.ndarray,
        split: str = selection_split,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        observed = []
        baseline = []
        total_absolute_error = 0.0
        block_report = {}
        for name, values in blocks.items():
            candidate = prediction(
                values["prior"],
                values["residual"],
                selected,
            )
            counts = exact_counts(
                candidate,
                values["target"],
                values["high"],
                values["masks"][split],
            )
            observed.extend(counts)
            baseline.extend(values["current_counts"][split])
            total_absolute_error += float(
                np.abs(candidate - values["target"])[
                    values["masks"][split]
                ].mean()
            )
            block_report[name] = {
                "current": list(values["current_counts"][split]),
                "candidate": list(counts),
            }
        margins = tuple(
            value - minimum
            for value, minimum in zip(
                observed,
                baseline,
                strict=True,
            )
        )
        non_regression = all(value >= 0 for value in margins)
        key = (
            int(non_regression),
            min(margins),
            sum(margins),
            -total_absolute_error,
            -sum(abs(float(BLENDS[int(value)])) for value in selected),
        )
        return key, {
            "split": split,
            "selected_blends": [
                float(BLENDS[int(value)]) for value in selected
            ],
            "blocks": block_report,
            "margins": list(margins),
            "protected_non_regression": bool(non_regression),
            "key": list(key),
        }

    finals = []
    traces = []
    for start_blend in BLENDS:
        selected = np.full(
            len(STATE_FIELDS),
            BLENDS.index(start_blend),
            dtype=np.int64,
        )
        passes = []
        for pass_index in range(8):
            changed = False
            for field in range(len(STATE_FIELDS)):
                best = int(selected[field])
                best_key = None
                for candidate in range(len(BLENDS)):
                    proposal = selected.copy()
                    proposal[field] = candidate
                    key, _ = evaluate(proposal)
                    if best_key is None or key > best_key:
                        best_key = key
                        best = candidate
                if best != int(selected[field]):
                    selected[field] = best
                    changed = True
            _, state = evaluate(selected)
            passes.append({"pass": pass_index + 1, **state})
            if not changed:
                break
        finals.append(selected.copy())
        traces.append(
            {
                "start_blend": float(start_blend),
                "passes": passes,
            }
        )
    selected = max(finals, key=lambda values: evaluate(values)[0])
    _, selected_report = evaluate(selected)
    _, confirmation_report = evaluate(selected, "confirmation")
    _, full_report = evaluate(selected, "full")
    promotion_passed = bool(
        selected_report["protected_non_regression"]
        and confirmation_report["protected_non_regression"]
        and full_report["protected_non_regression"]
    )
    artifact = {
        "version": "protected_calibrated_round2_forward_v9_one_seed",
        "model": "frozen_v7_plus_residual_tree_v9",
        "route_scope": "predict_forward_from_state_v1",
        "forward_artifact": str(forward_path),
        "forward_artifact_sha256": sha256(forward_path),
        "tree_artifact": str(tree_path),
        "tree_artifact_sha256": sha256(tree_path),
        "field_blend": {
            field: float(BLENDS[int(selected[index])])
            for index, field in enumerate(STATE_FIELDS)
        },
        "field_tree_source": {field: "primary" for field in STATE_FIELDS},
        "secondary_tree_artifact": None,
        "secondary_tree_artifact_sha256": None,
        "current_baseline": str(current_path),
        "current_baseline_sha256": sha256(current_path),
        "protected_calibration": {
            "selection_split": selection_split,
            "selected": selected_report,
            "confirmation": confirmation_report,
            "full": full_report,
            "promotion_passed": promotion_passed,
        },
        "system_validation_used_for_training": False,
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "selected": selected_report,
        "selection_split": selection_split,
        "confirmation": confirmation_report,
        "full": full_report,
        "promotion_passed": promotion_passed,
        "trace": traces,
        "source_contract": {
            "current_baseline": str(current_path),
            "tree_artifact": str(tree_path),
            "system_validation_used": False,
            "protected_split_confirmation": bool(
                args.split_confirmation
            ),
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
