#!/usr/bin/env python3
"""Calibrate strict-forward field blends under protected non-regression."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import load_grid_arrays
from physics_structured_rebuild_v9.evaluate_forward_candidate_protected import (
    high_mask,
    read_jsonl,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
)
from physics_structured_rebuild_v9.strict_forward_runtime import (
    load_strict_forward_correction_runtime_v9,
    sha256,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_OLD = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/val.jsonl"
)
DEFAULT_DIFFICULT = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl"
)
BLENDS = np.asarray(
    [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 1.25],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--current",
        type=Path,
        default=DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
    )
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--split-confirmation",
        action="store_true",
        help=(
            "Choose blends on even-numbered groups and reserve odd-numbered "
            "groups for untouched confirmation."
        ),
    )
    return parser.parse_args()


def block_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    high: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(prediction), dtype=np.bool_)
    error = np.abs(prediction - target)
    passed = error <= 1.0
    exact = np.all(passed, axis=1)
    return {
        "count": int(mask.sum()),
        "strict_count": int(exact[mask].sum()),
        "high_count": int(exact[mask & high].sum()),
        "mae": float(error[mask].mean()),
        "per_field_pass": passed[mask].mean(axis=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    candidate_path = args.candidate.resolve()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    torch, device = configure(int(args.seed), args.device)
    current, _ = load_forward_selector_ensemble_runtime_v9(
        args.current.resolve(),
        torch,
        device,
    )
    candidate, artifact = load_strict_forward_correction_runtime_v9(
        candidate_path,
        torch,
        device,
    )
    original_blend = np.asarray(
        artifact["field_blend"],
        dtype=np.float32,
    )
    if np.any(original_blend <= 0.0):
        raise ValueError(
            "protected calibration requires nonzero source field blends"
        )

    blocks = {}
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        rows = read_jsonl(path)
        arrays = load_grid_arrays(path, include_legacy_features=False)
        current_prediction = current.predict_changes(rows).reshape(-1, 5)
        candidate_prediction = candidate.predict_changes(rows).reshape(-1, 5)
        unit_correction = (
            candidate_prediction - current_prediction
        ) / original_blend[None, :]
        group_index = np.repeat(
            np.arange(arrays.group_count, dtype=np.int64),
            len(current_prediction) // arrays.group_count,
        )
        if len(group_index) != len(current_prediction):
            raise ValueError("protected forward group shape differs")
        blocks[name] = {
            "current": current_prediction,
            "unit_correction": unit_correction,
            "target": np.asarray(
                arrays.normalized_changes,
                dtype=np.float32,
            ),
            "high": high_mask(arrays.group_count),
            "masks": {
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
            },
        }

    baseline = {
        name: {
            split: block_metrics(
                block["current"],
                block["target"],
                block["high"],
                mask,
            )
            for split, mask in block["masks"].items()
        }
        for name, block in blocks.items()
    }
    selection_split = (
        "calibration" if args.split_confirmation else "full"
    )

    def evaluate(
        blend: np.ndarray,
        split: str = selection_split,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        observed = {}
        margins = []
        maes = []
        for name, block in blocks.items():
            prediction = (
                block["current"]
                + block["unit_correction"] * blend[None, :]
            )
            metrics = block_metrics(
                prediction,
                block["target"],
                block["high"],
                block["masks"][split],
            )
            observed[name] = metrics
            margins.extend(
                [
                    metrics["strict_count"]
                    - baseline[name][split]["strict_count"],
                    metrics["high_count"]
                    - baseline[name][split]["high_count"],
                ]
            )
            maes.append(metrics["mae"])
        passed = min(margins) >= 0
        key = (
            int(passed),
            min(margins) if passed else -10**9,
            sum(margins) if passed else -10**9,
            -float(np.mean(maes)),
            -float(np.abs(blend).sum()),
        )
        return key, {
            "split": split,
            "blend": blend.tolist(),
            "blocks": observed,
            "margins": margins,
            "protected_non_regression": bool(passed),
        }

    starts = [
        np.zeros(5, dtype=np.float32),
        np.minimum(original_blend, 0.25),
        np.minimum(original_blend, 0.50),
        original_blend.copy(),
    ]
    finals = []
    trace = []
    for start_index, initial in enumerate(starts):
        selected = initial.copy()
        passes = []
        for pass_index in range(5):
            changed = False
            for field in range(5):
                candidates = []
                for value in BLENDS:
                    proposal = selected.copy()
                    proposal[field] = float(value)
                    key, details = evaluate(proposal)
                    candidates.append((key, -float(value), proposal, details))
                best = max(candidates, key=lambda value: value[:2])
                if not np.array_equal(selected, best[2]):
                    selected = best[2]
                    changed = True
            _, details = evaluate(selected)
            passes.append(
                {
                    "pass": pass_index + 1,
                    **details,
                }
            )
            if not changed:
                break
        key, details = evaluate(selected)
        finals.append((key, -float(np.abs(selected).sum()), selected, details))
        trace.append(
            {
                "start": start_index,
                "initial": initial.tolist(),
                "passes": passes,
            }
        )

    best = max(finals, key=lambda value: value[:2])
    selected_blend = best[2]
    selected_details = best[3]
    if not selected_details["protected_non_regression"]:
        raise RuntimeError("zero correction must satisfy protected constraints")
    _, confirmation_details = evaluate(
        selected_blend,
        "confirmation",
    )
    _, full_details = evaluate(selected_blend, "full")
    promotion_passed = bool(
        selected_details["protected_non_regression"]
        and confirmation_details["protected_non_regression"]
        and full_details["protected_non_regression"]
    )
    calibrated = copy.deepcopy(artifact)
    calibrated["version"] = "strict_forward_protected_calibrated_v9_one_seed"
    calibrated["field_blend"] = selected_blend
    calibrated["protected_calibration"] = {
        "source_candidate": str(candidate_path),
        "source_candidate_sha256": sha256(candidate_path),
        "selected": selected_details,
        "selection_split": selection_split,
        "confirmation": confirmation_details,
        "full": full_details,
        "promotion_passed": promotion_passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibrated, output)
    report = {
        "version": calibrated["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "source_candidate": str(candidate_path),
        "original_blend": original_blend.tolist(),
        "baseline": baseline,
        "selected": selected_details,
        "selection_split": selection_split,
        "confirmation": confirmation_details,
        "full": full_details,
        "promotion_passed": promotion_passed,
        "trace": trace,
        "source_contract": {
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
