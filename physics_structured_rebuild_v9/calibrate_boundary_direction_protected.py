#!/usr/bin/env python3
"""Calibrate boundary-direction rules under protected non-regression."""

from __future__ import annotations

import argparse
import copy
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
from physics_structured_rebuild_v9.boundary_direction_runtime import (
    correction_features,
    load_boundary_direction_correction_runtime_v9,
    sha256,
)
from physics_structured_rebuild_v9.evaluate_forward_candidate_protected import (
    high_mask,
    read_jsonl,
)
from physics_structured_rebuild_v9.train_boundary_direction_correction import (
    BOUNDARY_LIMITS,
    CONFIDENCE_LIMITS,
    MARGIN_LIMITS,
    model_probabilities,
)

DEFAULT_OLD = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/val.jsonl"
)
DEFAULT_DIFFICULT = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
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
            "Select rules on even-numbered groups and reserve odd-numbered "
            "groups for confirmation."
        ),
    )
    return parser.parse_args()


def rule_choices() -> list[dict[str, float] | None]:
    return [None] + [
        {
            "boundary_limit": float(boundary),
            "confidence_limit": float(confidence),
            "margin_limit": float(margin),
        }
        for boundary in BOUNDARY_LIMITS
        for confidence in CONFIDENCE_LIMITS
        for margin in MARGIN_LIMITS
    ]


def field_predictions(
    base: np.ndarray,
    threshold: np.ndarray,
    probabilities: np.ndarray,
    choices: list[dict[str, float] | None],
) -> list[list[np.ndarray]]:
    ordered = np.sort(probabilities, axis=2)
    model_class = probabilities.argmax(axis=2)
    distance = np.abs(np.abs(threshold) - 1.0)
    output = []
    for field in range(5):
        field_output = []
        for rule in choices:
            prediction = base[:, field].astype(np.int8, copy=True)
            if rule is not None:
                apply = (
                    (
                        distance[:, field]
                        <= float(rule["boundary_limit"])
                    )
                    & (
                        ordered[:, field, -1]
                        >= float(rule["confidence_limit"])
                    )
                    & (
                        ordered[:, field, -1]
                        - ordered[:, field, -2]
                        >= float(rule["margin_limit"])
                    )
                    & (model_class[:, field] != base[:, field])
                )
                prediction[apply] = model_class[apply, field].astype(
                    np.int8
                )
            field_output.append(prediction)
        output.append(field_output)
    return output


def main() -> None:
    args = parse_args()
    candidate_path = args.candidate.resolve()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    torch, device = configure(int(args.seed), args.device)
    runtime, artifact = load_boundary_direction_correction_runtime_v9(
        candidate_path,
        torch,
        device,
    )
    choices = rule_choices()
    blocks = {}
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        rows = read_jsonl(path)
        arrays = load_grid_arrays(path, include_legacy_features=False)
        physical, prior, threshold, base = runtime.base_grid(rows)
        current_forward = runtime.current_forward.predict_changes(rows)
        values = correction_features(
            physical.reshape(-1, physical.shape[-1]),
            prior.reshape(-1, 5),
            current_forward.reshape(-1, 5),
            threshold.reshape(-1, 5),
            base,
        )
        probabilities = model_probabilities(runtime.models, values)
        truth = np.asarray(arrays.labels, dtype=np.int8)
        base = base.astype(np.int8)
        high = high_mask(arrays.group_count)
        base_exact = np.all(base == truth, axis=1)
        group_index = np.repeat(
            np.arange(arrays.group_count, dtype=np.int64),
            len(base) // arrays.group_count,
        )
        if len(group_index) != len(base):
            raise ValueError("protected direction group shape differs")
        split_masks = {
            "full": np.ones(len(base), dtype=np.bool_),
            "calibration": (
                group_index % 2 == 0
                if args.split_confirmation
                else np.ones(len(base), dtype=np.bool_)
            ),
            "confirmation": (
                group_index % 2 == 1
                if args.split_confirmation
                else np.ones(len(base), dtype=np.bool_)
            ),
        }
        blocks[name] = {
            "truth": truth,
            "high": high,
            "split_masks": split_masks,
            "baseline": {
                split: {
                    "joint": int(base_exact[mask].sum()),
                    "high": int(base_exact[mask & high].sum()),
                }
                for split, mask in split_masks.items()
            },
            "field_predictions": field_predictions(
                base,
                threshold.reshape(-1, 5),
                probabilities,
                choices,
            ),
        }

    selection_split = (
        "calibration" if args.split_confirmation else "full"
    )

    def evaluate(
        indices: list[int],
        split: str = selection_split,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        observed = {}
        margins = []
        total_field_correct = 0
        total_corrections = 0
        for name, block in blocks.items():
            prediction = np.column_stack(
                [
                    block["field_predictions"][field][indices[field]]
                    for field in range(5)
                ]
            )
            truth = block["truth"]
            mask = block["split_masks"][split]
            exact = np.all(prediction == truth, axis=1)
            joint = int(exact[mask].sum())
            high = int(exact[mask & block["high"]].sum())
            baseline = block["baseline"][split]
            margins.extend(
                [
                    joint - baseline["joint"],
                    high - baseline["high"],
                ]
            )
            total_field_correct += int(
                (prediction[mask] == truth[mask]).sum()
            )
            baseline_prediction = np.column_stack(
                [
                    block["field_predictions"][field][0]
                    for field in range(5)
                ]
            )
            total_corrections += int(
                (
                    prediction[mask] != baseline_prediction[mask]
                ).sum()
            )
            observed[name] = {
                "split": split,
                "count": int(mask.sum()),
                "joint_count": joint,
                "high_count": high,
                "baseline_joint_count": baseline["joint"],
                "baseline_high_count": baseline["high"],
            }
        passed = min(margins) >= 0
        key = (
            int(passed),
            min(margins) if passed else -10**9,
            sum(margins) if passed else -10**9,
            total_field_correct,
            -total_corrections,
        )
        return key, {
            "choice_indices": list(indices),
            "rules": [choices[index] for index in indices],
            "blocks": observed,
            "margins": margins,
            "total_field_correct": total_field_correct,
            "total_corrections": total_corrections,
            "protected_non_regression": bool(passed),
        }

    original_indices = []
    for rule in artifact["rules"]:
        try:
            original_indices.append(choices.index(rule))
        except ValueError:
            original_indices.append(0)
    starts = [[0] * 5, original_indices]
    finals = []
    trace = []
    for start_index, initial in enumerate(starts):
        selected = list(initial)
        passes = []
        for pass_index in range(5):
            changed = False
            for field in range(5):
                candidates = []
                for choice_index in range(len(choices)):
                    proposal = list(selected)
                    proposal[field] = choice_index
                    key, details = evaluate(proposal)
                    candidates.append(
                        (
                            key,
                            choice_index == 0,
                            -choice_index,
                            choice_index,
                            details,
                        )
                    )
                best = max(candidates, key=lambda value: value[:3])
                if selected[field] != best[3]:
                    selected[field] = best[3]
                    changed = True
            _, details = evaluate(selected)
            passes.append({"pass": pass_index + 1, **details})
            if not changed:
                break
        key, details = evaluate(selected)
        finals.append((key, selected, details))
        trace.append(
            {
                "start": start_index,
                "initial": initial,
                "passes": passes,
            }
        )
    best = max(finals, key=lambda value: value[0])
    selected_details = best[2]
    if not selected_details["protected_non_regression"]:
        raise RuntimeError("zero rules must satisfy protected constraints")
    _, confirmation_details = evaluate(
        best[1],
        "confirmation",
    )
    _, full_details = evaluate(best[1], "full")
    promotion_passed = bool(
        selected_details["protected_non_regression"]
        and confirmation_details["protected_non_regression"]
        and full_details["protected_non_regression"]
    )
    calibrated = copy.deepcopy(artifact)
    calibrated["version"] = (
        "boundary_direction_protected_calibrated_state_v9_one_seed"
    )
    calibrated["rules"] = selected_details["rules"]
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
    with output.open("wb") as stream:
        pickle.dump(calibrated, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "version": calibrated["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "source_candidate": str(candidate_path),
        "original_rules": artifact["rules"],
        "selected": selected_details,
        "selection_split": selection_split,
        "confirmation": confirmation_details,
        "full": full_details,
        "promotion_passed": promotion_passed,
        "trace": trace,
        "source_contract": {
            "protected_split_confirmation": bool(
                args.split_confirmation
            ),
            "calibration_groups": (
                "even" if args.split_confirmation else "all"
            ),
            "confirmation_groups": (
                "odd" if args.split_confirmation else "all"
            ),
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
