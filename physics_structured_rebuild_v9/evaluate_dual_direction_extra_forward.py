#!/usr/bin/env python3
"""Evaluate dual-source direction and Extra Trees forward end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joint_forward_direction_v7.evaluate_system as base_evaluator
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_COMBINED_NATURAL_INVERSE_V9,
    DEFAULT_COMPLEXITY_MIX_FORWARD,
    DEFAULT_DUAL_SOURCE_IMAGE_HYBRID,
    DEFAULT_DUAL_SOURCE_FORWARD_IMAGE,
    DEFAULT_DUAL_SOURCE_FORWARD_STATE,
    DEFAULT_DUAL_SOURCE_STATE_HYBRID,
    DEFAULT_EXTRA_TREES_FORWARD_IMAGE,
    DEFAULT_EXTRA_TREES_FORWARD_STATE,
    DEFAULT_EXTRA_FORWARD_DIRECTION_IMAGE_HYBRID,
    DEFAULT_EXTRA_FORWARD_DIRECTION_STATE_HYBRID,
    DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
    DEFAULT_INVERSE_ENUMERATOR_SELECTOR_V9,
    DEFAULT_LEAKAGE_FREE_INVERSE_SELECTOR_V9,
    DEFAULT_NATURAL_GRID_FORWARD_IMAGE,
    DEFAULT_NATURAL_GRID_FORWARD_STATE,
    DEFAULT_RESTRICTED_FORWARD_SELECTOR_V9,
    DEFAULT_RESIDUAL_FORWARD_STATE,
    DEFAULT_TRANSFORMER_INVERSE_V8,
    OrchestratedCombinedNaturalInverseRuntimeV9,
    OrchestratedDualDirectionExtraForwardInverseRuntimeV9,
    OrchestratedDualForwardAccumulatedRuntimeV9,
    OrchestratedComplexityMixForwardRuntimeV9,
    OrchestratedExtraForwardDirectionSelectedEnumeratorRuntimeV9,
    OrchestratedForwardSelectorEnsembleRuntimeV9,
    OrchestratedLeakageFreeSelectorRuntimeV9,
    OrchestratedNaturalVisualForwardRuntimeV9,
    OrchestratedNaturalGridForwardRuntimeV9,
    OrchestratedRestrictedForwardSelectorRuntimeV9,
    OrchestratedSelectedEnumeratorRuntimeV9,
    OrchestratedSplitForwardInverseRuntimeV9,
)

DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "orchestrated_dual_direction_extra_forward_v9_validation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--split-inverse-forward",
        action="store_true",
        help=(
            "Use the retained histogram residual forward model only for "
            "numerical inverse candidate enumeration."
        ),
    )
    parser.add_argument(
        "--inverse-selector",
        action="store_true",
        help="Use the protected dual-forward inverse enumerator selector.",
    )
    parser.add_argument(
        "--learned-inverse-selector",
        action="store_true",
        help="Use the training-only learned dual-forward inverse selector.",
    )
    parser.add_argument(
        "--extra-forward-direction",
        action="store_true",
        help="Use the dual-tree direction calibration with Extra Trees thresholds.",
    )
    parser.add_argument(
        "--dual-source-forward",
        action="store_true",
        help="Use the calibrated per-field dual-source forward ensemble.",
    )
    parser.add_argument(
        "--complexity-mixed-forward",
        action="store_true",
        help="Use the action-complexity-specific forward mixture candidate.",
    )
    parser.add_argument(
        "--natural-grid-forward",
        action="store_true",
        help="Use the 81k natural-grid augmented direct forward candidate.",
    )
    parser.add_argument(
        "--combined-natural-inverse-ranker",
        action="store_true",
        help=(
            "Use the disjoint 2,000-group inverse ranker with the retained "
            "natural-grid forward enumerator."
        ),
    )
    parser.add_argument(
        "--natural-visual-forward",
        action="store_true",
        help=(
            "Also use the retained natural-grid forward model inside the "
            "frozen modular visual inverse scorer."
        ),
    )
    parser.add_argument(
        "--restricted-forward-selector",
        action="store_true",
        help=(
            "Use the protected state-only forward expert selector on top of "
            "the retained natural visual/inverse stack."
        ),
    )
    parser.add_argument(
        "--forward-selector-ensemble",
        action="store_true",
        help=(
            "Use the protected second-stage state-forward selector ensemble "
            "on top of the retained natural visual/inverse stack."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    details = output.with_suffix(".details.jsonl")
    if output.exists() or details.exists():
        raise RuntimeError(f"refusing to overwrite completed output: {output}")
    original_class = base_evaluator.OrchestratedSharedRuntimeV7
    original_argv = sys.argv
    selector_ensemble_enabled = bool(args.forward_selector_ensemble)
    restricted_forward_enabled = bool(
        args.restricted_forward_selector or selector_ensemble_enabled
    )
    natural_visual_enabled = bool(
        args.natural_visual_forward or restricted_forward_enabled
    )
    combined_inverse_enabled = bool(
        args.combined_natural_inverse_ranker or natural_visual_enabled
    )
    selector_enabled = bool(
        args.inverse_selector
        or args.learned_inverse_selector
        or args.natural_grid_forward
    ) and not combined_inverse_enabled
    runtime_class = (
        OrchestratedForwardSelectorEnsembleRuntimeV9
        if selector_ensemble_enabled
        else OrchestratedRestrictedForwardSelectorRuntimeV9
        if restricted_forward_enabled
        else OrchestratedNaturalVisualForwardRuntimeV9
        if natural_visual_enabled
        else OrchestratedCombinedNaturalInverseRuntimeV9
        if combined_inverse_enabled
        else OrchestratedLeakageFreeSelectorRuntimeV9
        if args.learned_inverse_selector
        else OrchestratedNaturalGridForwardRuntimeV9
        if args.natural_grid_forward
        else OrchestratedComplexityMixForwardRuntimeV9
        if args.complexity_mixed_forward
        else OrchestratedDualForwardAccumulatedRuntimeV9
        if (
            selector_enabled
            and args.extra_forward_direction
            and args.dual_source_forward
        )
        else (
            OrchestratedExtraForwardDirectionSelectedEnumeratorRuntimeV9
            if selector_enabled and args.extra_forward_direction
            else (
                OrchestratedSelectedEnumeratorRuntimeV9
                if selector_enabled
                else (
                    OrchestratedSplitForwardInverseRuntimeV9
                    if args.split_inverse_forward
                    else OrchestratedDualDirectionExtraForwardInverseRuntimeV9
                )
            )
        )
    )
    base_evaluator.OrchestratedSharedRuntimeV7 = runtime_class
    sys.argv = [
        original_argv[0],
        "--output",
        str(output),
        "--details",
        str(details),
    ]
    try:
        base_evaluator.main()
    finally:
        base_evaluator.OrchestratedSharedRuntimeV7 = original_class
        sys.argv = original_argv
    report = json.loads(output.read_text(encoding="utf-8"))
    report["evaluation_version"] = (
        "saved_qwen_plus_forward_selector_ensemble_v9"
        if selector_ensemble_enabled
        else "saved_qwen_plus_restricted_forward_selector_v9"
        if restricted_forward_enabled
        else "saved_qwen_plus_natural_visual_forward_v9"
        if natural_visual_enabled
        else "saved_qwen_plus_combined_natural_inverse_ranker_v9"
        if combined_inverse_enabled
        else "saved_qwen_plus_leakage_free_inverse_selector_v9"
        if args.learned_inverse_selector
        else "saved_qwen_plus_natural_grid_forward_v9"
        if args.natural_grid_forward
        else "saved_qwen_plus_complexity_mixed_forward_v9"
        if args.complexity_mixed_forward
        else "saved_qwen_plus_accumulated_dual_forward_v9"
        if args.dual_source_forward
        else (
            "saved_qwen_plus_extra_forward_dual_direction_inverse_selector_v9"
        )
        if args.inverse_selector and args.extra_forward_direction
        else (
            "saved_qwen_plus_dual_direction_extra_forward_inverse_selector_v9"
            if selector_enabled
            else (
                "saved_qwen_plus_dual_direction_extra_forward_split_inverse_v9"
                if args.split_inverse_forward
                else "saved_qwen_plus_dual_direction_extra_forward_inverse_v9"
            )
        )
    )
    report["scope"] = (
        "direction selects HistGradientBoosting or Extra Trees independently "
        "per route and field; forward requests use calibrated Extra Trees "
        "residuals; numerical-state inverse uses "
        + (
            "the disjoint-data adapted v8 ranker, while visual inverse uses "
            "the frozen visual scorer with natural-grid forward candidates; "
            if natural_visual_enabled
            else "the disjoint-data adapted v8 ranker and retained natural-"
            "grid forward enumerator; "
            if combined_inverse_enabled
            else
            "a training-only learned dual-forward enumerator selector; "
            if args.learned_inverse_selector
            else "a protected dual-forward enumerator selector; "
            if selector_enabled
            else (
                "retained histogram residual forward enumeration; "
                if args.split_inverse_forward
                else "Extra Trees forward enumeration; "
            )
        )
        + (
            "visual inverse, measurement, Qwen, and deployed defaults remain "
            "frozen"
        )
    )
    extra_direction_enabled = bool(
        args.extra_forward_direction
        or args.dual_source_forward
        or args.complexity_mixed_forward
        or args.natural_grid_forward
        or args.learned_inverse_selector
        or combined_inverse_enabled
    )
    state_direction_path = (
        DEFAULT_EXTRA_FORWARD_DIRECTION_STATE_HYBRID
        if extra_direction_enabled
        else DEFAULT_DUAL_SOURCE_STATE_HYBRID
    )
    image_direction_path = (
        DEFAULT_EXTRA_FORWARD_DIRECTION_IMAGE_HYBRID
        if extra_direction_enabled
        else DEFAULT_DUAL_SOURCE_IMAGE_HYBRID
    )
    state_forward_path = (
        DEFAULT_NATURAL_GRID_FORWARD_STATE
        if args.natural_grid_forward or combined_inverse_enabled
        else DEFAULT_COMPLEXITY_MIX_FORWARD
        if args.complexity_mixed_forward
        else DEFAULT_DUAL_SOURCE_FORWARD_STATE
        if args.dual_source_forward
        else DEFAULT_EXTRA_TREES_FORWARD_STATE
    )
    image_forward_path = (
        DEFAULT_NATURAL_GRID_FORWARD_IMAGE
        if args.natural_grid_forward or combined_inverse_enabled
        else DEFAULT_COMPLEXITY_MIX_FORWARD
        if args.complexity_mixed_forward
        else DEFAULT_DUAL_SOURCE_FORWARD_IMAGE
        if args.dual_source_forward
        else DEFAULT_EXTRA_TREES_FORWARD_IMAGE
    )
    artifact_paths = {
        "state_direction_v9": state_direction_path,
        "image_direction_v9": image_direction_path,
        "state_forward_v9": state_forward_path,
        "image_forward_v9": image_forward_path,
        "transformer_numerical_inverse_v8": (
            DEFAULT_COMBINED_NATURAL_INVERSE_V9
            if combined_inverse_enabled
            else DEFAULT_TRANSFORMER_INVERSE_V8
        ),
    }
    if args.split_inverse_forward or selector_enabled or combined_inverse_enabled:
        artifact_paths["histogram_forward_for_inverse_v9"] = (
            DEFAULT_NATURAL_GRID_FORWARD_STATE
            if combined_inverse_enabled
            else DEFAULT_RESIDUAL_FORWARD_STATE
        )
    if selector_enabled:
        artifact_paths["inverse_enumerator_selector_v9"] = (
            DEFAULT_LEAKAGE_FREE_INVERSE_SELECTOR_V9
            if args.learned_inverse_selector
            else DEFAULT_INVERSE_ENUMERATOR_SELECTOR_V9
        )
    if natural_visual_enabled:
        artifact_paths["natural_grid_forward_for_visual_inverse_v9"] = (
            DEFAULT_NATURAL_GRID_FORWARD_STATE
        )
    if restricted_forward_enabled:
        artifact_paths["restricted_forward_selector_v9"] = (
            DEFAULT_RESTRICTED_FORWARD_SELECTOR_V9
        )
    if selector_ensemble_enabled:
        artifact_paths.pop("restricted_forward_selector_v9", None)
        artifact_paths["forward_selector_ensemble_v9"] = (
            DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9
        )
    for name, path in artifact_paths.items():
        report["artifacts"][name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path.resolve()),
        }
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = report["metrics"]
    print(
        json.dumps(
            {
                "output": str(output),
                "record_count": report["record_count"],
                "direction_all_five_exact": metrics[
                    "end_to_end_direction_physical_all_five_exact"
                ],
                "forward_strict_all_five": metrics[
                    "end_to_end_forward_physical_strict_all_five_success"
                ],
                "forward_correctly_routed": metrics[
                    "correctly_routed_forward_physical_strict_all_five_success"
                ],
                "inverse_target_reached": metrics[
                    "end_to_end_inverse_target_reached_rate"
                ],
                "measurement_strict_all_five": metrics[
                    "end_to_end_visual_measurement_strict_all_five_success"
                ],
                "forward_by_route": metrics["forward_physical_by_route"],
                "inverse_by_route": metrics["inverse_physical_by_route"],
                "direction_by_route": metrics[
                    "direction_physical_by_route"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
