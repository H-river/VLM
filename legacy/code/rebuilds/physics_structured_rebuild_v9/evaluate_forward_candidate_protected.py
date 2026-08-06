#!/usr/bin/env python3
"""Compare a residual-forward candidate with the current protected ensemble."""

from __future__ import annotations

import argparse
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
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_extension_runtime_v9,
    load_forward_selector_ensemble_runtime_v9,
    load_residual_forward_runtime_v9,
)
from physics_structured_rebuild_v9.strict_forward_runtime import (
    load_strict_forward_correction_runtime_v9,
)
from physics_structured_rebuild_v9.full_basis_forward_surface import (
    load_full_basis_forward_surface_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
)
from specialist_rebuild_v2.common import ACTION_FIELDS, fixed_action_grid

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        type=Path,
        default=DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--candidate-kind",
        choices=(
            "residual",
            "strict_correction",
            "selector_extension",
            "full_basis",
        ),
        default="residual",
    )
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
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def high_mask(group_count: int) -> np.ndarray:
    per_action = np.asarray(
        [
            sum(abs(float(action[field])) > 0.0 for field in ACTION_FIELDS)
            >= 3
            for action in fixed_action_grid()
        ],
        dtype=np.bool_,
    )
    return np.tile(per_action, group_count)


def metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    passed = np.abs(prediction - target) <= 1.0
    exact = np.all(passed, axis=1)
    return exact, {
        "count": int(len(exact)),
        "strict_all_five_count": int(exact.sum()),
        "strict_all_five_success": float(exact.mean()),
        "high_complexity_count": int(high.sum()),
        "high_complexity_success_count": int(exact[high].sum()),
        "high_complexity_success": float(exact[high].mean()),
        "per_field_tolerance_pass": passed.mean(axis=0).tolist(),
        "mae_in_tolerance_units": float(np.abs(prediction - target).mean()),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    torch, device = configure(int(args.seed), args.device)
    current, _ = load_forward_selector_ensemble_runtime_v9(
        args.current.resolve(),
        torch,
        device,
    )
    if args.candidate_kind == "strict_correction":
        candidate, _ = load_strict_forward_correction_runtime_v9(
            args.candidate.resolve(),
            torch,
            device,
        )
    elif args.candidate_kind == "selector_extension":
        candidate, _ = load_forward_selector_extension_runtime_v9(
            args.candidate.resolve(),
            torch,
            device,
        )
    elif args.candidate_kind == "full_basis":
        candidate, _ = load_full_basis_forward_surface_runtime_v9(
            args.candidate.resolve(),
            torch,
            device,
        )
    else:
        candidate, _ = load_residual_forward_runtime_v9(
            args.candidate.resolve(),
            torch,
            device,
        )
    blocks = {}
    promotion = True
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        rows = read_jsonl(path)
        arrays = load_grid_arrays(path, include_legacy_features=False)
        current_prediction = current.predict_changes(rows).reshape(-1, 5)
        candidate_prediction = candidate.predict_changes(rows).reshape(-1, 5)
        high = high_mask(arrays.group_count)
        current_success, current_metrics = metrics(
            current_prediction,
            arrays.normalized_changes,
            high,
        )
        candidate_success, candidate_metrics = metrics(
            candidate_prediction,
            arrays.normalized_changes,
            high,
        )
        block_pass = (
            candidate_metrics["strict_all_five_count"]
            >= current_metrics["strict_all_five_count"]
            and candidate_metrics["high_complexity_success_count"]
            >= current_metrics["high_complexity_success_count"]
        )
        promotion = promotion and block_pass
        blocks[name] = {
            "current": current_metrics,
            "candidate": candidate_metrics,
            "candidate_only_success_count": int(
                (candidate_success & ~current_success).sum()
            ),
            "current_only_success_count": int(
                (current_success & ~candidate_success).sum()
            ),
            "oracle_union_success_count": int(
                (current_success | candidate_success).sum()
            ),
            "promotion_passed": bool(block_pass),
        }
        print(json.dumps({"block": name, **blocks[name]}, sort_keys=True))
    report = {
        "version": "forward_candidate_protected_v9_one_seed",
        "candidate_kind": str(args.candidate_kind),
        "blocks": blocks,
        "promotion_passed": bool(promotion),
        "source_contract": {
            "system_validation_used": False,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
