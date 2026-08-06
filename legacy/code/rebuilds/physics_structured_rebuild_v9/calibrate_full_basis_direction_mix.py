#!/usr/bin/env python3
"""Select a fixed per-field direction mix with protected split confirmation."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import labels_from_normalized_change, load_grid_arrays
from physics_structured_rebuild_v9.boundary_direction_runtime import (
    load_boundary_direction_correction_runtime_v9,
)
from physics_structured_rebuild_v9.full_basis_forward_surface import (
    load_full_basis_forward_surface_runtime_v9,
)
from specialist_rebuild_v2.common import DIRECTION_FIELDS, read_jsonl

DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-direction",
        type=Path,
        default=DEFAULT_RUN / "boundary_direction_correction_state_v9.pkl",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=DEFAULT_RUN / "full_basis_forward_surface_v9.pt",
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


def metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    correct = prediction == target
    exact = np.all(correct, axis=1)
    return {
        "count": int(mask.sum()),
        "all_five_count": int(exact[mask].sum()),
        "all_five_exact": float(exact[mask].mean()),
        "per_field_accuracy": correct[mask].mean(axis=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    torch, device = configure(int(args.seed), args.device)
    base_direction, _ = load_boundary_direction_correction_runtime_v9(
        args.base_direction.resolve(), torch, device
    )
    candidate, _ = load_full_basis_forward_surface_runtime_v9(
        args.candidate.resolve(), torch, device
    )
    blocks = {}
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        rows = read_jsonl(path)
        arrays = load_grid_arrays(path, include_legacy_features=False)
        _, _, _, current_grid = base_direction.base_grid(rows)
        current = current_grid.reshape(-1, 5)
        alternative = labels_from_normalized_change(
            candidate.predict_changes(rows).reshape(-1, 5)
        )
        group_index = np.repeat(np.arange(arrays.group_count), 81)
        blocks[name] = {
            "current": current,
            "alternative": alternative,
            "target": arrays.labels,
            "calibration": group_index % 2 == 0,
            "confirmation": group_index % 2 == 1,
        }
    candidates = []
    for raw_choice in itertools.product((False, True), repeat=5):
        choice = np.asarray(raw_choice, dtype=np.bool_)
        margins = []
        reports = {}
        for name, values in blocks.items():
            selected = np.where(
                choice[None, :],
                values["alternative"],
                values["current"],
            )
            current_metric = metrics(
                values["current"], values["target"], values["calibration"]
            )
            selected_metric = metrics(
                selected, values["target"], values["calibration"]
            )
            margin = (
                selected_metric["all_five_count"]
                - current_metric["all_five_count"]
            )
            margins.append(margin)
            reports[name] = {
                "current": current_metric,
                "selected": selected_metric,
                "margin": int(margin),
            }
        nonregression = all(margin >= 0 for margin in margins)
        key = (
            int(nonregression),
            min(margins),
            sum(margins),
            -int(choice.sum()),
            tuple(int(value) for value in choice),
        )
        candidates.append((key, choice, reports))
    best = max(candidates, key=lambda row: row[0])
    selected_choice = best[1]
    confirmation = {}
    confirmation_margins = []
    for name, values in blocks.items():
        selected = np.where(
            selected_choice[None, :],
            values["alternative"],
            values["current"],
        )
        current_metric = metrics(
            values["current"], values["target"], values["confirmation"]
        )
        selected_metric = metrics(
            selected, values["target"], values["confirmation"]
        )
        margin = (
            selected_metric["all_five_count"]
            - current_metric["all_five_count"]
        )
        confirmation_margins.append(margin)
        confirmation[name] = {
            "current": current_metric,
            "selected": selected_metric,
            "margin": int(margin),
        }
    report = {
        "version": "full_basis_direction_field_mix_v9_one_seed",
        "selected_source": {
            field: ("full_basis" if selected_choice[index] else "accepted")
            for index, field in enumerate(DIRECTION_FIELDS)
        },
        "calibration": best[2],
        "confirmation": confirmation,
        "confirmation_passed": bool(
            all(margin >= 0 for margin in confirmation_margins)
            and sum(confirmation_margins) > 0
        ),
        "source_contract": {
            "protected_validation_used_as_split_calibrator": True,
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
