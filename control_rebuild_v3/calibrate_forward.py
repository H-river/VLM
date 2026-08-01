#!/usr/bin/env python3
"""Select and package a validation-safe blend of v2 and joint v3 forward models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import (
    group_arrays,
    inverse_pair_arrays,
    read_json,
    read_jsonl,
)
from control_rebuild_v3.models import joint_forward_model
from control_rebuild_v3.train_forward import (
    configure,
    forward_metrics,
    inverse_selection_metrics,
    predict_all,
    predicted_states,
)
from specialist_rebuild_v2.train_models import (
    load_forward_predictor,
    predict_grid_states,
)


DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_V2_RUN = REPO_ROOT.parent / "VLM_runs/specialist_rebuild_v2_one_seed"
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_runs/control_rebuild_v3_one_seed"
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=DEFAULT_DATA)
    parser.add_argument("v2_run", type=Path, nargs="?", default=DEFAULT_V2_RUN)
    parser.add_argument("output_dir", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def select_blend_candidate(
    candidates: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Maximize inverse success subject to not degrading either forward metric."""

    strict_floor = float(baseline["forward"]["strict_all_five_success"])
    mae_ceiling = float(baseline["forward"]["mae_in_tolerance_units"])
    eligible = [
        row
        for row in candidates
        if float(row["forward"]["strict_all_five_success"])
        >= strict_floor - 1e-12
        and float(row["forward"]["mae_in_tolerance_units"])
        <= mae_ceiling + 1e-12
    ]
    if not eligible:
        raise RuntimeError("no blend satisfies the v2 forward safety constraints")
    return max(
        eligible,
        key=lambda row: (
            float(row["inverse_selection"]["target_success_feasible"]),
            float(row["forward"]["strict_all_five_success"]),
            -float(row["forward"]["mae_in_tolerance_units"]),
            -float(row["alpha"]),
        ),
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    v2_run = args.v2_run.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    torch, device = configure(int(config["seed"]), args.device)

    rows = read_jsonl(data_dir / "grids/val.jsonl")
    pairs = read_jsonl(data_dir / "inverse/val.jsonl")
    contexts, current, tolerance, target, group_ids = group_arrays(rows)
    grid_map = {str(row["group_id"]): row for row in rows}
    group_position = {
        group_id: index for index, group_id in enumerate(group_ids)
    }
    group_index, _, desired, positives, statuses = inverse_pair_arrays(
        pairs, grid_map, group_position
    )

    legacy_model, legacy_artifact = load_forward_predictor(
        v2_run, torch, device
    )
    legacy_map = predict_grid_states(
        rows, legacy_model, legacy_artifact, torch, device
    )
    legacy_states = np.stack(
        [legacy_map[group_id] for group_id in group_ids]
    )
    legacy_change = (
        legacy_states - current[:, None, :]
    ) / tolerance[:, None, :]

    joint_path = output_dir / "forward_control_v3.pt"
    joint_artifact = torch.load(
        joint_path, map_location="cpu", weights_only=False
    )
    joint_model = joint_forward_model(
        torch,
        int(joint_artifact["context_dim"]),
        int(joint_artifact["basis_dim"]),
    ).to(device)
    joint_model.load_state_dict(joint_artifact["state_dict"])
    normalized_context = (
        contexts - np.asarray(joint_artifact["context_mean"])
    ) / np.asarray(joint_artifact["context_scale"])
    basis = torch.as_tensor(
        np.asarray(joint_artifact["action_basis"], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    joint_change = predict_all(
        torch, joint_model, normalized_context, basis, device
    )

    candidates = []
    for alpha in config["forward"]["blend_alpha_candidates"]:
        alpha = float(alpha)
        predicted = (
            (1.0 - alpha) * legacy_change + alpha * joint_change
        ).astype(np.float32)
        candidates.append(
            {
                "alpha": alpha,
                "forward": forward_metrics(target, predicted),
                "inverse_selection": inverse_selection_metrics(
                    predicted_states(current, tolerance, predicted),
                    group_index,
                    desired,
                    positives,
                    statuses,
                ),
            }
        )
    baseline = next(row for row in candidates if row["alpha"] == 0.0)
    selected = select_blend_candidate(candidates, baseline)

    artifact_path = output_dir / "forward_control_v3_calibrated.pt"
    torch.save(
        {
            "version": config["version"],
            "seed": int(config["seed"]),
            "model": "blended_forward_control_v3",
            "blend_alpha": float(selected["alpha"]),
            "selection_rule": (
                "maximize validation reachable inverse success while "
                "strict success >= v2 and MAE <= v2"
            ),
            "legacy_component": legacy_artifact,
            "joint_component": joint_artifact,
        },
        artifact_path,
    )
    summary = {
        "version": config["version"],
        "artifact": str(artifact_path.resolve()),
        "validation_pairs": len(pairs),
        "validation_transitions": len(rows) * 81,
        "baseline_v2": baseline,
        "joint_v3": next(row for row in candidates if row["alpha"] == 1.0),
        "selected": selected,
        "candidates": candidates,
        "held_out_test_used": False,
    }
    summary_path = output_dir / "forward_control_v3_calibration.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
