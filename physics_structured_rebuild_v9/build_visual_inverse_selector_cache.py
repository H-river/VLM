#!/usr/bin/env python3
"""Build leakage-free train and validation features for visual inverse selection."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v3.train_forward import configure
from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_residual_forward_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_NATURAL_GRID_FORWARD_STATE,
)
from physics_structured_rebuild_v9.visual_inverse_selector import (
    visual_selector_features,
)

DEFAULT_QWEN = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_OVERLAY = (
    REPO_ROOT.parent
    / "VLM_runs/direction_rebuild_v4_tree_one_seed"
    / "candidate_overlay_direction_v4_manifest.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "visual_inverse_selector_cache_v9.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--v4-overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument(
        "--natural-forward",
        type=Path,
        default=DEFAULT_NATURAL_GRID_FORWARD_STATE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role_path(
    data_dir: Path,
    row: dict[str, Any],
    role: str,
) -> Path:
    image_name = str(row["target_decision"]["image_roles"][role])
    index = int(image_name.removeprefix("image_"))
    return (data_dir / str(row["images"][index])).resolve()


def build_split(
    split: str,
    data_dir: Path,
    visual: Any,
    natural_forward: Any,
) -> dict[str, np.ndarray]:
    canonical = [
        row
        for row in read_jsonl(data_dir / f"canonical/{split}.jsonl")
        if row["target_decision"].get("route_name")
        == "select_inverse_action_from_images_v1"
    ]
    private_by_group = {
        str(row["group_id"]): row
        for row in read_jsonl(
            data_dir / f"private/source_cases/{split}.jsonl"
        )
    }
    private_rows = [private_by_group[str(row["group_id"])] for row in canonical]
    setups = [row["target_decision"]["arguments"]["setup"] for row in canonical]
    current_sensor: list[np.ndarray] = []
    desired_sensor: list[np.ndarray] = []
    for index, row in enumerate(canonical):
        calibration = row["target_decision"]["arguments"]["image_calibration"]
        current_sensor.append(
            visual.measure_image(
                role_path(data_dir, row, "current_beam"),
                calibration,
            )["beam_state"]
        )
        desired_sensor.append(
            visual.measure_image(
                role_path(data_dir, row, "desired_beam"),
                calibration,
            )["beam_state"]
        )
        if (index + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "split": split,
                        "measured": index + 1,
                        "count": len(canonical),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    current = np.asarray(current_sensor, dtype=np.float32)
    desired = np.asarray(desired_sensor, dtype=np.float32)
    original_forward = visual.forward
    ids = [str(row["example_id"]) for row in canonical]
    visual.forward = natural_forward
    primary = visual.predict_from_states(
        setups,
        current,
        desired,
        group_ids=ids,
    )
    visual.forward = original_forward
    secondary = visual.predict_from_states(
        setups,
        current,
        desired,
        group_ids=ids,
    )
    visual.forward = original_forward
    primary_selected = np.asarray(
        primary["selected_indices"],
        dtype=np.int64,
    )
    secondary_selected = np.asarray(
        secondary["selected_indices"],
        dtype=np.int64,
    )
    primary_success: list[bool] = []
    secondary_success: list[bool] = []
    for index, private in enumerate(private_rows):
        primary_success.append(
            private_inverse_target_reached(
                private["setup"],
                ACTION_GRID[int(primary_selected[index])],
                private["desired_beam_state"],
            )
        )
        secondary_success.append(
            private_inverse_target_reached(
                private["setup"],
                ACTION_GRID[int(secondary_selected[index])],
                private["desired_beam_state"],
            )
        )
    primary_success_array = np.asarray(primary_success, dtype=np.bool_)
    secondary_success_array = np.asarray(secondary_success, dtype=np.bool_)
    print(
        json.dumps(
            {
                "split": split,
                "count": len(canonical),
                "primary_success": int(primary_success_array.sum()),
                "secondary_success": int(secondary_success_array.sum()),
                "pair_oracle": int(
                    (primary_success_array | secondary_success_array).sum()
                ),
                "disagreement": int(
                    (primary_selected != secondary_selected).sum()
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {
        "features": visual_selector_features(
            setups,
            current,
            desired,
            primary,
            secondary,
        ),
        "primary_success": primary_success_array,
        "secondary_success": secondary_success_array,
        "primary_selected": primary_selected,
        "secondary_selected": secondary_selected,
        "group_ids": np.asarray(
            [str(row["group_id"]) for row in canonical],
            dtype=np.str_,
        ),
        "gamma": np.asarray(
            [
                float(
                    row["target_decision"]["arguments"][
                        "image_calibration"
                    ]["gamma"]
                )
                for row in canonical
            ],
            dtype=np.float32,
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    data_dir = args.qwen_data.resolve()
    torch, device = configure(int(args.seed), args.device)
    backend = OrchestratedSpecialistRuntimeV4.from_manifest(
        torch,
        args.v4_overlay.resolve(),
        device,
    )
    natural_forward, _ = load_residual_forward_runtime_v9(
        args.natural_forward.resolve(),
        torch,
        device,
    )
    split_arrays = {
        split: build_split(
            split,
            data_dir,
            backend.visual,
            natural_forward,
        )
        for split in ("train", "val")
    }
    flat = {
        f"{split}_{key}": value
        for split, arrays in split_arrays.items()
        for key, value in arrays.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **flat)
    report = {
        "version": "visual_inverse_selector_cache_v9_one_seed",
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "feature_count": int(split_arrays["train"]["features"].shape[1]),
        "splits": {
            split: {
                "count": int(len(arrays["features"])),
                "primary_success_count": int(
                    arrays["primary_success"].sum()
                ),
                "secondary_success_count": int(
                    arrays["secondary_success"].sum()
                ),
                "pair_oracle_success_count": int(
                    (
                        arrays["primary_success"]
                        | arrays["secondary_success"]
                    ).sum()
                ),
                "action_disagreement_count": int(
                    (
                        arrays["primary_selected"]
                        != arrays["secondary_selected"]
                    ).sum()
                ),
            }
            for split, arrays in split_arrays.items()
        },
        "source_contract": {
            "qwen_train": str(
                (data_dir / "canonical/train.jsonl").resolve()
            ),
            "qwen_validation": str(
                (data_dir / "canonical/val.jsonl").resolve()
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

