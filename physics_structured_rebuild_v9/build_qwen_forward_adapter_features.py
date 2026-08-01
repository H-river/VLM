#!/usr/bin/env python3
"""Build state/image features for natural-request forward adaptation."""

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

from control_rebuild_v3.common import tolerance_from_current
from control_rebuild_v3.train_forward import configure
from control_rebuild_v3.visual_inverse import sensor_to_base_legacy
from control_rebuild_v4.inverse_data import state_mapping
from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from physics_structured_rebuild_v9.calibrate_system_direction import (
    action_index,
    read_jsonl,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_residual_forward_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_DUAL_SOURCE_FORWARD_IMAGE,
    DEFAULT_DUAL_SOURCE_FORWARD_STATE,
)
from specialist_rebuild_v2.common import STATE_FIELDS, forward_feature

DEFAULT_DATA = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_adaptation/train.jsonl"
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
    / "qwen_forward_adapter_features.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument(
        "--state-forward",
        type=Path,
        default=DEFAULT_DUAL_SOURCE_FORWARD_STATE,
    )
    parser.add_argument(
        "--image-forward",
        type=Path,
        default=DEFAULT_DUAL_SOURCE_FORWARD_IMAGE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_predictions(
    model: Any,
    rows: list[dict[str, Any]],
    actions: list[dict[str, float]],
) -> np.ndarray:
    grid = model.predict_changes(rows)
    return np.stack(
        [
            grid[index, action_index(action)]
            for index, action in enumerate(actions)
        ]
    ).astype(np.float32)


def route_arrays(
    source: list[dict[str, Any]],
    currents: list[dict[str, float]],
    base_prediction: np.ndarray,
) -> dict[str, np.ndarray]:
    actions = [row["action"] for row in source]
    features = np.stack(
        [
            forward_feature(row["setup"], currents[index], actions[index])
            for index, row in enumerate(source)
        ]
    ).astype(np.float32)
    model_features = np.concatenate(
        [features, base_prediction],
        axis=1,
    ).astype(np.float32)
    truth_change = np.asarray(
        [
            [float(row["truth_change"][field]) for field in STATE_FIELDS]
            for row in source
        ],
        dtype=np.float32,
    )
    input_tolerance = np.stack(
        [tolerance_from_current(current) for current in currents]
    ).astype(np.float32)
    scoring_tolerance = np.stack(
        [
            tolerance_from_current(row["current_beam_state"])
            for row in source
        ]
    ).astype(np.float32)
    target_normalized = truth_change / input_tolerance
    residual_target = target_normalized - base_prediction
    return {
        "features": model_features,
        "base_prediction": base_prediction,
        "target_normalized": target_normalized,
        "residual_target": residual_target,
        "truth_change": truth_change,
        "input_tolerance": input_tolerance,
        "scoring_tolerance": scoring_tolerance,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    summary_path = output.with_suffix(".json")
    if output.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    data_path = args.data.resolve()
    source = read_jsonl(data_path)
    if len(source) != 1000:
        raise ValueError("expected 1,000 adaptation records")
    qwen_data = args.qwen_data.resolve()
    torch, device = configure(int(args.seed), args.device)

    measurement = OrchestratedSpecialistRuntimeV4.from_manifest(
        torch,
        args.overlay.resolve(),
        device,
    )
    image_currents = []
    for index, row in enumerate(source):
        measured = measurement.visual.measure_image(
            qwen_data / str(row["images"][0]),
            row["image_calibration"],
        )
        image_currents.append(
            state_mapping(
                sensor_to_base_legacy(
                    measured["beam_state"],
                    [row["setup"]],
                )[0]
            )
        )
        if (index + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "measured_images": index + 1,
                        "count": len(source),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    actions = [row["action"] for row in source]
    state_currents = [row["current_beam_state"] for row in source]
    state_rows = [
        {
            "group_id": str(row["group_id"]),
            "setup": row["setup"],
            "current_beam_state": state_currents[index],
        }
        for index, row in enumerate(source)
    ]
    state_model, _ = load_residual_forward_runtime_v9(
        args.state_forward.resolve(),
        torch,
        device,
    )
    state_prediction = selected_predictions(state_model, state_rows, actions)
    del state_model

    image_rows = [
        {
            "group_id": str(row["group_id"]),
            "setup": row["setup"],
            "current_beam_state": image_currents[index],
        }
        for index, row in enumerate(source)
    ]
    image_model, _ = load_residual_forward_runtime_v9(
        args.image_forward.resolve(),
        torch,
        device,
    )
    image_prediction = selected_predictions(image_model, image_rows, actions)
    del image_model

    route_data = {
        "state": route_arrays(
            source,
            state_currents,
            state_prediction,
        ),
        "image": route_arrays(
            source,
            image_currents,
            image_prediction,
        ),
    }
    cache: dict[str, np.ndarray] = {
        "group_ids": np.asarray(
            [str(row["group_id"]) for row in source],
            dtype=np.str_,
        )
    }
    baseline = {}
    for route, arrays in route_data.items():
        for name, values in arrays.items():
            cache[f"{route}_{name}"] = values
        error = (
            np.abs(
                arrays["base_prediction"] * arrays["input_tolerance"]
                - arrays["truth_change"]
            )
            / arrays["scoring_tolerance"]
        )
        exact = np.all(error <= 1.0, axis=1)
        baseline[route] = {
            "count": len(exact),
            "strict_all_five_count": int(exact.sum()),
            "strict_all_five_success": float(exact.mean()),
            "mae_in_scoring_tolerance_units": float(error.mean()),
            "per_field_tolerance_pass": {
                field: float((error[:, index] <= 1.0).mean())
                for index, field in enumerate(STATE_FIELDS)
            },
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **cache)
    summary = {
        "version": "qwen_forward_adapter_features_v9",
        "count": len(source),
        "feature_count": int(route_data["state"]["features"].shape[1]),
        "baseline": baseline,
        "cache": str(output),
        "cache_sha256": sha256(output),
        "source_contract": {
            "adaptation_data": str(data_path),
            "adaptation_data_sha256": sha256(data_path),
            "state_forward": str(args.state_forward.resolve()),
            "image_forward": str(args.image_forward.resolve()),
            "validation_files_opened": [],
            "held_out_test_files_opened": [],
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
