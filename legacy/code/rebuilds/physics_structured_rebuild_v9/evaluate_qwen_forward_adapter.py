#!/usr/bin/env python3
"""Evaluate frozen natural-request forward adapters on the fixed system cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.visual_inverse import sensor_to_base_legacy
from control_rebuild_v4.inverse_data import state_mapping
from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.calibrate_system_direction import read_jsonl
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_DUAL_SOURCE_FORWARD_IMAGE,
    DEFAULT_DUAL_SOURCE_FORWARD_STATE,
)
from specialist_rebuild_v2.common import STATE_FIELDS, forward_feature

ROUTES = (
    "predict_forward_from_state_v1",
    "predict_forward_from_image_v1",
)
DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_QWEN = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_CACHE = (
    DEFAULT_RUN
    / "forward_dual_source_calibrated/forward_system_cache.npz"
)
DEFAULT_ADAPTER = DEFAULT_RUN / "qwen_forward_adapter"
DEFAULT_OVERLAY = (
    REPO_ROOT.parent
    / "VLM_runs/direction_rebuild_v4_tree_one_seed"
    / "candidate_overlay_direction_v4_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--system-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--state-adapter", type=Path)
    parser.add_argument("--image-adapter", type=Path)
    parser.add_argument("--disable-state-adapter", action="store_true")
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN / "qwen_forward_adapter_validation.json",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_predictions(cache: Any) -> np.ndarray:
    prior = np.asarray(cache["prior"], dtype=np.float32)
    primary = np.asarray(cache["residual"], dtype=np.float32)
    secondary = np.asarray(cache["secondary_residual"], dtype=np.float32)
    routes = np.asarray(cache["routes"])
    output = prior.copy()
    for route, base_path in zip(
        ROUTES,
        (
            DEFAULT_DUAL_SOURCE_FORWARD_STATE,
            DEFAULT_DUAL_SOURCE_FORWARD_IMAGE,
        ),
        strict=True,
    ):
        with base_path.resolve().open("rb") as stream:
            base_artifact = pickle.load(stream)
        mask = routes == route
        for field_index, field in enumerate(STATE_FIELDS):
            source = base_artifact["field_tree_source"][field]
            residual = primary if source == "primary" else secondary
            output[mask, field_index] = (
                prior[mask, field_index]
                + float(base_artifact["field_blend"][field])
                * residual[mask, field_index]
            )
    return output


def adapter_prediction(
    artifact: dict[str, Any],
    features: np.ndarray,
) -> np.ndarray:
    if artifact["estimator"] in {
        "random_forest_multioutput",
        "ridge_multioutput",
    }:
        return np.asarray(
            artifact["models"].predict(features),
            dtype=np.float32,
        )
    return np.stack(
        [model.predict(features) for model in artifact["models"]],
        axis=1,
    ).astype(np.float32)


def metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    input_tolerance: np.ndarray,
    scoring_tolerance: np.ndarray,
) -> dict[str, Any]:
    error = (
        np.abs(prediction * input_tolerance - truth) / scoring_tolerance
    )
    passed = error <= 1.0
    exact = np.all(passed, axis=1)
    return {
        "count": len(exact),
        "strict_all_five_count": int(exact.sum()),
        "strict_all_five_success": float(exact.mean()),
        "mae_in_scoring_tolerance_units": float(error.mean()),
        "per_field_tolerance_pass": {
            field: float(passed[:, index].mean())
            for index, field in enumerate(STATE_FIELDS)
        },
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    qwen_data = args.qwen_data.resolve()
    cache_path = args.system_cache.resolve()
    cache = np.load(cache_path, allow_pickle=False)
    routes = np.asarray(cache["routes"])
    canonical = [
        row
        for row in read_jsonl(qwen_data / "canonical/val.jsonl")
        if str(row["category"]) in ROUTES
    ]
    if len(canonical) != len(routes):
        raise ValueError("forward validation row count differs")
    if not np.array_equal(
        routes,
        np.asarray([str(row["category"]) for row in canonical]),
    ):
        raise ValueError("forward validation route order differs")

    torch, device = configure(int(args.seed), args.device)
    measurement = OrchestratedSpecialistRuntimeV4.from_manifest(
        torch,
        args.overlay.resolve(),
        device,
    )
    currents = []
    for row in canonical:
        arguments = row["target_decision"]["arguments"]
        if str(row["category"]) == ROUTES[0]:
            currents.append(arguments["current_beam_state"])
        else:
            measured = measurement.visual.measure_image(
                qwen_data / str(row["images"][0]),
                arguments["image_calibration"],
            )
            currents.append(
                state_mapping(
                    sensor_to_base_legacy(
                        measured["beam_state"],
                        [arguments["setup"]],
                    )[0]
                )
            )

    base = current_predictions(cache)
    candidate = base.copy()
    artifact_records = {}
    default_paths = {
        ROUTES[0]: args.adapter_dir.resolve() / "forward_state_adapter.pkl",
        ROUTES[1]: args.adapter_dir.resolve() / "forward_image_adapter.pkl",
    }
    if args.state_adapter is not None:
        default_paths[ROUTES[0]] = args.state_adapter.resolve()
    if args.image_adapter is not None:
        default_paths[ROUTES[1]] = args.image_adapter.resolve()
    for route in ROUTES:
        if route == ROUTES[0] and args.disable_state_adapter:
            continue
        artifact_path = default_paths[route]
        with artifact_path.open("rb") as stream:
            artifact = pickle.load(stream)
        mask = routes == route
        route_rows = np.flatnonzero(mask)
        features = np.stack(
            [
                np.concatenate(
                    [
                        forward_feature(
                            canonical[index]["target_decision"]["arguments"][
                                "setup"
                            ],
                            currents[index],
                            canonical[index]["target_decision"]["arguments"][
                                "action"
                            ],
                        ),
                        base[index],
                    ]
                )
                for index in route_rows
            ]
        ).astype(np.float32)
        residual = adapter_prediction(artifact, features)
        blend = np.asarray(
            [float(artifact["field_blend"][field]) for field in STATE_FIELDS],
            dtype=np.float32,
        )
        candidate[mask] = base[mask] + residual * blend[None, :]
        artifact_records[route] = {
            "path": str(artifact_path),
            "sha256": sha256(artifact_path),
        }

    truth = np.asarray(cache["truth_change"], dtype=np.float32)
    input_tolerance = np.asarray(cache["input_tolerance"], dtype=np.float32)
    scoring_tolerance = np.asarray(
        cache["scoring_tolerance"],
        dtype=np.float32,
    )
    report = {
        "version": "qwen_forward_adapter_validation_v9_one_seed",
        "combined": {
            "baseline": metrics(
                base,
                truth,
                input_tolerance,
                scoring_tolerance,
            ),
            "candidate": metrics(
                candidate,
                truth,
                input_tolerance,
                scoring_tolerance,
            ),
        },
        "by_route": {
            route: {
                "baseline": metrics(
                    base[routes == route],
                    truth[routes == route],
                    input_tolerance[routes == route],
                    scoring_tolerance[routes == route],
                ),
                "candidate": metrics(
                    candidate[routes == route],
                    truth[routes == route],
                    input_tolerance[routes == route],
                    scoring_tolerance[routes == route],
                ),
            }
            for route in ROUTES
        },
        "artifacts": artifact_records,
        "source_contract": {
            "system_cache": str(cache_path),
            "system_cache_sha256": sha256(cache_path),
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
