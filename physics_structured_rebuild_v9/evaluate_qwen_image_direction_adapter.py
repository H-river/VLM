#!/usr/bin/env python3
"""Evaluate a frozen image-direction adapter on the fixed system validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from control_rebuild_v3.visual_inverse import sensor_to_base_legacy
from control_rebuild_v4.evaluate_orchestrated_system import (
    simulator_forward_truth,
)
from control_rebuild_v4.inverse_data import state_mapping
from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from direction_rebuild_v4.data import CLASSES
from physics_structured_rebuild_v9.calibrate_system_direction import (
    action_index,
    read_jsonl,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_residual_forward_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_DUAL_SOURCE_FORWARD_IMAGE,
    DEFAULT_EXTRA_FORWARD_DIRECTION_IMAGE_HYBRID,
    HybridDirectionRuntimeV9,
)
from specialist_rebuild_v2.common import DIRECTION_FIELDS, forward_feature

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_QWEN = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_OVERLAY = (
    REPO_ROOT.parent
    / "VLM_runs/direction_rebuild_v4_tree_one_seed"
    / "candidate_overlay_direction_v4_manifest.json"
)
DEFAULT_ARTIFACT = (
    DEFAULT_RUN
    / "qwen_clean_image_direction_adapter/direction_image_adapter.pkl"
)
DEFAULT_OUTPUT = (
    DEFAULT_RUN / "qwen_clean_image_direction_adapter_validation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
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


def metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    correct = predicted == target
    exact = np.all(correct, axis=1)
    return {
        "count": len(exact),
        "all_five_exact_count": int(exact.sum()),
        "all_five_exact": float(exact.mean()),
        "per_field_accuracy": {
            field: float(correct[:, index].mean())
            for index, field in enumerate(DIRECTION_FIELDS)
        },
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    artifact_path = args.artifact.resolve()
    with artifact_path.open("rb") as stream:
        artifact = pickle.load(stream)
    if artifact.get("model") != "qwen_distribution_direction_correction_gate_v9":
        raise ValueError("unexpected direction adapter artifact")
    base_direction_path = DEFAULT_EXTRA_FORWARD_DIRECTION_IMAGE_HYBRID.resolve()
    if (
        str(base_direction_path) != artifact["base_direction_artifact"]
        or sha256(base_direction_path)
        != artifact["base_direction_artifact_sha256"]
    ):
        raise ValueError("direction adapter base artifact differs")

    qwen_data = args.qwen_data.resolve()
    canonical = [
        row
        for row in read_jsonl(qwen_data / "canonical/val.jsonl")
        if str(row["category"]) == "predict_direction_from_image_v1"
    ]
    if len(canonical) != 150:
        raise ValueError("expected 150 image-direction validation requests")
    private_by_group = {
        str(row["group_id"]): row
        for row in read_jsonl(qwen_data / "private/source_cases/val.jsonl")
    }
    torch, device = configure(int(args.seed), args.device)
    measurement = OrchestratedSpecialistRuntimeV4.from_manifest(
        torch,
        args.overlay.resolve(),
        device,
    )
    base_direction = HybridDirectionRuntimeV9(
        torch,
        base_direction_path,
        device,
    )
    base_forward, _ = load_residual_forward_runtime_v9(
        DEFAULT_DUAL_SOURCE_FORWARD_IMAGE,
        torch,
        device,
    )
    currents = []
    prediction_rows = []
    for row in canonical:
        arguments = row["target_decision"]["arguments"]
        measured = measurement.visual.measure_image(
            qwen_data / str(row["images"][0]),
            arguments["image_calibration"],
        )
        current = state_mapping(
            sensor_to_base_legacy(
                measured["beam_state"],
                [arguments["setup"]],
            )[0]
        )
        currents.append(current)
        prediction_rows.append(
            {
                "group_id": str(row["example_id"]),
                "setup": arguments["setup"],
                "current_beam_state": current,
            }
        )
    forward_grid = base_forward.predict_changes(prediction_rows)
    class_index = {str(value): index for index, value in enumerate(CLASSES)}
    base_labels = np.empty((len(canonical), len(DIRECTION_FIELDS)), dtype=np.int8)
    features = []
    targets = np.empty_like(base_labels)
    for index, row in enumerate(canonical):
        arguments = row["target_decision"]["arguments"]
        base_result = base_direction.predict_one(
            arguments["setup"],
            currents[index],
            arguments["action"],
        )
        base_labels[index] = [
            class_index[str(base_result["directions"][field])]
            for field in DIRECTION_FIELDS
        ]
        forward_prediction = forward_grid[
            index,
            action_index(arguments["action"]),
        ]
        features.append(
            np.concatenate(
                [
                    forward_feature(
                        arguments["setup"],
                        currents[index],
                        arguments["action"],
                    ),
                    forward_prediction,
                    np.eye(3, dtype=np.float32)[base_labels[index]].reshape(-1),
                ]
            )
        )
        private = private_by_group[str(row["group_id"])]
        truth = simulator_forward_truth(private, arguments["action"])
        targets[index] = [
            class_index[str(truth["directions"][field])]
            for field in DIRECTION_FIELDS
        ]
        if (index + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "evaluated": index + 1,
                        "count": len(canonical),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    feature_array = np.asarray(features, dtype=np.float32)
    probabilities = np.zeros(
        (len(canonical), len(DIRECTION_FIELDS), 3),
        dtype=np.float32,
    )
    row_indices = np.arange(len(canonical))
    for field, model in enumerate(artifact["models"]):
        classes = np.asarray(model.classes_, dtype=np.int64)
        probabilities[
            row_indices[:, None],
            field,
            classes[None, :],
        ] = model.predict_proba(feature_array)
    learned = probabilities.argmax(axis=2)
    confidence = probabilities.max(axis=2)
    thresholds = np.asarray(
        [
            float(artifact["confidence_threshold"][field])
            for field in DIRECTION_FIELDS
        ],
        dtype=np.float32,
    )
    candidate = np.where(
        confidence >= thresholds[None, :],
        learned,
        base_labels,
    ).astype(np.int8)
    report = {
        "version": "qwen_clean_image_direction_adapter_validation_v9_one_seed",
        "baseline": metrics(base_labels, targets),
        "candidate": metrics(candidate, targets),
        "changed_field_count": int(np.sum(candidate != base_labels)),
        "changed_request_count": int(
            np.sum(np.any(candidate != base_labels, axis=1))
        ),
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "source_contract": {
            "qwen_validation": str(
                (qwen_data / "canonical/val.jsonl").resolve()
            ),
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
