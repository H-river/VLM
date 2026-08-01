#!/usr/bin/env python3
"""Build natural-request direction labels and retained-model predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from control_rebuild_v3.visual_inverse import sensor_to_base_legacy
from control_rebuild_v4.inverse_data import state_mapping
from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from direction_rebuild_v4.data import CLASSES
from physics_structured_rebuild_v9.calibrate_system_direction import read_jsonl
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_EXTRA_FORWARD_DIRECTION_IMAGE_HYBRID,
    DEFAULT_EXTRA_FORWARD_DIRECTION_STATE_HYBRID,
    HybridDirectionRuntimeV9,
)
from specialist_rebuild_v2.common import DIRECTION_FIELDS

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_DATA = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_adaptation/train.jsonl"
)
DEFAULT_SOURCE = (
    REPO_ROOT.parent
    / "VLM_data/qwen_orchestration/v1/private/source_cases/train.jsonl"
)
DEFAULT_QWEN = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_FORWARD_FEATURES = DEFAULT_RUN / "qwen_forward_adapter_features.npz"
DEFAULT_OVERLAY = (
    REPO_ROOT.parent
    / "VLM_runs/direction_rebuild_v4_tree_one_seed"
    / "candidate_overlay_direction_v4_manifest.json"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "qwen_direction_adapter_features.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument(
        "--forward-features",
        type=Path,
        default=DEFAULT_FORWARD_FEATURES,
    )
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
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


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    summary_path = output.with_suffix(".json")
    if output.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    data_path = args.data.resolve()
    rows = read_jsonl(data_path)
    source_path = args.source.resolve()
    source_by_group = {
        str(row["group_id"]): row for row in read_jsonl(source_path)
    }
    if len(rows) != 1000 or len(source_by_group) != 1000:
        raise ValueError("expected 1,000 natural-request training groups")
    forward_path = args.forward_features.resolve()
    forward = np.load(forward_path, allow_pickle=False)
    group_ids = [str(value) for value in forward["group_ids"]]
    if group_ids != [str(row["group_id"]) for row in rows]:
        raise ValueError("forward feature order differs from adaptation rows")
    class_index = {str(value): index for index, value in enumerate(CLASSES)}
    labels = np.asarray(
        [
            [
                class_index[str(row["truth_directions"][field])]
                for field in DIRECTION_FIELDS
            ]
            for row in rows
        ],
        dtype=np.int8,
    )
    perturbations = np.asarray(
        [
            str(source_by_group[group_id]["perturbation_family"])
            for group_id in group_ids
        ],
        dtype=np.str_,
    )

    torch, device = configure(int(args.seed), args.device)
    measurement = OrchestratedSpecialistRuntimeV4.from_manifest(
        torch,
        args.overlay.resolve(),
        device,
    )
    image_currents = []
    qwen_data = args.qwen_data.resolve()
    for row in rows:
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

    route_currents = {
        "state": [row["current_beam_state"] for row in rows],
        "image": image_currents,
    }
    artifacts = {
        "state": DEFAULT_EXTRA_FORWARD_DIRECTION_STATE_HYBRID,
        "image": DEFAULT_EXTRA_FORWARD_DIRECTION_IMAGE_HYBRID,
    }
    cache: dict[str, np.ndarray] = {
        "group_ids": np.asarray(group_ids, dtype=np.str_),
        "labels": labels,
        "perturbations": perturbations,
    }
    baseline = {}
    for route in ("state", "image"):
        model = HybridDirectionRuntimeV9(
            torch,
            artifacts[route],
            device,
        )
        predictions = np.empty_like(labels)
        for index, row in enumerate(rows):
            result = model.predict_one(
                row["setup"],
                route_currents[route][index],
                row["action"],
            )
            predictions[index] = [
                class_index[str(result["directions"][field])]
                for field in DIRECTION_FIELDS
            ]
            if (index + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "route": route,
                            "predicted": index + 1,
                            "count": len(rows),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        exact = np.all(predictions == labels, axis=1)
        cache[f"{route}_features"] = np.asarray(
            forward[f"{route}_features"],
            dtype=np.float32,
        )
        cache[f"{route}_base_labels"] = predictions
        baseline[route] = {
            "count": len(exact),
            "all_five_exact_count": int(exact.sum()),
            "all_five_exact": float(exact.mean()),
            "per_field_accuracy": {
                field: float((predictions[:, index] == labels[:, index]).mean())
                for index, field in enumerate(DIRECTION_FIELDS)
            },
        }
        del model

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **cache)
    summary = {
        "version": "qwen_direction_adapter_features_v9",
        "count": len(rows),
        "feature_count": int(cache["state_features"].shape[1]),
        "baseline": baseline,
        "cache": str(output),
        "cache_sha256": sha256(output),
        "source_contract": {
            "adaptation_data": str(data_path),
            "adaptation_data_sha256": sha256(data_path),
            "forward_features": str(forward_path),
            "forward_features_sha256": sha256(forward_path),
            "state_direction": str(artifacts["state"].resolve()),
            "image_direction": str(artifacts["image"].resolve()),
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
