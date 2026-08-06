#!/usr/bin/env python3
"""Evaluate a frozen state-direction adapter on fixed system validation."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from control_rebuild_v4.evaluate_orchestrated_system import (
    simulator_forward_truth,
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
    DEFAULT_DUAL_SOURCE_FORWARD_STATE,
    DEFAULT_EXTRA_FORWARD_DIRECTION_STATE_HYBRID,
    HybridDirectionRuntimeV9,
)
from physics_structured_rebuild_v9.train_qwen_state_direction_adapter import (
    apply_selection,
    augmented,
    metrics,
    sha256,
)
from specialist_rebuild_v2.common import DIRECTION_FIELDS, forward_feature

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_QWEN = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_ARTIFACT = DEFAULT_RUN / "qwen_state_direction_adapter_v9.pkl"
DEFAULT_OUTPUT = DEFAULT_RUN / "qwen_state_direction_adapter_validation.json"
ROUTE = "predict_direction_from_state_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    artifact_path = args.artifact.resolve()
    with artifact_path.open("rb") as stream:
        artifact = pickle.load(stream)
    if artifact.get("model") != "qwen_state_direction_hgb_correction_gate_v9":
        raise ValueError("unexpected state direction adapter")
    base_path = DEFAULT_EXTRA_FORWARD_DIRECTION_STATE_HYBRID.resolve()
    if (
        Path(str(artifact["base_direction_artifact"])).resolve() != base_path
        or artifact["base_direction_artifact_sha256"] != sha256(base_path)
    ):
        raise ValueError("state direction adapter base differs")
    qwen_data = args.qwen_data.resolve()
    canonical = [
        row
        for row in read_jsonl(qwen_data / "canonical/val.jsonl")
        if row["target_decision"].get("route_name") == ROUTE
    ]
    if len(canonical) != 150:
        raise ValueError("expected 150 state direction requests")
    private_by_group = {
        str(row["group_id"]): row
        for row in read_jsonl(qwen_data / "private/source_cases/val.jsonl")
    }
    torch, device = configure(int(args.seed), args.device)
    base_direction = HybridDirectionRuntimeV9(torch, base_path, device)
    base_forward, _ = load_residual_forward_runtime_v9(
        DEFAULT_DUAL_SOURCE_FORWARD_STATE,
        torch,
        device,
    )
    rows = [
        {
            "group_id": str(row["example_id"]),
            "setup": row["target_decision"]["arguments"]["setup"],
            "current_beam_state": row["target_decision"]["arguments"][
                "current_beam_state"
            ],
        }
        for row in canonical
    ]
    forward_grid = base_forward.predict_changes(rows)
    class_index = {str(value): index for index, value in enumerate(CLASSES)}
    base_labels = np.empty((len(canonical), 5), dtype=np.int8)
    targets = np.empty_like(base_labels)
    features = []
    for index, row in enumerate(canonical):
        arguments = row["target_decision"]["arguments"]
        base_result = base_direction.predict_one(
            arguments["setup"],
            arguments["current_beam_state"],
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
                        arguments["current_beam_state"],
                        arguments["action"],
                    ),
                    forward_prediction,
                ]
            )
        )
        private = private_by_group[str(row["group_id"])]
        truth = simulator_forward_truth(private, arguments["action"])
        targets[index] = [
            class_index[str(truth["directions"][field])]
            for field in DIRECTION_FIELDS
        ]
    model_features = augmented(
        np.asarray(features, dtype=np.float32),
        base_labels,
    )
    probabilities = np.zeros((len(canonical), 5, 3), dtype=np.float32)
    positions = np.arange(len(canonical))
    for field, model in enumerate(artifact["models"]):
        classes = np.asarray(model.classes_, dtype=np.int64)
        probabilities[
            positions[:, None],
            field,
            classes[None, :],
        ] = model.predict_proba(model_features)
    candidate = apply_selection(
        base_labels,
        probabilities,
        list(artifact["selection"]),
    )
    report = {
        "version": "qwen_state_direction_adapter_validation_v9_one_seed",
        "baseline": metrics(base_labels, targets),
        "candidate": metrics(candidate, targets),
        "changed_field_count": int(np.sum(candidate != base_labels)),
        "changed_request_count": int(
            np.any(candidate != base_labels, axis=1).sum()
        ),
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "source_contract": {
            "system_validation_used_for_training": False,
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
