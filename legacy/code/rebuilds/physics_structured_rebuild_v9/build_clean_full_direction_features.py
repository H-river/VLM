#!/usr/bin/env python3
"""Build derived direction features from all existing clean forward grids."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import labels_from_normalized_change, load_grid_arrays
from physics_structured_rebuild_v9.full_basis_forward_surface import (
    load_full_basis_forward_surface_runtime_v9,
)
from physics_structured_rebuild_v9.train_forward_tree_residual import sha256
from specialist_rebuild_v2.common import SETUP_FIELDS, STATE_FIELDS

DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
DEFAULT_FORWARD = DEFAULT_RUN / "full_basis_forward_surface_v9.pt"
DEFAULT_NATURAL = DEFAULT_RUN / "clean_nonoverlap_forward_training_features.npz"
DEFAULT_OUTPUT = DEFAULT_RUN / "clean_full_direction_features_v9.npz"
DEFAULT_GRIDS = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/train.jsonl",
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/train.jsonl",
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v5_numerical/grids/train.jsonl",
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/targeted_bundle/grids/train.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-train", type=Path, nargs="+", default=list(DEFAULT_GRIDS)
    )
    parser.add_argument("--natural-cache", type=Path, default=DEFAULT_NATURAL)
    parser.add_argument("--full-basis-forward", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def minimal_rows(path: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    ids = []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            group_id = str(raw["group_id"])
            ids.append(group_id)
            rows.append(
                {
                    "group_id": group_id,
                    "setup": raw["setup"],
                    "current_beam_state": raw["current_beam_state"],
                }
            )
    return np.asarray(ids, dtype=np.str_), rows


def rows_from_contexts(
    contexts: np.ndarray,
    ids: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for group_id, context in zip(ids, contexts, strict=True):
        current = np.asarray(context[12:17], dtype=np.float64).copy()
        current[-1] = math.expm1(float(current[-1]))
        rows.append(
            {
                "group_id": str(group_id),
                "setup": {
                    field: float(context[index])
                    for index, field in enumerate(SETUP_FIELDS)
                },
                "current_beam_state": {
                    field: float(current[index])
                    for index, field in enumerate(STATE_FIELDS)
                },
            }
        )
    return rows


def predictions(
    runtime: object,
    rows: list[dict[str, object]],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    prior_parts = []
    candidate_parts = []
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        prior, correction = runtime.predict_correction(chunk)
        candidate = (
            prior
            + correction
            * runtime.field_blend[None, None, :]
        )
        prior_parts.append(prior.astype(np.float32))
        candidate_parts.append(candidate.astype(np.float32))
    return np.concatenate(prior_parts), np.concatenate(candidate_parts)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    torch, device = configure(int(args.seed), args.device)
    runtime, _ = load_full_basis_forward_surface_runtime_v9(
        args.full_basis_forward.resolve(), torch, device
    )
    feature_parts = []
    label_parts = []
    normalized_parts = []
    id_parts = []
    source_report = []
    for raw_path in args.grid_train:
        path = raw_path.resolve()
        arrays = load_grid_arrays(path, include_legacy_features=False)
        ids, rows = minimal_rows(path)
        if len(ids) != arrays.group_count:
            raise ValueError("direction source group count differs")
        prior, candidate = predictions(
            runtime, rows, int(args.chunk_size)
        )
        features = np.concatenate(
            [
                arrays.features,
                prior.reshape(-1, 5),
                candidate.reshape(-1, 5),
            ],
            axis=1,
        ).astype(np.float32)
        feature_parts.append(features)
        label_parts.append(arrays.labels.astype(np.int8))
        normalized_parts.append(arrays.normalized_changes.astype(np.float32))
        id_parts.append(ids)
        source_report.append(
            {"path": str(path), "group_count": int(arrays.group_count)}
        )
    natural_path = args.natural_cache.resolve()
    with np.load(natural_path, allow_pickle=False) as natural:
        ids = np.asarray(natural["group_ids"], dtype=np.str_)
        flat = np.asarray(natural["grid_features"], dtype=np.float32)
        normalized = np.asarray(
            natural["grid_target_normalized"], dtype=np.float32
        )
        contexts = flat.reshape(len(ids), 81, -1)[:, 40, :17]
        rows = rows_from_contexts(contexts, ids)
        prior, candidate = predictions(
            runtime, rows, int(args.chunk_size)
        )
        feature_parts.append(
            np.concatenate(
                [flat[:, :46], prior.reshape(-1, 5), candidate.reshape(-1, 5)],
                axis=1,
            ).astype(np.float32)
        )
        label_parts.append(
            labels_from_normalized_change(normalized).astype(np.int8)
        )
        normalized_parts.append(normalized)
        id_parts.append(ids)
    source_report.append(
        {"path": str(natural_path), "group_count": int(len(ids))}
    )
    group_ids = np.concatenate(id_parts)
    if len(set(map(str, group_ids))) != len(group_ids):
        raise ValueError("clean direction group identifiers overlap")
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    normalized = np.concatenate(normalized_parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=features,
        labels=labels,
        normalized_changes=normalized,
        group_ids=group_ids,
    )
    report = {
        "version": "clean_full_direction_features_v9",
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "feature_count": int(features.shape[1]),
        "group_count": int(len(group_ids)),
        "transition_count": int(len(features)),
        "sources": source_report,
        "source_contract": {
            "generated_setups": 0,
            "generated_images": 0,
            "protected_validation_used": False,
            "system_validation_used": False,
            "held_out_test_files_opened": [],
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
