#!/usr/bin/env python3
"""Report v10 train/development distribution and tolerance-boundary coverage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_structured_rebuild_v10.contracts import (
    SETUP_FIELDS,
    STATE_FIELDS,
    group_targets,
)

DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/physics_structured_rebuild_v10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def split_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = np.asarray([group_targets(row)[1] for row in rows])
    boundary = np.abs(np.abs(normalized) - 1.0) <= 0.20
    cardinality = Counter()
    interaction = Counter()
    visual_conditions = Counter()
    clipping = []
    captured = []
    camera_distance = []
    focus = []
    for row in rows:
        focus.append(float(row["group_auxiliary"]["geometric_focus_residual"]))
        for candidate in row["candidates"]:
            auxiliary = candidate["auxiliary"]
            cardinality[int(auxiliary["action_cardinality"])] += 1
            interaction[str(auxiliary["interaction_category"])] += 1
            clipping.append(float(auxiliary["clipping_fraction"]))
            captured.append(float(auxiliary["captured_optical_power"]))
            camera_distance.append(
                float(auxiliary["distance_to_camera_boundary_px"])
            )
        visual_conditions.update(
            request["condition"] for request in row["visual_requests"]
        )
    setup_values = {
        field: quantiles(
            np.asarray([float(row["setup"][field]) for row in rows])
        )
        for field in SETUP_FIELDS
    }
    return {
        "groups": len(rows),
        "transitions": 81 * len(rows),
        "regime_counts": dict(
            sorted(Counter(row["regime"] for row in rows).items())
        ),
        "setup_quantiles": setup_values,
        "action_cardinality_counts": {
            str(key): value for key, value in sorted(cardinality.items())
        },
        "interaction_category_counts": dict(sorted(interaction.items())),
        "visual_condition_counts": dict(sorted(visual_conditions.items())),
        "near_direction_tolerance_boundary_rate": {
            field: float(boundary[:, :, index].mean())
            for index, field in enumerate(STATE_FIELDS)
        },
        "groups_with_any_near_tolerance_transition": int(
            boundary.any(axis=(1, 2)).sum()
        ),
        "auxiliary_quantiles": {
            "clipping_fraction": quantiles(np.asarray(clipping)),
            "captured_optical_power": quantiles(np.asarray(captured)),
            "distance_to_camera_boundary_px": quantiles(
                np.asarray(camera_distance)
            ),
            "geometric_focus_residual": quantiles(np.asarray(focus)),
        },
        "route_case_counts": {
            "state_forward": len(rows),
            "image_forward": 3 * len(rows),
            "state_direction": len(rows),
            "image_direction": 3 * len(rows),
            "numerical_inverse": 4 * len(rows),
            "visual_inverse": 3 * len(rows),
        },
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    manifest = json.loads(
        (data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    reports = {
        split: split_report(
            read_jsonl(data_dir / "grids" / f"{split}.jsonl")
        )
        for split in ("train", "development")
    }
    result = {
        "version": "physics_structured_rebuild_v10_distribution_coverage",
        "splits_opened": ["train", "development"],
        "locked_test_labels_opened": False,
        "reports": reports,
        "cross_split_setup_hash_overlap": manifest[
            "cross_split_setup_hash_overlap"
        ],
        "cross_split_context_hash_overlap": manifest[
            "cross_split_context_hash_overlap"
        ],
        "complete": True,
    }
    output = data_dir / "distribution_coverage_pre_freeze.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite coverage report: {output}")
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

