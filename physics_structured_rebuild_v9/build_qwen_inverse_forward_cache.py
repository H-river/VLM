#!/usr/bin/env python3
"""Enumerate natural inverse requests with both frozen forward candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.forward_runtime import (
    load_residual_forward_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_EXTRA_TREES_FORWARD_STATE,
    DEFAULT_RESIDUAL_FORWARD_STATE,
)

DEFAULT_DATA = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9"
    / "qwen_inverse_adaptation/train.npz"
)
DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "qwen_inverse_forward_cache.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--primary-forward",
        type=Path,
        default=DEFAULT_RESIDUAL_FORWARD_STATE,
    )
    parser.add_argument(
        "--secondary-forward",
        type=Path,
        default=DEFAULT_EXTRA_TREES_FORWARD_STATE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_from_context(
    contexts: np.ndarray,
    group_ids: np.ndarray,
) -> list[dict[str, Any]]:
    from specialist_rebuild_v2.common import SETUP_FIELDS, STATE_FIELDS

    rows = []
    for index, context in enumerate(contexts):
        setup = {
            field: float(context[position])
            for position, field in enumerate(SETUP_FIELDS)
        }
        current_values = np.asarray(context[12:17], dtype=np.float64).copy()
        current_values[-1] = math.expm1(float(current_values[-1]))
        current = {
            field: float(current_values[position])
            for position, field in enumerate(STATE_FIELDS)
        }
        rows.append(
            {
                "group_id": str(group_ids[index]),
                "setup": setup,
                "current_beam_state": current,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    summary_path = output.with_suffix(".json")
    if output.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    data_path = args.data.resolve()
    arrays = np.load(data_path, allow_pickle=False)
    contexts = np.asarray(arrays["contexts"], dtype=np.float32)
    group_ids = np.asarray(arrays["group_ids"])
    count = len(contexts)
    primary_states = np.empty((count, 81, 5), dtype=np.float32)
    secondary_states = np.empty_like(primary_states)

    torch, device = configure(int(args.seed), args.device)
    primary_path = args.primary_forward.resolve()
    secondary_path = args.secondary_forward.resolve()
    primary, _ = load_residual_forward_runtime_v9(
        primary_path,
        torch,
        device,
    )
    secondary, _ = load_residual_forward_runtime_v9(
        secondary_path,
        torch,
        device,
    )
    for start in range(0, count, int(args.chunk_size)):
        stop = min(start + int(args.chunk_size), count)
        rows = rows_from_context(
            contexts[start:stop],
            group_ids[start:stop],
        )
        primary_states[start:stop] = primary.predict_states(rows)
        secondary_states[start:stop] = secondary.predict_states(rows)
        print(
            json.dumps(
                {"enumerated": stop, "count": count},
                sort_keys=True,
            ),
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        group_ids=group_ids,
        contexts=contexts,
        desired=np.asarray(arrays["desired"], dtype=np.float32),
        positives=np.asarray(arrays["positives"], dtype=np.bool_),
        primary_states=primary_states,
        secondary_states=secondary_states,
        true_candidate_states=np.asarray(
            arrays["candidate_states"],
            dtype=np.float32,
        ),
    )
    summary = {
        "version": "qwen_inverse_forward_cache_v9",
        "count": count,
        "cache": str(output),
        "cache_sha256": sha256(output),
        "source_contract": {
            "natural_inverse_data": str(data_path),
            "natural_inverse_data_sha256": sha256(data_path),
            "primary_forward": str(primary_path),
            "primary_forward_sha256": sha256(primary_path),
            "secondary_forward": str(secondary_path),
            "secondary_forward_sha256": sha256(secondary_path),
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
