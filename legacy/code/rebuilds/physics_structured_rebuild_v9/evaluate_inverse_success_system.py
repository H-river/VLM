#!/usr/bin/env python3
"""Replay a protected inverse-success ranker on state-inverse validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.calibrate_system_direction import read_jsonl
from physics_structured_rebuild_v9.forward_runtime import (
    load_residual_forward_runtime_v9,
)
from physics_structured_rebuild_v9.inverse_runtime_v8 import (
    load_inverse_runtime_v8,
)
from physics_structured_rebuild_v9.inverse_success_runtime import (
    load_inverse_success_ranker_runtime_v9,
    sha256,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_COMBINED_NATURAL_INVERSE_V9,
    DEFAULT_NATURAL_GRID_FORWARD_STATE,
)
from Qwen_orchestration.scripts.evaluate_end_to_end import (
    private_inverse_target_reached,
)
from specialist_rebuild_v2.common import raw_state_array

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_QWEN = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_CANDIDATE = DEFAULT_RUN / "inverse_success_ranker_v9.pkl"
DEFAULT_OUTPUT = (
    DEFAULT_RUN / "inverse_success_ranker_system_validation.json"
)
ROUTE = "select_inverse_action_from_states_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_NATURAL_GRID_FORWARD_STATE)
    parser.add_argument("--base-artifact", type=Path, default=DEFAULT_COMBINED_NATURAL_INVERSE_V9)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    canonical = [
        row
        for row in read_jsonl(
            args.qwen_data.resolve() / "canonical/val.jsonl"
        )
        if row["target_decision"].get("route_name") == ROUTE
    ]
    if len(canonical) != 150:
        raise ValueError("expected 150 state-inverse validation requests")
    arguments = [row["target_decision"]["arguments"] for row in canonical]
    rows = [
        {
            "group_id": str(row["example_id"]),
            "setup": values["setup"],
            "current_beam_state": values["current_beam_state"],
        }
        for row, values in zip(canonical, arguments, strict=True)
    ]
    setups = [values["setup"] for values in arguments]
    current = np.asarray(
        [raw_state_array(values["current_beam_state"]) for values in arguments],
        dtype=np.float32,
    )
    desired = np.asarray(
        [raw_state_array(values["desired_beam_state"]) for values in arguments],
        dtype=np.float32,
    )

    torch, device = configure(int(args.seed), args.device)
    forward_path = args.forward_artifact.resolve()
    base_path = args.base_artifact.resolve()
    candidate_path = args.candidate.resolve()
    forward, _ = load_residual_forward_runtime_v9(
        forward_path,
        torch,
        device,
    )
    base, _ = load_inverse_runtime_v8(base_path, torch, device)
    candidate, artifact = load_inverse_success_ranker_runtime_v9(
        candidate_path,
        torch,
        device,
    )
    states = forward.predict_states(rows)
    base_result = base.score_requests(
        setups,
        current,
        desired,
        states,
    )
    candidate_result = candidate.score_requests(
        setups,
        current,
        desired,
        states,
    )
    base_actions = list(base_result["selected_actions"])
    candidate_actions = list(candidate_result["selected_actions"])
    base_success = np.asarray(
        [
            private_inverse_target_reached(
                setups[index],
                base_actions[index],
                arguments[index]["desired_beam_state"],
            )
            for index in range(len(arguments))
        ],
        dtype=np.bool_,
    )
    candidate_success = np.asarray(
        [
            private_inverse_target_reached(
                setups[index],
                candidate_actions[index],
                arguments[index]["desired_beam_state"],
            )
            for index in range(len(arguments))
        ],
        dtype=np.bool_,
    )
    changed = np.asarray(
        [
            base_actions[index] != candidate_actions[index]
            for index in range(len(arguments))
        ],
        dtype=np.bool_,
    )
    report: dict[str, Any] = {
        "version": "inverse_success_ranker_system_validation_v9_one_seed",
        "route": ROUTE,
        "count": int(len(arguments)),
        "baseline": {
            "success_count": int(base_success.sum()),
            "success_rate": float(base_success.mean()),
        },
        "candidate": {
            "success_count": int(candidate_success.sum()),
            "success_rate": float(candidate_success.mean()),
        },
        "changed_action_count": int(changed.sum()),
        "candidate_only_success_count": int(
            (candidate_success & ~base_success).sum()
        ),
        "baseline_only_success_count": int(
            (base_success & ~candidate_success).sum()
        ),
        "oracle_union_success_count": int(
            (base_success | candidate_success).sum()
        ),
        "runtime_gate_applied_count": int(
            np.asarray(candidate_result["gate_applied"]).sum()
        ),
        "promotion_passed": bool(
            candidate_success.sum() >= base_success.sum()
        ),
        "artifacts": {
            "forward": {
                "path": str(forward_path),
                "sha256": sha256(forward_path),
            },
            "base_inverse": {
                "path": str(base_path),
                "sha256": sha256(base_path),
            },
            "candidate": {
                "path": str(candidate_path),
                "sha256": sha256(candidate_path),
                "quality_weight": artifact["quality_weight"],
                "model_weight": artifact["model_weight"],
                "gate_rule": artifact["gate_rule"],
            },
        },
        "simulator_at_inference": False,
        "private_scoring_note": (
            "Simulator calls occur only after both actions are selected."
        ),
        "source_contract": {
            "system_validation_used_for_training": False,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
