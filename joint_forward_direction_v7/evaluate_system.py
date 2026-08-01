#!/usr/bin/env python3
"""Evaluate saved Qwen decisions with the shared forward-direction v7 overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Qwen_orchestration.scripts.evaluate_end_to_end as frozen_evaluator
from control_rebuild_v3.train_forward import configure
from control_rebuild_v4.evaluate_orchestrated_system import (
    RecordingRuntime,
    physical_forward_direction_metrics,
    subset_by_category,
    task_ready_counts,
)
from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from joint_forward_direction_v7.orchestrated_runtime import (
    OrchestratedSharedRuntimeV7,
)

DEFAULT_QWEN_DATA = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "Qwen_orchestration/results/v1/stage2_safe_runtime_v1"
    / "checkpoint-1000_all.jsonl"
)
DEFAULT_V4_OVERLAY = (
    REPO_ROOT.parent
    / "VLM_runs/direction_rebuild_v4_tree_one_seed"
    / "candidate_overlay_direction_v4_manifest.json"
)
DEFAULT_V5_RUN = (
    REPO_ROOT.parent / "VLM_runs/control_rebuild_v5_one_seed"
)
DEFAULT_V7_RUN = (
    REPO_ROOT.parent / "VLM_runs/joint_forward_direction_v7_one_seed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN_DATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--v4-overlay", type=Path, default=DEFAULT_V4_OVERLAY)
    parser.add_argument("--v5-run", type=Path, default=DEFAULT_V5_RUN)
    parser.add_argument("--v7-run", type=Path, default=DEFAULT_V7_RUN)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-per-category", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    qwen_data = args.qwen_data.resolve()
    v5_run = args.v5_run.resolve()
    v7_run = args.v7_run.resolve()
    shared_artifact = v7_run / "shared_forward_direction_v7.pt"
    forward_artifact = v5_run / "forward_tree_v5.pkl"
    inverse_artifact = v5_run / "inverse_tree_v5.pkl"
    output = (
        args.output.resolve()
        if args.output is not None
        else v7_run / "orchestrated_system_shared_v7_validation.json"
    )
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if bool(existing.get("complete")):
            raise RuntimeError(
                f"refusing to overwrite completed validation: {output}"
            )

    canonical = frozen_evaluator.read_jsonl(
        qwen_data / "canonical/val.jsonl"
    )
    canonical = subset_by_category(canonical, args.max_per_category)
    selected_ids = {str(row["example_id"]) for row in canonical}
    predictions = [
        row
        for row in frozen_evaluator.read_jsonl(args.predictions.resolve())
        if str(row["example_id"]) in selected_ids
    ]
    if len(predictions) != len(canonical):
        raise ValueError(
            f"saved Qwen predictions cover {len(predictions)}/"
            f"{len(canonical)} records"
        )
    private_rows = frozen_evaluator.read_jsonl(
        qwen_data / "private/source_cases/val.jsonl"
    )
    torch, device = configure(20260731, args.device)
    v4_backend = OrchestratedSpecialistRuntimeV4.from_manifest(
        torch,
        args.v4_overlay.resolve(),
        device,
    )
    shared_v7 = OrchestratedSharedRuntimeV7(
        torch,
        v4_backend,
        shared_artifact,
        forward_artifact,
        inverse_artifact,
        device,
    )
    recording_runtime = RecordingRuntime(shared_v7)

    started = time.perf_counter()
    original_runtime = frozen_evaluator.OrchestrationRuntime
    original_inverse_scorer = frozen_evaluator.private_inverse_target_reached
    inverse_replay_results: list[bool] = []

    def recording_inverse_scorer(*values, **named):
        result = bool(original_inverse_scorer(*values, **named))
        inverse_replay_results.append(result)
        return result

    frozen_evaluator.OrchestrationRuntime = lambda: recording_runtime
    frozen_evaluator.private_inverse_target_reached = recording_inverse_scorer
    try:
        metrics, details = frozen_evaluator.evaluate(
            predictions,
            canonical,
            private_rows,
            qwen_data,
        )
    finally:
        frozen_evaluator.OrchestrationRuntime = original_runtime
        frozen_evaluator.private_inverse_target_reached = original_inverse_scorer

    inverse_rows = [
        row
        for row in canonical
        if row["target_decision"]["status"] == "ready"
        and row["target_decision"]["task_type"] == "inverse_control"
    ]
    if len(inverse_replay_results) != 2 * len(inverse_rows):
        raise RuntimeError("inverse replay recorder count differs")
    inverse_by_route: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "count": 0,
            "end_to_end_success_count": 0,
            "correctly_routed_success_count": 0,
        }
    )
    for index, row in enumerate(inverse_rows):
        route = str(row["target_decision"]["route_name"])
        values = inverse_by_route[route]
        values["count"] += 1
        values["end_to_end_success_count"] += int(
            inverse_replay_results[2 * index]
        )
        values["correctly_routed_success_count"] += int(
            inverse_replay_results[2 * index + 1]
        )
    metrics["inverse_physical_by_route"] = {
        route: {
            **values,
            "end_to_end_target_reached": (
                values["end_to_end_success_count"] / values["count"]
            ),
            "correctly_routed_target_reached": (
                values["correctly_routed_success_count"] / values["count"]
            ),
        }
        for route, values in sorted(inverse_by_route.items())
    }
    physical_metrics, physical_details = physical_forward_direction_metrics(
        canonical,
        predictions,
        private_rows,
        recording_runtime,
    )
    metrics.update(physical_metrics)
    for detail in details:
        detail.update(physical_details.get(str(detail["example_id"]), {}))

    report = {
        "evaluation_version": (
            "saved_qwen_checkpoint1000_plus_shared_forward_direction_v7"
        ),
        "scope": (
            "forward state/image and direction state/image routes use shared "
            "v7; numerical and visual inverse, measurement, and Qwen remain "
            "frozen at their prior candidates"
        ),
        "device": str(device),
        "record_count": len(canonical),
        "task_ready_counts": task_ready_counts(canonical),
        "max_per_category": args.max_per_category,
        "artifacts": {
            "shared_forward_direction_v7": {
                "path": str(shared_artifact),
                "sha256": sha256(shared_artifact),
            },
            "numerical_forward_v5_for_inverse_only": {
                "path": str(forward_artifact),
                "sha256": sha256(forward_artifact),
            },
            "numerical_inverse_v5": {
                "path": str(inverse_artifact),
                "sha256": sha256(inverse_artifact),
            },
        },
        "simulator_at_inference": False,
        "private_scoring_note": (
            "Simulator calls occur only after inference to obtain private "
            "physical ground truth."
        ),
        "metrics": metrics,
        "complete": True,
        "seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    details_path = (
        args.details.resolve()
        if args.details is not None
        else output.with_suffix(".details.jsonl")
    )
    with details_path.open("w", encoding="utf-8") as stream:
        for row in details:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
