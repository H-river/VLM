#!/usr/bin/env python3
"""Resumable real corrected-simulator rollouts for every fixed-gain plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from continuous_control_v12.contracts import metrics_dict, stable_seed
from continuous_control_v12.world_model import load_forward_ensemble

from .core import FIXED_CONTROLLER_CONFIG, PLAN_NAMES, FixedGainPlanController, canonical_hash


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = REPOSITORY_ROOT / "artifacts/qwen_reasoning_plan_selector"
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "runs/overnight_v12_semantics_20260731_002709/models/lc_128g_v2/continuous_forward_v12_128g.pt"
DEFAULT_BASE_CONFIG = REPOSITORY_ROOT / "optical_sim/configs/base_config.yaml"
SEED_ROOT = 2026080411


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_initial_capture(artifact: Path, group: Mapping[str, Any]) -> dict[str, Any]:
    cache = artifact / str(group["initial_capture_cache"])
    if sha256_path(cache) != str(group["initial_capture_cache_sha256"]):
        raise RuntimeError(f"initial corrected-simulator cache hash mismatch: {cache}")
    with np.load(cache, allow_pickle=False) as payload:
        metrics = np.asarray(payload["metrics"], dtype=np.float64)
        intensity = np.asarray(payload["intensity"], dtype=np.float32)
    return {"metrics": metrics_dict(metrics), "intensity": intensity}


def _episode_key(group_id: str, plan_name: str, seed_index: int) -> tuple[str, str, int]:
    return group_id, plan_name, seed_index


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    artifact = args.artifact.resolve()
    rollout_config = json.loads((artifact / "rollout_config.json").read_text())
    if bool(rollout_config.get("formal_frozen_evaluation_enabled", True)):
        raise RuntimeError("frozen/protected evaluation must remain disabled")
    if rollout_config["fixed_controller_config"] != FIXED_CONTROLLER_CONFIG:
        raise RuntimeError("rollout config differs from code-locked fixed controller")
    if sha256_path(args.checkpoint.resolve()) != rollout_config["source_files"]["learned_h1_checkpoint_sha256"]:
        raise RuntimeError("Learned-H1 checkpoint hash differs from manifest")
    groups = read_jsonl(artifact / "split_manifests/groups.jsonl")
    if args.group_limit is not None:
        groups = groups[: int(args.group_limit)]
    output = args.output.resolve() if args.output else artifact / (
        "timing_smoke_results.jsonl" if args.smoke else "rollout_results.jsonl"
    )
    if args.smoke and output.name == "rollout_results.jsonl":
        raise RuntimeError("smoke output may not replace formal rollout results")
    completed: set[tuple[str, str, int]] = set()
    prior_rows: list[dict[str, Any]] = []
    if output.exists():
        prior_rows = read_jsonl(output)
        for row in prior_rows:
            key = _episode_key(str(row["group_id"]), str(row["plan_name"]), int(row["seed_index"]))
            if key in completed:
                raise RuntimeError(f"duplicate completed rollout key: {key}")
            completed.add(key)
    device = str(args.device)
    model = load_forward_ensemble(args.checkpoint.resolve(), device_name=device)
    if bool(model.image_conditioning):
        raise RuntimeError("must use the frozen numerical Learned-H1 checkpoint")
    controller = FixedGainPlanController(
        model=model,
        base_config_path=str(args.base_config.resolve()),
    )
    requested_plans = list(PLAN_NAMES)
    seed_count = 1 if args.smoke else 3
    expected = len(groups) * len(requested_plans) * seed_count
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output.exists() else "w"
    executed_now = 0
    episode_seconds = []
    with output.open(mode, encoding="utf-8", buffering=1) as stream:
        for group in groups:
            if bool(group.get("frozen_or_protected", True)):
                raise RuntimeError("non-candidate group found in rollout manifest")
            initial_capture = _load_initial_capture(artifact, group)
            seeds = [
                int(stable_seed(SEED_ROOT, str(group["group_id"]), seed_index) % (2**32))
                for seed_index in range(seed_count)
            ]
            for plan_name in requested_plans:
                for seed_index, cem_seed in enumerate(seeds):
                    key = _episode_key(str(group["group_id"]), plan_name, seed_index)
                    if key in completed:
                        continue
                    episode_started = time.perf_counter()
                    result = controller.run_episode(
                        group,
                        plan_name=plan_name,
                        cem_seed=cem_seed,
                        initial_capture=initial_capture,
                    )
                    elapsed = time.perf_counter() - episode_started
                    row = {
                        "schema_version": "qwen_reasoning_plan_rollout_v1",
                        "candidate_only": True,
                        "formal_frozen_evaluation_enabled": False,
                        "seed_index": seed_index,
                        "matched_seed_policy": "identical cem_seed across plans for group and seed_index",
                        "episode_wall_seconds": elapsed,
                        **result,
                    }
                    if row["fixed_controller_config_hash"] != canonical_hash(FIXED_CONTROLLER_CONFIG):
                        raise RuntimeError("episode controller config hash drift")
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    completed.add(key)
                    executed_now += 1
                    episode_seconds.append(elapsed)
                    if args.max_episodes is not None and executed_now >= int(args.max_episodes):
                        break
                if args.max_episodes is not None and executed_now >= int(args.max_episodes):
                    break
            if args.max_episodes is not None and executed_now >= int(args.max_episodes):
                break
    status = {
        "candidate_only": True,
        "formal_frozen_evaluation_enabled": False,
        "output": str(output),
        "output_sha256": sha256_path(output),
        "smoke": bool(args.smoke),
        "groups_requested": len(groups),
        "plans": requested_plans,
        "seeds_per_group_plan": seed_count,
        "expected_episodes_for_invocation": expected,
        "completed_unique_episodes_in_output": len(completed),
        "executed_this_invocation": executed_now,
        "mean_episode_seconds_this_invocation": None if not episode_seconds else float(np.mean(episode_seconds)),
        "p95_episode_seconds_this_invocation": None if not episode_seconds else float(np.quantile(episode_seconds, 0.95)),
        "invocation_wall_seconds": time.perf_counter() - started,
        "complete": len(completed) == expected and args.max_episodes is None,
        "device": device,
        "checkpoint_sha256": sha256_path(args.checkpoint.resolve()),
        "fixed_controller_config_hash": canonical_hash(FIXED_CONTROLLER_CONFIG),
    }
    status_path = artifact / ("timing_smoke.json" if args.smoke else "rollout_run_status.json")
    atomic_json(status_path, status)
    print(json.dumps(status, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--group-limit", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
