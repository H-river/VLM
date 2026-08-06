#!/usr/bin/env python3
"""Final-only extra CEM seeds for high-power confirmation estimates."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from continuous_control_v12.contracts import metrics_dict, stable_seed
from continuous_control_v12.world_model import load_forward_ensemble

from .core import FIXED_CONTROLLER_CONFIG, PLAN_NAMES, FixedGainPlanController, canonical_hash
from .run_rollouts import DEFAULT_BASE_CONFIG, DEFAULT_CHECKPOINT, SEED_ROOT


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/qwen_reasoning_plan_selector"
EXTRA_SEED_INDICES = tuple(range(3, 13))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    started = time.perf_counter()
    for seed in (2026080401, 2026080402, 2026080403):
        manifest = json.loads((ARTIFACT / f"training/seed_{seed}/run_manifest.latest.json").read_text())
        if manifest.get("status") != "completed":
            raise RuntimeError("extra confirmation may run only after all training seeds complete")
    config = json.loads((ARTIFACT / "rollout_config.json").read_text())
    if config["fixed_controller_config"] != FIXED_CONTROLLER_CONFIG or config["formal_frozen_evaluation_enabled"]:
        raise RuntimeError("fixed controller or candidate-only invariant changed")
    confirmation_path = ARTIFACT / "split_manifests/confirmation_frozen.json"
    if sha256(confirmation_path) != config["confirmation_manifest_sha256"]:
        raise RuntimeError("confirmation manifest changed")
    groups = [row for row in read_jsonl(ARTIFACT / "split_manifests/groups.jsonl") if row["split"] == "candidate_confirmation"]
    model = load_forward_ensemble(DEFAULT_CHECKPOINT, device_name="cuda")
    controller = FixedGainPlanController(model=model, base_config_path=str(DEFAULT_BASE_CONFIG))
    output = ARTIFACT / "confirmation_extra_10seed_results.jsonl"
    completed = set()
    if output.exists():
        for row in read_jsonl(output):
            completed.add((row["group_id"], row["plan_name"], int(row["seed_index"])))
    mode = "a" if output.exists() else "w"
    executed = 0
    durations = []
    with output.open(mode, encoding="utf-8", buffering=1) as stream:
        for group in groups:
            cache = ARTIFACT / group["initial_capture_cache"]
            if sha256(cache) != group["initial_capture_cache_sha256"]:
                raise RuntimeError("initial confirmation capture hash mismatch")
            with np.load(cache, allow_pickle=False) as payload:
                initial = {"metrics": metrics_dict(np.asarray(payload["metrics"], dtype=np.float64)), "intensity": np.asarray(payload["intensity"], dtype=np.float32)}
            for plan in PLAN_NAMES:
                for seed_index in EXTRA_SEED_INDICES:
                    key = (group["group_id"], plan, seed_index)
                    if key in completed:
                        continue
                    cem_seed = int(stable_seed(SEED_ROOT, group["group_id"], seed_index) % (2**32))
                    one = time.perf_counter()
                    result = controller.run_episode(group, plan_name=plan, cem_seed=cem_seed, initial_capture=initial)
                    elapsed = time.perf_counter() - one
                    row = {"schema_version": "qwen_reasoning_plan_confirmation_extra_v1", "candidate_only": True, "formal_frozen_evaluation_enabled": False, "final_confirmation_only": True, "seed_index": seed_index, "episode_wall_seconds": elapsed, **result}
                    if row["fixed_controller_config_hash"] != canonical_hash(FIXED_CONTROLLER_CONFIG):
                        raise RuntimeError("controller config drift")
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
                    completed.add(key); executed += 1; durations.append(elapsed)
    status = {"candidate_only": True, "confirmation_groups": len(groups), "plans": list(PLAN_NAMES), "extra_seeds": len(EXTRA_SEED_INDICES), "extra_seed_indices": list(EXTRA_SEED_INDICES), "expected_episodes": len(groups) * len(PLAN_NAMES) * len(EXTRA_SEED_INDICES), "completed_episodes": len(completed), "executed_this_invocation": executed, "wall_seconds": time.perf_counter() - started, "mean_episode_seconds": float(np.mean(durations)), "p95_episode_seconds": float(np.quantile(durations, 0.95)), "output_sha256": sha256(output), "checkpoint_sha256": sha256(DEFAULT_CHECKPOINT), "fixed_controller_config_hash": canonical_hash(FIXED_CONTROLLER_CONFIG), "frozen_or_protected_enabled": False}
    (ARTIFACT / "confirmation_extra_10seed_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
