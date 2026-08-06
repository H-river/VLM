#!/usr/bin/env python3
"""Open frozen candidate-confirmation inputs only after all SFT seeds finish."""

from __future__ import annotations

import json
from pathlib import Path

from .export_sft import DEFAULT_ARTIFACT, SYSTEM_PROMPT, USER_PREFIX, _repo_relative, _visible_state, read_jsonl, sha256_path, write_jsonl


def main() -> None:
    artifact = DEFAULT_ARTIFACT.resolve()
    for seed in (2026080401, 2026080402, 2026080403):
        manifest = json.loads((artifact / f"training/seed_{seed}/run_manifest.latest.json").read_text())
        if manifest.get("status") != "completed":
            raise RuntimeError("all three SFT seeds must finish before confirmation is opened")
    rollout_config = json.loads((artifact / "rollout_config.json").read_text())
    confirmation_manifest = artifact / "split_manifests/confirmation_frozen.json"
    if sha256_path(confirmation_manifest) != rollout_config["confirmation_manifest_sha256"]:
        raise RuntimeError("confirmation manifest hash changed")
    groups = read_jsonl(artifact / "split_manifests/groups.jsonl")
    rollouts = read_jsonl(artifact / "rollout_results.jsonl")
    representatives = {str(row["group_id"]): row for row in rollouts if row["plan_name"] == "direct_all_five" and int(row["seed_index"]) == 0}
    rows = []
    for group in groups:
        if group["split"] != "candidate_confirmation":
            continue
        group_id = str(group["group_id"])
        image_path = artifact / str(group["initial_sensor_image"])
        visible = _visible_state(group, representatives[group_id])
        rows.append({"group_id": group_id, "images": [_repo_relative(image_path)], "prompt": [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PREFIX + json.dumps(visible, sort_keys=True, separators=(",", ":"), allow_nan=False)}]}], "image_sha256": sha256_path(image_path), "candidate_only": True})
    rows.sort(key=lambda row: row["group_id"])
    path = artifact / "training/confirmation_inputs.jsonl"
    write_jsonl(path, rows)
    print(json.dumps({"records": len(rows), "path": str(path), "sha256": sha256_path(path), "contains_labels": False}, sort_keys=True))


if __name__ == "__main__":
    main()
