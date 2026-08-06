#!/usr/bin/env python3
"""Generate strict Qwen predictions for final physical probe interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

from .inference import QwenSelectorBackend


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/qwen_reasoning_plan_selector"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    seed = args.seed
    config = yaml.safe_load((ROOT / f"qwen_reasoning_plan_selector_candidate/configs/training_seed_{seed}.yaml").read_text())
    manifest = json.loads((ARTIFACT / f"training/seed_{seed}/run_manifest.latest.json").read_text())
    checkpoint = Path(manifest["training_result"]["best_dev_checkpoint"])
    backend = QwenSelectorBackend.load(config=config, adapter_path=checkpoint)
    inputs = read_jsonl(ARTIFACT / "probe_coupling_confirmation/inputs.jsonl")
    outputs = []
    for row in inputs:
        image_path = ROOT / row["images"][0]
        if sha256(image_path) != row["image_sha256"]:
            raise RuntimeError("probe-coupling image hash mismatch")
        result = backend.generate(row["prompt"], image_path, seed=seed)
        outputs.append({"group_id": row["group_id"], "seed": seed, "checkpoint": str(checkpoint), "checkpoint_adapter_hashes": backend.adapter_hashes, **result.to_dict()})
    output = ARTIFACT / f"probe_coupling_confirmation/predictions_seed_{seed}.jsonl"
    temporary = output.with_suffix(f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in outputs:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"seed": seed, "records": len(outputs), "valid_json": sum(row["valid_json"] for row in outputs), "output": str(output), "sha256": sha256(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
