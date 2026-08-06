#!/usr/bin/env python3
"""Generate strict selector predictions in the dependency-light Qwen environment."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    seed = args.seed
    config_path = ROOT / f"qwen_reasoning_plan_selector_candidate/configs/training_seed_{seed}.yaml"
    config = yaml.safe_load(config_path.read_text())
    manifest_path = ARTIFACT / f"training/seed_{seed}/run_manifest.latest.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = Path(manifest["training_result"]["best_dev_checkpoint"])
    backend = QwenSelectorBackend.load(config=config, adapter_path=checkpoint)
    input_path = ARTIFACT / "training/confirmation_inputs.jsonl"
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    outputs = []
    for row in rows:
        image_path = ROOT / row["images"][0]
        if sha256(image_path) != row["image_sha256"]:
            raise ValueError("confirmation input image hash mismatch")
        result = backend.generate(row["prompt"], image_path, seed=seed)
        outputs.append({"group_id": row["group_id"], "seed": seed, "checkpoint": str(checkpoint), "checkpoint_adapter_hashes": backend.adapter_hashes, **result.to_dict()})
    output_path = ARTIFACT / f"predictions/confirmation_seed_{seed}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in outputs:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(temporary, output_path)
    print(json.dumps({"seed": seed, "records": len(outputs), "valid_json": sum(row["valid_json"] for row in outputs), "output": str(output_path), "sha256": sha256(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
