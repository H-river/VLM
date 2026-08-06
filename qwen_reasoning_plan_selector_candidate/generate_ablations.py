#!/usr/bin/env python3
"""Generate primary-seed image/probe ablations in the Qwen environment."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .inference import QwenSelectorBackend
from .protocol import USER_PREFIX


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/qwen_reasoning_plan_selector"
SEED = 2026080401


def main() -> None:
    config = yaml.safe_load((ROOT / f"qwen_reasoning_plan_selector_candidate/configs/training_seed_{SEED}.yaml").read_text())
    manifest = json.loads((ARTIFACT / f"training/seed_{SEED}/run_manifest.latest.json").read_text())
    backend = QwenSelectorBackend.load(config=config, adapter_path=Path(manifest["training_result"]["best_dev_checkpoint"]))
    inputs = [json.loads(line) for line in (ARTIFACT / "training/confirmation_inputs.jsonl").read_text().splitlines() if line.strip()]
    ablation_dir = ARTIFACT / "ablations"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    black = ablation_dir / "masked_sensor.png"
    Image.fromarray(np.zeros((256, 256), dtype=np.uint8), mode="L").save(black)
    outputs = []
    for index, row in enumerate(inputs):
        text = row["prompt"][1]["content"][1]["text"]
        visible = json.loads(text[len(USER_PREFIX):])
        probe_visible = copy.deepcopy(visible)
        probe_visible["learned_h1_probes"] = list(reversed(probe_visible["learned_h1_probes"]))
        values = list(probe_visible["ensemble_uncertainty_summary"].values())[::-1]
        probe_visible["ensemble_uncertainty_summary"] = dict(zip(probe_visible["ensemble_uncertainty_summary"], values, strict=True))
        probe_prompt = copy.deepcopy(row["prompt"])
        probe_prompt[1]["content"][1]["text"] = USER_PREFIX + json.dumps(probe_visible, sort_keys=True, separators=(",", ":"), allow_nan=False)
        cases = {
            "image_mask": (row["prompt"], black),
            "image_shuffle": (row["prompt"], ROOT / inputs[(index + 1) % len(inputs)]["images"][0]),
            "probe_shuffle": (probe_prompt, ROOT / row["images"][0]),
        }
        for name, (prompt, image_path) in cases.items():
            result = backend.generate(prompt, image_path, seed=SEED)
            outputs.append({"group_id": row["group_id"], "ablation": name, "seed": SEED, **result.to_dict()})
    path = ablation_dir / f"predictions_seed_{SEED}.jsonl"
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in outputs:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(temporary, path)
    print(json.dumps({"records": len(outputs), "valid": sum(row["valid_json"] for row in outputs), "path": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
