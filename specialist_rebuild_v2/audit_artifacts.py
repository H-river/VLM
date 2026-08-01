#!/usr/bin/env python3
"""Audit the completed dataset and all five trained checkpoint artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_MODELS = {
    "direction": "direction_v2.pt",
    "forward": "forward_v2.pt",
    "inverse": "inverse_ranker_v2.pt",
    "measurement": "measurement_v2.pt",
    "visual_inverse": "visual_inverse_v2.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required JSON file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    config = read_json(data_dir / "config.json")
    manifest = read_json(data_dir / "manifest.json")
    training = read_json(run_dir / "training_summary.json")

    expected_groups = sum(int(value) for value in config["group_counts"].values())
    expected_transitions = expected_groups * 81
    expected_images = expected_groups * (
        1 + int(config["visual_targets_per_group"])
    )
    if int(manifest["total_groups"]) != expected_groups:
        raise RuntimeError("manifest group total does not match configuration")
    if int(manifest["total_transitions"]) != expected_transitions:
        raise RuntimeError("manifest transition total does not match configuration")
    if int(manifest["total_images"]) != expected_images:
        raise RuntimeError("manifest image total does not match configuration")
    if training["version"] != config["version"]:
        raise RuntimeError("training and dataset versions differ")
    if int(training["seed"]) != int(config["training"]["seed"]):
        raise RuntimeError("training seed differs from the configured seed")
    if set(training["models"]) != set(EXPECTED_MODELS):
        raise RuntimeError(
            f"training summary model set is {sorted(training['models'])}"
        )

    import torch

    model_results: dict[str, Any] = {}
    for name, filename in EXPECTED_MODELS.items():
        path = run_dir / filename
        if not path.is_file():
            raise RuntimeError(f"checkpoint is missing: {path}")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        state = artifact.get("state_dict")
        if not isinstance(state, dict) or not state:
            raise RuntimeError(f"{name}: state_dict is empty")
        nonfinite = [
            key
            for key, value in state.items()
            if torch.is_floating_point(value)
            and not bool(torch.isfinite(value).all())
        ]
        if nonfinite:
            raise RuntimeError(f"{name}: non-finite tensors: {nonfinite[:5]}")
        tensor_values = sum(int(value.numel()) for value in state.values())
        declared = int(training["models"][name]["parameter_count"])
        if tensor_values != declared:
            raise RuntimeError(
                f"{name}: state tensors {tensor_values} != declared {declared}"
            )
        validation = training["models"][name].get("validation")
        if not isinstance(validation, dict) or not validation:
            raise RuntimeError(f"{name}: validation result is missing")
        model_results[name] = {
            "artifact": str(path),
            "bytes": path.stat().st_size,
            "parameter_count": declared,
            "best_epoch": int(training["models"][name]["best_epoch"]),
            "validation": validation,
        }

    checksum_path = data_dir / "checksums.sha256"
    checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
    if len(checksum_lines) != int(manifest["checksum_file_count"]):
        raise RuntimeError("checksum file count differs from manifest")

    result = {
        "passed": True,
        "dataset": {
            "version": config["version"],
            "groups": expected_groups,
            "transitions": expected_transitions,
            "images": expected_images,
            "visual_pairs": expected_groups
            * int(config["visual_targets_per_group"]),
            "checksummed_files": len(checksum_lines),
        },
        "training": {
            "seed": int(training["seed"]),
            "device": training["device"],
            "models": model_results,
        },
    }
    output = run_dir / "completion_audit.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
