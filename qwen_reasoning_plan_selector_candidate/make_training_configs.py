#!/usr/bin/env python3
"""Materialize three hash-pinned configs after the SFT gate/export."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts/qwen_reasoning_plan_selector"
SEEDS = (2026080401, 2026080402, 2026080403)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads((PACKAGE / "training_protocol.json").read_text())
    report_path = ARTIFACT / "training/sft_export_report.json"
    report = json.loads(report_path.read_text())
    train_path = ROOT / report["records"]["train"]["path"]
    dev_path = ROOT / report["records"]["dev"]["path"]
    template = yaml.safe_load((ROOT / "supervisor_v1_1_candidate/configs/training_full.yaml").read_text())
    template["schema_version"] = "qwen_reasoning_plan_selector_training_v1"
    template["run_name"] = "qwen_reasoning_plan_selector_candidate_3seed_200step"
    template["training_seeds"] = list(SEEDS)
    template["model"]["id"] = protocol["base_model"]["id"]
    template["model"]["source_id"] = protocol["base_model"]["source_id"]
    template["model"]["revision"] = protocol["base_model"]["revision"]
    template["model"]["processor_revision"] = protocol["base_model"]["revision"]
    template["model"]["expected_local_snapshot_tree_sha256"] = protocol["base_model"]["snapshot_tree_sha256"]
    template["data"].update({
        "train_jsonl": train_path.resolve().relative_to(ROOT).as_posix(),
        "dev_jsonl": dev_path.resolve().relative_to(ROOT).as_posix(),
        "train_sha256": sha256(train_path),
        "dev_sha256": sha256(dev_path),
        "export_report": report_path.resolve().relative_to(ROOT).as_posix(),
        "export_report_sha256": sha256(report_path),
        "train_export_report_key": "train",
        "dev_export_report_key": "dev",
        "image_root": ".",
        "max_train_samples": None,
        "max_dev_samples": None,
        "allowed_train_splits": ["train"],
        "allowed_dev_splits": ["dev"],
        "prohibit_frozen_splits": True,
    })
    config_dir = PACKAGE / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        config = copy.deepcopy(template)
        config["training"]["seed"] = seed
        config["training"]["data_seed"] = seed
        config["training"]["output_dir"] = f"artifacts/qwen_reasoning_plan_selector/training/seed_{seed}"
        path = config_dir / f"training_seed_{seed}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(f"{path} {sha256(path)}")


if __name__ == "__main__":
    main()
