#!/usr/bin/env python3
"""Generate a hash-pinned candidate QLoRA config from legal train/dev exports."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from qwen_vl_supervisor_v1.train_qlora import load_yaml

from .training import (
    PACKAGE_ROOT,
    REPOSITORY_ROOT,
    TRAINING_SEEDS,
    _guard_candidate_path,
    validate_candidate_config,
    validate_official_sft_export_bundle,
)


DEFAULT_BASE_CONFIG = REPOSITORY_ROOT / "qwen_vl_supervisor_v1/configs/training_server.yaml"
DEFAULT_PROTOCOL = PACKAGE_ROOT / "protocol/meta_controller_protocol.json"


def _relative_repo(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def build_training_config(
    *,
    train_jsonl: Path,
    dev_jsonl: Path,
    export_report: Path,
    output_dir: Path,
    seed: int,
    base_config: Path = DEFAULT_BASE_CONFIG,
    protocol_path: Path = DEFAULT_PROTOCOL,
    train_report_key: str = "train",
    dev_report_key: str = "dev",
) -> dict[str, Any]:
    """Return one complete config; no model, image, or prediction is opened."""

    if seed not in TRAINING_SEEDS:
        raise ValueError(f"seed must be one of {list(TRAINING_SEEDS)}")
    if train_report_key != "train" or dev_report_key != "dev":
        raise ValueError("official export report keys must remain train and dev")
    train_jsonl = _guard_candidate_path(train_jsonl, role="train_jsonl")
    dev_jsonl = _guard_candidate_path(dev_jsonl, role="dev_jsonl")
    export_report = _guard_candidate_path(export_report, role="export_report")
    output_dir = _guard_candidate_path(output_dir, role="output_dir", must_exist=False)
    protocol_path = protocol_path.resolve()
    if protocol_path != DEFAULT_PROTOCOL.resolve():
        raise ValueError("training protocol path must remain the preregistered candidate protocol")
    base_config = base_config.resolve()
    if base_config != DEFAULT_BASE_CONFIG.resolve():
        raise ValueError("base training config must remain the audited supervisor config")

    export_bundle = validate_official_sft_export_bundle(
        train_path=train_jsonl,
        dev_path=dev_jsonl,
        report_path=export_report,
    )

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen_training = protocol["training"]
    config = copy.deepcopy(load_yaml(base_config))
    config["schema_version"] = "qwen_h1_meta_training_v0"
    config["run_name"] = f"qwen_h1_meta_v0_seed_{seed}"
    config["training_seeds"] = list(TRAINING_SEEDS)
    config["generation"] = copy.deepcopy(frozen_training["generation"])

    config["model"].update(
        {
            "id": frozen_training["base_model"],
            "source_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "revision": frozen_training["base_revision"],
            "processor_revision": frozen_training["base_revision"],
            "expected_local_snapshot_tree_sha256": frozen_training[
                "base_snapshot_tree_sha256"
            ],
            "local_files_only": True,
        }
    )
    config["data"].update(
        {
            "train_jsonl": _relative_repo(train_jsonl),
            "dev_jsonl": _relative_repo(dev_jsonl),
            "train_sha256": export_bundle["train_sha256"],
            "dev_sha256": export_bundle["dev_sha256"],
            "export_report": _relative_repo(export_report),
            "export_report_sha256": export_bundle["report_sha256"],
            "data_audit": _relative_repo(export_bundle["audit_path"]),
            "data_audit_sha256": export_bundle["audit_sha256"],
            "train_export_report_key": train_report_key,
            "dev_export_report_key": dev_report_key,
            "image_root": ".",
            "max_train_samples": None,
            "max_dev_samples": None,
            "allowed_train_splits": ["train"],
            "allowed_dev_splits": ["dev"],
            "prohibit_frozen_splits": True,
        }
    )
    config["lora"].update(
        {
            "r": int(frozen_training["lora_rank"]),
            "alpha": int(frozen_training["lora_alpha"]),
            "dropout": float(frozen_training["lora_dropout"]),
        }
    )
    config["training"].update(
        {
            "output_dir": _relative_repo(output_dir),
            "max_steps": int(frozen_training["max_steps"]),
            "per_device_train_batch_size": int(frozen_training["batch_size"]),
            "per_device_eval_batch_size": int(frozen_training["eval_batch_size"]),
            "gradient_accumulation_steps": int(
                frozen_training["gradient_accumulation_steps"]
            ),
            "learning_rate": float(frozen_training["learning_rate"]),
            "seed": seed,
            "data_seed": seed,
            "eval_steps": int(frozen_training["eval_steps"]),
            "save_steps": int(frozen_training["save_steps"]),
            "save_total_limit": int(frozen_training["max_steps"])
            // int(frozen_training["save_steps"]),
            "gradient_checkpointing": bool(
                frozen_training["gradient_checkpointing"]
            ),
            "resume_from_checkpoint": None,
        }
    )
    validate_candidate_config(config)
    return config


def write_yaml(path: Path, config: Mapping[str, Any]) -> None:
    import yaml

    path = _guard_candidate_path(path, role="config_output", must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-report-key", default="train")
    parser.add_argument("--dev-report-key", default="dev")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_training_config(
        train_jsonl=args.train_jsonl,
        dev_jsonl=args.dev_jsonl,
        export_report=args.export_report,
        output_dir=args.output_dir,
        seed=args.seed,
        train_report_key=args.train_report_key,
        dev_report_key=args.dev_report_key,
    )
    write_yaml(args.output, config)
    print(
        json.dumps(
            {
                "status": "candidate_training_config_written",
                "output": str(args.output.resolve()),
                "seed": args.seed,
                "train_sha256": config["data"]["train_sha256"],
                "dev_sha256": config["data"]["dev_sha256"],
                "formal_frozen_evaluation_enabled": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
