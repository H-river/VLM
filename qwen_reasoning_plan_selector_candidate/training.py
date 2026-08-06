#!/usr/bin/env python3
"""Candidate-only QLoRA wrapper for the fixed-gain plan selector."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from qwen_vl_supervisor_v1 import train_qlora as base

from .protocol import SYSTEM_PROMPT, USER_PREFIX
from .selector_contract import canonical_selector_text, parse_selector_output


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/training_seed_2026080401.yaml"
TRAINING_SEEDS = (2026080401, 2026080402, 2026080403)
PROTOCOL = json.loads((PACKAGE_ROOT / "training_protocol.json").read_text())
_PATCH_LOCK = threading.RLock()
_BASE_VALIDATE_CONFIG = base.validate_config
_BASE_RUNTIME_MANIFEST_BASE = base.runtime_manifest_base


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _candidate_path(value: str | Path, *, must_exist: bool = True) -> Path:
    path = _repo_path(value)
    try:
        path.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"candidate path escapes repository: {path}") from exc
    forbidden = {"frozen", "protected", "heldout", "held_out"}
    if any(part.lower() in forbidden for part in path.parts):
        raise ValueError(f"protected path is forbidden: {path}")
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def _completion_text(row: Mapping[str, Any]) -> str:
    completion = row.get("completion")
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValueError("completion must contain exactly one assistant message")
    message = completion[0]
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        raise ValueError("completion role must be assistant")
    content = message.get("content")
    if not isinstance(content, list) or len(content) != 1:
        raise ValueError("assistant completion must contain one item")
    item = content[0]
    if not isinstance(item, Mapping) or set(item) != {"type", "text"} or item.get("type") != "text":
        raise ValueError("assistant completion must be one text item")
    if not isinstance(item.get("text"), str):
        raise ValueError("assistant completion text is missing")
    return str(item["text"])


def validate_prebuilt_row(
    row: Mapping[str, Any], *, allowed_splits: set[str], image_root: Path,
    verify_image: bool,
) -> Path:
    expected = {"example_id", "split", "images", "prompt", "completion", "metadata"}
    if set(row) != expected:
        raise ValueError(f"unexpected row fields: {sorted(set(row) ^ expected)}")
    split = base.row_split(row)
    if split not in allowed_splits or split not in {"train", "dev"}:
        raise ValueError(f"illegal candidate split: {split}")
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise ValueError("prompt must be system,user")
    if prompt[0] != {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}:
        raise ValueError("system prompt is not the frozen selector prompt")
    if prompt[1].get("role") != "user" or not isinstance(prompt[1].get("content"), list):
        raise ValueError("user prompt is malformed")
    content = prompt[1]["content"]
    if len(content) != 2 or content[0] != {"type": "image"}:
        raise ValueError("one image must precede structured state")
    if set(content[1]) != {"type", "text"} or content[1].get("type") != "text":
        raise ValueError("structured state item is malformed")
    text = content[1].get("text")
    if not isinstance(text, str) or not text.startswith(USER_PREFIX):
        raise ValueError("user prompt prefix changed")
    visible = json.loads(text[len(USER_PREFIX):])
    if not isinstance(visible, dict):
        raise ValueError("visible state must be an object")
    forbidden = {"group_id", "base_state_id", "split", "oracle", "outcome", "q_goal"}
    if forbidden & set(visible):
        raise ValueError("hidden identity/outcome leaked into prompt")
    target = parse_selector_output(_completion_text(row))
    if canonical_selector_text(target) != _completion_text(row):
        raise ValueError("selector completion is not canonical whole-string JSON")
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ValueError("exactly one image path is required")
    image_path = Path(images[0]).expanduser()
    image_path = image_path.resolve() if image_path.is_absolute() else (image_root / image_path).resolve()
    _candidate_path(image_path)
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("candidate_only") is not True:
        raise ValueError("candidate-only provenance is required")
    if metadata.get("confirmation_excluded") is not True:
        raise ValueError("confirmation exclusion provenance is required")
    if metadata.get("label_source") != "three_seed_real_corrected_simulator_closed_loop_ranking":
        raise ValueError("label source is not the formal real-rollout ranking")
    if verify_image:
        from PIL import Image
        with Image.open(image_path) as image:
            image.verify()
        if sha256_path(image_path) != metadata.get("image_sha256"):
            raise ValueError("image hash mismatch")
    return image_path


def validate_candidate_config(config: Mapping[str, Any]) -> None:
    _BASE_VALIDATE_CONFIG(config)
    if config.get("schema_version") != "qwen_reasoning_plan_selector_training_v1":
        raise ValueError("wrong candidate training schema")
    if config.get("training_seeds") != list(TRAINING_SEEDS):
        raise ValueError("training seeds differ from preregistration")
    model, train, lora, data = config["model"], config["training"], config["lora"], config["data"]
    declared_model = PROTOCOL["base_model"]
    if Path(str(model["id"])).resolve() != Path(declared_model["id"]).resolve():
        raise ValueError("base model path differs from preregistration")
    if model.get("source_id") != declared_model["source_id"] or model.get("revision") != declared_model["revision"]:
        raise ValueError("base model identity differs from preregistration")
    if model.get("processor_revision") != declared_model["revision"]:
        raise ValueError("processor revision differs from preregistration")
    if model.get("expected_local_snapshot_tree_sha256") != declared_model["snapshot_tree_sha256"]:
        raise ValueError("base snapshot hash differs from preregistration")
    if model.get("local_files_only") is not True or model.get("require_cuda") is not True:
        raise ValueError("training must use local model on CUDA")
    declared_train = PROTOCOL["training"]
    for key in ("max_steps", "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps", "learning_rate", "eval_steps", "save_steps", "metric_for_best_model", "load_best_model_at_end", "completion_only_loss", "bf16"):
        if train.get(key) != declared_train[key]:
            raise ValueError(f"training.{key} differs from preregistration")
    if int(train.get("seed", -1)) not in TRAINING_SEEDS or train.get("data_seed") != train.get("seed"):
        raise ValueError("training seed is not preregistered")
    if (lora.get("r"), lora.get("alpha"), lora.get("dropout")) != (16, 32, 0.05):
        raise ValueError("LoRA config differs from preregistration")
    report_path = _candidate_path(data["export_report"])
    if sha256_path(report_path) != data["export_report_sha256"]:
        raise ValueError("SFT report hash mismatch")
    report = json.loads(report_path.read_text())
    if report.get("version") != "qwen_reasoning_plan_selector_sft_export_v1":
        raise ValueError("wrong SFT export report")
    if report.get("candidate_only") is not True or report.get("hard_labels_from_real_rollouts_only") is not True:
        raise ValueError("SFT provenance gate failed")
    gate_path = REPOSITORY_ROOT / "artifacts/qwen_reasoning_plan_selector/anti_collapse_audit.json"
    gate = json.loads(gate_path.read_text())
    if gate.get("gate_passed") is not True or sha256_path(gate_path) != report.get("gate_sha256"):
        raise ValueError("anti-collapse gate is not pinned PASS")
    for kind in ("train", "dev"):
        path = _candidate_path(data[f"{kind}_jsonl"])
        if sha256_path(path) != data[f"{kind}_sha256"]:
            raise ValueError(f"{kind} export hash mismatch")
        if report["records"][kind]["sha256"] != data[f"{kind}_sha256"]:
            raise ValueError(f"{kind} report entry mismatch")
    _candidate_path(train["output_dir"], must_exist=False)


def prepare_rows(config: Mapping[str, Any], *, verify_images: bool = True) -> base.PreparedRows:
    validate_candidate_config(config)
    data = config["data"]
    train_path, dev_path = _candidate_path(data["train_jsonl"]), _candidate_path(data["dev_jsonl"])
    image_root = _repo_path(data.get("image_root", "."))
    train_rows, dev_rows = base.read_jsonl(train_path), base.read_jsonl(dev_path)
    report = json.loads(_candidate_path(data["export_report"]).read_text())
    if len(train_rows) != int(report["records"]["train"]["count"]) or len(dev_rows) != int(report["records"]["dev"]["count"]):
        raise ValueError("SFT record count differs from report")
    for row in train_rows:
        validate_prebuilt_row(row, allowed_splits={"train"}, image_root=image_root, verify_image=verify_images)
    for row in dev_rows:
        validate_prebuilt_row(row, allowed_splits={"dev"}, image_root=image_root, verify_image=verify_images)
    return base.PreparedRows(train_path, dev_path, image_root, train_rows, dev_rows)


def row_to_example(row: Mapping[str, Any], image_root: Path) -> dict[str, Any]:
    path = validate_prebuilt_row(row, allowed_splits={str(base.row_split(row))}, image_root=image_root, verify_image=False)
    return {"images": [base.load_rgb_image(path)], "prompt": row["prompt"], "completion": row["completion"]}


def candidate_runtime_manifest_base(**kwargs: Any) -> dict[str, Any]:
    manifest = _BASE_RUNTIME_MANIFEST_BASE(**kwargs)
    manifest["manifest_version"] = "qwen_reasoning_plan_selector_training_run_v1"
    manifest["independent_adapter_name"] = "qwen_reasoning_plan_selector"
    manifest["training_protocol_sha256"] = sha256_path(PACKAGE_ROOT / "training_protocol.json")
    manifest["data"]["model_visible_allowlist_validator"] = "qwen_reasoning_plan_selector_candidate.training.validate_prebuilt_row"
    manifest["data"]["frozen_or_protected_predictions_opened"] = False
    return manifest


@contextmanager
def candidate_training_engine() -> Any:
    replacements = {"validate_config": validate_candidate_config, "prepare_rows": prepare_rows, "row_to_example": row_to_example, "runtime_manifest_base": candidate_runtime_manifest_base}
    with _PATCH_LOCK:
        old = {key: getattr(base, key) for key in replacements}
        try:
            for key, value in replacements.items():
                setattr(base, key, value)
            yield
        finally:
            for key, value in old.items():
                setattr(base, key, value)


def train(config: Mapping[str, Any], config_path: Path, **kwargs: Any) -> dict[str, Any]:
    validate_candidate_config(config)
    with candidate_training_engine():
        return base.train(config, config_path, **kwargs)


def main(argv: Sequence[str] | None = None) -> None:
    values = list(argv or sys.argv[1:])
    if not any(value == "--config" or value.startswith("--config=") for value in values):
        values = ["--config", str(DEFAULT_CONFIG), *values]
    args = base.parse_args(values)
    config, config_path = base.resolve_config(args)
    if args.print_resolved_config:
        print(json.dumps(config, indent=2, sort_keys=True)); return
    if args.validate_only:
        rows = prepare_rows(config, verify_images=True)
        print(json.dumps({"status": "valid", "train_records": len(rows.train_rows), "dev_records": len(rows.dev_rows), "model_loaded": False}, sort_keys=True)); return
    manifest = train(config, config_path, local_rank_arg=args.local_rank, save_at_global_step=args.save_at_global_step, stop_after_global_step=args.stop_after_global_step)
    if int(base.os.environ.get("RANK", "0")) == 0:
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
