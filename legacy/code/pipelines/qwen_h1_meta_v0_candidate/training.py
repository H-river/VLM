#!/usr/bin/env python3
"""Candidate-only wrapper around the audited supervisor QLoRA engine.

The heavy trainer/checkpoint machinery is reused from
``qwen_vl_supervisor_v1.train_qlora``.  Only its supervisor-specific exported
row validator and runtime-manifest label are replaced, temporarily and within
this process.  The original module and its files are never modified.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from qwen_vl_supervisor_v1 import train_qlora as base

from .contracts import MetaContractError, parse_meta_input, parse_meta_output


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/training_server.yaml"
TRAINING_SEEDS = (2026080201, 2026080202, 2026080203)
OFFICIAL_SFT_EXPORT_REPORT_VERSION = "qwen_h1_meta_v0_sft_export_report_v1"
OFFICIAL_DATA_AUDIT_VERSION = "qwen_h1_meta_v0_data_audit_v1"
EXPECTED_SPLIT_RECORDS = {"train": 48, "dev": 24}
EXPECTED_AUDIT_RECORDS = sum(EXPECTED_SPLIT_RECORDS.values())
EXPECTED_KNOWN_IDENTITY_COUNT = 298
PROHIBITED_COMPONENTS = {
    "frozen",
    "protected",
    "heldout",
    "held_out",
    "severity_ood",
    "frozen_iid",
    "frozen_ood",
}
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PATCH_LOCK = threading.RLock()

_BASE_VALIDATE_CONFIG = base.validate_config
_BASE_RUNTIME_MANIFEST_BASE = base.runtime_manifest_base


def sha256_path(path: Path) -> str:
    return base.sha256_path(path)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _load_prompt_freeze() -> tuple[str, dict[str, Any]]:
    manifest_path = PACKAGE_ROOT / "protocol/prompt_freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError("prompt freeze manifest is missing files")
    for name, expected in files.items():
        path = PACKAGE_ROOT / "protocol" / str(name)
        if sha256_path(path) != expected:
            raise RuntimeError(f"frozen prompt file hash mismatch: {path}")
    system = (PACKAGE_ROOT / "protocol/system_prompt.txt").read_text(encoding="utf-8")
    contract = json.loads(
        (PACKAGE_ROOT / "protocol/prompt_contract.json").read_text(encoding="utf-8")
    )
    return system, contract


SYSTEM_PROMPT, PROMPT_CONTRACT = _load_prompt_freeze()
USER_TEXT_PREFIX = str(PROMPT_CONTRACT["user_text_prefix"])


def canonical_meta_text(value: Mapping[str, Any]) -> str:
    return _canonical_json(parse_meta_output(value).to_dict())


def validate_prompt(prompt: Any) -> dict[str, Any]:
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise ValueError("prompt must contain exactly system and user messages")
    if [message.get("role") for message in prompt if isinstance(message, Mapping)] != [
        "system",
        "user",
    ]:
        raise ValueError("prompt roles must be exactly system,user")
    if prompt[0].get("content") != [{"type": "text", "text": SYSTEM_PROMPT}]:
        raise ValueError("system prompt differs from frozen prompt bytes")
    content = prompt[1].get("content")
    if (
        not isinstance(content, list)
        or len(content) != 2
        or content[0] != {"type": "image"}
        or not isinstance(content[1], Mapping)
        or set(content[1]) != {"type", "text"}
        or content[1].get("type") != "text"
        or not isinstance(content[1].get("text"), str)
    ):
        raise ValueError("user content must be image first and one structured text item")
    text = content[1]["text"]
    if not text.startswith(USER_TEXT_PREFIX):
        raise ValueError("user text prefix differs from frozen prompt contract")
    payload = text[len(USER_TEXT_PREFIX) :]
    parsed = parse_meta_input(payload).to_dict()
    if text != USER_TEXT_PREFIX + _canonical_json(parsed):
        raise ValueError("runtime state is not in frozen compact canonical serialization")
    return parsed


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _guard_candidate_path(path: Path, *, role: str, must_exist: bool = True) -> Path:
    resolved = path.expanduser().resolve()
    lowered = {part.lower() for part in resolved.parts}
    forbidden = sorted(lowered & PROHIBITED_COMPONENTS)
    if forbidden:
        raise ValueError(f"{role} path contains forbidden component(s) {forbidden}: {resolved}")
    if not _is_relative_to(resolved, PACKAGE_ROOT):
        raise ValueError(f"{role} must remain inside {PACKAGE_ROOT}: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _load_json_object(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not readable valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object: {path}")
    return value


def _report_reference(value: Any, *, report_path: Path, role: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role}.path must be a non-empty string")
    candidate = Path(value).expanduser()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (report_path.parent / candidate).resolve()
    )
    return _guard_candidate_path(resolved, role=f"{role}.path")


def _require_sha256(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not HEX_SHA256.fullmatch(value):
        raise ValueError(f"{role} must be a lowercase SHA-256")
    return value


def _require_mapping(value: Any, *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be an object")
    return value


def _validate_pass_audit(audit: Mapping[str, Any]) -> None:
    required = {
        "version",
        "candidate_only",
        "audited_splits",
        "record_count",
        "known_identity_registry_count",
        "identity_overlap",
        "exact_visible_collision",
        "rounded_4dp_visible_collision",
        "near_visible_collision",
        "visible_only_classifier",
        "visible_plus_hidden_setup_classifier",
        "hidden_setup_macro_f1_gain",
        "thresholds",
        "gate",
        "red_reasons",
        "sft_export_permitted",
        "diagnostics",
    }
    missing = sorted(required - set(audit))
    if missing:
        raise ValueError(f"official data audit is missing required fields: {missing}")
    if audit.get("version") != OFFICIAL_DATA_AUDIT_VERSION:
        raise ValueError("data audit version is not the official candidate version")
    if audit.get("candidate_only") is not True:
        raise ValueError("data audit must be candidate_only=true")
    if audit.get("audited_splits") != ["train", "dev"]:
        raise ValueError("data audit must cover exactly train and dev")
    if int(audit.get("record_count", -1)) != EXPECTED_AUDIT_RECORDS:
        raise ValueError(
            f"data audit must cover exactly {EXPECTED_AUDIT_RECORDS} train+dev records"
        )
    if int(audit.get("known_identity_registry_count", -1)) != EXPECTED_KNOWN_IDENTITY_COUNT:
        raise ValueError("data audit identity registry count is not the frozen 298")
    if (
        audit.get("gate") != "PASS"
        or audit.get("sft_export_permitted") is not True
        or audit.get("red_reasons") != []
    ):
        raise ValueError("information-sufficiency gate is not an unqualified PASS")

    identity = _require_mapping(audit.get("identity_overlap"), role="identity_overlap")
    if int(identity.get("known_overlap_count", -1)) != 0:
        raise ValueError("data audit reports known-identity overlap")
    if int(identity.get("cross_split_overlap_count", -1)) != 0:
        raise ValueError("data audit reports candidate cross-split identity overlap")

    thresholds = _require_mapping(audit.get("thresholds"), role="thresholds")
    expected_thresholds = {
        "exact_conflict_count_max": 0.0,
        "rounded_4dp_conflict_count_max": 0.0,
        "near_conflict_rate_max": 0.20,
        "hidden_setup_macro_f1_gain_max": 0.15,
        "visible_classifier_macro_f1_min": 0.50,
    }
    for name, expected in expected_thresholds.items():
        try:
            actual = float(thresholds[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"data audit threshold {name!r} is missing or invalid") from exc
        if actual != expected:
            raise ValueError(f"data audit threshold {name!r} differs from preregistration")

    exact = _require_mapping(
        audit.get("exact_visible_collision"), role="exact_visible_collision"
    )
    rounded = _require_mapping(
        audit.get("rounded_4dp_visible_collision"),
        role="rounded_4dp_visible_collision",
    )
    near = _require_mapping(
        audit.get("near_visible_collision"), role="near_visible_collision"
    )
    visible = _require_mapping(
        audit.get("visible_only_classifier"), role="visible_only_classifier"
    )
    if int(exact.get("conflicting_collision_group_count", -1)) != 0:
        raise ValueError("data audit exact-visible conflict result is not PASS")
    if int(rounded.get("conflicting_collision_group_count", -1)) != 0:
        raise ValueError("data audit rounded-visible conflict result is not PASS")
    try:
        near_rate = float(near["conflicting_near_pair_rate"])
        hidden_gain = float(audit["hidden_setup_macro_f1_gain"])
        visible_f1 = float(visible["macro_f1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("data audit PASS statistics are missing or invalid") from exc
    if near_rate > expected_thresholds["near_conflict_rate_max"]:
        raise ValueError("data audit near-visible conflict result contradicts PASS")
    if hidden_gain > expected_thresholds["hidden_setup_macro_f1_gain_max"]:
        raise ValueError("data audit hidden-setup gain result contradicts PASS")
    if visible_f1 < expected_thresholds["visible_classifier_macro_f1_min"]:
        raise ValueError("data audit visible-only classifier result contradicts PASS")


def validate_official_sft_export_bundle(
    *,
    train_path: Path,
    dev_path: Path,
    report_path: Path,
    expected_train_sha256: str | None = None,
    expected_dev_sha256: str | None = None,
    expected_report_sha256: str | None = None,
    expected_audit_path: Path | None = None,
    expected_audit_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the complete PASS-gated 48/24 export and all hash links."""

    train_path = _guard_candidate_path(train_path, role="train_jsonl")
    dev_path = _guard_candidate_path(dev_path, role="dev_jsonl")
    report_path = _guard_candidate_path(report_path, role="export_report")
    expected_root = report_path.parent
    official_paths = {
        "train": (expected_root / "prebuilt_chat_train.jsonl").resolve(),
        "dev": (expected_root / "prebuilt_chat_dev.jsonl").resolve(),
    }
    if report_path.name != "sft_export_report.json":
        raise ValueError("export report must be the official sft_export_report.json")
    if train_path != official_paths["train"] or dev_path != official_paths["dev"]:
        raise ValueError("train/dev files must be the official prebuilt_chat exports")

    actual_hashes = {
        "train": sha256_path(train_path),
        "dev": sha256_path(dev_path),
        "report": sha256_path(report_path),
    }
    for role, expected in (
        ("train", expected_train_sha256),
        ("dev", expected_dev_sha256),
        ("report", expected_report_sha256),
    ):
        if expected is not None:
            expected = _require_sha256(expected, role=f"expected_{role}_sha256")
            if actual_hashes[role] != expected:
                raise ValueError(f"{role} export hash differs from the pinned config")

    report = _load_json_object(report_path, role="SFT export report")
    expected_report_keys = {
        "version",
        "candidate_only",
        "information_gate",
        "data_audit",
        "records",
    }
    if set(report) != expected_report_keys:
        raise ValueError(
            "official SFT export report fields differ; "
            f"missing={sorted(expected_report_keys - set(report))}, "
            f"unexpected={sorted(set(report) - expected_report_keys)}"
        )
    if report.get("version") != OFFICIAL_SFT_EXPORT_REPORT_VERSION:
        raise ValueError("SFT export report version is not the official candidate version")
    if report.get("candidate_only") is not True:
        raise ValueError("SFT export report must be candidate_only=true")
    if report.get("information_gate") != "PASS":
        raise ValueError("SFT export report information gate is not PASS")

    records = _require_mapping(report.get("records"), role="SFT export records")
    if set(records) != set(EXPECTED_SPLIT_RECORDS):
        raise ValueError("SFT export report records must contain exactly train and dev")
    for split, expected_count in EXPECTED_SPLIT_RECORDS.items():
        entry = _require_mapping(records.get(split), role=f"records.{split}")
        if set(entry) != {"path", "records", "sha256"}:
            raise ValueError(f"records.{split} must contain exactly path, records, sha256")
        referenced = _report_reference(
            entry.get("path"), report_path=report_path, role=f"records.{split}"
        )
        if referenced != official_paths[split]:
            raise ValueError(f"records.{split}.path does not reference the official export")
        if int(entry.get("records", -1)) != expected_count:
            raise ValueError(f"records.{split} must declare exactly {expected_count} rows")
        declared_hash = _require_sha256(
            entry.get("sha256"), role=f"records.{split}.sha256"
        )
        if declared_hash != actual_hashes[split]:
            raise ValueError(f"records.{split} SHA-256 does not match the export")
        actual_count = len(base.read_jsonl(official_paths[split]))
        if actual_count != expected_count:
            raise ValueError(
                f"{split} export must contain exactly {expected_count} rows, got {actual_count}"
            )

    audit_entry = _require_mapping(report.get("data_audit"), role="data_audit")
    if set(audit_entry) != {"path", "sha256"}:
        raise ValueError("data_audit must contain exactly path and sha256")
    audit_path = _report_reference(
        audit_entry.get("path"), report_path=report_path, role="data_audit"
    )
    official_audit_path = (expected_root / "reports" / "data_audit.json").resolve()
    if audit_path != official_audit_path:
        raise ValueError("SFT export report does not reference the official data audit")
    audit_hash = sha256_path(audit_path)
    declared_audit_hash = _require_sha256(
        audit_entry.get("sha256"), role="data_audit.sha256"
    )
    if audit_hash != declared_audit_hash:
        raise ValueError("data audit hash differs from the SFT export report")
    if expected_audit_path is not None:
        expected_audit_path = _guard_candidate_path(
            expected_audit_path, role="expected_data_audit"
        )
        if audit_path != expected_audit_path:
            raise ValueError("data audit path differs from the pinned config")
    if expected_audit_sha256 is not None:
        expected_audit_sha256 = _require_sha256(
            expected_audit_sha256, role="expected_data_audit_sha256"
        )
        if audit_hash != expected_audit_sha256:
            raise ValueError("data audit hash differs from the pinned config")
    audit = _load_json_object(audit_path, role="data audit")
    _validate_pass_audit(audit)
    return {
        "train_sha256": actual_hashes["train"],
        "dev_sha256": actual_hashes["dev"],
        "report_sha256": actual_hashes["report"],
        "audit_path": audit_path,
        "audit_sha256": audit_hash,
        "audit": audit,
        "report": report,
    }


def validate_candidate_config(config: Mapping[str, Any]) -> None:
    """Apply the inherited engine contract plus candidate preregistration."""

    _BASE_VALIDATE_CONFIG(config)
    if config.get("schema_version") != "qwen_h1_meta_training_v0":
        raise ValueError("schema_version must be qwen_h1_meta_training_v0")
    if not str(config.get("run_name", "")).startswith("qwen_h1_meta_v0"):
        raise ValueError("run_name must use the independent qwen_h1_meta_v0 namespace")
    if tuple(config.get("training_seeds", ())) != TRAINING_SEEDS:
        raise ValueError(f"training_seeds must remain {list(TRAINING_SEEDS)}")

    protocol = _load_json_object(
        PACKAGE_ROOT / "protocol/meta_controller_protocol.json",
        role="frozen meta-controller protocol",
    )
    frozen_training = _require_mapping(protocol.get("training"), role="protocol.training")
    if config.get("generation") != frozen_training.get("generation"):
        raise ValueError("candidate generation settings differ from preregistration")

    training = config["training"]
    if int(training["max_steps"]) != 200:
        raise ValueError("candidate max_steps is preregistered to 200")
    if int(training["eval_steps"]) != 25 or int(training["save_steps"]) != 25:
        raise ValueError("candidate eval_steps/save_steps are preregistered to 25")
    if int(training["gradient_accumulation_steps"]) != 4:
        raise ValueError("candidate gradient_accumulation_steps is preregistered to 4")
    if int(training["per_device_train_batch_size"]) != 1:
        raise ValueError("candidate per-device train batch size is preregistered to 1")
    if int(training["per_device_eval_batch_size"]) != 1:
        raise ValueError("candidate per-device eval batch size is preregistered to 1")
    if float(training["learning_rate"]) != 0.0002:
        raise ValueError("candidate learning_rate is preregistered to 2e-4")
    if training.get("gradient_checkpointing") is not True:
        raise ValueError("candidate gradient checkpointing must remain enabled")
    if training.get("bf16") is not True or training.get("fp16") is not False:
        raise ValueError("candidate precision must remain bf16=true, fp16=false")
    if int(training.get("seed", -1)) not in TRAINING_SEEDS:
        raise ValueError("training.seed is not one of the three preregistered seeds")
    if int(training.get("data_seed", -1)) != int(training.get("seed", -2)):
        raise ValueError("training.data_seed must equal training.seed")
    if int(config["lora"]["r"]) != 16 or int(config["lora"]["alpha"]) != 32:
        raise ValueError("candidate LoRA rank/alpha are preregistered to 16/32")
    if float(config["lora"]["dropout"]) != 0.05:
        raise ValueError("candidate LoRA dropout is preregistered to 0.05")

    model = config["model"]
    expected_model_path = Path(str(frozen_training["base_model"])).resolve()
    if Path(str(model.get("id", ""))).expanduser().resolve() != expected_model_path:
        raise ValueError("candidate model.id must be the preregistered local snapshot")
    if model.get("source_id") != "Qwen/Qwen2.5-VL-3B-Instruct":
        raise ValueError("candidate model.source_id differs from the frozen Qwen identity")
    for key in ("revision", "processor_revision"):
        if model.get(key) != frozen_training["base_revision"]:
            raise ValueError(f"candidate model.{key} differs from preregistration")
    if (
        model.get("expected_local_snapshot_tree_sha256")
        != frozen_training["base_snapshot_tree_sha256"]
    ):
        raise ValueError("candidate model snapshot tree hash differs from preregistration")
    if model.get("local_files_only") is not True:
        raise ValueError("candidate base model loading must remain local-files-only")
    if model.get("torch_dtype") != "bfloat16":
        raise ValueError("candidate model torch_dtype must remain bfloat16")
    quantization = _require_mapping(
        model.get("quantization"), role="model.quantization"
    )
    if quantization.get("load_in_4bit") is not True:
        raise ValueError("candidate base model must remain 4-bit")
    if str(quantization.get("quant_type", "")).lower() != "nf4":
        raise ValueError("candidate quantization type must remain NF4")
    if quantization.get("compute_dtype") != "bfloat16":
        raise ValueError("candidate NF4 compute dtype must remain bfloat16")
    if quantization.get("double_quant") is not True:
        raise ValueError("candidate NF4 double quantization must remain enabled")

    data = config["data"]
    required_data_fields = {
        "train_jsonl",
        "dev_jsonl",
        "train_sha256",
        "dev_sha256",
        "export_report",
        "export_report_sha256",
        "data_audit",
        "data_audit_sha256",
        "train_export_report_key",
        "dev_export_report_key",
    }
    missing_data = sorted(required_data_fields - set(data))
    if missing_data:
        raise ValueError(f"candidate data config is missing gate fields: {missing_data}")
    if data.get("train_export_report_key") != "train":
        raise ValueError("train_export_report_key must remain train")
    if data.get("dev_export_report_key") != "dev":
        raise ValueError("dev_export_report_key must remain dev")
    if data.get("max_train_samples") is not None or data.get("max_dev_samples") is not None:
        raise ValueError("candidate training must use the complete 48/24 exports")
    if data.get("allowed_train_splits") != ["train"]:
        raise ValueError("allowed_train_splits must be exactly [train]")
    if data.get("allowed_dev_splits") != ["dev"]:
        raise ValueError("allowed_dev_splits must be exactly [dev]")
    if data.get("prohibit_frozen_splits") is not True:
        raise ValueError("prohibit_frozen_splits must remain true")

    train_path = _guard_candidate_path(
        _repo_path(data["train_jsonl"]), role="data.train_jsonl"
    )
    dev_path = _guard_candidate_path(
        _repo_path(data["dev_jsonl"]), role="data.dev_jsonl"
    )
    report_path = _guard_candidate_path(
        _repo_path(data["export_report"]), role="data.export_report"
    )
    audit_path = _guard_candidate_path(
        _repo_path(data["data_audit"]), role="data.data_audit"
    )
    validate_official_sft_export_bundle(
        train_path=train_path,
        dev_path=dev_path,
        report_path=report_path,
        expected_train_sha256=str(data["train_sha256"]),
        expected_dev_sha256=str(data["dev_sha256"]),
        expected_report_sha256=str(data["export_report_sha256"]),
        expected_audit_path=audit_path,
        expected_audit_sha256=str(data["data_audit_sha256"]),
    )
    _guard_candidate_path(
        _repo_path(training["output_dir"]),
        role="training.output_dir",
        must_exist=False,
    )


def _image_placeholder_count(messages: Any) -> int:
    if not isinstance(messages, list):
        raise ValueError("prompt must be a list")
    count = 0
    for message in messages:
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), list):
            raise ValueError("prompt messages require list-valued content")
        count += sum(
            isinstance(item, Mapping) and item.get("type") == "image"
            for item in message["content"]
        )
    return count


def validate_prebuilt_row(
    row: Mapping[str, Any],
    *,
    allowed_splits: set[str],
    image_root: Path,
    verify_image: bool,
) -> Path:
    """Validate one exact prompt-contract row without supervisor target code."""

    expected_row_keys = {"example_id", "split", "images", "prompt", "completion", "metadata"}
    if set(row) != expected_row_keys:
        raise ValueError(
            f"{row.get('example_id', '<unknown>')}: unexpected row fields "
            f"{sorted(set(row) - expected_row_keys)}; missing={sorted(expected_row_keys - set(row))}"
        )
    example_id = row.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise ValueError("example_id must be a non-empty string")
    split = base.row_split(row)
    if split not in allowed_splits:
        raise ValueError(f"{example_id}: split {split!r} not in {sorted(allowed_splits)}")
    if split not in {"train", "dev"}:
        raise ValueError(f"{example_id}: only candidate train/dev rows are accepted")

    prompt = row.get("prompt")
    try:
        validate_prompt(prompt)
    except (MetaContractError, ValueError) as exc:
        raise ValueError(f"{example_id}: {exc}") from exc
    if _image_placeholder_count(prompt) != 1:
        raise ValueError(f"{example_id}: exactly one current image placeholder is required")

    completion = row.get("completion")
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValueError(f"{example_id}: completion must contain one assistant message")
    message = completion[0]
    if (
        not isinstance(message, Mapping)
        or set(message) != {"role", "content"}
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), list)
        or len(message["content"]) != 1
        or not isinstance(message["content"][0], Mapping)
        or set(message["content"][0]) != {"type", "text"}
        or message["content"][0].get("type") != "text"
        or not isinstance(message["content"][0].get("text"), str)
    ):
        raise ValueError(f"{example_id}: assistant completion must be exactly one text item")
    completion_text = message["content"][0]["text"]
    target = parse_meta_output(completion_text).to_dict()
    if completion_text != canonical_meta_text(target):
        raise ValueError(f"{example_id}: assistant target is not frozen compact canonical JSON")

    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ValueError(f"{example_id}: exactly one image path is required")
    image_path = Path(images[0]).expanduser()
    image_path = (
        image_path.resolve()
        if image_path.is_absolute()
        else (image_root.resolve() / image_path).resolve()
    )
    image_path = _guard_candidate_path(image_path, role=f"{example_id}.image")

    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{example_id}: metadata must be an object")
    expected_hash = metadata.get("image_sha256")
    if not isinstance(expected_hash, str) or not HEX_SHA256.fullmatch(expected_hash):
        raise ValueError(f"{example_id}: metadata.image_sha256 must be a lowercase SHA-256")
    if verify_image:
        from PIL import Image

        with Image.open(image_path) as image:
            if image.size != (1024, 1024):
                raise ValueError(
                    f"{example_id}: current sensor-frame image must be 1024x1024, "
                    f"got {image.size}"
                )
            image.verify()
        actual = sha256_path(image_path)
        if actual != expected_hash:
            raise ValueError(
                f"{example_id}: image SHA-256 mismatch; expected {expected_hash}, got {actual}"
            )
    return image_path


def prepare_rows(
    config: Mapping[str, Any], *, verify_images: bool = True
) -> base.PreparedRows:
    validate_candidate_config(config)
    data = config["data"]
    train_path = _guard_candidate_path(_repo_path(data["train_jsonl"]), role="train_jsonl")
    dev_path = _guard_candidate_path(_repo_path(data["dev_jsonl"]), role="dev_jsonl")
    report_path = _guard_candidate_path(_repo_path(data["export_report"]), role="export_report")
    image_root = _repo_path(data.get("image_root", "."))

    expected_train_hash = str(data["train_sha256"])
    expected_dev_hash = str(data["dev_sha256"])
    validate_official_sft_export_bundle(
        train_path=train_path,
        dev_path=dev_path,
        report_path=report_path,
        expected_train_sha256=expected_train_hash,
        expected_dev_sha256=expected_dev_hash,
        expected_report_sha256=str(data["export_report_sha256"]),
        expected_audit_path=_guard_candidate_path(
            _repo_path(data["data_audit"]), role="data_audit"
        ),
        expected_audit_sha256=str(data["data_audit_sha256"]),
    )

    train_rows = base.limit_rows(base.read_jsonl(train_path), data.get("max_train_samples"))
    dev_rows = base.limit_rows(base.read_jsonl(dev_path), data.get("max_dev_samples"))
    if len(train_rows) != EXPECTED_SPLIT_RECORDS["train"]:
        raise ValueError("candidate training requires all 48 train records")
    if len(dev_rows) != EXPECTED_SPLIT_RECORDS["dev"]:
        raise ValueError("candidate development requires all 24 dev records")
    for row in train_rows:
        validate_prebuilt_row(
            row,
            allowed_splits=set(data["allowed_train_splits"]),
            image_root=image_root,
            verify_image=verify_images,
        )
    for row in dev_rows:
        validate_prebuilt_row(
            row,
            allowed_splits=set(data["allowed_dev_splits"]),
            image_root=image_root,
            verify_image=verify_images,
        )
    return base.PreparedRows(train_path, dev_path, image_root, train_rows, dev_rows)


def row_to_example(row: Mapping[str, Any], image_root: Path) -> dict[str, Any]:
    path = validate_prebuilt_row(
        row,
        allowed_splits={str(base.row_split(row))},
        image_root=image_root,
        verify_image=False,
    )
    return {
        "images": [base.load_rgb_image(path)],
        "prompt": row["prompt"],
        "completion": row["completion"],
    }


def candidate_runtime_manifest_base(**kwargs: Any) -> dict[str, Any]:
    manifest = _BASE_RUNTIME_MANIFEST_BASE(**kwargs)
    manifest["manifest_version"] = "qwen_h1_meta_training_run_v0"
    manifest["base_training_engine"] = "qwen_vl_supervisor_v1.train_qlora"
    manifest["independent_adapter_name"] = "qwen_h1_meta_v0"
    manifest["data"]["model_visible_allowlist_validator"] = (
        "qwen_h1_meta_v0_candidate.training.validate_prebuilt_row"
    )
    manifest["data"]["frozen_or_protected_predictions_opened"] = False
    return manifest


@contextmanager
def candidate_training_engine() -> Any:
    """Temporarily inject candidate contracts into the unchanged engine."""

    replacements = {
        "validate_config": validate_candidate_config,
        "prepare_rows": prepare_rows,
        "row_to_example": row_to_example,
        "runtime_manifest_base": candidate_runtime_manifest_base,
    }
    with _PATCH_LOCK:
        previous = {name: getattr(base, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(base, name, value)
            yield
        finally:
            for name, value in previous.items():
                setattr(base, name, value)


def train(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    local_rank_arg: int | None = None,
    save_at_global_step: int | None = None,
    stop_after_global_step: int | None = None,
) -> dict[str, Any]:
    validate_candidate_config(config)
    with candidate_training_engine():
        return base.train(
            config,
            config_path,
            local_rank_arg=local_rank_arg,
            save_at_global_step=save_at_global_step,
            stop_after_global_step=stop_after_global_step,
        )


def _with_default_config(argv: Sequence[str] | None) -> list[str]:
    values = list(argv or [])
    if not any(value == "--config" or value.startswith("--config=") for value in values):
        values = ["--config", str(DEFAULT_CONFIG), *values]
    return values


def main(argv: Sequence[str] | None = None) -> None:
    args = base.parse_args(_with_default_config(argv))
    config, config_path = base.resolve_config(args)
    validate_candidate_config(config)
    if args.print_resolved_config:
        print(json.dumps(config, indent=2, sort_keys=True))
        return
    if args.validate_only:
        rows = prepare_rows(config, verify_images=True)
        model_cfg = config["model"]
        model_source_identity = base.resolve_model_source_identity(
            model_id=str(model_cfg["id"]),
            source_id=str(model_cfg.get("source_id", model_cfg["id"])),
            revision=str(model_cfg["revision"]),
            processor_revision=str(model_cfg["processor_revision"]),
            repository_root=REPOSITORY_ROOT,
            expected_local_tree_sha256=model_cfg.get(
                "expected_local_snapshot_tree_sha256"
            ),
        )
        print(
            json.dumps(
                {
                    "status": "valid_candidate_meta_training_inputs",
                    "config": str(config_path),
                    "train_jsonl": str(rows.train_path),
                    "train_records": len(rows.train_rows),
                    "dev_jsonl": str(rows.dev_path),
                    "dev_records": len(rows.dev_rows),
                    "images_verified": len(rows.train_rows) + len(rows.dev_rows),
                    "model_source_identity": model_source_identity,
                    "model_loaded": False,
                    "frozen_or_protected_predictions_opened": False,
                    "validator": "qwen_h1_meta_v0_candidate.training.validate_prebuilt_row",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    manifest = train(
        config,
        config_path,
        local_rank_arg=args.local_rank,
        save_at_global_step=args.save_at_global_step,
        stop_after_global_step=args.stop_after_global_step,
    )
    if int(base.os.environ.get("RANK", "0")) == 0:
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
