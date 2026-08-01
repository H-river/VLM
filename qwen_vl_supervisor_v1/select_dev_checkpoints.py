#!/usr/bin/env python3
"""Select one Qwen-VL checkpoint per seed using complete dev-only evidence.

The selector does not run a model.  It enumerates the exact checkpoint schedule
implied by the hashed server config, verifies every generation/evaluation
artifact, recomputes the offline report from raw development predictions, and
then applies the frozen ranking rule.  Missing checkpoints or evidence are
fatal; protected/frozen use cannot be represented as successful selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluate_offline import evaluate_files
from .train_qlora import checkpoint_step, load_yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "qwen_vl_supervisor_dev_checkpoint_selection_v1.0.0"
TREE_HASH_ALGORITHM = "sha256_tree_relative_path_nul_file_sha256_lf_v1"
FILE_HASH_ALGORITHM = "sha256_file_v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CHECKPOINT_NAME = re.compile(r"^checkpoint-([0-9]+)$")
RANKING_RULE = (
    "highest_joint_diagnosis_policy_action_exact_accuracy",
    "highest_diagnosis_macro_f1",
    "highest_valid_json_rate",
    "lowest_optimizer_step",
    "lexicographically_lowest_checkpoint_tree_sha256",
)
TOP_KEYS = {
    "schema_version",
    "artifact_kind",
    "selection_split",
    "ranking_rule",
    "training_config",
    "dev_data",
    "training_seeds",
    "checkpoint_schedule",
    "seed_results",
    "coverage",
    "protected_or_frozen_data_used",
    "frozen_predictions_opened",
    "selector",
}
CANDIDATE_KEYS = {
    "training_seed",
    "optimizer_step",
    "selection_rank",
    "checkpoint",
    "generation_report",
    "predictions",
    "offline_report",
    "dev_coverage",
    "ranking_metrics",
    "eligible",
    "protected_or_frozen_data_used",
    "frozen_predictions_opened",
}
PROTECTED_FRAGMENTS = ("frozen", "heldout", "held_out", "severity_ood", "test")


class SelectionError(ValueError):
    """Raised when checkpoint-selection evidence violates the frozen contract."""


def _exact_keys(value: Any, expected: set[str], location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionError(f"{location} must be an object")
    actual = set(value)
    if actual != expected:
        raise SelectionError(
            f"{location} keys differ; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    return value


def _read_json(path: Path, *, location: str) -> dict[str, Any]:
    _require_regular_file(path, location=location)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                SelectionError(f"{location} contains non-finite JSON value {item}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"cannot read {location} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SelectionError(f"{location} must contain one JSON object")
    return value


def _read_jsonl(path: Path, *, location: str) -> list[dict[str, Any]]:
    _require_regular_file(path, location=location)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    parse_constant=lambda item: (_ for _ in ()).throw(
                        SelectionError(
                            f"{location}:{line_number} contains non-finite JSON value {item}"
                        )
                    ),
                )
            except json.JSONDecodeError as exc:
                raise SelectionError(f"invalid {location} JSONL at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SelectionError(f"{location}:{line_number} must be an object")
            rows.append(row)
    if not rows:
        raise SelectionError(f"{location} is empty: {path}")
    return rows


def _require_regular_file(path: Path, *, location: str) -> None:
    if path.is_symlink():
        raise SelectionError(f"{location} must not be a symbolic link: {path}")
    if not path.is_file():
        raise SelectionError(f"{location} does not exist or is not a regular file: {path}")


def _sha256_file(path: Path) -> str:
    _require_regular_file(path, location="hashed file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(path: Path) -> str:
    if path.is_symlink():
        raise SelectionError(f"checkpoint must not be a symbolic link: {path}")
    if not path.is_dir():
        raise SelectionError(f"checkpoint is not a directory: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file() or item.is_symlink())
    if not files:
        raise SelectionError(f"checkpoint tree is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise SelectionError(f"checkpoint tree contains a symbolic link: {item}")
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _inside(root: Path, path: Path, *, location: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SelectionError(f"{location} escapes repository root: {resolved}") from exc
    return resolved


def _display(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        # Production inputs are required inside the repository.  This fallback
        # only permits the selector's own immutable implementation/schema to
        # remain identifiable when the pure builder is tested in a temp root.
        return str(path.resolve())


def _file_entry(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": _display(root, path),
        "hash_algorithm": FILE_HASH_ALGORITHM,
        "sha256": _sha256_file(path),
    }


def _tree_entry(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": _display(root, path),
        "hash_algorithm": TREE_HASH_ALGORITHM,
        "sha256": _hash_tree(path),
    }


def _finite_unit(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SelectionError(f"{location} must be finite and in [0,1], got {value!r}")
    return result


def _training_contract(config: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    seeds = config.get("training_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise SelectionError("training config must declare unique integer training_seeds")
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise SelectionError("training config has no training object")
    max_steps = int(training.get("max_steps", 0))
    save_steps = int(training.get("save_steps", 0))
    save_total_limit = int(training.get("save_total_limit", 0))
    if max_steps <= 0 or save_steps <= 0:
        raise SelectionError("training max_steps and save_steps must be positive")
    steps = list(range(save_steps, max_steps + 1, save_steps))
    if not steps or steps[-1] != max_steps:
        steps.append(max_steps)
    if save_total_limit < len(steps):
        raise SelectionError(
            "save_total_limit cannot retain the complete configured checkpoint schedule: "
            f"need {len(steps)}, configured {save_total_limit}"
        )
    return list(seeds), steps


def _validate_dev_inputs(
    *, root: Path, config: Mapping[str, Any], dev_sft: Path, dev_manifest: Path
) -> tuple[int, str, str]:
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise SelectionError("training config has no data object")
    configured_dev = _resolve(root, str(data.get("dev_jsonl", "")))
    if configured_dev != dev_sft:
        raise SelectionError(
            f"--dev-sft must equal config.data.dev_jsonl: {dev_sft} != {configured_dev}"
        )
    configured_dev_text = str(data.get("dev_jsonl", "")).lower()
    if any(fragment in configured_dev_text for fragment in PROTECTED_FRAGMENTS):
        raise SelectionError("development SFT path looks protected or frozen")
    expected_sft_hash = str(data.get("dev_sha256", ""))
    actual_sft_hash = _sha256_file(dev_sft)
    if actual_sft_hash != expected_sft_hash:
        raise SelectionError(
            f"development SFT hash mismatch: expected {expected_sft_hash}, got {actual_sft_hash}"
        )

    sft_rows = _read_jsonl(dev_sft, location="development SFT")
    manifest_rows = _read_jsonl(dev_manifest, location="development manifest")
    sft_ids: list[str] = []
    for index, row in enumerate(sft_rows):
        if row.get("split") != "dev":
            raise SelectionError(f"development SFT row {index} is not split='dev'")
        sample_id = row.get("example_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise SelectionError(f"development SFT row {index} has no example_id")
        sft_ids.append(sample_id)
    manifest_ids: list[str] = []
    for index, row in enumerate(manifest_rows):
        if row.get("split") != "dev":
            raise SelectionError(f"development manifest row {index} is not split='dev'")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise SelectionError(f"development manifest row {index} has no sample_id")
        manifest_ids.append(sample_id)
    if len(set(sft_ids)) != len(sft_ids) or len(set(manifest_ids)) != len(manifest_ids):
        raise SelectionError("development SFT/manifest sample IDs must be unique")
    if set(sft_ids) != set(manifest_ids):
        raise SelectionError("development SFT and source manifest sample IDs differ")
    return len(manifest_ids), actual_sft_hash, _sha256_file(dev_manifest)


def _checkpoint_directories(seed_dir: Path) -> dict[int, Path]:
    if seed_dir.is_symlink() or not seed_dir.is_dir():
        raise SelectionError(f"training seed directory is missing or symbolic: {seed_dir}")
    checkpoints: dict[int, Path] = {}
    for path in seed_dir.iterdir():
        match = CHECKPOINT_NAME.fullmatch(path.name)
        if match is None:
            continue
        if path.is_symlink() or not path.is_dir():
            raise SelectionError(f"checkpoint path is not a regular directory: {path}")
        step = int(match.group(1))
        if step in checkpoints:
            raise SelectionError(f"duplicate checkpoint step {step} under {seed_dir}")
        checkpoints[step] = path.resolve()
    return checkpoints


def _validate_seed_run_manifest(
    *,
    root: Path,
    seed_dir: Path,
    seed: int,
    max_step: int,
    config_hash: str,
    config: Mapping[str, Any],
) -> dict[str, str]:
    path = seed_dir / "run_manifest.latest.json"
    report = _read_json(path, location=f"seed {seed} training run manifest")
    if report.get("status") != "completed":
        raise SelectionError(f"seed {seed} training run is not completed")
    training = report.get("training")
    result = report.get("training_result")
    data = report.get("data")
    safe = report.get("safe_checkpointing")
    if not all(isinstance(value, Mapping) for value in (training, result, data, safe)):
        raise SelectionError(f"seed {seed} run manifest is missing required objects")
    if (
        training.get("seed") != seed
        or training.get("data_seed") != seed
        or int(training.get("max_steps", -1)) != max_step
        or int(result.get("global_step", -1)) != max_step
    ):
        raise SelectionError(f"seed {seed} run manifest seed/step contract differs")
    if report.get("config_sha256") != config_hash:
        raise SelectionError(f"seed {seed} run manifest used a different training config")
    configured_data = config["data"]
    if (
        data.get("train_sha256") != configured_data.get("train_sha256")
        or data.get("dev_sha256") != configured_data.get("dev_sha256")
        or data.get("frozen_predictions_opened") is not False
    ):
        raise SelectionError(f"seed {seed} run manifest data provenance differs")
    if safe.get("legacy_optimizer_or_rng_pickle_loaded") is not False:
        raise SelectionError(f"seed {seed} run manifest does not prove safe checkpoint loading")
    return _file_entry(root, path)


def _validate_generation_report(
    *,
    report_path: Path,
    predictions_path: Path,
    checkpoint: Path,
    seed: int,
    dev_count: int,
    dev_sft_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    report = _read_json(report_path, location="development generation report")
    data = report.get("data")
    adapter = report.get("adapter")
    progress = report.get("progress")
    decoding = report.get("decoding")
    image_integrity = report.get("image_integrity")
    config = report.get("config")
    if not all(
        isinstance(value, Mapping)
        for value in (data, adapter, progress, decoding, image_integrity, config)
    ):
        raise SelectionError(f"generation report lacks required objects: {report_path}")
    if (
        report.get("status") != "completed"
        or report.get("seed") != seed
        or data.get("split") != "dev"
        or data.get("sha256") != dev_sft_hash
        or data.get("records") != dev_count
        or data.get("frozen_or_protected_predictions_opened") is not False
        or decoding.get("json_extraction_or_repair") is not False
        or progress.get("completed_records") != dev_count
        or progress.get("expected_records") != dev_count
        or image_integrity.get("verified_records_before_model_load") != dev_count
        or config.get("sha256") != config_hash
    ):
        raise SelectionError(f"generation report violates dev-only completeness: {report_path}")
    if Path(str(adapter.get("path"))).resolve() != checkpoint:
        raise SelectionError(f"generation report adapter does not match checkpoint: {report_path}")
    if Path(str(report.get("output"))).resolve() != predictions_path:
        raise SelectionError(f"generation report output path does not match predictions: {report_path}")
    hashes = adapter.get("hashes")
    if not isinstance(hashes, Mapping):
        raise SelectionError(f"generation report has no adapter hashes: {report_path}")
    possible_adapter = {
        name: checkpoint / name
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "generation_config.json",
        )
    }
    required_adapter = {
        name: path for name, path in possible_adapter.items() if path.is_file()
    }
    if "adapter_model.bin" in required_adapter:
        raise SelectionError("pickle adapter_model.bin is forbidden; safetensors is required")
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(required_adapter):
        raise SelectionError(f"checkpoint lacks the required safetensors adapter: {checkpoint}")
    if set(hashes) != set(required_adapter):
        raise SelectionError(
            f"generation report adapter fingerprint differs from checkpoint: {report_path}"
        )
    for name, path in required_adapter.items():
        if hashes[name] != _sha256_file(path):
            raise SelectionError(f"generation adapter hash mismatch for {path}")
    return report


def _candidate(
    *,
    root: Path,
    seed: int,
    step: int,
    checkpoint: Path,
    evidence_dir: Path,
    dev_manifest: Path,
    dev_count: int,
    dev_sft_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    if checkpoint_step(checkpoint) != step:
        raise SelectionError(f"checkpoint step verification failed: {checkpoint}")
    for required in ("adapter_config.json", "adapter_model.safetensors"):
        _require_regular_file(checkpoint / required, location="checkpoint adapter")

    predictions = evidence_dir / "predictions_dev.jsonl"
    generation_report = evidence_dir / "predictions_dev.jsonl.report.json"
    offline_report = evidence_dir / "offline_dev.json"
    _require_regular_file(predictions, location="development predictions")
    _validate_generation_report(
        report_path=generation_report,
        predictions_path=predictions,
        checkpoint=checkpoint,
        seed=seed,
        dev_count=dev_count,
        dev_sft_hash=dev_sft_hash,
        config_hash=config_hash,
    )
    stored_report = _read_json(offline_report, location="development offline report")
    recomputed_report = evaluate_files(
        dev_manifest,
        predictions,
        expected_seeds=[seed],
    )
    if stored_report != recomputed_report:
        raise SelectionError(
            f"stored offline report is not the exact current reducer result: {offline_report}"
        )
    enforcement = stored_report.get("seed_enforcement")
    per_seed = stored_report.get("per_seed")
    if (
        enforcement != {"enabled": True, "expected_seeds": [seed]}
        or not isinstance(per_seed, list)
        or len(per_seed) != 1
        or per_seed[0].get("seed") != seed
        or per_seed[0].get("manifest_count") != dev_count
        or per_seed[0].get("supplied_prediction_count") != dev_count
        or per_seed[0].get("missing_prediction_count") != 0
        or per_seed[0].get("coverage_rate") != 1.0
    ):
        raise SelectionError(f"offline report lacks exact seed/coverage evidence: {offline_report}")
    metrics = {
        "joint_exact_accuracy": _finite_unit(
            per_seed[0].get("joint_exact_accuracy"), location="joint exact accuracy"
        ),
        "diagnosis_macro_f1": _finite_unit(
            per_seed[0].get("diagnosis_macro_f1"), location="diagnosis macro F1"
        ),
        "valid_json_rate": _finite_unit(
            per_seed[0].get("valid_json_rate"), location="valid JSON rate"
        ),
    }
    return {
        "training_seed": seed,
        "optimizer_step": step,
        "selection_rank": 0,
        "checkpoint": _tree_entry(root, checkpoint),
        "generation_report": _file_entry(root, generation_report),
        "predictions": _file_entry(root, predictions),
        "offline_report": _file_entry(root, offline_report),
        "dev_coverage": {
            "manifest_records": dev_count,
            "supplied_predictions": dev_count,
            "missing_predictions": 0,
            "coverage_rate": 1.0,
            "expected_seed_enforced": True,
        },
        "ranking_metrics": metrics,
        "eligible": True,
        "protected_or_frozen_data_used": False,
        "frozen_predictions_opened": False,
    }


def _rank_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, int, str]:
    metrics = candidate["ranking_metrics"]
    return (
        -float(metrics["joint_exact_accuracy"]),
        -float(metrics["diagnosis_macro_f1"]),
        -float(metrics["valid_json_rate"]),
        int(candidate["optimizer_step"]),
        str(candidate["checkpoint"]["sha256"]),
    )


def build_selection_artifact(
    *,
    repository_root: Path,
    training_config: Path,
    training_root: Path,
    evidence_root: Path,
    dev_sft: Path,
    dev_manifest: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    paths = {
        name: _inside(root, path, location=name)
        for name, path in {
            "training config": training_config,
            "training root": training_root,
            "evidence root": evidence_root,
            "development SFT": dev_sft,
            "development manifest": dev_manifest,
        }.items()
    }
    training_config = paths["training config"]
    training_root = paths["training root"]
    evidence_root = paths["evidence root"]
    dev_sft = paths["development SFT"]
    dev_manifest = paths["development manifest"]
    _require_regular_file(training_config, location="training config")
    if training_root.is_symlink() or not training_root.is_dir():
        raise SelectionError(f"training root is missing or symbolic: {training_root}")
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise SelectionError(f"evidence root is missing or symbolic: {evidence_root}")

    config = load_yaml(training_config)
    seeds, expected_steps = _training_contract(config)
    config_hash = _sha256_file(training_config)
    dev_count, dev_sft_hash, dev_manifest_hash = _validate_dev_inputs(
        root=root,
        config=config,
        dev_sft=dev_sft,
        dev_manifest=dev_manifest,
    )

    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = training_root / f"seed_{seed}"
        observed = _checkpoint_directories(seed_dir)
        if sorted(observed) != expected_steps:
            raise SelectionError(
                f"seed {seed} checkpoint coverage mismatch; "
                f"expected={expected_steps}, observed={sorted(observed)}"
            )
        run_manifest = _validate_seed_run_manifest(
            root=root,
            seed_dir=seed_dir,
            seed=seed,
            max_step=expected_steps[-1],
            config_hash=config_hash,
            config=config,
        )
        candidates = [
            _candidate(
                root=root,
                seed=seed,
                step=step,
                checkpoint=observed[step],
                evidence_dir=evidence_root / f"seed_{seed}" / f"checkpoint-{step}",
                dev_manifest=dev_manifest,
                dev_count=dev_count,
                dev_sft_hash=dev_sft_hash,
                config_hash=config_hash,
            )
            for step in expected_steps
        ]
        candidates.sort(key=_rank_key)
        for rank, candidate in enumerate(candidates, start=1):
            candidate["selection_rank"] = rank
        selected = candidates[0]
        seed_results.append(
            {
                "training_seed": seed,
                "training_run_manifest": run_manifest,
                "expected_checkpoint_steps": expected_steps,
                "observed_checkpoint_steps": sorted(observed),
                "complete_checkpoint_coverage": True,
                "candidates": candidates,
                "selected": {
                    "optimizer_step": selected["optimizer_step"],
                    "selection_rank": 1,
                    "checkpoint": selected["checkpoint"],
                    "ranking_metrics": selected["ranking_metrics"],
                },
            }
        )

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "deterministic_dev_only_checkpoint_selection",
        "selection_split": "dev",
        "ranking_rule": list(RANKING_RULE),
        "training_config": _file_entry(root, training_config),
        "dev_data": {
            "sft": _file_entry(root, dev_sft),
            "source_manifest": _file_entry(root, dev_manifest),
            "record_count": dev_count,
            "sample_id_sets_equal": True,
        },
        "training_seeds": seeds,
        "checkpoint_schedule": {
            "expected_steps": expected_steps,
            "expected_checkpoints_per_seed": len(expected_steps),
            "source": "training.max_steps_training.save_steps_and_save_total_limit",
        },
        "seed_results": seed_results,
        "coverage": {
            "expected_training_seeds": len(seeds),
            "observed_training_seeds": len(seed_results),
            "expected_checkpoints_total": len(seeds) * len(expected_steps),
            "observed_checkpoints_total": sum(
                len(result["candidates"]) for result in seed_results
            ),
            "complete": True,
        },
        "protected_or_frozen_data_used": False,
        "frozen_predictions_opened": False,
        "selector": {
            "implementation": _file_entry(root, Path(__file__).resolve()),
            "json_schema": _file_entry(
                root,
                Path(__file__).resolve().parent
                / "schema/dev_selection_artifact.schema.json",
            ),
            "offline_report_recomputed_from_raw_predictions": True,
            "timestamp_in_artifact": False,
        },
    }
    validate_selection_artifact(artifact)
    # Ensure the separately computed values were not accidentally shadowed.
    if artifact["dev_data"]["sft"]["sha256"] != dev_sft_hash:
        raise AssertionError("development SFT hash changed during selection")
    if artifact["dev_data"]["source_manifest"]["sha256"] != dev_manifest_hash:
        raise AssertionError("development manifest hash changed during selection")
    return artifact


def _validate_artifact_path_entry(
    value: Any, *, location: str, tree: bool = False
) -> None:
    entry = _exact_keys(value, {"path", "hash_algorithm", "sha256"}, location)
    if not isinstance(entry["path"], str) or not entry["path"]:
        raise SelectionError(f"{location}.path must be a nonempty string")
    expected_algorithm = TREE_HASH_ALGORITHM if tree else FILE_HASH_ALGORITHM
    if entry["hash_algorithm"] != expected_algorithm:
        raise SelectionError(f"{location}.hash_algorithm changed")
    if not isinstance(entry["sha256"], str) or not HEX64.fullmatch(entry["sha256"]):
        raise SelectionError(f"{location}.sha256 is invalid")


def validate_selection_artifact(artifact: Any) -> None:
    """Strict equivalent validator for the published JSON Schema."""

    artifact = _exact_keys(artifact, TOP_KEYS, "selection artifact")
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise SelectionError("selection artifact schema_version changed")
    if artifact["artifact_kind"] != "deterministic_dev_only_checkpoint_selection":
        raise SelectionError("selection artifact kind changed")
    if artifact["selection_split"] != "dev":
        raise SelectionError("checkpoint selection must use dev only")
    if artifact["ranking_rule"] != list(RANKING_RULE):
        raise SelectionError("checkpoint ranking rule changed")
    if artifact["protected_or_frozen_data_used"] is not False:
        raise SelectionError("protected/frozen data use is forbidden")
    if artifact["frozen_predictions_opened"] is not False:
        raise SelectionError("frozen predictions must remain unopened")

    _validate_artifact_path_entry(artifact["training_config"], location="training_config")
    dev_data = _exact_keys(
        artifact["dev_data"],
        {"sft", "source_manifest", "record_count", "sample_id_sets_equal"},
        "dev_data",
    )
    _validate_artifact_path_entry(dev_data["sft"], location="dev_data.sft")
    _validate_artifact_path_entry(
        dev_data["source_manifest"], location="dev_data.source_manifest"
    )
    if (
        isinstance(dev_data["record_count"], bool)
        or not isinstance(dev_data["record_count"], int)
        or dev_data["record_count"] <= 0
        or dev_data["sample_id_sets_equal"] is not True
    ):
        raise SelectionError("dev_data does not prove a nonempty equal-ID development set")

    selector = _exact_keys(
        artifact["selector"],
        {
            "implementation",
            "json_schema",
            "offline_report_recomputed_from_raw_predictions",
            "timestamp_in_artifact",
        },
        "selector",
    )
    _validate_artifact_path_entry(selector["implementation"], location="selector.implementation")
    _validate_artifact_path_entry(selector["json_schema"], location="selector.json_schema")
    if (
        selector["offline_report_recomputed_from_raw_predictions"] is not True
        or selector["timestamp_in_artifact"] is not False
    ):
        raise SelectionError("selector audit flags changed")

    seeds = artifact["training_seeds"]
    schedule = artifact["checkpoint_schedule"]
    results = artifact["seed_results"]
    coverage = artifact["coverage"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise SelectionError("artifact training_seeds must be a unique nonempty list")
    schedule = _exact_keys(
        schedule,
        {"expected_steps", "expected_checkpoints_per_seed", "source"},
        "checkpoint_schedule",
    )
    if not isinstance(results, list):
        raise SelectionError("artifact seed_results are malformed")
    steps = schedule.get("expected_steps")
    if (
        not isinstance(steps, list)
        or not steps
        or any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps)
        or steps != sorted(set(steps))
    ):
        raise SelectionError("artifact expected checkpoint steps are invalid")
    if schedule.get("expected_checkpoints_per_seed") != len(steps):
        raise SelectionError("artifact checkpoint count disagrees with schedule")
    if schedule.get("source") != "training.max_steps_training.save_steps_and_save_total_limit":
        raise SelectionError("artifact checkpoint schedule source changed")
    if [result.get("training_seed") for result in results if isinstance(result, Mapping)] != seeds:
        raise SelectionError("artifact seed results do not exactly cover the declared seed order")

    total = 0
    for result in results:
        result = _exact_keys(
            result,
            {
                "training_seed",
                "training_run_manifest",
                "expected_checkpoint_steps",
                "observed_checkpoint_steps",
                "complete_checkpoint_coverage",
                "candidates",
                "selected",
            },
            "seed result",
        )
        _validate_artifact_path_entry(
            result["training_run_manifest"], location="seed result training_run_manifest"
        )
        if (
            result.get("expected_checkpoint_steps") != steps
            or result.get("observed_checkpoint_steps") != steps
            or result.get("complete_checkpoint_coverage") is not True
        ):
            raise SelectionError("seed result does not prove complete checkpoint coverage")
        candidates = result.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(steps):
            raise SelectionError("seed result candidate count is incomplete")
        total += len(candidates)
        ranks = []
        for candidate in candidates:
            candidate = _exact_keys(candidate, CANDIDATE_KEYS, "selection candidate")
            if (
                candidate["training_seed"] != result["training_seed"]
                or candidate["optimizer_step"] not in steps
                or candidate["eligible"] is not True
                or candidate["protected_or_frozen_data_used"] is not False
                or candidate["frozen_predictions_opened"] is not False
            ):
                raise SelectionError("candidate violates seed/eligibility/protection contract")
            ranks.append(candidate["selection_rank"])
            metrics = candidate["ranking_metrics"]
            if not isinstance(metrics, Mapping):
                raise SelectionError("candidate ranking_metrics must be an object")
            for name in (
                "joint_exact_accuracy",
                "diagnosis_macro_f1",
                "valid_json_rate",
            ):
                _finite_unit(metrics.get(name), location=f"candidate {name}")
            if candidate["dev_coverage"] != {
                "manifest_records": artifact["dev_data"]["record_count"],
                "supplied_predictions": artifact["dev_data"]["record_count"],
                "missing_predictions": 0,
                "coverage_rate": 1.0,
                "expected_seed_enforced": True,
            }:
                raise SelectionError("candidate development coverage is incomplete")
            for file_key in ("generation_report", "predictions", "offline_report"):
                _validate_artifact_path_entry(
                    candidate[file_key], location=f"candidate {file_key}"
                )
            _validate_artifact_path_entry(
                candidate["checkpoint"], location="candidate checkpoint", tree=True
            )
        if ranks != list(range(1, len(candidates) + 1)):
            raise SelectionError("candidate selection ranks are incomplete or unordered")
        if candidates != sorted(candidates, key=_rank_key):
            raise SelectionError("candidate list is not in frozen ranking order")
        selected = result.get("selected")
        if not isinstance(selected, Mapping) or selected != {
            "optimizer_step": candidates[0]["optimizer_step"],
            "selection_rank": 1,
            "checkpoint": candidates[0]["checkpoint"],
            "ranking_metrics": candidates[0]["ranking_metrics"],
        }:
            raise SelectionError("selected checkpoint is not rank one")

    if coverage != {
        "expected_training_seeds": len(seeds),
        "observed_training_seeds": len(results),
        "expected_checkpoints_total": len(seeds) * len(steps),
        "observed_checkpoints_total": total,
        "complete": True,
    }:
        raise SelectionError("artifact global checkpoint coverage is inconsistent")


def _canonical_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _common_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.repository_root).resolve()
    return {
        "repository_root": root,
        "training_config": _resolve(root, args.training_config),
        "training_root": _resolve(root, args.training_root),
        "evidence_root": _resolve(root, args.evidence_root),
        "dev_sft": _resolve(root, args.dev_sft),
        "dev_manifest": _resolve(root, args.dev_manifest),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("select", "verify"))
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--dev-sft", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = _common_paths(args)
    expected = build_selection_artifact(**paths)
    artifact_path = _resolve(paths["repository_root"], args.artifact)
    _inside(paths["repository_root"], artifact_path, location="selection artifact")

    if args.mode == "select":
        if artifact_path.exists() or artifact_path.is_symlink():
            raise SelectionError(f"refusing to overwrite selection artifact: {artifact_path}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(_canonical_text(expected), encoding="utf-8")
        output = {
            "artifact": _display(paths["repository_root"], artifact_path),
            "sha256": _sha256_file(artifact_path),
            "selected": [result["selected"] for result in expected["seed_results"]],
        }
    else:
        observed = _read_json(artifact_path, location="selection artifact")
        validate_selection_artifact(observed)
        if observed != expected:
            raise SelectionError(
                "selection artifact differs from recomputed checkpoints/evidence/config"
            )
        output = {
            "artifact": _display(paths["repository_root"], artifact_path),
            "sha256": _sha256_file(artifact_path),
            "verification": "passed_exact_recomputation",
        }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
