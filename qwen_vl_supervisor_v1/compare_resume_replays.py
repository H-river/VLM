#!/usr/bin/env python3
"""Prove that a resumed step exactly reproduces an uninterrupted reference step.

The comparison is deliberately stricter than a loss-only replay check.  It binds
the two replay checkpoints, their trainer logs, adapter tensors, safe
optimizer/scheduler state, and the candidate run's recorded restore invariants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


TRAIN_FIELDS = (
    "global_step",
    "epoch",
    "loss",
    "entropy",
    "grad_norm",
    "learning_rate",
    "mean_token_accuracy",
    "num_tokens",
)
EVAL_FIELDS = (
    "global_step",
    "epoch",
    "eval_loss",
    "eval_entropy",
    "eval_mean_token_accuracy",
    "eval_num_tokens",
)
SAFE_OPTIMIZER_JSON = "optimizer_scheduler.safe.json"
SAFE_OPTIMIZER_TENSORS = "optimizer_scheduler.safe.safetensors"
ADAPTER_TENSORS = "adapter_model.safetensors"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(item: str) -> Any:
    raise ValueError(f"non-finite JSON value: {item}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_nonfinite,
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _require_checkpoint(path: Path) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    for name in (
        "trainer_state.json",
        ADAPTER_TENSORS,
        SAFE_OPTIMIZER_JSON,
        SAFE_OPTIMIZER_TENSORS,
    ):
        _require_file(path / name)
    return path


def _file_record(path: Path) -> dict[str, Any]:
    path = _require_file(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }


def _files_byte_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _unique_step_row(
    rows: list[dict[str, Any]], step: int, marker_field: str
) -> dict[str, Any] | None:
    matches = [row for row in rows if row.get("global_step") == step and marker_field in row]
    if len(matches) > 1:
        raise ValueError(
            f"expected at most one step-{step} row containing {marker_field}, found {len(matches)}"
        )
    return matches[0] if matches else None


def _compare_fields(
    reference: dict[str, Any], candidate: dict[str, Any], fields: Iterable[str]
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for field in fields:
        if field not in reference or field not in candidate:
            raise ValueError(f"required deterministic field is missing: {field}")
        comparison[field] = {
            "reference": reference[field],
            "candidate": candidate[field],
            "exact_equal": reference[field] == candidate[field],
        }
    return comparison


def compare_log_rows(
    *, reference_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], replay_step: int
) -> dict[str, Any]:
    """Compare deterministic train fields and, when both exist, eval fields."""

    reference_train = _unique_step_row(reference_rows, replay_step, "loss")
    candidate_train = _unique_step_row(candidate_rows, replay_step, "loss")
    if reference_train is None or candidate_train is None:
        raise ValueError(f"both logs must contain exactly one training row for step {replay_step}")
    train_fields = _compare_fields(reference_train, candidate_train, TRAIN_FIELDS)
    train_exact = all(item["exact_equal"] for item in train_fields.values())

    reference_eval = _unique_step_row(reference_rows, replay_step, "eval_loss")
    candidate_eval = _unique_step_row(candidate_rows, replay_step, "eval_loss")
    eval_fields: dict[str, Any] = {}
    eval_compared = reference_eval is not None and candidate_eval is not None
    if eval_compared:
        assert reference_eval is not None and candidate_eval is not None
        eval_fields = _compare_fields(reference_eval, candidate_eval, EVAL_FIELDS)
    eval_exact: bool | None = (
        all(item["exact_equal"] for item in eval_fields.values()) if eval_compared else None
    )
    return {
        "train": {
            "required_fields": list(TRAIN_FIELDS),
            "fields": train_fields,
            "all_exact_equal": train_exact,
        },
        "eval": {
            "reference_present": reference_eval is not None,
            "candidate_present": candidate_eval is not None,
            "compared": eval_compared,
            "comparison_policy": "compare deterministic eval fields only when both logs contain them",
            "fields": eval_fields,
            "all_exact_equal": eval_exact,
        },
        "all_applicable_fields_exact_equal": train_exact and (eval_exact is not False),
    }


def compare_safetensors_exact(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    """Compare SafeTensors keys, dtype, shape, and every tensor bit exactly."""

    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - exercised by the real environment
        raise RuntimeError("torch and safetensors are required for replay comparison") from exc

    reference_path = _require_file(reference_path)
    candidate_path = _require_file(candidate_path)
    reference_record = _file_record(reference_path)
    candidate_record = _file_record(candidate_path)
    raw_equal = _files_byte_equal(reference_path, candidate_path)
    mismatch_examples: list[dict[str, Any]] = []
    dtype_histogram: dict[str, int] = {}
    total_elements = 0
    metadata_equal = False
    dtype_equal = False
    shape_equal = False
    values_equal = False
    with safe_open(str(reference_path), framework="pt", device="cpu") as reference, safe_open(
        str(candidate_path), framework="pt", device="cpu"
    ) as candidate:
        reference_keys = sorted(reference.keys())
        candidate_keys = sorted(candidate.keys())
        keys_equal = reference_keys == candidate_keys
        metadata_equal = reference.metadata() == candidate.metadata()
        dtype_equal = keys_equal
        shape_equal = keys_equal
        values_equal = keys_equal
        for key in sorted(set(reference_keys) | set(candidate_keys)):
            if key not in reference_keys or key not in candidate_keys:
                if len(mismatch_examples) < 8:
                    mismatch_examples.append(
                        {
                            "key": key,
                            "reason": "missing_key",
                            "reference_present": key in reference_keys,
                            "candidate_present": key in candidate_keys,
                        }
                    )
                continue
            reference_tensor = reference.get_tensor(key)
            candidate_tensor = candidate.get_tensor(key)
            dtype = str(reference_tensor.dtype)
            dtype_histogram[dtype] = dtype_histogram.get(dtype, 0) + 1
            total_elements += reference_tensor.numel()
            this_dtype_equal = reference_tensor.dtype == candidate_tensor.dtype
            this_shape_equal = tuple(reference_tensor.shape) == tuple(candidate_tensor.shape)
            dtype_equal = dtype_equal and this_dtype_equal
            shape_equal = shape_equal and this_shape_equal
            this_value_equal = False
            if this_dtype_equal and this_shape_equal:
                # Byte views make NaN payloads and signed zero part of the exact comparison.
                this_value_equal = torch.equal(
                    reference_tensor.contiguous().view(torch.uint8),
                    candidate_tensor.contiguous().view(torch.uint8),
                )
            values_equal = values_equal and this_value_equal
            if (not this_dtype_equal or not this_shape_equal or not this_value_equal) and len(
                mismatch_examples
            ) < 8:
                mismatch_examples.append(
                    {
                        "key": key,
                        "reason": "tensor_mismatch",
                        "reference_dtype": str(reference_tensor.dtype),
                        "candidate_dtype": str(candidate_tensor.dtype),
                        "reference_shape": list(reference_tensor.shape),
                        "candidate_shape": list(candidate_tensor.shape),
                        "exact_value_bits_equal": this_value_equal,
                    }
                )

    exact = (
        keys_equal
        and metadata_equal
        and dtype_equal
        and shape_equal
        and values_equal
        and raw_equal
    )
    return {
        "reference": reference_record,
        "candidate": candidate_record,
        "tensor_count": len(reference_keys),
        "total_elements": total_elements,
        "dtype_histogram": dict(sorted(dtype_histogram.items())),
        "keys_exact_equal": keys_equal,
        "metadata_exact_equal": metadata_equal,
        "dtypes_exact_equal": dtype_equal,
        "shapes_exact_equal": shape_equal,
        "value_bits_exact_equal": values_equal,
        "raw_file_bytes_exact_equal": raw_equal,
        "mismatch_examples": mismatch_examples,
        "exact_equal": exact,
    }


def _compare_safe_optimizer_scheduler(
    reference_checkpoint: Path, candidate_checkpoint: Path
) -> dict[str, Any]:
    reference_json_path = reference_checkpoint / SAFE_OPTIMIZER_JSON
    candidate_json_path = candidate_checkpoint / SAFE_OPTIMIZER_JSON
    reference_tensor_path = reference_checkpoint / SAFE_OPTIMIZER_TENSORS
    candidate_tensor_path = candidate_checkpoint / SAFE_OPTIMIZER_TENSORS
    reference_json = read_json(reference_json_path)
    candidate_json = read_json(candidate_json_path)
    tensors = compare_safetensors_exact(reference_tensor_path, candidate_tensor_path)
    reference_declared_sha = reference_json.get("tensor_sha256")
    candidate_declared_sha = candidate_json.get("tensor_sha256")
    reference_sha_verified = reference_declared_sha == tensors["reference"]["sha256"]
    candidate_sha_verified = candidate_declared_sha == tensors["candidate"]["sha256"]
    json_bytes_equal = _files_byte_equal(reference_json_path, candidate_json_path)
    decoded_equal = reference_json == candidate_json
    exact = (
        json_bytes_equal
        and decoded_equal
        and reference_sha_verified
        and candidate_sha_verified
        and tensors["exact_equal"]
    )
    return {
        "reference_json": _file_record(reference_json_path),
        "candidate_json": _file_record(candidate_json_path),
        "json_raw_bytes_exact_equal": json_bytes_equal,
        "decoded_json_exact_equal": decoded_equal,
        "declared_tensor_sha256_verified": {
            "reference": reference_sha_verified,
            "candidate": candidate_sha_verified,
        },
        "tensors": tensors,
        "exact_equal": exact,
    }


def _nested(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _resolved_manifest_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).resolve()


def _bf16_audit_is_exact(audit: dict[str, Any], source_checkpoint: Path) -> bool:
    if audit.get("phase") != "trainer_resume":
        return False
    if _resolved_manifest_path(audit.get("checkpoint")) != source_checkpoint:
        return False
    if audit.get("dtype_and_element_count_match") is not True:
        return False
    calls = audit.get("load_calls")
    if not isinstance(calls, list) or not calls:
        return False
    if any(not isinstance(call, dict) or call.get("effective_autocast_adapter_dtype") is not False for call in calls):
        return False
    summaries = [audit.get(name) for name in ("saved_adapter", "trainable_before_load", "trainable_after_load")]
    if any(not isinstance(summary, dict) for summary in summaries):
        return False
    element_counts = {summary.get("total_elements") for summary in summaries}
    if len(element_counts) != 1 or None in element_counts:
        return False
    return all(
        summary.get("dtypes_to_elements") == {"torch.bfloat16": summary.get("total_elements")}
        for summary in summaries
    )


def _candidate_restore_invariants(
    *,
    candidate_manifest: dict[str, Any],
    candidate_checkpoint: Path,
    candidate_log: Path,
    reference_checkpoint: Path,
    reference_manifest: dict[str, Any],
    source_checkpoint: Path,
    source_step: int,
    replay_step: int,
) -> dict[str, Any]:
    resume = candidate_manifest.get("resume_audit")
    if not isinstance(resume, dict):
        raise ValueError("candidate manifest is missing resume_audit")
    safe_optimizer = resume.get("safe_optimizer_scheduler_restore")
    safe_rng = resume.get("safe_rng_restore")
    adapter_invariant = candidate_manifest.get("adapter_dtype_invariant")
    if not isinstance(safe_optimizer, dict) or not isinstance(safe_rng, dict):
        raise ValueError("candidate manifest is missing safe restore audit details")
    if not isinstance(adapter_invariant, dict):
        raise ValueError("candidate manifest is missing adapter dtype invariant")
    restore_audits = adapter_invariant.get("restore_audits")
    if not isinstance(restore_audits, list):
        restore_audits = []

    reference_training = reference_manifest.get("training")
    candidate_training = candidate_manifest.get("training")
    identity_fields = {
        "config_sha256": reference_manifest.get("config_sha256")
        == candidate_manifest.get("config_sha256"),
        "train_export_sha256": _nested(reference_manifest, "data", "train_sha256")
        == _nested(candidate_manifest, "data", "train_sha256"),
        "dev_export_sha256": _nested(reference_manifest, "data", "dev_sha256")
        == _nested(candidate_manifest, "data", "dev_sha256"),
        "export_report_sha256": _nested(reference_manifest, "data", "export_report_sha256")
        == _nested(candidate_manifest, "data", "export_report_sha256"),
        "model_snapshot_tree_sha256": _nested(
            reference_manifest, "model", "source_identity", "fingerprint", "tree_sha256"
        )
        == _nested(candidate_manifest, "model", "source_identity", "fingerprint", "tree_sha256"),
        "training_configuration": reference_training == candidate_training,
        "training_seeds": reference_manifest.get("training_seeds")
        == candidate_manifest.get("training_seeds"),
    }
    checks = {
        "candidate_status_completed": candidate_manifest.get("status") == "completed",
        "resume_requested": resume.get("requested") is True,
        "resume_status_passed": resume.get("status") == "passed",
        "resume_source_checkpoint_exact": _resolved_manifest_path(resume.get("checkpoint"))
        == source_checkpoint,
        "expected_checkpoint_step_exact": resume.get("expected_checkpoint_step") == source_step,
        "observed_start_step_exact": resume.get("observed_start_step") == source_step,
        "first_completed_step_exact": resume.get("first_completed_step") == replay_step,
        "immediate_next_step_recorded": resume.get("next_step_executed_after_full_state_restore")
        is True,
        "full_state_restore_verified": resume.get("full_state_restore_verified") is True,
        "trainer_data_skip_enabled": resume.get("trainer_data_skip_enabled") is True
        and _nested(candidate_manifest, "training", "ignore_data_skip") is False,
        "full_determinism_enabled": _nested(candidate_manifest, "training", "full_determinism")
        is True,
        "safe_optimizer_checkpoint_exact": _resolved_manifest_path(
            safe_optimizer.get("checkpoint")
        )
        == source_checkpoint,
        "safe_optimizer_step_exact": safe_optimizer.get("global_step") == source_step,
        "optimizer_parameter_name_mapping_verified": safe_optimizer.get(
            "optimizer_parameter_name_mapping_verified"
        )
        is True,
        "optimizer_state_exact_after_restore": _nested(
            safe_optimizer, "optimizer_state_exact_after_restore", "exact_equal"
        )
        is True
        and _nested(safe_optimizer, "optimizer_state_exact_after_restore", "mismatch_count") == 0,
        "scheduler_state_exact_after_restore": _nested(
            safe_optimizer, "scheduler_state_exact_after_restore", "exact_equal"
        )
        is True
        and _nested(safe_optimizer, "scheduler_state_exact_after_restore", "mismatch_count") == 0,
        "safe_optimizer_serialization": safe_optimizer.get("serialization")
        == "strict_json_plus_safetensors_no_pickle",
        "safe_rng_checkpoint_exact": _resolved_manifest_path(safe_rng.get("checkpoint"))
        == source_checkpoint,
        "safe_rng_step_exact": safe_rng.get("global_step") == source_step,
        "safe_rng_components_restored": set(safe_rng.get("restored", []))
        >= {"python", "numpy", "torch_cpu", "torch_cuda"},
        "safe_rng_serialization": safe_rng.get("serialization")
        == "strict_json_plus_safetensors_no_pickle",
        "legacy_pickle_never_loaded": _nested(
            candidate_manifest, "safe_checkpointing", "legacy_optimizer_or_rng_pickle_loaded"
        )
        is False,
        "adapter_required_dtype_bfloat16": adapter_invariant.get("required_dtype")
        == "torch.bfloat16",
        "adapter_trainer_resume_audit_exact": any(
            isinstance(audit, dict) and _bf16_audit_is_exact(audit, source_checkpoint)
            for audit in restore_audits
        ),
        "scheduler_horizon_unchanged_by_stop": _nested(
            candidate_manifest,
            "audit_execution_controls",
            "scheduler_horizon_changed_by_stop_control",
        )
        is False,
        "forced_checkpoint_step_exact": _nested(
            candidate_manifest, "audit_execution_controls", "forced_checkpoint_global_step"
        )
        == replay_step,
        "controlled_stop_step_exact": _nested(
            candidate_manifest, "audit_execution_controls", "stop_after_global_step"
        )
        == replay_step,
        "candidate_result_step_exact": _nested(
            candidate_manifest, "training_result", "global_step"
        )
        == replay_step,
        "candidate_result_checkpoint_exact": _resolved_manifest_path(
            _nested(candidate_manifest, "training_result", "latest_checkpoint")
        )
        == candidate_checkpoint,
        "candidate_result_log_exact": _resolved_manifest_path(
            _nested(candidate_manifest, "training_result", "log_jsonl")
        )
        == candidate_log,
        "source_and_reference_replay_share_run": source_checkpoint.parent
        == reference_checkpoint.parent,
    }
    checks.update({f"run_identity_{name}": result for name, result in identity_fields.items()})
    all_verified = all(checks.values())
    return {
        "candidate_manifest_status": candidate_manifest.get("status"),
        "source_checkpoint": str(source_checkpoint),
        "source_step": source_step,
        "replay_step": replay_step,
        "checks": checks,
        "all_verified": all_verified,
    }


def _manifest_and_log(
    checkpoint: Path, explicit_manifest: Path | None, explicit_log: Path | None
) -> tuple[Path, dict[str, Any], Path]:
    run_dir = checkpoint.parent
    manifest_path = _require_file(explicit_manifest or (run_dir / "run_manifest.latest.json"))
    manifest = read_json(manifest_path)
    recorded_log = _nested(manifest, "training_result", "log_jsonl")
    log_path = _require_file(explicit_log or Path(str(recorded_log)))
    if log_path.parent != run_dir:
        raise ValueError(f"trainer log must be inside its checkpoint run directory: {log_path}")
    return manifest_path, manifest, log_path


def compare_replay_checkpoints(
    *,
    reference_checkpoint: Path,
    candidate_checkpoint: Path,
    reference_manifest: Path | None = None,
    candidate_manifest: Path | None = None,
    reference_log: Path | None = None,
    candidate_log: Path | None = None,
    replay_step: int | None = None,
) -> dict[str, Any]:
    """Return a strict resume proof or raise when any required invariant differs."""

    reference_checkpoint = _require_checkpoint(reference_checkpoint)
    candidate_checkpoint = _require_checkpoint(candidate_checkpoint)
    reference_manifest_path, reference_manifest_value, reference_log_path = _manifest_and_log(
        reference_checkpoint, reference_manifest, reference_log
    )
    candidate_manifest_path, candidate_manifest_value, candidate_log_path = _manifest_and_log(
        candidate_checkpoint, candidate_manifest, candidate_log
    )
    reference_state = read_json(reference_checkpoint / "trainer_state.json")
    candidate_state = read_json(candidate_checkpoint / "trainer_state.json")
    reference_step = reference_state.get("global_step")
    candidate_step = candidate_state.get("global_step")
    if not isinstance(reference_step, int) or not isinstance(candidate_step, int):
        raise ValueError("checkpoint trainer_state.global_step must be an integer")
    if reference_step != candidate_step:
        raise ValueError(
            f"reference step {reference_step} does not equal candidate step {candidate_step}"
        )
    if replay_step is not None and replay_step != reference_step:
        raise ValueError(
            f"requested replay step {replay_step} does not equal checkpoint step {reference_step}"
        )
    replay_step = reference_step

    source_checkpoint_value = _nested(candidate_manifest_value, "resume_audit", "checkpoint")
    source_checkpoint = _resolved_manifest_path(source_checkpoint_value)
    if source_checkpoint is None or not source_checkpoint.is_dir():
        raise ValueError("candidate manifest resume_audit.checkpoint is missing or invalid")
    source_state = read_json(_require_file(source_checkpoint / "trainer_state.json"))
    source_step = source_state.get("global_step")
    if not isinstance(source_step, int):
        raise ValueError("resume source trainer_state.global_step must be an integer")
    if replay_step != source_step + 1:
        raise ValueError(
            f"replay step {replay_step} must immediately follow source checkpoint step {source_step}"
        )

    logs = compare_log_rows(
        reference_rows=read_jsonl(reference_log_path),
        candidate_rows=read_jsonl(candidate_log_path),
        replay_step=replay_step,
    )
    adapter = compare_safetensors_exact(
        reference_checkpoint / ADAPTER_TENSORS,
        candidate_checkpoint / ADAPTER_TENSORS,
    )
    optimizer_scheduler = _compare_safe_optimizer_scheduler(
        reference_checkpoint, candidate_checkpoint
    )
    invariants = _candidate_restore_invariants(
        candidate_manifest=candidate_manifest_value,
        candidate_checkpoint=candidate_checkpoint,
        candidate_log=candidate_log_path,
        reference_checkpoint=reference_checkpoint,
        reference_manifest=reference_manifest_value,
        source_checkpoint=source_checkpoint,
        source_step=source_step,
        replay_step=replay_step,
    )
    verified = (
        logs["all_applicable_fields_exact_equal"]
        and adapter["exact_equal"]
        and optimizer_scheduler["exact_equal"]
        and invariants["all_verified"]
    )
    result = {
        "schema": "qwen_vl_resume_replay_comparison_v2",
        "reference": {
            "checkpoint": str(reference_checkpoint),
            "checkpoint_trainer_state": _file_record(reference_checkpoint / "trainer_state.json"),
            "run_manifest": _file_record(reference_manifest_path),
            "trainer_log": _file_record(reference_log_path),
        },
        "candidate": {
            "checkpoint": str(candidate_checkpoint),
            "checkpoint_trainer_state": _file_record(candidate_checkpoint / "trainer_state.json"),
            "run_manifest": _file_record(candidate_manifest_path),
            "trainer_log": _file_record(candidate_log_path),
        },
        "step_relation": {
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_global_step": source_step,
            "reference_replay_checkpoint_global_step": reference_step,
            "candidate_replay_checkpoint_global_step": candidate_step,
            "replay_step": replay_step,
            "is_immediate_successor": replay_step == source_step + 1,
        },
        "log_comparison": logs,
        "adapter_comparison": adapter,
        "optimizer_scheduler_comparison": optimizer_scheduler,
        "candidate_restore_invariants": invariants,
        "deterministic_resume_reproduction_verified": verified,
        "claim_scope": (
            "The candidate safe restore exactly reproduced the uninterrupted reference at the "
            "immediate next training step, including deterministic log fields, BF16 adapter "
            "tensor bits, and safe optimizer/scheduler bytes. This is an engineering replay "
            "check, not a model-quality or frozen-evaluation claim."
        ),
    }
    if not verified:
        failed: list[str] = []
        if not logs["all_applicable_fields_exact_equal"]:
            failed.append("logs")
        if not adapter["exact_equal"]:
            failed.append("adapter")
        if not optimizer_scheduler["exact_equal"]:
            failed.append("optimizer_scheduler")
        if not invariants["all_verified"]:
            failed.append("candidate_restore_invariants")
        raise ValueError("deterministic resume replay failed: " + ", ".join(failed))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--reference-log", type=Path)
    parser.add_argument("--candidate-log", type=Path)
    parser.add_argument("--replay-step", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_replay_checkpoints(
        reference_checkpoint=args.reference_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        reference_manifest=args.reference_manifest,
        candidate_manifest=args.candidate_manifest,
        reference_log=args.reference_log,
        candidate_log=args.candidate_log,
        replay_step=args.replay_step,
    )
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output = args.output.resolve()
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite existing comparison: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
