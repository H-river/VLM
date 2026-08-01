#!/usr/bin/env python3
"""Verify an uninterrupted and resumed run at one post-update checkpoint.

The comparison deliberately separates the pre-update Trainer log emitted for
the replayed optimizer step from the post-update adapter, optimizer/scheduler,
and RNG state saved in that step's checkpoint.  Development evaluation is
optional because an audit checkpoint need not coincide with ``eval_steps``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from qwen_vl_supervisor_v1.train_qlora import (  # noqa: E402
    SAFE_OPTIMIZER_JSON,
    SAFE_OPTIMIZER_TENSORS,
    SAFE_RNG_JSON,
    SAFE_RNG_TENSORS,
    exact_nested_state_comparison,
    load_safe_state_bundle,
    sha256_path,
)


TRAIN_FIELDS = (
    "loss",
    "grad_norm",
    "learning_rate",
    "entropy",
    "num_tokens",
    "mean_token_accuracy",
    "epoch",
)
EVAL_FIELDS = (
    "eval_loss",
    "eval_entropy",
    "eval_num_tokens",
    "eval_mean_token_accuracy",
    "epoch",
)


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"{path}: non-finite JSON constant {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(
                line,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    ValueError(f"{path}:{line_number}: non-finite JSON constant {item}")
                ),
            )
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _unique_step_row(
    rows: Sequence[Mapping[str, Any]], step: int, required_field: str
) -> Mapping[str, Any] | None:
    matches = [row for row in rows if row.get("global_step") == step and required_field in row]
    if len(matches) > 1:
        raise ValueError(
            f"expected at most one step-{step} row containing {required_field}, "
            f"found {len(matches)}"
        )
    return matches[0] if matches else None


def _required_step_row(
    rows: Sequence[Mapping[str, Any]], step: int, required_field: str
) -> Mapping[str, Any]:
    row = _unique_step_row(rows, step, required_field)
    if row is None:
        raise ValueError(f"missing step-{step} row containing {required_field}")
    return row


def _field_comparison(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], fields: Sequence[str]
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for field in fields:
        if field not in reference or field not in candidate:
            raise ValueError(f"required comparison field is missing: {field}")
        comparison[field] = {
            "reference": reference[field],
            "candidate": candidate[field],
            "exact_equal": reference[field] == candidate[field],
        }
    return comparison


def _resolved_manifest(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run_manifest.latest.json"
    manifest = _strict_json(manifest_path)
    log_path = Path(manifest["training_result"]["log_jsonl"]).resolve()
    if not log_path.is_relative_to(run_dir):
        raise ValueError(f"manifest log path escapes its run directory: {log_path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"manifest training log is missing: {log_path}")
    return manifest_path, manifest


def _checkpoint_step(checkpoint: Path) -> int:
    state = _strict_json(checkpoint / "trainer_state.json")
    return int(state["global_step"])


def _adapter_comparison(reference: Path, candidate: Path, torch: Any, load_file: Any) -> dict[str, Any]:
    reference_path = reference / "adapter_model.safetensors"
    candidate_path = candidate / "adapter_model.safetensors"
    reference_tensors = load_file(str(reference_path), device="cpu")
    candidate_tensors = load_file(str(candidate_path), device="cpu")
    exact = exact_nested_state_comparison(
        reference_tensors, candidate_tensors, torch_module=torch, max_examples=12
    )
    reference_dtypes = Counter(str(tensor.dtype) for tensor in reference_tensors.values())
    candidate_dtypes = Counter(str(tensor.dtype) for tensor in candidate_tensors.values())
    reference_elements = sum(tensor.numel() for tensor in reference_tensors.values())
    candidate_elements = sum(tensor.numel() for tensor in candidate_tensors.values())
    reference_sha = sha256_path(reference_path)
    candidate_sha = sha256_path(candidate_path)
    return {
        "reference": {
            "path": str(reference_path),
            "sha256": reference_sha,
            "tensor_count": len(reference_tensors),
            "total_elements": reference_elements,
            "dtypes_to_tensor_count": dict(sorted(reference_dtypes.items())),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": candidate_sha,
            "tensor_count": len(candidate_tensors),
            "total_elements": candidate_elements,
            "dtypes_to_tensor_count": dict(sorted(candidate_dtypes.items())),
        },
        "file_bytes_exact_equal": reference_sha == candidate_sha,
        "tensor_keys_dtypes_shapes_values_exact_equal": exact,
        "all_tensors_bfloat16": (
            set(reference_dtypes) == {"torch.bfloat16"}
            and set(candidate_dtypes) == {"torch.bfloat16"}
        ),
    }


def _safe_bundle_comparison(
    reference: Path,
    candidate: Path,
    *,
    json_name: str,
    tensor_name: str,
    torch: Any,
    load_file: Any,
) -> dict[str, Any]:
    reference_value = load_safe_state_bundle(
        directory=reference,
        json_name=json_name,
        tensor_name=tensor_name,
        load_file=load_file,
    )
    candidate_value = load_safe_state_bundle(
        directory=candidate,
        json_name=json_name,
        tensor_name=tensor_name,
        load_file=load_file,
    )
    exact = exact_nested_state_comparison(
        reference_value, candidate_value, torch_module=torch, max_examples=12
    )
    reference_json_sha = sha256_path(reference / json_name)
    candidate_json_sha = sha256_path(candidate / json_name)
    reference_tensor_sha = sha256_path(reference / tensor_name)
    candidate_tensor_sha = sha256_path(candidate / tensor_name)
    return {
        "reference": {
            "json_path": str(reference / json_name),
            "json_sha256": reference_json_sha,
            "tensor_path": str(reference / tensor_name),
            "tensor_sha256": reference_tensor_sha,
        },
        "candidate": {
            "json_path": str(candidate / json_name),
            "json_sha256": candidate_json_sha,
            "tensor_path": str(candidate / tensor_name),
            "tensor_sha256": candidate_tensor_sha,
        },
        "json_bytes_exact_equal": reference_json_sha == candidate_json_sha,
        "tensor_file_bytes_exact_equal": reference_tensor_sha == candidate_tensor_sha,
        "decoded_state_exact_equal": exact,
    }


def _model_identity(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["model"]["source_identity"]


def _all_true(values: Mapping[str, bool]) -> bool:
    return all(value is True for value in values.values())


def compare_midresume(
    *,
    reference_dir: Path,
    candidate_dir: Path,
    replay_step: int,
    reference_replay_checkpoint: Path | None = None,
    candidate_replay_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Return a byte- and value-exact mid-resume comparison artifact."""

    import torch
    from safetensors.torch import load_file

    reference_dir = reference_dir.resolve()
    candidate_dir = candidate_dir.resolve()
    reference_checkpoint = (
        reference_replay_checkpoint.resolve()
        if reference_replay_checkpoint is not None
        else reference_dir / f"checkpoint-{replay_step}"
    )
    candidate_checkpoint = (
        candidate_replay_checkpoint.resolve()
        if candidate_replay_checkpoint is not None
        else candidate_dir / f"checkpoint-{replay_step}"
    )
    for checkpoint in (reference_checkpoint, candidate_checkpoint):
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"replay checkpoint is missing: {checkpoint}")
        if _checkpoint_step(checkpoint) != replay_step:
            raise ValueError(f"replay checkpoint global step is not {replay_step}: {checkpoint}")

    reference_manifest_path, reference_manifest = _resolved_manifest(reference_dir)
    candidate_manifest_path, candidate_manifest = _resolved_manifest(candidate_dir)
    reference_log_path = Path(reference_manifest["training_result"]["log_jsonl"]).resolve()
    candidate_log_path = Path(candidate_manifest["training_result"]["log_jsonl"]).resolve()
    reference_rows = _read_jsonl(reference_log_path)
    candidate_rows = _read_jsonl(candidate_log_path)
    reference_train = _required_step_row(reference_rows, replay_step, "loss")
    candidate_train = _required_step_row(candidate_rows, replay_step, "loss")
    training_fields = _field_comparison(reference_train, candidate_train, TRAIN_FIELDS)

    reference_eval = _unique_step_row(reference_rows, replay_step, "eval_loss")
    candidate_eval = _unique_step_row(candidate_rows, replay_step, "eval_loss")
    if (reference_eval is None) != (candidate_eval is None):
        raise ValueError("only one run contains development evaluation at the replay step")
    if reference_eval is None:
        evaluation = {
            "performed_at_replay_step": False,
            "required_for_resume_verification": False,
            "comparison": None,
            "reason": (
                "The forced audit checkpoint does not coincide with eval_steps. Resume evidence "
                "uses exact pre-update training metrics and exact post-update checkpoint state."
            ),
        }
    else:
        assert candidate_eval is not None
        eval_fields = _field_comparison(reference_eval, candidate_eval, EVAL_FIELDS)
        evaluation = {
            "performed_at_replay_step": True,
            "required_for_resume_verification": False,
            "comparison": eval_fields,
            "all_fields_exact_equal": all(
                item["exact_equal"] for item in eval_fields.values()
            ),
        }

    adapter = _adapter_comparison(reference_checkpoint, candidate_checkpoint, torch, load_file)
    optimizer = _safe_bundle_comparison(
        reference_checkpoint,
        candidate_checkpoint,
        json_name=SAFE_OPTIMIZER_JSON,
        tensor_name=SAFE_OPTIMIZER_TENSORS,
        torch=torch,
        load_file=load_file,
    )
    rng = _safe_bundle_comparison(
        reference_checkpoint,
        candidate_checkpoint,
        json_name=SAFE_RNG_JSON,
        tensor_name=SAFE_RNG_TENSORS,
        torch=torch,
        load_file=load_file,
    )

    prior_step = replay_step - 1
    reference_prior_train = _required_step_row(reference_rows, prior_step, "loss")
    resume = candidate_manifest["resume_audit"]
    optimizer_restore = resume["safe_optimizer_scheduler_restore"]
    reference_identity = _model_identity(reference_manifest)
    candidate_identity = _model_identity(candidate_manifest)
    reference_controls = reference_manifest["audit_execution_controls"]
    candidate_controls = candidate_manifest["audit_execution_controls"]
    expected_resume_checkpoint = (reference_dir / f"checkpoint-{prior_step}").resolve()
    candidate_final_adapter = candidate_manifest["adapter_dtype_invariant"]["final_adapter"]
    adapter_exact = adapter["tensor_keys_dtypes_shapes_values_exact_equal"]
    optimizer_exact = optimizer["decoded_state_exact_equal"]
    rng_exact = rng["decoded_state_exact_equal"]
    restore_audits = candidate_manifest["adapter_dtype_invariant"]["restore_audits"]

    checks = {
        "both_manifests_completed": (
            reference_manifest.get("status") == "completed"
            and candidate_manifest.get("status") == "completed"
        ),
        "reference_is_fresh_candidate_is_resume": (
            reference_manifest["resume_audit"]["requested"] is False
            and resume["requested"] is True
        ),
        "candidate_resumed_immediately_before_replay_step": (
            Path(resume["checkpoint"]).resolve() == expected_resume_checkpoint
            and resume["expected_checkpoint_step"] == prior_step
            and resume["observed_start_step"] == prior_step
            and resume["first_completed_step"] == replay_step
            and resume["next_step_executed_after_full_state_restore"] is True
            and resume["full_state_restore_verified"] is True
            and resume["status"] == "passed"
        ),
        "safe_restore_parameter_mapping_verified": (
            optimizer_restore["optimizer_parameter_name_mapping_verified"] is True
        ),
        "safe_restore_optimizer_exact_before_step": (
            optimizer_restore["optimizer_state_exact_after_restore"]["exact_equal"] is True
            and optimizer_restore["optimizer_state_exact_after_restore"]["mismatch_count"] == 0
        ),
        "safe_restore_scheduler_exact_before_step": (
            optimizer_restore["scheduler_state_exact_after_restore"]["exact_equal"] is True
            and optimizer_restore["scheduler_state_exact_after_restore"]["mismatch_count"] == 0
        ),
        "trl_token_counter_restored": (
            optimizer_restore["trl_total_train_tokens_restored"]
            == int(reference_prior_train["num_tokens"])
        ),
        "training_step_fields_exact": all(
            item["exact_equal"] for item in training_fields.values()
        ),
        "optional_eval_exact_if_present": (
            evaluation["comparison"] is None
            or evaluation.get("all_fields_exact_equal") is True
        ),
        "adapter_file_bytes_exact": adapter["file_bytes_exact_equal"] is True,
        "adapter_keys_dtypes_shapes_values_exact": (
            adapter_exact["exact_equal"] is True and adapter_exact["mismatch_count"] == 0
        ),
        "adapter_is_all_bfloat16": adapter["all_tensors_bfloat16"] is True,
        "optimizer_json_and_tensor_bytes_exact": (
            optimizer["json_bytes_exact_equal"] is True
            and optimizer["tensor_file_bytes_exact_equal"] is True
        ),
        "optimizer_decoded_state_exact": (
            optimizer_exact["exact_equal"] is True and optimizer_exact["mismatch_count"] == 0
        ),
        "rng_json_and_tensor_bytes_exact": (
            rng["json_bytes_exact_equal"] is True
            and rng["tensor_file_bytes_exact_equal"] is True
        ),
        "rng_decoded_state_exact": (
            rng_exact["exact_equal"] is True and rng_exact["mismatch_count"] == 0
        ),
        "same_config_data_seed_and_scheduler_horizon": (
            reference_manifest["config_sha256"] == candidate_manifest["config_sha256"]
            and reference_manifest["data"] == candidate_manifest["data"]
            and reference_manifest["training_seeds"] == candidate_manifest["training_seeds"]
            and reference_manifest["training"]["max_steps"]
            == candidate_manifest["training"]["max_steps"]
            == reference_controls["configured_scheduler_max_steps"]
            == candidate_controls["configured_scheduler_max_steps"]
            and reference_controls["scheduler_horizon_changed_by_stop_control"] is False
            and candidate_controls["scheduler_horizon_changed_by_stop_control"] is False
        ),
        "full_determinism_enabled_both_runs": (
            reference_manifest["training"]["full_determinism"] is True
            and candidate_manifest["training"]["full_determinism"] is True
        ),
        "forced_replay_checkpoint_and_controlled_candidate_stop": (
            reference_controls["forced_checkpoint_global_step"] == replay_step
            and candidate_controls["forced_checkpoint_global_step"] == replay_step
            and reference_controls["stop_after_global_step"] is None
            and candidate_controls["stop_after_global_step"] == replay_step
        ),
        "local_model_snapshot_identity_exact_and_verified": (
            reference_identity == candidate_identity
            and reference_identity["kind"] == "local_snapshot"
            and reference_identity["expected_tree_sha256_verified"] is True
            and reference_identity["expected_tree_sha256"]
            == reference_identity["fingerprint"]["tree_sha256"]
        ),
        "adapter_restore_dtype_audits_passed": (
            candidate_manifest["adapter_dtype_invariant"]["final_adapter_matches"] is True
            and all(item["dtype_and_element_count_match"] is True for item in restore_audits)
        ),
        "candidate_final_adapter_is_replay_checkpoint": (
            Path(candidate_final_adapter["path"]).resolve()
            == (candidate_dir / "final_adapter/adapter_model.safetensors").resolve()
            and candidate_final_adapter["sha256"] == adapter["candidate"]["sha256"]
            and candidate_final_adapter["tensor_count"] == adapter["candidate"]["tensor_count"]
            and candidate_final_adapter["total_elements"] == adapter["candidate"]["total_elements"]
        ),
        "manifest_restore_counts_match_checkpoint_comparison": (
            optimizer_restore["optimizer_parameter_count"] == adapter["candidate"]["tensor_count"]
            and optimizer_restore["optimizer_state_exact_after_restore"]["tensor_count"]
            == optimizer_exact["tensor_count"]
            and optimizer_restore["optimizer_state_exact_after_restore"]["tensor_elements"]
            == optimizer_exact["tensor_elements"]
        ),
    }
    verified = _all_true(checks)
    artifact = {
        "schema": "qwen_vl_midresume_exact_comparison_v1",
        "artifact_role": "authoritative_stepwise_resume_reproduction_evidence",
        "authoritative": True,
        "reference": {
            "run_dir": str(reference_dir),
            "manifest": str(reference_manifest_path),
            "manifest_sha256": sha256_path(reference_manifest_path),
            "run_id": reference_manifest["run_id"],
            "log": str(reference_log_path),
            "log_sha256": sha256_path(reference_log_path),
            "replay_checkpoint": str(reference_checkpoint),
        },
        "candidate": {
            "run_dir": str(candidate_dir),
            "manifest": str(candidate_manifest_path),
            "manifest_sha256": sha256_path(candidate_manifest_path),
            "run_id": candidate_manifest["run_id"],
            "log": str(candidate_log_path),
            "log_sha256": sha256_path(candidate_log_path),
            "resume_checkpoint": str(expected_resume_checkpoint),
            "replay_checkpoint": str(candidate_checkpoint),
        },
        "resume_transition": {
            "checkpoint_global_step": prior_step,
            "replayed_global_step": replay_step,
            "configured_scheduler_max_steps": candidate_manifest["training"]["max_steps"],
            "candidate_stopped_after_replay_step": True,
        },
        "pre_update_training_log_comparison": {
            "fields": training_fields,
            "all_fields_exact_equal": all(
                item["exact_equal"] for item in training_fields.values()
            ),
        },
        "development_evaluation_comparison": evaluation,
        "post_update_checkpoint_comparison": {
            "adapter": adapter,
            "optimizer_scheduler": optimizer,
            "rng": rng,
        },
        "candidate_pre_step_restore_audit": {
            "optimizer_parameter_count": optimizer_restore["optimizer_parameter_count"],
            "optimizer_state_exact_after_restore": optimizer_restore[
                "optimizer_state_exact_after_restore"
            ],
            "scheduler_state_exact_after_restore": optimizer_restore[
                "scheduler_state_exact_after_restore"
            ],
            "optimizer_parameter_name_mapping_verified": optimizer_restore[
                "optimizer_parameter_name_mapping_verified"
            ],
            "trl_total_train_tokens_restored": optimizer_restore[
                "trl_total_train_tokens_restored"
            ],
            "safe_rng_restore": resume["safe_rng_restore"],
        },
        "model_source_identity": reference_identity,
        "verification_checks": checks,
        "all_verification_checks_passed": verified,
        "numerical_reproduction_verified": verified,
        "claim_scope": (
            "An independent strict-safe restore of the uninterrupted step-10 checkpoint "
            "reproduced step 11 exactly in the declared training metrics and every saved "
            "adapter, optimizer/scheduler, and RNG tensor. This is engineering resume evidence, "
            "not a model-quality or frozen-evaluation claim."
        ),
        "supersession_note": (
            "Only the deterministic_pinned reference/candidate pair named above is authoritative. "
            "Earlier midresume directories are preserved as diagnostic evidence and are superseded."
        ),
    }
    return artifact


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--replay-step", type=int, required=True)
    parser.add_argument("--reference-replay-checkpoint", type=Path)
    parser.add_argument("--candidate-replay-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifact = compare_midresume(
        reference_dir=args.reference_dir,
        candidate_dir=args.candidate_dir,
        replay_step=args.replay_step,
        reference_replay_checkpoint=args.reference_replay_checkpoint,
        candidate_replay_checkpoint=args.candidate_replay_checkpoint,
    )
    _atomic_json(args.output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False))
    if artifact["numerical_reproduction_verified"] is not True:
        raise SystemExit("mid-resume numerical reproduction verification failed")


if __name__ == "__main__":
    main()
