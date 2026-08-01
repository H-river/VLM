import copy
from pathlib import Path

import pytest

from qwen_vl_supervisor_v1.compare_resume_replays import (
    compare_log_rows,
    compare_replay_checkpoints,
    compare_safetensors_exact,
)


MODULE_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ARTIFACTS = MODULE_ROOT / "artifacts/training"
REFERENCE_RUN = (
    TRAINING_ARTIFACTS / "smoke_qwen25vl_3b_midresume_reference_deterministic_pinned"
)
CANDIDATE_RUN = (
    TRAINING_ARTIFACTS / "smoke_qwen25vl_3b_midresume_candidate_deterministic_pinned"
)


def _train_row(step: int = 11) -> dict[str, float | int]:
    return {
        "global_step": step,
        "epoch": 0.5,
        "loss": 0.25,
        "entropy": 0.1,
        "grad_norm": 1.125,
        "learning_rate": 0.0001,
        "mean_token_accuracy": 0.9,
        "num_tokens": 1234.0,
    }


def test_completed_pinned_resume_is_exact_at_step_11() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    result = compare_replay_checkpoints(
        reference_checkpoint=REFERENCE_RUN / "checkpoint-11",
        candidate_checkpoint=CANDIDATE_RUN / "checkpoint-11",
        replay_step=11,
    )
    assert result["deterministic_resume_reproduction_verified"] is True
    assert result["step_relation"]["source_checkpoint_global_step"] == 10
    assert result["step_relation"]["is_immediate_successor"] is True
    assert result["log_comparison"]["train"]["fields"]["grad_norm"]["exact_equal"]
    assert result["log_comparison"]["train"]["fields"]["num_tokens"]["exact_equal"]
    assert result["log_comparison"]["eval"]["compared"] is False
    assert result["adapter_comparison"]["tensor_count"] == 824
    assert result["adapter_comparison"]["dtype_histogram"] == {"torch.bfloat16": 824}
    assert result["adapter_comparison"]["value_bits_exact_equal"] is True
    assert result["optimizer_scheduler_comparison"]["exact_equal"] is True
    assert result["candidate_restore_invariants"]["all_verified"] is True


def test_grad_norm_and_num_tokens_are_required_exact_fields() -> None:
    reference = _train_row()
    candidate = copy.deepcopy(reference)
    candidate["grad_norm"] = 1.0
    result = compare_log_rows(
        reference_rows=[reference], candidate_rows=[candidate], replay_step=11
    )
    assert result["all_applicable_fields_exact_equal"] is False
    assert result["train"]["fields"]["grad_norm"]["exact_equal"] is False

    candidate = copy.deepcopy(reference)
    candidate["num_tokens"] = 1235.0
    result = compare_log_rows(
        reference_rows=[reference], candidate_rows=[candidate], replay_step=11
    )
    assert result["all_applicable_fields_exact_equal"] is False
    assert result["train"]["fields"]["num_tokens"]["exact_equal"] is False


def test_eval_fields_are_compared_only_when_both_logs_have_eval() -> None:
    train = _train_row()
    reference_eval = {
        "global_step": 11,
        "epoch": 0.5,
        "eval_loss": 0.2,
        "eval_entropy": 0.1,
        "eval_mean_token_accuracy": 0.95,
        "eval_num_tokens": 1234.0,
    }
    one_sided = compare_log_rows(
        reference_rows=[train, reference_eval], candidate_rows=[train], replay_step=11
    )
    assert one_sided["eval"]["reference_present"] is True
    assert one_sided["eval"]["candidate_present"] is False
    assert one_sided["eval"]["compared"] is False
    assert one_sided["all_applicable_fields_exact_equal"] is True

    candidate_eval = copy.deepcopy(reference_eval)
    candidate_eval["eval_loss"] = 0.3
    both = compare_log_rows(
        reference_rows=[train, reference_eval],
        candidate_rows=[train, candidate_eval],
        replay_step=11,
    )
    assert both["eval"]["compared"] is True
    assert both["eval"]["all_exact_equal"] is False
    assert both["all_applicable_fields_exact_equal"] is False


def test_safetensor_comparison_rejects_dtype_and_value_drift(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safe_torch = pytest.importorskip("safetensors.torch")
    reference = tmp_path / "reference.safetensors"
    equal = tmp_path / "equal.safetensors"
    value_drift = tmp_path / "value_drift.safetensors"
    dtype_drift = tmp_path / "dtype_drift.safetensors"
    payload = {"weight": torch.tensor([1.0, -0.0], dtype=torch.bfloat16)}
    safe_torch.save_file(payload, reference)
    safe_torch.save_file(payload, equal)
    safe_torch.save_file(
        {"weight": torch.tensor([1.0, 0.5], dtype=torch.bfloat16)}, value_drift
    )
    safe_torch.save_file(
        {"weight": torch.tensor([1.0, -0.0], dtype=torch.float32)}, dtype_drift
    )

    assert compare_safetensors_exact(reference, equal)["exact_equal"] is True
    value_result = compare_safetensors_exact(reference, value_drift)
    assert value_result["value_bits_exact_equal"] is False
    assert value_result["exact_equal"] is False
    dtype_result = compare_safetensors_exact(reference, dtype_drift)
    assert dtype_result["dtypes_exact_equal"] is False
    assert dtype_result["exact_equal"] is False
