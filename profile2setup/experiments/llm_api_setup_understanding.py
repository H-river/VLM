"""End-to-end experiment runner for LLM API optical setup understanding."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from profile2setup.data_prep.build_llm_api_sft_dataset import build_llm_api_sft_dataset
from profile2setup.evaluation.llm_api_eval import evaluate_llm_api_predictions
from profile2setup.llm_api.inference import run_llm_api_inference
from profile2setup.llm_api.sft_jobs import (
    create_sft_job,
    load_job_metadata,
    save_job_metadata,
    update_job_metadata,
    upload_training_file,
    upload_validation_file,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_json(obj: Any, path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _load_json(path) -> dict | None:
    json_path = Path(path)
    if not json_path.is_file():
        return None
    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else None


def _path_exists(path) -> bool:
    return Path(path).is_file()


def _mean_dict_values(values: dict | None) -> float | None:
    if not isinstance(values, dict):
        return None
    nums = [float(value) for value in values.values() if isinstance(value, (int, float))]
    return None if not nums else float(np.mean(np.asarray(nums, dtype=np.float64)))


def _api_summary(report: dict | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    numerical = report.get("numerical_metrics") or {}
    return {
        "valid_json_rate": (report.get("format_metrics") or {}).get("valid_json_rate"),
        "schema_valid_rate": (report.get("format_metrics") or {}).get("schema_valid_rate"),
        "canonical_variable_rate": (report.get("format_metrics") or {}).get("canonical_variable_rate"),
        "changed_variable_f1": (report.get("understanding_metrics") or {}).get("changed_variable_f1"),
        "observed_profile_change_accuracy": (report.get("understanding_metrics") or {}).get(
            "observed_profile_change_accuracy"
        ),
        "routed_setup_mae": numerical.get("routed_setup_mae"),
        "predicted_setup_mae": numerical.get("predicted_setup_mae"),
        "predicted_delta_mae": numerical.get("predicted_delta_mae"),
        "normalized_profile_mse": (report.get("simulator_metrics") or {}).get("normalized_profile_mse"),
        "counts": report.get("counts"),
    }


def _profile2setup_summary(report: dict | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    routed = (((report.get("physical_metrics") or {}).get("routed_setup") or {}).get("mae") or {})
    return {
        "num_examples": report.get("num_examples"),
        "task_type_counts": report.get("task_type_counts"),
        "routed_setup_mae": _mean_dict_values(routed),
        "per_variable_routed_setup_mae": routed,
        "loss": report.get("loss"),
    }


def _record_step(steps: list[dict[str, Any]], name: str, status: str, **extra) -> None:
    steps.append({"name": name, "status": status, **extra})


def _run_profile2setup_baseline(
    *,
    checkpoint_path,
    test_jsonl,
    out_path,
    variables_config,
    max_examples: int | None,
    device: str,
    strict: bool,
) -> tuple[dict | None, str | None]:
    if checkpoint_path is None:
        return None, "checkpoint not provided"
    if not Path(checkpoint_path).is_file():
        return None, f"checkpoint not found: {checkpoint_path}"
    try:
        from profile2setup.evaluation.evaluate_model import evaluate_checkpoint
    except Exception as exc:
        return None, f"baseline evaluator unavailable: {type(exc).__name__}: {exc}"

    try:
        return evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            data_path=test_jsonl,
            out_path=out_path,
            variables_config_path=variables_config,
            device=device,
            max_examples=max_examples,
            strict=strict,
        ), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def run_llm_api_setup_experiment(
    *,
    provider: str,
    base_model: str,
    sft_model: str | None,
    train_jsonl,
    val_jsonl,
    test_jsonl,
    out_dir,
    profile2setup_checkpoint=None,
    max_test_examples: int | None = None,
    run_simulator: bool = False,
    simulation_policy: str = "target_base",
    variables_config="profile2setup/configs/variables.yaml",
    dry_run: bool = False,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    image_detail: str = "low",
    reuse_rendered_images: bool = False,
    include_composite: bool = False,
    image_mode: str = "base64",
    sft_train_limit: int | None = None,
    sft_val_limit: int | None = None,
    build_sft: bool = True,
    reuse_sft_jsonl: bool = False,
    skip_base_inference: bool = False,
    base_predictions=None,
    skip_base_eval: bool = False,
    create_sft: bool = False,
    sft_job_metadata=None,
    check_sft_job: bool = False,
    skip_sft_inference: bool = False,
    sft_predictions=None,
    skip_sft_eval: bool = False,
    run_profile2setup_baseline: bool = False,
    profile2setup_baseline_json=None,
    device: str = "auto",
    strict_baseline: bool = False,
) -> dict[str, Any]:
    """Run or reuse each stage of the LLM API setup-understanding experiment."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []

    paths = {
        "sft_train_jsonl": out_root / "sft_data" / "train.jsonl",
        "sft_val_jsonl": out_root / "sft_data" / "val.jsonl",
        "sft_train_images": out_root / "sft_data" / "images" / "train",
        "sft_val_images": out_root / "sft_data" / "images" / "val",
        "base_predictions": Path(base_predictions) if base_predictions else out_root / "base_api_predictions.jsonl",
        "base_eval": out_root / "base_api_eval.json",
        "sft_predictions": Path(sft_predictions) if sft_predictions else out_root / "sft_api_predictions.jsonl",
        "sft_eval": out_root / "sft_api_eval.json",
        "sft_job_metadata": Path(sft_job_metadata) if sft_job_metadata else out_root / "sft_job" / "job.json",
        "profile2setup_baseline": Path(profile2setup_baseline_json)
        if profile2setup_baseline_json
        else out_root / "profile2setup_baseline_eval.json",
        "comparison_report": out_root / "comparison_report.json",
        "base_images": out_root / "images" / "base",
        "sft_images": out_root / "images" / "sft",
    }

    sft_train_ready = _path_exists(paths["sft_train_jsonl"])
    sft_val_ready = _path_exists(paths["sft_val_jsonl"])
    if build_sft:
        if reuse_sft_jsonl and sft_train_ready and sft_val_ready:
            _record_step(
                steps,
                "build_sft_jsonl",
                "reused",
                train_jsonl=str(paths["sft_train_jsonl"]),
                val_jsonl=str(paths["sft_val_jsonl"]),
            )
        else:
            train_summary = build_llm_api_sft_dataset(
                input_path=train_jsonl,
                out_path=paths["sft_train_jsonl"],
                image_out_dir=paths["sft_train_images"],
                limit=sft_train_limit,
                split="train",
                include_composite=include_composite,
                image_mode=image_mode,
                image_detail=image_detail,
                strict=False,
            )
            val_summary = build_llm_api_sft_dataset(
                input_path=val_jsonl,
                out_path=paths["sft_val_jsonl"],
                image_out_dir=paths["sft_val_images"],
                limit=sft_val_limit,
                split="val",
                include_composite=include_composite,
                image_mode=image_mode,
                image_detail=image_detail,
                strict=False,
            )
            _record_step(
                steps,
                "build_sft_jsonl",
                "completed",
                train_summary=train_summary,
                val_summary=val_summary,
            )
    else:
        _record_step(steps, "build_sft_jsonl", "skipped")

    base_eval_report = None
    if skip_base_inference:
        _record_step(steps, "base_inference", "skipped", predictions=str(paths["base_predictions"]))
    elif _path_exists(paths["base_predictions"]) and base_predictions is not None:
        _record_step(steps, "base_inference", "reused", predictions=str(paths["base_predictions"]))
    else:
        base_infer = run_llm_api_inference(
            provider=provider,
            model=base_model,
            data_path=test_jsonl,
            out_path=paths["base_predictions"],
            image_out_dir=paths["base_images"],
            limit=max_test_examples,
            dry_run=dry_run,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            image_detail=image_detail,
            reuse_rendered_images=reuse_rendered_images,
        )
        _record_step(steps, "base_inference", "completed", summary=base_infer)

    if skip_base_eval:
        _record_step(steps, "base_eval", "skipped")
    elif _path_exists(paths["base_predictions"]):
        base_eval_report = evaluate_llm_api_predictions(
            predictions_path=paths["base_predictions"],
            data_path=test_jsonl,
            out_path=paths["base_eval"],
            variables_config_path=variables_config,
            run_simulator=run_simulator,
            simulation_policy=simulation_policy,
            max_examples=max_test_examples,
        )
        _record_step(steps, "base_eval", "completed", out=str(paths["base_eval"]))
    else:
        _record_step(steps, "base_eval", "skipped", reason="base predictions not found")

    sft_job = None
    if create_sft:
        train_file = upload_training_file(paths["sft_train_jsonl"], provider=provider, dry_run=dry_run)
        val_file = upload_validation_file(paths["sft_val_jsonl"], provider=provider, dry_run=dry_run)
        job = create_sft_job(
            provider=provider,
            base_model=base_model,
            training_file_id=train_file["id"],
            validation_file_id=val_file["id"],
            dry_run=dry_run,
        )
        metadata = {
            "provider": provider,
            "base_model": base_model,
            "training_file": train_file,
            "validation_file": val_file,
            "job": job,
            "job_id": job.get("id"),
            "status": job.get("status"),
            "fine_tuned_model": job.get("fine_tuned_model"),
            "dry_run": dry_run,
        }
        sft_job = save_job_metadata(paths["sft_job_metadata"], metadata)
        _record_step(steps, "create_sft_job", "completed", metadata_path=str(paths["sft_job_metadata"]))
    else:
        _record_step(steps, "create_sft_job", "skipped")

    if check_sft_job:
        if _path_exists(paths["sft_job_metadata"]):
            sft_job = update_job_metadata(paths["sft_job_metadata"], provider=provider, dry_run=dry_run)
            _record_step(steps, "check_sft_job", "completed", metadata_path=str(paths["sft_job_metadata"]))
        else:
            _record_step(steps, "check_sft_job", "skipped", reason="job metadata not found")
    elif sft_job is None and _path_exists(paths["sft_job_metadata"]):
        sft_job = load_job_metadata(paths["sft_job_metadata"])

    sft_model_id = (sft_model or "").strip() or None
    if sft_model_id is None and isinstance(sft_job, dict):
        value = sft_job.get("fine_tuned_model")
        sft_model_id = value if isinstance(value, str) and value else None

    sft_eval_report = None
    if not sft_model_id:
        _record_step(steps, "sft_inference", "skipped", reason="fine-tuned model ID not supplied or unavailable")
    elif skip_sft_inference:
        _record_step(steps, "sft_inference", "skipped", predictions=str(paths["sft_predictions"]))
    elif _path_exists(paths["sft_predictions"]) and sft_predictions is not None:
        _record_step(steps, "sft_inference", "reused", predictions=str(paths["sft_predictions"]))
    else:
        sft_infer = run_llm_api_inference(
            provider=provider,
            model=sft_model_id,
            data_path=test_jsonl,
            out_path=paths["sft_predictions"],
            image_out_dir=paths["sft_images"],
            limit=max_test_examples,
            dry_run=dry_run,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            image_detail=image_detail,
            reuse_rendered_images=reuse_rendered_images,
        )
        _record_step(steps, "sft_inference", "completed", summary=sft_infer)

    if skip_sft_eval:
        _record_step(steps, "sft_eval", "skipped")
    elif sft_model_id and _path_exists(paths["sft_predictions"]):
        sft_eval_report = evaluate_llm_api_predictions(
            predictions_path=paths["sft_predictions"],
            data_path=test_jsonl,
            out_path=paths["sft_eval"],
            variables_config_path=variables_config,
            run_simulator=run_simulator,
            simulation_policy=simulation_policy,
            max_examples=max_test_examples,
        )
        _record_step(steps, "sft_eval", "completed", out=str(paths["sft_eval"]))
    else:
        _record_step(steps, "sft_eval", "skipped", reason="sft predictions unavailable")

    baseline_report = None
    if not run_profile2setup_baseline:
        _record_step(steps, "profile2setup_baseline", "skipped")
    elif _path_exists(paths["profile2setup_baseline"]) and profile2setup_baseline_json:
        baseline_report = _load_json(paths["profile2setup_baseline"])
        _record_step(steps, "profile2setup_baseline", "reused", out=str(paths["profile2setup_baseline"]))
    else:
        baseline_report, baseline_error = _run_profile2setup_baseline(
            checkpoint_path=profile2setup_checkpoint,
            test_jsonl=test_jsonl,
            out_path=paths["profile2setup_baseline"],
            variables_config=variables_config,
            max_examples=max_test_examples,
            device=device,
            strict=strict_baseline,
        )
        if baseline_report is None:
            _record_step(steps, "profile2setup_baseline", "skipped", reason=baseline_error)
        else:
            _record_step(steps, "profile2setup_baseline", "completed", out=str(paths["profile2setup_baseline"]))

    # Load reports when a previous stage was reused or produced by another process.
    base_eval_report = base_eval_report or _load_json(paths["base_eval"])
    sft_eval_report = sft_eval_report or _load_json(paths["sft_eval"])
    baseline_report = baseline_report or _load_json(paths["profile2setup_baseline"])

    final_report = {
        "created_at": _utc_now(),
        "provider": provider,
        "base_model": base_model,
        "sft_model": sft_model_id,
        "inputs": {
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "test_jsonl": str(test_jsonl),
            "variables_config": str(variables_config),
            "profile2setup_checkpoint": None
            if profile2setup_checkpoint is None
            else str(profile2setup_checkpoint),
        },
        "paths": {key: str(value) for key, value in paths.items()},
        "steps": steps,
        "comparison": {
            "base_api": _api_summary(base_eval_report),
            "sft_api": _api_summary(sft_eval_report),
            "profile2setup_baseline": _profile2setup_summary(baseline_report),
        },
        "full_reports": {
            "base_api_eval": base_eval_report,
            "sft_api_eval": sft_eval_report,
            "profile2setup_baseline_eval": baseline_report,
            "sft_job": sft_job,
        },
    }
    _save_json(final_report, paths["comparison_report"])
    return final_report
