"""Optional reasoned VLM front-end prediction CLI for profile2setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from profile2setup.reasoning_vlm.image_rendering import render_reasoning_images
from profile2setup.reasoning_vlm.intent_features import (
    build_allowed_change_mask,
    build_fixed_change_mask,
    build_relevance_prior,
    extract_canonical_prompt,
)
from profile2setup.reasoning_vlm.vlm_parser import build_vlm_user_payload, get_reasoning_command
from profile2setup.schema import VARIABLE_ORDER, validate_setup_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional reasoned VLM front-end profile2setup prediction")
    parser.add_argument("--checkpoint", default=None, help="Optional profile2setup checkpoint path")
    parser.add_argument("--current-profile", required=True, help="Current profile intensity.npy")
    parser.add_argument("--target-profile", required=True, help="Target profile intensity.npy")
    parser.add_argument("--current-setup", default=None, help="JSON file containing canonical current setup")
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument(
        "--variables-config",
        default="profile2setup/configs/variables.yaml",
        help="Variables YAML config",
    )
    parser.add_argument("--config", default=None, help="Optional checkpoint config override")
    parser.add_argument("--out-dir", required=True, help="Output artifact directory")
    parser.add_argument(
        "--reasoning-mode",
        choices=["mock_rule_based", "json_file", "manual_text"],
        default="mock_rule_based",
    )
    parser.add_argument("--reasoning-json", default=None, help="Reasoning JSON for mode=json_file")
    parser.add_argument("--manual-text", default=None, help="Reasoning text for mode=manual_text")
    parser.add_argument(
        "--task-type",
        choices=["edit", "paired_no_setup"],
        default=None,
        help="Optional task type override; inferred from current setup when omitted",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _load_current_setup(path: str | None) -> dict | None:
    if path is None:
        return None
    with open(path, "r") as f:
        setup = json.load(f)
    if not validate_setup_dict(setup):
        raise ValueError("current setup JSON must be a canonical profile2setup v2 setup dict")
    return setup


def _build_record(args: argparse.Namespace, current_setup: dict | None, canonical_prompt: str | None = None) -> dict:
    task_type = args.task_type
    if task_type is None:
        task_type = "edit" if current_setup is not None else "paired_no_setup"
    return {
        "id": "reasoned_vlm_predict",
        "task_type": task_type,
        "prompt": canonical_prompt or args.prompt,
        "current_profile_path": args.current_profile,
        "target_profile_path": args.target_profile,
        "current_setup": current_setup,
        "target_setup": None,
        "target_delta": None,
        "profile_loss_reference": {
            "current_profile_path": args.current_profile,
            "target_profile_path": args.target_profile,
        },
    }


def _vector_to_dict(values) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != len(VARIABLE_ORDER):
        raise ValueError(f"Expected vector length {len(VARIABLE_ORDER)}, got {arr.shape[0]}")
    return {name: float(arr[idx]) for idx, name in enumerate(VARIABLE_ORDER)}


def _constraint_violations(raw_delta, masked_delta, fixed_change_mask: list[float]) -> list[dict[str, float | str]]:
    raw = np.asarray(raw_delta, dtype=np.float64).reshape(-1)
    masked = np.asarray(masked_delta, dtype=np.float64).reshape(-1)
    violations = []
    for idx, variable in enumerate(VARIABLE_ORDER):
        if fixed_change_mask[idx] <= 0.0:
            continue
        if abs(float(raw[idx])) > 1e-8:
            violations.append(
                {
                    "variable": variable,
                    "raw_delta_norm": float(raw[idx]),
                    "applied_delta_norm": float(masked[idx]),
                }
            )
    return violations


def _run_model_prediction(
    *,
    checkpoint: str,
    record: dict,
    variables_config_path: str,
    config_path: str | None,
    device: str,
    allowed_change_mask: list[float],
    fixed_change_mask: list[float],
) -> dict:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for checkpoint-backed prediction") from exc

    from profile2setup.inference.controller import Profile2SetupController, assert_no_forbidden_v2_fields
    from profile2setup.inference.routing import (
        apply_allowed_change_mask_to_delta,
        apply_fixed_change_mask_to_delta,
        route_setup_prediction,
    )
    from profile2setup.training.normalization import denormalize_delta_vector, denormalize_setup_vector

    controller = Profile2SetupController(
        checkpoint_path=checkpoint,
        device=device,
        variables_config_path=variables_config_path,
        config_path=config_path,
    )
    sample = controller._build_one_sample(record)
    batch = {
        "profile": sample["profile"].unsqueeze(0).to(controller.device),
        "prompt_tokens": sample["prompt_tokens"].unsqueeze(0).to(controller.device),
        "current_setup": sample["current_setup"].unsqueeze(0).to(controller.device),
        "setup_present": sample["setup_present"].unsqueeze(0).to(controller.device),
        "intent_features": sample["intent_features"].unsqueeze(0).to(controller.device),
    }

    with torch.no_grad():
        outputs = controller.model(
            batch["profile"],
            batch["prompt_tokens"],
            batch["current_setup"],
            setup_present=batch["setup_present"],
            intent_features=batch["intent_features"],
        )
        raw_delta = outputs["delta"]
        masked_delta = apply_allowed_change_mask_to_delta(raw_delta, allowed_change_mask)
        masked_delta = apply_fixed_change_mask_to_delta(masked_delta, fixed_change_mask)
        masked_outputs = dict(outputs)
        masked_outputs["delta"] = masked_delta
        routed = route_setup_prediction(
            masked_outputs,
            batch["current_setup"],
            batch["setup_present"],
            prefer_absolute_when_setup_missing=True,
        )
        change_confidence = torch.sigmoid(outputs["change_logits"])

    absolute_norm = outputs["absolute"].detach().cpu().numpy()[0]
    raw_delta_norm = raw_delta.detach().cpu().numpy()[0]
    masked_delta_norm = masked_delta.detach().cpu().numpy()[0]
    routed_norm = routed.detach().cpu().numpy()[0]
    setup_present = int(float(sample["setup_present"].detach().cpu().item()) > 0.0)

    result = {
        "status": "predicted",
        "checkpoint_path": str(checkpoint),
        "record_id": sample["record_id"],
        "task_type": sample["task_type"],
        "prompt": sample["prompt"],
        "setup_present": setup_present,
        "allowed_change_mask": allowed_change_mask,
        "fixed_change_mask": fixed_change_mask,
        "predicted_absolute_norm": _vector_to_dict(absolute_norm),
        "predicted_delta_raw_norm": _vector_to_dict(raw_delta_norm),
        "predicted_delta_masked_norm": _vector_to_dict(masked_delta_norm),
        "predicted_routed_setup_norm": _vector_to_dict(routed_norm),
        "predicted_absolute_physical": denormalize_setup_vector(absolute_norm, controller.variables_config),
        "predicted_delta_raw_physical": denormalize_delta_vector(raw_delta_norm, controller.variables_config),
        "predicted_delta_masked_physical": denormalize_delta_vector(masked_delta_norm, controller.variables_config),
        "predicted_routed_setup_physical": denormalize_setup_vector(routed_norm, controller.variables_config),
        "change_confidence": _vector_to_dict(change_confidence.detach().cpu().numpy()[0]),
        "constraint_violations": _constraint_violations(raw_delta_norm, masked_delta_norm, fixed_change_mask),
    }
    assert_no_forbidden_v2_fields(result)
    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current_setup = _load_current_setup(args.current_setup)
    rendered = render_reasoning_images(args.current_profile, args.target_profile, out_dir)
    payload = build_vlm_user_payload(
        args.prompt,
        rendered["current_profile"],
        rendered["target_profile"],
        rendered["difference_profile"],
        current_setup=current_setup,
    )
    _write_json(out_dir / "vlm_user_payload.json", payload)

    base_record = _build_record(args, current_setup)
    reasoning_command = get_reasoning_command(
        mode=args.reasoning_mode,
        record=base_record,
        rendered_image_paths=rendered,
        json_file=args.reasoning_json,
        manual_text=args.manual_text,
    )
    _write_json(out_dir / "reasoned_command.json", reasoning_command)

    canonical_prompt = extract_canonical_prompt(reasoning_command)
    allowed_change_mask = build_allowed_change_mask(reasoning_command)
    fixed_change_mask = build_fixed_change_mask(reasoning_command)
    relevance_prior = build_relevance_prior(reasoning_command)
    intent_features = {
        "canonical_prompt": canonical_prompt,
        "allowed_change_mask": allowed_change_mask,
        "fixed_change_mask": fixed_change_mask,
        "relevance_prior": relevance_prior,
    }
    _write_json(out_dir / "intent_features.json", intent_features)

    prediction_record = _build_record(args, current_setup, canonical_prompt=canonical_prompt)
    prediction_record["reasoning_command"] = reasoning_command
    if args.checkpoint is None:
        prediction = {
            "status": "skipped_checkpoint_not_provided",
            "message": "Rendered images and reasoned_command.json were saved; pass --checkpoint to run model inference.",
        }
    else:
        try:
            prediction = _run_model_prediction(
                checkpoint=args.checkpoint,
                record=prediction_record,
                variables_config_path=args.variables_config,
                config_path=args.config,
                device=args.device,
                allowed_change_mask=allowed_change_mask,
                fixed_change_mask=fixed_change_mask,
            )
        except Exception as exc:  # noqa: BLE001
            prediction = {
                "status": "model_inference_failed",
                "message": (
                    "Rendered images, reasoned_command.json, and intent_features.json were saved; "
                    f"checkpoint-backed inference failed at integration point: {type(exc).__name__}: {exc}"
                ),
            }

    _write_json(out_dir / "prediction.json", prediction)
    print(json.dumps(
        {
            "out_dir": str(out_dir),
            "reasoned_command": str(out_dir / "reasoned_command.json"),
            "intent_features": str(out_dir / "intent_features.json"),
            "prediction": str(out_dir / "prediction.json"),
            "prediction_status": prediction["status"],
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
