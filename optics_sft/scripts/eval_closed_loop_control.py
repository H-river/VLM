#!/usr/bin/env python3
"""Closed-loop simulator evaluation for physics-aware VLM control adapters."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optical_sim.src.optical_elements import OpticalSetup, setup_from_dict
from optics_sft.physics.prompt_builder import build_physics_prompt
from optics_sft.physics.rendering import save_intensity_png
from optics_sft.physics.sim_adapter import (
    apply_action_to_setup,
    residual_error_px,
    setup_to_safe_metadata,
    simulate_and_measure,
)


CONTROL_KEYS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
OSCILLATION_EPSILON_PX = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run closed-loop simulator evaluation for physics-aware VLM adapters."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("optics_sft/configs/qwen25vl_3b_qlora_physics_mixed.yaml"),
    )
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("../VLM_runs/closed_loop_eval_tmp"))
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--success-threshold-px", type=float, default=2.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use ground-truth actions from private_eval and skip model loading.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install PyYAML to read configs.") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def setup_snapshot_to_config(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    cfg = {
        "source": dict(snapshot.get("source", {})),
        "lens": dict(snapshot.get("lens", {})),
        "sensor": dict(snapshot.get("sensor", {})),
        "geometry": dict(snapshot.get("geometry", {})),
        "camera": dict(snapshot.get("camera", {})),
        "alignment": dict(snapshot.get("alignment", {})),
        "simulation": dict(snapshot.get("simulation", {})),
    }
    cfg["alignment"].setdefault("x_offset", 0.0)
    cfg["alignment"].setdefault("y_offset", 0.0)
    cfg["alignment"].setdefault("tilt_x", 0.0)
    cfg["alignment"].setdefault("tilt_y", 0.0)
    cfg["alignment"].setdefault("defocus", 0.0)
    return cfg


def setup_from_snapshot(snapshot: Mapping[str, Any]) -> OpticalSetup:
    return setup_from_dict(setup_snapshot_to_config(snapshot))


def simulator_config(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    private_eval = row.get("private_eval")
    if not isinstance(private_eval, Mapping):
        return None
    config = private_eval.get("simulator_config")
    return config if isinstance(config, Mapping) else None


def reconstruct_current_setup(row: Mapping[str, Any]) -> tuple[OpticalSetup | None, str | None]:
    private_eval = row.get("private_eval")
    if not isinstance(private_eval, Mapping):
        return None, "private_eval_missing"

    config = simulator_config(row)
    setup_keys = ("current_setup", "initial_setup", "setup")
    if config is not None:
        for key in setup_keys:
            snapshot = config.get(key)
            if isinstance(snapshot, Mapping):
                return setup_from_snapshot(snapshot), None

    for key in setup_keys:
        snapshot = private_eval.get(key)
        if isinstance(snapshot, Mapping):
            return setup_from_snapshot(snapshot), None

    return None, "reconstructable_current_setup_missing"


def reconstruct_target_setup(row: Mapping[str, Any]) -> OpticalSetup | None:
    config = simulator_config(row)
    if config is not None:
        snapshot = config.get("target_setup")
        if isinstance(snapshot, Mapping):
            return setup_from_snapshot(snapshot)
    private_eval = row.get("private_eval")
    if isinstance(private_eval, Mapping):
        snapshot = private_eval.get("target_setup")
        if isinstance(snapshot, Mapping):
            return setup_from_snapshot(snapshot)
    return None


def target_state(row: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str | None]:
    private_eval = row.get("private_eval")
    if not isinstance(private_eval, Mapping):
        return None, "private_eval_missing"
    state = private_eval.get("target_state")
    if isinstance(state, Mapping):
        return state, None
    return None, "target_state_missing"


def render_options(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    private_eval = row.get("private_eval")
    if not isinstance(private_eval, Mapping):
        return None
    config = private_eval.get("simulator_config")
    if isinstance(config, Mapping) and isinstance(config.get("render_options"), Mapping):
        return config["render_options"]
    return None


def prompt_target_image_path(row: Mapping[str, Any], image_root: Path | None, work_dir: Path) -> Path | None:
    prompt_inputs = row.get("prompt_inputs")
    if isinstance(prompt_inputs, Mapping):
        images = prompt_inputs.get("images")
        if isinstance(images, Mapping):
            path_value = images.get("target_image_path")
            if isinstance(path_value, str) and path_value:
                path = Path(path_value)
                if path.is_absolute():
                    return path
                if image_root is not None:
                    return image_root / path

    target_setup = reconstruct_target_setup(row)
    if target_setup is None:
        return None
    rendered = simulate_and_measure(target_setup)
    path = work_dir / "targets" / f"{row.get('sample_id', 'sample')}_target.png"
    save_intensity_png(rendered["intensity"], path, render_options(row))
    return path


def action_from_mapping(obj: Mapping[str, Any] | None, context: str) -> tuple[dict[str, float] | None, str | None]:
    if not isinstance(obj, Mapping):
        return None, f"{context}_missing"
    missing = [key for key in CONTROL_KEYS if key not in obj]
    if missing:
        return None, f"{context}_missing_keys:{','.join(missing)}"
    non_numeric = [key for key in CONTROL_KEYS if not is_number(obj.get(key))]
    if non_numeric:
        return None, f"{context}_non_numeric_keys:{','.join(non_numeric)}"
    return {key: float(obj[key]) for key in CONTROL_KEYS}, None


def extract_control_action(parsed_prediction: Mapping[str, Any] | None) -> tuple[dict[str, float] | None, str | None]:
    if not isinstance(parsed_prediction, Mapping):
        return None, "prediction_json_missing"
    plan = parsed_prediction.get("control_plan")
    action, error = action_from_mapping(plan if isinstance(plan, Mapping) else None, "control_plan")
    if action is not None:
        return action, None

    recommended_steps = parsed_prediction.get("recommended_steps")
    if isinstance(recommended_steps, list) and recommended_steps:
        first_step = recommended_steps[0]
        if isinstance(first_step, Mapping):
            step_action = first_step.get("action")
            action, step_error = action_from_mapping(
                step_action if isinstance(step_action, Mapping) else None,
                "recommended_steps[0].action",
            )
            if action is not None:
                return action, None
            return None, f"{error}; {step_error}"
    return None, error


def dry_run_action(row: Mapping[str, Any], step_index: int) -> tuple[dict[str, float] | None, str | None]:
    private_eval = row.get("private_eval")
    if not isinstance(private_eval, Mapping):
        return None, "private_eval_missing"

    all_actions = private_eval.get("all_actions")
    if isinstance(all_actions, list):
        if step_index >= len(all_actions):
            return None, "ground_truth_action_exhausted"
        action_obj = all_actions[step_index]
        return action_from_mapping(action_obj if isinstance(action_obj, Mapping) else None, "all_actions")

    true_plan = private_eval.get("true_control_plan")
    if isinstance(true_plan, Mapping):
        if step_index > 0:
            return None, "single_step_true_control_plan_exhausted"
        return action_from_mapping(true_plan, "true_control_plan")

    target = row.get("target")
    if isinstance(target, Mapping):
        plan = target.get("control_plan")
        if isinstance(plan, Mapping):
            if step_index > 0:
                return None, "single_step_target_control_plan_exhausted"
            return action_from_mapping(plan, "target.control_plan")

        recommended_steps = target.get("recommended_steps")
        if isinstance(recommended_steps, list):
            if step_index >= len(recommended_steps):
                return None, "target_recommended_steps_exhausted"
            step = recommended_steps[step_index]
            if isinstance(step, Mapping):
                return action_from_mapping(
                    step.get("action") if isinstance(step.get("action"), Mapping) else None,
                    "target.recommended_steps.action",
                )

    return None, "ground_truth_action_missing"


def build_step_prompt_row(
    source_row: Mapping[str, Any],
    current_image_path: Path,
    target_image_path: Path,
    setup: OpticalSetup,
) -> dict[str, Any]:
    return {
        "sample_id": source_row.get("sample_id"),
        "sample_type": "inverse_control",
        "prompt_inputs": {
            "images": {
                "current_image_path": current_image_path.name,
                "target_image_path": target_image_path.name,
            },
            "safe_setup_metadata": setup_to_safe_metadata(setup),
        },
        "target": {},
        "private_eval": {},
        "split_tags": ["closed_loop_eval"],
    }


def load_rgb_image(path: Path) -> Any:
    from PIL import Image

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    image = Image.open(path).convert("RGB")
    image.load()
    return image


def build_user_messages(prompt: str, image_count: int) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(image_count)]
            + [{"type": "text", "text": prompt}],
        }
    ]


def generation_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    generation_cfg = config.get("generation", {})
    if not isinstance(generation_cfg, Mapping):
        generation_cfg = {}
    do_sample = bool(generation_cfg.get("do_sample", False))
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation_cfg.get("max_new_tokens", 512)),
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs["temperature"] = float(generation_cfg.get("temperature", 0.7))
    return kwargs


def first_json_object_text(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        direct_error = str(exc)
    else:
        if isinstance(parsed, dict):
            return parsed, None
        return None, "generated text is valid JSON but not an object"

    block = first_json_object_text(text)
    if block is None:
        return None, direct_error
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "extracted JSON is not an object"
    return parsed, None


def dtype_from_name(torch: Any, name: str) -> Any:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


def require_inference_imports() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Missing inference dependencies. Install torch, transformers, peft, bitsandbytes, and Pillow."
        ) from exc

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def load_model_and_processor(config: Mapping[str, Any], adapter_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    deps = require_inference_imports()
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, Mapping):
        raise ValueError("Config model section must be an object.")
    model_name = str(model_cfg["name"])
    local_files_only = bool(model_cfg.get("local_files_only", True))
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))

    processor = deps["AutoProcessor"].from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    quantization_config = deps["BitsAndBytesConfig"](
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype_from_name(
            deps["torch"],
            str(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        ),
        bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
    )
    base_model = deps["AutoModelForImageTextToText"].from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model = deps["PeftModel"].from_pretrained(
        base_model,
        adapter_path,
        local_files_only=True,
    )
    model.eval()
    return processor, model, deps


def infer_action(
    prompt: str,
    current_image_path: Path,
    target_image_path: Path,
    config: Mapping[str, Any],
    processor: Any,
    model: Any,
    deps: Mapping[str, Any],
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    messages = build_user_messages(prompt, image_count=2)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[load_rgb_image(current_image_path), load_rgb_image(target_image_path)],
        return_tensors="pt",
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with deps["torch"].no_grad():
        generated = model.generate(
            **inputs,
            **generation_kwargs(config),
        )
    prompt_len = inputs["input_ids"].shape[-1]
    raw_text = processor.batch_decode(
        generated[:, prompt_len:],
        skip_special_tokens=True,
    )[0].strip()
    parsed_json, parse_error = extract_json_object(raw_text)
    action, action_error = extract_control_action(parsed_json)
    metadata = {
        "raw_prediction_text": raw_text,
        "parsed_json": parsed_json,
        "parse_error": parse_error,
        "action_error": action_error,
    }
    return action, metadata


def residual_sequence_has_oscillation(residuals: list[float]) -> bool:
    signs: list[int] = []
    for previous, current in zip(residuals, residuals[1:]):
        delta = current - previous
        if delta > OSCILLATION_EPSILON_PX:
            signs.append(1)
        elif delta < -OSCILLATION_EPSILON_PX:
            signs.append(-1)
    return any(a != b for a, b in zip(signs, signs[1:]))


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_case(
    row: dict[str, Any],
    *,
    config: Mapping[str, Any],
    image_root: Path | None,
    work_dir: Path,
    max_steps: int,
    success_threshold_px: float,
    dry_run: bool,
    processor: Any = None,
    model: Any = None,
    deps: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id", "sample"))
    detail: dict[str, Any] = {
        "sample_id": row.get("sample_id"),
        "sample_type": row.get("sample_type"),
        "dry_run": dry_run,
        "success": False,
        "diverged": False,
        "oscillated": False,
        "invalid_action_count": 0,
        "action_request_count": 0,
        "steps": [],
        "error": None,
    }

    setup, setup_error = reconstruct_current_setup(row)
    if setup is None:
        detail["error"] = setup_error
        return detail
    target, target_error = target_state(row)
    if target is None:
        detail["error"] = target_error
        return detail

    case_dir = work_dir / "images" / sample_id
    case_dir.mkdir(parents=True, exist_ok=True)
    target_image_path = prompt_target_image_path(row, image_root, case_dir)
    if not dry_run and target_image_path is None:
        detail["error"] = "target_image_missing"
        return detail

    residuals: list[float] = []
    current_setup = setup
    initial_residual: float | None = None
    steps_to_success: int | None = None

    for step_index in range(max_steps + 1):
        simulation = simulate_and_measure(current_setup)
        current_state = simulation["state"]
        residual_before = residual_error_px(current_state, target)
        residuals.append(residual_before)
        if initial_residual is None:
            initial_residual = residual_before

        current_image_path = case_dir / f"step_{step_index:02d}_current.png"
        save_intensity_png(simulation["intensity"], current_image_path, render_options(row))

        if residual_before <= success_threshold_px:
            detail["success"] = True
            steps_to_success = step_index
            break
        if step_index >= max_steps:
            break

        prompt = None
        prediction_metadata: dict[str, Any] = {}
        if dry_run:
            action, action_error = dry_run_action(row, step_index)
            prediction_metadata["action_error"] = action_error
        else:
            if target_image_path is None:
                detail["error"] = "target_image_missing"
                break
            prompt_row = build_step_prompt_row(row, current_image_path, target_image_path, current_setup)
            prompt = build_physics_prompt(prompt_row)
            action, prediction_metadata = infer_action(
                prompt=prompt,
                current_image_path=current_image_path,
                target_image_path=target_image_path,
                config=config,
                processor=processor,
                model=model,
                deps=deps or {},
            )

        detail["action_request_count"] += 1
        step_detail: dict[str, Any] = {
            "step_index": step_index,
            "current_image_path": str(current_image_path),
            "target_image_path": str(target_image_path) if target_image_path is not None else None,
            "residual_before_px": residual_before,
            "state_before": current_state,
            "action": action,
            "prompt": prompt,
            **prediction_metadata,
        }

        if action is None:
            detail["invalid_action_count"] += 1
            step_detail["invalid_action"] = True
            step_detail["error"] = prediction_metadata.get("action_error") or "action_missing"
            detail["steps"].append(step_detail)
            break

        current_setup = apply_action_to_setup(current_setup, action)
        after = simulate_and_measure(current_setup)
        residual_after = residual_error_px(after["state"], target)
        step_detail.update(
            {
                "invalid_action": False,
                "residual_after_px": residual_after,
                "state_after": after["state"],
            }
        )
        detail["steps"].append(step_detail)

    final_residual = residuals[-1] if residuals else None
    detail.update(
        {
            "initial_residual_px": initial_residual,
            "final_residual_px": final_residual,
            "steps_to_success": steps_to_success,
            "diverged": (
                initial_residual is not None
                and final_residual is not None
                and final_residual > initial_residual
            ),
            "oscillated": residual_sequence_has_oscillation(residuals),
            "residuals_px": residuals,
        }
    )
    return detail


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(details)
    valid_cases = [row for row in details if row.get("initial_residual_px") is not None]
    final_residuals = [float(row["final_residual_px"]) for row in valid_cases if is_number(row.get("final_residual_px"))]
    successful = [row for row in valid_cases if row.get("success")]
    steps_to_success = [
        float(row["steps_to_success"])
        for row in successful
        if is_number(row.get("steps_to_success"))
    ]
    action_requests = sum(int(row.get("action_request_count", 0)) for row in details)
    invalid_actions = sum(int(row.get("invalid_action_count", 0)) for row in details)
    error_counts: dict[str, int] = {}
    for row in details:
        error = row.get("error")
        if error:
            error_counts[str(error)] = error_counts.get(str(error), 0) + 1
        for step in row.get("steps", []):
            if isinstance(step, Mapping) and step.get("invalid_action"):
                step_error = step.get("error") or "invalid_action"
                error_counts[str(step_error)] = error_counts.get(str(step_error), 0) + 1

    return {
        "count": count,
        "valid_case_count": len(valid_cases),
        "success_rate": rate(len(successful), len(valid_cases)),
        "mean_steps_to_success": safe_mean(steps_to_success),
        "median_final_residual_px": safe_median(final_residuals),
        "mean_final_residual_px": safe_mean(final_residuals),
        "divergence_rate": rate(sum(1 for row in valid_cases if row.get("diverged")), len(valid_cases)),
        "oscillation_rate": rate(sum(1 for row in valid_cases if row.get("oscillated")), len(valid_cases)),
        "invalid_action_rate": rate(invalid_actions, action_requests),
        "invalid_action_count": invalid_actions,
        "action_request_count": action_requests,
        "error_counts": error_counts,
    }


def main() -> None:
    args = parse_args()
    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative")
    if args.success_threshold_px < 0.0:
        raise ValueError("--success-threshold-px must be non-negative")

    config = load_yaml(args.config)
    if not args.dry_run:
        if args.adapter_path is None:
            raise ValueError("--adapter-path is required unless --dry-run is set")
        if args.image_root is None:
            raise ValueError("--image-root is required unless --dry-run is set")
        if not args.image_root.exists():
            raise FileNotFoundError(f"Image root does not exist: {args.image_root}")
        processor, model, deps = load_model_and_processor(config, args.adapter_path)
    else:
        processor = model = None
        deps = None

    rows = [
        row
        for row in read_jsonl(args.test_jsonl, args.max_samples)
        if row.get("sample_type") in {"inverse_control", "trajectory"}
    ]
    args.work_dir.mkdir(parents=True, exist_ok=True)

    details = [
        evaluate_case(
            row,
            config=config,
            image_root=args.image_root,
            work_dir=args.work_dir,
            max_steps=args.max_steps,
            success_threshold_px=args.success_threshold_px,
            dry_run=args.dry_run,
            processor=processor,
            model=model,
            deps=deps,
        )
        for row in rows
    ]
    report = {
        "config": str(args.config),
        "test_jsonl": str(args.test_jsonl),
        "image_root": str(args.image_root) if args.image_root is not None else None,
        "adapter_path": str(args.adapter_path) if args.adapter_path is not None else None,
        "work_dir": str(args.work_dir),
        "dry_run": args.dry_run,
        "max_steps": args.max_steps,
        "success_threshold_px": args.success_threshold_px,
        "summary": summarize(details),
        "cases": details,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
