"""Qualitative prediction visualization for profile2setup v2 checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from profile2setup.inference.controller import (
    Profile2SetupController,
    assert_no_forbidden_v2_fields,
)
from profile2setup.schema import VARIABLE_ORDER
from profile2setup.training.dataset import filter_records, load_jsonl
from profile2setup.training.preprocessing import (
    load_intensity,
    normalize_intensity,
    resize_intensity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize profile2setup v2 predictions")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--data", required=True, help="Evaluation JSONL path")
    parser.add_argument(
        "--variables-config",
        default="profile2setup/configs/variables.yaml",
        help="Variables YAML config",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for per-example PNG/JSON files")
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--config", default=None, help="Optional model config override")
    parser.add_argument(
        "--task-filter",
        choices=["absolute", "edit", "paired_no_setup", "paired-no-setup"],
        default=None,
    )
    parser.add_argument(
        "--simulate-predicted-profile",
        action="store_true",
        help="Run the optical simulator for predicted setups when record metadata supports it",
    )
    parser.add_argument(
        "--simulation-policy",
        choices=["target_base", "current_base", "auto"],
        default="target_base",
    )
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict", dest="strict", action="store_true")
    strict_group.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "record"


def _short_text(value: str, max_len: int = 80) -> str:
    value = " ".join(str(value).split())
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _resolve_profile_path(record: dict, top_key: str, ref_key: str) -> str | None:
    value = record.get(top_key)
    if value:
        return str(value)
    ref = record.get("profile_loss_reference") or {}
    if isinstance(ref, dict) and ref.get(ref_key):
        return str(ref[ref_key])
    return None


def _profile_paths(record: dict) -> dict:
    current = _resolve_profile_path(record, "current_profile_path", "current_profile_path")
    target = _resolve_profile_path(record, "target_profile_path", "target_profile_path")
    groundtruth = target or current
    return {
        "current_profile_path": current,
        "target_profile_path": target,
        "groundtruth_profile_path": groundtruth,
    }


def _load_display_profile(path: str | None, input_size: int, normalize_mode: str) -> np.ndarray | None:
    if path is None:
        return None
    arr = load_intensity(path)
    arr = normalize_intensity(arr, mode=normalize_mode)
    arr = resize_intensity(arr, size=input_size)
    return np.asarray(arr, dtype=np.float32)


def _display_array(arr: np.ndarray | None, input_size: int) -> np.ndarray:
    if arr is None:
        return np.zeros((input_size, input_size), dtype=np.float32)
    return arr


def _abs_error(predicted: dict, target: dict | None) -> dict | None:
    if target is None:
        return None
    return {name: float(abs(float(predicted[name]) - float(target[name]))) for name in VARIABLE_ORDER}


def _maybe_simulate_profile(
    record: dict,
    predicted_setup: dict,
    simulation_policy: str,
    input_size: int,
    normalize_mode: str,
) -> tuple[np.ndarray | None, str | None]:
    try:
        from profile2setup.evaluation.closed_loop import (
            apply_predicted_setup_to_sim_config,
            load_base_simulator_config_from_metadata,
            resolve_metadata_paths,
            simulate_intensity_from_config,
            _choose_policy_config,
            _eligibility_skip_reason,
        )

        paths = resolve_metadata_paths(record)
        reason = _eligibility_skip_reason(record, paths)
        if reason is not None:
            return None, reason
        target_config = load_base_simulator_config_from_metadata(paths["target_metadata_path"])
        _, base_config, _ = _choose_policy_config(paths, target_config, simulation_policy)
        sim_config = apply_predicted_setup_to_sim_config(base_config, predicted_setup)
        intensity = simulate_intensity_from_config(sim_config)
        intensity = normalize_intensity(intensity, mode=normalize_mode)
        intensity = resize_intensity(intensity, size=input_size)
        return intensity, None
    except Exception as exc:
        return None, str(exc)


def _save_visualization(
    path: Path,
    current_profile: np.ndarray | None,
    groundtruth_profile: np.ndarray | None,
    predicted_profile: np.ndarray | None,
    task_type: str,
    prompt: str,
    input_size: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    current = _display_array(current_profile, input_size)
    groundtruth = _display_array(groundtruth_profile, input_size)

    if predicted_profile is not None:
        panels = [
            ("current profile", current),
            ("groundtruth profile", groundtruth),
            ("predicted simulated profile", predicted_profile),
            ("abs predicted-target", np.abs(predicted_profile - groundtruth)),
        ]
    else:
        if current_profile is not None and groundtruth_profile is not None:
            diff = groundtruth - current
        else:
            diff = np.zeros((input_size, input_size), dtype=np.float32)
        panels = [
            ("current profile", current),
            ("groundtruth profile", groundtruth),
            ("target-current difference", diff),
        ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, image) in zip(axes, panels):
        im = ax.imshow(image, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{task_type}: {_short_text(prompt)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2, sort_keys=True)


def visualize_predictions(
    checkpoint_path,
    data_path,
    variables_config_path,
    out_dir,
    num_examples=10,
    device="auto",
    config_path=None,
    task_filter=None,
    strict=True,
    simulate_predicted_profile=False,
    simulation_policy="target_base",
) -> dict:
    """Run qualitative prediction visualization and write per-example outputs."""
    checkpoint = Path(checkpoint_path)
    data = Path(data_path)
    variables_config = Path(variables_config_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not data.exists():
        raise FileNotFoundError(f"Data JSONL not found: {data}")
    if not variables_config.exists():
        raise FileNotFoundError(f"Variables config not found: {variables_config}")

    controller = Profile2SetupController(
        checkpoint_path=checkpoint,
        device=device,
        variables_config_path=variables_config,
        config_path=config_path,
    )
    records = filter_records(load_jsonl(data), task_filter=task_filter)
    records = records[: int(num_examples)]
    if not records:
        raise RuntimeError("No records selected for visualization")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    skipped = []

    for idx, record in enumerate(records):
        record_id = str(record.get("id") or f"row_{idx}")
        safe_id = f"{idx:03d}_{_safe_filename(record_id)}"
        try:
            assert_no_forbidden_v2_fields(record)
            prediction = controller.predict_record(record)
            predicted_setup = prediction["predicted_routed_setup_physical"]
            groundtruth_setup = prediction["target_setup_physical"] or record.get("target_setup")
            predicted_delta = prediction["predicted_delta_physical"] if prediction["setup_present"] else None
            groundtruth_delta = prediction["target_delta_physical"] or record.get("target_delta")
            paths = _profile_paths(record)

            current_profile = _load_display_profile(
                paths["current_profile_path"], controller.input_size, controller.normalize_mode
            )
            groundtruth_profile = _load_display_profile(
                paths["groundtruth_profile_path"], controller.input_size, controller.normalize_mode
            )

            predicted_profile = None
            predicted_profile_status = "skipped_not_requested"
            if simulate_predicted_profile:
                predicted_profile, reason = _maybe_simulate_profile(
                    record,
                    predicted_setup,
                    simulation_policy,
                    controller.input_size,
                    controller.normalize_mode,
                )
                predicted_profile_status = "available" if predicted_profile is not None else f"skipped: {reason}"

            png_path = output_dir / f"{safe_id}.png"
            json_path = output_dir / f"{safe_id}.json"
            _save_visualization(
                png_path,
                current_profile,
                groundtruth_profile,
                predicted_profile,
                prediction["task_type"],
                prediction["prompt"],
                controller.input_size,
            )

            summary = {
                "record_id": record_id,
                "task_type": prediction["task_type"],
                "prompt": prediction["prompt"],
                "setup_present": prediction["setup_present"],
                "predicted_setup_physical": predicted_setup,
                "groundtruth_setup_physical": groundtruth_setup,
                "absolute_error_physical": _abs_error(predicted_setup, groundtruth_setup),
                "predicted_delta_physical": predicted_delta,
                "groundtruth_delta_physical": groundtruth_delta,
                "profile_paths_used": paths,
                "checkpoint_path": str(checkpoint),
                "prediction_png_path": str(png_path),
                "prediction_json_path": str(json_path),
                "predicted_profile_available": predicted_profile is not None,
                "predicted_profile_status": predicted_profile_status,
            }
            assert_no_forbidden_v2_fields(summary)
            _write_json(json_path, summary)
            examples.append(summary)
        except Exception as exc:
            if strict:
                raise
            skipped.append({"record_id": record_id, "reason": str(exc)})

    result = {
        "checkpoint_path": str(checkpoint),
        "data_path": str(data),
        "variables_config_path": str(variables_config),
        "out_dir": str(output_dir),
        "num_examples_requested": int(num_examples),
        "num_examples_written": len(examples),
        "simulate_predicted_profile": bool(simulate_predicted_profile),
        "simulation_policy": simulation_policy,
        "examples": examples,
        "skipped": skipped,
    }
    assert_no_forbidden_v2_fields(result)
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    args = parse_args()
    result = visualize_predictions(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        variables_config_path=args.variables_config,
        out_dir=args.out_dir,
        num_examples=args.num_examples,
        device=args.device,
        config_path=args.config,
        task_filter=args.task_filter,
        strict=args.strict,
        simulate_predicted_profile=args.simulate_predicted_profile,
        simulation_policy=args.simulation_policy,
    )
    print(f"saved summary JSON: {Path(args.out_dir) / 'summary.json'}")
    print(f"examples written: {result['num_examples_written']}")
    if result["skipped"]:
        print(f"examples skipped: {len(result['skipped'])}")


if __name__ == "__main__":
    main()
