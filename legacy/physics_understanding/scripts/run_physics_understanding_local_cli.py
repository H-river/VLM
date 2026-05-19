"""Run local Profile2SetupModel baseline on physics-understanding probes."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from profile2setup.evaluation.param_metrics import (
    FALLBACK_TOLERANCES,
    denormalize_delta_vector,
    denormalize_setup_vector,
    load_tolerances,
)
from legacy.physics_understanding.evaluation.physics_understanding_schema import (
    CANONICAL_VARIABLE_ORDER,
    load_probe_jsonl,
)
from profile2setup.inference.routing import route_setup_prediction
from profile2setup.models import build_model_from_config
from profile2setup.training.dataset import Profile2SetupDataset, profile2setup_collate_fn
from profile2setup.training.normalization import load_variables_config
from profile2setup.training.text import load_vocab
from profile2setup.training.utils import get_device, move_batch_to_device

OBSERVED_PROFILE_CHANGE_FIELDS = (
    "centroid_x",
    "centroid_y",
    "beam_width_x",
    "beam_width_y",
    "peak_intensity",
    "total_intensity",
)
IMAGE_INPUT_MODES = {"images_only", "prompt_plus_images", "shuffled_prompt", "conflict"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local Profile2SetupModel as an image-driven baseline on "
            "physics-understanding probes."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Path to local profile2setup checkpoint")
    parser.add_argument("--probes", required=True, help="Input physics-understanding probe JSONL")
    parser.add_argument("--out", required=True, help="Output local prediction JSONL")
    parser.add_argument("--variables-config", required=True, help="Variables YAML config")
    parser.add_argument("--vocab", required=True, help="Tokenizer vocabulary JSON")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--max-text-len", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--normalize-mode",
        choices=("max", "max_log", "none"),
        default=None,
        help="Profile normalization mode. Defaults to checkpoint config or max_log.",
    )
    return parser.parse_args()


def _load_checkpoint(path: Path, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _validate_checkpoint_order(checkpoint: dict) -> None:
    order = checkpoint.get("variable_order")
    if order is not None and list(order) != list(CANONICAL_VARIABLE_ORDER):
        raise ValueError(
            "Checkpoint variable_order must match canonical order "
            f"{CANONICAL_VARIABLE_ORDER}; got {order}"
        )


def _resolve_path(path_value: Any, repo_root: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _usable_profile_input(probe: dict, repo_root: Path) -> tuple[bool, str | None]:
    current = _resolve_path(probe.get("current_profile_path"), repo_root)
    target = _resolve_path(probe.get("target_profile_path"), repo_root)
    if current is None and target is None:
        return False, "missing current_profile_path and target_profile_path"
    missing: list[str] = []
    if current is not None and not current.exists():
        missing.append(f"current_profile_path not found: {current}")
    if target is not None and not target.exists():
        missing.append(f"target_profile_path not found: {target}")
    if missing:
        return False, "; ".join(missing)
    return True, None


def _local_prompt(probe: dict) -> str:
    if probe.get("input_mode") == "images_only":
        return "Use the provided profiles to infer the setup change."
    return str(probe.get("prompt") or "")


def _probe_to_dataset_record(probe: dict) -> dict[str, Any]:
    return {
        "id": probe["probe_id"],
        "task_type": probe["task_type"],
        "prompt": _local_prompt(probe),
        "current_profile_path": probe.get("current_profile_path"),
        "target_profile_path": probe.get("target_profile_path"),
        "current_setup": probe.get("current_setup"),
        "target_setup": None,
        "target_delta": None,
        "profile_loss_reference": {
            "current_profile_path": probe.get("current_profile_path"),
            "target_profile_path": probe.get("target_profile_path"),
        },
    }


def _write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _vector_to_setup(values: Any, variables_config: dict) -> dict[str, float]:
    setup = denormalize_setup_vector(np.asarray(values, dtype=np.float64), variables_config)
    return {name: float(setup[name]) for name in CANONICAL_VARIABLE_ORDER}


def _vector_to_delta(values: Any, variables_config: dict) -> dict[str, float]:
    delta = denormalize_delta_vector(np.asarray(values, dtype=np.float64), variables_config)
    return {name: float(delta[name]) for name in CANONICAL_VARIABLE_ORDER}


def _change_direction(delta: dict[str, float], tolerances: dict[str, float]) -> dict[str, str]:
    directions: dict[str, str] = {}
    for name in CANONICAL_VARIABLE_ORDER:
        value = float(delta[name])
        tolerance = float(tolerances[name])
        if value > tolerance:
            directions[name] = "increase"
        elif value < -tolerance:
            directions[name] = "decrease"
        else:
            directions[name] = "unchanged"
    return directions


def _changed_variables(direction: dict[str, str]) -> list[str]:
    return [name for name in CANONICAL_VARIABLE_ORDER if direction[name] != "unchanged"]


def _confidence_from_logits(change_logits: np.ndarray | None) -> tuple[float | None, dict[str, float] | None]:
    if change_logits is None:
        return None, None
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(change_logits, dtype=np.float64).reshape(-1)))
    if probabilities.shape[0] != len(CANONICAL_VARIABLE_ORDER):
        return None, None
    per_variable = {
        name: float(probabilities[idx])
        for idx, name in enumerate(CANONICAL_VARIABLE_ORDER)
    }
    return float(np.mean(probabilities)), per_variable


def _prediction_object(
    *,
    probe: dict,
    predicted_setup: dict[str, float],
    predicted_delta: dict[str, float],
    confidence: float | None,
    tolerances: dict[str, float],
) -> dict[str, Any]:
    direction = _change_direction(predicted_delta, tolerances)
    current_setup = probe.get("current_setup") if isinstance(probe.get("current_setup"), dict) else None
    return {
        "valid": True,
        "task_type": probe["task_type"],
        "observed_profile_change": {
            field: "unknown" for field in OBSERVED_PROFILE_CHANGE_FIELDS
        },
        "setup_understanding": {
            "current_setup": current_setup,
            "target_setup": predicted_setup,
            "changed_variables": _changed_variables(direction),
            "change_direction": direction,
            "notes": (
                "Local Profile2SetupModel baseline prediction. This row is not evidence "
                "that the local model understood the prompt semantics."
            ),
        },
        "predicted_delta": predicted_delta,
        "predicted_setup": predicted_setup,
        "confidence": 0.0 if confidence is None else float(confidence),
        "reasoning_summary": (
            "Image/text-conditioned local PyTorch baseline output; no native rejection head."
        ),
        "rejection_reason": "",
    }


def _base_row(probe: dict, *, checkpoint: str) -> dict[str, Any]:
    return {
        "probe_id": probe["probe_id"],
        "record_id": probe["probe_id"],
        "probe_type": probe["probe_type"],
        "input_mode": probe["input_mode"],
        "model": "local_profile2setup",
        "checkpoint": checkpoint,
        "raw_response": "",
        "prediction": None,
        "status": "pending",
        "error": None,
        "valid_json": False,
        "invalid_forced_prediction": False,
        "local_baseline_note": (
            "Local Profile2SetupModel baseline; use this to test image-driven behavior, "
            "not language understanding."
        ),
    }


def _not_applicable_row(probe: dict, *, checkpoint: str, reason: str) -> dict[str, Any]:
    row = _base_row(probe, checkpoint=checkpoint)
    row.update(
        {
            "status": "not_applicable",
            "error": reason,
            "valid_json": False,
        }
    )
    return row


def _build_runnable_records(
    probes: list[dict],
    *,
    repo_root: Path,
    checkpoint: str,
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    dataset_records: list[dict] = []
    runnable_probes: list[dict] = []
    rows_by_probe_id: dict[str, dict] = {}
    for probe in probes:
        row = _base_row(probe, checkpoint=checkpoint)
        rows_by_probe_id[probe["probe_id"]] = row

        if probe.get("input_mode") == "prompt_only":
            rows_by_probe_id[probe["probe_id"]] = _not_applicable_row(
                probe,
                checkpoint=checkpoint,
                reason="prompt_only probe has no profile tensor input for the local model",
            )
            continue

        usable, reason = _usable_profile_input(probe, repo_root)
        if not usable:
            rows_by_probe_id[probe["probe_id"]] = _not_applicable_row(
                probe,
                checkpoint=checkpoint,
                reason=reason or "no usable profile input",
            )
            continue

        dataset_records.append(_probe_to_dataset_record(probe))
        runnable_probes.append(probe)
    return dataset_records, runnable_probes, rows_by_probe_id


def run_physics_understanding_local(
    *,
    checkpoint_path,
    probes_path,
    out_path,
    variables_config_path,
    vocab_path,
    input_size: int = 128,
    max_text_len: int = 32,
    limit: int | None = None,
    batch_size: int = 32,
    device: str = "auto",
    normalize_mode: str | None = None,
    repo_root=None,
) -> dict[str, Any]:
    probes_file = Path(probes_path)
    checkpoint_file = Path(checkpoint_path)
    output_file = Path(out_path)
    variables_file = Path(variables_config_path)
    vocab_file = Path(vocab_path)
    root = Path(repo_root) if repo_root is not None else Path.cwd()

    probes = load_probe_jsonl(probes_file)
    if limit is not None:
        probes = probes[: int(limit)]

    checkpoint = _load_checkpoint(checkpoint_file, map_location="cpu")
    _validate_checkpoint_order(checkpoint)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing config")
    data_cfg = config.get("data") or {}
    resolved_normalize_mode = normalize_mode or data_cfg.get("normalize_mode", "max_log")

    vocab = load_vocab(vocab_file)
    variables_config = load_variables_config(variables_file)
    tolerances = load_tolerances(variables_config) if variables_config else dict(FALLBACK_TOLERANCES)
    resolved_device = get_device(device)

    model = build_model_from_config(config, vocab_size=len(vocab)).to(resolved_device)
    model_state = checkpoint.get("model_state_dict")
    if model_state is None:
        raise ValueError("Checkpoint is missing model_state_dict")
    model.load_state_dict(model_state)
    model.eval()

    dataset_records, runnable_probes, rows_by_probe_id = _build_runnable_records(
        probes,
        repo_root=root,
        checkpoint=str(checkpoint_file),
    )

    completed = 0
    errors = 0
    forced_invalid = 0
    if dataset_records:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_jsonl = Path(tmpdir) / "local_probe_dataset.jsonl"
            _write_jsonl(dataset_records, tmp_jsonl)
            dataset = Profile2SetupDataset(
                jsonl_path=tmp_jsonl,
                variables_config_path=variables_file,
                vocab=vocab,
                input_size=int(input_size),
                max_text_len=int(max_text_len),
                normalize_mode=resolved_normalize_mode,
                strict=True,
            )
            if len(dataset) != len(runnable_probes):
                raise RuntimeError(
                    f"dataset length mismatch: {len(dataset)} vs {len(runnable_probes)} runnable probes"
                )
            loader = DataLoader(
                dataset,
                batch_size=int(batch_size),
                shuffle=False,
                num_workers=0,
                collate_fn=profile2setup_collate_fn,
            )

            cursor = 0
            with torch.no_grad():
                for batch in loader:
                    probes_for_batch = runnable_probes[cursor : cursor + len(batch["record_id"])]
                    cursor += len(probes_for_batch)
                    batch = move_batch_to_device(batch, resolved_device)
                    try:
                        outputs = model(
                            batch["profile"],
                            batch["prompt_tokens"],
                            batch["current_setup"],
                            setup_present=batch["setup_present"],
                            intent_features=batch.get("intent_features"),
                        )
                        routed = route_setup_prediction(
                            outputs,
                            batch["current_setup"],
                            batch["setup_present"],
                            prefer_absolute_when_setup_missing=True,
                        )
                        absolute_np = outputs["absolute"].detach().cpu().numpy()
                        delta_np = outputs["delta"].detach().cpu().numpy()
                        routed_np = routed.detach().cpu().numpy()
                        logits_np = None
                        if "change_logits" in outputs:
                            logits_np = outputs["change_logits"].detach().cpu().numpy()

                        for idx, probe in enumerate(probes_for_batch):
                            predicted_setup = _vector_to_setup(routed_np[idx], variables_config)
                            predicted_delta = _vector_to_delta(delta_np[idx], variables_config)
                            confidence, change_confidence = _confidence_from_logits(
                                None if logits_np is None else logits_np[idx]
                            )
                            prediction = _prediction_object(
                                probe=probe,
                                predicted_setup=predicted_setup,
                                predicted_delta=predicted_delta,
                                confidence=confidence,
                                tolerances=tolerances,
                            )
                            row = rows_by_probe_id[probe["probe_id"]]
                            invalid_forced = not bool(probe.get("expected_valid", True))
                            if invalid_forced:
                                forced_invalid += 1
                            row.update(
                                {
                                    "status": "completed",
                                    "valid_json": True,
                                    "prediction": prediction,
                                    "predicted_absolute_setup": _vector_to_setup(
                                        absolute_np[idx], variables_config
                                    ),
                                    "change_confidence": change_confidence,
                                    "invalid_forced_prediction": invalid_forced,
                                    "local_input_note": (
                                        "Prompt tokens were passed through the trained local text encoder, "
                                        "but this baseline has no rejection head and should not be interpreted "
                                        "as prompt understanding."
                                    ),
                                }
                            )
                            completed += 1
                    except Exception as exc:  # noqa: BLE001 - record batch failure as per-probe errors.
                        message = f"{type(exc).__name__}: {exc}"
                        for probe in probes_for_batch:
                            rows_by_probe_id[probe["probe_id"]].update(
                                {
                                    "status": "error",
                                    "error": message,
                                    "valid_json": False,
                                }
                            )
                            errors += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for probe in probes:
            f.write(json.dumps(rows_by_probe_id[probe["probe_id"]], sort_keys=True) + "\n")

    status_counts: dict[str, int] = {}
    for row in rows_by_probe_id.values():
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "checkpoint": str(checkpoint_file),
        "probes": str(probes_file),
        "out": str(output_file),
        "variables_config": str(variables_file),
        "vocab": str(vocab_file),
        "input_size": int(input_size),
        "max_text_len": int(max_text_len),
        "normalize_mode": resolved_normalize_mode,
        "device": str(resolved_device),
        "loaded_probes": len(load_probe_jsonl(probes_file)),
        "processed_probes": len(probes),
        "runnable_probes": len(dataset_records),
        "completed": completed,
        "not_applicable": status_counts.get("not_applicable", 0),
        "errors": errors,
        "invalid_forced_predictions": forced_invalid,
        "status_counts": dict(sorted(status_counts.items())),
    }


def main() -> None:
    args = parse_args()
    summary = run_physics_understanding_local(
        checkpoint_path=args.checkpoint,
        probes_path=args.probes,
        out_path=args.out,
        variables_config_path=args.variables_config,
        vocab_path=args.vocab,
        input_size=args.input_size,
        max_text_len=args.max_text_len,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        normalize_mode=args.normalize_mode,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
