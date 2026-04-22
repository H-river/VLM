"""Build edit-task JSONL dataset from simulator outputs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml

from ..schema import compute_delta_setup, validate_dataset_record
from .extract_profile_features import compute_profile_features_from_path
from .extract_setup import find_sample_dirs, load_sample


def _warn(msg: str) -> None:
    print(f"[warn] {msg}")


def _load_prompts(prompts_path: str | Path) -> dict:
    with open(prompts_path, "r") as f:
        prompts = yaml.safe_load(f)
    if not isinstance(prompts, dict):
        raise ValueError(f"Invalid prompts file: {prompts_path}")
    return prompts


def _merge_metrics(metadata_metrics: dict, profile_features: dict) -> dict:
    out = dict(metadata_metrics or {})
    out.update(profile_features or {})
    return out


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def compute_metrics_delta(current_metrics: dict, target_metrics: dict) -> dict:
    """Compute numeric metric deltas as target - current for common keys."""
    delta = {}
    for key in current_metrics.keys() & target_metrics.keys():
        cval = _numeric(current_metrics.get(key))
        tval = _numeric(target_metrics.get(key))
        if cval is None or tval is None:
            continue
        delta[key] = float(tval - cval)
    return delta


def _pick_metric(metrics: dict, *keys: str) -> float | None:
    for key in keys:
        value = _numeric(metrics.get(key))
        if value is not None:
            return value
    return None


def _choose_phrase(prompts: dict, key: str, rng: random.Random, fallback: str) -> str:
    metric_edit = prompts.get("metric_edit", {}) or {}
    options = metric_edit.get(key) or [fallback]
    return rng.choice(options)


def choose_edit_prompt(current_metrics, target_metrics, metrics_delta, prompts, rng) -> str:
    """Choose edit prompt using metric deltas when available."""
    del metrics_delta  # currently not required for branching

    generic_prompts = prompts.get("edit") or [
        "Change the current beam to match the target profile."
    ]

    phrases: list[str] = []

    cur_x = _pick_metric(current_metrics, "centroid_x")
    tgt_x = _pick_metric(target_metrics, "centroid_x")
    if cur_x is not None and tgt_x is not None:
        dx = tgt_x - cur_x
        if dx > 2e-4:
            phrases.append(_choose_phrase(prompts, "move_right", rng, "move the beam right"))
        elif dx < -2e-4:
            phrases.append(_choose_phrase(prompts, "move_left", rng, "move the beam left"))
    else:
        cur_x_px = _pick_metric(current_metrics, "centroid_x_px")
        tgt_x_px = _pick_metric(target_metrics, "centroid_x_px")
        if cur_x_px is not None and tgt_x_px is not None:
            dx_px = tgt_x_px - cur_x_px
            if dx_px > 0.5:
                phrases.append(_choose_phrase(prompts, "move_right", rng, "move the beam right"))
            elif dx_px < -0.5:
                phrases.append(_choose_phrase(prompts, "move_left", rng, "move the beam left"))

    cur_y = _pick_metric(current_metrics, "centroid_y")
    tgt_y = _pick_metric(target_metrics, "centroid_y")
    if cur_y is not None and tgt_y is not None:
        # Physical centroid_y convention: larger y means beam is higher.
        dy = tgt_y - cur_y
        if dy > 2e-4:
            phrases.append(_choose_phrase(prompts, "move_up", rng, "move the beam up"))
        elif dy < -2e-4:
            phrases.append(_choose_phrase(prompts, "move_down", rng, "move the beam down"))
    else:
        cur_y_px = _pick_metric(current_metrics, "centroid_y_px")
        tgt_y_px = _pick_metric(target_metrics, "centroid_y_px")
        if cur_y_px is not None and tgt_y_px is not None:
            # Pixel-space centroid_y_px follows array row index; larger is lower.
            dy_px = tgt_y_px - cur_y_px
            if dy_px > 0.5:
                phrases.append(_choose_phrase(prompts, "move_down", rng, "move the beam down"))
            elif dy_px < -0.5:
                phrases.append(_choose_phrase(prompts, "move_up", rng, "move the beam up"))

    cur_sx = _pick_metric(current_metrics, "sigma_x")
    cur_sy = _pick_metric(current_metrics, "sigma_y")
    tgt_sx = _pick_metric(target_metrics, "sigma_x")
    tgt_sy = _pick_metric(target_metrics, "sigma_y")

    if None not in (cur_sx, cur_sy, tgt_sx, tgt_sy):
        cur_size = (cur_sx + cur_sy) / 2.0
        tgt_size = (tgt_sx + tgt_sy) / 2.0
        if tgt_size < cur_size - 1e-4:
            phrases.append(_choose_phrase(prompts, "smaller", rng, "make the beam smaller"))
        elif tgt_size > cur_size + 1e-4:
            phrases.append(_choose_phrase(prompts, "larger", rng, "make the beam larger"))
    else:
        cur_sx_px = _pick_metric(current_metrics, "sigma_x_px")
        cur_sy_px = _pick_metric(current_metrics, "sigma_y_px")
        tgt_sx_px = _pick_metric(target_metrics, "sigma_x_px")
        tgt_sy_px = _pick_metric(target_metrics, "sigma_y_px")
        if None not in (cur_sx_px, cur_sy_px, tgt_sx_px, tgt_sy_px):
            cur_size_px = (cur_sx_px + cur_sy_px) / 2.0
            tgt_size_px = (tgt_sx_px + tgt_sy_px) / 2.0
            if tgt_size_px < cur_size_px - 0.3:
                phrases.append(_choose_phrase(prompts, "smaller", rng, "make the beam smaller"))
            elif tgt_size_px > cur_size_px + 0.3:
                phrases.append(_choose_phrase(prompts, "larger", rng, "make the beam larger"))

    if len(phrases) >= 2:
        return f"{phrases[0]} and {phrases[1]}"
    if len(phrases) == 1:
        return phrases[0]

    return rng.choice(generic_prompts)


def build_edit_dataset(
    sim_dir,
    out_path,
    num_pairs=50000,
    prompts_path="profile2setup/configs/prompts.yaml",
    strict=True,
    seed=42,
    limit_samples=None,
    allow_self_pairs=False,
):
    """Build edit dataset where current+target profile maps to target delta/setup."""
    rng = random.Random(seed)
    prompts = _load_prompts(prompts_path)

    sample_dirs = find_sample_dirs(sim_dir)
    if limit_samples is not None:
        sample_dirs = sample_dirs[: int(limit_samples)]

    valid_samples = []
    skipped_samples = 0

    for sample_dir in sample_dirs:
        try:
            sample = load_sample(sample_dir, strict=strict)
            if sample is None:
                skipped_samples += 1
                if not strict:
                    _warn(f"Skipping invalid sample (load_sample returned None): {sample_dir}")
                continue
            profile_features = compute_profile_features_from_path(sample["profile_path"])
            sample["metrics"] = _merge_metrics(sample.get("metrics", {}), profile_features)
            valid_samples.append(sample)
        except Exception as exc:
            if strict:
                raise
            skipped_samples += 1
            _warn(f"Skipping sample {sample_dir}: {exc}")

    if not valid_samples:
        raise ValueError("No valid samples available for edit dataset")
    if not allow_self_pairs and len(valid_samples) < 2:
        raise ValueError("Need at least 2 valid samples when allow_self_pairs=False")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    requested_pairs = int(num_pairs)
    written_pairs = 0
    skipped_pairs = 0
    attempts = 0
    max_attempts = max(requested_pairs * 20, 1000)

    with open(out_path, "w") as f:
        while written_pairs < requested_pairs and attempts < max_attempts:
            attempts += 1
            current = rng.choice(valid_samples)
            target = rng.choice(valid_samples)

            if not allow_self_pairs and current["sample_id"] == target["sample_id"]:
                skipped_pairs += 1
                continue

            try:
                target_delta = compute_delta_setup(current["setup"], target["setup"])
                metrics_delta = compute_metrics_delta(current["metrics"], target["metrics"])
                prompt = choose_edit_prompt(
                    current["metrics"],
                    target["metrics"],
                    metrics_delta,
                    prompts,
                    rng,
                )

                # target_setup/target_delta are supervised labels.
                # profile_loss_reference stores current/target profiles for later
                # simulation/profile-loss or closed-loop comparison; in the
                # current implementation this is for later evaluation/closed-loop
                # use unless a differentiable simulator or surrogate model is added.
                record = {
                    "id": f"edit_{current['sample_id']}__{target['sample_id']}__{written_pairs}",
                    "task_type": "edit",
                    "prompt": prompt,
                    "current_profile_path": current["profile_path"],
                    "target_profile_path": target["profile_path"],
                    "current_setup": current["setup"],
                    "target_setup": target["setup"],
                    "target_delta": target_delta,
                    "current_metrics": current["metrics"],
                    "target_metrics": target["metrics"],
                    "metrics_delta": metrics_delta,
                    "profile_loss_reference": {
                        "current_profile_path": current["profile_path"],
                        "target_profile_path": target["profile_path"],
                        "current_metadata_path": current["metadata_path"],
                        "target_metadata_path": target["metadata_path"],
                    },
                    "source_run_current": current["sample_id"],
                    "source_run_target": target["sample_id"],
                    "current_metadata_path": current["metadata_path"],
                    "target_metadata_path": target["metadata_path"],
                }

                if not validate_dataset_record(record, strict=strict):
                    skipped_pairs += 1
                    if not strict:
                        _warn(
                            "Skipping invalid edit record for pair "
                            f"{current['sample_id']} -> {target['sample_id']}"
                        )
                    continue
                if set(record["target_delta"].keys()) != set(record["target_setup"].keys()):
                    raise ValueError("target_delta keys do not match canonical setup keys")

                f.write(json.dumps(record) + "\n")
                written_pairs += 1

            except Exception as exc:
                if strict:
                    raise
                skipped_pairs += 1
                _warn(
                    f"Skipping pair {current['sample_id']} -> {target['sample_id']}: {exc}"
                )

    if strict and written_pairs < requested_pairs:
        raise RuntimeError(
            f"Could not generate requested pairs: requested={requested_pairs}, "
            f"written={written_pairs}, attempts={attempts}"
        )

    print(
        "edit dataset summary: "
        f"found samples={len(sample_dirs)}, "
        f"valid samples={len(valid_samples)}, "
        f"requested pairs={requested_pairs}, "
        f"written pairs={written_pairs}, "
        f"skipped samples={skipped_samples}, "
        f"skipped pairs={skipped_pairs}"
    )

    return {
        "found_samples": len(sample_dirs),
        "valid_samples": len(valid_samples),
        "requested_pairs": requested_pairs,
        "written_pairs": written_pairs,
        "skipped_samples": skipped_samples,
        "skipped_pairs": skipped_pairs,
        "out_path": str(out_path),
    }
