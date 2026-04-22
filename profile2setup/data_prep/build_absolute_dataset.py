"""Build absolute-task JSONL dataset from simulator outputs."""

from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from ..schema import validate_dataset_record
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


def build_absolute_dataset(
    sim_dir,
    out_path,
    prompts_path="profile2setup/configs/prompts.yaml",
    strict=True,
    limit=None,
    seed=42,
):
    """Build absolute dataset where target profile + prompt maps to target_setup."""
    rng = random.Random(seed)
    prompts = _load_prompts(prompts_path)
    absolute_prompts = prompts.get("absolute") or []
    if not absolute_prompts:
        raise ValueError("No absolute prompts found in prompts config")

    sample_dirs = find_sample_dirs(sim_dir)
    if limit is not None:
        sample_dirs = sample_dirs[: int(limit)]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with open(out_path, "w") as f:
        for sample_dir in sample_dirs:
            try:
                sample = load_sample(sample_dir, strict=strict)
                if sample is None:
                    skipped += 1
                    if not strict:
                        _warn(f"Skipping invalid sample (load_sample returned None): {sample_dir}")
                    continue

                profile_features = compute_profile_features_from_path(sample["profile_path"])
                target_metrics = _merge_metrics(sample.get("metrics", {}), profile_features)

                prompt = rng.choice(absolute_prompts)

                # target_setup is the supervised setup label.
                # target_profile_path is the input profile for absolute tasks.
                # profile_loss_reference stores the desired profile for later
                # simulation/profile-loss or closed-loop comparison; in the
                # current implementation this is for later evaluation/closed-loop
                # use unless a differentiable simulator or surrogate model is added.
                record = {
                    "id": f"abs_{sample['sample_id']}",
                    "task_type": "absolute",
                    "prompt": prompt,
                    "current_profile_path": None,
                    "target_profile_path": sample["profile_path"],
                    "current_setup": None,
                    "target_setup": sample["setup"],
                    "target_delta": None,
                    "current_metrics": None,
                    "target_metrics": target_metrics,
                    "profile_loss_reference": {
                        "target_profile_path": sample["profile_path"],
                        "target_metadata_path": sample["metadata_path"],
                    },
                    "source_run": sample["sample_id"],
                    "target_metadata_path": sample["metadata_path"],
                }

                if not validate_dataset_record(record, strict=strict):
                    skipped += 1
                    if not strict:
                        _warn(f"Skipping invalid absolute record for sample: {sample_dir}")
                    continue

                f.write(json.dumps(record) + "\n")
                written += 1

            except Exception as exc:
                if strict:
                    raise
                skipped += 1
                _warn(f"Skipping sample {sample_dir}: {exc}")

    print(
        "absolute dataset summary: "
        f"found samples={len(sample_dirs)}, written records={written}, skipped records={skipped}"
    )

    return {
        "found_samples": len(sample_dirs),
        "written_records": written,
        "skipped_records": skipped,
        "out_path": str(out_path),
    }
