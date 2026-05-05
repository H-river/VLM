"""Build VLM SFT JSONL data for the profile2setup reasoning layer."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from profile2setup.reasoning_vlm.image_rendering import render_reasoning_images
from profile2setup.reasoning_vlm.sft_dataset import build_sft_record

NEGATIVE_PROMPTS = (
    "move the beam left and right",
    "make the beam wider and narrower",
    "keep the camera fixed but only move the camera",
    "change the wavelength",
    "change the beam color",
    "increase laser power",
)


def _safe_id(value: Any) -> str:
    text = str(value or "record")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return safe or "record"


def _load_jsonl(path: Path):
    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} record must be a JSON object")
            yield line_number, record


def _profile_paths(record: dict) -> tuple[str | None, str | None]:
    current = record.get("current_profile_path")
    target = record.get("target_profile_path")
    return (
        current if isinstance(current, str) and current else None,
        target if isinstance(target, str) and target else None,
    )


def _render_for_record(record: dict, image_out_dir: Path, index: int) -> dict[str, str]:
    current_path, target_path = _profile_paths(record)
    if current_path is None or target_path is None:
        raise ValueError("record requires both current_profile_path and target_profile_path")
    record_dir = image_out_dir / f"{index:06d}_{_safe_id(record.get('id'))}"
    return render_reasoning_images(current_path, target_path, record_dir)


def _negative_record(base_record: dict, prompt: str, index: int) -> dict:
    record = deepcopy(base_record)
    record["id"] = f"negative_{index:03d}_{_safe_id(prompt)}"
    record["prompt"] = prompt
    return record


def build_vlm_sft_dataset(
    input_path,
    out_path,
    image_out_dir,
    limit: int | None = None,
    include_negative_examples: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Read profile2setup JSONL records and write multimodal reasoning SFT JSONL."""
    input_file = Path(input_path)
    output_file = Path(out_path)
    image_dir = Path(image_out_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    read_records = 0
    written_records = 0
    skipped_missing_profiles = 0
    skipped_errors = 0
    first_usable_record: dict | None = None

    with open(output_file, "w") as out_f:
        for line_number, record in _load_jsonl(input_file):
            read_records += 1
            if limit is not None and written_records >= int(limit):
                break

            current_path, target_path = _profile_paths(record)
            if current_path is None or target_path is None:
                skipped_missing_profiles += 1
                continue

            try:
                image_paths = _render_for_record(record, image_dir, written_records)
                sft_record = build_sft_record(record, image_paths)
                out_f.write(json.dumps(sft_record, sort_keys=True) + "\n")
                written_records += 1
                if first_usable_record is None:
                    first_usable_record = record
            except Exception:
                if strict:
                    raise
                skipped_errors += 1
                continue

        negative_written = 0
        if include_negative_examples and first_usable_record is not None:
            for negative_index, prompt in enumerate(NEGATIVE_PROMPTS):
                record = _negative_record(first_usable_record, prompt, negative_index)
                try:
                    image_paths = _render_for_record(
                        record,
                        image_dir,
                        written_records + negative_written,
                    )
                    sft_record = build_sft_record(record, image_paths)
                    out_f.write(json.dumps(sft_record, sort_keys=True) + "\n")
                    negative_written += 1
                except Exception:
                    if strict:
                        raise
                    skipped_errors += 1
                    continue

    summary = {
        "input_path": str(input_file),
        "out_path": str(output_file),
        "image_out_dir": str(image_dir),
        "read_records": read_records,
        "written_records": written_records,
        "negative_written": negative_written if include_negative_examples else 0,
        "skipped_missing_profiles": skipped_missing_profiles,
        "skipped_errors": skipped_errors,
    }
    return summary
