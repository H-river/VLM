"""Build multimodal LLM API SFT JSONL from profile2setup records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from profile2setup.llm_api.image_rendering import (
    load_intensity_npy,
    render_composite_png,
    render_difference_png,
    render_profile_png,
)
from profile2setup.llm_api.sft_records import build_sft_record, write_jsonl


def _safe_id(value: Any) -> str:
    text = str(value or "record")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return safe or "record"


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
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


def _resolve_path(path_value: Any, repo_root: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _profile_paths(record: dict, repo_root: Path) -> tuple[Path | None, Path | None]:
    return (
        _resolve_path(record.get("current_profile_path"), repo_root),
        _resolve_path(record.get("target_profile_path"), repo_root),
    )


def _required_profile_fields(record: dict) -> tuple[str, ...]:
    task_type = record.get("task_type")
    if task_type == "absolute":
        return ("target_profile_path",)
    return ("current_profile_path", "target_profile_path")


def _missing_required_profiles(record: dict) -> list[str]:
    return [
        field
        for field in _required_profile_fields(record)
        if not isinstance(record.get(field), str) or not record.get(field)
    ]


def _load_optional_profile(path: Path | None) -> Any:
    if path is None:
        return None
    return load_intensity_npy(path)


def _render_images_for_record(
    record: dict,
    *,
    current_profile: Path | None,
    target_profile: Path | None,
    image_out_dir: Path,
    index: int,
    include_composite: bool,
    size: int,
    normalize_mode: str,
) -> dict[str, str]:
    current = _load_optional_profile(current_profile)
    target = _load_optional_profile(target_profile)
    if current is None and target is None:
        raise ValueError("record must provide at least one intensity.npy profile")
    if current is not None and target is not None and current.shape != target.shape:
        raise ValueError(f"current and target shapes must match, got {current.shape} and {target.shape}")

    record_dir = image_out_dir / f"{index:06d}_{_safe_id(record.get('id'))}"
    image_paths: dict[str, str] = {}
    if current is not None:
        image_paths["current_profile"] = render_profile_png(
            current,
            record_dir / "current_profile.png",
            label="CURRENT",
            size=size,
            mode=normalize_mode,
        )
    if target is not None:
        image_paths["target_profile"] = render_profile_png(
            target,
            record_dir / "target_profile.png",
            label="TARGET",
            size=size,
            mode=normalize_mode,
        )
    if current is not None and target is not None:
        image_paths["difference_profile"] = render_difference_png(
            current,
            target,
            record_dir / "difference_profile.png",
            size=size,
            mode=normalize_mode,
        )
        if include_composite:
            image_paths["composite_profile"] = render_composite_png(
                current,
                target,
                record_dir / "composite_profile.png",
                size=size,
                mode=normalize_mode,
            )
    return image_paths


def build_llm_api_sft_dataset(
    input_path,
    out_path,
    image_out_dir,
    *,
    limit: int | None = None,
    split: str | None = None,
    include_composite: bool = False,
    image_mode: str = "base64",
    image_detail: str = "low",
    strict: bool = False,
    repo_root=None,
    size: int = 512,
    normalize_mode: str = "max",
    public_prefix: str | None = None,
) -> dict[str, Any]:
    """Render images and write API-compatible multimodal SFT records."""
    input_file = Path(input_path)
    output_file = Path(out_path)
    image_dir = Path(image_out_dir)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    image_dir.mkdir(parents=True, exist_ok=True)

    read_records = 0
    written_records = 0
    skipped_missing_profiles = 0
    skipped_errors = 0
    errors: list[dict[str, Any]] = []
    sft_records: list[dict] = []

    for line_number, record in _load_jsonl(input_file):
        if limit is not None and written_records >= int(limit):
            break
        read_records += 1

        current_profile, target_profile = _profile_paths(record, root)
        missing_profiles = _missing_required_profiles(record)
        if missing_profiles:
            skipped_missing_profiles += 1
            if strict:
                raise ValueError(
                    f"{input_file}:{line_number} missing required profile path(s) "
                    f"for task_type={record.get('task_type')}: {missing_profiles}"
                )
            continue

        try:
            image_paths = _render_images_for_record(
                record,
                current_profile=current_profile,
                target_profile=target_profile,
                image_out_dir=image_dir,
                index=written_records,
                include_composite=include_composite,
                size=size,
                normalize_mode=normalize_mode,
            )
            sft_record = build_sft_record(
                record,
                image_paths,
                image_mode=image_mode,
                image_detail=image_detail,
                image_base_dir=output_file.parent,
                public_prefix=public_prefix,
            )
            sft_records.append(sft_record)
            written_records += 1
        except Exception as exc:  # noqa: BLE001 - report bad records as data unless strict.
            skipped_errors += 1
            errors.append(
                {
                    "line_number": line_number,
                    "record_id": record.get("id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if strict:
                raise

    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows_written = write_jsonl(sft_records, output_file)
    summary = {
        "input_path": str(input_file),
        "out_path": str(output_file),
        "image_out_dir": str(image_dir),
        "split": split,
        "image_mode": image_mode,
        "image_detail": image_detail,
        "include_composite": include_composite,
        "read_records": read_records,
        "written_records": rows_written,
        "skipped_missing_profiles": skipped_missing_profiles,
        "skipped_errors": skipped_errors,
        "errors": errors[:10],
    }
    return summary
