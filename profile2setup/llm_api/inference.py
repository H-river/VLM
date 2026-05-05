"""Run multimodal LLM API inference over profile2setup JSONL records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .client import call_multimodal_model
from .image_rendering import (
    load_intensity_npy,
    render_composite_png,
    render_difference_png,
    render_profile_png,
)
from .prompts import build_messages
from .validator import parse_llm_json_text


def _safe_id(value: Any) -> str:
    text = str(value or "record")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return safe or "record"


def load_jsonl_records(path):
    """Yield ``(line_number, record)`` pairs from a JSONL file."""
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{jsonl_path}:{line_number} record must be a JSON object")
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


def _load_optional_profile(path: Path | None):
    if path is None:
        return None
    return load_intensity_npy(path)


def render_or_reuse_images(
    record: dict,
    *,
    current_profile: Path | None,
    target_profile: Path | None,
    image_out_dir: Path,
    index: int,
    reuse_rendered_images: bool = False,
    image_size: int = 512,
    normalize_mode: str = "max",
    include_composite: bool = True,
) -> dict[str, str]:
    """Render or reuse available profile PNGs for one record."""
    record_dir = image_out_dir / f"{index:06d}_{_safe_id(record.get('id'))}"
    expected_paths: dict[str, Path] = {}
    if current_profile is not None:
        expected_paths["current_profile"] = record_dir / "current_profile.png"
    if target_profile is not None:
        expected_paths["target_profile"] = record_dir / "target_profile.png"
    if current_profile is not None and target_profile is not None:
        expected_paths["difference_profile"] = record_dir / "difference_profile.png"
        if include_composite:
            expected_paths["composite_profile"] = record_dir / "composite_profile.png"
    if reuse_rendered_images and expected_paths and all(path.exists() for path in expected_paths.values()):
        return {key: str(path) for key, path in expected_paths.items()}

    current = _load_optional_profile(current_profile)
    target = _load_optional_profile(target_profile)
    if current is None and target is None:
        raise ValueError("record must provide at least one intensity.npy profile")
    if current is not None and target is not None and current.shape != target.shape:
        raise ValueError(f"current and target shapes must match, got {current.shape} and {target.shape}")

    rendered: dict[str, str] = {}
    if current is not None:
        rendered["current_profile"] = render_profile_png(
            current,
            record_dir / "current_profile.png",
            label="CURRENT",
            size=image_size,
            mode=normalize_mode,
        )
    if target is not None:
        rendered["target_profile"] = render_profile_png(
            target,
            record_dir / "target_profile.png",
            label="TARGET",
            size=image_size,
            mode=normalize_mode,
        )
    if current is not None and target is not None:
        rendered["difference_profile"] = render_difference_png(
            current,
            target,
            record_dir / "difference_profile.png",
            size=image_size,
            mode=normalize_mode,
        )
        if include_composite:
            rendered["composite_profile"] = render_composite_png(
                current,
                target,
                record_dir / "composite_profile.png",
                size=image_size,
                mode=normalize_mode,
            )
    return rendered


def _base_prediction_record(
    *,
    line_number: int,
    record: dict,
    image_paths: dict | None = None,
) -> dict[str, Any]:
    out = {
        "line_number": line_number,
        "record_id": record.get("id"),
        "task_type": record.get("task_type"),
    }
    if image_paths is not None:
        out["image_paths"] = image_paths
    return out


def _parse_prediction(raw_text: str) -> tuple[dict | None, str | None]:
    try:
        return parse_llm_json_text(raw_text), None
    except Exception as exc:  # noqa: BLE001 - parser errors are prediction data.
        return None, f"{type(exc).__name__}: {exc}"


def run_llm_api_inference(
    *,
    data_path,
    out_path,
    image_out_dir,
    provider: str = "openai",
    model: str,
    limit: int | None = None,
    dry_run: bool = False,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    image_detail: str = "low",
    reuse_rendered_images: bool = False,
    repo_root=None,
    image_size: int = 512,
    normalize_mode: str = "max",
    use_response_format: bool = True,
) -> dict[str, Any]:
    """Run inference and write one prediction/status JSON object per line."""
    data_file = Path(data_path)
    output_file = Path(out_path)
    image_dir = Path(image_out_dir)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    read_records = 0
    attempted_records = 0
    written_records = 0
    skipped_missing_profiles = 0
    valid_json = 0
    invalid_json = 0
    api_errors = 0

    with output_file.open("w", encoding="utf-8") as out_f:
        for line_number, record in load_jsonl_records(data_file):
            if limit is not None and attempted_records >= int(limit):
                break
            read_records += 1

            current_profile, target_profile = _profile_paths(record, root)
            missing_profiles = _missing_required_profiles(record)
            if missing_profiles:
                skipped_missing_profiles += 1
                result = _base_prediction_record(line_number=line_number, record=record)
                result.update(
                    {
                        "status": "skipped",
                        "skip_reason": (
                            "missing required profile path(s) "
                            f"for task_type={record.get('task_type')}: {missing_profiles}"
                        ),
                        "valid_json": False,
                    }
                )
                out_f.write(json.dumps(result, sort_keys=True) + "\n")
                written_records += 1
                continue

            result = _base_prediction_record(line_number=line_number, record=record)
            try:
                image_paths = render_or_reuse_images(
                    record,
                    current_profile=current_profile,
                    target_profile=target_profile,
                    image_out_dir=image_dir,
                    index=attempted_records,
                    reuse_rendered_images=reuse_rendered_images,
                    image_size=image_size,
                    normalize_mode=normalize_mode,
                )
                messages = build_messages(record, image_paths, image_detail=image_detail)
                response = call_multimodal_model(
                    provider=provider,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    dry_run=dry_run,
                    use_response_format=use_response_format,
                )
                attempted_records += 1
                result.update(
                    {
                        "status": "dry_run" if dry_run else "completed",
                        "provider": provider,
                        "model": model,
                        "image_paths": image_paths,
                        "metadata": response.get("metadata", {}),
                    }
                )
                if dry_run:
                    result["request_payload"] = response.get("request_payload")
                    result["valid_json"] = None
                else:
                    raw_text = str(response.get("raw_text") or "")
                    parsed, error = _parse_prediction(raw_text)
                    result["raw_response"] = raw_text
                    if parsed is None:
                        result["valid_json"] = False
                        result["validation_error"] = error
                        invalid_json += 1
                    else:
                        result["valid_json"] = True
                        result["prediction"] = parsed
                        valid_json += 1
            except Exception as exc:  # noqa: BLE001 - API failures should be logged per record.
                attempted_records += 1
                api_errors += 1
                result.update(
                    {
                        "status": "error",
                        "provider": provider,
                        "model": model,
                        "valid_json": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

            out_f.write(json.dumps(result, sort_keys=True) + "\n")
            written_records += 1

    return {
        "data_path": str(data_file),
        "out_path": str(output_file),
        "image_out_dir": str(image_dir),
        "provider": provider,
        "model": model,
        "dry_run": dry_run,
        "read_records": read_records,
        "attempted_records": attempted_records,
        "written_records": written_records,
        "skipped_missing_profiles": skipped_missing_profiles,
        "valid_json": valid_json,
        "invalid_json": invalid_json,
        "api_errors": api_errors,
    }
