"""Run LLM/API inference for physics-understanding diagnostic probes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from legacy.physics_understanding.evaluation.physics_understanding_schema import (
    CANONICAL_VARIABLE_ORDER,
    INPUT_MODES,
    load_probe_jsonl,
)
from profile2setup.llm_api.client import call_multimodal_model
from profile2setup.llm_api.inference import render_or_reuse_images
from profile2setup.llm_api.sft_records import image_file_to_data_url
from profile2setup.llm_api.validator import parse_llm_json_text

IMAGE_INPUT_MODES = {"images_only", "prompt_plus_images", "shuffled_prompt", "conflict"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multimodal LLM inference over physics-understanding probes."
    )
    parser.add_argument("--provider", default="openai", choices=("openai",))
    parser.add_argument("--model", required=True, help="Provider model name")
    parser.add_argument("--probes", required=True, help="Input physics-understanding probe JSONL")
    parser.add_argument("--out", required=True, help="Output prediction/status JSONL")
    parser.add_argument("--image-out-dir", required=True, help="Directory for rendered probe images")
    parser.add_argument("--limit", type=int, default=None, help="Maximum probes to attempt")
    parser.add_argument("--dry-run", action="store_true", help="Build request payloads without sending API calls")
    parser.add_argument(
        "--input-mode-override",
        choices=("prompt_only", "images_only", "prompt_plus_images"),
        default=None,
        help="Override probe input mode for ablations.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
    parser.add_argument("--reuse-rendered-images", action="store_true")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--normalize-mode",
        choices=("max", "percentile", "log_max"),
        default="max",
    )
    parser.add_argument(
        "--no-response-format",
        action="store_true",
        help="Disable provider structured-output format and rely on strict prompt instructions only.",
    )
    return parser.parse_args()


def _safe_id(value: Any) -> str:
    text = str(value or "probe")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return safe or "probe"


def _resolve_path(path_value: Any, repo_root: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _json_or_null(value: Any) -> str:
    return json.dumps(value if value is not None else None, sort_keys=True)


def _system_prompt() -> str:
    return (
        "You are an optical setup understanding and physics diagnostic model. "
        "Return only strict JSON using the profile2setup response schema. "
        "Use exactly these canonical variables: "
        + ", ".join(CANONICAL_VARIABLE_ORDER)
        + ". Do not introduce non-canonical setup variable names. "
        "If the prompt, images, or setup context are contradictory or physically inconsistent, "
        "return valid=false and explain the rejection briefly in rejection_reason. "
        "Do not include markdown, code fences, or hidden reasoning."
    )


def _effective_prompt(probe: dict, input_mode: str) -> str:
    if input_mode == "images_only":
        return "Use the provided profiles to infer the setup change."
    return str(probe.get("prompt") or "")


def _input_text(probe: dict, input_mode: str) -> str:
    fixed_variables = probe.get("fixed_variables") or []
    allowed_variables = probe.get("allowed_variables")
    text_lines = [
        "Predict the profile2setup strict JSON response for this diagnostic probe.",
        f"probe_type: {probe.get('probe_type')}",
        f"task_type: {probe.get('task_type')}",
        f"input_mode: {input_mode}",
        f"prompt: {_effective_prompt(probe, input_mode)}",
        "canonical_variables: " + ", ".join(CANONICAL_VARIABLE_ORDER),
        "current_setup: " + _json_or_null(probe.get("current_setup")),
        "fixed_variables: " + _json_or_null(fixed_variables),
        "allowed_variables: " + _json_or_null(allowed_variables),
        (
            "If the prompt conflicts with the image evidence or setup context, return "
            "valid=false and include a rejection_reason containing conflict, contradict, "
            "or inconsistent when appropriate."
        ),
        (
            "Return fields valid, task_type, observed_profile_change, setup_understanding, "
            "predicted_delta, predicted_setup, confidence, reasoning_summary, and rejection_reason."
        ),
        "Do not include markdown, code fences, or chain-of-thought.",
    ]
    if input_mode in {"conflict", "shuffled_prompt"}:
        text_lines.append("This probe may intentionally mismatch prompt and image evidence.")
    if input_mode == "prompt_only":
        text_lines.append("No profile images are provided for this probe.")
    elif input_mode == "images_only":
        text_lines.append("Ignore the original probe wording; use the generic image-only prompt above.")
    else:
        text_lines.append(
            "Images are provided in this order when available: current profile, target profile, "
            "target-current difference, composite."
        )
    return "\n".join(text_lines)


def _image_content(path: str, *, image_detail: str) -> dict[str, Any]:
    return {
        "type": "input_image",
        "image_url": image_file_to_data_url(path),
        "detail": image_detail,
    }


def build_probe_messages(
    probe: dict,
    image_paths: dict[str, str],
    *,
    input_mode: str,
    image_detail: str,
) -> list[dict[str, Any]]:
    """Build Responses-API messages for one diagnostic probe."""
    user_content: list[dict[str, Any]] = [
        {"type": "input_text", "text": _input_text(probe, input_mode)}
    ]
    for key in ("current_profile", "target_profile", "difference_profile", "composite_profile"):
        if key in image_paths:
            user_content.append(_image_content(image_paths[key], image_detail=image_detail))
    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": _system_prompt()}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _probe_render_record(probe: dict) -> dict[str, Any]:
    return {
        "id": probe.get("probe_id"),
        "task_type": probe.get("task_type"),
        "current_profile_path": probe.get("current_profile_path"),
        "target_profile_path": probe.get("target_profile_path"),
        "current_setup": probe.get("current_setup"),
        "prompt": probe.get("prompt"),
    }


def _render_images_for_probe(
    probe: dict,
    *,
    image_out_dir: Path,
    index: int,
    repo_root: Path,
    reuse_rendered_images: bool,
    image_size: int,
    normalize_mode: str,
) -> dict[str, str]:
    current_profile = _resolve_path(probe.get("current_profile_path"), repo_root)
    target_profile = _resolve_path(probe.get("target_profile_path"), repo_root)
    if current_profile is None and target_profile is None:
        raise ValueError("probe has no current_profile_path or target_profile_path")
    return render_or_reuse_images(
        _probe_render_record(probe),
        current_profile=current_profile,
        target_profile=target_profile,
        image_out_dir=image_out_dir,
        index=index,
        reuse_rendered_images=reuse_rendered_images,
        image_size=image_size,
        normalize_mode=normalize_mode,
        include_composite=True,
    )


def _parse_prediction(raw_text: str) -> tuple[dict | None, str | None]:
    try:
        return parse_llm_json_text(raw_text), None
    except Exception as exc:  # noqa: BLE001 - parse failures are prediction data.
        return None, f"{type(exc).__name__}: {exc}"


def _base_output_row(
    *,
    line_number: int,
    probe: dict,
    model: str,
    input_mode: str,
) -> dict[str, Any]:
    return {
        "line_number": line_number,
        "probe_id": probe.get("probe_id"),
        "probe_type": probe.get("probe_type"),
        "input_mode": input_mode,
        "original_input_mode": probe.get("input_mode"),
        "model": model,
        "raw_response": "",
        "prediction": None,
        "status": "pending",
        "error": None,
    }


def run_physics_understanding_llm(
    *,
    model: str,
    probes_path,
    out_path,
    image_out_dir,
    provider: str = "openai",
    limit: int | None = None,
    dry_run: bool = False,
    input_mode_override: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    image_detail: str = "low",
    reuse_rendered_images: bool = False,
    repo_root=None,
    image_size: int = 512,
    normalize_mode: str = "max",
    use_response_format: bool = True,
) -> dict[str, Any]:
    """Run LLM/API inference over validated physics-understanding probes."""
    probes_file = Path(probes_path)
    output_file = Path(out_path)
    image_dir = Path(image_out_dir)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    probes = load_probe_jsonl(probes_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    attempted = 0
    written = 0
    dry_run_rows = 0
    completed = 0
    skipped = 0
    valid_json = 0
    invalid_json = 0
    api_errors = 0

    with output_file.open("w", encoding="utf-8") as out_f:
        for line_number, probe in enumerate(probes, start=1):
            if limit is not None and attempted >= int(limit):
                break
            input_mode = input_mode_override or probe.get("input_mode")
            if input_mode not in INPUT_MODES:
                raise ValueError(f"{probes_file}:{line_number} invalid input_mode: {input_mode}")

            attempted += 1
            row = _base_output_row(
                line_number=line_number,
                probe=probe,
                model=model,
                input_mode=str(input_mode),
            )
            row["provider"] = provider

            try:
                image_paths: dict[str, str] = {}
                if input_mode in IMAGE_INPUT_MODES:
                    image_paths = _render_images_for_probe(
                        probe,
                        image_out_dir=image_dir,
                        index=attempted - 1,
                        repo_root=root,
                        reuse_rendered_images=reuse_rendered_images,
                        image_size=image_size,
                        normalize_mode=normalize_mode,
                    )
                    row["image_paths"] = image_paths

                messages = build_probe_messages(
                    probe,
                    image_paths,
                    input_mode=str(input_mode),
                    image_detail=image_detail,
                )
                response = call_multimodal_model(
                    provider=provider,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    dry_run=dry_run,
                    use_response_format=use_response_format,
                )
                row["metadata"] = response.get("metadata", {})
                if dry_run:
                    row["status"] = "dry_run"
                    row["request_payload"] = response.get("request_payload")
                    row["valid_json"] = None
                    dry_run_rows += 1
                else:
                    raw_text = str(response.get("raw_text") or "")
                    row["raw_response"] = raw_text
                    parsed, parse_error = _parse_prediction(raw_text)
                    if parsed is None:
                        row["status"] = "invalid_json"
                        row["valid_json"] = False
                        row["error"] = parse_error
                        invalid_json += 1
                    else:
                        row["status"] = "completed"
                        row["valid_json"] = True
                        row["prediction"] = parsed
                        valid_json += 1
                        completed += 1
            except Exception as exc:  # noqa: BLE001 - keep per-probe failures in the JSONL.
                message = f"{type(exc).__name__}: {exc}"
                if input_mode in IMAGE_INPUT_MODES and (
                    "no current_profile_path or target_profile_path" in message
                    or "not found" in message.lower()
                ):
                    row["status"] = "skipped"
                    skipped += 1
                else:
                    row["status"] = "error"
                    api_errors += 1
                row["valid_json"] = False
                row["error"] = message

            out_f.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1

    return {
        "probes_path": str(probes_file),
        "out_path": str(output_file),
        "image_out_dir": str(image_dir),
        "provider": provider,
        "model": model,
        "dry_run": bool(dry_run),
        "input_mode_override": input_mode_override,
        "temperature": float(temperature),
        "loaded_probes": len(probes),
        "attempted_probes": attempted,
        "written_rows": written,
        "dry_run_rows": dry_run_rows,
        "completed": completed,
        "skipped": skipped,
        "valid_json": valid_json,
        "invalid_json": invalid_json,
        "api_errors": api_errors,
    }


def main() -> None:
    args = parse_args()
    summary = run_physics_understanding_llm(
        provider=args.provider,
        model=args.model,
        probes_path=Path(args.probes),
        out_path=Path(args.out),
        image_out_dir=Path(args.image_out_dir),
        limit=args.limit,
        dry_run=args.dry_run,
        input_mode_override=args.input_mode_override,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        image_detail=args.image_detail,
        reuse_rendered_images=args.reuse_rendered_images,
        image_size=args.image_size,
        normalize_mode=args.normalize_mode,
        use_response_format=not args.no_response_format,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
