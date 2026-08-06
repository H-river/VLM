"""SFT job creation and tracking for profile2setup LLM API fine-tuning."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .validator import validate_llm_output

SUPPORTED_PROVIDERS = ("openai",)
DEFAULT_API_KEY_ENV = {"openai": "OPENAI_API_KEY"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provider_check(provider: str) -> None:
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"provider must be one of {list(SUPPORTED_PROVIDERS)}")


def _object_to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    out: dict[str, Any] = {}
    for key in (
        "id",
        "object",
        "bytes",
        "created_at",
        "filename",
        "purpose",
        "status",
        "model",
        "fine_tuned_model",
        "training_file",
        "validation_file",
        "finished_at",
        "estimated_finish",
        "error",
        "trained_tokens",
    ):
        value = getattr(obj, key, None)
        if value is not None:
            out[key] = value
    return out


def _openai_client(api_key_env: str = "OPENAI_API_KEY"):
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"missing API key environment variable: {api_key_env}")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required for provider=openai API calls") from exc
    return OpenAI(api_key=api_key)


def validate_sft_jsonl(path) -> dict[str, Any]:
    """Validate an SFT JSONL file and its strict assistant JSON labels."""
    file_path = Path(path)
    if file_path.suffix != ".jsonl":
        raise ValueError(f"SFT file must be .jsonl: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"SFT file not found: {file_path}")

    records = 0
    assistant_labels = 0
    image_messages = 0
    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{file_path}:{line_number} record must be a JSON object")
            messages = obj.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(f"{file_path}:{line_number} missing messages list")
            roles = [message.get("role") for message in messages if isinstance(message, dict)]
            if roles[:3] != ["system", "user", "assistant"]:
                raise ValueError(
                    f"{file_path}:{line_number} messages must start with system, user, assistant"
                )
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError(f"{file_path}:{line_number} message must be an object")

            user_content = messages[1].get("content")
            if not isinstance(user_content, list):
                raise ValueError(f"{file_path}:{line_number} user content must be a list")
            has_text = any(isinstance(part, dict) and part.get("type") == "text" for part in user_content)
            has_image = any(isinstance(part, dict) and part.get("type") == "image_url" for part in user_content)
            if not has_text:
                raise ValueError(f"{file_path}:{line_number} user content missing text part")
            if not has_image:
                raise ValueError(f"{file_path}:{line_number} user content missing image_url part")
            image_messages += int(has_image)

            assistant_content = messages[2].get("content")
            if not isinstance(assistant_content, str) or not assistant_content.strip():
                raise ValueError(f"{file_path}:{line_number} assistant content must be a JSON string")
            try:
                assistant_json = json.loads(assistant_content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{file_path}:{line_number} assistant content is not valid JSON: {exc}"
                ) from exc
            validate_llm_output(assistant_json)
            assistant_labels += 1
            records += 1

    if records == 0:
        raise ValueError(f"SFT file has no records: {file_path}")

    return {
        "path": str(file_path),
        "bytes": file_path.stat().st_size,
        "records": records,
        "assistant_labels": assistant_labels,
        "records_with_image_url": image_messages,
    }


def upload_training_file(
    path,
    *,
    provider: str = "openai",
    dry_run: bool = False,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Upload a training JSONL file and return provider file metadata."""
    return _upload_sft_file(
        path,
        provider=provider,
        role="training",
        dry_run=dry_run,
        api_key_env=api_key_env,
    )


def upload_validation_file(
    path,
    *,
    provider: str = "openai",
    dry_run: bool = False,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Upload a validation JSONL file and return provider file metadata."""
    return _upload_sft_file(
        path,
        provider=provider,
        role="validation",
        dry_run=dry_run,
        api_key_env=api_key_env,
    )


def _upload_sft_file(
    path,
    *,
    provider: str,
    role: str,
    dry_run: bool,
    api_key_env: str | None,
) -> dict[str, Any]:
    _provider_check(provider)
    validation = validate_sft_jsonl(path)
    request = {
        "provider": provider,
        "operation": "files.create",
        "path": validation["path"],
        "purpose": "fine-tune",
        "role": role,
    }
    if dry_run:
        return {
            "dry_run": True,
            "id": f"dry-run-{role}-file",
            "request": request,
            "validation": validation,
        }

    if provider == "openai":
        env_name = api_key_env or DEFAULT_API_KEY_ENV[provider]
        client = _openai_client(env_name)
        with Path(path).open("rb") as f:
            file_obj = client.files.create(file=f, purpose="fine-tune")
        data = _object_to_dict(file_obj)
        data["validation"] = validation
        return data

    raise ValueError(f"unsupported provider: {provider}")


def create_sft_job(
    provider: str,
    base_model: str,
    training_file_id: str,
    validation_file_id: str | None = None,
    suffix: str | None = None,
    *,
    dry_run: bool = False,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Create a provider SFT job and return job metadata."""
    _provider_check(provider)
    if not base_model:
        raise ValueError("base_model is required")
    if not training_file_id:
        raise ValueError("training_file_id is required")

    request: dict[str, Any] = {
        "model": base_model,
        "training_file": training_file_id,
    }
    if validation_file_id:
        request["validation_file"] = validation_file_id
    if suffix:
        request["suffix"] = suffix

    if dry_run:
        return {
            "dry_run": True,
            "id": "dry-run-sft-job",
            "provider": provider,
            "request": {
                "operation": "fine_tuning.jobs.create",
                **request,
            },
            "status": "dry_run",
            "fine_tuned_model": None,
        }

    if provider == "openai":
        env_name = api_key_env or DEFAULT_API_KEY_ENV[provider]
        client = _openai_client(env_name)
        job = client.fine_tuning.jobs.create(**request)
        data = _object_to_dict(job)
        data["provider"] = provider
        return data

    raise ValueError(f"unsupported provider: {provider}")


def retrieve_sft_job(
    job_id: str,
    *,
    provider: str = "openai",
    dry_run: bool = False,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Retrieve a provider SFT job by ID."""
    _provider_check(provider)
    if not job_id:
        raise ValueError("job_id is required")
    if dry_run:
        return {
            "dry_run": True,
            "id": job_id,
            "provider": provider,
            "request": {
                "operation": "fine_tuning.jobs.retrieve",
                "job_id": job_id,
            },
            "status": "dry_run",
            "fine_tuned_model": None,
        }

    if provider == "openai":
        env_name = api_key_env or DEFAULT_API_KEY_ENV[provider]
        client = _openai_client(env_name)
        job = client.fine_tuning.jobs.retrieve(job_id)
        data = _object_to_dict(job)
        data["provider"] = provider
        return data

    raise ValueError(f"unsupported provider: {provider}")


def save_job_metadata(path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write SFT job metadata JSON and return the saved object."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = dict(metadata or {})
    saved["metadata_path"] = str(out_path)
    saved["saved_at"] = _utc_now()
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2, sort_keys=True)
    return saved


def load_job_metadata(path) -> dict[str, Any]:
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"job metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"job metadata must be a JSON object: {metadata_path}")
    return obj


def update_job_metadata(
    path,
    *,
    provider: str | None = None,
    dry_run: bool = False,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Retrieve latest job state and update a metadata JSON file."""
    metadata = load_job_metadata(path)
    provider_name = provider or metadata.get("provider") or "openai"
    job_id = metadata.get("job_id") or metadata.get("id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("job metadata missing job_id")

    latest = retrieve_sft_job(
        job_id,
        provider=provider_name,
        dry_run=dry_run,
        api_key_env=api_key_env,
    )
    updated = dict(metadata)
    updated["provider"] = provider_name
    if "created_job" not in updated and "job" in updated:
        updated["created_job"] = updated["job"]
    updated["job"] = latest
    updated["latest_job"] = latest
    updated["job_id"] = latest.get("id", job_id)
    updated["status"] = latest.get("status", updated.get("status"))
    updated["fine_tuned_model"] = latest.get("fine_tuned_model", updated.get("fine_tuned_model"))
    updated["updated_at"] = _utc_now()
    return save_job_metadata(path, updated)
