"""Provider client abstraction for profile2setup multimodal LLM API calls."""

from __future__ import annotations

import os
from typing import Any

from .schema import default_output_schema

SUPPORTED_PROVIDERS = ("openai",)


def build_openai_request_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    use_response_format: bool = True,
) -> dict[str, Any]:
    """Build an OpenAI Responses API payload."""
    payload: dict[str, Any] = {
        "model": model,
        "input": messages,
        "temperature": float(temperature),
    }
    if max_output_tokens is not None:
        payload["max_output_tokens"] = int(max_output_tokens)
    if use_response_format:
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": "profile2setup_llm_output",
                "strict": True,
                "schema": default_output_schema(),
            }
        }
    return payload


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    if isinstance(response, dict):
        value = response.get("output_text")
        if isinstance(value, str):
            return value
        output = response.get("output")
    else:
        output = getattr(response, "output", None)

    parts: list[str] = []
    if isinstance(output, list):
        for item in output:
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                else:
                    text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts)


def _response_metadata(response: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in ("id", "model", "created_at", "status"):
        value = getattr(response, name, None)
        if value is not None:
            metadata[name] = value
    usage = getattr(response, "usage", None)
    if usage is not None:
        if hasattr(usage, "model_dump"):
            metadata["usage"] = usage.model_dump()
        elif isinstance(usage, dict):
            metadata["usage"] = usage
        else:
            metadata["usage"] = str(usage)
    return metadata


def call_multimodal_model(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    dry_run: bool = False,
    api_key_env: str = "OPENAI_API_KEY",
    use_response_format: bool = True,
) -> dict[str, Any]:
    """Call a multimodal provider model and return raw text plus metadata."""
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"provider must be one of {list(SUPPORTED_PROVIDERS)}")

    payload = build_openai_request_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        use_response_format=use_response_format,
    )
    if dry_run:
        return {
            "provider": provider,
            "model": model,
            "raw_text": "",
            "metadata": {"dry_run": True},
            "request_payload": payload,
        }

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"missing API key environment variable: {api_key_env}")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required for provider=openai API calls") from exc

    client = OpenAI(api_key=api_key)
    response = client.responses.create(**payload)
    return {
        "provider": provider,
        "model": model,
        "raw_text": _extract_response_text(response),
        "metadata": _response_metadata(response),
    }
