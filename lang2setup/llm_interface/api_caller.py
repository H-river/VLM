"""
api_caller.py
Thin wrapper around LLM API calls (OpenAI-compatible).
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional

import yaml
from pathlib import Path

_DEFAULT_LLM_CFG = Path(__file__).parent.parent / "configs" / "llm.yaml"


def load_llm_config(path: str | Path = _DEFAULT_LLM_CFG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def call_llm(messages: List[Dict[str, str]],
             config: dict | None = None) -> str:
    """
    Send messages to an LLM and return the raw assistant response text.

    Requires the `openai` package and OPENAI_API_KEY env var set.
    Extend for Anthropic / local models as needed.
    """
    if config is None:
        config = load_llm_config()
    cfg = config["llm"]

    if cfg["provider"] == "openai":
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=messages,
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        return response.choices[0].message.content

    elif cfg["provider"] == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        # Extract system message
        system = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_msgs.append(m)
        response = client.messages.create(
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
            system=system,
            messages=user_msgs,
        )
        return response.content[0].text

    else:
        raise ValueError(f"Unsupported LLM provider: {cfg['provider']}")
