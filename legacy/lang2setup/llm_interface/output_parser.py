"""
output_parser.py
End-to-end: query → prediction with validation, retry, and fallback.
"""
from __future__ import annotations

from typing import Dict, Optional

from .prompt_builder import build_prompt
from .api_caller import call_llm, load_llm_config
from .schema import parse_llm_output


def predict_with_llm(query: str,
                     few_shot_examples: list | None = None,
                     retrieval_fallback=None,
                     config: dict | None = None) -> Dict[str, int]:
    """
    Full prediction pipeline: build prompt → call LLM → parse → validate.

    Parameters
    ----------
    query : natural-language beam description
    few_shot_examples : retrieved examples for few-shot context
    retrieval_fallback : a callable(query) → dict, used if LLM fails
    config : LLM config dict

    Returns
    -------
    Validated setup dict {"id", "x_bin", "y_bin", "angle_bin"}.
    """
    if config is None:
        config = load_llm_config()

    max_retries = config.get("max_retries", 2)
    messages = build_prompt(query, few_shot_examples)

    for attempt in range(max_retries):
        try:
            raw = call_llm(messages, config)
            result = parse_llm_output(raw)
            if result is not None:
                return result
        except Exception:
            continue

    # Fallback
    if retrieval_fallback is not None:
        return retrieval_fallback(query)

    # Last resort: return center
    return {"id": 0, "x_bin": 10, "y_bin": 10, "angle_bin": 10}
