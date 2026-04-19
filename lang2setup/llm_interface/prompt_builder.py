"""
prompt_builder.py
Build structured prompts for LLM-based setup prediction.
"""
from __future__ import annotations

from typing import Dict, List, Any

SYSTEM_PROMPT = """\
You are an optical setup assistant. Given a natural-language description of a \
desired laser beam profile, you predict the setup parameters as a JSON object.

The setup has these discrete parameters:
- "id": always 0 (single Gaussian beam family)
- "x_bin": integer 0–20, lateral position (0=far left, 10=centered, 20=far right)
- "y_bin": integer 0–20, vertical position (0=far down, 10=centered, 20=far up)
- "angle_bin": integer 0–20, tilt angle (0=extreme negative, 10=no tilt, 20=extreme positive)

If vertical position is not mentioned, assume y_bin=10 (centered).
Respond ONLY with a JSON object, no explanation. Example:
{"id": 0, "x_bin": 6, "y_bin": 10, "angle_bin": 14}
"""


def build_prompt(query: str,
                 few_shot_examples: List[Dict[str, Any]] | None = None) -> list:
    """
    Build a chat-style message list for the LLM.

    Parameters
    ----------
    query : the user's natural-language beam description
    few_shot_examples : list of {"text": ..., "target": {...}} dicts

    Returns
    -------
    List of {"role": ..., "content": ...} messages.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if few_shot_examples:
        for ex in few_shot_examples:
            messages.append({"role": "user", "content": ex["text"]})
            import json
            messages.append({"role": "assistant",
                             "content": json.dumps(ex["target"])})

    messages.append({"role": "user", "content": query})
    return messages
