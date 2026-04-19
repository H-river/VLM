"""
rule_based.py
Keyword → bin mapping baseline.
v3: 21 keywords per axis, one per bin (0–20).
"""
from __future__ import annotations

import re
from typing import Dict

_C = 10  # center bin


# ── Ordered keyword lists ──
# Each list has 21 entries, index = bin number.
# Matching is done longest-first to avoid substring collisions.

_X_KEYWORDS = [
    # bin: keyword fragment (matched in lowered text)
    (0,  "extreme far left"),
    (1,  "very far to the left"),     # "shifted very far to the left"
    (2,  "far to the left"),
    (3,  "well to the left"),
    (4,  "moderately to the left"),
    (5,  "noticeably to the left"),
    (6,  "slightly to the left"),
    (7,  "barely to the left"),
    (8,  "just left"),
    (9,  "marginally left"),
    (10, "centered"),
    (11, "marginally right"),
    (12, "just right"),
    (13, "barely to the right"),
    (14, "slightly to the right"),
    (15, "noticeably to the right"),
    (16, "moderately to the right"),
    (17, "well to the right"),
    (18, "far to the right"),
    (19, "very far to the right"),
    (20, "extreme far right"),
]

_Y_KEYWORDS = [
    (0,  "extreme far bottom"),
    (1,  "very far downward"),
    (2,  "far downward"),
    (3,  "well downward"),
    (4,  "moderately downward"),
    (5,  "noticeably downward"),
    (6,  "slightly downward"),
    (7,  "barely downward"),
    (8,  "just below"),
    (9,  "marginally below"),
    (10, "vertically centered"),
    (11, "marginally above"),
    (12, "just above"),
    (13, "barely upward"),
    (14, "slightly upward"),
    (15, "noticeably upward"),
    (16, "moderately upward"),
    (17, "well upward"),
    (18, "far upward"),
    (19, "very far upward"),
    (20, "extreme far top"),
]

_TILT_KEYWORDS = [
    (0,  "extreme negative"),
    (1,  "very strong negative"),
    (2,  "strong negative"),
    (3,  "firm negative"),
    (4,  "moderate negative"),
    (5,  "noticeable negative"),
    (6,  "slight negative"),
    (7,  "mild negative"),
    (8,  "weak negative"),
    (9,  "minimal negative"),
    (10, "no tilt"),
    (11, "minimal positive"),
    (12, "weak positive"),
    (13, "mild positive"),
    (14, "slight positive"),
    (15, "noticeable positive"),
    (16, "moderate positive"),
    (17, "firm positive"),
    (18, "strong positive"),
    (19, "very strong positive"),
    (20, "extreme positive"),
]


def _match_keywords(text: str, keyword_list: list[tuple[int, str]], default: int = _C) -> int:
    """Match longest keyword first to avoid substring collisions."""
    # Sort by keyword length descending — longer phrases matched first
    sorted_kw = sorted(keyword_list, key=lambda x: len(x[1]), reverse=True)
    for bin_idx, kw in sorted_kw:
        if kw in text:
            return bin_idx
    return default


def predict_rule_based(text: str) -> Dict[str, int]:
    """
    Heuristic keyword-based prediction. Returns bin dict.
    """
    text_lower = text.lower()
    return {
        "id": 0,
        "x_bin": _match_keywords(text_lower, _X_KEYWORDS),
        "y_bin": _match_keywords(text_lower, _Y_KEYWORDS),
        "angle_bin": _match_keywords(text_lower, _TILT_KEYWORDS),
    }
