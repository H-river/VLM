"""Strict actuator-free JSON contract for Qwen plan selection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .protocol import PLAN_NAMES


OUTPUT_KEYS = ("plan_ranking", "selected_plan")


@dataclass(frozen=True)
class SelectorDecision:
    plan_ranking: tuple[str, ...]
    selected_plan: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_ranking": list(self.plan_ranking),
            "selected_plan": self.selected_plan,
        }


def parse_selector_output(value: str | Mapping[str, Any]) -> SelectorDecision:
    if isinstance(value, str):
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            output: dict[str, Any] = {}
            for key, nested in pairs:
                if key in output:
                    raise ValueError(f"duplicate JSON key: {key}")
                output[key] = nested
            return output

        try:
            value = json.loads(
                value,
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON token: {token}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid whole-string selector JSON: {exc}") from exc
    if not isinstance(value, Mapping) or tuple(value.keys()) != OUTPUT_KEYS:
        raise ValueError(f"selector output keys and order must be exactly {OUTPUT_KEYS}")
    ranking = value["plan_ranking"]
    selected = value["selected_plan"]
    if not isinstance(ranking, list) or any(not isinstance(item, str) for item in ranking):
        raise ValueError("plan_ranking must be a string list")
    if tuple(sorted(ranking)) != tuple(sorted(PLAN_NAMES)) or len(ranking) != len(PLAN_NAMES):
        raise ValueError("plan_ranking must be an exact permutation of executable plans")
    if not isinstance(selected, str) or selected not in PLAN_NAMES:
        raise ValueError("selected_plan is not executable")
    if selected != ranking[0]:
        raise ValueError("selected_plan must equal the first ranked plan")
    return SelectorDecision(tuple(ranking), selected)


def canonical_selector_text(value: SelectorDecision | Mapping[str, Any]) -> str:
    decision = value if isinstance(value, SelectorDecision) else parse_selector_output(value)
    return json.dumps(
        decision.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
