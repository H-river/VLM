"""Training targets and assistant JSON for text optics SFT."""

from __future__ import annotations

import copy
import json
from typing import Any, Literal, Mapping

TrainingTargetMode = Literal["full", "compact", "control_plan_only"]
TRAINING_TARGET_MODES: tuple[TrainingTargetMode, ...] = (
    "full",
    "compact",
    "control_plan_only",
)

COMPACT_TARGET_KEYS = ("task", "perception", "control_plan", "confidence")
DEFAULT_TASK = "physics_aware_beam_alignment"


def normalize_training_target(
    training_target: str | None = None,
    *,
    compact_target: bool = False,
) -> TrainingTargetMode:
    """Resolve training target mode from explicit value or legacy compact_target flag."""
    if training_target is not None:
        if training_target not in TRAINING_TARGET_MODES:
            raise ValueError(
                f"Unsupported training_target: {training_target!r}. "
                f"Expected one of {TRAINING_TARGET_MODES}"
            )
        return training_target  # type: ignore[return-value]
    if compact_target:
        return "compact"
    return "full"


def compact_target(full_target: Mapping[str, Any]) -> dict[str, Any]:
    """Keep fields used for compact supervision and eval perception checks."""
    compact: dict[str, Any] = {}
    for key in COMPACT_TARGET_KEYS:
        if key in full_target:
            compact[key] = copy.deepcopy(full_target[key])
    if "task" not in compact:
        compact["task"] = DEFAULT_TASK
    if not isinstance(compact.get("control_plan"), dict):
        raise ValueError("compact target requires target.control_plan")
    return compact


def control_plan_only_target(full_target: Mapping[str, Any]) -> dict[str, Any]:
    """Supervise only task + control_plan to reduce template memorization."""
    plan = full_target.get("control_plan")
    if not isinstance(plan, dict):
        raise ValueError("control_plan_only target requires target.control_plan")
    return {
        "task": full_target.get("task", DEFAULT_TASK),
        "control_plan": copy.deepcopy(plan),
    }


def build_training_target(
    full_target: Mapping[str, Any],
    mode: TrainingTargetMode,
) -> dict[str, Any]:
    if mode == "full":
        return copy.deepcopy(dict(full_target))
    if mode == "compact":
        return compact_target(full_target)
    if mode == "control_plan_only":
        return control_plan_only_target(full_target)
    raise ValueError(f"Unsupported training target mode: {mode!r}")


def training_target_json(full_target: Mapping[str, Any], mode: TrainingTargetMode) -> str:
    return json.dumps(build_training_target(full_target, mode), sort_keys=True)


def apply_training_messages(
    row: Mapping[str, Any],
    mode: TrainingTargetMode,
) -> list[dict[str, str]]:
    from optics_sft.physics.text_prompt_builder import build_text_prompt

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"Row {row.get('sample_id')} is missing messages")
    target = row.get("target")
    if not isinstance(target, Mapping):
        raise ValueError(f"Row {row.get('sample_id')} is missing target")
    return [
        {"role": "user", "content": build_text_prompt(row, training_target=mode)},
        {"role": "assistant", "content": training_target_json(target, mode)},
    ]


def apply_compact_training_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Backward-compatible alias for compact training rows."""
    return apply_training_messages(row, "compact")


def compact_target_json(full_target: Mapping[str, Any]) -> str:
    return training_target_json(full_target, "compact")
