"""Smoke checks for Stage 1 profile2setup reasoning VLM helpers."""

from __future__ import annotations

from copy import deepcopy

from legacy.reasoning_vlm import (
    CANONICAL_VARIABLE_ORDER,
    build_allowed_change_mask,
    build_fixed_change_mask,
    build_relevance_prior,
    default_reasoning_command,
    validate_reasoning_command,
)


def _expect_value_error(command: dict, expected_fragment: str) -> None:
    try:
        validate_reasoning_command(command)
    except ValueError as exc:
        if expected_fragment not in str(exc):
            raise AssertionError(
                f"expected error containing {expected_fragment!r}, got {exc!s}"
            ) from exc
        return
    raise AssertionError("expected ValueError")


def _command_with_fixed_camera() -> dict:
    command = default_reasoning_command()
    command["task_type"] = "paired_no_setup"
    command["canonical_prompt"] = "move the target profile upward while keeping camera offsets fixed"
    command["requested_goal"] = "move target profile upward"
    command["constraints"]["fixed_variables"] = ["camera_x", "camera_y"]
    command["constraints"]["allowed_variables"] = []
    command["constraints"]["change_mask_prior"].update(
        {
            "lens_y": "likely",
            "camera_x": "fixed",
            "camera_y": "fixed",
        }
    )
    command["control_plan"]["likely_relevant_variables"] = ["lens_y"]
    command["control_plan"]["avoid_variables"] = ["camera_x", "camera_y"]
    return command


def main() -> None:
    command = _command_with_fixed_camera()
    validate_reasoning_command(command)

    legacy_root = "alignment"
    legacy_command = deepcopy(command)
    legacy_command["constraints"]["fixed_variables"] = [f"{legacy_root}_x"]
    _expect_value_error(legacy_command, "legacy variable")

    unknown_command = deepcopy(command)
    unknown_command["control_plan"]["likely_relevant_variables"] = ["mirror_angle"]
    _expect_value_error(unknown_command, "non-canonical variable")

    allowed_mask = build_allowed_change_mask(command)
    fixed_mask = build_fixed_change_mask(command)
    relevance = build_relevance_prior(command)

    camera_x_idx = CANONICAL_VARIABLE_ORDER.index("camera_x")
    camera_y_idx = CANONICAL_VARIABLE_ORDER.index("camera_y")
    for idx in (camera_x_idx, camera_y_idx):
        if allowed_mask[idx] != 0.0:
            raise AssertionError(f"fixed camera variable has allowed mask {allowed_mask[idx]}")
        if fixed_mask[idx] != 1.0:
            raise AssertionError(f"fixed camera variable has fixed mask {fixed_mask[idx]}")
        if relevance[idx] != 0.0:
            raise AssertionError(f"fixed camera variable has relevance {relevance[idx]}")

    if len(allowed_mask) != 7 or len(fixed_mask) != 7 or len(relevance) != 7:
        raise AssertionError("reasoning masks must have length 7")

    print("reasoning VLM smoke test passed")
    print(f"allowed_change_mask: {allowed_mask}")
    print(f"fixed_change_mask: {fixed_mask}")
    print(f"relevance_prior: {relevance}")


if __name__ == "__main__":
    main()
