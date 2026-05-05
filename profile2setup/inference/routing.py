"""Inference routing helpers for profile2setup input modes."""

from __future__ import annotations

import torch


def _mask_to_tensor(mask, delta: torch.Tensor, *, name: str) -> torch.Tensor:
    mask_tensor = torch.as_tensor(mask, dtype=delta.dtype, device=delta.device)
    if mask_tensor.ndim == 1:
        if mask_tensor.shape[0] != delta.shape[-1]:
            raise ValueError(
                f"{name} length must match delta width; got {mask_tensor.shape[0]} and {delta.shape[-1]}"
            )
        mask_tensor = mask_tensor.unsqueeze(0)
    elif mask_tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [V] or [B, V]; got {tuple(mask_tensor.shape)}")

    if mask_tensor.shape[-1] != delta.shape[-1]:
        raise ValueError(
            f"{name} width must match delta width; got {mask_tensor.shape[-1]} and {delta.shape[-1]}"
        )
    if mask_tensor.shape[0] not in {1, delta.shape[0]}:
        raise ValueError(
            f"{name} batch size must be 1 or match delta batch; got {mask_tensor.shape[0]} and {delta.shape[0]}"
        )
    return mask_tensor


def apply_allowed_change_mask_to_delta(delta: torch.Tensor, allowed_change_mask) -> torch.Tensor:
    """Zero delta entries that are not allowed to change."""
    mask = _mask_to_tensor(allowed_change_mask, delta, name="allowed_change_mask")
    return delta * mask


def apply_fixed_change_mask_to_delta(delta: torch.Tensor, fixed_change_mask) -> torch.Tensor:
    """Zero delta entries for variables that must remain fixed."""
    mask = _mask_to_tensor(fixed_change_mask, delta, name="fixed_change_mask")
    return delta * (1.0 - mask)


def route_setup_prediction(
    outputs: dict[str, torch.Tensor],
    current_setup: torch.Tensor,
    setup_present: torch.Tensor | None,
    *,
    prefer_absolute_when_setup_missing: bool = True,
) -> torch.Tensor:
    """Select absolute or delta-composed setup predictions by setup availability."""
    for key in ("absolute", "delta"):
        if key not in outputs:
            raise KeyError(f"outputs missing required key: {key}")
    absolute = outputs["absolute"]
    delta = outputs["delta"]
    if absolute.shape != delta.shape:
        raise ValueError(
            "absolute and delta predictions must have the same shape; "
            f"got absolute={tuple(absolute.shape)}, delta={tuple(delta.shape)}"
        )
    if current_setup.shape != absolute.shape:
        raise ValueError(
            "current_setup must match prediction shape; "
            f"got current_setup={tuple(current_setup.shape)}, prediction={tuple(absolute.shape)}"
        )

    if setup_present is None:
        setup_present = torch.ones(absolute.shape[0], 1, dtype=absolute.dtype, device=absolute.device)
    elif setup_present.ndim == 1:
        setup_present = setup_present.unsqueeze(-1)
    elif setup_present.ndim != 2 or setup_present.shape[-1] != 1:
        raise ValueError(
            "setup_present must have shape [B] or [B, 1]; "
            f"got shape={tuple(setup_present.shape)}"
        )
    if setup_present.shape[0] != absolute.shape[0]:
        raise ValueError(
            "setup_present batch size must match predictions; "
            f"got setup_present={setup_present.shape[0]}, predictions={absolute.shape[0]}"
        )

    setup_present = setup_present.to(dtype=absolute.dtype, device=absolute.device)
    delta_composed = current_setup.to(dtype=absolute.dtype, device=absolute.device) + delta
    if not prefer_absolute_when_setup_missing:
        return delta_composed
    return setup_present * delta_composed + (1.0 - setup_present) * absolute
