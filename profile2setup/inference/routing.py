"""Inference routing helpers for profile2setup input modes."""

from __future__ import annotations

import torch


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
