"""Masked loss helpers for profile2setup multi-head training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(loss: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return a mask-normalized mean, yielding zero when the mask is all zero."""
    if loss.shape[0] != mask.shape[0]:
        raise ValueError(
            "loss and mask batch sizes must match; "
            f"got loss={loss.shape[0]}, mask={mask.shape[0]}"
        )
    if mask.ndim == 1:
        mask = mask.unsqueeze(-1)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=loss.dtype, device=loss.device)
    masked = loss * mask
    denom = mask.expand_as(loss).sum().clamp_min(eps)
    return masked.sum() / denom


def compute_profile2setup_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    absolute_weight: float = 1.0,
    delta_weight: float = 1.0,
    change_weight: float = 0.5,
    constraint_weight: float = 0.0,
    fixed_change_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute mask-normalized absolute, delta, change, and optional constraint losses."""
    required_outputs = ["absolute", "delta", "change_logits"]
    required_batch = [
        "target_setup",
        "target_delta",
        "change_mask",
        "absolute_loss_mask",
        "delta_loss_mask",
        "change_loss_mask",
    ]
    for key in required_outputs:
        if key not in outputs:
            raise KeyError(f"outputs missing required key: {key}")
    for key in required_batch:
        if key not in batch:
            raise KeyError(f"batch missing required key: {key}")

    absolute_loss = masked_mean(
        F.mse_loss(outputs["absolute"], batch["target_setup"], reduction="none"),
        batch["absolute_loss_mask"],
    )
    delta_loss = masked_mean(
        F.mse_loss(outputs["delta"], batch["target_delta"], reduction="none"),
        batch["delta_loss_mask"],
    )
    change_loss = masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["change_logits"],
            batch["change_mask"],
            reduction="none",
        ),
        batch["change_loss_mask"],
    )

    if fixed_change_mask is None:
        fixed_change_mask = batch.get("fixed_change_mask")

    constraint_loss = outputs["delta"].new_zeros(())
    if float(constraint_weight) > 0.0 and fixed_change_mask is not None and "delta" in outputs:
        mask = fixed_change_mask
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask.to(dtype=outputs["delta"].dtype, device=outputs["delta"].device)
        constraint_loss = torch.mean(torch.abs(outputs["delta"] * mask))

    total = (
        float(absolute_weight) * absolute_loss
        + float(delta_weight) * delta_loss
        + float(change_weight) * change_loss
        + float(constraint_weight) * constraint_loss
    )
    return {
        "loss": total,
        "absolute_loss": absolute_loss,
        "delta_loss": delta_loss,
        "change_loss": change_loss,
        "constraint_loss": constraint_loss,
    }
