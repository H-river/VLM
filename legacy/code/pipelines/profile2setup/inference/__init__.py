"""Inference package for profile2setup v2."""

_HAS_TORCH_ROUTING = True
try:
    from .routing import (
        apply_allowed_change_mask_to_delta,
        apply_fixed_change_mask_to_delta,
        route_setup_prediction,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _HAS_TORCH_ROUTING = False

__all__ = []

if _HAS_TORCH_ROUTING:
    __all__.extend(
        [
            "apply_allowed_change_mask_to_delta",
            "apply_fixed_change_mask_to_delta",
            "route_setup_prediction",
        ]
    )
