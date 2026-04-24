"""Inference package for profile2setup v2."""

_HAS_TORCH_ROUTING = True
try:
    from .routing import route_setup_prediction
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _HAS_TORCH_ROUTING = False

__all__ = []

if _HAS_TORCH_ROUTING:
    __all__.append("route_setup_prediction")
