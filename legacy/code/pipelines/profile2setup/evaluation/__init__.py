"""Evaluation package for profile2setup v2."""

try:
    from .baselines import (
        BaseBaseline,
        MeanAbsoluteBaseline,
        MeanDeltaBaseline,
        NearestNeighborProfileBaseline,
        ZeroDeltaMeanAbsoluteBaseline,
    )
    from .evaluate_model import evaluate_checkpoint, evaluate_outputs_over_loader
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    BaseBaseline = None
    MeanAbsoluteBaseline = None
    MeanDeltaBaseline = None
    NearestNeighborProfileBaseline = None
    ZeroDeltaMeanAbsoluteBaseline = None
    evaluate_checkpoint = None
    evaluate_outputs_over_loader = None

from .reasoning_vlm_eval import evaluate_reasoning_vlm_jsonl

__all__ = [
    "BaseBaseline",
    "MeanAbsoluteBaseline",
    "MeanDeltaBaseline",
    "NearestNeighborProfileBaseline",
    "ZeroDeltaMeanAbsoluteBaseline",
    "evaluate_checkpoint",
    "evaluate_outputs_over_loader",
    "evaluate_reasoning_vlm_jsonl",
]
