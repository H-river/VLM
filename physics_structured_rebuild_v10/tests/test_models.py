from __future__ import annotations

import numpy as np
import torch

from physics_structured_rebuild_v10.contracts import (
    ACTION_NORMALIZED,
    explicit_action_features,
)
from physics_structured_rebuild_v10.models import (
    build_forward_model,
    forward_call,
)


def test_forward_models_emit_complete_zero_anchored_surfaces() -> None:
    config = {
        "dimension": 32,
        "dropout": 0.0,
        "heads": 4,
        "layers": 1,
        "explicit_action_dim": explicit_action_features().shape[1],
    }
    context = torch.randn(3, 17)
    explicit = torch.tensor(explicit_action_features())
    categories = torch.tensor((ACTION_NORMALIZED + 1).astype(np.int64))
    for architecture in ("pointwise_action_id", "structured_81_action"):
        model = build_forward_model(torch, architecture, config)
        mean, log_variance = forward_call(
            model,
            architecture,
            context,
            explicit,
            categories,
        )
        assert mean.shape == (3, 81, 5)
        assert torch.all(mean[:, 40, :] == 0)
        if architecture == "structured_81_action":
            assert log_variance is not None
            assert log_variance.shape == (3, 81, 5)

