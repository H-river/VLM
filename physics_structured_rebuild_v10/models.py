"""Pointwise control and structured 81-action response-surface models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_forward_model(
    torch: Any,
    architecture: str,
    config: Mapping[str, Any],
) -> Any:
    dimension = int(config["dimension"])
    dropout = float(config["dropout"])

    class PointwiseActionIdSurface(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context = torch.nn.Sequential(
                torch.nn.Linear(17, dimension),
                torch.nn.LayerNorm(dimension),
                torch.nn.GELU(),
                torch.nn.Linear(dimension, dimension),
                torch.nn.GELU(),
            )
            self.action_id = torch.nn.Embedding(81, dimension)
            self.output = torch.nn.Sequential(
                torch.nn.Linear(2 * dimension, 2 * dimension),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(2 * dimension, 5),
            )
            self.register_buffer(
                "nonzero_mask",
                torch.ones(1, 81, 1, dtype=torch.float32),
            )
            self.nonzero_mask[:, 40, :] = 0.0

        def forward(self, context: Any) -> tuple[Any, None]:
            batch = context.shape[0]
            encoded = self.context(context)[:, None, :].expand(-1, 81, -1)
            action = self.action_id.weight[None, :, :].expand(batch, -1, -1)
            mean = self.output(torch.cat([encoded, action], dim=2))
            return mean * self.nonzero_mask, None

    class StructuredActionSurface(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            explicit_dim = int(config["explicit_action_dim"])
            self.context = torch.nn.Sequential(
                torch.nn.Linear(17, dimension),
                torch.nn.LayerNorm(dimension),
                torch.nn.GELU(),
                torch.nn.Linear(dimension, dimension),
            )
            self.actuator_effects = torch.nn.ModuleList(
                [torch.nn.Embedding(3, dimension) for _ in range(4)]
            )
            self.explicit_interactions = torch.nn.Sequential(
                torch.nn.Linear(explicit_dim, dimension),
                torch.nn.LayerNorm(dimension),
                torch.nn.GELU(),
                torch.nn.Linear(dimension, dimension),
            )
            layer = torch.nn.TransformerEncoderLayer(
                d_model=dimension,
                nhead=int(config.get("heads", 4)),
                dim_feedforward=3 * dimension,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.group_interactions = torch.nn.TransformerEncoder(
                layer,
                num_layers=int(config.get("layers", 2)),
                norm=torch.nn.LayerNorm(dimension),
            )

            def head(output: int) -> Any:
                return torch.nn.Sequential(
                    torch.nn.Linear(2 * dimension, dimension),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(dimension, output),
                )

            self.centroid_head = head(2)
            self.width_head = head(2)
            self.peak_head = head(1)
            self.log_variance_head = head(5)
            self.register_buffer(
                "nonzero_mask",
                torch.ones(1, 81, 1, dtype=torch.float32),
            )
            self.nonzero_mask[:, 40, :] = 0.0

        def forward(
            self,
            context: Any,
            explicit_actions: Any,
            actuator_categories: Any,
        ) -> tuple[Any, Any]:
            batch = context.shape[0]
            context_encoded = self.context(context)
            individual = 0.0
            for actuator, embedding in enumerate(self.actuator_effects):
                individual = individual + embedding(
                    actuator_categories[:, actuator]
                )
            action = (
                self.explicit_interactions(explicit_actions)
                + individual
            )
            tokens = action[None, :, :].expand(batch, -1, -1)
            tokens = tokens + context_encoded[:, None, :]
            encoded = self.group_interactions(tokens)
            context_broadcast = context_encoded[:, None, :].expand(-1, 81, -1)
            combined = torch.cat([encoded, context_broadcast], dim=2)
            mean = torch.cat(
                [
                    self.centroid_head(combined),
                    self.width_head(combined),
                    self.peak_head(combined),
                ],
                dim=2,
            )
            log_variance = self.log_variance_head(combined).clamp(-6.0, 4.0)
            return mean * self.nonzero_mask, log_variance

    if architecture == "pointwise_action_id":
        return PointwiseActionIdSurface()
    if architecture == "structured_81_action":
        return StructuredActionSurface()
    raise ValueError(f"unknown forward architecture: {architecture}")


def forward_call(
    model: Any,
    architecture: str,
    context: Any,
    explicit_actions: Any,
    actuator_categories: Any,
) -> tuple[Any, Any | None]:
    if architecture == "pointwise_action_id":
        return model(context)
    return model(context, explicit_actions, actuator_categories)


def build_inverse_ranker(torch: Any, config: Mapping[str, Any]) -> Any:
    """Direct multi-positive scorer over all 81 structured legal actions."""

    class DirectInverseRanker(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimension = int(config["dimension"])
            dropout = float(config["dropout"])
            self.request = torch.nn.Sequential(
                torch.nn.Linear(int(config["request_dim"]), dimension),
                torch.nn.LayerNorm(dimension),
                torch.nn.GELU(),
                torch.nn.Linear(dimension, dimension),
            )
            self.action = torch.nn.Sequential(
                torch.nn.Linear(int(config["action_dim"]), dimension),
                torch.nn.LayerNorm(dimension),
                torch.nn.GELU(),
                torch.nn.Linear(dimension, dimension),
            )
            layer = torch.nn.TransformerEncoderLayer(
                d_model=dimension,
                nhead=4,
                dim_feedforward=3 * dimension,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.candidates = torch.nn.TransformerEncoder(
                layer,
                num_layers=2,
                norm=torch.nn.LayerNorm(dimension),
            )
            self.score = torch.nn.Sequential(
                torch.nn.Linear(2 * dimension, dimension),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(dimension, 1),
            )

        def forward(self, request: Any, actions: Any) -> Any:
            batch = request.shape[0]
            request_encoded = self.request(request)
            tokens = self.action(actions)[None, :, :].expand(batch, -1, -1)
            tokens = self.candidates(tokens + request_encoded[:, None, :])
            request_broadcast = request_encoded[:, None, :].expand(-1, 81, -1)
            return self.score(
                torch.cat([tokens, request_broadcast], dim=2)
            ).squeeze(-1)

    return DirectInverseRanker()


def build_visual_inverse_ranker(torch: Any, config: Mapping[str, Any]) -> Any:
    """Shared-image encoder with direct scores for all legal actions."""

    class VisualInverseRanker(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimension = int(config["dimension"])
            dropout = float(config["dropout"])
            self.image = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, 5, stride=2, padding=2),
                torch.nn.GELU(),
                torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(64, 96, 3, stride=2, padding=1),
                torch.nn.GELU(),
                torch.nn.AdaptiveAvgPool2d((2, 2)),
                torch.nn.Flatten(),
                torch.nn.Linear(96 * 4, dimension),
                torch.nn.LayerNorm(dimension),
                torch.nn.GELU(),
            )
            self.numeric_auxiliary = torch.nn.Sequential(
                torch.nn.Linear(int(config["context_dim"]), dimension),
                torch.nn.LayerNorm(dimension),
                torch.nn.GELU(),
            )
            self.pair = torch.nn.Sequential(
                torch.nn.Linear(5 * dimension, 2 * dimension),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(2 * dimension, dimension),
            )
            self.action = torch.nn.Sequential(
                torch.nn.Linear(int(config["action_dim"]), dimension),
                torch.nn.LayerNorm(dimension),
                torch.nn.GELU(),
                torch.nn.Linear(dimension, dimension),
            )
            self.score = torch.nn.Sequential(
                torch.nn.Linear(2 * dimension, dimension),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(dimension, 1),
            )

        def forward(
            self,
            current_image: Any,
            target_image: Any,
            numeric_context: Any,
            actions: Any,
        ) -> Any:
            combined_images = torch.cat([current_image, target_image], dim=0)
            encoded = self.image(combined_images)
            current, target = encoded.chunk(2, dim=0)
            numeric = self.numeric_auxiliary(numeric_context)
            pair = self.pair(
                torch.cat(
                    [
                        current,
                        target,
                        target - current,
                        current * target,
                        numeric,
                    ],
                    dim=1,
                )
            )
            action = self.action(actions)[None, :, :].expand(
                len(pair), -1, -1
            )
            pair_broadcast = pair[:, None, :].expand(-1, 81, -1)
            return self.score(
                torch.cat([action, pair_broadcast], dim=2)
            ).squeeze(-1)

    return VisualInverseRanker()
