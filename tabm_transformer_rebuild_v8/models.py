"""TabM and token-Transformer model definitions."""

from __future__ import annotations

from typing import Any


def build_forward_direction_model(
    torch: Any,
    architecture: str,
    input_dim: int,
    config: dict[str, Any],
) -> Any:
    if architecture == "tabm":
        import tabm

        class TabMForwardDirection(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = tabm.TabM.make(
                    n_num_features=input_dim,
                    cat_cardinalities=[],
                    d_out=20,
                    k=int(config["k"]),
                    n_blocks=int(config["n_blocks"]),
                    d_block=int(config["d_block"]),
                    dropout=float(config["dropout"]),
                    arch_type="tabm",
                )

            def forward(self, values: Any) -> tuple[Any, Any]:
                output = self.model(values)
                change = output[..., :5]
                direction = output[..., 5:].reshape(
                    output.shape[0],
                    output.shape[1],
                    5,
                    3,
                )
                return change, direction

        return TabMForwardDirection()

    if architecture != "transformer":
        raise ValueError(f"unknown architecture: {architecture}")

    class FeatureTokenTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimension = int(config["dimension"])
            self.feature_weight = torch.nn.Parameter(
                torch.empty(input_dim, dimension)
            )
            self.feature_bias = torch.nn.Parameter(
                torch.empty(input_dim, dimension)
            )
            self.feature_identity = torch.nn.Parameter(
                torch.empty(input_dim, dimension)
            )
            self.cls = torch.nn.Parameter(torch.empty(1, 1, dimension))
            layer = torch.nn.TransformerEncoderLayer(
                d_model=dimension,
                nhead=int(config["heads"]),
                dim_feedforward=int(config["feedforward"]),
                dropout=float(config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(
                layer,
                num_layers=int(config["layers"]),
                norm=torch.nn.LayerNorm(dimension),
            )
            self.output = torch.nn.Linear(dimension, 20)
            torch.nn.init.normal_(self.feature_weight, std=0.02)
            torch.nn.init.normal_(self.feature_bias, std=0.02)
            torch.nn.init.normal_(self.feature_identity, std=0.02)
            torch.nn.init.normal_(self.cls, std=0.02)

        def forward(self, values: Any) -> tuple[Any, Any]:
            tokens = (
                values[:, :, None] * self.feature_weight[None, :, :]
                + self.feature_bias[None, :, :]
                + self.feature_identity[None, :, :]
            )
            cls = self.cls.expand(len(values), -1, -1)
            encoded = self.encoder(torch.cat([cls, tokens], dim=1))
            output = self.output(encoded[:, 0])
            return output[:, :5], output[:, 5:].reshape(-1, 5, 3)

    return FeatureTokenTransformer()


def build_inverse_model(
    torch: Any,
    architecture: str,
    context_dim: int,
    candidate_dim: int,
    status_dim: int,
    candidate_count: int,
    config: dict[str, Any],
) -> Any:
    if architecture == "tabm":
        import tabm

        class TabMInverse(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                common = {
                    "k": int(config["k"]),
                    "n_blocks": int(config["n_blocks"]),
                    "d_block": int(config["d_block"]),
                    "dropout": float(config["dropout"]),
                    "arch_type": "tabm",
                }
                self.candidate = tabm.TabM.make(
                    n_num_features=context_dim + candidate_dim,
                    cat_cardinalities=[],
                    d_out=1,
                    **common,
                )
                self.status = tabm.TabM.make(
                    n_num_features=context_dim + status_dim,
                    cat_cardinalities=[],
                    d_out=3,
                    **common,
                )

            def forward(
                self,
                context: Any,
                candidates: Any,
                status_features: Any,
            ) -> tuple[Any, Any]:
                repeated = context[:, None, :].expand(
                    -1,
                    candidates.shape[1],
                    -1,
                )
                combined = torch.cat([repeated, candidates], dim=-1)
                correction = self.candidate(
                    combined.reshape(-1, combined.shape[-1])
                )
                correction = correction.reshape(
                    len(context),
                    candidates.shape[1],
                    correction.shape[-2],
                )
                status_input = torch.cat([context, status_features], dim=-1)
                return correction, self.status(status_input)

        return TabMInverse()

    if architecture != "transformer":
        raise ValueError(f"unknown architecture: {architecture}")

    class CandidateSetTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimension = int(config["dimension"])
            self.context = torch.nn.Linear(context_dim, dimension)
            self.candidate = torch.nn.Linear(candidate_dim, dimension)
            self.action_identity = torch.nn.Parameter(
                torch.empty(1, candidate_count, dimension)
            )
            self.context_identity = torch.nn.Parameter(
                torch.empty(1, 1, dimension)
            )
            layer = torch.nn.TransformerEncoderLayer(
                d_model=dimension,
                nhead=int(config["heads"]),
                dim_feedforward=int(config["feedforward"]),
                dropout=float(config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(
                layer,
                num_layers=int(config["layers"]),
                norm=torch.nn.LayerNorm(dimension),
            )
            self.score = torch.nn.Linear(dimension, 1)
            self.status = torch.nn.Sequential(
                torch.nn.Linear(dimension + status_dim, dimension),
                torch.nn.GELU(),
                torch.nn.Linear(dimension, 3),
            )
            torch.nn.init.normal_(self.action_identity, std=0.02)
            torch.nn.init.normal_(self.context_identity, std=0.02)

        def forward(
            self,
            context: Any,
            candidates: Any,
            status_features: Any,
        ) -> tuple[Any, Any]:
            context_token = (
                self.context(context)[:, None, :] + self.context_identity
            )
            candidate_tokens = (
                self.candidate(candidates)
                + self.action_identity[:, : candidates.shape[1], :]
            )
            encoded = self.encoder(
                torch.cat([context_token, candidate_tokens], dim=1)
            )
            correction = self.score(encoded[:, 1:]).squeeze(-1)
            status = self.status(
                torch.cat([encoded[:, 0], status_features], dim=-1)
            )
            return correction, status

    return CandidateSetTransformer()
