"""Fusion model for profile2setup v2 Stage 4."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from profile2setup.schema import VARIABLE_ORDER

from .heads import MultiVariableHeads
from .profile_encoder import ProfileEncoder
from .setup_encoder import SetupEncoder
from .text_encoder import SimpleTextEncoder


def _require_canonical_variable_order() -> None:
    expected = [
        "source_to_lens",
        "lens_to_camera",
        "focal_length",
        "lens_x",
        "lens_y",
        "camera_x",
        "camera_y",
    ]
    if list(VARIABLE_ORDER) != expected:
        raise ValueError(
            "profile2setup.schema.VARIABLE_ORDER is not canonical for v2. "
            f"Expected {expected}, got {list(VARIABLE_ORDER)}"
        )


class Profile2SetupModel(nn.Module):
    """Stage 4 multimodal model for profile2setup v2."""

    def __init__(
        self,
        vocab_size: int,
        input_channels: int = 4,
        num_variables: int = 7,
        profile_dim: int = 256,
        token_dim: int = 128,
        text_dim: int = 128,
        setup_dim: int = 64,
        fused_dim: int = 256,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        _require_canonical_variable_order()

        if num_variables != len(VARIABLE_ORDER):
            raise ValueError(
                "num_variables must match canonical v2 variable count "
                f"{len(VARIABLE_ORDER)}, got {num_variables}"
            )

        self.num_variables = int(num_variables)

        self.profile_encoder = ProfileEncoder(
            in_channels=input_channels,
            profile_dim=profile_dim,
            base_channels=32,
            dropout=dropout,
        )
        self.text_encoder = SimpleTextEncoder(
            vocab_size=vocab_size,
            token_dim=token_dim,
            text_dim=text_dim,
            pad_id=pad_id,
            dropout=dropout,
        )
        self.setup_encoder = SetupEncoder(
            input_dim=self.num_variables,
            setup_dim=setup_dim,
            hidden_dim=64,
            dropout=dropout,
        )

        fusion_in_dim = profile_dim + text_dim + setup_dim
        fusion_layers: list[nn.Module] = [
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0.0:
            fusion_layers.append(nn.Dropout(dropout))
        fusion_layers.extend(
            [
                nn.Linear(fusion_hidden_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU(),
            ]
        )
        if dropout > 0.0:
            fusion_layers.append(nn.Dropout(dropout))
        self.fusion_mlp = nn.Sequential(*fusion_layers)

        self.heads = MultiVariableHeads(
            fused_dim=fused_dim,
            num_variables=self.num_variables,
            hidden_dim=128,
            dropout=dropout,
        )

    def forward(
        self,
        profile: torch.Tensor,
        prompt_tokens: torch.Tensor,
        current_setup: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if profile.shape[0] != prompt_tokens.shape[0] or profile.shape[0] != current_setup.shape[0]:
            raise ValueError(
                "Batch sizes must match across inputs; "
                f"got profile={profile.shape[0]}, prompt_tokens={prompt_tokens.shape[0]}, "
                f"current_setup={current_setup.shape[0]}"
            )

        profile_emb = self.profile_encoder(profile)
        text_emb = self.text_encoder(prompt_tokens)
        setup_emb = self.setup_encoder(current_setup)

        fused_in = torch.cat([profile_emb, text_emb, setup_emb], dim=-1)
        fused = self.fusion_mlp(fused_in)
        return self.heads(fused)


def build_model_from_config(config: dict[str, Any], vocab_size: int) -> Profile2SetupModel:
    """Build Profile2SetupModel from a config dict with safe defaults."""
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")

    model_cfg = config.get("model") or {}
    if not isinstance(model_cfg, dict):
        raise ValueError("config['model'] must be a dict when provided")

    return Profile2SetupModel(
        vocab_size=int(vocab_size),
        input_channels=4,
        num_variables=len(VARIABLE_ORDER),
        profile_dim=int(model_cfg.get("profile_dim", 256)),
        token_dim=int(model_cfg.get("token_dim", 128)),
        text_dim=int(model_cfg.get("text_dim", 128)),
        setup_dim=int(model_cfg.get("setup_dim", 64)),
        fused_dim=int(model_cfg.get("fused_dim", 256)),
        fusion_hidden_dim=int(model_cfg.get("fusion_hidden_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pad_id=int(model_cfg.get("pad_id", 0)),
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
