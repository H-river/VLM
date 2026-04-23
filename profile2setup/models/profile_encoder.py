"""Profile encoder for profile2setup v2 Stage 4 model."""

from __future__ import annotations

import torch
from torch import nn


class ProfileEncoder(nn.Module):
    """Lightweight CNN encoder for 4-channel beam profile tensors."""

    def __init__(
        self,
        in_channels: int = 4,
        profile_dim: int = 256,
        base_channels: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if profile_dim <= 0:
            raise ValueError(f"profile_dim must be positive, got {profile_dim}")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be >= 0, got {dropout}")

        self.in_channels = int(in_channels)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        projection_layers: list[nn.Module] = [
            nn.Linear(c4, profile_dim),
            nn.LayerNorm(profile_dim),
            nn.GELU(),
        ]
        if dropout > 0.0:
            projection_layers.append(nn.Dropout(dropout))
        self.projection = nn.Sequential(*projection_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "ProfileEncoder expects input shape [B, C, H, W]; "
                f"got ndim={x.ndim} with shape={tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"ProfileEncoder expects channel dimension {self.in_channels}; "
                f"got {x.shape[1]} with shape={tuple(x.shape)}"
            )

        feats = self.features(x)
        return self.projection(feats)
