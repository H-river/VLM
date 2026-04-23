"""Text encoder for profile2setup v2 Stage 4 model."""

from __future__ import annotations

import torch
from torch import nn


class SimpleTextEncoder(nn.Module):
    """Embedding + masked mean pooling text encoder."""

    def __init__(
        self,
        vocab_size: int,
        token_dim: int = 128,
        text_dim: int = 128,
        pad_id: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}")
        if text_dim <= 0:
            raise ValueError(f"text_dim must be positive, got {text_dim}")
        if pad_id < 0:
            raise ValueError(f"pad_id must be >= 0, got {pad_id}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be >= 0, got {dropout}")

        self.pad_id = int(pad_id)
        self.embedding = nn.Embedding(vocab_size, token_dim, padding_idx=self.pad_id)

        projection_layers: list[nn.Module] = [
            nn.Linear(token_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
        ]
        if dropout > 0.0:
            projection_layers.append(nn.Dropout(dropout))
        self.projection = nn.Sequential(*projection_layers)

    def forward(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        if prompt_tokens.ndim != 2:
            raise ValueError(
                "SimpleTextEncoder expects token ids shape [B, T]; "
                f"got ndim={prompt_tokens.ndim} with shape={tuple(prompt_tokens.shape)}"
            )

        tokens = prompt_tokens.long()
        token_emb = self.embedding(tokens)

        mask = (tokens != self.pad_id).unsqueeze(-1)
        masked_sum = (token_emb * mask.to(token_emb.dtype)).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1).to(token_emb.dtype)
        pooled = masked_sum / counts

        return self.projection(pooled)
