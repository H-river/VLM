"""Runtime for a strict-all-five correction over the accepted forward ensemble."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v5.forward_runtime import ZERO_ACTION_INDEX
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from specialist_rebuild_v2.common import forward_feature


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_strict_forward_model(torch: Any, config: Mapping[str, Any]) -> Any:
    nn = torch.nn

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, dropout: float) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(width * 2, width),
            )

        def forward(self, values: Any) -> Any:
            return values + self.layers(values)

    class StrictForwardCorrection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            input_dim = int(config["input_dim"])
            width = int(config["width"])
            depth = int(config["depth"])
            dropout = float(config["dropout"])
            self.input = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.SiLU(),
                nn.LayerNorm(width),
            )
            self.blocks = nn.ModuleList(
                [ResidualBlock(width, dropout) for _ in range(depth)]
            )
            self.output_norm = nn.LayerNorm(width)
            self.output = nn.Linear(width, 5)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, values: Any) -> Any:
            hidden = self.input(values)
            for block in self.blocks:
                hidden = block(hidden)
            return self.output(self.output_norm(hidden))

    return StrictForwardCorrection()


class StrictForwardCorrectionRuntimeV9:
    """Apply a neural correction while preserving the accepted ensemble."""

    def __init__(
        self,
        torch: Any,
        artifact_path: Path,
        device: Any,
    ) -> None:
        path = artifact_path.resolve()
        artifact = torch.load(path, map_location=device, weights_only=False)
        if artifact.get("model") != "strict_forward_correction_v9":
            raise ValueError("unexpected strict-forward artifact")
        current_path = Path(str(artifact["current_selector"])).resolve()
        if sha256(current_path) != artifact["current_selector_sha256"]:
            raise ValueError("strict-forward current-selector checksum differs")
        self.current, _ = load_forward_selector_ensemble_runtime_v9(
            current_path,
            torch,
            device,
        )
        self.model = build_strict_forward_model(
            torch,
            artifact["architecture"],
        ).to(device)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()
        self.mean = torch.as_tensor(
            artifact["input_mean"],
            dtype=torch.float32,
            device=device,
        )
        self.scale = torch.as_tensor(
            artifact["input_scale"],
            dtype=torch.float32,
            device=device,
        )
        self.correction_scale = torch.as_tensor(
            artifact["correction_scale"],
            dtype=torch.float32,
            device=device,
        )
        self.field_blend = torch.as_tensor(
            artifact["field_blend"],
            dtype=torch.float32,
            device=device,
        )
        self.torch = torch
        self.device = device
        self.version = str(artifact["version"])

    @staticmethod
    def _physical_features(
        rows: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        return np.asarray(
            [
                [
                    forward_feature(
                        row["setup"],
                        row["current_beam_state"],
                        action,
                    )
                    for action in ACTION_GRID
                ]
                for row in rows
            ],
            dtype=np.float32,
        )

    def predict_changes(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 256,
    ) -> np.ndarray:
        if not rows:
            return np.empty((0, len(ACTION_GRID), 5), dtype=np.float32)
        outputs = []
        torch = self.torch
        with torch.inference_mode():
            for start in range(0, len(rows), group_batch):
                selected = rows[start : start + group_batch]
                physical = self._physical_features(selected)
                prior = self.current.base.predict_changes(selected)
                current = self.current.predict_changes(
                    selected,
                    group_batch=group_batch,
                )
                flat = np.concatenate(
                    [
                        physical.reshape(-1, physical.shape[-1]),
                        prior.reshape(-1, prior.shape[-1]),
                        current.reshape(-1, current.shape[-1]),
                    ],
                    axis=1,
                ).astype(np.float32)
                values = torch.as_tensor(
                    flat,
                    dtype=torch.float32,
                    device=self.device,
                )
                correction = (
                    self.model((values - self.mean) / self.scale)
                    * self.correction_scale[None, :]
                    * self.field_blend[None, :]
                )
                predicted = (
                    torch.as_tensor(
                        current.reshape(-1, 5),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    + correction
                ).reshape(current.shape)
                predicted[:, ZERO_ACTION_INDEX, :] = 0.0
                outputs.append(predicted.detach().cpu().numpy())
        return np.concatenate(outputs).astype(np.float32)


def load_strict_forward_correction_runtime_v9(
    path: Path,
    torch: Any,
    device: Any,
) -> tuple[StrictForwardCorrectionRuntimeV9, dict[str, Any]]:
    artifact = torch.load(
        path.resolve(),
        map_location="cpu",
        weights_only=False,
    )
    return StrictForwardCorrectionRuntimeV9(torch, path, device), artifact
