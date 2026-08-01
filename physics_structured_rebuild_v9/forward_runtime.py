"""Runtime for calibrated boosted residual corrections over frozen forward v7."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID, tolerance_from_current
from control_rebuild_v5.forward_runtime import ZERO_ACTION_INDEX
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    STATE_FIELDS,
    forward_feature,
    raw_state_array,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResidualForwardRuntimeV9:
    """Add calibrated tree residuals to the frozen v7 normalized changes."""

    def __init__(
        self,
        torch: Any,
        artifact_path: Path,
        device: Any,
    ) -> None:
        path = artifact_path.resolve()
        with path.open("rb") as stream:
            artifact = pickle.load(stream)
        if artifact.get("model") != "frozen_v7_plus_residual_tree_v9":
            raise ValueError("unexpected residual-forward calibration artifact")
        base_path = Path(str(artifact["forward_artifact"])).resolve()
        tree_path = Path(str(artifact["tree_artifact"])).resolve()
        if sha256(base_path) != artifact["forward_artifact_sha256"]:
            raise ValueError("residual-forward base checksum differs")
        if sha256(tree_path) != artifact["tree_artifact_sha256"]:
            raise ValueError("residual-forward tree checksum differs")
        with tree_path.open("rb") as stream:
            tree = pickle.load(stream)
        if tree.get("model") not in {
            "v7_stacked_five_head_hist_gradient_boosting_forward_v9",
            "v7_stacked_five_head_extra_trees_forward_v9",
            "v7_stacked_five_head_lightgbm_forward_v9",
        }:
            raise ValueError("unexpected residual-forward tree artifact")
        self.base, _ = load_forward_direction_runtime_v7(
            base_path,
            torch,
            device,
        )
        self.models = list(tree["models"])
        self.secondary_models = None
        self.complexity_primary_blend = None
        self.complexity_secondary_blend = None
        secondary_tree_path = artifact.get("secondary_tree_artifact")
        declared_sources = artifact.get("field_tree_source")
        complexity_blends = artifact.get("complexity_field_blend")
        if secondary_tree_path is not None:
            secondary_path = Path(str(secondary_tree_path)).resolve()
            if (
                sha256(secondary_path)
                != artifact["secondary_tree_artifact_sha256"]
            ):
                raise ValueError("secondary residual-forward checksum differs")
            with secondary_path.open("rb") as stream:
                secondary_tree = pickle.load(stream)
            if secondary_tree.get("feature_mode") != tree.get("feature_mode"):
                raise ValueError("residual-forward feature modes differ")
            secondary_models = list(secondary_tree["models"])
            if complexity_blends is not None:
                if declared_sources is not None:
                    raise ValueError(
                        "complexity mixtures cannot declare field sources"
                    )
                self.secondary_models = secondary_models
                primary_weights = np.zeros(
                    (5, len(STATE_FIELDS)),
                    dtype=np.float32,
                )
                secondary_weights = np.zeros_like(primary_weights)
                if set(complexity_blends) != {"1", "2", "3", "4"}:
                    raise ValueError(
                        "complexity residual-forward bins differ"
                    )
                for complexity in range(1, 5):
                    declared_fields = complexity_blends[str(complexity)]
                    if set(declared_fields) != set(STATE_FIELDS):
                        raise ValueError(
                            "complexity residual-forward fields differ"
                        )
                    for field_index, field in enumerate(STATE_FIELDS):
                        weights = declared_fields[field]
                        if set(weights) != {"primary", "secondary"}:
                            raise ValueError(
                                "complexity residual-forward sources differ"
                            )
                        primary_weights[complexity, field_index] = float(
                            weights["primary"]
                        )
                        secondary_weights[complexity, field_index] = float(
                            weights["secondary"]
                        )
                self.complexity_primary_blend = primary_weights
                self.complexity_secondary_blend = secondary_weights
            elif declared_sources is None:
                raise ValueError("residual-forward field sources are missing")
            elif set(declared_sources) != set(STATE_FIELDS):
                raise ValueError("residual-forward field sources differ")
            else:
                for index, field in enumerate(STATE_FIELDS):
                    source = str(declared_sources[field])
                    if source == "secondary":
                        self.models[index] = secondary_models[index]
                    elif source != "primary":
                        raise ValueError(
                            f"unsupported residual-forward source: {source}"
                        )
        elif complexity_blends is not None:
            raise ValueError("complexity mixtures require a secondary tree")
        elif declared_sources is not None and any(
            str(declared_sources[field]) != "primary"
            for field in STATE_FIELDS
        ):
            raise ValueError("secondary forward source has no secondary tree")
        if len(self.models) != len(STATE_FIELDS):
            raise ValueError("residual-forward tree must contain five heads")
        declared = artifact["field_blend"]
        if set(declared) != set(STATE_FIELDS):
            raise ValueError("residual-forward field blends differ")
        self.field_blend = np.asarray(
            [float(declared[field]) for field in STATE_FIELDS],
            dtype=np.float32,
        )
        self.version = str(artifact["version"])

    @staticmethod
    def _features(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
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
        group_batch: int = 512,
    ) -> np.ndarray:
        del group_batch
        if not rows:
            return np.empty((0, len(ACTION_GRID), 5), dtype=np.float32)
        features = self._features(rows)
        prior = self.base.predict_changes(rows)
        flat = np.concatenate(
            [
                features.reshape(-1, features.shape[-1]),
                prior.reshape(-1, prior.shape[-1]),
            ],
            axis=1,
        )
        residual = np.stack(
            [model.predict(flat) for model in self.models],
            axis=1,
        ).reshape(prior.shape)
        if self.complexity_primary_blend is None:
            prediction = (
                prior
                + residual.astype(np.float32)
                * self.field_blend[None, None, :]
            )
        else:
            assert self.secondary_models is not None
            assert self.complexity_secondary_blend is not None
            secondary_residual = np.stack(
                [model.predict(flat) for model in self.secondary_models],
                axis=1,
            ).reshape(prior.shape)
            complexity = np.asarray(
                [
                    sum(
                        abs(float(action[field])) > 0.0
                        for field in ACTION_FIELDS
                    )
                    for action in ACTION_GRID
                ],
                dtype=np.int64,
            )
            primary_weight = self.complexity_primary_blend[complexity]
            secondary_weight = self.complexity_secondary_blend[complexity]
            prediction = (
                prior
                + residual.astype(np.float32) * primary_weight[None, :, :]
                + secondary_residual.astype(np.float32)
                * secondary_weight[None, :, :]
            )
        prediction[:, ZERO_ACTION_INDEX, :] = 0.0
        return prediction.astype(np.float32)

    def predict_states(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 512,
    ) -> np.ndarray:
        changes = self.predict_changes(rows, group_batch=group_batch)
        current = np.asarray(
            [
                raw_state_array(row["current_beam_state"])
                for row in rows
            ],
            dtype=np.float32,
        )
        tolerance = np.asarray(
            [
                tolerance_from_current(row["current_beam_state"])
                for row in rows
            ],
            dtype=np.float32,
        )
        return current[:, None, :] + changes * tolerance[:, None, :]


class ForwardExpertSelectorRuntimeV9:
    """Select a calibrated forward expert independently for each grid action."""

    def __init__(
        self,
        torch: Any,
        artifact_path: Path,
        route_key: str,
        device: Any,
        current: ResidualForwardRuntimeV9 | None = None,
    ) -> None:
        path = artifact_path.resolve()
        with path.open("rb") as stream:
            artifact = pickle.load(stream)
        if artifact.get("model") != (
            "route_restricted_hgb_forward_expert_selector_v9"
        ):
            raise ValueError("unexpected forward-expert selector artifact")
        if route_key not in {"state", "image"}:
            raise ValueError(f"unsupported forward selector route: {route_key}")
        current_path = (
            Path(str(artifact["current_dir"])).resolve()
            / f"forward_{route_key}.pkl"
        )
        alternative_path = (
            Path(str(artifact["alternative_dir"])).resolve()
            / f"forward_{route_key}.pkl"
        )
        if sha256(current_path) != artifact[f"current_{route_key}_sha256"]:
            raise ValueError("forward selector current checksum differs")
        if (
            sha256(alternative_path)
            != artifact[f"alternative_{route_key}_sha256"]
        ):
            raise ValueError("forward selector alternative checksum differs")
        if current is None:
            current, _ = load_residual_forward_runtime_v9(
                current_path,
                torch,
                device,
            )
        self.current = current
        self.alternative, _ = load_residual_forward_runtime_v9(
            alternative_path,
            torch,
            device,
        )
        route = artifact["routes"][route_key]
        self.enabled = bool(route["enabled"])
        self.classifier = route["classifier"]
        self.threshold = float(route["threshold"])
        self.base = self.current.base
        self.version = str(artifact["version"])

    def predict_changes(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 512,
    ) -> np.ndarray:
        if not rows:
            return np.empty((0, len(ACTION_GRID), 5), dtype=np.float32)
        current = self.current.predict_changes(rows, group_batch=group_batch)
        if not self.enabled:
            return current
        alternative = self.alternative.predict_changes(
            rows,
            group_batch=group_batch,
        )
        prior = self.base.predict_changes(rows)
        flat = np.concatenate(
            [
                prior.reshape(-1, prior.shape[-1]),
                current.reshape(-1, current.shape[-1]),
                alternative.reshape(-1, alternative.shape[-1]),
                np.abs(current - alternative).reshape(
                    -1,
                    current.shape[-1],
                ),
            ],
            axis=1,
        ).astype(np.float32)
        probability = self.classifier.predict_proba(flat)[:, 1]
        choose_alternative = (
            probability.reshape(current.shape[:2]) >= self.threshold
        )
        return np.where(
            choose_alternative[:, :, None],
            alternative,
            current,
        ).astype(np.float32)

    def predict_states(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 512,
    ) -> np.ndarray:
        changes = self.predict_changes(rows, group_batch=group_batch)
        current = np.asarray(
            [
                raw_state_array(row["current_beam_state"])
                for row in rows
            ],
            dtype=np.float32,
        )
        tolerance = np.asarray(
            [
                tolerance_from_current(row["current_beam_state"])
                for row in rows
            ],
            dtype=np.float32,
        )
        return current[:, None, :] + changes * tolerance[:, None, :]


class ForwardSelectorEnsembleRuntimeV9:
    """Select between two independently protected forward selectors."""

    def __init__(
        self,
        torch: Any,
        artifact_path: Path,
        device: Any,
    ) -> None:
        path = artifact_path.resolve()
        with path.open("rb") as stream:
            artifact = pickle.load(stream)
        if artifact.get("model") != "hgb_forward_selector_ensemble_v9":
            raise ValueError("unexpected forward-selector ensemble artifact")
        primary_path = Path(str(artifact["primary_selector"])).resolve()
        secondary_path = Path(str(artifact["secondary_selector"])).resolve()
        if sha256(primary_path) != artifact["primary_selector_sha256"]:
            raise ValueError("primary forward-selector checksum differs")
        if sha256(secondary_path) != artifact["secondary_selector_sha256"]:
            raise ValueError("secondary forward-selector checksum differs")
        self.primary = ForwardExpertSelectorRuntimeV9(
            torch,
            primary_path,
            "state",
            device,
        )
        self.secondary = ForwardExpertSelectorRuntimeV9(
            torch,
            secondary_path,
            "state",
            device,
        )
        self.classifier = artifact["classifier"]
        self.threshold = float(artifact["threshold"])
        self.base = self.primary.base
        self.version = str(artifact["version"])

    def predict_changes(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 512,
    ) -> np.ndarray:
        if not rows:
            return np.empty((0, len(ACTION_GRID), 5), dtype=np.float32)
        primary = self.primary.predict_changes(
            rows,
            group_batch=group_batch,
        )
        secondary = self.secondary.predict_changes(
            rows,
            group_batch=group_batch,
        )
        prior = self.base.predict_changes(rows)
        flat = np.concatenate(
            [
                prior.reshape(-1, prior.shape[-1]),
                primary.reshape(-1, primary.shape[-1]),
                secondary.reshape(-1, secondary.shape[-1]),
                np.abs(primary - secondary).reshape(
                    -1,
                    primary.shape[-1],
                ),
            ],
            axis=1,
        ).astype(np.float32)
        probability = self.classifier.predict_proba(flat)[:, 1]
        choose_secondary = (
            probability.reshape(primary.shape[:2]) >= self.threshold
        )
        return np.where(
            choose_secondary[:, :, None],
            secondary,
            primary,
        ).astype(np.float32)

    def predict_states(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 512,
    ) -> np.ndarray:
        changes = self.predict_changes(rows, group_batch=group_batch)
        current = np.asarray(
            [
                raw_state_array(row["current_beam_state"])
                for row in rows
            ],
            dtype=np.float32,
        )
        tolerance = np.asarray(
            [
                tolerance_from_current(row["current_beam_state"])
                for row in rows
            ],
            dtype=np.float32,
        )
        return current[:, None, :] + changes * tolerance[:, None, :]


class ForwardSelectorExtensionRuntimeV9:
    """Select conservatively between the current ensemble and one new expert."""

    def __init__(
        self,
        torch: Any,
        artifact_path: Path,
        device: Any,
    ) -> None:
        path = artifact_path.resolve()
        with path.open("rb") as stream:
            artifact = pickle.load(stream)
        if artifact.get("model") != "hgb_forward_selector_extension_v9":
            raise ValueError("unexpected forward-selector extension artifact")
        current_path = Path(str(artifact["current_selector"])).resolve()
        candidate_path = Path(str(artifact["candidate_artifact"])).resolve()
        if sha256(current_path) != artifact["current_selector_sha256"]:
            raise ValueError("forward extension current checksum differs")
        if sha256(candidate_path) != artifact["candidate_artifact_sha256"]:
            raise ValueError("forward extension candidate checksum differs")
        self.current, _ = load_forward_selector_ensemble_runtime_v9(
            current_path,
            torch,
            device,
        )
        self.candidate, _ = load_residual_forward_runtime_v9(
            candidate_path,
            torch,
            device,
        )
        self.classifier = artifact["classifier"]
        self.threshold = float(artifact["threshold"])
        self.physical_feature_count = int(artifact["physical_feature_count"])
        self.base = self.current.base
        self.version = str(artifact["version"])

    def predict_changes(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 512,
    ) -> np.ndarray:
        if not rows:
            return np.empty((0, len(ACTION_GRID), 5), dtype=np.float32)
        current = self.current.predict_changes(
            rows,
            group_batch=group_batch,
        )
        candidate = self.candidate.predict_changes(
            rows,
            group_batch=group_batch,
        )
        prior = self.base.predict_changes(rows)
        engineered = ResidualForwardRuntimeV9._features(rows)
        physical = np.concatenate([engineered, prior], axis=2)
        if physical.shape[2] != self.physical_feature_count:
            raise ValueError("forward extension physical feature count differs")
        flat = np.concatenate(
            [
                physical.reshape(-1, physical.shape[-1]),
                prior.reshape(-1, prior.shape[-1]),
                current.reshape(-1, current.shape[-1]),
                candidate.reshape(-1, candidate.shape[-1]),
                np.abs(candidate - current).reshape(
                    -1,
                    current.shape[-1],
                ),
            ],
            axis=1,
        ).astype(np.float32)
        probability = self.classifier.predict_proba(flat)[:, 1]
        choose_candidate = (
            probability.reshape(current.shape[:2]) >= self.threshold
        )
        return np.where(
            choose_candidate[:, :, None],
            candidate,
            current,
        ).astype(np.float32)

    def predict_states(
        self,
        rows: Sequence[Mapping[str, Any]],
        group_batch: int = 512,
    ) -> np.ndarray:
        changes = self.predict_changes(rows, group_batch=group_batch)
        current = np.asarray(
            [
                raw_state_array(row["current_beam_state"])
                for row in rows
            ],
            dtype=np.float32,
        )
        tolerance = np.asarray(
            [
                tolerance_from_current(row["current_beam_state"])
                for row in rows
            ],
            dtype=np.float32,
        )
        return current[:, None, :] + changes * tolerance[:, None, :]


def load_residual_forward_runtime_v9(
    artifact_path: Path,
    torch: Any,
    device: Any,
) -> tuple[ResidualForwardRuntimeV9, dict[str, Any]]:
    path = artifact_path.resolve()
    with path.open("rb") as stream:
        artifact = pickle.load(stream)
    return ResidualForwardRuntimeV9(torch, path, device), artifact


def load_forward_expert_selector_runtime_v9(
    artifact_path: Path,
    route_key: str,
    torch: Any,
    device: Any,
    current: ResidualForwardRuntimeV9 | None = None,
) -> tuple[ForwardExpertSelectorRuntimeV9, dict[str, Any]]:
    path = artifact_path.resolve()
    with path.open("rb") as stream:
        artifact = pickle.load(stream)
    return (
        ForwardExpertSelectorRuntimeV9(
            torch,
            path,
            route_key,
            device,
            current=current,
        ),
        artifact,
    )


def load_forward_selector_ensemble_runtime_v9(
    artifact_path: Path,
    torch: Any,
    device: Any,
) -> tuple[ForwardSelectorEnsembleRuntimeV9, dict[str, Any]]:
    path = artifact_path.resolve()
    with path.open("rb") as stream:
        artifact = pickle.load(stream)
    return ForwardSelectorEnsembleRuntimeV9(torch, path, device), artifact


def load_forward_selector_extension_runtime_v9(
    artifact_path: Path,
    torch: Any,
    device: Any,
) -> tuple[ForwardSelectorExtensionRuntimeV9, dict[str, Any]]:
    path = artifact_path.resolve()
    with path.open("rb") as stream:
        artifact = pickle.load(stream)
    return ForwardSelectorExtensionRuntimeV9(
        torch,
        path,
        device,
    ), artifact
