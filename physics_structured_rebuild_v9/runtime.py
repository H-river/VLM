"""Inference overlays for the promoted boundary-only direction hybrid."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from control_rebuild_v5.forward_runtime import load_forward_runtime_v5
from control_rebuild_v5.inverse_runtime import load_inverse_runtime_v5
from direction_rebuild_v4.data import CLASSES, labels_from_normalized_change
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from physics_structured_rebuild_v9.inverse_runtime_v8 import (
    load_inverse_runtime_v8,
)
from physics_structured_rebuild_v9.inverse_selector_runtime import (
    InverseEnumeratorSelectorRuntimeV9,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_expert_selector_runtime_v9,
    load_forward_selector_ensemble_runtime_v9,
    load_residual_forward_runtime_v9,
)
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    DIRECTION_FIELDS,
    STATE_FIELDS,
    forward_feature,
)

DEFAULT_HYBRID = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed/hybrid_direction"
    / "hybrid_direction.pkl"
)
SYSTEM_AWARE_HYBRID_DIR = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "hybrid_direction_system_aware"
)
DEFAULT_SYSTEM_AWARE_STATE_HYBRID = (
    SYSTEM_AWARE_HYBRID_DIR / "hybrid_direction_state.pkl"
)
DEFAULT_SYSTEM_AWARE_IMAGE_HYBRID = (
    SYSTEM_AWARE_HYBRID_DIR / "hybrid_direction_image.pkl"
)
TARGETED_TREE_HYBRID_DIR = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "hybrid_direction_targeted_tree"
)
DEFAULT_TARGETED_TREE_STATE_HYBRID = (
    TARGETED_TREE_HYBRID_DIR / "hybrid_direction_state.pkl"
)
DEFAULT_TARGETED_TREE_IMAGE_HYBRID = (
    TARGETED_TREE_HYBRID_DIR / "hybrid_direction_image.pkl"
)
DEFAULT_TRANSFORMER_INVERSE_V8 = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/tabm_transformer_rebuild_v8_one_seed"
    / "transformer/inverse.pt"
)
DEFAULT_COMBINED_NATURAL_INVERSE_V9 = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "combined_natural_inverse_ranker_adaptation/inverse.pt"
)
DEFAULT_INVERSE_ENUMERATOR_SELECTOR_V9 = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "inverse_enumerator_selector_v9.json"
)
DEFAULT_LEAKAGE_FREE_INVERSE_SELECTOR_V9 = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "inverse_selector_leakage_free_hgb_v9.pkl"
)
FIELDWISE_HYBRID_DIR = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "hybrid_direction_fieldwise"
)
DEFAULT_FIELDWISE_STATE_HYBRID = (
    FIELDWISE_HYBRID_DIR / "hybrid_direction_state.pkl"
)
DEFAULT_FIELDWISE_IMAGE_HYBRID = (
    FIELDWISE_HYBRID_DIR / "hybrid_direction_image.pkl"
)
FIELD_MIX_HYBRID_DIR = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "hybrid_direction_field_mix"
)
DEFAULT_FIELD_MIX_STATE_HYBRID = (
    FIELD_MIX_HYBRID_DIR / "hybrid_direction_state.pkl"
)
DEFAULT_FIELD_MIX_IMAGE_HYBRID = (
    FIELD_MIX_HYBRID_DIR / "hybrid_direction_image.pkl"
)
DUAL_SOURCE_HYBRID_DIR = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "hybrid_direction_dual_source_fieldwise"
)
DEFAULT_DUAL_SOURCE_STATE_HYBRID = (
    DUAL_SOURCE_HYBRID_DIR / "hybrid_direction_state.pkl"
)
DEFAULT_DUAL_SOURCE_IMAGE_HYBRID = (
    DUAL_SOURCE_HYBRID_DIR / "hybrid_direction_image.pkl"
)
EXTRA_FORWARD_DUAL_TREE_HYBRID_DIR = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "hybrid_direction_extra_forward_dual_tree"
)
DEFAULT_EXTRA_FORWARD_DIRECTION_STATE_HYBRID = (
    EXTRA_FORWARD_DUAL_TREE_HYBRID_DIR / "hybrid_direction_state.pkl"
)
DEFAULT_EXTRA_FORWARD_DIRECTION_IMAGE_HYBRID = (
    EXTRA_FORWARD_DUAL_TREE_HYBRID_DIR / "hybrid_direction_image.pkl"
)
RESIDUAL_FORWARD_DIR = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "forward_residual_calibrated"
)
DEFAULT_RESIDUAL_FORWARD_STATE = (
    RESIDUAL_FORWARD_DIR / "forward_state.pkl"
)
DEFAULT_RESIDUAL_FORWARD_IMAGE = (
    RESIDUAL_FORWARD_DIR / "forward_image.pkl"
)
EXTRA_TREES_FORWARD_DIR = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "forward_extra_trees_calibrated"
)
DEFAULT_EXTRA_TREES_FORWARD_STATE = (
    EXTRA_TREES_FORWARD_DIR / "forward_state.pkl"
)
DEFAULT_EXTRA_TREES_FORWARD_IMAGE = (
    EXTRA_TREES_FORWARD_DIR / "forward_image.pkl"
)
DUAL_SOURCE_FORWARD_DIR = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "forward_dual_source_calibrated"
)
DEFAULT_DUAL_SOURCE_FORWARD_STATE = (
    DUAL_SOURCE_FORWARD_DIR / "forward_state.pkl"
)
DEFAULT_DUAL_SOURCE_FORWARD_IMAGE = (
    DUAL_SOURCE_FORWARD_DIR / "forward_image.pkl"
)
NATURAL_GRID_FORWARD_DIR = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "forward_natural_grid_extra_calibrated"
)
DEFAULT_NATURAL_GRID_FORWARD_STATE = (
    NATURAL_GRID_FORWARD_DIR / "forward_state.pkl"
)
DEFAULT_NATURAL_GRID_FORWARD_IMAGE = (
    NATURAL_GRID_FORWARD_DIR / "forward_image.pkl"
)
DEFAULT_RESTRICTED_FORWARD_SELECTOR_V9 = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "forward_natural_expert_selector_restricted_v9.pkl"
)
DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9 = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "forward_selector_ensemble_v9.pkl"
)
DEFAULT_COMPLEXITY_MIX_FORWARD = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "forward_complexity_mix/forward_complexity_mix.pkl"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HybridDirectionRuntimeV9:
    """Use v7 thresholds except for confident near-boundary tree corrections."""

    def __init__(
        self,
        torch: Any,
        artifact_path: Path,
        device: Any,
    ) -> None:
        path = artifact_path.resolve()
        with path.open("rb") as stream:
            artifact = pickle.load(stream)
        if artifact.get("model") != "frozen_v7_forward_plus_boundary_tree_v4":
            raise ValueError("unexpected hybrid-direction artifact")
        forward_path = Path(str(artifact["forward_artifact"])).resolve()
        tree_path = Path(str(artifact["tree_artifact"])).resolve()
        if sha256(forward_path) != artifact["forward_artifact_sha256"]:
            raise ValueError("hybrid forward checksum differs")
        if sha256(tree_path) != artifact["tree_artifact_sha256"]:
            raise ValueError("hybrid tree checksum differs")
        with tree_path.open("rb") as stream:
            tree = pickle.load(stream)
        self.shared, _ = load_forward_direction_runtime_v7(
            forward_path,
            torch,
            device,
        )
        self.models = list(tree["models"])
        self.tree_feature_mode = str(
            tree.get("feature_mode", "engineered_46")
        )
        if self.tree_feature_mode not in {
            "engineered_46",
            "engineered_46_plus_forward_v7_change_5",
            (
                "engineered_46_plus_forward_v7_change_5"
                "_plus_residual_forward_5"
            ),
        }:
            raise ValueError("unsupported hybrid tree feature mode")
        secondary_tree_path = artifact.get("secondary_tree_artifact")
        declared_tree_sources = artifact.get("field_tree_source")
        if secondary_tree_path is not None:
            secondary_path = Path(str(secondary_tree_path)).resolve()
            if (
                sha256(secondary_path)
                != artifact["secondary_tree_artifact_sha256"]
            ):
                raise ValueError("hybrid secondary-tree checksum differs")
            with secondary_path.open("rb") as stream:
                secondary_tree = pickle.load(stream)
            if (
                str(secondary_tree.get("feature_mode", "engineered_46"))
                != self.tree_feature_mode
            ):
                raise ValueError("hybrid direction-tree feature modes differ")
            secondary_models = list(secondary_tree["models"])
            if declared_tree_sources is None:
                raise ValueError("hybrid field tree sources are missing")
            if set(declared_tree_sources) != set(DIRECTION_FIELDS):
                raise ValueError(
                    "hybrid field-tree-source keys differ from direction fields"
                )
            for index, field in enumerate(DIRECTION_FIELDS):
                source = str(declared_tree_sources[field])
                if source == "secondary":
                    self.models[index] = secondary_models[index]
                elif source != "primary":
                    raise ValueError(
                        f"unsupported hybrid field tree source: {source}"
                    )
        elif declared_tree_sources is not None and any(
            str(declared_tree_sources[field]) != "primary"
            for field in DIRECTION_FIELDS
        ):
            raise ValueError("secondary field source has no secondary tree")
        self.residual_forward_models = None
        residual_forward_tree_path = tree.get(
            "residual_forward_tree_artifact"
        )
        if residual_forward_tree_path is not None:
            residual_tree_path = Path(
                str(residual_forward_tree_path)
            ).resolve()
            expected_residual_sha = tree.get(
                "residual_forward_tree_artifact_sha256"
            )
            if (
                expected_residual_sha is not None
                and sha256(residual_tree_path) != expected_residual_sha
            ):
                raise ValueError(
                    "hybrid residual-forward tree checksum differs"
                )
            with residual_tree_path.open("rb") as stream:
                residual_tree = pickle.load(stream)
            self.residual_forward_models = list(
                residual_tree["models"]
            )
        threshold_forward_path = artifact.get("threshold_forward_artifact")
        if threshold_forward_path is None:
            self.threshold_forward = self.shared
        else:
            threshold_path = Path(str(threshold_forward_path)).resolve()
            if (
                sha256(threshold_path)
                != artifact["threshold_forward_artifact_sha256"]
            ):
                raise ValueError(
                    "hybrid threshold-forward checksum differs"
                )
            self.threshold_forward, _ = (
                load_residual_forward_runtime_v9(
                    threshold_path,
                    torch,
                    device,
                )
            )
        self.calibration = dict(artifact["calibration"])
        declared_field_calibration = artifact.get("field_calibration")
        if declared_field_calibration is None:
            self.field_calibration = [
                dict(self.calibration) for _ in DIRECTION_FIELDS
            ]
        else:
            if set(declared_field_calibration) != set(DIRECTION_FIELDS):
                raise ValueError(
                    "hybrid field-calibration keys differ from direction fields"
                )
            self.field_calibration = [
                dict(declared_field_calibration[field])
                for field in DIRECTION_FIELDS
            ]
        self.version = str(artifact["version"])

    def predict_one(
        self,
        setup: Mapping[str, Any],
        current: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> dict[str, Any]:
        action_index = next(
            (
                index
                for index, candidate in enumerate(ACTION_GRID)
                if all(
                    float(action[field]) == float(candidate[field])
                    for field in ACTION_FIELDS
                )
            ),
            None,
        )
        if action_index is None:
            raise ValueError("hybrid direction accepts only the 81-action grid")
        row = {
            "group_id": "hybrid_direction_v9",
            "setup": dict(setup),
            "current_beam_state": dict(current),
        }
        v7_changes = self.shared.predict_changes([row])[0, action_index]
        changes = self.threshold_forward.predict_changes([row])[
            0,
            action_index,
        ]
        threshold = labels_from_normalized_change(changes)
        features = forward_feature(setup, current, action)[None, :]
        if self.tree_feature_mode != "engineered_46":
            features = np.concatenate(
                [features, v7_changes[None, :].astype(np.float32)],
                axis=1,
            )
        if (
            self.tree_feature_mode
            == (
                "engineered_46_plus_forward_v7_change_5"
                "_plus_residual_forward_5"
            )
        ):
            if self.residual_forward_models is None:
                raise ValueError(
                    "hybrid residual-forward models are unavailable"
                )
            residual = np.stack(
                [
                    model.predict(features)[0]
                    for model in self.residual_forward_models
                ]
            ).astype(np.float32)
            features = np.concatenate(
                [features, residual[None, :]],
                axis=1,
            )
        probabilities = np.zeros((5, 3), dtype=np.float32)
        for field, model in enumerate(self.models):
            predicted = model.predict_proba(features)[0]
            probabilities[
                field,
                np.asarray(model.classes_, dtype=np.int64),
            ] = predicted
        ordered = np.sort(probabilities, axis=1)
        tree_class = probabilities.argmax(axis=1)
        distance = np.abs(np.abs(changes) - 1.0)
        boundary_limit = np.asarray(
            [
                calibration["boundary_limit"]
                for calibration in self.field_calibration
            ],
            dtype=np.float32,
        )
        confidence_limit = np.asarray(
            [
                calibration["confidence_limit"]
                for calibration in self.field_calibration
            ],
            dtype=np.float32,
        )
        margin_limit = np.asarray(
            [
                calibration["margin_limit"]
                for calibration in self.field_calibration
            ],
            dtype=np.float32,
        )
        apply = (
            (distance <= boundary_limit)
            & (ordered[:, -1] >= confidence_limit)
            & (
                ordered[:, -1] - ordered[:, -2]
                >= margin_limit
            )
            & (tree_class != threshold)
        )
        predicted = np.where(apply, tree_class, threshold).astype(np.int64)
        return {
            "directions": {
                field: CLASSES[int(predicted[index])]
                for index, field in enumerate(DIRECTION_FIELDS)
            },
            "normalized_change": {
                field: float(changes[index])
                for index, field in enumerate(STATE_FIELDS)
            },
            "tree_correction_applied": {
                field: bool(apply[index])
                for index, field in enumerate(DIRECTION_FIELDS)
            },
            "model_version": self.version,
            "simulator_at_inference": False,
        }


class OrchestratedHybridRuntimeV9:
    """Replace only direction routes while preserving frozen v7/v5 backends."""

    FORWARD_ROUTES = {
        "predict_forward_from_state_v1",
        "predict_forward_from_image_v1",
    }
    DIRECTION_ROUTES = {
        "predict_direction_from_state_v1",
        "predict_direction_from_image_v1",
    }

    def __init__(
        self,
        torch: Any,
        backend: OrchestratedSpecialistRuntimeV4,
        shared_artifact: Path,
        v5_forward_artifact: Path,
        v5_inverse_artifact: Path,
        device: Any,
    ) -> None:
        self.backend = backend
        self.shared, _ = load_forward_direction_runtime_v7(
            shared_artifact,
            torch,
            device,
        )
        self.hybrid_direction = HybridDirectionRuntimeV9(
            torch,
            DEFAULT_HYBRID,
            device,
        )
        self.v5_forward, _ = load_forward_runtime_v5(v5_forward_artifact)
        self.v5_inverse, _ = load_inverse_runtime_v5(v5_inverse_artifact)

    def dispatch(
        self,
        decision: Mapping[str, Any],
        available_images: Mapping[str, str | Path] | None = None,
    ) -> dict[str, Any]:
        route = str(decision.get("route_name"))
        if route in self.FORWARD_ROUTES:
            self.backend.forward = self.shared
        elif route in self.DIRECTION_ROUTES:
            self.backend.direction = self.hybrid_direction
        elif route == "select_inverse_action_from_states_v1":
            self.backend.forward = self.v5_forward
            self.backend.inverse = self.v5_inverse
        return self.backend.dispatch(decision, available_images)


class OrchestratedSystemAwareHybridRuntimeV9(OrchestratedHybridRuntimeV9):
    """Use independently pinned state/image direction calibrations."""

    STATE_HYBRID_ARTIFACT = DEFAULT_SYSTEM_AWARE_STATE_HYBRID
    IMAGE_HYBRID_ARTIFACT = DEFAULT_SYSTEM_AWARE_IMAGE_HYBRID

    def __init__(
        self,
        torch: Any,
        backend: OrchestratedSpecialistRuntimeV4,
        shared_artifact: Path,
        v5_forward_artifact: Path,
        v5_inverse_artifact: Path,
        device: Any,
    ) -> None:
        super().__init__(
            torch,
            backend,
            shared_artifact,
            v5_forward_artifact,
            v5_inverse_artifact,
            device,
        )
        self.state_hybrid_direction = HybridDirectionRuntimeV9(
            torch,
            self.STATE_HYBRID_ARTIFACT,
            device,
        )
        self.image_hybrid_direction = HybridDirectionRuntimeV9(
            torch,
            self.IMAGE_HYBRID_ARTIFACT,
            device,
        )

    def dispatch(
        self,
        decision: Mapping[str, Any],
        available_images: Mapping[str, str | Path] | None = None,
    ) -> dict[str, Any]:
        route = str(decision.get("route_name"))
        if route == "predict_direction_from_state_v1":
            self.hybrid_direction = self.state_hybrid_direction
        elif route == "predict_direction_from_image_v1":
            self.hybrid_direction = self.image_hybrid_direction
        return super().dispatch(decision, available_images)


class OrchestratedTargetedTreeHybridRuntimeV9(
    OrchestratedSystemAwareHybridRuntimeV9
):
    """Use the targeted-data, v7-change-aware direction tree gates."""

    STATE_HYBRID_ARTIFACT = DEFAULT_TARGETED_TREE_STATE_HYBRID
    IMAGE_HYBRID_ARTIFACT = DEFAULT_TARGETED_TREE_IMAGE_HYBRID


class OrchestratedTargetedTreeV8InverseRuntimeV9(
    OrchestratedTargetedTreeHybridRuntimeV9
):
    """Use targeted direction plus v8 numerical-state inverse ranking."""

    def __init__(
        self,
        torch: Any,
        backend: OrchestratedSpecialistRuntimeV4,
        shared_artifact: Path,
        v5_forward_artifact: Path,
        v5_inverse_artifact: Path,
        device: Any,
    ) -> None:
        super().__init__(
            torch,
            backend,
            shared_artifact,
            v5_forward_artifact,
            v5_inverse_artifact,
            device,
        )
        self.v5_inverse, _ = load_inverse_runtime_v8(
            DEFAULT_TRANSFORMER_INVERSE_V8,
            torch,
            device,
        )


class OrchestratedFieldwiseV8InverseRuntimeV9(
    OrchestratedTargetedTreeV8InverseRuntimeV9
):
    """Use fieldwise direction gates plus v8 numerical-state inverse."""

    STATE_HYBRID_ARTIFACT = DEFAULT_FIELDWISE_STATE_HYBRID
    IMAGE_HYBRID_ARTIFACT = DEFAULT_FIELDWISE_IMAGE_HYBRID


class OrchestratedBestCycleRuntimeV9(
    OrchestratedFieldwiseV8InverseRuntimeV9
):
    """Combine fieldwise direction, residual forward, and v8 state inverse."""

    STATE_FORWARD_ARTIFACT = DEFAULT_RESIDUAL_FORWARD_STATE
    IMAGE_FORWARD_ARTIFACT = DEFAULT_RESIDUAL_FORWARD_IMAGE

    def __init__(
        self,
        torch: Any,
        backend: OrchestratedSpecialistRuntimeV4,
        shared_artifact: Path,
        v5_forward_artifact: Path,
        v5_inverse_artifact: Path,
        device: Any,
    ) -> None:
        super().__init__(
            torch,
            backend,
            shared_artifact,
            v5_forward_artifact,
            v5_inverse_artifact,
            device,
        )
        self.state_residual_forward, _ = (
            load_residual_forward_runtime_v9(
                self.STATE_FORWARD_ARTIFACT,
                torch,
                device,
            )
        )
        self.image_residual_forward, _ = (
            load_residual_forward_runtime_v9(
                self.IMAGE_FORWARD_ARTIFACT,
                torch,
                device,
            )
        )

    def dispatch(
        self,
        decision: Mapping[str, Any],
        available_images: Mapping[str, str | Path] | None = None,
    ) -> dict[str, Any]:
        route = str(decision.get("route_name"))
        if route == "predict_forward_from_state_v1":
            self.shared = self.state_residual_forward
        elif route == "predict_forward_from_image_v1":
            self.shared = self.image_residual_forward
        return super().dispatch(decision, available_images)


class OrchestratedBestForwardInverseRuntimeV9(
    OrchestratedBestCycleRuntimeV9
):
    """Also enumerate numerical inverse candidates with residual forward v9."""

    def __init__(
        self,
        torch: Any,
        backend: OrchestratedSpecialistRuntimeV4,
        shared_artifact: Path,
        v5_forward_artifact: Path,
        v5_inverse_artifact: Path,
        device: Any,
    ) -> None:
        super().__init__(
            torch,
            backend,
            shared_artifact,
            v5_forward_artifact,
            v5_inverse_artifact,
            device,
        )
        self.v5_forward = self.state_residual_forward


class OrchestratedFieldMixBestForwardInverseRuntimeV9(
    OrchestratedBestForwardInverseRuntimeV9
):
    """Use the field-mixed direction candidate in the accumulated best stack."""

    STATE_HYBRID_ARTIFACT = DEFAULT_FIELD_MIX_STATE_HYBRID
    IMAGE_HYBRID_ARTIFACT = DEFAULT_FIELD_MIX_IMAGE_HYBRID


class OrchestratedDualDirectionExtraForwardInverseRuntimeV9(
    OrchestratedBestForwardInverseRuntimeV9
):
    """Use dual-source direction and Extra Trees forward in the best stack."""

    STATE_HYBRID_ARTIFACT = DEFAULT_DUAL_SOURCE_STATE_HYBRID
    IMAGE_HYBRID_ARTIFACT = DEFAULT_DUAL_SOURCE_IMAGE_HYBRID
    STATE_FORWARD_ARTIFACT = DEFAULT_EXTRA_TREES_FORWARD_STATE
    IMAGE_FORWARD_ARTIFACT = DEFAULT_EXTRA_TREES_FORWARD_IMAGE


class OrchestratedSplitForwardInverseRuntimeV9(
    OrchestratedDualDirectionExtraForwardInverseRuntimeV9
):
    """Use Extra Trees for forward calls but histogram forward for inverse."""

    def __init__(
        self,
        torch: Any,
        backend: OrchestratedSpecialistRuntimeV4,
        shared_artifact: Path,
        v5_forward_artifact: Path,
        v5_inverse_artifact: Path,
        device: Any,
    ) -> None:
        super().__init__(
            torch,
            backend,
            shared_artifact,
            v5_forward_artifact,
            v5_inverse_artifact,
            device,
        )
        self.v5_forward, _ = load_residual_forward_runtime_v9(
            DEFAULT_RESIDUAL_FORWARD_STATE,
            torch,
            device,
        )


class OrchestratedSelectedEnumeratorRuntimeV9(
    OrchestratedSplitForwardInverseRuntimeV9
):
    """Use a protected score rule to select one of two inverse enumerators."""

    SELECTOR_ARTIFACT = DEFAULT_INVERSE_ENUMERATOR_SELECTOR_V9

    def __init__(
        self,
        torch: Any,
        backend: OrchestratedSpecialistRuntimeV4,
        shared_artifact: Path,
        v5_forward_artifact: Path,
        v5_inverse_artifact: Path,
        device: Any,
    ) -> None:
        super().__init__(
            torch,
            backend,
            shared_artifact,
            v5_forward_artifact,
            v5_inverse_artifact,
            device,
        )
        selector_path = self.SELECTOR_ARTIFACT.resolve()
        if selector_path.suffix == ".json":
            import json

            selector = json.loads(selector_path.read_text(encoding="utf-8"))
        else:
            with selector_path.open("rb") as stream:
                selector = pickle.load(stream)
        selector_model = selector.get("model")
        if selector_model not in {
            "v8_ranker_dual_forward_score_selector_v9",
            "v8_ranker_dual_forward_hgb_selector_v9",
        }:
            raise ValueError("unexpected inverse-enumerator selector")
        primary_path = Path(
            str(selector["primary_forward_artifact"])
        ).resolve()
        secondary_path = Path(
            str(selector["secondary_forward_artifact"])
        ).resolve()
        inverse_path = Path(str(selector["inverse_artifact"])).resolve()
        if (
            sha256(primary_path)
            != selector["primary_forward_artifact_sha256"]
            or sha256(secondary_path)
            != selector["secondary_forward_artifact_sha256"]
            or sha256(inverse_path)
            != selector["inverse_artifact_sha256"]
        ):
            raise ValueError("inverse-enumerator selector checksum differs")
        if primary_path != DEFAULT_RESIDUAL_FORWARD_STATE.resolve():
            raise ValueError("inverse selector primary forward differs")
        if inverse_path != DEFAULT_TRANSFORMER_INVERSE_V8.resolve():
            raise ValueError("inverse selector ranker differs")
        secondary_forward, _ = load_residual_forward_runtime_v9(
            secondary_path,
            torch,
            device,
        )
        if selector_model == "v8_ranker_dual_forward_score_selector_v9":
            self.v5_inverse = InverseEnumeratorSelectorRuntimeV9(
                self.v5_inverse,
                secondary_forward,
                threshold=float(selector["threshold"]),
            )
        else:
            self.v5_inverse = InverseEnumeratorSelectorRuntimeV9(
                self.v5_inverse,
                secondary_forward,
                classifier=selector["classifier"],
                feature_names=selector["feature_names"],
            )


class OrchestratedExtraForwardDirectionSelectedEnumeratorRuntimeV9(
    OrchestratedSelectedEnumeratorRuntimeV9
):
    """Use improved forward thresholds in direction plus inverse selection."""

    STATE_HYBRID_ARTIFACT = DEFAULT_EXTRA_FORWARD_DIRECTION_STATE_HYBRID
    IMAGE_HYBRID_ARTIFACT = DEFAULT_EXTRA_FORWARD_DIRECTION_IMAGE_HYBRID


class OrchestratedDualForwardAccumulatedRuntimeV9(
    OrchestratedExtraForwardDirectionSelectedEnumeratorRuntimeV9
):
    """Use the dual-source forward ensemble for direct forward requests."""

    STATE_FORWARD_ARTIFACT = DEFAULT_DUAL_SOURCE_FORWARD_STATE
    IMAGE_FORWARD_ARTIFACT = DEFAULT_DUAL_SOURCE_FORWARD_IMAGE


class OrchestratedLeakageFreeSelectorRuntimeV9(
    OrchestratedDualForwardAccumulatedRuntimeV9
):
    """Use the training-only learned inverse selector in the accumulated stack."""

    SELECTOR_ARTIFACT = DEFAULT_LEAKAGE_FREE_INVERSE_SELECTOR_V9


class OrchestratedNaturalGridForwardRuntimeV9(
    OrchestratedDualForwardAccumulatedRuntimeV9
):
    """Use the natural-grid augmented forward for direct forward requests."""

    STATE_FORWARD_ARTIFACT = DEFAULT_NATURAL_GRID_FORWARD_STATE
    IMAGE_FORWARD_ARTIFACT = DEFAULT_NATURAL_GRID_FORWARD_IMAGE


class OrchestratedCombinedNaturalInverseRuntimeV9(
    OrchestratedNaturalGridForwardRuntimeV9
):
    """Use the protected natural forward and disjoint-data inverse ranker."""

    INVERSE_ARTIFACT = DEFAULT_COMBINED_NATURAL_INVERSE_V9

    def __init__(
        self,
        torch: Any,
        backend: OrchestratedSpecialistRuntimeV4,
        shared_artifact: Path,
        v5_forward_artifact: Path,
        v5_inverse_artifact: Path,
        device: Any,
    ) -> None:
        super().__init__(
            torch,
            backend,
            shared_artifact,
            v5_forward_artifact,
            v5_inverse_artifact,
            device,
        )
        self.v5_forward = self.state_residual_forward
        self.v5_inverse, _ = load_inverse_runtime_v8(
            self.INVERSE_ARTIFACT,
            torch,
            device,
        )


class OrchestratedNaturalVisualForwardRuntimeV9(
    OrchestratedCombinedNaturalInverseRuntimeV9
):
    """Also use the retained natural forward inside visual inverse scoring."""

    def __init__(
        self,
        torch: Any,
        backend: OrchestratedSpecialistRuntimeV4,
        shared_artifact: Path,
        v5_forward_artifact: Path,
        v5_inverse_artifact: Path,
        device: Any,
    ) -> None:
        super().__init__(
            torch,
            backend,
            shared_artifact,
            v5_forward_artifact,
            v5_inverse_artifact,
            device,
        )
        self.backend.visual.forward = self.state_residual_forward
        self.backend.visual.inverse = self.v5_inverse


class OrchestratedRestrictedForwardSelectorRuntimeV9(
    OrchestratedNaturalVisualForwardRuntimeV9
):
    """Select the complementary forward expert only for direct state calls."""

    FORWARD_SELECTOR_ARTIFACT = DEFAULT_RESTRICTED_FORWARD_SELECTOR_V9

    def __init__(
        self,
        torch: Any,
        backend: OrchestratedSpecialistRuntimeV4,
        shared_artifact: Path,
        v5_forward_artifact: Path,
        v5_inverse_artifact: Path,
        device: Any,
    ) -> None:
        super().__init__(
            torch,
            backend,
            shared_artifact,
            v5_forward_artifact,
            v5_inverse_artifact,
            device,
        )
        self.state_residual_forward, _ = (
            load_forward_expert_selector_runtime_v9(
                self.FORWARD_SELECTOR_ARTIFACT,
                "state",
                torch,
                device,
                current=self.state_residual_forward,
            )
        )


class OrchestratedForwardSelectorEnsembleRuntimeV9(
    OrchestratedNaturalVisualForwardRuntimeV9
):
    """Use the protected selector ensemble only for direct state forward."""

    FORWARD_SELECTOR_ARTIFACT = DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9

    def __init__(
        self,
        torch: Any,
        backend: OrchestratedSpecialistRuntimeV4,
        shared_artifact: Path,
        v5_forward_artifact: Path,
        v5_inverse_artifact: Path,
        device: Any,
    ) -> None:
        super().__init__(
            torch,
            backend,
            shared_artifact,
            v5_forward_artifact,
            v5_inverse_artifact,
            device,
        )
        self.state_residual_forward, _ = (
            load_forward_selector_ensemble_runtime_v9(
                self.FORWARD_SELECTOR_ARTIFACT,
                torch,
                device,
            )
        )


class OrchestratedComplexityMixForwardRuntimeV9(
    OrchestratedDualForwardAccumulatedRuntimeV9
):
    """Use complexity-aware mixtures only for direct forward requests."""

    STATE_FORWARD_ARTIFACT = DEFAULT_COMPLEXITY_MIX_FORWARD
    IMAGE_FORWARD_ARTIFACT = DEFAULT_COMPLEXITY_MIX_FORWARD
