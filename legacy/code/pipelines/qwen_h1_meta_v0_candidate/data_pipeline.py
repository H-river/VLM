"""Frozen candidate-data generation and audit helpers for qwen_h1_meta_v0.

This module deliberately keeps evaluator-only provenance separate from the
model-visible chat payload.  It is candidate infrastructure only: importing it
does not read any dataset, start training, or run a frozen evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
PROTOCOL_ROOT = PACKAGE_ROOT / "protocol"
KNOWN_IDENTITY_REGISTRY_RELATIVE_PATH = Path(
    "qwen_h1_meta_v0_candidate/configs/known_identity_blocklist.json"
)
# This is the identity-only snapshot captured before candidate generation.  The
# preregistration freezes its cardinality but (unlike the protocol files) did
# not duplicate its digest, so the generator pins the snapshot bytes here.
KNOWN_IDENTITY_REGISTRY_SHA256 = (
    "9256e245f7f76fe81b789242526cc840d9c1378520c9ead91873ab58101b747d"
)

ACTION_FIELDS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
POSITION_FIELDS = (
    "lens_x_mm",
    "lens_y_mm",
    "camera_x_mm",
    "camera_y_mm",
)
METRIC_FIELDS = (
    "centroid_x_px",
    "centroid_y_px",
    "sigma_x_px",
    "sigma_y_px",
    "peak_intensity",
)
SETUP_COMMON_FIELDS = (
    "wavelength_nm",
    "beam_waist_mm",
    "power_w",
    "lens_focal_length_mm",
    "lens_aperture_mm",
    "source_to_lens_mm",
    "lens_to_camera_mm",
    "pixel_size_um",
)
REGIME_CYCLE = (
    "ordinary",
    "focusing",
    "clipping",
    "camera_boundary",
    "tolerance_boundary",
    "high_offset_interaction",
)
SPLIT_SPECS = {
    "train": {"count": 16, "setup_seed": 2026080207},
    "dev": {"count": 8, "setup_seed": 2026080208},
    "candidate_eval": {"count": 12, "setup_seed": 2026080209},
}
TARGET_SEED = 2026080210
ORACLE_SEED = 2026080211
SHUFFLE_SEED = 2026080213
TARGET_VARIANTS = (
    ("one_step_mixed", (0.8, -0.8, 0.75, -0.75)),
    ("multi_step_opposite", (-1.8, 1.4, 1.6, -1.2)),
    ("near_target", (0.2, 0.2, -0.2, -0.2)),
)
HISTORY_ACTION_FRACTIONS = (
    (0.25, -0.25, 0.0, 0.0),
    (0.0, 0.0, 0.25, -0.25),
    (-0.125, 0.125, -0.125, 0.125),
)
SCORE_WEIGHTS = {
    # The frozen preregistration contains both the base max-error term and an
    # additional 0.10 max-error term, hence the combined coefficient 1.10.
    "max_normalized_error": 1.10,
    "mean_normalized_error": 0.15,
    "normalized_l1_movement": 0.02,
    "clipping_fraction": 1.0,
    "boundary_indicator": 1.0,
    "mean_normalized_uncertainty": 0.10,
    "projected_or_invalid": 10.0,
}
MIN_GUIDED_IMPROVEMENT = 0.02
HIGH_GUIDED_IMPROVEMENT = 0.10

FROZEN_FILE_MANIFESTS = (
    "protocol_freeze_manifest.json",
    "protocol_addendum_freeze_manifest.json",
    "prompt_freeze_manifest.json",
)
MODEL_VISIBLE_FORBIDDEN_KEYS = {
    "setup_id",
    "group_id",
    "episode_id",
    "record_id",
    "example_id",
    "target_counterfactual_id",
    "split",
    "image_path",
    "path",
    "setup_context",
    "setup_hash",
    "context_hash",
    "generator_family",
    "q_goal_mm",
    "q_goal",
    "target_positions_mm",
    "future_state",
    "future_metrics",
    "oracle_action",
    "oracle_label",
    "rollout_success",
    "unexecuted_default_action",
    "default_h1_action",
}
MODEL_VISIBLE_FORBIDDEN_TEXT = (
    "/home/",
    "qwen_h1_meta_v0_candidate/data/",
    "qh1meta_train_",
    "qh1meta_dev_",
    "qh1meta_eval_",
)


class DataGenerationError(RuntimeError):
    """Raised when candidate generation cannot satisfy the frozen contract."""


class InformationSufficiencyError(DataGenerationError):
    """Raised after a RED information gate; SFT export must not proceed."""


class PlannerBackend(Protocol):
    """Minimal injected planner interface used by the oracle generator."""

    def plan(
        self,
        *,
        configuration: Mapping[str, Any],
        setup_context: Mapping[str, Any],
        positions_mm: np.ndarray,
        current_metrics: np.ndarray,
        target_metrics: np.ndarray,
        tolerance_reference: np.ndarray,
        seed: int,
    ) -> Mapping[str, Any]:
        """Return at least ``selected_action`` for one frozen configuration."""


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_identity_registry_path() -> Path:
    return (REPO_ROOT / KNOWN_IDENTITY_REGISTRY_RELATIVE_PATH).resolve()


def validate_identity_registry_freeze(path: Path) -> Path:
    """Reject alternate registries before opening them, then verify the freeze.

    The caller intentionally invokes this before any protocol/source reads so
    an attacker-controlled ``--identity-registry`` path is never opened.
    """

    candidate = path.expanduser().resolve(strict=False)
    official = _official_identity_registry_path()
    if candidate != official:
        raise DataGenerationError(
            "--identity-registry must equal the preregistered candidate identity "
            f"snapshot path exactly: {official}"
        )
    if not official.is_file():
        raise DataGenerationError(
            f"preregistered identity registry is missing: {official}"
        )
    actual_hash = _sha256_file(official)
    if actual_hash != KNOWN_IDENTITY_REGISTRY_SHA256:
        raise DataGenerationError(
            "preregistered identity registry SHA-256 mismatch; refusing generation"
        )
    return official


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_text(path, "".join(_compact_json(row) + "\n" for row in rows))


def verify_frozen_protocol(protocol_root: Path = PROTOCOL_ROOT) -> dict[str, str]:
    """Verify every hash named by the three freeze manifests."""

    verified: dict[str, str] = {}
    for manifest_name in FROZEN_FILE_MANIFESTS:
        manifest_path = protocol_root / manifest_name
        manifest = _read_json(manifest_path)
        declared_files = manifest.get("files")
        if declared_files is None and "file" in manifest and "sha256" in manifest:
            declared_files = {manifest["file"]: manifest["sha256"]}
        if not isinstance(declared_files, Mapping) or not declared_files:
            raise DataGenerationError(
                f"freeze manifest does not declare any files: {manifest_path}"
            )
        for relative_name, expected_hash in declared_files.items():
            if Path(str(relative_name)).name != str(relative_name):
                raise DataGenerationError(
                    f"unsafe frozen relative filename: {relative_name!r}"
                )
            target = protocol_root / relative_name
            actual_hash = _sha256_file(target)
            if actual_hash != expected_hash:
                raise DataGenerationError(
                    f"frozen protocol hash mismatch for {target}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            previous = verified.get(relative_name)
            if previous is not None and previous != actual_hash:
                raise DataGenerationError(
                    f"inconsistent frozen hashes for {relative_name}"
                )
            verified[relative_name] = actual_hash
        verified[manifest_name] = _sha256_file(manifest_path)
    return dict(sorted(verified.items()))


def _canonical_direction(direction: Mapping[str, str]) -> dict[str, str]:
    return {field: str(direction[field]) for field in ACTION_FIELDS}


def enumerate_oracle_configurations(
    local_directional_priors: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Enumerate the frozen default plus 35 guided configurations."""

    required = {"all_actuators", "lens_only", "camera_only"}
    if set(local_directional_priors) != required:
        raise DataGenerationError(
            "local directional priors must contain exactly all_actuators, "
            "lens_only, and camera_only"
        )

    configurations: list[dict[str, Any]] = [
        {
            "configuration_id": "default",
            "decision": "run_default_h1",
            "objective_profile": "balanced",
            "mask_profile": "default",
            "directional_prior": {field: "unknown" for field in ACTION_FIELDS},
            "step_scale": "default",
            "risk_mode": "standard",
            "template_family": "default",
        }
    ]

    def append_guided(
        objective: str,
        mask: str,
        scale: str,
        risk: str,
        direction: Mapping[str, str],
        family: str,
    ) -> None:
        configuration_id = (
            f"guided__{family}__objective-{objective}__mask-{mask}"
            f"__scale-{scale}__risk-{risk}"
        )
        configurations.append(
            {
                "configuration_id": configuration_id,
                "decision": "run_guided_h1",
                "objective_profile": objective,
                "mask_profile": mask,
                "directional_prior": _canonical_direction(direction),
                "step_scale": scale,
                "risk_mode": risk,
                "template_family": family,
            }
        )

    for objective in (
        "balanced",
        "centroid_priority",
        "width_priority",
        "intensity_priority",
    ):
        for scale in ("fine", "medium", "default"):
            for risk in ("conservative", "standard"):
                append_guided(
                    objective,
                    "all_actuators",
                    scale,
                    risk,
                    local_directional_priors["all_actuators"],
                    "primary_local_gradient",
                )

    for mask in ("lens_only", "camera_only"):
        for scale in ("fine", "medium"):
            for risk in ("conservative", "standard"):
                append_guided(
                    "balanced",
                    mask,
                    scale,
                    risk,
                    local_directional_priors[mask],
                    f"{mask.split('_')[0]}_local_gradient",
                )

    unknown_direction = {field: "unknown" for field in ACTION_FIELDS}
    for scale in ("fine", "medium", "default"):
        append_guided(
            "balanced",
            "default",
            scale,
            "standard",
            unknown_direction,
            "unbiased_unknown",
        )

    if len(configurations) != 36:
        raise AssertionError(f"expected 36 configurations, got {len(configurations)}")
    identifiers = [item["configuration_id"] for item in configurations]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("configuration identifiers are not unique")
    return configurations


def canonical_one_step_score(
    *,
    next_metrics: Sequence[float],
    target_metrics: Sequence[float],
    tolerance_reference: Sequence[float],
    action_mm: Sequence[float],
    action_limit_mm: Sequence[float],
    clipping_fraction: float = 0.0,
    boundary_indicator: float = 0.0,
    mean_normalized_uncertainty: float = 0.0,
    projected: bool = False,
    invalid: bool = False,
) -> tuple[float, dict[str, float]]:
    """Compute the preregistered actual/predicted one-step score."""

    next_vector = np.asarray(next_metrics, dtype=np.float64)
    target_vector = np.asarray(target_metrics, dtype=np.float64)
    tolerance_vector = np.asarray(tolerance_reference, dtype=np.float64)
    action_vector = np.asarray(action_mm, dtype=np.float64)
    action_limit_vector = np.asarray(action_limit_mm, dtype=np.float64)
    if (
        next_vector.shape != (5,)
        or target_vector.shape != (5,)
        or tolerance_vector.shape != (5,)
        or action_vector.shape != (4,)
        or action_limit_vector.shape != (4,)
    ):
        raise ValueError("canonical score received a vector with an invalid shape")
    if np.any(tolerance_vector <= 0.0) or np.any(action_limit_vector <= 0.0):
        raise ValueError("tolerance and action limits must be strictly positive")

    normalized_error = np.abs(next_vector - target_vector) / tolerance_vector
    normalized_l1_movement = float(
        np.sum(np.abs(action_vector) / action_limit_vector)
    )
    components = {
        "max_normalized_error": float(np.max(normalized_error)),
        "mean_normalized_error": float(np.mean(normalized_error)),
        "normalized_l1_movement": normalized_l1_movement,
        "clipping_fraction": float(max(0.0, clipping_fraction)),
        "boundary_indicator": float(np.clip(float(boundary_indicator), 0.0, 1.0)),
        "mean_normalized_uncertainty": float(
            max(0.0, mean_normalized_uncertainty)
        ),
        "projected_or_invalid": float(bool(projected or invalid)),
    }
    score = sum(SCORE_WEIGHTS[name] * value for name, value in components.items())
    if not math.isfinite(score):
        raise ValueError("canonical score is not finite")
    return float(score), components


def shortlist_guided_candidates(
    evaluated: Sequence[Mapping[str, Any]], size: int = 6
) -> list[dict[str, Any]]:
    """Return the lowest predicted-score guided candidates with lexical ties."""

    guided = [item for item in evaluated if item["configuration_id"] != "default"]
    if len(guided) < size:
        raise DataGenerationError(
            f"need at least {size} guided candidates, received {len(guided)}"
        )
    ranked = sorted(
        guided,
        key=lambda item: (
            float(item["predicted_score"]),
            str(item["configuration_id"]),
        ),
    )
    return [dict(item) for item in ranked[:size]]


def choose_actual_label(
    default_result: Mapping[str, Any],
    guided_results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose default or guided using actual simulator scores and frozen ties."""

    default_score = float(default_result["actual_score"])
    objective_rank = {
        "balanced": 0,
        "centroid_priority": 1,
        "width_priority": 2,
        "intensity_priority": 3,
    }
    mask_rank = {"default": 0, "all_actuators": 1, "lens_only": 2, "camera_only": 3}
    scale_rank = {"fine": 0, "medium": 1, "default": 2}
    risk_rank = {"conservative": 0, "standard": 1}
    ranked_guided = sorted(
        guided_results,
        key=lambda item: (
            float(item["actual_score"]),
            objective_rank[str(item["configuration"]["objective_profile"])],
            mask_rank[str(item["configuration"]["mask_profile"])],
            scale_rank[str(item["configuration"]["step_scale"])],
            risk_rank[str(item["configuration"]["risk_mode"])],
            str(item["configuration_id"]),
        ),
    )
    if not ranked_guided:
        raise DataGenerationError("actual-label selection received no guided result")
    best_guided = dict(ranked_guided[0])
    improvement = default_score - float(best_guided["actual_score"])
    if improvement + 1e-12 < MIN_GUIDED_IMPROVEMENT:
        selected = dict(default_result)
        selected["decision"] = "run_default_h1"
    else:
        selected = best_guided
        selected["decision"] = "run_guided_h1"
    audit = {
        "default_actual_score": default_score,
        "best_guided_actual_score": float(best_guided["actual_score"]),
        "best_guided_configuration_id": best_guided["configuration_id"],
        "guided_improvement": float(improvement),
        "minimum_guided_improvement": MIN_GUIDED_IMPROVEMENT,
        "selected_configuration_id": selected["configuration_id"],
    }
    return selected, audit


def setup_common_fingerprint(setup_context: Mapping[str, Any]) -> tuple[float, ...]:
    """Identity-only rounded common-field fingerprint (nine decimals)."""

    return tuple(round(float(setup_context[field]), 9) for field in SETUP_COMMON_FIELDS)


@dataclass(frozen=True)
class KnownIdentityRegistry:
    setup_hashes: frozenset[str]
    common_fingerprints: frozenset[tuple[float, ...]]
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> "KnownIdentityRegistry":
        payload = _read_json(path)
        if tuple(payload.get("common_field_order", ())) != SETUP_COMMON_FIELDS:
            raise DataGenerationError(
                "identity registry common-field order does not match corrected-v12"
            )
        entries = payload.get("entries", [])
        setup_hashes = frozenset(str(item["setup_hash"]) for item in entries)
        fingerprints = frozenset(
            tuple(float(value) for value in item["common_fields_rounded_9"])
            for item in entries
        )
        expected = int(payload.get("expected_distinct_count", len(entries)))
        if len(entries) != expected or len(setup_hashes) != expected:
            raise DataGenerationError(
                f"identity registry count mismatch: expected {expected}, "
                f"found {len(entries)} rows and {len(setup_hashes)} hashes"
            )
        return cls(setup_hashes, fingerprints, path)

    def overlaps(self, setup_hash_value: str, setup_context: Mapping[str, Any]) -> dict[str, bool]:
        return {
            "exact_setup_hash": setup_hash_value in self.setup_hashes,
            "rounded_common_fields": (
                setup_common_fingerprint(setup_context) in self.common_fingerprints
            ),
        }


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def validate_model_visible_state(state: Mapping[str, Any]) -> None:
    forbidden = sorted(set(_walk_keys(state)) & MODEL_VISIBLE_FORBIDDEN_KEYS)
    if forbidden:
        raise DataGenerationError(
            f"model-visible state contains forbidden keys: {forbidden}"
        )
    serialized = _compact_json(state)
    for fragment in MODEL_VISIBLE_FORBIDDEN_TEXT:
        if fragment in serialized:
            raise DataGenerationError(
                f"model-visible state contains forbidden text fragment: {fragment!r}"
            )


def build_prebuilt_chat_row(
    *,
    manifest_row: Mapping[str, Any],
    system_prompt: str,
    prompt_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one exact frozen chat row without evaluator-only provenance."""

    state = manifest_row["model_visible_input"]
    completion = manifest_row["oracle_output"]
    validate_model_visible_state(state)
    prefix = str(prompt_contract["user_text_prefix"])
    user_text = prefix + _compact_json(state)
    for fragment in MODEL_VISIBLE_FORBIDDEN_TEXT:
        if fragment in user_text:
            raise DataGenerationError(
                f"rendered user prompt contains forbidden fragment: {fragment!r}"
            )
    images = [str(manifest_row["image"]["storage_path"])]
    if len(images) != int(prompt_contract["image_placeholders"]):
        raise DataGenerationError("image count violates frozen prompt contract")
    return {
        "example_id": str(manifest_row["record_id"]),
        "split": str(manifest_row["split"]),
        "images": images,
        "prompt": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": _compact_json(completion)}],
            }
        ],
        "metadata": {
            "candidate_only": True,
            "record_id": str(manifest_row["record_id"]),
            "setup_hash": str(manifest_row["identity"]["setup_hash"]),
            "image_sha256": str(manifest_row["image"]["sha256"]),
            "oracle_configuration_id": str(
                manifest_row["oracle_audit"]["selected_configuration_id"]
            ),
        },
    }


def _round_recursive(value: Any, decimals: int) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        number = round(float(value), decimals)
        return 0.0 if number == 0.0 else number
    if isinstance(value, list):
        return [_round_recursive(item, decimals) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _round_recursive(nested, decimals)
            for key, nested in value.items()
        }
    raise TypeError(f"unsupported value in rounded signature: {type(value)!r}")


def _label_key(row: Mapping[str, Any]) -> str:
    return _compact_json(row["oracle_output"])


def _collision_statistics(
    rows: Sequence[Mapping[str, Any]], *, decimals: int | None
) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    for row in rows:
        state: Any = row["model_visible_input"]
        if decimals is not None:
            state = _round_recursive(state, decimals)
        signature_payload = {
            "image_sha256": row["image"]["sha256"],
            "state": state,
        }
        signature = hashlib.sha256(_compact_json(signature_payload).encode()).hexdigest()
        groups.setdefault(signature, []).append(_label_key(row))
    collision_groups = [labels for labels in groups.values() if len(labels) > 1]
    conflicting = [labels for labels in collision_groups if len(set(labels)) > 1]
    rate = len(conflicting) / len(collision_groups) if collision_groups else 0.0
    return {
        "unique_visible_signatures": len(groups),
        "collision_group_count": len(collision_groups),
        "conflicting_collision_group_count": len(conflicting),
        "conflicting_collision_group_rate": rate,
    }


def _flatten_numeric(value: Any, output: list[float]) -> None:
    if isinstance(value, bool):
        output.append(float(value))
    elif isinstance(value, (int, float)):
        output.append(float(value))
    elif isinstance(value, str):
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        output.append(int.from_bytes(digest[:4], "big") / 2**32)
    elif value is None:
        output.append(0.0)
    elif isinstance(value, list):
        for item in value:
            _flatten_numeric(item, output)
    elif isinstance(value, Mapping):
        for key in sorted(value):
            _flatten_numeric(value[key], output)
    else:
        raise TypeError(f"unsupported feature value {type(value)!r}")


def _visible_feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    features: list[list[float]] = []
    for row in rows:
        vector: list[float] = []
        _flatten_numeric(row["model_visible_input"], vector)
        _flatten_numeric(row["image"].get("audit_features", {}), vector)
        features.append(vector)
    widths = {len(vector) for vector in features}
    if len(widths) != 1:
        raise DataGenerationError(f"visible feature widths differ: {sorted(widths)}")
    return np.asarray(features, dtype=np.float64)


def _hidden_setup_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [float(row["evaluator_only"]["setup_context"][field]) for field in SETUP_COMMON_FIELDS]
            for row in rows
        ],
        dtype=np.float64,
    )


def _macro_f1(labels: Sequence[str], predictions: Sequence[str]) -> float:
    classes = sorted(set(labels) | set(predictions))
    scores: list[float] = []
    for label in classes:
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(labels, predictions)
        )
        false_positive = sum(
            actual != label and predicted == label
            for actual, predicted in zip(labels, predictions)
        )
        false_negative = sum(
            actual == label and predicted != label
            for actual, predicted in zip(labels, predictions)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores)) if scores else 0.0


def _grouped_nearest_centroid_f1(
    features: np.ndarray,
    labels: Sequence[str],
    groups: Sequence[str],
    folds: int = 4,
) -> dict[str, Any]:
    unique_groups = sorted(set(groups))
    if len(unique_groups) < folds:
        raise DataGenerationError("not enough setup groups for grouped classifier audit")
    assignment = {
        group: int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % folds
        for group in unique_groups
    }
    predictions: list[str] = []
    actuals: list[str] = []
    for fold in range(folds):
        train_indices = [index for index, group in enumerate(groups) if assignment[group] != fold]
        test_indices = [index for index, group in enumerate(groups) if assignment[group] == fold]
        if not train_indices or not test_indices:
            continue
        train = features[train_indices]
        mean = train.mean(axis=0)
        scale = train.std(axis=0)
        scale[scale < 1e-9] = 1.0
        train = (train - mean) / scale
        test = (features[test_indices] - mean) / scale
        classes = sorted(set(labels[index] for index in train_indices))
        centroids = {
            label: train[
                [labels[index] == label for index in train_indices]
            ].mean(axis=0)
            for label in classes
        }
        for local_index, global_index in enumerate(test_indices):
            predicted = min(
                classes,
                key=lambda label: (
                    float(np.sum((test[local_index] - centroids[label]) ** 2)),
                    label,
                ),
            )
            predictions.append(predicted)
            actuals.append(labels[global_index])
    return {
        "method": "four_fold_setup_grouped_nearest_centroid",
        "evaluated_records": len(actuals),
        "macro_f1": _macro_f1(actuals, predictions),
    }


def _near_visible_conflicts(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    features = _visible_feature_matrix(rows)
    feature_min = features.min(axis=0)
    feature_span = features.max(axis=0) - feature_min
    feature_span[feature_span < 1e-9] = 1.0
    normalized = (features - feature_min) / feature_span
    near_pairs = 0
    conflicting_pairs = 0
    minimum_conflicting_distance: float | None = None
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            distance = float(np.sqrt(np.mean((normalized[left] - normalized[right]) ** 2)))
            if distance <= threshold:
                near_pairs += 1
                if _label_key(rows[left]) != _label_key(rows[right]):
                    conflicting_pairs += 1
                    if minimum_conflicting_distance is None:
                        minimum_conflicting_distance = distance
                    else:
                        minimum_conflicting_distance = min(
                            minimum_conflicting_distance, distance
                        )
    rate = conflicting_pairs / near_pairs if near_pairs else 0.0
    return {
        "distance_definition": (
            "RMS distance after per-feature train+dev min-max scaling; "
            "the visible image is represented by scalar summaries and a "
            "16x16 spatial-mean signature"
        ),
        "threshold": threshold,
        "near_pair_count": near_pairs,
        "conflicting_near_pair_count": conflicting_pairs,
        "conflicting_near_pair_rate": rate,
        "minimum_conflicting_distance": minimum_conflicting_distance,
    }


def information_sufficiency_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    known_identity_count: int,
    identity_overlap_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the preregistered train+dev information and leakage hard gate."""

    audit_rows = [row for row in rows if row["split"] in {"train", "dev"}]
    expected_rows = SPLIT_SPECS["train"]["count"] * 3 + SPLIT_SPECS["dev"]["count"] * 3
    if len(audit_rows) != expected_rows:
        raise DataGenerationError(
            f"information audit requires all {expected_rows} train+dev rows; "
            f"received {len(audit_rows)}"
        )
    for row in audit_rows:
        validate_model_visible_state(row["model_visible_input"])

    exact = _collision_statistics(audit_rows, decimals=None)
    rounded = _collision_statistics(audit_rows, decimals=4)
    near = _near_visible_conflicts(audit_rows, threshold=0.05)
    labels = [str(row["oracle_audit"]["selected_configuration_id"]) for row in audit_rows]
    groups = [str(row["evaluator_only"]["setup_id"]) for row in audit_rows]
    visible_features = _visible_feature_matrix(audit_rows)
    hidden_features = _hidden_setup_matrix(audit_rows)
    visible_classifier = _grouped_nearest_centroid_f1(visible_features, labels, groups)
    augmented_classifier = _grouped_nearest_centroid_f1(
        np.concatenate([visible_features, hidden_features], axis=1), labels, groups
    )
    hidden_gain = augmented_classifier["macro_f1"] - visible_classifier["macro_f1"]

    red_reasons: list[str] = []
    if exact["conflicting_collision_group_count"] > 0:
        red_reasons.append("exact_visible_label_conflict")
    if rounded["conflicting_collision_group_count"] > 0:
        red_reasons.append("rounded_4dp_visible_label_conflict")
    if near["conflicting_near_pair_rate"] > 0.20:
        red_reasons.append("near_visible_conflict_rate_above_0.20")
    if hidden_gain > 0.15:
        red_reasons.append("hidden_setup_classifier_macro_f1_gain_above_0.15")
    if visible_classifier["macro_f1"] < 0.50:
        red_reasons.append("visible_classifier_macro_f1_below_0.50")
    if int(identity_overlap_audit.get("known_overlap_count", 0)) != 0:
        red_reasons.append("known_identity_overlap")
    if int(identity_overlap_audit.get("cross_split_overlap_count", 0)) != 0:
        red_reasons.append("candidate_cross_split_identity_overlap")

    return {
        "version": "qwen_h1_meta_v0_data_audit_v1",
        "candidate_only": True,
        "audited_splits": ["train", "dev"],
        "record_count": len(audit_rows),
        "known_identity_registry_count": known_identity_count,
        "identity_overlap": dict(identity_overlap_audit),
        "exact_visible_collision": exact,
        "rounded_4dp_visible_collision": rounded,
        "near_visible_collision": near,
        "visible_only_classifier": visible_classifier,
        "visible_plus_hidden_setup_classifier": augmented_classifier,
        "hidden_setup_macro_f1_gain": float(hidden_gain),
        "thresholds": {
            "exact_conflict_count_max": 0,
            "rounded_4dp_conflict_count_max": 0,
            "near_conflict_rate_max": 0.20,
            "hidden_setup_macro_f1_gain_max": 0.15,
            "visible_classifier_macro_f1_min": 0.50,
        },
        "gate": "PASS" if not red_reasons else "RED_STOP",
        "red_reasons": red_reasons,
        "sft_export_permitted": not red_reasons,
    }


def export_prebuilt_chat(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    data_audit: Mapping[str, Any],
    protocol_root: Path = PROTOCOL_ROOT,
) -> dict[str, str]:
    """Export train/dev only after the hard information gate passes."""

    if data_audit.get("gate") != "PASS" or not data_audit.get("sft_export_permitted"):
        raise InformationSufficiencyError(
            "information-sufficiency gate is RED; refusing SFT export"
        )
    audit_path = (output_root / "reports" / "data_audit.json").resolve()
    if not audit_path.is_file():
        raise InformationSufficiencyError(
            "official data_audit.json must be written before SFT export"
        )
    persisted_audit = _read_json(audit_path)
    if persisted_audit != dict(data_audit):
        raise InformationSufficiencyError(
            "persisted data audit differs from the PASS audit supplied for export"
        )
    system_prompt = (protocol_root / "system_prompt.txt").read_text(encoding="utf-8")
    prompt_contract = _read_json(protocol_root / "prompt_contract.json")
    outputs: dict[str, str] = {}
    report_records: dict[str, Any] = {}
    for split in ("train", "dev"):
        split_rows = [row for row in rows if row["split"] == split]
        expected = SPLIT_SPECS[split]["count"] * 3
        if len(split_rows) != expected:
            raise DataGenerationError(
                f"{split} export expected {expected} rows, got {len(split_rows)}"
            )
        chat_rows = [
            build_prebuilt_chat_row(
                manifest_row=row,
                system_prompt=system_prompt,
                prompt_contract=prompt_contract,
            )
            for row in split_rows
        ]
        from qwen_h1_meta_v0_candidate.training import validate_prebuilt_row

        for chat_row in chat_rows:
            validate_prebuilt_row(
                chat_row,
                allowed_splits={split},
                image_root=REPO_ROOT,
                verify_image=True,
            )
        path = output_root / f"prebuilt_chat_{split}.jsonl"
        _atomic_write_jsonl(path, chat_rows)
        outputs[split] = str(path)
        report_records[split] = {
            "path": str(path),
            "records": len(chat_rows),
            "sha256": _sha256_file(path),
        }
    report_path = output_root / "sft_export_report.json"
    _atomic_write_json(
        report_path,
        {
            "version": "qwen_h1_meta_v0_sft_export_report_v1",
            "candidate_only": True,
            "information_gate": "PASS",
            "data_audit": {
                "path": str(audit_path),
                "sha256": _sha256_file(audit_path),
            },
            "records": report_records,
        },
    )
    outputs["report"] = str(report_path)
    return outputs


def verify_source_freeze() -> dict[str, str]:
    """Verify the source/checkpoint files consumed by candidate generation."""

    protocol = _read_json(PROTOCOL_ROOT / "meta_controller_protocol.json")
    source = protocol["source_freeze"]
    declared = {
        str(source["default_h1_protocol"]): str(
            source["default_h1_protocol_sha256"]
        ),
        str(source["forward_checkpoint"]): str(source["forward_checkpoint_sha256"]),
        str(source["simulator_config"]): str(source["simulator_config_sha256"]),
        "continuous_control_v12/mpc.py": str(source["cem_source_sha256"]),
        "continuous_control_v12/world_model.py": str(source["world_model_source_sha256"]),
    }
    verified: dict[str, str] = {}
    for relative_name, expected_hash in declared.items():
        path = REPO_ROOT / relative_name
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise DataGenerationError(
                f"source freeze mismatch for {path}: expected {expected_hash}, "
                f"got {actual_hash}"
            )
        verified[relative_name] = actual_hash
    locked = _read_json(REPO_ROOT / str(source["default_h1_protocol"]))
    base_path = REPO_ROOT / str(locked["base_simulator_config"])
    base_hash = _sha256_file(base_path)
    if base_hash != str(locked["base_simulator_config_sha256"]):
        raise DataGenerationError(f"base simulator config hash mismatch: {base_path}")
    verified[str(locked["base_simulator_config"])] = base_hash
    return verified


def _metric_dict(values: Sequence[float]) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (5,):
        raise ValueError("metric vector must contain five values")
    return {field: float(vector[index]) for index, field in enumerate(METRIC_FIELDS)}


def _action_dict(values: Sequence[float]) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (4,):
        raise ValueError("action vector must contain four values")
    return {field: float(vector[index]) for index, field in enumerate(ACTION_FIELDS)}


def _position_dict(values: Sequence[float]) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (4,):
        raise ValueError("position vector must contain four values")
    return {field: float(vector[index]) for index, field in enumerate(POSITION_FIELDS)}


def _metric_vector(values: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
    if isinstance(values, Mapping):
        return np.asarray([float(values[field]) for field in METRIC_FIELDS], dtype=np.float64)
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (5,):
        raise ValueError("metric vector must contain five values")
    return vector


def _action_vector(values: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
    if isinstance(values, Mapping):
        return np.asarray([float(values[field]) for field in ACTION_FIELDS], dtype=np.float64)
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (4,):
        raise ValueError("action vector must contain four values")
    return vector


def _forward_prediction(
    model: Any,
    *,
    setup_context: Mapping[str, Any],
    positions_mm: Sequence[float],
    current_metrics: Sequence[float],
    action_mm: Sequence[float],
) -> dict[str, Any]:
    prediction = model.predict(
        setup_context,
        np.asarray(positions_mm, dtype=np.float64),
        np.asarray(current_metrics, dtype=np.float64),
        np.asarray(action_mm, dtype=np.float64)[None, :],
    )
    auxiliary = {
        key: float(np.asarray(value).reshape(-1)[0])
        for key, value in prediction["auxiliary_predictions"].items()
    }
    return {
        "predicted_next_metrics": np.asarray(
            prediction["predicted_next_metrics"], dtype=np.float64
        )[0],
        "uncertainty": np.asarray(prediction["uncertainty"], dtype=np.float64)[0],
        "auxiliary_predictions": auxiliary,
    }


def _configuration_output(
    configuration: Mapping[str, Any],
    *,
    confidence: str,
    reason_codes: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": "qwen_h1_meta_v0",
        "decision": configuration["decision"],
        "observation_request": "reuse_current",
        "objective_profile": configuration["objective_profile"],
        "mask_profile": configuration["mask_profile"],
        "directional_prior": dict(configuration["directional_prior"]),
        "step_scale": configuration["step_scale"],
        "risk_mode": configuration["risk_mode"],
        "confidence": confidence,
        "reason_codes": list(dict.fromkeys(reason_codes))[:4],
    }


class FrozenH1PlannerBackend:
    """Adapter over the frozen default CEM and candidate guided-H1 hook."""

    def __init__(self, *, model: Any, bounds: Any) -> None:
        from qwen_h1_meta_v0_candidate.compiler import default_h1_config

        self.model = model
        self.bounds = bounds
        self.default_config = default_h1_config()

    def plan(
        self,
        *,
        configuration: Mapping[str, Any],
        setup_context: Mapping[str, Any],
        positions_mm: np.ndarray,
        current_metrics: np.ndarray,
        target_metrics: np.ndarray,
        tolerance_reference: np.ndarray,
        seed: int,
    ) -> Mapping[str, Any]:
        from continuous_control_v12.mpc import CEMMPC, learned_predictor

        predictor = learned_predictor(self.model, setup_context)
        if configuration["decision"] == "run_default_h1":
            planner = CEMMPC(
                bounds=self.bounds,
                predictor=predictor,
                config=self.default_config,
                seed=seed,
            )
            return planner.plan(
                positions_mm=positions_mm,
                current_metrics=current_metrics,
                target_metrics=target_metrics,
                allowed_dofs=("lens_x", "lens_y", "camera_x", "camera_y"),
                tolerance_reference=tolerance_reference,
            )

        from qwen_h1_meta_v0_candidate.compiler import compile_guidance
        from qwen_h1_meta_v0_candidate.contracts import parse_meta_output

        try:
            from qwen_h1_meta_v0_candidate.controller import plan_guided_h1
        except (ImportError, AttributeError) as exc:
            raise DataGenerationError(
                "candidate controller.plan_guided_h1 is unavailable; generator "
                "refuses to substitute an unfrozen planner"
            ) from exc
        decision = parse_meta_output(
            _configuration_output(
                configuration,
                confidence="medium",
                reason_codes=["recent_response_consistent"],
            )
        )
        guidance = compile_guidance(
            decision,
            default_bounds=self.bounds,
            default_config=self.default_config,
        )
        return plan_guided_h1(
            guidance=guidance,
            default_bounds=self.bounds,
            predictor=predictor,
            default_config=self.default_config,
            seed=seed,
            positions_mm=positions_mm,
            current_metrics=current_metrics,
            target_metrics=target_metrics,
            tolerance_reference=tolerance_reference,
        )


def _image_audit_features(image: np.ndarray) -> dict[str, Any]:
    values = np.asarray(image, dtype=np.float64)
    if values.shape != (1024, 1024):
        raise DataGenerationError(
            f"image audit requires 1024x1024 input, got {values.shape}"
        )
    spatial = values.reshape(16, 64, 16, 64).mean(axis=(1, 3)).reshape(-1)
    return {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "saturated_fraction": float(np.mean(values >= 1.0)),
        "spatial_mean_16x16": [float(value) for value in spatial],
    }


def _save_grayscale_png(path: Path, normalized_image: np.ndarray) -> str:
    from PIL import Image

    values = np.asarray(normalized_image, dtype=np.float64)
    if values.shape != (1024, 1024):
        raise DataGenerationError(f"expected a 1024x1024 image, got {values.shape}")
    if not np.isfinite(values).all():
        raise DataGenerationError("cannot export a non-finite image")
    pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        Image.fromarray(pixels, mode="L").save(temporary_path, format="PNG")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return _sha256_file(path)


def _actuator_semantics(action_high: Sequence[float]) -> list[dict[str, Any]]:
    descriptions = (
        ("move the lens in positive laboratory x", "move the lens in negative laboratory x"),
        ("move the lens in positive laboratory y", "move the lens in negative laboratory y"),
        ("move the camera in positive laboratory x", "move the camera in negative laboratory x"),
        ("move the camera in positive laboratory y", "move the camera in negative laboratory y"),
    )
    return [
        {
            "action_id": action,
            "position_id": position,
            "unit": "mm",
            "positive_command_semantics": descriptions[index][0],
            "negative_command_semantics": descriptions[index][1],
            "legal_per_step_bounds_mm": [
                -float(action_high[index]),
                float(action_high[index]),
            ],
            "absolute_limit_source": "repository_sampling_domain_not_hardware_limit",
        }
        for index, (action, position) in enumerate(zip(ACTION_FIELDS, POSITION_FIELDS))
    ]


class CandidateDataGenerator:
    """Generate the complete preregistered candidate dataset and oracle labels."""

    def __init__(
        self,
        *,
        output_root: Path,
        identity_registry_path: Path,
        device: str = "cpu",
        resume: bool = False,
        model: Any | None = None,
        planner_backend: PlannerBackend | None = None,
    ) -> None:
        # This check must remain first: no caller-selected registry is opened,
        # and no other generation input is consumed, until its path is proven
        # to be the one preregistered identity-only snapshot.
        identity_registry_path = validate_identity_registry_freeze(
            identity_registry_path
        )
        self.output_root = output_root.expanduser().resolve()
        try:
            self.output_root.relative_to(PACKAGE_ROOT.resolve())
        except ValueError as exc:
            raise DataGenerationError(
                f"candidate data output must remain inside {PACKAGE_ROOT}"
            ) from exc
        prohibited = {"frozen", "protected", "heldout", "held_out"}
        if {part.lower() for part in self.output_root.parts} & prohibited:
            raise DataGenerationError("candidate output path contains a prohibited component")
        if self.output_root.exists() and any(self.output_root.iterdir()) and not resume:
            raise DataGenerationError(
                f"output directory is non-empty; pass --resume to continue: {self.output_root}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.resume = bool(resume)
        self.protocol_hashes = verify_frozen_protocol()
        self.source_hashes = verify_source_freeze()
        self.identity_registry = KnownIdentityRegistry.load(identity_registry_path)
        preregistration = _read_json(
            PROTOCOL_ROOT / "data_generation_preregistration.json"
        )
        preregistered_count = int(
            preregistration["overlap"]["known_identity_count_from_read_only_audit"]
        )
        declared_count = int(
            _read_json(identity_registry_path)["expected_distinct_count"]
        )
        actual_count = len(self.identity_registry.setup_hashes)
        if (
            preregistered_count != 298
            or declared_count != preregistered_count
            or actual_count != preregistered_count
        ):
            raise DataGenerationError(
                "frozen preregistration requires exactly 298 distinct known identities"
            )

        from continuous_control_v12.contracts import Bounds
        from continuous_control_v12.simulator import default_simulator_fixed

        protocol = _read_json(PROTOCOL_ROOT / "meta_controller_protocol.json")
        source = protocol["source_freeze"]
        self.v12_config_path = REPO_ROOT / str(source["simulator_config"])
        self.v12_config = _read_json(self.v12_config_path)
        self.bounds = Bounds.from_config(self.v12_config)
        self.bounds.validate()
        locked = _read_json(REPO_ROOT / str(source["default_h1_protocol"]))
        self.base_config_path = (REPO_ROOT / str(locked["base_simulator_config"])).resolve()
        simulator = self.v12_config["simulator"]
        self.simulator_fixed = default_simulator_fixed(
            str(self.base_config_path),
            grid_size=int(simulator["data_grid_size"]),
            grid_extent_mm=float(simulator["data_grid_extent_mm"]),
            sensor_resolution=list(simulator["data_sensor_resolution"]),
            semantics=simulator["semantics"],
        )
        expected_fixed = {
            "grid_size": 1536,
            "grid_extent_mm": 6.25,
            "sensor_resolution_px": [1024, 1024],
            "simulator_semantics_version": "v12_sensor_power_semantics_v1",
            "sensor_sampling_method": "pixel_area_bilinear_intensity",
            "pixel_area_quadrature_order": 3,
        }
        for key, expected in expected_fixed.items():
            if self.simulator_fixed.get(key) != expected:
                raise DataGenerationError(
                    f"corrected-v12 simulator fixed field {key!r} is not frozen value {expected!r}"
                )

        if model is None:
            from continuous_control_v12.world_model import load_forward_ensemble

            checkpoint = REPO_ROOT / str(source["forward_checkpoint"])
            model = load_forward_ensemble(checkpoint, device_name=device)
        self.model = model
        self.planner_backend = planner_backend or FrozenH1PlannerBackend(
            model=self.model, bounds=self.bounds
        )
        self.simulator_calls = 0
        self.forward_calls = 0
        self.planner_calls = 0
        self.rejected_known_identity_candidates = 0
        self.rejected_candidate_identity_candidates = 0

    def _simulate(
        self, setup_context: Mapping[str, Any], positions: Sequence[float]
    ) -> dict[str, Any]:
        from continuous_control_v12.simulator import simulate_state

        self.simulator_calls += 1
        return simulate_state(
            setup_context,
            _position_dict(positions),
            self.simulator_fixed,
            str(self.base_config_path),
            self.bounds,
        )

    def _predict(
        self,
        *,
        setup_context: Mapping[str, Any],
        positions: Sequence[float],
        metrics: Sequence[float],
        action: Sequence[float],
    ) -> dict[str, Any]:
        self.forward_calls += 1
        return _forward_prediction(
            self.model,
            setup_context=setup_context,
            positions_mm=positions,
            current_metrics=metrics,
            action_mm=action,
        )

    @staticmethod
    def _setup_id(split: str, index: int) -> str:
        prefix = "eval" if split == "candidate_eval" else split
        return f"qh1meta_{prefix}_{index:04d}"

    def _sample_setup(
        self,
        *,
        split: str,
        index: int,
        accepted_hashes: set[str],
        accepted_fingerprints: set[tuple[float, ...]],
    ) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], int, str]:
        from continuous_control_v12.contracts import position_vector, setup_hash
        from continuous_control_v12.simulator import sample_group_setup

        setup_id = self._setup_id(split, index)
        regime = REGIME_CYCLE[index % len(REGIME_CYCLE)]
        for attempt in range(16):
            setup_context, positions_mapping = sample_group_setup(
                regime,
                setup_id,
                int(SPLIT_SPECS[split]["setup_seed"]),
                self.simulator_fixed,
                self.bounds,
                setup_attempt=attempt,
            )
            positions = position_vector(positions_mapping)
            capture = self._simulate(setup_context, positions)
            auxiliary = capture["auxiliary"]
            source_power = float(auxiliary.get("source_integrated_power_w", 0.0))
            captured = float(auxiliary.get("captured_power_w", 0.0))
            captured_fraction = captured / source_power if source_power > 0.0 else 0.0
            if not bool(auxiliary["simulator_valid"]) or captured_fraction < 0.01:
                continue
            identity_hash = setup_hash(setup_context, self.simulator_fixed)
            fingerprint = setup_common_fingerprint(setup_context)
            known = self.identity_registry.overlaps(identity_hash, setup_context)
            if any(known.values()):
                self.rejected_known_identity_candidates += 1
                continue
            if identity_hash in accepted_hashes or fingerprint in accepted_fingerprints:
                self.rejected_candidate_identity_candidates += 1
                continue
            return dict(setup_context), positions, capture, attempt, identity_hash
        raise DataGenerationError(
            f"{setup_id}: no acceptable setup in the frozen maximum of 16 attempts"
        )

    def _generate_history(
        self,
        *,
        setup_context: Mapping[str, Any],
        initial_positions: np.ndarray,
        initial_capture: Mapping[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
        from continuous_control_v12.contracts import apply_action, project_action

        positions = initial_positions.copy()
        capture = dict(initial_capture)
        history: list[dict[str, Any]] = []
        for fractions in HISTORY_ACTION_FRACTIONS:
            current_metrics = _metric_vector(capture["metrics"])
            requested = np.asarray(fractions, dtype=np.float64) * self.bounds.action_high
            action = project_action(positions, requested, self.bounds)
            prediction = self._predict(
                setup_context=setup_context,
                positions=positions,
                metrics=current_metrics,
                action=action,
            )
            next_positions = apply_action(positions, action, self.bounds)
            next_capture = self._simulate(setup_context, next_positions)
            if not bool(next_capture["auxiliary"]["simulator_valid"]):
                raise DataGenerationError("a frozen real-history transition is simulator-invalid")
            next_metrics = _metric_vector(next_capture["metrics"])
            measured_delta = next_metrics - current_metrics
            predicted_delta = prediction["predicted_next_metrics"] - current_metrics
            history.append(
                {
                    "valid": True,
                    "executed_action_mm": _action_dict(action),
                    "measured_beam_delta": _metric_dict(measured_delta),
                    "predicted_beam_delta": _metric_dict(predicted_delta),
                    "prediction_residual": _metric_dict(measured_delta - predicted_delta),
                    "ensemble_uncertainty": _metric_dict(prediction["uncertainty"]),
                    "padding_reason": "none",
                }
            )
            positions = np.asarray(next_positions, dtype=np.float64)
            capture = next_capture
        if len(history) != 3:
            raise AssertionError("history generation did not produce exactly three transitions")
        return positions, capture, history

    def _target_capture(
        self,
        *,
        setup_context: Mapping[str, Any],
        current_positions: np.ndarray,
        multiples: Sequence[float],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        target_positions = np.clip(
            current_positions
            + np.asarray(multiples, dtype=np.float64) * self.bounds.action_high,
            self.bounds.position_low,
            self.bounds.position_high,
        )
        return target_positions, self._simulate(setup_context, target_positions)

    def _actual_action_result(
        self,
        *,
        configuration_result: Mapping[str, Any],
        setup_context: Mapping[str, Any],
        positions: np.ndarray,
        target_metrics: np.ndarray,
        tolerance_reference: np.ndarray,
    ) -> dict[str, Any]:
        from continuous_control_v12.contracts import apply_action, project_action

        action = _action_vector(configuration_result["selected_action"])
        projected_action = project_action(positions, action, self.bounds)
        projection_changed = bool(configuration_result.get("projection_changed", False)) or (
            not np.allclose(action, projected_action, atol=1e-12, rtol=0.0)
        )
        next_positions = apply_action(positions, projected_action, self.bounds)
        capture = self._simulate(setup_context, next_positions)
        auxiliary = capture["auxiliary"]
        score, components = canonical_one_step_score(
            next_metrics=_metric_vector(capture["metrics"]),
            target_metrics=target_metrics,
            tolerance_reference=tolerance_reference,
            action_mm=projected_action,
            action_limit_mm=self.bounds.action_high,
            clipping_fraction=float(auxiliary.get("clipping_fraction", 0.0)),
            boundary_indicator=float(bool(auxiliary.get("camera_boundary_indicator", False))),
            projected=projection_changed,
            invalid=not bool(auxiliary.get("simulator_valid", False)),
        )
        output = dict(configuration_result)
        output.update(
            {
                "selected_action": _action_dict(projected_action),
                "actual_score": score,
                "actual_score_components": components,
                "actual_next_metrics": _metric_dict(_metric_vector(capture["metrics"])),
                "simulator_valid": bool(auxiliary.get("simulator_valid", False)),
                "projection_changed": projection_changed,
            }
        )
        return output

    def _local_directional_priors(
        self,
        *,
        setup_context: Mapping[str, Any],
        positions: np.ndarray,
        target_metrics: np.ndarray,
        tolerance_reference: np.ndarray,
    ) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
        from continuous_control_v12.contracts import apply_action, project_action

        directions: dict[str, str] = {}
        probe_audit: dict[str, Any] = {}
        for axis, field in enumerate(ACTION_FIELDS):
            scores: dict[str, float] = {}
            for sign_name, sign in (("decrease", -1.0), ("increase", 1.0)):
                requested = np.zeros(4, dtype=np.float64)
                requested[axis] = sign * 0.5 * self.bounds.action_high[axis]
                action = project_action(positions, requested, self.bounds)
                next_positions = apply_action(positions, action, self.bounds)
                capture = self._simulate(setup_context, next_positions)
                auxiliary = capture["auxiliary"]
                score, _ = canonical_one_step_score(
                    next_metrics=_metric_vector(capture["metrics"]),
                    target_metrics=target_metrics,
                    tolerance_reference=tolerance_reference,
                    action_mm=action,
                    action_limit_mm=self.bounds.action_high,
                    clipping_fraction=float(auxiliary.get("clipping_fraction", 0.0)),
                    boundary_indicator=float(
                        bool(auxiliary.get("camera_boundary_indicator", False))
                    ),
                    projected=not np.allclose(action, requested, atol=1e-12, rtol=0.0),
                    invalid=not bool(auxiliary.get("simulator_valid", False)),
                )
                scores[sign_name] = score
            if scores["decrease"] == scores["increase"]:
                direction = "hold"
            elif scores["decrease"] < scores["increase"]:
                direction = "decrease"
            else:
                direction = "increase"
            directions[field] = direction
            probe_audit[field] = {
                "decrease_actual_score": scores["decrease"],
                "increase_actual_score": scores["increase"],
                "selected_direction": direction,
            }
        return (
            {
                "all_actuators": dict(directions),
                "lens_only": {
                    field: directions[field] if index < 2 else "hold"
                    for index, field in enumerate(ACTION_FIELDS)
                },
                "camera_only": {
                    field: directions[field] if index >= 2 else "hold"
                    for index, field in enumerate(ACTION_FIELDS)
                },
            },
            probe_audit,
        )

    def _predicted_configuration_result(
        self,
        *,
        configuration: Mapping[str, Any],
        setup_id: str,
        target_id: str,
        setup_context: Mapping[str, Any],
        positions: np.ndarray,
        current_metrics: np.ndarray,
        target_metrics: np.ndarray,
        tolerance_reference: np.ndarray,
    ) -> dict[str, Any]:
        from continuous_control_v12.contracts import project_action, stable_seed

        seed = stable_seed(ORACLE_SEED, setup_id, target_id, configuration["configuration_id"])
        self.planner_calls += 1
        plan = self.planner_backend.plan(
            configuration=configuration,
            setup_context=setup_context,
            positions_mm=positions,
            current_metrics=current_metrics,
            target_metrics=target_metrics,
            tolerance_reference=tolerance_reference,
            seed=seed,
        )
        selected_action = _action_vector(
            plan.get("selected_effective_action", plan["selected_action"])
        )
        requested_action = _action_vector(
            plan.get("selected_requested_action", selected_action)
        )
        effective_action = project_action(positions, selected_action, self.bounds)
        projection_changed = (
            not np.allclose(requested_action, selected_action, atol=1e-12, rtol=0.0)
            or not np.allclose(selected_action, effective_action, atol=1e-12, rtol=0.0)
        )
        physical = self._predict(
            setup_context=setup_context,
            positions=positions,
            metrics=current_metrics,
            action=effective_action,
        )
        auxiliary = physical["auxiliary_predictions"]
        predicted_score, components = canonical_one_step_score(
            next_metrics=physical["predicted_next_metrics"],
            target_metrics=target_metrics,
            tolerance_reference=tolerance_reference,
            action_mm=effective_action,
            action_limit_mm=self.bounds.action_high,
            clipping_fraction=float(auxiliary.get("clipping_fraction", 0.0)),
            boundary_indicator=float(auxiliary.get("camera_boundary_probability", 0.0)),
            mean_normalized_uncertainty=float(np.mean(physical["uncertainty"])),
            projected=projection_changed,
            invalid=False,
        )
        return {
            "configuration_id": configuration["configuration_id"],
            "configuration": dict(configuration),
            "planner_seed": int(seed),
            "selected_action": _action_dict(effective_action),
            "predicted_next_metrics": _metric_dict(physical["predicted_next_metrics"]),
            "predicted_uncertainty": _metric_dict(physical["uncertainty"]),
            "predicted_auxiliary": dict(auxiliary),
            "predicted_score": predicted_score,
            "predicted_score_components": components,
            "projection_changed": projection_changed,
        }

    @staticmethod
    def _reason_codes(
        *,
        current_metrics: np.ndarray,
        target_metrics: np.ndarray,
        tolerance_reference: np.ndarray,
        history: Sequence[Mapping[str, Any]],
        current_uncertainty: np.ndarray,
    ) -> list[str]:
        normalized = np.abs(target_metrics - current_metrics) / tolerance_reference
        reasons: list[str] = []
        if float(normalized.max()) <= 1.0:
            reasons.append("near_target")
        else:
            category_scores = {
                "centroid_error_dominant": float(normalized[:2].max()),
                "width_error_dominant": float(normalized[2:4].max()),
                "intensity_error_dominant": float(normalized[4]),
            }
            reasons.append(
                max(
                    category_scores,
                    key=lambda name: (category_scores[name], -list(category_scores).index(name)),
                )
            )
        valid_history = [item for item in history if item["valid"]]
        if len(valid_history) < 3:
            reasons.append("insufficient_history")
        else:
            residuals = np.asarray(
                [
                    [float(item["prediction_residual"][field]) for field in METRIC_FIELDS]
                    for item in valid_history
                ],
                dtype=np.float64,
            )
            normalized_residual = np.mean(np.abs(residuals) / tolerance_reference[None, :])
            uncertainty = np.mean(
                [
                    np.mean(
                        [float(item["ensemble_uncertainty"][field]) for field in METRIC_FIELDS]
                    )
                    for item in valid_history
                ]
            )
            reasons.append(
                "recent_response_mismatch"
                if normalized_residual > max(1.0, 2.0 * uncertainty)
                else "recent_response_consistent"
            )
        if float(np.mean(current_uncertainty)) > 1.0:
            reasons.append("elevated_uncertainty")
        return list(dict.fromkeys(reasons))[:4]

    def _build_model_visible_input(
        self,
        *,
        positions: np.ndarray,
        current_metrics: np.ndarray,
        target_metrics: np.ndarray,
        tolerance_reference: np.ndarray,
        history: Sequence[Mapping[str, Any]],
        current_uncertainty: np.ndarray,
        measurement_validity: Mapping[str, str],
        remaining_budget: Mapping[str, int],
    ) -> dict[str, Any]:
        state = {
            "schema_version": "qwen_h1_meta_input_v0",
            "current_beam_image": {
                "role": "current_sensor_frame_beam_image",
                "coordinate_frame": "camera_sensor_array",
                "display_normalization": "per_image_peak_normalized_for_qwen_only",
                "width_px": 1024,
                "height_px": 1024,
            },
            "current_beam_state": _metric_dict(current_metrics),
            "target_beam_state": _metric_dict(target_metrics),
            "normalized_signed_error": _metric_dict(
                (target_metrics - current_metrics) / tolerance_reference
            ),
            "actuator_positions_mm": _position_dict(positions),
            "actuator_semantics": _actuator_semantics(self.bounds.action_high),
            "history": [dict(item) for item in history],
            "forward_uncertainty": {
                "per_metric": _metric_dict(current_uncertainty),
                "mean": float(np.mean(current_uncertainty)),
                "maximum": float(np.max(current_uncertainty)),
            },
            "remaining_budget": {
                "measurement_steps": int(remaining_budget["measurement_steps"]),
                "control_steps": int(remaining_budget["control_steps"]),
            },
            "measurement_validity": dict(measurement_validity),
        }
        validate_model_visible_state(state)
        from qwen_h1_meta_v0_candidate.contracts import parse_meta_input

        return parse_meta_input(state).to_dict()

    def _oracle_for_target(
        self,
        *,
        setup_id: str,
        target_id: str,
        setup_context: Mapping[str, Any],
        positions: np.ndarray,
        current_metrics: np.ndarray,
        target_metrics: np.ndarray,
        tolerance_reference: np.ndarray,
        history: Sequence[Mapping[str, Any]],
        current_uncertainty: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        local_priors, probe_audit = self._local_directional_priors(
            setup_context=setup_context,
            positions=positions,
            target_metrics=target_metrics,
            tolerance_reference=tolerance_reference,
        )
        configurations = enumerate_oracle_configurations(local_priors)
        predicted = [
            self._predicted_configuration_result(
                configuration=configuration,
                setup_id=setup_id,
                target_id=target_id,
                setup_context=setup_context,
                positions=positions,
                current_metrics=current_metrics,
                target_metrics=target_metrics,
                tolerance_reference=tolerance_reference,
            )
            for configuration in configurations
        ]
        default_result = next(
            item for item in predicted if item["configuration_id"] == "default"
        )
        shortlist = shortlist_guided_candidates(predicted, size=6)
        actual_default = self._actual_action_result(
            configuration_result=default_result,
            setup_context=setup_context,
            positions=positions,
            target_metrics=target_metrics,
            tolerance_reference=tolerance_reference,
        )
        actual_guided = [
            self._actual_action_result(
                configuration_result=item,
                setup_context=setup_context,
                positions=positions,
                target_metrics=target_metrics,
                tolerance_reference=tolerance_reference,
            )
            for item in shortlist
        ]
        selected, label_audit = choose_actual_label(actual_default, actual_guided)
        reasons = self._reason_codes(
            current_metrics=current_metrics,
            target_metrics=target_metrics,
            tolerance_reference=tolerance_reference,
            history=history,
            current_uncertainty=current_uncertainty,
        )
        improvement = float(label_audit["guided_improvement"])
        confidence = (
            "high"
            if selected["decision"] == "run_guided_h1"
            and improvement >= HIGH_GUIDED_IMPROVEMENT
            else "medium"
        )
        output = _configuration_output(
            selected["configuration"],
            confidence=confidence,
            reason_codes=reasons,
        )
        from qwen_h1_meta_v0_candidate.contracts import parse_meta_output

        output = parse_meta_output(output).to_dict()
        oracle_audit = {
            **label_audit,
            "configuration_count": len(configurations),
            "predicted_shortlist_size": len(shortlist),
            "local_direction_probes": probe_audit,
            "predicted_configurations": predicted,
            "actual_default": actual_default,
            "actual_shortlist": actual_guided,
        }
        return output, oracle_audit

    @staticmethod
    def _defensive_output(decision: str) -> dict[str, Any]:
        default_configuration = {
            "decision": decision,
            "objective_profile": "balanced",
            "mask_profile": "default",
            "directional_prior": {field: "unknown" for field in ACTION_FIELDS},
            "step_scale": "default",
            "risk_mode": "standard",
        }
        if decision == "reobserve":
            reason = "measurement_requires_revalidation"
            observation_request = "revalidate"
        elif decision == "stop":
            reason = "budget_exhausted"
            observation_request = "reuse_current"
        else:
            raise ValueError(f"unsupported defensive decision: {decision}")
        output = _configuration_output(
            default_configuration,
            confidence="high",
            reason_codes=[reason],
        )
        output["observation_request"] = observation_request
        from qwen_h1_meta_v0_candidate.contracts import parse_meta_output

        return parse_meta_output(output).to_dict()

    def _generate_setup_rows(
        self,
        *,
        split: str,
        index: int,
        setup_context: Mapping[str, Any],
        initial_positions: np.ndarray,
        initial_capture: Mapping[str, Any],
        setup_attempt: int,
        identity_hash: str,
    ) -> list[dict[str, Any]]:
        from continuous_control_v12.contracts import tolerance_vector

        setup_id = self._setup_id(split, index)
        positions, current_capture, history = self._generate_history(
            setup_context=setup_context,
            initial_positions=initial_positions,
            initial_capture=initial_capture,
        )
        current_metrics = _metric_vector(current_capture["metrics"])
        tolerance_reference = tolerance_vector(current_metrics)
        current_prediction = self._predict(
            setup_context=setup_context,
            positions=positions,
            metrics=current_metrics,
            action=np.zeros(4, dtype=np.float64),
        )
        current_uncertainty = np.asarray(
            current_prediction["uncertainty"], dtype=np.float64
        )

        image_path = self.output_root / "images" / split / f"{setup_id}.png"
        image_hash = _save_grayscale_png(
            image_path, np.asarray(current_capture["image_normalized"])
        )
        try:
            storage_path = str(image_path.relative_to(REPO_ROOT))
        except ValueError:
            storage_path = str(image_path)
        image_metadata = {
            "storage_path": storage_path,
            "sha256": image_hash,
            "width_px": 1024,
            "height_px": 1024,
            "mode": "L",
            "normalization": "per_image_peak_normalized_for_qwen_only",
            "audit_features": _image_audit_features(
                np.asarray(current_capture["image_normalized"])
            ),
        }

        rows: list[dict[str, Any]] = []
        for target_index, (target_id, multiples) in enumerate(TARGET_VARIANTS):
            target_positions, target_capture = self._target_capture(
                setup_context=setup_context,
                current_positions=positions,
                multiples=multiples,
            )
            target_metrics = _metric_vector(target_capture["metrics"])
            measurement_validity: dict[str, str] = {
                "state": "valid",
                "supervisor_diagnosis": "nominal",
                "measurement_policy": "standard",
            }
            remaining_budget = {"measurement_steps": 2, "control_steps": 4}
            defensive: str | None = None
            if target_id == "near_target" and index % 8 == 0:
                measurement_validity = {
                    "state": "requires_recovery",
                    "supervisor_diagnosis": "sensor_saturation",
                    "measurement_policy": "lower_exposure_reacquire",
                }
                defensive = "reobserve"
            elif target_id == "near_target" and index % 8 == 1:
                remaining_budget["control_steps"] = 0
                defensive = "stop"

            model_visible_input = self._build_model_visible_input(
                positions=positions,
                current_metrics=current_metrics,
                target_metrics=target_metrics,
                tolerance_reference=tolerance_reference,
                history=history,
                current_uncertainty=current_uncertainty,
                measurement_validity=measurement_validity,
                remaining_budget=remaining_budget,
            )
            if defensive is None:
                oracle_output, oracle_audit = self._oracle_for_target(
                    setup_id=setup_id,
                    target_id=target_id,
                    setup_context=setup_context,
                    positions=positions,
                    current_metrics=current_metrics,
                    target_metrics=target_metrics,
                    tolerance_reference=tolerance_reference,
                    history=history,
                    current_uncertainty=current_uncertainty,
                )
            else:
                oracle_output = self._defensive_output(defensive)
                oracle_audit = {
                    "selected_configuration_id": defensive,
                    "defensive_rule": defensive,
                    "configuration_count": 0,
                    "predicted_shortlist_size": 0,
                    "local_direction_probes": {},
                    "predicted_configurations": [],
                    "actual_default": None,
                    "actual_shortlist": [],
                }
            record_id = f"{setup_id}__target-{target_index:02d}-{target_id}"
            rows.append(
                {
                    "schema_version": "qwen_h1_meta_v0_candidate_record_v1",
                    "record_id": record_id,
                    "split": split,
                    "image": dict(image_metadata),
                    "model_visible_input": model_visible_input,
                    "oracle_output": oracle_output,
                    "identity": {
                        "setup_hash": identity_hash,
                        "common_fields_rounded_9": list(
                            setup_common_fingerprint(setup_context)
                        ),
                    },
                    "oracle_audit": oracle_audit,
                    "evaluator_only": {
                        "setup_id": setup_id,
                        "episode_id": record_id,
                        "target_counterfactual_id": target_id,
                        "setup_index": index,
                        "target_index": target_index,
                        "regime": REGIME_CYCLE[index % len(REGIME_CYCLE)],
                        "setup_attempt": setup_attempt,
                        "setup_context": dict(setup_context),
                        "simulator_fixed": dict(self.simulator_fixed),
                        "target_positions_mm": _position_dict(target_positions),
                        "target_seed": TARGET_SEED,
                        "oracle_seed": ORACLE_SEED,
                        "target_simulator_valid": bool(
                            target_capture["auxiliary"]["simulator_valid"]
                        ),
                    },
                }
            )
        return rows

    def _load_existing_partials(self) -> dict[tuple[str, int], list[dict[str, Any]]]:
        if not self.resume:
            return {}
        partials: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for split, spec in SPLIT_SPECS.items():
            for index in range(int(spec["count"])):
                path = self.output_root / "partial" / split / f"{self._setup_id(split, index)}.json"
                if not path.exists():
                    continue
                payload = _read_json(path)
                if payload.get("protocol_hashes") != self.protocol_hashes:
                    raise DataGenerationError(f"stale/tampered protocol hashes in partial: {path}")
                if payload.get("source_hashes") != self.source_hashes:
                    raise DataGenerationError(f"stale/tampered source hashes in partial: {path}")
                rows = payload.get("rows")
                if not isinstance(rows, list) or len(rows) != 3:
                    raise DataGenerationError(f"invalid resume partial: {path}")
                if any(row.get("split") != split for row in rows):
                    raise DataGenerationError(f"resume partial split mismatch: {path}")
                if any(
                    row.get("evaluator_only", {}).get("setup_id")
                    != self._setup_id(split, index)
                    for row in rows
                ):
                    raise DataGenerationError(f"resume partial setup mismatch: {path}")
                if any(
                    int(row.get("evaluator_only", {}).get("setup_index", -1)) != index
                    for row in rows
                ):
                    raise DataGenerationError(f"resume partial setup index mismatch: {path}")
                target_ids = {
                    str(row.get("evaluator_only", {}).get("target_counterfactual_id"))
                    for row in rows
                }
                if target_ids != {name for name, _ in TARGET_VARIANTS}:
                    raise DataGenerationError(f"resume partial target registry mismatch: {path}")
                identity_hashes = {
                    str(row.get("identity", {}).get("setup_hash")) for row in rows
                }
                if len(identity_hashes) != 1:
                    raise DataGenerationError(f"resume partial identity mismatch: {path}")
                contexts = {
                    _compact_json(row.get("evaluator_only", {}).get("setup_context"))
                    for row in rows
                }
                if len(contexts) != 1:
                    raise DataGenerationError(f"resume partial setup-context mismatch: {path}")
                expected_fingerprint = list(
                    setup_common_fingerprint(rows[0]["evaluator_only"]["setup_context"])
                )
                if any(
                    list(row.get("identity", {}).get("common_fields_rounded_9", []))
                    != expected_fingerprint
                    for row in rows
                ):
                    raise DataGenerationError(f"resume partial fingerprint mismatch: {path}")
                from qwen_h1_meta_v0_candidate.contracts import (
                    parse_meta_input,
                    parse_meta_output,
                )

                for row in rows:
                    parse_meta_input(row["model_visible_input"])
                    parse_meta_output(row["oracle_output"])
                    validate_model_visible_state(row["model_visible_input"])
                image_path = Path(rows[0]["image"]["storage_path"])
                if not image_path.is_absolute():
                    image_path = REPO_ROOT / image_path
                try:
                    image_path.resolve().relative_to(self.output_root)
                except ValueError as exc:
                    raise DataGenerationError(
                        f"resume partial image is outside output root: {image_path}"
                    ) from exc
                if any(
                    row["image"]["storage_path"] != rows[0]["image"]["storage_path"]
                    or row["image"]["sha256"] != rows[0]["image"]["sha256"]
                    for row in rows
                ):
                    raise DataGenerationError(f"resume partial image metadata mismatch: {path}")
                if _sha256_file(image_path) != rows[0]["image"]["sha256"]:
                    raise DataGenerationError(f"resume image hash mismatch: {image_path}")
                partials[(split, index)] = rows
        return partials

    def _identity_overlap_summary(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        split_hashes: dict[str, set[str]] = {split: set() for split in SPLIT_SPECS}
        split_fingerprints: dict[str, set[tuple[float, ...]]] = {
            split: set() for split in SPLIT_SPECS
        }
        known_overlap_hashes: set[str] = set()
        seen_setup_hashes: set[str] = set()
        for row in rows:
            split = str(row["split"])
            identity_hash = str(row["identity"]["setup_hash"])
            split_hashes[split].add(identity_hash)
            split_fingerprints[split].add(
                tuple(float(value) for value in row["identity"]["common_fields_rounded_9"])
            )
            if identity_hash not in seen_setup_hashes:
                seen_setup_hashes.add(identity_hash)
                setup_context = row["evaluator_only"]["setup_context"]
                if any(self.identity_registry.overlaps(identity_hash, setup_context).values()):
                    known_overlap_hashes.add(identity_hash)
        exact_cross: set[str] = set()
        rounded_cross: set[tuple[float, ...]] = set()
        split_names = list(SPLIT_SPECS)
        for left_index, left in enumerate(split_names):
            for right in split_names[left_index + 1 :]:
                exact_cross |= split_hashes[left] & split_hashes[right]
                rounded_cross |= split_fingerprints[left] & split_fingerprints[right]
        return {
            "known_overlap_count": len(known_overlap_hashes),
            "cross_split_exact_setup_hash_overlap_count": len(exact_cross),
            "cross_split_rounded_fingerprint_overlap_count": len(rounded_cross),
            "cross_split_overlap_count": len(exact_cross) + len(rounded_cross),
            "distinct_setups_by_split": {
                split: len(values) for split, values in split_hashes.items()
            },
        }

    def generate(self) -> dict[str, Any]:
        """Run all preregistered splits, audit, and conditionally export SFT rows."""

        import time

        started = time.monotonic()
        partials = self._load_existing_partials()
        accepted_hashes: set[str] = set()
        accepted_fingerprints: set[tuple[float, ...]] = set()
        for rows in partials.values():
            setup_context = rows[0]["evaluator_only"]["setup_context"]
            identity_hash = str(rows[0]["identity"]["setup_hash"])
            fingerprint = setup_common_fingerprint(setup_context)
            overlap = self.identity_registry.overlaps(identity_hash, setup_context)
            if any(overlap.values()):
                raise DataGenerationError("resume partial overlaps the known identity registry")
            if identity_hash in accepted_hashes or fingerprint in accepted_fingerprints:
                raise DataGenerationError("duplicate identity found across resume partials")
            accepted_hashes.add(identity_hash)
            accepted_fingerprints.add(fingerprint)

        all_rows: list[dict[str, Any]] = []
        for split, spec in SPLIT_SPECS.items():
            for index in range(int(spec["count"])):
                key = (split, index)
                if key in partials:
                    rows = partials[key]
                else:
                    (
                        setup_context,
                        initial_positions,
                        initial_capture,
                        setup_attempt,
                        identity_hash,
                    ) = self._sample_setup(
                        split=split,
                        index=index,
                        accepted_hashes=accepted_hashes,
                        accepted_fingerprints=accepted_fingerprints,
                    )
                    accepted_hashes.add(identity_hash)
                    accepted_fingerprints.add(setup_common_fingerprint(setup_context))
                    rows = self._generate_setup_rows(
                        split=split,
                        index=index,
                        setup_context=setup_context,
                        initial_positions=initial_positions,
                        initial_capture=initial_capture,
                        setup_attempt=setup_attempt,
                        identity_hash=identity_hash,
                    )
                    partial_path = (
                        self.output_root
                        / "partial"
                        / split
                        / f"{self._setup_id(split, index)}.json"
                    )
                    _atomic_write_json(
                        partial_path,
                        {
                            "version": "qwen_h1_meta_v0_setup_partial_v1",
                            "candidate_only": True,
                            "protocol_hashes": self.protocol_hashes,
                            "source_hashes": self.source_hashes,
                            "rows": rows,
                        },
                    )
                all_rows.extend(rows)

        expected_total = sum(int(spec["count"]) * 3 for spec in SPLIT_SPECS.values())
        if len(all_rows) != expected_total:
            raise DataGenerationError(
                f"complete candidate generation expected {expected_total} rows, got {len(all_rows)}"
            )
        identity_overlap = self._identity_overlap_summary(all_rows)
        identity_overlap["rejected_known_identity_candidates"] = (
            self.rejected_known_identity_candidates
        )
        identity_overlap["rejected_candidate_identity_candidates"] = (
            self.rejected_candidate_identity_candidates
        )
        if identity_overlap["cross_split_overlap_count"]:
            raise DataGenerationError("candidate setup identities overlap across splits")

        from continuous_control_v12.contracts import stable_seed

        manifest_paths: dict[str, Path] = {}
        ordered_rows: list[dict[str, Any]] = []
        for split, spec in SPLIT_SPECS.items():
            split_rows = [row for row in all_rows if row["split"] == split]
            expected = int(spec["count"]) * 3
            if len(split_rows) != expected:
                raise DataGenerationError(
                    f"{split} expected {expected} rows, got {len(split_rows)}"
                )
            random.Random(stable_seed(SHUFFLE_SEED, split)).shuffle(split_rows)
            manifest_path = self.output_root / f"manifest_{split}.jsonl"
            _atomic_write_jsonl(manifest_path, split_rows)
            manifest_paths[split] = manifest_path
            ordered_rows.extend(split_rows)

        audit = information_sufficiency_audit(
            ordered_rows,
            known_identity_count=len(self.identity_registry.setup_hashes),
            identity_overlap_audit=identity_overlap,
        )
        audit["diagnostics"] = dataset_diagnostics(ordered_rows)
        audit_path = self.output_root / "reports" / "data_audit.json"
        _atomic_write_json(audit_path, audit)

        exports: dict[str, str] = {}
        if audit["gate"] == "PASS":
            exports = export_prebuilt_chat(
                rows=ordered_rows,
                output_root=self.output_root,
                data_audit=audit,
            )

        generation_config = {
            "version": "qwen_h1_meta_v0_generation_config_v1",
            "candidate_only": True,
            "formal_frozen_evaluation_enabled": False,
            "output_root": str(self.output_root),
            "identity_registry": {
                "path": str(self.identity_registry.source_path),
                "sha256": _sha256_file(self.identity_registry.source_path),
                "count": len(self.identity_registry.setup_hashes),
            },
            "splits": SPLIT_SPECS,
            "target_seed": TARGET_SEED,
            "oracle_seed": ORACLE_SEED,
            "shuffle_seed": SHUFFLE_SEED,
            "target_variants": dict(TARGET_VARIANTS),
            "history_action_fractions": list(HISTORY_ACTION_FRACTIONS),
            "protocol_hashes": self.protocol_hashes,
            "source_hashes": self.source_hashes,
            "simulator_fixed": self.simulator_fixed,
        }
        config_path = self.output_root / "generation_config.json"
        _atomic_write_json(config_path, generation_config)
        elapsed = time.monotonic() - started
        report = {
            "version": "qwen_h1_meta_v0_generation_report_v1",
            "candidate_only": True,
            "scientific_conclusion": False,
            "gate": audit["gate"],
            "sft_exported": bool(exports),
            "elapsed_seconds": elapsed,
            "simulator_calls_this_process": self.simulator_calls,
            "forward_physical_scoring_calls_this_process": self.forward_calls,
            "planner_calls_this_process": self.planner_calls,
            "rows": {
                split: int(spec["count"]) * 3 for split, spec in SPLIT_SPECS.items()
            },
            "manifests": {
                split: {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                }
                for split, path in manifest_paths.items()
            },
            "data_audit": {"path": str(audit_path), "sha256": _sha256_file(audit_path)},
            "generation_config": {
                "path": str(config_path),
                "sha256": _sha256_file(config_path),
            },
            "exports": exports,
        }
        report_path = self.output_root / "reports" / "generation_report.json"
        _atomic_write_json(report_path, report)
        report["report_path"] = str(report_path)
        report["report_sha256"] = _sha256_file(report_path)
        return report


def _distribution(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _entropy_bits(values: Sequence[str]) -> float:
    counts = np.asarray(list(_distribution(values).values()), dtype=np.float64)
    if counts.size == 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _discrete_mutual_information(labels: Sequence[str], bins: Sequence[int]) -> float:
    if len(labels) != len(bins):
        raise ValueError("mutual-information inputs have different lengths")
    total = float(len(labels))
    if total == 0.0:
        return 0.0
    joint: dict[tuple[str, int], int] = {}
    label_counts: dict[str, int] = {}
    bin_counts: dict[int, int] = {}
    for label, bin_value in zip(labels, bins):
        joint[(label, int(bin_value))] = joint.get((label, int(bin_value)), 0) + 1
        label_counts[label] = label_counts.get(label, 0) + 1
        bin_counts[int(bin_value)] = bin_counts.get(int(bin_value), 0) + 1
    mutual_information = 0.0
    for (label, bin_value), count in joint.items():
        p_joint = count / total
        p_label = label_counts[label] / total
        p_bin = bin_counts[bin_value] / total
        mutual_information += p_joint * math.log2(p_joint / (p_label * p_bin))
    return float(mutual_information)


def _field_dependence_diagnostics(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    label_entropy = _entropy_bits(labels)
    output: dict[str, Any] = {}
    field_names = [
        "current_beam_image",
        "current_beam_image_pixels_summary",
        "current_beam_state",
        "target_beam_state",
        "normalized_signed_error",
        "actuator_positions_mm",
        "actuator_semantics",
        "history",
        "forward_uncertainty",
        "remaining_budget",
        "measurement_validity",
    ]
    for field in field_names:
        vectors: list[list[float]] = []
        for row in rows:
            vector: list[float] = []
            source = (
                row["image"]["audit_features"]
                if field == "current_beam_image_pixels_summary"
                else row["model_visible_input"][field]
            )
            _flatten_numeric(source, vector)
            vectors.append(vector)
        matrix = np.asarray(vectors, dtype=np.float64)
        component_mi: list[float] = []
        for component in range(matrix.shape[1]):
            values = matrix[:, component]
            edges = np.unique(np.quantile(values, [0.25, 0.50, 0.75]))
            bins = np.digitize(values, edges, right=True)
            component_mi.append(_discrete_mutual_information(labels, bins.tolist()))
        maximum = max(component_mi, default=0.0)
        mean = float(np.mean(component_mi)) if component_mi else 0.0
        output[field] = {
            "method": "per_scalar_empirical_quartile_mutual_information",
            "component_count": len(component_mi),
            "maximum_mi_bits": maximum,
            "mean_mi_bits": mean,
            "maximum_normalized_mi": (
                maximum / label_entropy if label_entropy > 0.0 else 0.0
            ),
            "mean_normalized_mi": mean / label_entropy if label_entropy > 0.0 else 0.0,
        }
    return output


def dataset_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Non-gating diagnostics requested for data adequacy and leakage review."""

    audit_rows = [row for row in rows if row["split"] in {"train", "dev"}]
    configuration_labels = [
        str(row["oracle_audit"]["selected_configuration_id"]) for row in audit_rows
    ]
    decisions = [str(row["oracle_output"]["decision"]) for row in audit_rows]
    setup_target_counts: dict[str, int] = {}
    setup_image_hashes: dict[str, set[str]] = {}
    for row in rows:
        setup_id = str(row["evaluator_only"]["setup_id"])
        setup_target_counts[setup_id] = setup_target_counts.get(setup_id, 0) + 1
        setup_image_hashes.setdefault(setup_id, set()).add(str(row["image"]["sha256"]))

    target_distribution: dict[str, Any] = {}
    for target_id, _ in TARGET_VARIANTS:
        target_rows = [
            row
            for row in audit_rows
            if row["evaluator_only"]["target_counterfactual_id"] == target_id
        ]
        errors = np.asarray(
            [
                [
                    float(row["model_visible_input"]["normalized_signed_error"][field])
                    for field in METRIC_FIELDS
                ]
                for row in target_rows
            ],
            dtype=np.float64,
        )
        target_distribution[target_id] = {
            "records": len(target_rows),
            "per_metric_mean": _metric_dict(errors.mean(axis=0)),
            "per_metric_standard_deviation": _metric_dict(errors.std(axis=0)),
            "maximum_absolute_normalized_error_mean": float(
                np.mean(np.max(np.abs(errors), axis=1))
            ),
        }

    direction_values = [
        str(direction)
        for row in audit_rows
        for direction in row["oracle_output"]["directional_prior"].values()
    ]
    split_counts: dict[str, Any] = {}
    for split in SPLIT_SPECS:
        split_rows = [row for row in rows if row["split"] == split]
        split_counts[split] = {
            "records": len(split_rows),
            "setups": len(
                {str(row["evaluator_only"]["setup_id"]) for row in split_rows}
            ),
            "episodes": len(
                {str(row["evaluator_only"]["episode_id"]) for row in split_rows}
            ),
        }
    forbidden_key_occurrences = sum(
        len(set(_walk_keys(row["model_visible_input"])) & MODEL_VISIBLE_FORBIDDEN_KEYS)
        for row in rows
    )
    return {
        "diagnostic_splits_for_label_statistics": ["train", "dev"],
        "split_setup_episode_counts": split_counts,
        "counterfactual_grouping": {
            "setups_with_exactly_three_targets": sum(
                count == 3 for count in setup_target_counts.values()
            ),
            "setup_count": len(setup_target_counts),
            "setups_with_one_shared_current_image": sum(
                len(hashes) == 1 for hashes in setup_image_hashes.values()
            ),
        },
        "model_visible_id_or_path_key_occurrences": forbidden_key_occurrences,
        "label": {
            "configuration_distribution": _distribution(configuration_labels),
            "configuration_entropy_bits": _entropy_bits(configuration_labels),
            "decision_distribution": _distribution(decisions),
            "decision_entropy_bits": _entropy_bits(decisions),
            "default_guided_defensive_balance": {
                "run_default_h1": decisions.count("run_default_h1"),
                "run_guided_h1": decisions.count("run_guided_h1"),
                "reobserve_or_stop": sum(
                    decision in {"reobserve", "stop"} for decision in decisions
                ),
            },
        },
        "output_field_distributions": {
            "objective_profile": _distribution(
                [str(row["oracle_output"]["objective_profile"]) for row in audit_rows]
            ),
            "mask_profile": _distribution(
                [str(row["oracle_output"]["mask_profile"]) for row in audit_rows]
            ),
            "step_scale": _distribution(
                [str(row["oracle_output"]["step_scale"]) for row in audit_rows]
            ),
            "risk_mode": _distribution(
                [str(row["oracle_output"]["risk_mode"]) for row in audit_rows]
            ),
            "direction_tokens": _distribution(direction_values),
        },
        "target_distribution": target_distribution,
        "per_visible_field_dependence": _field_dependence_diagnostics(
            audit_rows, configuration_labels
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or audit qwen_h1_meta_v0 candidate-only data"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-protocol")
    verify.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
    generate = subparsers.add_parser(
        "generate",
        help="run the complete 16/8/12 preregistered candidate generation",
    )
    generate.add_argument(
        "--output-root",
        type=Path,
        default=PACKAGE_ROOT / "data/generated_v1",
    )
    generate.add_argument(
        "--identity-registry",
        type=Path,
        default=PACKAGE_ROOT / "configs/known_identity_blocklist.json",
    )
    generate.add_argument("--device", default="cpu")
    generate.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "verify-protocol":
        verified = verify_frozen_protocol(args.protocol_root)
        source = verify_source_freeze()
        print(
            _compact_json(
                {"status": "PASS", "protocol_verified": verified, "source_verified": source}
            )
        )
        return 0
    if args.command == "generate":
        generator = CandidateDataGenerator(
            output_root=args.output_root,
            identity_registry_path=args.identity_registry,
            device=str(args.device),
            resume=bool(args.resume),
        )
        report = generator.generate()
        print(_compact_json(report))
        return 0 if report["gate"] == "PASS" else 3
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
