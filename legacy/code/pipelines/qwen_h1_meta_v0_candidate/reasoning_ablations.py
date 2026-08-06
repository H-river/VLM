#!/usr/bin/env python3
"""Candidate-only Qwen-H1 reasoning-ablation generation and diagnostics.

This module deliberately does not dispatch actions.  Every generated decision,
including ``full_input`` when produced by this harness, is labelled for
simulator/shadow consumption only.  The frozen protocol is read, never edited.

Two generation surfaces are provided:

* :func:`generate_reasoning_ablation_predictions` produces initial-state raw
  continuations for the complete fresh ``candidate_eval`` manifest.
* :class:`LiveReasoningAblationPolicy` is injectable into
  :mod:`qwen_h1_meta_v0_candidate.evaluate_closed_loop`; it regenerates from
  the updated runtime state and simulator sensor image at every policy call.

Neither surface loads a model until the CLI ``generate`` command is actually
invoked.  Unit tests use the same code with a mock ``IndependentMetaAdapter``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, MutableSequence, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from continuous_control_v12.contracts import tolerance_vector
from specialist_rebuild_v2.common import ACTION_FIELDS, STATE_FIELDS

from . import training
from .contracts import MetaContractError, load_protocol_json, parse_meta_input, parse_meta_output
from .evaluate_closed_loop import (
    ClosedLoopEvaluationError,
    normalize_candidate_eval_manifest as normalize_closed_loop_candidate_eval_manifest,
)
from .evaluate_offline import (
    OfflineEvaluationError,
    evaluate_candidate_eval_predictions,
)
from .inference import IndependentMetaAdapter, load_independent_adapter, parse_generated_text


PROTOCOL = load_protocol_json("meta_controller_protocol.json")
ABLATIONS = tuple(str(value) for value in PROTOCOL["ablations"])
EXPECTED_ABLATIONS = (
    "full_input",
    "blank_image",
    "deterministic_image_shuffle",
    "metrics_blank",
    "current_state_shuffle",
    "target_shuffle",
    "target_replace_current",
    "history_blank",
    "history_shuffle",
    "actuator_semantics_shuffle",
    "uncertainty_blank",
    "reason_codes_disabled",
)
if ABLATIONS != EXPECTED_ABLATIONS:
    raise RuntimeError("frozen reasoning-ablation registry differs from implementation")

ABLATION_SEED = int(PROTOCOL["data"]["split_shuffle_seed"])
if ABLATION_SEED != 2026080213:
    raise RuntimeError("frozen reasoning-ablation seed differs from 2026080213")

MODEL_SEEDS = tuple(int(value) for value in PROTOCOL["training"]["seeds"])
if MODEL_SEEDS != training.TRAINING_SEEDS:
    raise RuntimeError("frozen Qwen seeds differ from candidate training seeds")

EXPECTED_CANDIDATE_EVAL_RECORDS = int(PROTOCOL["data"]["candidate_eval_episodes"])
EXPECTED_CANDIDATE_EVAL_SETUPS = int(PROTOCOL["data"]["candidate_eval_setups"])
TARGETS_PER_SETUP = int(PROTOCOL["data"]["targets_per_setup"])
MAX_NEW_TOKENS = 256

# This is a hard execution boundary, not merely report prose.  Consumers must
# reject a row if either value has been changed.
EXECUTION_SCOPE = "simulator_or_shadow_only"
HARDWARE_DISPATCH_ALLOWED = False
ACTION_EXECUTION = "not_executed_by_reasoning_ablation_harness"

SCALAR_CONTROL_FIELDS = (
    "decision",
    "observation_request",
    "objective_profile",
    "mask_profile",
    "step_scale",
    "risk_mode",
    "confidence",
)
SHUFFLE_ABLATIONS = frozenset(
    {
        "deterministic_image_shuffle",
        "current_state_shuffle",
        "target_shuffle",
        "history_shuffle",
    }
)
HEX_SHA256 = frozenset("0123456789abcdef")


class ReasoningAblationError(ValueError):
    """A stable candidate harness validation failure."""


@dataclass(frozen=True)
class CandidateEvalSample:
    sample_id: str
    setup_hash: str
    runtime_input: dict[str, Any]
    oracle_output: dict[str, Any]
    image_path: Path
    image_sha256: str
    record_sha256: str
    record: dict[str, Any]


@dataclass(frozen=True)
class AblatedInput:
    ablation: str
    runtime_input: dict[str, Any]
    image_path: Path
    image_sha256: str
    donor_sample_id: str | None
    donor_setup_hash: str | None
    operation: dict[str, Any]
    physical_validation: dict[str, str]


DonorRuntimeProvider = Callable[[Any, Mapping[str, Any]], Mapping[str, Any] | None]
DonorImageProvider = Callable[[Any, Mapping[str, Any]], np.ndarray | None]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX_SHA256 for character in value)
    )


def _sample_id(record: Mapping[str, Any], index: int) -> str:
    value = record.get("record_id", record.get("sample_id"))
    if not isinstance(value, str) or not value:
        raise ReasoningAblationError(f"manifest row {index} has no stable record_id")
    return value


def _resolve_image(record: Mapping[str, Any], sample_id: str, *, verify: bool) -> tuple[Path, str]:
    image = record.get("image")
    if not isinstance(image, Mapping):
        raise ReasoningAblationError(f"{sample_id}: missing image metadata")
    storage_path = image.get("storage_path")
    expected_hash = image.get("sha256")
    if not isinstance(storage_path, str) or not storage_path:
        raise ReasoningAblationError(f"{sample_id}: invalid image.storage_path")
    if not _valid_sha256(expected_hash):
        raise ReasoningAblationError(f"{sample_id}: invalid image.sha256")
    image_path = training._guard_candidate_path(
        training._repo_path(storage_path),
        role=f"candidate_eval image for {sample_id}",
        must_exist=verify,
    )
    if verify:
        if not image_path.is_file():
            raise ReasoningAblationError(f"{sample_id}: image is not a file")
        observed_hash = training.sha256_path(image_path)
        if observed_hash != expected_hash:
            raise ReasoningAblationError(
                f"{sample_id}: image SHA-256 mismatch ({observed_hash} != {expected_hash})"
            )
        from PIL import Image

        with Image.open(image_path) as opened:
            if opened.size != (1024, 1024):
                raise ReasoningAblationError(
                    f"{sample_id}: Qwen image must be exactly 1024x1024"
                )
            opened.verify()
    return image_path, str(expected_hash)


def normalize_candidate_eval_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    verify_images: bool = True,
    require_preregistered_cardinality: bool = True,
) -> list[CandidateEvalSample]:
    """Strictly normalize a fresh candidate-eval manifest.

    Only model-visible input, oracle output, irreversible setup hash, and
    candidate image metadata are inspected.  Frozen/protected path components
    are rejected by the shared candidate path guard.
    """

    if not records:
        raise ReasoningAblationError("candidate_eval manifest is empty")
    if require_preregistered_cardinality:
        try:
            # Reuse the authoritative evaluator identity/seed contract so the
            # ablation harness cannot accept a stale or differently indexed
            # 36-row manifest merely because its setup hashes have cardinality.
            normalize_closed_loop_candidate_eval_manifest(
                records,
                require_preregistered_cardinality=True,
            )
        except ClosedLoopEvaluationError as exc:
            raise ReasoningAblationError(
                f"candidate_eval identity registry mismatch: {exc}"
            ) from exc
    samples: list[CandidateEvalSample] = []
    seen: set[str] = set()
    setup_members: defaultdict[str, list[str]] = defaultdict(list)
    for index, source in enumerate(records):
        if not isinstance(source, Mapping):
            raise ReasoningAblationError(f"manifest row {index} is not an object")
        record = copy.deepcopy(dict(source))
        sample_id = _sample_id(record, index)
        if sample_id in seen:
            raise ReasoningAblationError(f"duplicate candidate_eval record_id: {sample_id}")
        seen.add(sample_id)
        if record.get("split") != "candidate_eval":
            raise ReasoningAblationError(
                f"{sample_id}: reasoning harness accepts candidate_eval only"
            )
        identity = record.get("identity")
        if not isinstance(identity, Mapping) or not _valid_sha256(identity.get("setup_hash")):
            raise ReasoningAblationError(
                f"{sample_id}: identity.setup_hash must be an irreversible SHA-256"
            )
        setup_hash = str(identity["setup_hash"])
        visible = record.get("model_visible_input")
        if not isinstance(visible, Mapping):
            raise ReasoningAblationError(f"{sample_id}: missing model_visible_input")
        try:
            runtime_input = parse_meta_input(visible).to_dict()
            oracle = parse_meta_output(record.get("oracle_output")).to_dict()
        except (MetaContractError, TypeError) as exc:
            raise ReasoningAblationError(f"{sample_id}: invalid strict contract: {exc}") from exc
        image_path, image_hash = _resolve_image(record, sample_id, verify=verify_images)
        setup_members[setup_hash].append(sample_id)
        samples.append(
            CandidateEvalSample(
                sample_id=sample_id,
                setup_hash=setup_hash,
                runtime_input=runtime_input,
                oracle_output=oracle,
                image_path=image_path,
                image_sha256=image_hash,
                record_sha256=_sha256_json(record),
                record=record,
            )
        )
    if require_preregistered_cardinality:
        if len(samples) != EXPECTED_CANDIDATE_EVAL_RECORDS:
            raise ReasoningAblationError(
                "preregistered candidate_eval requires "
                f"{EXPECTED_CANDIDATE_EVAL_RECORDS} records, got {len(samples)}"
            )
        if len(setup_members) != EXPECTED_CANDIDATE_EVAL_SETUPS:
            raise ReasoningAblationError(
                "preregistered candidate_eval requires "
                f"{EXPECTED_CANDIDATE_EVAL_SETUPS} setup hashes, got {len(setup_members)}"
            )
        wrong = {
            setup_hash: len(members)
            for setup_hash, members in setup_members.items()
            if len(members) != TARGETS_PER_SETUP
        }
        if wrong:
            raise ReasoningAblationError(
                f"each setup requires {TARGETS_PER_SETUP} target episodes: {wrong}"
            )
    return sorted(samples, key=lambda sample: sample.sample_id)


def _member_sort_key(sample: CandidateEvalSample) -> tuple[int, str]:
    evaluator = sample.record.get("evaluator_only")
    if isinstance(evaluator, Mapping):
        target_index = evaluator.get("target_index")
        if isinstance(target_index, int) and not isinstance(target_index, bool):
            return int(target_index), sample.sample_id
    return sys.maxsize, sample.sample_id


def setup_aware_donor_map(
    samples: Sequence[CandidateEvalSample], *, seed: int = ABLATION_SEED
) -> dict[str, CandidateEvalSample]:
    """Return a deterministic, cross-setup donor for every target episode."""

    if int(seed) != ABLATION_SEED:
        raise ReasoningAblationError(f"shuffle seed is frozen to {ABLATION_SEED}")
    grouped: defaultdict[str, list[CandidateEvalSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.setup_hash].append(sample)
    if len(grouped) < 2:
        raise ReasoningAblationError("setup-aware shuffle requires at least two setups")
    setup_order = sorted(
        grouped,
        key=lambda setup_hash: (
            hashlib.sha256(f"{seed}|setup|{setup_hash}".encode()).hexdigest(),
            setup_hash,
        ),
    )
    result: dict[str, CandidateEvalSample] = {}
    for receiver_index, receiver_setup in enumerate(setup_order):
        donor_setup = setup_order[(receiver_index + 1) % len(setup_order)]
        receivers = sorted(grouped[receiver_setup], key=_member_sort_key)
        donors = sorted(grouped[donor_setup], key=_member_sort_key)
        for index, receiver in enumerate(receivers):
            donor = donors[index % len(donors)]
            if donor.setup_hash == receiver.setup_hash:
                raise AssertionError("setup-aware donor unexpectedly retained setup identity")
            result[receiver.sample_id] = donor
    if set(result) != {sample.sample_id for sample in samples}:
        raise AssertionError("setup-aware donor map is incomplete")
    return result


def _zero_metrics() -> dict[str, float]:
    return {field: 0.0 for field in STATE_FIELDS}


def _zero_action() -> dict[str, float]:
    return {field: 0.0 for field in ACTION_FIELDS}


def _blank_history() -> list[dict[str, Any]]:
    return [
        {
            "valid": False,
            "executed_action_mm": _zero_action(),
            "measured_beam_delta": _zero_metrics(),
            "predicted_beam_delta": _zero_metrics(),
            "prediction_residual": _zero_metrics(),
            "ensemble_uncertainty": _zero_metrics(),
            "padding_reason": "no_history",
        }
        for _ in range(3)
    ]


def _recompute_normalized_error(payload: dict[str, Any]) -> None:
    current = np.asarray(
        [float(payload["current_beam_state"][field]) for field in STATE_FIELDS],
        dtype=np.float64,
    )
    target = np.asarray(
        [float(payload["target_beam_state"][field]) for field in STATE_FIELDS],
        dtype=np.float64,
    )
    tolerances = tolerance_vector(current)
    values = (target - current) / tolerances
    payload["normalized_signed_error"] = {
        field: float(values[index]) for index, field in enumerate(STATE_FIELDS)
    }


def _shuffle_semantic_text(
    semantics: Sequence[Mapping[str, Any]], *, setup_hash: str
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows = [copy.deepcopy(dict(row)) for row in semantics]
    order = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(
            f"{ABLATION_SEED}|semantics|{setup_hash}|{rows[index]['action_id']}".encode()
        ).hexdigest(),
    )
    source_for: dict[int, int] = {
        order[index]: order[(index + 1) % len(order)] for index in range(len(order))
    }
    audit: dict[str, str] = {}
    original = copy.deepcopy(rows)
    for receiver_index, source_index in source_for.items():
        rows[receiver_index]["positive_command_semantics"] = original[source_index][
            "positive_command_semantics"
        ]
        rows[receiver_index]["negative_command_semantics"] = original[source_index][
            "negative_command_semantics"
        ]
        audit[str(rows[receiver_index]["action_id"])] = str(
            original[source_index]["action_id"]
        )
    if any(receiver == source for receiver, source in audit.items()):
        raise AssertionError("actuator semantic shuffle must have no fixed points")
    return rows, audit


def canonical_ablation(value: str) -> str:
    name = "full_input" if value == "full" else str(value)
    if name not in ABLATIONS:
        raise ReasoningAblationError(
            f"unknown ablation {value!r}; expected one of {list(ABLATIONS)}"
        )
    return name


def apply_ablation(
    sample: CandidateEvalSample,
    *,
    ablation: str,
    donor: CandidateEvalSample | None,
    blank_image_path: Path,
    blank_image_sha256: str,
    donor_runtime_input: Mapping[str, Any] | None = None,
    donor_image_path: Path | None = None,
    donor_image_sha256: str | None = None,
) -> AblatedInput:
    """Apply one frozen-name intervention without mutating source records."""

    ablation = canonical_ablation(ablation)
    payload = copy.deepcopy(sample.runtime_input)
    image_path = sample.image_path
    image_hash = sample.image_sha256
    operation: dict[str, Any] = {
        "name": ablation,
        "shuffle_seed": ABLATION_SEED,
        "schema_revalidated": True,
        "relational_consistency": "preserved",
    }
    physical_validation = {
        "actuator_semantics": "not_applicable",
        "counterfactual_physics": "not_claimed",
    }
    donor_payload: dict[str, Any] | None = None
    if ablation in SHUFFLE_ABLATIONS:
        if donor is None:
            raise ReasoningAblationError(f"{ablation} requires a setup-aware donor")
        source = donor.runtime_input if donor_runtime_input is None else donor_runtime_input
        donor_payload = parse_meta_input(source).to_dict()
        operation["donor_sample_id"] = donor.sample_id
        operation["donor_setup_hash"] = donor.setup_hash
        operation["setup_preserving"] = False
        operation["cross_setup"] = True

    if ablation in {"full_input", "reason_codes_disabled"}:
        operation["fields_changed"] = []
        if ablation == "reason_codes_disabled":
            operation["reason_codes_control_use"] = "disabled"
            operation["control_fields"] = "unchanged_from_full_input"
    elif ablation == "blank_image":
        image_path, image_hash = blank_image_path, blank_image_sha256
        operation["fields_changed"] = ["current_beam_image_pixels"]
    elif ablation == "deterministic_image_shuffle":
        assert donor is not None
        image_path = donor.image_path if donor_image_path is None else donor_image_path
        image_hash = donor.image_sha256 if donor_image_sha256 is None else donor_image_sha256
        operation["fields_changed"] = ["current_beam_image_pixels"]
    elif ablation == "metrics_blank":
        # Blank every beam metric channel, including the derived signed error.
        # This is the image/semantics/budget-only condition, not a fabricated
        # physically consistent beam state.
        payload["current_beam_state"] = _zero_metrics()
        payload["target_beam_state"] = _zero_metrics()
        payload["normalized_signed_error"] = _zero_metrics()
        operation["fields_changed"] = [
            "current_beam_state",
            "target_beam_state",
            "normalized_signed_error",
        ]
        operation["relational_consistency"] = "zero_sentinel_not_physical_measurement"
    elif ablation == "current_state_shuffle":
        assert donor_payload is not None
        payload["current_beam_state"] = copy.deepcopy(donor_payload["current_beam_state"])
        _recompute_normalized_error(payload)
        operation["fields_changed"] = [
            "current_beam_state",
            "normalized_signed_error",
        ]
    elif ablation == "target_shuffle":
        assert donor_payload is not None
        payload["target_beam_state"] = copy.deepcopy(donor_payload["target_beam_state"])
        _recompute_normalized_error(payload)
        operation["fields_changed"] = [
            "target_beam_state",
            "normalized_signed_error",
        ]
    elif ablation == "target_replace_current":
        payload["target_beam_state"] = copy.deepcopy(payload["current_beam_state"])
        payload["normalized_signed_error"] = _zero_metrics()
        operation["fields_changed"] = [
            "target_beam_state",
            "normalized_signed_error",
        ]
    elif ablation == "history_blank":
        payload["history"] = _blank_history()
        operation["fields_changed"] = ["history"]
    elif ablation == "history_shuffle":
        assert donor_payload is not None
        payload["history"] = copy.deepcopy(donor_payload["history"])
        operation["fields_changed"] = ["history"]
    elif ablation == "actuator_semantics_shuffle":
        shuffled, source_map = _shuffle_semantic_text(
            payload["actuator_semantics"], setup_hash=sample.setup_hash
        )
        payload["actuator_semantics"] = shuffled
        operation["fields_changed"] = [
            "actuator_semantics.*.positive_command_semantics",
            "actuator_semantics.*.negative_command_semantics",
        ]
        operation["semantic_text_source_action"] = source_map
        operation["identity_bounds_positions_unchanged"] = True
        physical_validation = {
            "actuator_semantics": "unverified",
            "counterfactual_physics": "unverified_do_not_infer_hardware_mapping",
        }
    elif ablation == "uncertainty_blank":
        payload["forward_uncertainty"] = {
            "per_metric": _zero_metrics(),
            "mean": 0.0,
            "maximum": 0.0,
        }
        for history_row in payload["history"]:
            history_row["ensemble_uncertainty"] = _zero_metrics()
        operation["fields_changed"] = [
            "forward_uncertainty",
            "history.*.ensemble_uncertainty",
        ]
    else:  # pragma: no cover - registry exhaustiveness guard
        raise AssertionError(f"unhandled ablation: {ablation}")

    try:
        payload = parse_meta_input(payload).to_dict()
    except MetaContractError as exc:
        raise ReasoningAblationError(
            f"{sample.sample_id}/{ablation}: ablation broke strict input schema: {exc}"
        ) from exc
    image_path = training._guard_candidate_path(
        image_path, role=f"{ablation} generation image", must_exist=True
    )
    if not _valid_sha256(image_hash) or training.sha256_path(image_path) != image_hash:
        raise ReasoningAblationError(
            f"{sample.sample_id}/{ablation}: selected image hash mismatch"
        )
    return AblatedInput(
        ablation=ablation,
        runtime_input=payload,
        image_path=image_path,
        image_sha256=str(image_hash),
        donor_sample_id=None if donor is None else donor.sample_id,
        donor_setup_hash=None if donor is None else donor.setup_hash,
        operation=operation,
        physical_validation=physical_validation,
    )


def ensure_blank_image(path: Path) -> tuple[Path, str]:
    """Create or verify the frozen-size, all-zero candidate blank image."""

    path = training._guard_candidate_path(path, role="ablation blank image", must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    if path.exists():
        with Image.open(path) as opened:
            values = np.asarray(opened.convert("L"), dtype=np.uint8)
        if values.shape != (1024, 1024) or np.any(values != 0):
            raise ReasoningAblationError(
                f"refusing to overwrite noncanonical blank-image asset: {path}"
            )
    else:
        Image.new("L", (1024, 1024), 0).save(path, format="PNG")
    return path, training.sha256_path(path)


def build_generation_prompt(runtime_input: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = parse_meta_input(runtime_input).to_dict()
    prompt = [
        {"role": "system", "content": [{"type": "text", "text": training.SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": training.USER_TEXT_PREFIX + _canonical_json(payload),
                },
            ],
        },
    ]
    training.validate_prompt(prompt)
    return prompt


def _adapter_metadata(adapter: IndependentMetaAdapter) -> dict[str, Any]:
    backend = adapter.backend
    return {
        "adapter_name": adapter.adapter_name,
        "adapter_hashes": dict(getattr(backend, "adapter_hashes", {})),
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def _generate_raw(
    adapter: IndependentMetaAdapter,
    *,
    prompt: list[dict[str, Any]],
    image_path: Path,
    model_seed: int,
) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    raw = adapter.backend(prompt, image_path, int(model_seed), MAX_NEW_TOKENS)
    elapsed = time.perf_counter() - started
    if not isinstance(raw, str):
        raise TypeError("generation backend must return the unmodified decoded string")
    generated = parse_generated_text(raw, latency_seconds=elapsed)
    return raw, generated.to_dict()


def _prediction_row(
    *,
    sample: CandidateEvalSample,
    condition: AblatedInput,
    seed: int,
    raw: str,
    generated: Mapping[str, Any],
    generation_metadata: Mapping[str, Any],
    generation_reused: bool,
) -> dict[str, Any]:
    parsed = generated.get("parsed")
    return {
        "schema_version": "qwen_h1_meta_reasoning_ablation_prediction_v1",
        "sample_id": sample.sample_id,
        "record_id": sample.sample_id,
        "episode_id": sample.sample_id,
        "step": 0,
        "setup_hash": sample.setup_hash,
        "ablation": condition.ablation,
        "ablation_seed": ABLATION_SEED,
        "seed": int(seed),
        # `prediction` is the untouched decoded continuation.  parsed_prediction
        # is audit-only and is never substituted back into the control path.
        "prediction": raw,
        "raw_prediction": raw,
        "parsed_prediction": parsed,
        "valid_json": bool(generated["valid_json"]),
        "parse_error_code": generated.get("error_code"),
        "parse_error_message": generated.get("error_message"),
        "latency_seconds": float(generated["latency_seconds"]),
        "runtime_input_sha256": _sha256_json(condition.runtime_input),
        "manifest_record_sha256": sample.record_sha256,
        "image": {
            "storage_path": str(condition.image_path),
            "sha256": condition.image_sha256,
        },
        "donor": (
            None
            if condition.donor_sample_id is None
            else {
                "sample_id": condition.donor_sample_id,
                "setup_hash": condition.donor_setup_hash,
                "setup_aware": True,
            }
        ),
        "operation": copy.deepcopy(condition.operation),
        "physical_validation": copy.deepcopy(condition.physical_validation),
        "reason_codes_control_use": (
            "disabled"
            if condition.ablation == "reason_codes_disabled"
            else "audit_only_not_direct_action"
        ),
        "control_fields_unchanged_from_full_input": (
            True if condition.ablation == "reason_codes_disabled" else None
        ),
        "source_generation_ablation": (
            "full_input" if condition.ablation == "reason_codes_disabled" else condition.ablation
        ),
        "generation_reused": bool(generation_reused),
        "execution_scope": EXECUTION_SCOPE,
        "hardware_dispatch_allowed": HARDWARE_DISPATCH_ALLOWED,
        "action_execution": ACTION_EXECUTION,
        "formal_frozen_evaluation_enabled": False,
        "generation_metadata": copy.deepcopy(dict(generation_metadata)),
    }


def generate_reasoning_ablation_predictions_for_seed(
    *,
    samples: Sequence[CandidateEvalSample],
    adapter: IndependentMetaAdapter,
    seed: int,
    blank_image_path: Path,
    blank_image_sha256: str,
    generation_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate all conditions for one adapter without loading another model."""

    if int(seed) not in MODEL_SEEDS:
        raise ReasoningAblationError(f"model seed must be one of {list(MODEL_SEEDS)}")
    donors = setup_aware_donor_map(samples)
    metadata = _adapter_metadata(adapter)
    metadata.update(dict(generation_metadata or {}))
    output: list[dict[str, Any]] = []
    full_rows: dict[str, dict[str, Any]] = {}
    for ablation in ABLATIONS:
        for sample in samples:
            donor = donors[sample.sample_id] if ablation in SHUFFLE_ABLATIONS else None
            condition = apply_ablation(
                sample,
                ablation=ablation,
                donor=donor,
                blank_image_path=blank_image_path,
                blank_image_sha256=blank_image_sha256,
            )
            if ablation == "reason_codes_disabled":
                full = full_rows[sample.sample_id]
                generated = {
                    "parsed": copy.deepcopy(full["parsed_prediction"]),
                    "valid_json": full["valid_json"],
                    "error_code": full["parse_error_code"],
                    "error_message": full["parse_error_message"],
                    "latency_seconds": full["latency_seconds"],
                }
                row = _prediction_row(
                    sample=sample,
                    condition=condition,
                    seed=int(seed),
                    raw=str(full["prediction"]),
                    generated=generated,
                    generation_metadata=metadata,
                    generation_reused=True,
                )
                if row["prediction"] != full["prediction"]:
                    raise AssertionError("reason_codes_disabled changed raw full prediction")
            else:
                prompt = build_generation_prompt(condition.runtime_input)
                raw, generated = _generate_raw(
                    adapter,
                    prompt=prompt,
                    image_path=condition.image_path,
                    model_seed=int(seed),
                )
                row = _prediction_row(
                    sample=sample,
                    condition=condition,
                    seed=int(seed),
                    raw=raw,
                    generated=generated,
                    generation_metadata=metadata,
                    generation_reused=False,
                )
            output.append(row)
            if ablation == "full_input":
                full_rows[sample.sample_id] = row
    expected = len(samples) * len(ABLATIONS)
    if len(output) != expected:
        raise AssertionError(f"expected {expected} seed rows, generated {len(output)}")
    return output


def generate_reasoning_ablation_predictions(
    *,
    manifest_records: Sequence[Mapping[str, Any]],
    adapters: Mapping[int, IndependentMetaAdapter],
    blank_image_path: Path,
    verify_images: bool = True,
    require_preregistered_cardinality: bool = True,
    generation_metadata: Mapping[int, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Mock-testable three-adapter initial-state generation harness."""

    if set(int(seed) for seed in adapters) != set(MODEL_SEEDS):
        raise ReasoningAblationError(
            f"exactly the three candidate adapters {list(MODEL_SEEDS)} are required"
        )
    samples = normalize_candidate_eval_manifest(
        manifest_records,
        verify_images=verify_images,
        require_preregistered_cardinality=require_preregistered_cardinality,
    )
    blank_path, blank_hash = ensure_blank_image(blank_image_path)
    rows: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        rows.extend(
            generate_reasoning_ablation_predictions_for_seed(
                samples=samples,
                adapter=adapters[seed],
                seed=seed,
                blank_image_path=blank_path,
                blank_image_sha256=blank_hash,
                generation_metadata=(
                    None if generation_metadata is None else generation_metadata.get(seed)
                ),
            )
        )
    expected = len(samples) * len(ABLATIONS) * len(MODEL_SEEDS)
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} total rows, generated {len(rows)}")
    return rows


def _flatten_control(payload: Mapping[str, Any]) -> dict[str, str]:
    parsed = parse_meta_output(payload).to_dict()
    output = {field: str(parsed[field]) for field in SCALAR_CONTROL_FIELDS}
    for actuator in ACTION_FIELDS:
        output[f"directional_prior.{actuator}"] = str(
            parsed["directional_prior"][actuator]
        )
    return output


def _prediction_regret(record: Mapping[str, Any]) -> float | None:
    value = record.get("configuration_regret")
    if value is None:
        selected = record.get("selected_configuration_cost")
        oracle = record.get("oracle_configuration_cost")
        if selected is not None and oracle is not None:
            value = float(selected) - float(oracle)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < -1e-12:
        raise ReasoningAblationError("configuration regret must be finite and nonnegative")
    return max(0.0, number)


def _validate_prediction_grid(
    records: Sequence[Mapping[str, Any]],
    samples: Sequence[CandidateEvalSample],
    *,
    require_complete_grid: bool,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    sample_ids = {sample.sample_id for sample in samples}
    output: dict[tuple[str, int, str], dict[str, Any]] = {}
    for index, source in enumerate(records):
        row = copy.deepcopy(dict(source))
        sample_id = row.get("sample_id", row.get("record_id"))
        if not isinstance(sample_id, str) or sample_id not in sample_ids:
            raise ReasoningAblationError(
                f"prediction row {index} references unknown candidate sample {sample_id!r}"
            )
        ablation = canonical_ablation(str(row.get("ablation")))
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in MODEL_SEEDS:
            raise ReasoningAblationError(
                f"{sample_id}/{ablation}: invalid model seed {seed!r}"
            )
        step = row.get("step", 0)
        if step != 0:
            raise ReasoningAblationError(
                "offline reasoning diagnostics accept only initial-state step=0 rows"
            )
        if not isinstance(row.get("prediction"), str):
            raise ReasoningAblationError(f"{sample_id}/{ablation}: prediction is not raw text")
        if row.get("execution_scope") != EXECUTION_SCOPE:
            raise ReasoningAblationError(f"{sample_id}/{ablation}: unsafe execution scope")
        if row.get("hardware_dispatch_allowed") is not False:
            raise ReasoningAblationError(
                f"{sample_id}/{ablation}: hardware dispatch must remain false"
            )
        if ablation == "actuator_semantics_shuffle":
            physical = row.get("physical_validation")
            if not isinstance(physical, Mapping) or physical.get("actuator_semantics") != "unverified":
                raise ReasoningAblationError(
                    "actuator_semantics_shuffle must remain physically unverified"
                )
        key = (ablation, int(seed), sample_id)
        if key in output:
            raise ReasoningAblationError(f"duplicate prediction grid key: {key}")
        output[key] = row
    if require_complete_grid:
        expected = {
            (ablation, seed, sample_id)
            for ablation in ABLATIONS
            for seed in MODEL_SEEDS
            for sample_id in sample_ids
        }
        missing = expected - set(output)
        extra = set(output) - expected
        if missing or extra:
            raise ReasoningAblationError(
                f"incomplete ablation grid: missing={len(missing)}, extra={len(extra)}"
            )
    for seed in MODEL_SEEDS:
        for sample_id in sample_ids:
            full = output.get(("full_input", seed, sample_id))
            disabled = output.get(("reason_codes_disabled", seed, sample_id))
            if full is not None and disabled is not None:
                if disabled["prediction"] != full["prediction"]:
                    raise ReasoningAblationError(
                        "reason_codes_disabled must preserve the exact full raw prediction"
                    )
                if disabled.get("control_fields_unchanged_from_full_input") is not True:
                    raise ReasoningAblationError(
                        "reason_codes_disabled lacks the control-field invariance marker"
                    )
    return output


def _control_diagnostics(
    *,
    samples: Sequence[CandidateEvalSample],
    grid: Mapping[tuple[str, int, str], Mapping[str, Any]],
    ablation: str,
) -> dict[str, Any]:
    truth = {sample.sample_id: _flatten_control(sample.oracle_output) for sample in samples}
    field_names = tuple(next(iter(truth.values())))
    per_seed: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        correct = {field: 0 for field in field_names}
        valid = 0
        configuration_exact = 0
        reason_exact = 0
        regrets: list[float] = []
        for sample in samples:
            row = grid[(ablation, seed, sample.sample_id)]
            try:
                parsed = parse_meta_output(row["prediction"]).to_dict()
            except MetaContractError:
                parsed = None
            if parsed is not None:
                valid += 1
                flattened = _flatten_control(parsed)
                for field in field_names:
                    correct[field] += int(flattened[field] == truth[sample.sample_id][field])
                configuration_exact += int(flattened == truth[sample.sample_id])
                reason_exact += int(parsed["reason_codes"] == sample.oracle_output["reason_codes"])
            regret = _prediction_regret(row)
            if regret is not None:
                regrets.append(regret)
        total = len(samples)
        per_seed.append(
            {
                "seed": seed,
                "records": total,
                "valid_json_rate": valid / total,
                "control_configuration_exact_match": configuration_exact / total,
                "reason_codes_exact_match": reason_exact / total,
                "field_accuracy": {
                    field: correct[field] / total for field in field_names
                },
                "configuration_regret": {
                    "evaluated_count": len(regrets),
                    "coverage_rate": len(regrets) / total,
                    "mean": statistics.fmean(regrets) if regrets else None,
                    "median": statistics.median(regrets) if regrets else None,
                    "maximum": max(regrets) if regrets else None,
                },
            }
        )
    return {
        "reason_codes_are_control_fields": False,
        "control_field_names": list(field_names),
        "per_seed": per_seed,
        "aggregate": {
            "control_configuration_exact_match_mean": statistics.fmean(
                row["control_configuration_exact_match"] for row in per_seed
            ),
            "field_accuracy_mean": {
                field: statistics.fmean(row["field_accuracy"][field] for row in per_seed)
                for field in field_names
            },
            "configuration_regret_mean": (
                statistics.fmean(
                    row["configuration_regret"]["mean"]
                    for row in per_seed
                    if row["configuration_regret"]["mean"] is not None
                )
                if any(
                    row["configuration_regret"]["mean"] is not None for row in per_seed
                )
                else None
            ),
        },
    }


def _paired_vs_full(
    *,
    samples: Sequence[CandidateEvalSample],
    grid: Mapping[tuple[str, int, str], Mapping[str, Any]],
    ablation: str,
) -> dict[str, Any]:
    field_names = list(_flatten_control(samples[0].oracle_output))
    per_seed: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        changed = {field: 0 for field in field_names}
        comparable = 0
        paired_regret_deltas: list[float] = []
        regret_by_field: dict[str, dict[str, list[float]]] = {
            field: {"changed": [], "unchanged": []} for field in field_names
        }
        for sample in samples:
            full_row = grid[("full_input", seed, sample.sample_id)]
            ablated_row = grid[(ablation, seed, sample.sample_id)]
            try:
                full = _flatten_control(parse_meta_output(full_row["prediction"]).to_dict())
                altered = _flatten_control(parse_meta_output(ablated_row["prediction"]).to_dict())
            except MetaContractError:
                full = altered = None
            if full is not None and altered is not None:
                comparable += 1
                for field in field_names:
                    changed[field] += int(full[field] != altered[field])
            full_regret = _prediction_regret(full_row)
            altered_regret = _prediction_regret(ablated_row)
            if full_regret is not None and altered_regret is not None:
                delta = altered_regret - full_regret
                paired_regret_deltas.append(delta)
                if full is not None and altered is not None:
                    for field in field_names:
                        group = "changed" if full[field] != altered[field] else "unchanged"
                        regret_by_field[field][group].append(delta)
        total = len(samples)
        per_seed.append(
            {
                "seed": seed,
                "records": total,
                "comparable_valid_pairs": comparable,
                "field_change_rate": {
                    field: changed[field] / total for field in field_names
                },
                "configuration_regret_delta_ablated_minus_full": {
                    "evaluated_count": len(paired_regret_deltas),
                    "coverage_rate": len(paired_regret_deltas) / total,
                    "mean": (
                        statistics.fmean(paired_regret_deltas)
                        if paired_regret_deltas
                        else None
                    ),
                    "median": (
                        statistics.median(paired_regret_deltas)
                        if paired_regret_deltas
                        else None
                    ),
                },
                "field_regret_diagnostics": {
                    field: {
                        group: {
                            "count": len(values),
                            "mean_regret_delta": (
                                statistics.fmean(values) if values else None
                            ),
                        }
                        for group, values in groups.items()
                    }
                    for field, groups in regret_by_field.items()
                },
            }
        )
    return {
        "direction": "ablated_minus_full",
        "per_seed": per_seed,
        "aggregate_field_change_rate": {
            field: statistics.fmean(row["field_change_rate"][field] for row in per_seed)
            for field in field_names
        },
        "configuration_regret_status": (
            "evaluated_in_supplied_simulator_or_shadow_records"
            if any(
                row["configuration_regret_delta_ablated_minus_full"]["evaluated_count"]
                for row in per_seed
            )
            else "unverified_no_regret_records_supplied"
        ),
    }


def evaluate_reasoning_ablations(
    *,
    manifest_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    seed_metadata: Mapping[int, Mapping[str, Any]] | None = None,
    require_preregistered_cardinality: bool = True,
) -> dict[str, Any]:
    """Evaluate every ablation and report field/configuration diagnostics."""

    samples = normalize_candidate_eval_manifest(
        manifest_records,
        verify_images=False,
        require_preregistered_cardinality=require_preregistered_cardinality,
    )
    grid = _validate_prediction_grid(
        prediction_records,
        samples,
        require_complete_grid=True,
    )
    manifest_for_offline = [sample.record for sample in samples]
    per_ablation: dict[str, Any] = {}
    for ablation in ABLATIONS:
        rows = [
            grid[(ablation, seed, sample.sample_id)]
            for seed in MODEL_SEEDS
            for sample in samples
        ]
        try:
            offline = evaluate_candidate_eval_predictions(
                manifest_records=manifest_for_offline,
                prediction_records=rows,
                expected_seeds=MODEL_SEEDS,
                seed_metadata=seed_metadata,
                require_preregistered_cardinality=require_preregistered_cardinality,
            )
        except OfflineEvaluationError as exc:
            raise ReasoningAblationError(
                f"offline evaluator rejected ablation {ablation}: {exc}"
            ) from exc
        per_ablation[ablation] = {
            "offline_evaluation": offline,
            "control_field_diagnostics": _control_diagnostics(
                samples=samples, grid=grid, ablation=ablation
            ),
            "paired_vs_full": _paired_vs_full(
                samples=samples, grid=grid, ablation=ablation
            ),
            "physical_validation": (
                {
                    "actuator_semantics": "unverified",
                    "claim_allowed": False,
                    "reason": "visible semantic reassignment is a counterfactual; no hardware mapping was measured",
                }
                if ablation == "actuator_semantics_shuffle"
                else {"status": "not_applicable"}
            ),
        }
    return {
        "version": "qwen_h1_meta_reasoning_ablation_offline_v1",
        "candidate_only": True,
        "scope": "fresh candidate_eval; simulator/shadow only; no hardware dispatch",
        "formal_frozen_evaluation_enabled": False,
        "execution_policy": {
            "scope": EXECUTION_SCOPE,
            "hardware_dispatch_allowed": HARDWARE_DISPATCH_ALLOWED,
            "missing_closed_loop_steps_count_as_qwen_performance": False,
        },
        "ablation_seed": ABLATION_SEED,
        "expected_seeds": list(MODEL_SEEDS),
        "ablations": list(ABLATIONS),
        "manifest_records": len(samples),
        "prediction_records": len(prediction_records),
        "full_input_offline_evaluation": per_ablation["full_input"][
            "offline_evaluation"
        ],
        "per_ablation": per_ablation,
        "closed_loop_trace_selector": {
            "keys": ["ablation", "seed", "episode_id", "step"],
            "raw_field": "prediction",
            "requires_dynamic_state_image_binding_for_live_claim": True,
        },
    }


def build_not_run_due_to_data_gate_report(
    *,
    gate_evidence: Mapping[str, Any],
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Build an explicit unavailable artifact after the preregistered RED gate.

    This contains no placeholder predictions, metrics, regret, or claimed model
    behavior.  It exists so downstream reporting distinguishes an intentional
    safety stop from a missing/corrupt result file.
    """

    evidence = copy.deepcopy(dict(gate_evidence))
    if not evidence:
        raise ReasoningAblationError("data-gate evidence must not be empty")
    manifest: dict[str, Any] | None = None
    if manifest_path is not None:
        resolved = training._guard_candidate_path(
            manifest_path, role="not-run candidate_eval manifest"
        )
        manifest = {
            "path": str(resolved),
            "sha256": training.sha256_path(resolved),
        }
    unavailable = {
        ablation: {
            "status": "unavailable",
            "reason": "not_run_due_to_information_sufficiency_gate",
            "predictions": None,
            "field_diagnostics": None,
            "configuration_regret": None,
            "closed_loop_performance": None,
            "fallback_counted_as_qwen_performance": False,
            "physical_validation": (
                "unverified"
                if ablation == "actuator_semantics_shuffle"
                else "not_run"
            ),
        }
        for ablation in ABLATIONS
    }
    return {
        # Keep the reporter-compatible result version; status/reason distinguish
        # this intentional no-run artifact from an evaluated result.
        "version": "qwen_h1_meta_reasoning_ablation_offline_v1",
        "candidate_only": True,
        "status": "unavailable",
        "reason": "not_run_due_to_information_sufficiency_gate",
        "reason_alias": "not_run_due_to_data_gate",
        "scientific_conclusion": False,
        "formal_frozen_evaluation_enabled": False,
        "execution_policy": {
            "scope": EXECUTION_SCOPE,
            "hardware_dispatch_allowed": HARDWARE_DISPATCH_ALLOWED,
        },
        "ablation_seed": ABLATION_SEED,
        "expected_seeds": list(MODEL_SEEDS),
        "ablations": list(ABLATIONS),
        "model_loaded": False,
        "inference_started": False,
        "predictions_generated": False,
        "closed_loop_started": False,
        "regret_evaluated": False,
        "manifest": manifest,
        "data_gate": {
            "status": "RED",
            "on_failure": PROTOCOL["information_sufficiency_gate"]["on_failure"],
            "evidence": evidence,
        },
        "full_input_offline_evaluation": None,
        "per_ablation": unavailable,
    }


def merge_configuration_regret(
    prediction_records: Sequence[Mapping[str, Any]],
    regret_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach externally measured simulator/shadow regret without fabrication."""

    mapping: dict[tuple[str, int, str, int], Mapping[str, Any]] = {}
    for index, row in enumerate(regret_records):
        sample_id = row.get("sample_id", row.get("record_id", row.get("episode_id")))
        ablation = canonical_ablation(str(row.get("ablation")))
        seed = row.get("seed")
        step = row.get("step", 0)
        if not isinstance(sample_id, str) or not sample_id:
            raise ReasoningAblationError(f"regret row {index} has no sample identity")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in MODEL_SEEDS:
            raise ReasoningAblationError(f"regret row {index} has invalid seed")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ReasoningAblationError(f"regret row {index} has invalid step")
        if row.get("execution_scope") != EXECUTION_SCOPE:
            raise ReasoningAblationError(
                f"regret row {index} was not measured in simulator/shadow scope"
            )
        if row.get("hardware_dispatch_allowed") is not False:
            raise ReasoningAblationError(f"regret row {index} permits hardware dispatch")
        _prediction_regret(row)
        key = (ablation, int(seed), sample_id, int(step))
        if key in mapping:
            raise ReasoningAblationError(f"duplicate regret key: {key}")
        mapping[key] = row
    output: list[dict[str, Any]] = []
    matched: set[tuple[str, int, str, int]] = set()
    for source in prediction_records:
        row = copy.deepcopy(dict(source))
        sample_id = str(row.get("sample_id", row.get("record_id", row.get("episode_id"))))
        key = (
            canonical_ablation(str(row.get("ablation"))),
            int(row["seed"]),
            sample_id,
            int(row.get("step", 0)),
        )
        regret = mapping.get(key)
        if regret is not None:
            matched.add(key)
            if regret.get("configuration_regret") is not None:
                row["configuration_regret"] = float(regret["configuration_regret"])
            else:
                row["selected_configuration_cost"] = float(
                    regret["selected_configuration_cost"]
                )
                row["oracle_configuration_cost"] = float(
                    regret["oracle_configuration_cost"]
                )
            row["regret_provenance"] = "externally_supplied_simulator_or_shadow"
        output.append(row)
    unmatched = set(mapping) - matched
    if unmatched:
        raise ReasoningAblationError(
            f"regret records reference {len(unmatched)} predictions that were not supplied"
        )
    return output


def select_closed_loop_trace(
    records: Sequence[Mapping[str, Any]], *, ablation: str, seed: int
) -> list[dict[str, Any]]:
    """Select raw rows in the exact shape consumed by ``TracePolicy``."""

    ablation = canonical_ablation(ablation)
    if int(seed) not in MODEL_SEEDS:
        raise ReasoningAblationError(f"seed must be one of {list(MODEL_SEEDS)}")
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for source in records:
        if source.get("ablation") != ablation or source.get("seed") != int(seed):
            continue
        if source.get("execution_scope") != EXECUTION_SCOPE or source.get(
            "hardware_dispatch_allowed"
        ) is not False:
            raise ReasoningAblationError("unsafe row rejected by closed-loop selector")
        episode_id = source.get("episode_id", source.get("record_id", source.get("sample_id")))
        step = source.get("step")
        if not isinstance(episode_id, str) or not episode_id:
            raise ReasoningAblationError("closed-loop row has no episode_id")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ReasoningAblationError("closed-loop row has no valid step")
        key = (episode_id, step)
        if key in seen:
            raise ReasoningAblationError(f"duplicate closed-loop trace key: {key}")
        seen.add(key)
        row = copy.deepcopy(dict(source))
        row["episode_id"] = episode_id
        selected.append(row)
    return sorted(selected, key=lambda row: (str(row["episode_id"]), int(row["step"])))


class LiveReasoningAblationPolicy:
    """Image-aware live policy for the candidate closed-loop evaluator.

    A single instance owns one adapter seed and one ablation.  This makes it
    practical to run methods sequentially without loading three base models.
    Cross-setup dynamic donors are supplied by callbacks.  If no dynamic donor
    runtime provider exists, the initial candidate-manifest donor is used and
    the trace says ``initial_manifest_only_unverified``.  Image shuffle is
    stricter: absence of a donor-image provider makes the policy unavailable,
    so a fallback can never be counted as Qwen performance.
    """

    requires_sensor_image = True
    live_image_generation = True
    execution_scope = EXECUTION_SCOPE
    hardware_dispatch_allowed = HARDWARE_DISPATCH_ALLOWED

    def __init__(
        self,
        *,
        adapter: IndependentMetaAdapter,
        seed: int,
        ablation: str,
        manifest_records: Sequence[Mapping[str, Any]],
        donor_runtime_provider: DonorRuntimeProvider | None = None,
        donor_image_provider: DonorImageProvider | None = None,
        trace_sink: MutableSequence[dict[str, Any]] | Callable[[dict[str, Any]], None] | None = None,
        require_preregistered_cardinality: bool = True,
    ) -> None:
        if int(seed) not in MODEL_SEEDS:
            raise ReasoningAblationError(f"seed must be one of {list(MODEL_SEEDS)}")
        self.adapter = adapter
        self.seed = int(seed)
        self.ablation = canonical_ablation(ablation)
        self.samples = normalize_candidate_eval_manifest(
            manifest_records,
            verify_images=False,
            require_preregistered_cardinality=require_preregistered_cardinality,
        )
        self.by_id = {sample.sample_id: sample for sample in self.samples}
        self.donors = setup_aware_donor_map(self.samples)
        self.donor_runtime_provider = donor_runtime_provider
        self.donor_image_provider = donor_image_provider
        self.trace_sink = trace_sink

    def _emit(self, row: dict[str, Any]) -> None:
        if self.trace_sink is None:
            return
        if callable(self.trace_sink):
            self.trace_sink(copy.deepcopy(row))
        else:
            self.trace_sink.append(copy.deepcopy(row))

    @staticmethod
    def _array_to_png(image: np.ndarray, directory: Path) -> tuple[Path, str]:
        from PIL import Image

        values = np.asarray(image, dtype=np.float64)
        if values.shape != (1024, 1024) or not np.isfinite(values).all():
            raise ReasoningAblationError(
                "live simulator sensor image must be finite and exactly 1024x1024"
            )
        pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
        path = directory / "current_sensor_frame.png"
        Image.fromarray(pixels, mode="L").save(path, format="PNG")
        return path, training.sha256_path(path)

    def decide(self, context: Any) -> Any:
        # Imported lazily to keep this module usable for offline generation and
        # mock tests without initializing a physical simulator backend.
        from .evaluate_closed_loop import (
            MetaPolicyResult,
            runtime_input_fingerprint,
            sensor_image_fingerprint,
        )

        sample = self.by_id.get(str(context.episode_id))
        if sample is None:
            return MetaPolicyResult(
                payload=None,
                available=False,
                unavailable_reason="live_ablation_unknown_candidate_episode",
            )
        if context.sensor_image_normalized is None:
            return MetaPolicyResult(
                payload=None,
                available=False,
                unavailable_reason="live_ablation_sensor_image_unavailable",
            )
        donor = self.donors[sample.sample_id] if self.ablation in SHUFFLE_ABLATIONS else None
        donor_runtime: Mapping[str, Any] | None = None
        donor_alignment = "not_applicable"
        if donor is not None and self.ablation != "deterministic_image_shuffle":
            if self.donor_runtime_provider is not None:
                donor_runtime = self.donor_runtime_provider(context, donor.record)
            if donor_runtime is None:
                donor_runtime = donor.runtime_input
                donor_alignment = "initial_manifest_only_unverified"
            else:
                donor_alignment = "dynamic_provider_supplied"

        own_image = np.asarray(context.sensor_image_normalized, dtype=np.float64)
        displayed_image = own_image
        donor_image_status = "not_applicable"
        if self.ablation == "blank_image":
            displayed_image = np.zeros((1024, 1024), dtype=np.float64)
        elif self.ablation == "deterministic_image_shuffle":
            assert donor is not None
            if self.donor_image_provider is None:
                return MetaPolicyResult(
                    payload=None,
                    available=False,
                    unavailable_reason="live_image_shuffle_requires_donor_image_provider",
                )
            supplied = self.donor_image_provider(context, donor.record)
            if supplied is None:
                return MetaPolicyResult(
                    payload=None,
                    available=False,
                    unavailable_reason="live_image_shuffle_donor_image_unavailable",
                )
            displayed_image = np.asarray(supplied, dtype=np.float64)
            donor_image_status = "dynamic_provider_supplied"

        with tempfile.TemporaryDirectory(
            prefix=".reasoning_ablation_live_", dir=training.PACKAGE_ROOT
        ) as directory_name:
            directory = Path(directory_name)
            own_path, own_hash = self._array_to_png(own_image, directory)
            blank_path = directory / "blank_sensor_frame.png"
            from PIL import Image

            Image.new("L", (1024, 1024), 0).save(blank_path, format="PNG")
            blank_hash = training.sha256_path(blank_path)
            displayed_path = own_path
            displayed_hash = own_hash
            if self.ablation in {"blank_image", "deterministic_image_shuffle"}:
                displayed_path, displayed_hash = self._array_to_png(
                    displayed_image, directory
                )
            condition = apply_ablation(
                CandidateEvalSample(
                    sample_id=sample.sample_id,
                    setup_hash=sample.setup_hash,
                    runtime_input=parse_meta_input(context.runtime_input).to_dict(),
                    oracle_output=sample.oracle_output,
                    image_path=own_path,
                    image_sha256=own_hash,
                    record_sha256=sample.record_sha256,
                    record=sample.record,
                ),
                ablation=self.ablation,
                donor=donor,
                blank_image_path=blank_path,
                blank_image_sha256=blank_hash,
                donor_runtime_input=donor_runtime,
                donor_image_path=(
                    displayed_path
                    if self.ablation == "deterministic_image_shuffle"
                    else None
                ),
                donor_image_sha256=(
                    displayed_hash
                    if self.ablation == "deterministic_image_shuffle"
                    else None
                ),
            )
            prompt = build_generation_prompt(condition.runtime_input)
            raw, generated = _generate_raw(
                self.adapter,
                prompt=prompt,
                image_path=condition.image_path,
                model_seed=self.seed,
            )

        trace = {
            "schema_version": "qwen_h1_meta_reasoning_ablation_live_trace_v1",
            "sample_id": sample.sample_id,
            "record_id": sample.sample_id,
            "episode_id": sample.sample_id,
            "step": int(context.step),
            "setup_hash": sample.setup_hash,
            "ablation": self.ablation,
            "ablation_seed": ABLATION_SEED,
            "seed": self.seed,
            "prediction": raw,
            "raw_prediction": raw,
            "parsed_prediction": generated.get("parsed"),
            "valid_json": bool(generated["valid_json"]),
            "parse_error_code": generated.get("error_code"),
            "parse_error_message": generated.get("error_message"),
            "latency_seconds": float(generated["latency_seconds"]),
            "runtime_input_sha256": runtime_input_fingerprint(context.runtime_input),
            "ablated_runtime_input_sha256": _sha256_json(condition.runtime_input),
            "sensor_image_fingerprint_sha256": sensor_image_fingerprint(
                context.sensor_image_normalized
            ),
            "displayed_image_fingerprint_sha256": sensor_image_fingerprint(
                displayed_image
            ),
            "donor": (
                None
                if donor is None
                else {
                    "sample_id": donor.sample_id,
                    "setup_hash": donor.setup_hash,
                    "runtime_alignment": donor_alignment,
                    "image_status": donor_image_status,
                }
            ),
            "operation": copy.deepcopy(condition.operation),
            "physical_validation": copy.deepcopy(condition.physical_validation),
            "reason_codes_control_use": (
                "disabled"
                if self.ablation == "reason_codes_disabled"
                else "audit_only_not_direct_action"
            ),
            "execution_scope": EXECUTION_SCOPE,
            "hardware_dispatch_allowed": HARDWARE_DISPATCH_ALLOWED,
            "action_execution": ACTION_EXECUTION,
            "formal_frozen_evaluation_enabled": False,
            "generation_metadata": _adapter_metadata(self.adapter),
        }
        self._emit(trace)
        return MetaPolicyResult(
            payload=raw,
            latency_seconds=float(generated["latency_seconds"]),
        )


def _read_records(path: Path, *, role: str) -> list[dict[str, Any]]:
    path = training._guard_candidate_path(path, role=role)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ReasoningAblationError(f"empty input: {path}")
    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        whole = None
    if isinstance(whole, list):
        records = whole
    elif isinstance(whole, Mapping) and isinstance(whole.get("records"), list):
        records = whole["records"]
    elif whole is None:
        try:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise ReasoningAblationError(f"invalid JSONL: {path}") from exc
    elif isinstance(whole, Mapping):
        records = [whole]
    else:
        raise ReasoningAblationError(f"records must be JSON objects: {path}")
    if not all(isinstance(row, dict) for row in records):
        raise ReasoningAblationError(f"records must be JSON objects: {path}")
    return records


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_seed_paths(values: Sequence[str], *, role: str) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ReasoningAblationError(f"--{role} must be SEED=/candidate/path")
        raw_seed, raw_path = raw.split("=", 1)
        try:
            seed = int(raw_seed)
        except ValueError as exc:
            raise ReasoningAblationError(f"invalid {role} seed: {raw_seed!r}") from exc
        if seed not in MODEL_SEEDS or seed in output:
            raise ReasoningAblationError(
                f"{role} seeds must be unique members of {list(MODEL_SEEDS)}"
            )
        output[seed] = training._guard_candidate_path(
            Path(raw_path), role=f"{role} for seed {seed}"
        )
    if set(output) != set(MODEL_SEEDS):
        raise ReasoningAblationError(
            f"exactly three --{role} values are required for {list(MODEL_SEEDS)}"
        )
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate all initial-state ablations")
    generate.add_argument("--manifest", type=Path, required=True)
    generate.add_argument(
        "--config", action="append", default=[], metavar="SEED=/candidate/config.yaml"
    )
    generate.add_argument(
        "--adapter", action="append", default=[], metavar="SEED=/candidate/adapter"
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--local-rank", type=int)

    evaluate = subparsers.add_parser("evaluate", help="evaluate supplied raw ablation rows")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--regret", type=Path)
    evaluate.add_argument("--seed-metadata", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)

    select = subparsers.add_parser("select-trace", help="filter by ablation and seed")
    select.add_argument("--predictions", type=Path, required=True)
    select.add_argument("--ablation", choices=("full",) + ABLATIONS, required=True)
    select.add_argument("--seed", type=int, choices=MODEL_SEEDS, required=True)
    select.add_argument("--output", type=Path, required=True)

    not_run = subparsers.add_parser(
        "not-run", help="write explicit unavailable result after the RED data gate"
    )
    not_run.add_argument("--gate-report", type=Path, required=True)
    not_run.add_argument("--manifest", type=Path)
    not_run.add_argument("--output", type=Path, required=True)
    return parser


def _main_generate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = training._guard_candidate_path(args.manifest, role="candidate_eval manifest")
    if "candidate_eval" not in manifest_path.name.lower():
        raise ReasoningAblationError(
            "manifest filename must explicitly identify candidate_eval"
        )
    output = training._guard_candidate_path(
        args.output, role="reasoning ablation predictions", must_exist=False
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate predictions: {output}")
    configs = _parse_seed_paths(args.config, role="config")
    adapters = _parse_seed_paths(args.adapter, role="adapter")
    records = _read_records(manifest_path, role="candidate_eval manifest")
    samples = normalize_candidate_eval_manifest(
        records, verify_images=True, require_preregistered_cardinality=True
    )
    blank_path, blank_hash = ensure_blank_image(
        output.parent / "reasoning_ablation_assets" / "blank_1024.png"
    )
    rows: list[dict[str, Any]] = []
    # Intentionally load and release one adapter at a time.  This command never
    # retains three quantized base models in GPU memory.
    for seed in MODEL_SEEDS:
        config = training.base.load_yaml(configs[seed])
        training.validate_candidate_config(config)
        adapter = load_independent_adapter(
            config=config,
            adapter_path=adapters[seed],
            local_rank=args.local_rank,
        )
        rows.extend(
            generate_reasoning_ablation_predictions_for_seed(
                samples=samples,
                adapter=adapter,
                seed=seed,
                blank_image_path=blank_path,
                blank_image_sha256=blank_hash,
                generation_metadata={
                    "config_path": str(configs[seed]),
                    "config_sha256": training.sha256_path(configs[seed]),
                    "adapter_path": str(adapters[seed]),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": training.sha256_path(manifest_path),
                },
            )
        )
        del adapter
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    _atomic_jsonl(output, rows)
    return {
        "status": "candidate_reasoning_ablation_generation_complete",
        "output": str(output),
        "records": len(rows),
        "manifest_records": len(samples),
        "ablations": list(ABLATIONS),
        "seeds": list(MODEL_SEEDS),
        "execution_scope": EXECUTION_SCOPE,
        "hardware_dispatch_allowed": False,
        "frozen_or_protected_predictions_opened": False,
    }


def _main_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _read_records(args.manifest, role="candidate_eval manifest")
    predictions = _read_records(args.predictions, role="reasoning predictions")
    if args.regret is not None:
        predictions = merge_configuration_regret(
            predictions, _read_records(args.regret, role="simulator regret")
        )
    seed_metadata = None
    if args.seed_metadata is not None:
        metadata_path = training._guard_candidate_path(
            args.seed_metadata, role="training seed metadata"
        )
        decoded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(decoded, Mapping):
            raise ReasoningAblationError("seed metadata must be a JSON object")
        seed_metadata = {int(seed): dict(value) for seed, value in decoded.items()}
    report = evaluate_reasoning_ablations(
        manifest_records=manifest,
        prediction_records=predictions,
        seed_metadata=seed_metadata,
        require_preregistered_cardinality=True,
    )
    output = training._guard_candidate_path(
        args.output, role="reasoning evaluation output", must_exist=False
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reasoning report: {output}")
    _atomic_json(output, report)
    return {
        "status": "candidate_reasoning_ablation_evaluation_complete",
        "output": str(output),
        "records": len(predictions),
        "execution_scope": EXECUTION_SCOPE,
        "hardware_dispatch_allowed": False,
    }


def _main_select(args: argparse.Namespace) -> dict[str, Any]:
    predictions = _read_records(args.predictions, role="reasoning predictions")
    rows = select_closed_loop_trace(
        predictions, ablation=args.ablation, seed=args.seed
    )
    output = training._guard_candidate_path(
        args.output, role="closed-loop trace selection", must_exist=False
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selected trace: {output}")
    _atomic_jsonl(output, rows)
    return {
        "status": "candidate_closed_loop_trace_selected",
        "output": str(output),
        "records": len(rows),
        "ablation": canonical_ablation(args.ablation),
        "seed": int(args.seed),
        "warning": "static initial-state rows contain step=0 only; missing dynamic steps must never count as Qwen performance",
    }


def _main_not_run(args: argparse.Namespace) -> dict[str, Any]:
    gate_path = training._guard_candidate_path(
        args.gate_report, role="information-sufficiency gate report"
    )
    decoded = json.loads(gate_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise ReasoningAblationError("information-sufficiency gate report must be an object")
    report = build_not_run_due_to_data_gate_report(
        gate_evidence={
            "report_path": str(gate_path),
            "report_sha256": training.sha256_path(gate_path),
            "report": decoded,
        },
        manifest_path=args.manifest,
    )
    output = training._guard_candidate_path(
        args.output, role="reasoning not-run output", must_exist=False
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reasoning not-run report: {output}")
    _atomic_json(output, report)
    return {
        "status": "unavailable",
        "reason": "not_run_due_to_information_sufficiency_gate",
        "output": str(output),
        "model_loaded": False,
        "inference_started": False,
        "closed_loop_started": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "generate":
        summary = _main_generate(args)
    elif args.command == "evaluate":
        summary = _main_evaluate(args)
    elif args.command == "select-trace":
        summary = _main_select(args)
    elif args.command == "not-run":
        summary = _main_not_run(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "ABLATIONS",
    "ABLATION_SEED",
    "ACTION_EXECUTION",
    "CandidateEvalSample",
    "EXECUTION_SCOPE",
    "HARDWARE_DISPATCH_ALLOWED",
    "LiveReasoningAblationPolicy",
    "MODEL_SEEDS",
    "ReasoningAblationError",
    "apply_ablation",
    "build_generation_prompt",
    "build_not_run_due_to_data_gate_report",
    "canonical_ablation",
    "ensure_blank_image",
    "evaluate_reasoning_ablations",
    "generate_reasoning_ablation_predictions",
    "generate_reasoning_ablation_predictions_for_seed",
    "merge_configuration_regret",
    "normalize_candidate_eval_manifest",
    "select_closed_loop_trace",
    "setup_aware_donor_map",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
