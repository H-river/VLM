#!/usr/bin/env python3
"""Candidate-only paired closed-loop evaluation on fresh candidate records.

This module never opens the image paths embedded in a manifest.  Meta-model
outputs come either from a live image-aware policy hook or state/image-bound
per-step traces, while the unchanged numeric forward ensemble and corrected
v12.1 simulator own prediction and physical transitions.  Missing traces remain
explicit ``null + reason`` in the method summary; a safe default fallback is
audited but is never reported as the missing method's performance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    Bounds,
    action_dict,
    action_vector,
    apply_action,
    metrics_dict,
    metrics_vector,
    position_dict,
    position_vector,
    stable_seed,
    tolerance_vector,
)
from continuous_control_v12.mpc import Predictor, learned_predictor
from continuous_control_v12.simulator import is_corrected_semantics, simulate_state
from continuous_control_v12.world_model import load_forward_ensemble
from qwen_vl_supervisor_v1.closed_loop_adapter import (
    adapt_supervisor_decision_or_safe_stop,
)

from .baselines import RuleMetaBaseline
from .compiler import (
    default_h1_config,
    dual_budget_h1_config,
    validate_preregistered_controller_config,
)
from .contracts import MetaControllerDecision, load_protocol_json, parse_meta_input
from .controller import (
    ActionEvaluator,
    MetaH1Controller,
    PhysicalActionEvaluation,
    forward_ensemble_action_evaluator,
)
from .integration import LoopBudget, integrate_supervisor_step


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
PROTOCOL_PATH = PACKAGE_ROOT / "protocol" / "meta_controller_protocol.json"
DEFAULT_FORWARD_CHECKPOINT = (
    REPOSITORY_ROOT
    / "runs/overnight_v12_semantics_20260731_002709/models/lc_128g_v2/continuous_forward_v12_128g.pt"
)
DEFAULT_SIMULATOR_CONFIG = (
    REPOSITORY_ROOT / "continuous_control_v12/config_v12_semantics_v2.json"
)
DEFAULT_BASE_CONFIG = REPOSITORY_ROOT / "optical_sim/configs/base_config.yaml"

HORIZON_STEPS = 4
PLANNER_SEED_ROOT = 2026080221
BOOTSTRAP_SEED = 2026080212
BOOTSTRAP_SAMPLES = 4000
QWEN_SEEDS = (2026080201, 2026080202, 2026080203)
TARGET_SEED = 2026080210
ORACLE_SEED = 2026080211
STANDARD_METHOD_NAMES = (
    "default_h1",
    "dual_budget_default_h1",
    "rule_guided_h1",
    "metrics_mlp_meta_h1",
    "qwen_guided_h1_seed_2026080201",
    "qwen_guided_h1_seed_2026080202",
    "qwen_guided_h1_seed_2026080203",
    "oracle_guided_h1",
    "random_or_frequency_meta_h1",
    "shadow_qwen_h1",
)


class ClosedLoopEvaluationError(ValueError):
    """Candidate evaluator input or fairness contract violation."""


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ClosedLoopEvaluationError(
                    f"{path}:{line_number}: invalid JSONL record"
                ) from exc
            if not isinstance(value, dict):
                raise ClosedLoopEvaluationError(
                    f"{path}:{line_number}: record must be an object"
                )
            rows.append(value)
    return rows


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_closed_loop_source_freeze(
    *,
    forward_checkpoint_path: Path,
    simulator_config_path: Path,
    base_config_path: Path,
) -> dict[str, dict[str, str]]:
    """Verify every numeric-control source and canonical config before loading."""

    source = load_protocol_json("meta_controller_protocol.json")["source_freeze"]

    def repository_path(raw: Any) -> Path:
        candidate = Path(str(raw)).expanduser()
        return (candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate).resolve()

    expected_paths = {
        "default_h1_protocol": repository_path(source["default_h1_protocol"]),
        "cem_source": (REPOSITORY_ROOT / "continuous_control_v12/mpc.py").resolve(),
        "world_model_source": (
            REPOSITORY_ROOT / "continuous_control_v12/world_model.py"
        ).resolve(),
        "forward_checkpoint": repository_path(source["forward_checkpoint"]),
        "simulator_config": repository_path(source["simulator_config"]),
    }
    expected_hashes = {
        "default_h1_protocol": str(source["default_h1_protocol_sha256"]),
        "cem_source": str(source["cem_source_sha256"]),
        "world_model_source": str(source["world_model_source_sha256"]),
        "forward_checkpoint": str(source["forward_checkpoint_sha256"]),
        "simulator_config": str(source["simulator_config_sha256"]),
    }
    supplied = {
        "forward_checkpoint": forward_checkpoint_path.expanduser().resolve(),
        "simulator_config": simulator_config_path.expanduser().resolve(),
    }
    for role, actual_path in supplied.items():
        if actual_path != expected_paths[role]:
            raise ClosedLoopEvaluationError(f"{role} path differs from freeze")
    verified: dict[str, dict[str, str]] = {}
    for role, path in expected_paths.items():
        try:
            actual_hash = _sha256_path(path)
        except OSError as exc:
            raise ClosedLoopEvaluationError(f"cannot read frozen {role}: {path}") from exc
        if actual_hash != expected_hashes[role]:
            raise ClosedLoopEvaluationError(f"{role} digest mismatch")
        verified[role] = {"path": str(path), "sha256": actual_hash}

    locked = _read_json(expected_paths["default_h1_protocol"])
    expected_base = repository_path(locked["base_simulator_config"])
    supplied_base = base_config_path.expanduser().resolve()
    if supplied_base != expected_base:
        raise ClosedLoopEvaluationError("base_config path differs from locked primary freeze")
    base_hash = _sha256_path(expected_base)
    if base_hash != str(locked["base_simulator_config_sha256"]):
        raise ClosedLoopEvaluationError("base_config digest mismatch")
    if repository_path(locked["checkpoint"]) != expected_paths["forward_checkpoint"]:
        raise ClosedLoopEvaluationError("locked primary checkpoint path conflicts with protocol")
    if str(locked["checkpoint_sha256"]) != expected_hashes["forward_checkpoint"]:
        raise ClosedLoopEvaluationError("locked primary checkpoint hash conflicts with protocol")
    if repository_path(locked["corrected_v12_config"]) != expected_paths["simulator_config"]:
        raise ClosedLoopEvaluationError("locked primary simulator path conflicts with protocol")
    if str(locked["corrected_v12_config_sha256"]) != expected_hashes["simulator_config"]:
        raise ClosedLoopEvaluationError("locked primary simulator hash conflicts with protocol")
    verified["base_config"] = {"path": str(expected_base), "sha256": base_hash}
    return verified


def _guard_candidate_path(
    path: Path,
    *,
    role: str,
    must_exist: bool,
) -> Path:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(PACKAGE_ROOT.resolve())
    except ValueError as exc:
        raise ClosedLoopEvaluationError(
            f"{role} must remain inside {PACKAGE_ROOT}"
        ) from exc
    forbidden_parts = {"protected", "frozen", "iid", "ood", "test", "tests"}
    if any(part.lower() in forbidden_parts for part in relative.parts):
        raise ClosedLoopEvaluationError(
            f"{role} path contains a forbidden split marker"
        )
    if must_exist and not resolved.is_file():
        raise ClosedLoopEvaluationError(f"{role} does not exist: {resolved}")
    return resolved


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def metric_value(value: Any, reason: str | None = None) -> dict[str, Any]:
    """Represent every optional metric as an explicit value/reason pair."""

    if value is None and not reason:
        raise ValueError("a null metric requires a reason")
    if value is not None and reason is not None:
        raise ValueError("a measured metric cannot have a missing-value reason")
    return {"value": value, "reason": reason}


def normalize_candidate_eval_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    require_preregistered_cardinality: bool = True,
) -> list[dict[str, Any]]:
    """Validate fresh candidate-eval identity without opening image content."""

    if not records:
        raise ClosedLoopEvaluationError("candidate-eval manifest is empty")
    output: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    setup_targets: defaultdict[str, set[int]] = defaultdict(set)
    for index, source in enumerate(records):
        row = copy.deepcopy(dict(source))
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ClosedLoopEvaluationError(f"manifest row {index} has no record_id")
        if record_id in record_ids:
            raise ClosedLoopEvaluationError(f"duplicate record_id: {record_id}")
        record_ids.add(record_id)
        if row.get("split") != "candidate_eval":
            raise ClosedLoopEvaluationError(
                f"{record_id}: evaluator accepts candidate_eval only"
            )
        if require_preregistered_cardinality and not record_id.startswith(
            "qh1meta_eval_"
        ):
            raise ClosedLoopEvaluationError(
                f"{record_id}: not a preregistered qh1meta_eval identity"
            )
        runtime_input = row.get("model_visible_input")
        if not isinstance(runtime_input, Mapping):
            raise ClosedLoopEvaluationError(
                f"{record_id}: missing model_visible_input"
            )
        row["model_visible_input"] = parse_meta_input(runtime_input).to_dict()
        evaluator = row.get("evaluator_only")
        if not isinstance(evaluator, Mapping):
            raise ClosedLoopEvaluationError(
                f"{record_id}: missing evaluator_only fields"
            )
        required = {
            "setup_id",
            "episode_id",
            "target_index",
            "target_counterfactual_id",
            "setup_context",
            "simulator_fixed",
        }
        missing = sorted(required - set(evaluator))
        if missing:
            raise ClosedLoopEvaluationError(
                f"{record_id}: evaluator_only missing {missing}"
            )
        setup_id = evaluator["setup_id"]
        target_index = evaluator["target_index"]
        if not isinstance(setup_id, str) or not setup_id:
            raise ClosedLoopEvaluationError(f"{record_id}: invalid setup_id")
        if isinstance(target_index, bool) or not isinstance(target_index, int):
            raise ClosedLoopEvaluationError(f"{record_id}: invalid target_index")
        if target_index in setup_targets[setup_id]:
            raise ClosedLoopEvaluationError(
                f"{record_id}: duplicate target_index within setup"
            )
        setup_targets[setup_id].add(target_index)
        # The image mapping is deliberately not resolved, hashed, or opened.
        output.append(row)
    if require_preregistered_cardinality:
        if len(output) != 36 or len(setup_targets) != 12:
            raise ClosedLoopEvaluationError(
                "preregistered candidate evaluation requires 12 setups x 3 targets"
            )
        if any(targets != {0, 1, 2} for targets in setup_targets.values()):
            raise ClosedLoopEvaluationError(
                "each candidate-eval setup must contain target indices 0,1,2"
            )
        expected_setups = {f"qh1meta_eval_{index:04d}" for index in range(12)}
        if set(setup_targets) != expected_setups:
            raise ClosedLoopEvaluationError(
                "candidate-eval setup registry differs from qh1meta_eval_0000..0011"
            )
        for row in output:
            evaluator = row["evaluator_only"]
            if evaluator.get("target_seed") != TARGET_SEED:
                raise ClosedLoopEvaluationError(
                    f"{row['record_id']}: target seed differs from preregistration"
                )
            if evaluator.get("oracle_seed") != ORACLE_SEED:
                raise ClosedLoopEvaluationError(
                    f"{row['record_id']}: oracle seed differs from preregistration"
                )
    return sorted(
        output,
        key=lambda row: (
            str(row["evaluator_only"]["setup_id"]),
            int(row["evaluator_only"]["target_index"]),
        ),
    )


@dataclass(frozen=True)
class PolicyContext:
    method_name: str
    episode_id: str
    setup_id: str
    step: int
    planner_seed: int
    runtime_input: Mapping[str, Any]
    manifest_record: Mapping[str, Any]
    sensor_image_normalized: np.ndarray | None = None


def runtime_input_fingerprint(runtime_input: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        runtime_input,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def sensor_image_fingerprint(image: np.ndarray) -> str:
    """Hash the exact 8-bit display pixels shown to an image-aware policy."""

    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ClosedLoopEvaluationError(
            "sensor image must be a nonempty finite two-dimensional array"
        )
    pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    header = f"{pixels.shape[0]}x{pixels.shape[1]}:L:".encode("ascii")
    return hashlib.sha256(header + pixels.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class MetaPolicyResult:
    payload: MetaControllerDecision | str | bytes | Mapping[str, Any] | None
    available: bool = True
    unavailable_reason: str | None = None
    latency_seconds: float | None = None
    configuration_regret: float | None = None

    def __post_init__(self) -> None:
        if self.available and self.payload is None:
            raise ValueError("available meta policy result requires a payload")
        if not self.available and not self.unavailable_reason:
            raise ValueError("unavailable meta policy result requires a reason")
        if self.latency_seconds is not None and (
            not math.isfinite(float(self.latency_seconds))
            or float(self.latency_seconds) < 0.0
        ):
            raise ValueError("policy latency must be finite and nonnegative")
        if self.configuration_regret is not None and (
            not math.isfinite(float(self.configuration_regret))
            or float(self.configuration_regret) < -1e-12
        ):
            raise ValueError("configuration regret must be finite and nonnegative")


class MetaPolicy(Protocol):
    requires_sensor_image: bool
    live_image_generation: bool

    def decide(self, context: PolicyContext) -> MetaPolicyResult:
        """Return an already-generated discrete meta decision."""


class RulePolicy:
    requires_sensor_image = False
    live_image_generation = False

    def __init__(self) -> None:
        self.baseline = RuleMetaBaseline()

    def decide(self, context: PolicyContext) -> MetaPolicyResult:
        started = time.perf_counter()
        payload = self.baseline.predict(context.runtime_input)
        return MetaPolicyResult(
            payload=payload,
            latency_seconds=time.perf_counter() - started,
        )


class CallablePolicy:
    """Small adapter used by trained baselines and synthetic tests."""

    def __init__(
        self,
        callback: Callable[[PolicyContext], MetaPolicyResult | Mapping[str, Any] | str],
        *,
        requires_sensor_image: bool = False,
    ) -> None:
        self.callback = callback
        self.requires_sensor_image = bool(requires_sensor_image)
        self.live_image_generation = bool(requires_sensor_image)

    def decide(self, context: PolicyContext) -> MetaPolicyResult:
        started = time.perf_counter()
        result = self.callback(context)
        elapsed = time.perf_counter() - started
        if isinstance(result, MetaPolicyResult):
            return result
        return MetaPolicyResult(payload=result, latency_seconds=elapsed)


class UnavailablePolicy:
    requires_sensor_image = False
    live_image_generation = False

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)

    def decide(self, context: PolicyContext) -> MetaPolicyResult:
        del context
        return MetaPolicyResult(
            payload=None,
            available=False,
            unavailable_reason=self.reason,
        )


class TracePolicy:
    """Per-episode/per-step output trace; it never repairs generated text."""

    live_image_generation = False

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        require_dynamic_image_binding: bool = False,
    ) -> None:
        self.require_dynamic_image_binding = bool(require_dynamic_image_binding)
        self.requires_sensor_image = self.require_dynamic_image_binding
        self.rows: dict[tuple[str, int], dict[str, Any]] = {}
        for index, source in enumerate(rows):
            row = dict(source)
            episode_id = row.get("episode_id", row.get("record_id", row.get("sample_id")))
            step = row.get("step")
            if not isinstance(episode_id, str) or not episode_id:
                raise ClosedLoopEvaluationError(
                    f"prediction trace row {index} has no episode_id"
                )
            if isinstance(step, bool) or not isinstance(step, int) or not (
                0 <= step < HORIZON_STEPS
            ):
                raise ClosedLoopEvaluationError(
                    f"prediction trace {episode_id}: step must be 0..{HORIZON_STEPS - 1}"
                )
            key = (episode_id, step)
            if key in self.rows:
                raise ClosedLoopEvaluationError(
                    f"duplicate prediction trace entry {episode_id} step {step}"
                )
            if "prediction" not in row:
                raise ClosedLoopEvaluationError(
                    f"prediction trace {episode_id} step {step} has no prediction"
                )
            if self.require_dynamic_image_binding:
                for field in (
                    "runtime_input_sha256",
                    "sensor_image_fingerprint_sha256",
                ):
                    value = row.get(field)
                    if (
                        not isinstance(value, str)
                        or len(value) != 64
                        or any(character not in "0123456789abcdef" for character in value)
                    ):
                        raise ClosedLoopEvaluationError(
                            f"dynamic trace {episode_id} step {step} requires {field}"
                        )
            self.rows[key] = row

    def decide(self, context: PolicyContext) -> MetaPolicyResult:
        row = self.rows.get((context.episode_id, context.step))
        if row is None:
            return MetaPolicyResult(
                payload=None,
                available=False,
                unavailable_reason="missing_per_step_prediction_trace",
            )
        if self.require_dynamic_image_binding:
            if context.sensor_image_normalized is None:
                return MetaPolicyResult(
                    payload=None,
                    available=False,
                    unavailable_reason="dynamic_trace_sensor_image_unavailable",
                )
            if row["runtime_input_sha256"] != runtime_input_fingerprint(
                context.runtime_input
            ):
                return MetaPolicyResult(
                    payload=None,
                    available=False,
                    unavailable_reason="dynamic_trace_runtime_state_mismatch",
                )
            if row["sensor_image_fingerprint_sha256"] != sensor_image_fingerprint(
                context.sensor_image_normalized
            ):
                return MetaPolicyResult(
                    payload=None,
                    available=False,
                    unavailable_reason="dynamic_trace_sensor_image_mismatch",
                )
        return MetaPolicyResult(
            payload=row["prediction"],
            latency_seconds=(
                None
                if row.get("latency_seconds") is None
                else float(row["latency_seconds"])
            ),
            configuration_regret=(
                None
                if row.get("configuration_regret") is None
                else float(row["configuration_regret"])
            ),
        )


class LiveImagePolicy(CallablePolicy):
    """Array-based hook for a live Qwen adapter owned by the caller."""

    def __init__(
        self,
        callback: Callable[[PolicyContext], MetaPolicyResult | Mapping[str, Any] | str],
    ) -> None:
        super().__init__(callback, requires_sensor_image=True)


class LiveImagePathPolicy:
    """Temporary-PNG hook for adapters whose API requires an image path."""

    requires_sensor_image = True
    live_image_generation = True

    def __init__(
        self,
        callback: Callable[
            [PolicyContext, Path], MetaPolicyResult | Mapping[str, Any] | str
        ],
    ) -> None:
        self.callback = callback

    def decide(self, context: PolicyContext) -> MetaPolicyResult:
        if context.sensor_image_normalized is None:
            return MetaPolicyResult(
                payload=None,
                available=False,
                unavailable_reason="live_policy_sensor_image_unavailable",
            )
        from PIL import Image

        pixels = np.rint(
            np.clip(context.sensor_image_normalized, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(
            prefix=".closed_loop_live_image_", dir=PACKAGE_ROOT
        ) as directory:
            image_path = Path(directory) / "current_sensor_frame.png"
            Image.fromarray(pixels, mode="L").save(image_path, format="PNG")
            result = self.callback(context, image_path)
        elapsed = time.perf_counter() - started
        if isinstance(result, MetaPolicyResult):
            return result
        return MetaPolicyResult(payload=result, latency_seconds=elapsed)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    feature_mode: str
    policy: MetaPolicy | None = None
    planner_config: Mapping[str, Any] | None = None
    planner_profile: str = "default_h1"

    def resolved_config(self) -> dict[str, Any]:
        config = default_h1_config() if self.planner_config is None else dict(self.planner_config)
        return validate_preregistered_controller_config(
            config,
            profile=self.planner_profile,
        )

    def __post_init__(self) -> None:
        if not self.name or "h3" in self.name.lower():
            raise ValueError("method name must be nonempty and H1-only")
        if self.feature_mode not in {"off", "shadow", "guarded"}:
            raise ValueError(f"invalid feature mode {self.feature_mode!r}")
        if self.feature_mode == "off" and self.policy is not None:
            raise ValueError("off methods cannot call a meta policy")
        self.resolved_config()


def standard_method_specs(
    policies: Mapping[str, MetaPolicy] | None = None,
) -> tuple[MethodSpec, ...]:
    """Build the preregistered comparison registry without invoking models."""

    supplied = {} if policies is None else dict(policies)
    allowed_policy_names = set(STANDARD_METHOD_NAMES) - {
        "default_h1",
        "dual_budget_default_h1",
        "rule_guided_h1",
    }
    unknown = sorted(set(supplied) - allowed_policy_names)
    if unknown:
        raise ClosedLoopEvaluationError(f"unknown method policy names: {unknown}")

    def policy(name: str) -> MetaPolicy:
        return supplied.get(
            name,
            UnavailablePolicy(f"no per-step prediction trace supplied for {name}"),
        )

    def qwen_policy(name: str) -> MetaPolicy:
        selected = policy(name)
        if isinstance(selected, UnavailablePolicy):
            return selected
        is_live = bool(getattr(selected, "live_image_generation", False))
        is_bound_trace = bool(
            isinstance(selected, TracePolicy)
            and selected.require_dynamic_image_binding
        )
        if not (is_live or is_bound_trace):
            raise ClosedLoopEvaluationError(
                f"{name} requires a live image-aware policy or a state/image-bound dynamic trace"
            )
        return selected

    default = default_h1_config()
    dual = dual_budget_h1_config()
    shadow_policy = supplied.get(
        "shadow_qwen_h1",
        qwen_policy("qwen_guided_h1_seed_2026080201"),
    )
    if not isinstance(shadow_policy, UnavailablePolicy):
        shadow_live = bool(getattr(shadow_policy, "live_image_generation", False))
        shadow_bound = bool(
            isinstance(shadow_policy, TracePolicy)
            and shadow_policy.require_dynamic_image_binding
        )
        if not (shadow_live or shadow_bound):
            raise ClosedLoopEvaluationError(
                "shadow_qwen_h1 requires live image generation or a bound dynamic trace"
            )
    return (
        MethodSpec("default_h1", "off", planner_config=default),
        MethodSpec(
            "dual_budget_default_h1",
            "off",
            planner_config=dual,
            planner_profile="dual_budget_default_h1",
        ),
        MethodSpec("rule_guided_h1", "guarded", policy=RulePolicy()),
        MethodSpec(
            "metrics_mlp_meta_h1",
            "guarded",
            policy=policy("metrics_mlp_meta_h1"),
        ),
        *(
            MethodSpec(
                f"qwen_guided_h1_seed_{seed}",
                "guarded",
                policy=qwen_policy(f"qwen_guided_h1_seed_{seed}"),
            )
            for seed in QWEN_SEEDS
        ),
        MethodSpec(
            "oracle_guided_h1",
            "guarded",
            policy=policy("oracle_guided_h1"),
        ),
        MethodSpec(
            "random_or_frequency_meta_h1",
            "guarded",
            policy=policy("random_or_frequency_meta_h1"),
        ),
        MethodSpec(
            "shadow_qwen_h1",
            "shadow",
            policy=shadow_policy,
        ),
    )


@dataclass(frozen=True)
class SimulationObservation:
    positions_mm: tuple[float, float, float, float]
    metrics: tuple[float, float, float, float, float]
    simulator_valid: bool
    clipping_fraction: float | None
    camera_boundary_indicator: bool | None
    raw_auxiliary: Mapping[str, Any]
    sensor_image_normalized: np.ndarray | None = None


class ClosedLoopBackend(Protocol):
    bounds: Bounds
    identity: Mapping[str, Any]

    def predictor(self, record: Mapping[str, Any]) -> Predictor:
        """Return the unchanged numeric predictor for one setup."""

    def action_evaluator(self, record: Mapping[str, Any]) -> ActionEvaluator:
        """Return physical ensemble/member predictions for arbitration."""

    def initial_sensor_image(
        self,
        record: Mapping[str, Any],
        positions_mm: np.ndarray,
    ) -> np.ndarray:
        """Generate the fresh current image only for image-aware policies."""

    def simulate(
        self,
        record: Mapping[str, Any],
        positions_mm: np.ndarray,
        action_mm: np.ndarray,
    ) -> SimulationObservation:
        """Execute one corrected deterministic simulator transition."""


class CorrectedSimulatorBackend:
    """Real unchanged-forward/corrected-simulator evaluation backend."""

    def __init__(
        self,
        *,
        model: Any,
        bounds: Bounds,
        simulator_config_path: Path,
        base_config_path: Path,
        forward_checkpoint_sha256: str,
    ) -> None:
        bounds.validate()
        self.model = model
        self.bounds = bounds
        self.simulator_config_path = simulator_config_path.resolve()
        self.base_config_path = base_config_path.resolve()
        if bool(getattr(model, "image_conditioning", False)):
            raise ClosedLoopEvaluationError(
                "closed-loop evaluator requires unchanged numeric forward ensemble"
            )
        model_bounds = getattr(model, "bounds", None)
        if model_bounds is not None:
            for field in (
                "action_low",
                "action_high",
                "position_low",
                "position_high",
            ):
                if not np.array_equal(
                    np.asarray(getattr(model_bounds, field)),
                    np.asarray(getattr(bounds, field)),
                ):
                    raise ClosedLoopEvaluationError(
                        f"forward ensemble {field} differs from corrected bounds"
                    )
        self.identity = {
            "backend": "unchanged_forward_ensemble_plus_corrected_v12_1_simulator",
            "forward_checkpoint_sha256": forward_checkpoint_sha256,
            "simulator_config": str(self.simulator_config_path),
            "simulator_config_sha256": _sha256_path(self.simulator_config_path),
            "base_config": str(self.base_config_path),
            "base_config_sha256": _sha256_path(self.base_config_path),
        }

    def predictor(self, record: Mapping[str, Any]) -> Predictor:
        return learned_predictor(
            self.model, record["evaluator_only"]["setup_context"]
        )

    def action_evaluator(self, record: Mapping[str, Any]) -> ActionEvaluator:
        return forward_ensemble_action_evaluator(
            self.model, record["evaluator_only"]["setup_context"]
        )

    def _capture_at_positions(
        self,
        record: Mapping[str, Any],
        positions_mm: np.ndarray,
    ) -> Mapping[str, Any]:
        simulator_fixed = record["evaluator_only"]["simulator_fixed"]
        if not is_corrected_semantics(simulator_fixed):
            raise ClosedLoopEvaluationError(
                "manifest simulator_fixed is not corrected v12.1 semantics"
            )
        return simulate_state(
            record["evaluator_only"]["setup_context"],
            position_dict(positions_mm),
            simulator_fixed,
            str(self.base_config_path),
            self.bounds,
        )

    def initial_sensor_image(
        self,
        record: Mapping[str, Any],
        positions_mm: np.ndarray,
    ) -> np.ndarray:
        capture = self._capture_at_positions(record, positions_mm)
        expected = metrics_vector(record["model_visible_input"]["current_beam_state"])
        observed = metrics_vector(capture["metrics"])
        if not np.allclose(observed, expected, atol=1e-8, rtol=1e-10):
            raise ClosedLoopEvaluationError(
                "regenerated initial simulator metrics differ from manifest state"
            )
        image = np.asarray(capture["image_normalized"], dtype=np.float64)
        if image.shape != (1024, 1024) or not np.isfinite(image).all():
            raise ClosedLoopEvaluationError(
                "corrected simulator did not return a finite 1024x1024 sensor image"
            )
        return image.copy()

    def simulate(
        self,
        record: Mapping[str, Any],
        positions_mm: np.ndarray,
        action_mm: np.ndarray,
    ) -> SimulationObservation:
        next_positions = apply_action(positions_mm, action_mm, self.bounds)
        capture = self._capture_at_positions(record, next_positions)
        auxiliary = dict(capture["auxiliary"])
        image = np.asarray(capture["image_normalized"], dtype=np.float64)
        return SimulationObservation(
            positions_mm=tuple(float(value) for value in next_positions),  # type: ignore[arg-type]
            metrics=tuple(float(value) for value in metrics_vector(capture["metrics"])),  # type: ignore[arg-type]
            simulator_valid=bool(auxiliary.get("simulator_valid", False)),
            clipping_fraction=(
                None
                if auxiliary.get("clipping_fraction") is None
                else float(auxiliary["clipping_fraction"])
            ),
            camera_boundary_indicator=(
                None
                if auxiliary.get("camera_boundary_indicator") is None
                else bool(auxiliary["camera_boundary_indicator"])
            ),
            raw_auxiliary=auxiliary,
            sensor_image_normalized=image.copy(),
        )


def normalized_components(
    metrics: Sequence[float],
    target: Sequence[float],
    reference: Sequence[float],
) -> np.ndarray:
    values = metrics_vector(metrics)
    target_values = metrics_vector(target)
    tolerances = tolerance_vector(reference)
    output = np.abs(values - target_values) / tolerances
    if not np.isfinite(output).all():
        raise ClosedLoopEvaluationError("normalized error is non-finite")
    return output


def strict_success(
    metrics: Sequence[float], target: Sequence[float], reference: Sequence[float]
) -> bool:
    return bool(np.all(normalized_components(metrics, target, reference) <= 1.0))


def canonical_state_objective(
    metrics: Sequence[float], target: Sequence[float], reference: Sequence[float]
) -> float:
    normalized = normalized_components(metrics, target, reference)
    return float(normalized.max() + 0.15 * normalized.mean())


def error_step_auc(distances: Sequence[float]) -> float:
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("distance trace must be a nonempty finite vector")
    if values.size == 1:
        return 0.0
    # Manual trapezoid rule is stable across NumPy 1.x/2.x API changes.
    return float(np.sum((values[:-1] + values[1:]) * 0.5))


def _planner_ood_values(plan: Mapping[str, Any] | None) -> list[float]:
    if plan is None:
        return []
    output: list[float] = []
    for row in plan.get("iteration_history", []):
        value = row.get("proposal_initially_out_of_feasible_bounds_fraction")
        if value is not None:
            number = float(value)
            if math.isfinite(number):
                output.append(number)
    return output


def _candidate_evaluations(controller_result: Any) -> int:
    total = 0
    for plan in (controller_result.default_plan, controller_result.guided_plan):
        if plan is not None:
            total += int(plan.get("candidate_sequences_evaluated", 0))
    return total


def _supervisor_decision(runtime_input: Mapping[str, Any], budget: LoopBudget):
    validity = runtime_input["measurement_validity"]
    diagnosis = validity["supervisor_diagnosis"]
    policy = validity["measurement_policy"]
    action = {
        ("nominal", "standard"): "execute",
        ("sensor_saturation", "lower_exposure_reacquire"): "reacquire",
        ("secondary_reflection", "primary_spot"): "switch_measurement",
    }.get((diagnosis, policy), "execute")
    return adapt_supervisor_decision_or_safe_stop(
        {
            "diagnosis": diagnosis,
            "measurement_policy": policy,
            "supervisor_action": action,
        },
        executed_steps=budget.executed_control_steps,
        remaining_step_budget=budget.control_steps,
        frozen_continuation_allows_next=budget.control_steps > 0,
    )


def _update_runtime_input(
    runtime_input: Mapping[str, Any],
    *,
    positions: np.ndarray,
    previous_metrics: np.ndarray,
    next_metrics: np.ndarray,
    target_metrics: np.ndarray,
    tolerance_reference: np.ndarray,
    action: np.ndarray,
    physical: PhysicalActionEvaluation,
    budget: LoopBudget,
    simulator_valid: bool,
) -> dict[str, Any]:
    output = copy.deepcopy(dict(runtime_input))
    predicted_delta = physical.next_vector() - previous_metrics
    measured_delta = next_metrics - previous_metrics
    history = list(output["history"])
    history.append(
        {
            "valid": True,
            "executed_action_mm": action_dict(action),
            "measured_beam_delta": metrics_dict(measured_delta),
            "predicted_beam_delta": metrics_dict(predicted_delta),
            "prediction_residual": metrics_dict(measured_delta - predicted_delta),
            "ensemble_uncertainty": metrics_dict(physical.uncertainty),
            "padding_reason": "none",
        }
    )
    output["history"] = history[-3:]
    output["current_beam_state"] = metrics_dict(next_metrics)
    output["normalized_signed_error"] = metrics_dict(
        (target_metrics - next_metrics) / tolerance_reference
    )
    output["actuator_positions_mm"] = position_dict(positions)
    uncertainty = physical.uncertainty_vector()
    output["forward_uncertainty"] = {
        "per_metric": metrics_dict(uncertainty),
        "mean": float(uncertainty.mean()),
        "maximum": float(uncertainty.max()),
    }
    output["remaining_budget"] = {
        "measurement_steps": budget.measurement_steps,
        "control_steps": budget.control_steps,
    }
    output["measurement_validity"] = (
        {
            "state": "valid",
            "supervisor_diagnosis": "nominal",
            "measurement_policy": "standard",
        }
        if simulator_valid
        else {
            "state": "invalid",
            "supervisor_diagnosis": "unknown",
            "measurement_policy": "unknown",
        }
    )
    return parse_meta_input(output).to_dict()


def _run_episode(
    *,
    record: Mapping[str, Any],
    method: MethodSpec,
    backend: ClosedLoopBackend,
    horizon_steps: int,
) -> dict[str, Any]:
    if horizon_steps != HORIZON_STEPS:
        raise ClosedLoopEvaluationError("candidate closed-loop horizon is frozen to 4")
    started_episode = time.perf_counter()
    record_id = str(record["record_id"])
    evaluator = record["evaluator_only"]
    setup_id = str(evaluator["setup_id"])
    target_id = str(evaluator["target_counterfactual_id"])
    runtime_input = parse_meta_input(record["model_visible_input"]).to_dict()
    positions = position_vector(runtime_input["actuator_positions_mm"])
    current = metrics_vector(runtime_input["current_beam_state"])
    target = metrics_vector(runtime_input["target_beam_state"])
    reference = current.copy()
    fixed_tolerances = tolerance_vector(reference)
    initial_state_cost = canonical_state_objective(current, target, reference)
    distances = [float(normalized_components(current, target, reference).max())]
    budget = LoopBudget(
        measurement_steps=int(runtime_input["remaining_budget"]["measurement_steps"]),
        control_steps=min(
            int(runtime_input["remaining_budget"]["control_steps"]), horizon_steps
        ),
        executed_control_steps=0,
    )
    predictor = backend.predictor(record)
    action_evaluator = backend.action_evaluator(record)
    planner_config = method.resolved_config()

    action_norms: list[float] = []
    uncertainty_values: list[float] = []
    ood_values: list[float] = []
    policy_latencies: list[float] = []
    planner_latencies: list[float] = []
    forward_latencies: list[float] = []
    simulator_latencies: list[float] = []
    configuration_regrets: list[float] = []
    fallback_reasons: Counter[str] = Counter()
    traces: list[dict[str, Any]] = []
    candidate_evaluation_count = 0
    policy_calls = 0
    policy_latency_missing = 0
    invalid_json_count = 0
    security_rejection_count = 0
    override_count = 0
    guided_request_count = 0
    guided_gate_count = 0
    guided_accept_count = 0
    safety_rejection_count = 0
    overshoot_count = 0
    reobserve_count = 0
    stop_count = 0
    method_input_complete = True
    incomplete_reasons: Counter[str] = Counter()
    termination_reason = "horizon_exhausted"
    steps_to_success: int | None = 0 if strict_success(current, target, reference) else None
    current_sensor_image: np.ndarray | None = None
    image_aware_policy = bool(
        method.policy is not None
        and getattr(method.policy, "requires_sensor_image", False)
    )
    live_image_generation = bool(
        method.policy is not None
        and getattr(method.policy, "live_image_generation", False)
    )

    for step in range(horizon_steps):
        if steps_to_success is not None:
            termination_reason = "strict_success"
            break
        planner_seed = stable_seed(
            PLANNER_SEED_ROOT, setup_id, target_id, step
        )
        policy_result: MetaPolicyResult | None = None
        meta_payload: Any = None
        supervisor = _supervisor_decision(runtime_input, budget)
        supervisor_blocks_meta = (
            not supervisor.audit.parse_valid
            or supervisor.decision.should_stop
            or not supervisor.decision.execute_frozen_cem
            or supervisor.decision.supervisor.diagnosis != "nominal"
            or supervisor.decision.supervisor.measurement_policy != "standard"
            or runtime_input["measurement_validity"]["state"] != "valid"
        )
        if method.policy is not None and not supervisor_blocks_meta:
            policy_calls += 1
            if image_aware_policy and current_sensor_image is None:
                try:
                    current_sensor_image = np.asarray(
                        backend.initial_sensor_image(record, positions.copy()),
                        dtype=np.float64,
                    )
                except Exception as exc:
                    method_input_complete = False
                    reason = f"initial_sensor_image_unavailable:{type(exc).__name__}"
                    incomplete_reasons[reason] += 1
                    policy_result = MetaPolicyResult(
                        payload=None,
                        available=False,
                        unavailable_reason=reason,
                    )
            try:
                if policy_result is None:
                    policy_result = method.policy.decide(
                        PolicyContext(
                            method_name=method.name,
                            episode_id=record_id,
                            setup_id=setup_id,
                            step=step,
                            planner_seed=planner_seed,
                            runtime_input=runtime_input,
                            manifest_record=record,
                            sensor_image_normalized=(
                                None
                                if current_sensor_image is None
                                else current_sensor_image.copy()
                            ),
                        )
                    )
            except Exception as exc:
                policy_result = MetaPolicyResult(
                    payload=None,
                    available=False,
                    unavailable_reason=f"policy_exception:{type(exc).__name__}",
                )
            if policy_result.latency_seconds is None:
                policy_latency_missing += 1
            else:
                policy_latencies.append(float(policy_result.latency_seconds))
            if not policy_result.available:
                method_input_complete = False
                incomplete_reasons[str(policy_result.unavailable_reason)] += 1
                meta_payload = "__missing_candidate_meta_prediction__"
            else:
                meta_payload = policy_result.payload
                if policy_result.configuration_regret is not None:
                    configuration_regrets.append(
                        max(0.0, float(policy_result.configuration_regret))
                    )

        integration = integrate_supervisor_step(
            supervisor,
            feature_mode=method.feature_mode,
            budget=budget,
            measurement_state=runtime_input["measurement_validity"]["state"],
            meta_payload=meta_payload,
        )
        if integration.meta_parse_valid is False:
            invalid_json_count += 1
        if integration.security_event:
            security_rejection_count += 1
        budget = integration.budget_after_authorization
        if integration.operation == "reobserve":
            reobserve_count += 1
            termination_reason = "reobserve_requested_no_recovery_backend"
            incomplete_reasons["sequential_measurement_recovery_not_implemented"] += 1
            traces.append(
                {
                    "step": step,
                    "planner_seed": planner_seed,
                    "integration": integration.to_dict(),
                    "action": None,
                }
            )
            break
        if integration.operation == "stop":
            stop_count += 1
            termination_reason = integration.reason
            traces.append(
                {
                    "step": step,
                    "planner_seed": planner_seed,
                    "integration": integration.to_dict(),
                    "action": None,
                }
            )
            break

        planner_started = time.perf_counter()
        controller = MetaH1Controller(
            default_bounds=backend.bounds,
            predictor=predictor,
            default_config=planner_config,
            seed=planner_seed,
            mode=method.feature_mode,
            planner_profile=method.planner_profile,
            action_evaluator=action_evaluator,
        )
        controller_result = controller.run(
            positions_mm=positions,
            current_metrics=current,
            target_metrics=target,
            guidance_payload=(
                integration.parsed_meta_decision
                if integration.parsed_meta_decision is not None
                else meta_payload
            ),
            tolerance_reference=reference,
            measurement_valid=True,
        )
        planner_latencies.append(time.perf_counter() - planner_started)
        if not controller_result.dispatch or controller_result.selected_action is None:
            method_input_complete = False
            incomplete_reasons["integration_controller_dispatch_mismatch"] += 1
            termination_reason = "integration_controller_dispatch_mismatch"
            break

        step_candidate_evaluations = _candidate_evaluations(controller_result)
        candidate_evaluation_count += step_candidate_evaluations
        ood_values.extend(_planner_ood_values(controller_result.default_plan))
        ood_values.extend(_planner_ood_values(controller_result.guided_plan))
        if controller_result.fallback_reason is not None:
            fallback_reasons[controller_result.fallback_reason] += 1
        if controller_result.guidance is not None and (
            controller_result.guidance.requested_decision == "run_guided_h1"
        ):
            guided_request_count += 1
            if controller_result.selected_source == "default":
                override_count += 1
        if controller_result.gate is not None:
            guided_gate_count += 1
            if controller_result.gate.accepted_guided:
                guided_accept_count += 1
            else:
                safety_rejection_count += 1

        action = action_vector(controller_result.selected_action)
        forward_started = time.perf_counter()
        physical = action_evaluator(positions.copy(), current.copy(), action.copy())
        forward_latencies.append(time.perf_counter() - forward_started)
        simulator_started = time.perf_counter()
        observation = backend.simulate(record, positions.copy(), action.copy())
        simulator_latencies.append(time.perf_counter() - simulator_started)
        next_positions = position_vector(observation.positions_mm)
        next_metrics = metrics_vector(observation.metrics)
        action_norms.append(float(np.linalg.norm(action / backend.bounds.action_high)))
        uncertainty_values.append(float(physical.uncertainty_vector().mean()))
        before_signed = current - target
        after_signed = next_metrics - target
        if bool(np.any(before_signed * after_signed < 0.0)):
            overshoot_count += 1
        distances.append(
            float(normalized_components(next_metrics, target, reference).max())
        )
        traces.append(
            {
                "step": step,
                "planner_seed": planner_seed,
                "integration": integration.to_dict(),
                "selected_source": controller_result.selected_source,
                "selected_action": action_dict(action),
                "positions_before_mm": position_dict(positions),
                "positions_after_mm": position_dict(next_positions),
                "metrics_before": metrics_dict(current),
                "metrics_after": metrics_dict(next_metrics),
                "canonical_objective_before": canonical_state_objective(
                    current, target, reference
                ),
                "canonical_objective_after": canonical_state_objective(
                    next_metrics, target, reference
                ),
                "actual_canonical_objective_reduction": (
                    canonical_state_objective(current, target, reference)
                    - canonical_state_objective(next_metrics, target, reference)
                ),
                "predicted_next_metrics": metrics_dict(physical.predicted_next_metrics),
                "mean_normalized_uncertainty": float(
                    physical.uncertainty_vector().mean()
                ),
                "candidate_evaluations": step_candidate_evaluations,
                "gate": (
                    None
                    if controller_result.gate is None
                    else controller_result.gate.to_dict()
                ),
                "fallback_reason": controller_result.fallback_reason,
                "simulator_valid": observation.simulator_valid,
                "clipping_fraction": observation.clipping_fraction,
                "camera_boundary_indicator": observation.camera_boundary_indicator,
            }
        )
        runtime_input = _update_runtime_input(
            runtime_input,
            positions=next_positions,
            previous_metrics=current,
            next_metrics=next_metrics,
            target_metrics=target,
            tolerance_reference=fixed_tolerances,
            action=action,
            physical=physical,
            budget=budget,
            simulator_valid=observation.simulator_valid,
        )
        positions = next_positions
        current = next_metrics
        current_sensor_image = (
            None
            if observation.sensor_image_normalized is None
            else np.asarray(
                observation.sensor_image_normalized, dtype=np.float64
            ).copy()
        )
        if not observation.simulator_valid:
            termination_reason = "corrected_simulator_invalid_after_action"
            break
        if strict_success(current, target, reference):
            steps_to_success = budget.executed_control_steps
            termination_reason = "strict_success"
            break

    final_components = normalized_components(current, target, reference)
    final_state_cost = canonical_state_objective(current, target, reference)
    executed_actions = len(action_norms)
    episode_complete = method_input_complete
    regret_complete = policy_calls > 0 and len(configuration_regrets) == policy_calls
    configuration_regret_metric = metric_value(
        float(np.mean(configuration_regrets)) if regret_complete else None,
        (
            None
            if regret_complete
            else "per-step oracle configuration cost was not supplied for every meta call"
        ),
    )
    return {
        "record_id": record_id,
        "episode_id": str(evaluator["episode_id"]),
        "setup_id": setup_id,
        "target_id": target_id,
        "target_index": int(evaluator["target_index"]),
        "target_seed": evaluator.get("target_seed"),
        "oracle_seed": evaluator.get("oracle_seed"),
        "method": method.name,
        "feature_mode": method.feature_mode,
        "planner_profile": method.planner_profile,
        "planner_config": planner_config,
        "policy_provenance": {
            "class": None if method.policy is None else type(method.policy).__name__,
            "requires_sensor_image": image_aware_policy,
            "live_image_generation": live_image_generation,
            "dynamic_trace_binding_required": bool(
                isinstance(method.policy, TracePolicy)
                and method.policy.require_dynamic_image_binding
            ),
        },
        "horizon_steps": horizon_steps,
        "method_input_complete": episode_complete,
        "incomplete_reasons": dict(sorted(incomplete_reasons.items())),
        "strict_all_five_success": bool(steps_to_success is not None),
        "steps_to_success": steps_to_success,
        "executed_control_steps": executed_actions,
        "final_normalized_error": float(final_components.max()),
        "final_normalized_error_per_metric": metrics_dict(final_components),
        "error_step_auc": error_step_auc(distances),
        "actual_canonical_objective_reduction": float(
            initial_state_cost - final_state_cost
        ),
        "initial_canonical_state_objective": initial_state_cost,
        "final_canonical_state_objective": final_state_cost,
        "action_norm_values": action_norms,
        "overshoot_count": overshoot_count,
        "out_of_domain_proposal_values": ood_values,
        "safety_rejection_count": safety_rejection_count,
        "security_rejection_count": security_rejection_count,
        "invalid_json_count": invalid_json_count,
        "override_count": override_count,
        "fallback_count": int(sum(fallback_reasons.values())),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "reobserve_count": reobserve_count,
        "stop_count": stop_count,
        "guided_request_count": guided_request_count,
        "guided_gate_count": guided_gate_count,
        "guided_accept_count": guided_accept_count,
        "configuration_regret": configuration_regret_metric,
        "ensemble_uncertainty_values": uncertainty_values,
        "policy_call_count": policy_calls,
        "policy_latency_missing_count": policy_latency_missing,
        "policy_latency_seconds": policy_latencies,
        "planner_latency_seconds": planner_latencies,
        "forward_rescore_latency_seconds": forward_latencies,
        "simulator_latency_seconds": simulator_latencies,
        "candidate_evaluation_count": candidate_evaluation_count,
        "termination_reason": termination_reason,
        "distance_trace": distances,
        "trace": traces,
        "wall_clock_seconds": time.perf_counter() - started_episode,
    }


def _mean_metric(
    values: Sequence[float], *, missing_reason: str
) -> dict[str, Any]:
    return metric_value(
        None if not values else float(np.mean(np.asarray(values, dtype=np.float64))),
        missing_reason if not values else None,
    )


def aggregate_method_episodes(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ClosedLoopEvaluationError("cannot aggregate an empty method")
    complete = all(bool(row["method_input_complete"]) for row in episodes)
    incomplete_reason = (
        None
        if complete
        else "one or more required per-step method predictions were unavailable"
    )
    successes = [float(bool(row["strict_all_five_success"])) for row in episodes]
    successful_steps = [
        float(row["steps_to_success"])
        for row in episodes
        if row["steps_to_success"] is not None
    ]
    performance = {
        "strict_all_five_success": metric_value(
            float(np.mean(successes)) if complete else None,
            incomplete_reason,
        ),
        "steps_to_success": metric_value(
            float(np.mean(successful_steps)) if complete and successful_steps else None,
            (
                incomplete_reason
                if not complete
                else "no successful episodes"
                if not successful_steps
                else None
            ),
        ),
        "final_normalized_error": metric_value(
            float(np.mean([float(row["final_normalized_error"]) for row in episodes]))
            if complete
            else None,
            incomplete_reason,
        ),
        "error_step_auc": metric_value(
            float(np.mean([float(row["error_step_auc"]) for row in episodes]))
            if complete
            else None,
            incomplete_reason,
        ),
        "actual_canonical_objective_reduction": metric_value(
            float(
                np.mean(
                    [
                        float(row["actual_canonical_objective_reduction"])
                        for row in episodes
                    ]
                )
            )
            if complete
            else None,
            incomplete_reason,
        ),
    }
    action_norms = [
        float(value) for row in episodes for value in row["action_norm_values"]
    ]
    ood = [
        float(value)
        for row in episodes
        for value in row["out_of_domain_proposal_values"]
    ]
    uncertainties = [
        float(value)
        for row in episodes
        for value in row["ensemble_uncertainty_values"]
    ]
    policy_latencies = [
        float(value)
        for row in episodes
        for value in row["policy_latency_seconds"]
    ]
    planner_latencies = [
        float(value)
        for row in episodes
        for value in row["planner_latency_seconds"]
    ]
    control_steps = sum(int(row["executed_control_steps"]) for row in episodes)
    policy_calls = sum(int(row["policy_call_count"]) for row in episodes)
    policy_latency_missing = sum(
        int(row["policy_latency_missing_count"]) for row in episodes
    )
    guided_requests = sum(int(row["guided_request_count"]) for row in episodes)
    guided_gates = sum(int(row["guided_gate_count"]) for row in episodes)
    regret_values = [
        float(row["configuration_regret"]["value"])
        for row in episodes
        if row["configuration_regret"]["value"] is not None
    ]
    regret_complete = bool(regret_values) and len(regret_values) == len(episodes)
    fallback_reasons: Counter[str] = Counter()
    incomplete_reasons: Counter[str] = Counter()
    for row in episodes:
        fallback_reasons.update(row["fallback_reasons"])
        incomplete_reasons.update(row["incomplete_reasons"])
    audit = {
        "action_norm": _mean_metric(
            action_norms, missing_reason="no continuous action was executed"
        ),
        "overshoot_frequency": metric_value(
            (
                sum(int(row["overshoot_count"]) for row in episodes) / control_steps
                if control_steps
                else None
            ),
            None if control_steps else "no continuous action was executed",
        ),
        "out_of_domain_proposal_frequency": _mean_metric(
            ood, missing_reason="planner proposal diagnostics were unavailable"
        ),
        "safety_rejection_rate": metric_value(
            (
                sum(int(row["safety_rejection_count"]) for row in episodes)
                / guided_gates
                if guided_gates
                else None
            ),
            None if guided_gates else "no guided proposal reached the safety gate",
        ),
        "security_rejection_count": sum(
            int(row["security_rejection_count"]) for row in episodes
        ),
        "invalid_json_rate": metric_value(
            (
                sum(int(row["invalid_json_count"]) for row in episodes) / policy_calls
                if policy_calls
                else None
            ),
            None if policy_calls else "method does not call a meta policy",
        ),
        "override_rate": metric_value(
            (
                sum(int(row["override_count"]) for row in episodes)
                / guided_requests
                if guided_requests
                else None
            ),
            None if guided_requests else "no guided decision was requested",
        ),
        "fallback_rate": metric_value(
            (
                sum(int(row["fallback_count"]) for row in episodes) / policy_calls
                if policy_calls
                else None
            ),
            None if policy_calls else "method does not call a meta policy",
        ),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "reobserve_rate": float(
            np.mean([bool(row["reobserve_count"]) for row in episodes])
        ),
        "stop_rate": float(np.mean([bool(row["stop_count"]) for row in episodes])),
        "guided_acceptance_rate": metric_value(
            (
                sum(int(row["guided_accept_count"]) for row in episodes)
                / guided_gates
                if guided_gates
                else None
            ),
            None if guided_gates else "no guided proposal reached the safety gate",
        ),
        "configuration_regret": metric_value(
            float(np.mean(regret_values)) if regret_complete else None,
            (
                None
                if regret_complete
                else "per-step oracle configuration cost was not available for every episode"
            ),
        ),
        "ensemble_uncertainty": _mean_metric(
            uncertainties, missing_reason="no physical action prediction was evaluated"
        ),
        "policy_latency_seconds": metric_value(
            (
                float(np.mean(policy_latencies))
                if policy_calls > 0 and policy_latency_missing == 0
                else None
            ),
            (
                None
                if policy_calls > 0 and policy_latency_missing == 0
                else "method does not call a meta policy"
                if policy_calls == 0
                else "generation latency was not supplied for every meta call"
            ),
        ),
        "planner_latency_seconds": _mean_metric(
            planner_latencies, missing_reason="no planner call was made"
        ),
        "candidate_evaluation_count": sum(
            int(row["candidate_evaluation_count"]) for row in episodes
        ),
        "candidate_evaluations_per_executed_step": metric_value(
            (
                sum(int(row["candidate_evaluation_count"]) for row in episodes)
                / control_steps
                if control_steps
                else None
            ),
            None if control_steps else "no continuous action was executed",
        ),
        "wall_clock_compute_seconds": float(
            sum(float(row["wall_clock_seconds"]) for row in episodes)
        ),
    }
    return {
        "method": str(episodes[0]["method"]),
        "policy_provenance": dict(episodes[0]["policy_provenance"]),
        "episode_count": len(episodes),
        "method_input_complete": complete,
        "incomplete_reasons": dict(sorted(incomplete_reasons.items())),
        "performance": performance,
        "audit": audit,
    }


def _paired_outcome(method: Mapping[str, Any], baseline: Mapping[str, Any]) -> int:
    method_success = bool(method["strict_all_five_success"])
    baseline_success = bool(baseline["strict_all_five_success"])
    if method_success != baseline_success:
        return 1 if method_success else -1
    method_error = float(method["final_normalized_error"])
    baseline_error = float(baseline["final_normalized_error"])
    if method_error < baseline_error - 1e-12:
        return 1
    if method_error > baseline_error + 1e-12:
        return -1
    method_steps = (
        math.inf if method["steps_to_success"] is None else int(method["steps_to_success"])
    )
    baseline_steps = (
        math.inf
        if baseline["steps_to_success"] is None
        else int(baseline["steps_to_success"])
    )
    if method_steps < baseline_steps:
        return 1
    if method_steps > baseline_steps:
        return -1
    return 0


def setup_bootstrap_interval(
    setup_values: Mapping[str, Sequence[float]],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if not setup_values:
        return metric_value(None, "no complete paired setup values")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    names = sorted(setup_values)
    per_setup = np.asarray(
        [float(np.mean(setup_values[name])) for name in names], dtype=np.float64
    )
    if not np.isfinite(per_setup).all():
        raise ValueError("bootstrap inputs contain non-finite values")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(per_setup), size=(samples, len(per_setup)))
    estimates = per_setup[draws].mean(axis=1)
    return metric_value(
        {
            "point_estimate": float(per_setup.mean()),
            "lower": float(np.quantile(estimates, 0.025)),
            "upper": float(np.quantile(estimates, 0.975)),
            "bootstrap_samples": int(samples),
            "bootstrap_seed": int(seed),
            "unit": "setup",
            "setup_count": len(names),
        }
    )


def paired_comparison(
    baseline_episodes: Sequence[Mapping[str, Any]],
    method_episodes: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    baseline = {str(row["record_id"]): row for row in baseline_episodes}
    method = {str(row["record_id"]): row for row in method_episodes}
    if set(baseline) != set(method):
        raise ClosedLoopEvaluationError("paired methods have different episode IDs")
    if not all(bool(row["method_input_complete"]) for row in method.values()):
        reason = "method has incomplete per-step prediction traces"
        return {
            "paired_win_loss_tie": metric_value(None, reason),
            "setup_bootstrap_95pct_ci": {
                key: metric_value(None, reason)
                for key in (
                    "strict_success_difference",
                    "final_error_improvement",
                    "objective_reduction_difference",
                )
            },
        }
    outcomes = [_paired_outcome(method[key], baseline[key]) for key in sorted(baseline)]
    by_setup: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key in sorted(baseline):
        left = baseline[key]
        right = method[key]
        setup_id = str(left["setup_id"])
        by_setup[setup_id]["strict_success_difference"].append(
            float(bool(right["strict_all_five_success"]))
            - float(bool(left["strict_all_five_success"]))
        )
        by_setup[setup_id]["final_error_improvement"].append(
            float(left["final_normalized_error"])
            - float(right["final_normalized_error"])
        )
        by_setup[setup_id]["objective_reduction_difference"].append(
            float(right["actual_canonical_objective_reduction"])
            - float(left["actual_canonical_objective_reduction"])
        )
    intervals: dict[str, Any] = {}
    for metric in (
        "strict_success_difference",
        "final_error_improvement",
        "objective_reduction_difference",
    ):
        intervals[metric] = setup_bootstrap_interval(
            {setup: values[metric] for setup, values in by_setup.items()},
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        )
    return {
        "paired_win_loss_tie": metric_value(
            {
                "wins": sum(value > 0 for value in outcomes),
                "losses": sum(value < 0 for value in outcomes),
                "ties": sum(value == 0 for value in outcomes),
                "paired_episodes": len(outcomes),
                "ordering": (
                    "strict success, then lower final normalized error, then fewer "
                    "steps to success; tolerance 1e-12"
                ),
            }
        ),
        "setup_bootstrap_95pct_ci": intervals,
    }


def _episode_outcome_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    """Timing-free signature used to verify repeated default runs before merge."""

    return {
        key: copy.deepcopy(row[key])
        for key in (
            "record_id",
            "setup_id",
            "target_id",
            "strict_all_five_success",
            "steps_to_success",
            "executed_control_steps",
            "final_normalized_error",
            "actual_canonical_objective_reduction",
            "candidate_evaluation_count",
            "termination_reason",
            "distance_trace",
        )
    }


def merge_closed_loop_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge method-subset reports after deterministic default verification."""

    if not reports:
        raise ClosedLoopEvaluationError("at least one partial report is required")
    first = copy.deepcopy(dict(reports[0]))
    expected_version = "qwen_h1_meta_v0_candidate_closed_loop_report_v1"
    if first.get("version") != expected_version:
        raise ClosedLoopEvaluationError("unexpected partial report version")
    reference_default = [
        _episode_outcome_signature(row)
        for row in first.get("episodes", {}).get("default_h1", [])
    ]
    if not reference_default:
        raise ClosedLoopEvaluationError("partial report has no default_h1 episodes")
    methods: dict[str, Any] = {
        "default_h1": copy.deepcopy(first["methods"]["default_h1"])
    }
    episodes: dict[str, Any] = {
        "default_h1": copy.deepcopy(first["episodes"]["default_h1"])
    }
    paired: dict[str, Any] = {}
    method_order = ["default_h1"]
    total_wall = 0.0
    limitations: list[str] = []
    manifest_hash: str | None = None
    for index, source in enumerate(reports):
        report = dict(source)
        if report.get("version") != expected_version:
            raise ClosedLoopEvaluationError(
                f"partial report {index} has a different version"
            )
        for key in ("manifest_record_count", "setup_count", "fairness"):
            if report.get(key) != first.get(key):
                raise ClosedLoopEvaluationError(
                    f"partial report {index} differs in {key}"
                )
        observed_default = [
            _episode_outcome_signature(row)
            for row in report.get("episodes", {}).get("default_h1", [])
        ]
        if observed_default != reference_default:
            raise ClosedLoopEvaluationError(
                f"partial report {index} does not reproduce default_h1 outcomes"
            )
        input_hash = report.get("inputs", {}).get("manifest_sha256")
        if input_hash is not None:
            if manifest_hash is None:
                manifest_hash = str(input_hash)
            elif str(input_hash) != manifest_hash:
                raise ClosedLoopEvaluationError(
                    "partial reports reference different candidate manifests"
                )
        for name in report.get("method_order", []):
            if name == "default_h1":
                continue
            if name in methods:
                raise ClosedLoopEvaluationError(
                    f"method {name!r} appears in more than one partial report"
                )
            methods[name] = copy.deepcopy(report["methods"][name])
            episodes[name] = copy.deepcopy(report["episodes"][name])
            paired[name] = copy.deepcopy(report["paired_vs_default"][name])
            method_order.append(name)
        total_wall += float(report.get("wall_clock_seconds", 0.0))
        for limitation in report.get("limitations", []):
            if limitation not in limitations:
                limitations.append(str(limitation))
    first["method_order"] = method_order
    first["methods"] = methods
    first["episodes"] = episodes
    first["paired_vs_default"] = paired
    first["wall_clock_seconds"] = total_wall
    first["limitations"] = limitations
    first["merged_partial_report_count"] = len(reports)
    first["merge_default_verification"] = (
        "timing-free default_h1 episode outcomes matched exactly"
    )
    return first


class PairedClosedLoopEvaluator:
    def __init__(
        self,
        *,
        backend: ClosedLoopBackend,
        methods: Sequence[MethodSpec],
        horizon_steps: int = HORIZON_STEPS,
        bootstrap_seed: int = BOOTSTRAP_SEED,
        bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    ) -> None:
        if horizon_steps != HORIZON_STEPS:
            raise ClosedLoopEvaluationError("candidate horizon must remain four steps")
        names = [method.name for method in methods]
        if not names or len(names) != len(set(names)):
            raise ClosedLoopEvaluationError("method names must be nonempty and unique")
        if "default_h1" not in names:
            raise ClosedLoopEvaluationError("paired evaluator requires default_h1")
        if any("h3" in name.lower() for name in names):
            raise ClosedLoopEvaluationError("H3 is forbidden")
        backend.bounds.validate()
        self.backend = backend
        self.methods = tuple(methods)
        self.horizon_steps = horizon_steps
        self.bootstrap_seed = int(bootstrap_seed)
        self.bootstrap_samples = int(bootstrap_samples)

    def evaluate(
        self,
        manifest_records: Sequence[Mapping[str, Any]],
        *,
        require_preregistered_cardinality: bool = True,
    ) -> dict[str, Any]:
        records = normalize_candidate_eval_manifest(
            manifest_records,
            require_preregistered_cardinality=require_preregistered_cardinality,
        )
        started = time.perf_counter()
        episodes_by_method: dict[str, list[dict[str, Any]]] = {}
        for method in self.methods:
            episodes_by_method[method.name] = [
                _run_episode(
                    record=record,
                    method=method,
                    backend=self.backend,
                    horizon_steps=self.horizon_steps,
                )
                for record in records
            ]
        summaries = {
            name: aggregate_method_episodes(rows)
            for name, rows in episodes_by_method.items()
        }
        default_rows = episodes_by_method["default_h1"]
        paired = {
            name: paired_comparison(
                default_rows,
                rows,
                bootstrap_seed=self.bootstrap_seed,
                bootstrap_samples=self.bootstrap_samples,
            )
            for name, rows in episodes_by_method.items()
            if name != "default_h1"
        }
        return {
            "version": "qwen_h1_meta_v0_candidate_closed_loop_report_v1",
            "candidate_only": True,
            "formal_frozen_evaluation_enabled": False,
            "scientific_conclusion": False,
            "evaluation_scope": {
                "level": "component_level_corrected_simulator_evaluation",
                "supervisor_state_source": (
                    "candidate manifest model_visible_input.measurement_validity; "
                    "synthetic candidate-data validity, not supervisor inference"
                ),
                "actual_supervisor_inference_executed": False,
                "reobserve_recovery_backend_validated": False,
                "full_stack_or_end_to_end_claim_allowed": False,
            },
            "manifest_record_count": len(records),
            "setup_count": len(
                {str(row["evaluator_only"]["setup_id"]) for row in records}
            ),
            "method_order": [method.name for method in self.methods],
            "fairness": {
                "paired": True,
                "horizon_steps": self.horizon_steps,
                "planner_seed_root": PLANNER_SEED_ROOT,
                "planner_seed_key": "setup_id,target_counterfactual_id,step; no method name",
                "candidate_manifest_registry": "qh1meta_eval_0000..0011 x target indices 0,1,2",
                "target_seed": TARGET_SEED,
                "oracle_seed": ORACLE_SEED,
                "simulator_transition_rule": (
                    "same deterministic corrected-v12.1 setup_context and simulator_fixed "
                    "record are reused for every paired method"
                ),
                "fixed_initial_tolerances": True,
                "bounds": {
                    "action_low": self.backend.bounds.action_low.tolist(),
                    "action_high": self.backend.bounds.action_high.tolist(),
                    "position_low": self.backend.bounds.position_low.tolist(),
                    "position_high": self.backend.bounds.position_high.tolist(),
                },
                "default_candidate_evaluations_per_step": 72,
                "guarded_total_candidate_evaluations_per_step": 144,
                "dual_budget_default_population": 48,
                "dual_budget_default_iterations": 3,
                "dual_budget_default_elites": default_h1_config()["elites"],
                "backend_identity": dict(self.backend.identity),
                "image_policy_rule": (
                    "only policies declaring requires_sensor_image trigger corrected-simulator "
                    "image generation; numeric/default methods never open manifest images"
                ),
            },
            "metric_definitions": {
                "strict_success": "all five fixed-tolerance normalized errors <= 1.0",
                "error_step_auc": "unnormalized trapezoidal area over max-error trace",
                "canonical_state_objective": "max normalized error + 0.15*mean normalized error",
                "action_norm": "L2 norm of action divided elementwise by default positive bounds",
                "overshoot": "any metric error changes strict sign across an executed transition",
                "out_of_domain_proposal": "CEM initially-out-of-feasible-bounds proposal fraction",
            },
            "methods": summaries,
            "paired_vs_default": paired,
            "episodes": episodes_by_method,
            "wall_clock_seconds": time.perf_counter() - started,
            "limitations": [
                "manifest image paths are never opened; live image-aware policies receive corrected-simulator images",
                "replayed Qwen traces must bind every step to the current runtime-input and sensor-image fingerprints",
                "sequential exposure/reflection recovery has no real repository backend; reobserve terminates with an explicit reason",
                "supervisor decisions are reconstructed from synthetic candidate-manifest validity fields; no frozen Qwen supervisor inference is run",
                "this is component-level simulator evaluation, not a full-stack or end-to-end validation",
                "configuration regret is null unless per-step oracle costs are supplied",
                "candidate results are not a formal frozen evaluation or scientific conclusion",
            ],
            "terminal_line": "QWEN-H1 META-CONTROLLER CANDIDATE — READY FOR HUMAN REVIEW",
        }


def _parse_prediction_args(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ClosedLoopEvaluationError(
                "--prediction must be METHOD=/candidate/path.jsonl"
            )
        method, raw_path = value.split("=", 1)
        if method not in STANDARD_METHOD_NAMES:
            raise ClosedLoopEvaluationError(f"unknown prediction method {method!r}")
        if method in {"default_h1", "dual_budget_default_h1", "rule_guided_h1"}:
            raise ClosedLoopEvaluationError(
                f"{method} does not accept an external prediction trace"
            )
        if method in output:
            raise ClosedLoopEvaluationError(
                f"duplicate prediction path for {method}"
            )
        output[method] = _guard_candidate_path(
            Path(raw_path), role=f"{method} prediction trace", must_exist=True
        )
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="candidate-only paired corrected-simulator closed-loop evaluation"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="candidate-only per-step raw-output JSONL; repeat per method",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        choices=STANDARD_METHOD_NAMES,
        default=[],
        help=(
            "evaluate a subset so only one live adapter need be loaded; "
            "default_h1 is always included"
        ),
    )
    parser.add_argument("--forward-checkpoint", type=Path, default=DEFAULT_FORWARD_CHECKPOINT)
    parser.add_argument("--simulator-config", type=Path, default=DEFAULT_SIMULATOR_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest_path = _guard_candidate_path(
        args.manifest, role="candidate-eval manifest", must_exist=True
    )
    if "candidate_eval" not in manifest_path.name.lower():
        raise ClosedLoopEvaluationError(
            "manifest filename must explicitly identify candidate_eval"
        )
    output_path = _guard_candidate_path(
        args.output, role="closed-loop output", must_exist=False
    )
    prediction_paths = _parse_prediction_args(args.prediction)
    checkpoint = args.forward_checkpoint.expanduser().resolve()
    simulator_config = args.simulator_config.expanduser().resolve()
    base_config = args.base_config.expanduser().resolve()
    source_freeze = verify_closed_loop_source_freeze(
        forward_checkpoint_path=checkpoint,
        simulator_config_path=simulator_config,
        base_config_path=base_config,
    )
    checkpoint_hash = source_freeze["forward_checkpoint"]["sha256"]
    simulator_payload = _read_json(simulator_config)
    bounds = Bounds.from_config(simulator_payload)
    bounds.validate()
    model = load_forward_ensemble(checkpoint, device_name=str(args.device))
    backend = CorrectedSimulatorBackend(
        model=model,
        bounds=bounds,
        simulator_config_path=simulator_config,
        base_config_path=base_config,
        forward_checkpoint_sha256=checkpoint_hash,
    )
    policies = {
        method: TracePolicy(
            _read_jsonl(path),
            require_dynamic_image_binding=(
                method.startswith("qwen_guided_h1_seed_")
                or method == "shadow_qwen_h1"
            ),
        )
        for method, path in prediction_paths.items()
    }
    records = _read_jsonl(manifest_path)
    all_methods = standard_method_specs(policies)
    if args.method:
        selected_names = set(args.method) | {"default_h1"}
        methods = tuple(
            method for method in all_methods if method.name in selected_names
        )
    else:
        methods = all_methods
    report = PairedClosedLoopEvaluator(
        backend=backend,
        methods=methods,
    ).evaluate(records, require_preregistered_cardinality=True)
    report["inputs"] = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_path(manifest_path),
        "prediction_traces": {
            name: {"path": str(path), "sha256": _sha256_path(path)}
            for name, path in sorted(prediction_paths.items())
        },
        "verified_source_freeze": source_freeze,
    }
    _atomic_write_json(output_path, report)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output_path),
                "methods": report["method_order"],
                "wall_clock_seconds": report["wall_clock_seconds"],
                "candidate_only": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "CallablePolicy",
    "ClosedLoopBackend",
    "ClosedLoopEvaluationError",
    "CorrectedSimulatorBackend",
    "HORIZON_STEPS",
    "LiveImagePathPolicy",
    "LiveImagePolicy",
    "MetaPolicy",
    "MetaPolicyResult",
    "MethodSpec",
    "PLANNER_SEED_ROOT",
    "PairedClosedLoopEvaluator",
    "PolicyContext",
    "RulePolicy",
    "STANDARD_METHOD_NAMES",
    "SimulationObservation",
    "TracePolicy",
    "UnavailablePolicy",
    "aggregate_method_episodes",
    "canonical_state_objective",
    "error_step_auc",
    "metric_value",
    "merge_closed_loop_reports",
    "normalize_candidate_eval_manifest",
    "paired_comparison",
    "setup_bootstrap_interval",
    "runtime_input_fingerprint",
    "sensor_image_fingerprint",
    "standard_method_specs",
    "strict_success",
    "verify_closed_loop_source_freeze",
]
