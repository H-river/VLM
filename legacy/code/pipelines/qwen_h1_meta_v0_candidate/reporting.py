#!/usr/bin/env python3
"""Deterministic, candidate-only reporting for the Qwen-H1 meta-controller.

The reporter is intentionally model-free.  It only reads already-produced
candidate evidence, renders bounded reports, and hashes candidate artifacts.
It never opens an image and it refuses evidence paths outside this package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_ROOT / "protocol/meta_controller_protocol.json"
STATUS_LINE = "CANDIDATE ONLY — NOT SEALED — FROZEN EVALUATION DISABLED"
TERMINAL_LINE = "QWEN-H1 META-CONTROLLER CANDIDATE — READY FOR HUMAN REVIEW"
TRAINING_SEEDS = (2026080201, 2026080202, 2026080203)
REPORT_FILENAMES = (
    "training_report.md",
    "regression_and_safety_report.md",
    "commands.md",
    "machine_summary.json",
    "final_report.md",
    "artifact_hashes.json",
)
JSON_RESULT_VERSIONS = {
    "data_audit": {"qwen_h1_meta_v0_data_audit_v1"},
    "offline": {
        "qwen_h1_meta_v0_offline_evaluation_v1",
        "qwen_h1_meta_reasoning_ablation_offline_v1",
    },
    "closed_loop": {"qwen_h1_meta_v0_candidate_closed_loop_report_v1"},
    "ablation": {"qwen_h1_meta_reasoning_ablation_offline_v1"},
    "execution": {"qwen_h1_meta_v0_execution_evidence_v1"},
}
FORBIDDEN_EVIDENCE_TOKENS = {"frozen", "protected", "iid", "ood", "test", "tests"}
FORBIDDEN_ARTIFACT_TOKENS = {"frozen", "protected", "iid", "ood"}
HASH_EXCLUDED_DIR_NAMES = {"__pycache__", ".pytest_cache", ".git", "partial"}
HASH_EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo", ".tmp", ".lock", ".incomplete"}
HASH_MANIFEST_FILENAME = "artifact_hashes.json"
PACKAGE_VERSIONS = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "Pillow",
    "PyYAML",
    "qwen-vl-utils",
    "numpy",
)


class ReportingError(ValueError):
    """Invalid or unsafe candidate reporting input."""


class ArtifactValidationError(ReportingError):
    """Generated artifacts do not match the deterministic manifest."""


@dataclass(frozen=True)
class ReportSources:
    """Candidate-only evidence files consumed by :func:`build_reports`."""

    data_audit: Path | None = None
    training_runs: tuple[Path, ...] = ()
    offline: Path | None = None
    closed_loop: Path | None = None
    ablation: Path | None = None
    regression: Path | None = None
    execution_evidence: Path | None = None
    command_log: Path | None = None


def _path_tokens(part: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", part.lower()) if token}


def _contains_forbidden_marker(path: Path) -> bool:
    return any(
        _path_tokens(part) & FORBIDDEN_EVIDENCE_TOKENS for part in path.parts
    )


def guard_candidate_path(
    path: Path,
    *,
    role: str,
    must_exist: bool = True,
    namespace_root: Path = PACKAGE_ROOT,
) -> Path:
    """Resolve a path and prove it remains inside the candidate namespace."""

    root = namespace_root.resolve()
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ReportingError(f"{role} must remain inside {root}") from exc
    if _contains_forbidden_marker(relative):
        raise ReportingError(f"{role} path contains a forbidden split marker")
    if must_exist and not resolved.is_file():
        raise ReportingError(f"{role} does not exist as a regular file: {resolved}")
    return resolved


def _reject_constant(value: str) -> None:
    raise ReportingError(f"non-finite JSON constant is forbidden: {value}")


def _read_json(path: Path, *, role: str, namespace_root: Path) -> dict[str, Any]:
    guarded = guard_candidate_path(
        path, role=role, must_exist=True, namespace_root=namespace_root
    )
    try:
        value = json.loads(
            guarded.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportingError(f"{role} is not valid UTF-8 JSON: {guarded}") from exc
    if not isinstance(value, dict):
        raise ReportingError(f"{role} must be a JSON object")
    _validate_finite_tree(value, role=role)
    return value


def _validate_finite_tree(value: Any, *, role: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReportingError(f"{role} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ReportingError(f"{role} contains a non-string object key")
            _validate_finite_tree(child, role=role)
        return
    if isinstance(value, list):
        for child in value:
            _validate_finite_tree(child, role=role)
        return
    raise ReportingError(f"{role} contains a non-JSON value")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(path, _canonical_json(value))


def _protocol(namespace_root: Path) -> dict[str, Any]:
    # A synthetic output root still consumes the one immutable package protocol.
    path = PROTOCOL_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != STATUS_LINE:
        raise ReportingError("meta-controller protocol candidate status changed")
    if value.get("final_terminal_line") != TERMINAL_LINE:
        raise ReportingError("meta-controller protocol terminal line changed")
    if tuple(value.get("training", {}).get("seeds", ())) != TRAINING_SEEDS:
        raise ReportingError("meta-controller protocol training seeds changed")
    return value


def _validate_common_candidate_evidence(value: Mapping[str, Any], *, role: str) -> None:
    if value.get("formal_frozen_evaluation_enabled") is True:
        # Accepted as RED evidence by verdict computation, never silently normalized.
        return
    if value.get("candidate_only") is False:
        raise ReportingError(f"{role} declares non-candidate scope")


def _validate_inputs(
    sources: ReportSources, *, namespace_root: Path
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    evidence: dict[str, Any] = {}
    provenance: dict[str, dict[str, Any]] = {}

    def load(role: str, path: Path | None) -> None:
        if path is None:
            return
        guarded = guard_candidate_path(
            path, role=role, must_exist=True, namespace_root=namespace_root
        )
        value = _read_json(guarded, role=role, namespace_root=namespace_root)
        _validate_common_candidate_evidence(value, role=role)
        expected = JSON_RESULT_VERSIONS.get(role)
        if expected is not None and value.get("version") not in expected:
            raise ReportingError(
                f"{role}.version must be one of {sorted(expected)!r}; "
                f"received {value.get('version')!r}"
            )
        evidence[role] = value
        provenance[role] = {
            "path": guarded.relative_to(namespace_root.resolve()).as_posix(),
            "sha256": _sha256_path(guarded),
            "bytes": guarded.stat().st_size,
        }

    load("data_audit", sources.data_audit)
    load("offline", sources.offline)
    load("closed_loop", sources.closed_loop)
    load("ablation", sources.ablation)
    load("regression", sources.regression)
    load("execution", sources.execution_evidence)

    training_runs: list[dict[str, Any]] = []
    for index, source in enumerate(sources.training_runs):
        role = f"training_run_{index}"
        guarded = guard_candidate_path(
            source, role=role, must_exist=True, namespace_root=namespace_root
        )
        value = _read_json(guarded, role=role, namespace_root=namespace_root)
        if value.get("manifest_version") != "qwen_h1_meta_training_run_v0":
            raise ReportingError(
                f"{role}.manifest_version must be qwen_h1_meta_training_run_v0"
            )
        if value.get("independent_adapter_name") != "qwen_h1_meta_v0":
            raise ReportingError(f"{role} is not the independent meta-controller adapter")
        data = value.get("data")
        if not isinstance(data, Mapping):
            raise ReportingError(f"{role}.data must be an object")
        if data.get("frozen_or_protected_predictions_opened") is not False:
            raise ReportingError(
                f"{role} must explicitly state that frozen/protected predictions were not opened"
            )
        seed = value.get("training", {}).get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ReportingError(f"{role} has no integer training.seed")
        training_runs.append(value)
        provenance[role] = {
            "path": guarded.relative_to(namespace_root.resolve()).as_posix(),
            "sha256": _sha256_path(guarded),
            "bytes": guarded.stat().st_size,
            "seed": seed,
        }
    seeds = [int(run["training"]["seed"]) for run in training_runs]
    if len(seeds) != len(set(seeds)):
        raise ReportingError("training run manifests contain duplicate seeds")
    if any(seed not in TRAINING_SEEDS for seed in seeds):
        raise ReportingError("training run manifest uses a non-preregistered seed")
    training_runs.sort(key=lambda run: int(run["training"]["seed"]))

    commands: list[dict[str, Any]] = []
    if "execution" in evidence:
        commands.extend(_commands_from_execution_evidence(evidence["execution"]))
    if sources.command_log is not None:
        guarded = guard_candidate_path(
            sources.command_log,
            role="command_log",
            must_exist=True,
            namespace_root=namespace_root,
        )
        commands.extend(_read_command_log(guarded))
        provenance["command_log"] = {
            "path": guarded.relative_to(namespace_root.resolve()).as_posix(),
            "sha256": _sha256_path(guarded),
            "bytes": guarded.stat().st_size,
        }
    deduplicated: list[dict[str, Any]] = []
    seen_commands: set[str] = set()
    for command in commands:
        text = str(command["command"])
        if text not in seen_commands:
            seen_commands.add(text)
            deduplicated.append(command)
    return evidence, provenance, deduplicated, training_runs


def _read_command_log(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        raw = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError:
        raw = [{"command": line} for line in text.splitlines() if line.strip()]
    if isinstance(raw, Mapping):
        if isinstance(raw.get("data_generation"), Mapping):
            return _commands_from_execution_evidence(raw)
        raw = raw.get("commands")
    if not isinstance(raw, list):
        raise ReportingError("command log must be a list or an object with commands")
    commands: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            item = {"command": item}
        if not isinstance(item, Mapping):
            raise ReportingError(f"command log entry {index} must be an object or string")
        command = item.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ReportingError(f"command log entry {index} has no command")
        normalized = dict(item)
        normalized["command"] = command.strip()
        _validate_finite_tree(normalized, role=f"command_log[{index}]")
        commands.append(normalized)
    return commands


def _commands_from_execution_evidence(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    generation = value.get("data_generation")
    if not isinstance(generation, Mapping):
        raise ReportingError("execution evidence has no data_generation object")
    command = generation.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ReportingError("execution evidence has no exact data-generation command")
    maximum_resident_set_kib = generation.get("maximum_resident_set_kib")
    peak_memory_bytes = (
        int(maximum_resident_set_kib) * 1024
        if isinstance(maximum_resident_set_kib, int)
        and not isinstance(maximum_resident_set_kib, bool)
        and maximum_resident_set_kib >= 0
        else None
    )
    entry: dict[str, Any] = {
        "command": command.strip(),
        "purpose": "Generate and audit preregistered candidate-only data",
        "exit_code": generation.get("exit_status"),
        "status": generation.get("exit_status_meaning"),
        "wall_time_seconds": generation.get("pipeline_elapsed_seconds"),
        "maximum_resident_set_kib": maximum_resident_set_kib,
        "peak_memory_bytes": peak_memory_bytes,
        "notes": (
            f"/usr/bin/time wall={generation.get('time_wall_clock')}; "
            f"CPU={generation.get('cpu_percent')}%; SFT exported="
            f"{generation.get('sft_exported')}"
        ),
    }
    _validate_finite_tree(entry, role="execution_evidence.data_generation")
    return [entry]


def _run_read_only(command: Sequence[str], *, cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=None if cwd is None else str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def collect_machine_summary(
    *, provenance: Mapping[str, Any], verdict: Mapping[str, Any]
) -> dict[str, Any]:
    """Collect stable, read-only software/GPU identity (never live utilization)."""

    packages: dict[str, str | None] = {}
    for name in PACKAGE_VERSIONS:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    gpu_rows: list[dict[str, Any]] = []
    gpu_query = _run_read_only(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_query:
        for line in gpu_query.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                continue
            try:
                index = int(fields[0])
                memory_mib = int(fields[3])
            except ValueError:
                continue
            gpu_rows.append(
                {
                    "index": index,
                    "name": fields[1],
                    "driver_version": fields[2],
                    "memory_total_mib": memory_mib,
                }
            )

    repository_root = PACKAGE_ROOT.parent
    commit = _run_read_only(["git", "rev-parse", "HEAD"], cwd=repository_root)
    branch = _run_read_only(
        ["git", "branch", "--show-current"], cwd=repository_root
    )
    return {
        "schema_version": "qwen_h1_meta_v0_machine_summary_v1",
        "status": STATUS_LINE,
        "candidate_only": True,
        "formal_frozen_evaluation_enabled": False,
        "collection_policy": (
            "read-only stable identity; excludes GPU utilization, hostname, username, "
            "environment variables, and timestamps"
        ),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "packages": packages,
        "gpu": {
            "query": "index,name,driver_version,memory.total only",
            "available": bool(gpu_rows),
            "devices": gpu_rows,
        },
        "repository": {"commit": commit, "branch": branch},
        "evidence_inputs": dict(sorted(provenance.items())),
        "report_verdict": dict(verdict),
    }


def _lookup(mapping: Mapping[str, Any] | None, *paths: str) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    for dotted in paths:
        value: Any = mapping
        for key in dotted.split("."):
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return None


def _bool_failure(
    mapping: Mapping[str, Any] | None,
    *,
    positive_paths: Sequence[str] = (),
    negative_paths: Sequence[str] = (),
) -> bool:
    positive = _lookup(mapping, *positive_paths)
    if positive is False:
        return True
    negative = _lookup(mapping, *negative_paths)
    return negative is True


def _bool_explicit_pass(mapping: Mapping[str, Any] | None, *paths: str) -> bool:
    return _lookup(mapping, *paths) is True


def _metric_number(value: Any) -> float | None:
    if isinstance(value, Mapping) and set(value) >= {"value", "reason"}:
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _method_metric(
    closed_loop: Mapping[str, Any] | None, method: str, *path: str
) -> float | None:
    value: Any = _lookup(closed_loop, f"methods.{method}")
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return _metric_number(value)


def _qwen_candidate_gain(closed_loop: Mapping[str, Any] | None) -> bool:
    if not isinstance(closed_loop, Mapping):
        return False
    if closed_loop.get("manifest_record_count") != 36 or closed_loop.get("setup_count") != 12:
        return False
    required_methods = {
        "default_h1",
        "dual_budget_default_h1",
        "rule_guided_h1",
        "metrics_mlp_meta_h1",
        "oracle_guided_h1",
        "random_or_frequency_meta_h1",
        "shadow_qwen_h1",
        *(f"qwen_guided_h1_seed_{seed}" for seed in TRAINING_SEEDS),
    }
    methods = closed_loop.get("methods")
    if not isinstance(methods, Mapping) or not required_methods.issubset(methods):
        return False
    if any(
        not isinstance(methods.get(name), Mapping)
        or methods[name].get("method_input_complete") is not True
        or methods[name].get("episode_count") != 36
        for name in required_methods
    ):
        return False
    qwen_names = [f"qwen_guided_h1_seed_{seed}" for seed in TRAINING_SEEDS]
    qwen = [
        _method_metric(
            closed_loop, name, "performance", "strict_all_five_success"
        )
        for name in qwen_names
    ]
    default = _method_metric(
        closed_loop, "default_h1", "performance", "strict_all_five_success"
    )
    dual = _method_metric(
        closed_loop,
        "dual_budget_default_h1",
        "performance",
        "strict_all_five_success",
    )
    reasonable = [
        _method_metric(
            closed_loop, name, "performance", "strict_all_five_success"
        )
        for name in ("rule_guided_h1", "metrics_mlp_meta_h1")
    ]
    if any(value is None for value in qwen) or default is None or dual is None:
        return False
    if not all(float(value) > default + 1e-12 and float(value) > dual + 1e-12 for value in qwen):
        return False
    if not all(
        (_paired_ci_lower(closed_loop, baseline, name) or -math.inf) > 0.0
        for baseline in ("default_h1", "dual_budget_default_h1")
        for name in qwen_names
    ):
        return False
    # Use one fixed reasonable comparator across all seeds. Choosing the weaker
    # baseline per seed (or taking min(rule, MLP)) would be opportunistic.
    reasonable_supported = False
    for baseline, baseline_value in zip(
        ("rule_guided_h1", "metrics_mlp_meta_h1"), reasonable, strict=True
    ):
        if baseline_value is None:
            continue
        if all(float(value) > baseline_value + 1e-12 for value in qwen) and all(
            (_paired_ci_lower(closed_loop, baseline, name) or -math.inf) > 0.0
            for name in qwen_names
        ):
            reasonable_supported = True
    return reasonable_supported


def _ci_lower(value: Any) -> float | None:
    if isinstance(value, Mapping) and set(value) >= {"value", "reason"}:
        value = value.get("value")
    if isinstance(value, Mapping):
        value = value.get("lower")
    return _metric_number(value)


def _paired_ci_lower(
    closed_loop: Mapping[str, Any], baseline: str, method: str
) -> float | None:
    suffix = "setup_bootstrap_95pct_ci.strict_success_difference"
    candidates: list[Any] = []
    if baseline == "default_h1":
        candidates.append(_lookup(closed_loop, f"paired_vs_default.{method}.{suffix}"))
    aliases = {
        "dual_budget_default_h1": "paired_vs_dual_budget_default",
        "rule_guided_h1": "paired_vs_rule_guided",
        "metrics_mlp_meta_h1": "paired_vs_metrics_mlp",
    }
    if baseline in aliases:
        candidates.append(_lookup(closed_loop, f"{aliases[baseline]}.{method}.{suffix}"))
    candidates.extend(
        [
            _lookup(closed_loop, f"paired_vs_baseline.{baseline}.{method}.{suffix}"),
            _lookup(closed_loop, f"paired_comparisons.{baseline}.{method}.{suffix}"),
        ]
    )
    for candidate in candidates:
        lower = _ci_lower(candidate)
        if lower is not None:
            return lower
    return None


def _ablation_seed_rows(
    ablation: Mapping[str, Any], name: str
) -> list[Mapping[str, Any]] | None:
    rows = _lookup(ablation, f"per_ablation.{name}.paired_vs_full.per_seed")
    if not isinstance(rows, list):
        return None
    normalized = [row for row in rows if isinstance(row, Mapping)]
    if {row.get("seed") for row in normalized} != set(TRAINING_SEEDS):
        return None
    return normalized


def _ablation_ci_lower(ablation: Mapping[str, Any], name: str) -> float | None:
    for path in (
        f"per_ablation.{name}.paired_vs_full.setup_bootstrap_95pct_ci.configuration_regret_delta_ablated_minus_full",
        f"per_ablation.{name}.setup_bootstrap_95pct_ci.configuration_regret_delta_ablated_minus_full",
    ):
        lower = _ci_lower(_lookup(ablation, path))
        if lower is not None:
            return lower
    return None


def _ablation_reasoning_support(
    ablation: Mapping[str, Any] | None,
    *,
    closed_loop_gain_supported: bool,
    regression: Mapping[str, Any] | None,
) -> bool:
    """Derive the fixed reasoning claim; never trust a reported verdict boolean."""

    if not isinstance(ablation, Mapping) or _declared_not_run(ablation):
        return False
    if ablation.get("version") != "qwen_h1_meta_reasoning_ablation_offline_v1":
        return False
    if ablation.get("manifest_records") != 36 or tuple(
        ablation.get("expected_seeds", ())
    ) != TRAINING_SEEDS:
        return False
    target_rows = _ablation_seed_rows(ablation, "target_shuffle")
    semantics_rows = _ablation_seed_rows(ablation, "actuator_semantics_shuffle")
    if target_rows is None or semantics_rows is None:
        return False

    def row_supports_change_and_regret(row: Mapping[str, Any]) -> bool:
        changes = row.get("field_change_rate")
        regret = row.get("configuration_regret_delta_ablated_minus_full")
        if not isinstance(changes, Mapping) or not isinstance(regret, Mapping):
            return False
        control_changes = [
            float(value)
            for field, value in changes.items()
            if (
                field.startswith("directional_prior.")
                or field
                in {
                    "decision",
                    "objective_profile",
                    "mask_profile",
                    "step_scale",
                    "risk_mode",
                }
            )
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]
        return (
            bool(control_changes)
            and max(control_changes) > 0.0
            and regret.get("evaluated_count") == 36
            and _metric_number(regret.get("coverage_rate")) == 1.0
            and (_metric_number(regret.get("mean")) or -math.inf) > 0.0
        )

    target_supported = all(row_supports_change_and_regret(row) for row in target_rows)
    semantics_supported = all(
        row_supports_change_and_regret(row) for row in semantics_rows
    )
    target_ci = (_ablation_ci_lower(ablation, "target_shuffle") or -math.inf) > 0.0
    semantics_ci = (
        _ablation_ci_lower(ablation, "actuator_semantics_shuffle") or -math.inf
    ) > 0.0
    semantics_physical = _lookup(
        ablation,
        "per_ablation.actuator_semantics_shuffle.physical_validation.claim_allowed",
    ) is True
    safety_paths = (
        ("safety.feature_flag_off_equivalent", "off_equivalence"),
        ("safety.compiler_bounds_preserved", "compiler_bounds_preserved"),
        (
            "safety.continuous_action_injection_rejected",
            "continuous_action_injection_rejected",
        ),
        ("safety.dispatch_gate_enforced", "dispatch_gate_enforced"),
        ("safety.h3_disabled", "h3_disabled"),
    )
    safety_supported = isinstance(regression, Mapping) and all(
        _bool_explicit_pass(regression, *paths) for paths in safety_paths
    )
    return all(
        (
            target_supported,
            semantics_supported,
            target_ci,
            semantics_ci,
            semantics_physical,
            closed_loop_gain_supported,
            safety_supported,
        )
    )


def _declared_not_run(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    status = str(value.get("status", "")).lower()
    reason = str(value.get("reason", value.get("unavailable_reason", ""))).lower()
    return status in {"unavailable", "not_run", "skipped"} and (
        "not_run_due_to_data_gate" in reason
        or "not_run_due_to_information_sufficiency_gate" in reason
    )


def compute_verdict(
    *, evidence: Mapping[str, Any], training_runs: Sequence[Mapping[str, Any]], commands: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compute conservative GO/YELLOW/RED without tuning to result values."""

    data = evidence.get("data_audit")
    offline = evidence.get("offline")
    execution = evidence.get("execution")
    closed = evidence.get("closed_loop")
    ablation = evidence.get("ablation")
    regression = evidence.get("regression")
    red_reasons: list[str] = []
    yellow_reasons: list[str] = []

    information_gate_red = bool(data is not None and (
        data.get("gate") != "PASS" or data.get("sft_export_permitted") is not True
    ))
    if information_gate_red:
        red_reasons.append("information_insufficient_or_data_gate_red")
    if isinstance(data, Mapping):
        overlap = data.get("identity_overlap", {})
        if int(overlap.get("known_overlap_count", 0) or 0) != 0 or int(
            overlap.get("cross_split_overlap_count", 0) or 0
        ) != 0:
            red_reasons.append("train_dev_or_protected_setup_leakage")

    if _bool_failure(
        regression,
        positive_paths=("safety.compiler_bounds_preserved", "compiler_bounds_preserved"),
        negative_paths=("safety.compiler_expands_action_bound", "compiler_expands_action_bound"),
    ):
        red_reasons.append("compiler_expands_action_bound")
    if _bool_failure(
        regression,
        positive_paths=(
            "safety.continuous_action_injection_rejected",
            "continuous_action_injection_rejected",
        ),
        negative_paths=(
            "safety.continuous_action_injection_accepted",
            "continuous_action_injection_accepted",
        ),
    ):
        red_reasons.append("continuous_action_injection_accepted")
    if _bool_failure(
        regression,
        positive_paths=("safety.feature_flag_off_equivalent", "off_equivalence"),
        negative_paths=("safety.off_equivalence_failed", "off_equivalence_failed"),
    ):
        red_reasons.append("off_mode_default_h1_not_reproduced")
    if _bool_failure(
        regression,
        positive_paths=("safety.dispatch_gate_enforced", "dispatch_gate_enforced"),
        negative_paths=("safety.dispatch_gate_bypass", "dispatch_gate_bypass"),
    ):
        red_reasons.append("dispatch_gate_bypass")
    if _bool_failure(
        regression,
        positive_paths=("safety.h3_disabled", "h3_disabled"),
        negative_paths=("safety.h3_invoked", "h3_invoked"),
    ):
        red_reasons.append("h3_invoked")
    if _bool_failure(
        regression,
        positive_paths=(
            "safety.existing_state_machine_tests_passed",
            "existing_state_machine_tests_passed",
        ),
        negative_paths=(
            "safety.existing_state_machine_regression",
            "existing_state_machine_regression",
        ),
    ):
        red_reasons.append("existing_state_machine_regression")
    if _bool_failure(
        regression,
        positive_paths=(
            "safety.canonical_objective_and_strict_success_preserved",
            "canonical_objective_and_strict_success_preserved",
        ),
        negative_paths=(
            "safety.canonical_objective_or_strict_success_changed",
            "canonical_objective_or_strict_success_changed",
        ),
    ):
        red_reasons.append("canonical_objective_or_strict_success_changed")
    if _bool_failure(
        regression,
        positive_paths=(
            "safety.frozen_or_protected_content_untouched",
            "frozen_or_protected_content_untouched",
        ),
        negative_paths=(
            "safety.frozen_or_protected_content_read_or_predicted",
            "frozen_or_protected_content_read_or_predicted",
        ),
    ):
        red_reasons.append("frozen_or_protected_image_read_or_prediction")

    for role, value in evidence.items():
        if isinstance(value, Mapping) and value.get("formal_frozen_evaluation_enabled") is True:
            red_reasons.append(f"{role}_formal_frozen_evaluation_enabled")
        declared = value.get("red_reasons") if isinstance(value, Mapping) else None
        if isinstance(declared, list):
            red_reasons.extend(str(reason) for reason in declared if reason)
    if isinstance(regression, Mapping):
        if str(regression.get("status", "")).upper() in {"FAIL", "FAILED", "RED", "RED_STOP"}:
            red_reasons.append("regression_evidence_declared_failure")
        declared_stops = regression.get("stop_conditions")
        if isinstance(declared_stops, Mapping):
            red_reasons.extend(
                str(name) for name, active in declared_stops.items() if active is True
            )
        elif isinstance(declared_stops, list):
            red_reasons.extend(str(name) for name in declared_stops if name)

    required_roles = ("data_audit", "offline", "closed_loop", "ablation", "regression")
    for role in required_roles:
        if role not in evidence:
            if not information_gate_red or role in {"data_audit", "regression"}:
                yellow_reasons.append(f"missing_{role}_evidence")
    seeds = {int(run["training"]["seed"]) for run in training_runs}
    if seeds != set(TRAINING_SEEDS) and not information_gate_red:
        yellow_reasons.append("three_preregistered_training_runs_incomplete")
    elif any(run.get("status") != "completed" for run in training_runs):
        yellow_reasons.append("one_or_more_training_runs_not_completed")
    if not commands:
        yellow_reasons.append("actual_command_ledger_missing")
    safety_pass_groups = (
        ("safety.feature_flag_off_equivalent", "off_equivalence"),
        ("safety.compiler_bounds_preserved", "compiler_bounds_preserved"),
        (
            "safety.continuous_action_injection_rejected",
            "continuous_action_injection_rejected",
        ),
        ("safety.dispatch_gate_enforced", "dispatch_gate_enforced"),
        ("safety.h3_disabled", "h3_disabled"),
        (
            "safety.existing_state_machine_tests_passed",
            "existing_state_machine_tests_passed",
        ),
        (
            "safety.canonical_objective_and_strict_success_preserved",
            "canonical_objective_and_strict_success_preserved",
        ),
        (
            "safety.frozen_or_protected_content_untouched",
            "frozen_or_protected_content_untouched",
        ),
    )
    if regression is not None and not all(
        _bool_explicit_pass(regression, *paths) for paths in safety_pass_groups
    ):
        yellow_reasons.append("safety_compatibility_evidence_incomplete")

    normalized_offline = _full_offline_evaluation(
        offline if isinstance(offline, Mapping) else None
    )
    if isinstance(normalized_offline, Mapping) and not information_gate_red:
        if tuple(normalized_offline.get("expected_seeds", ())) != TRAINING_SEEDS:
            yellow_reasons.append("offline_three_seed_reducer_incomplete")
        per_seed = normalized_offline.get("per_seed")
        if not isinstance(per_seed, list) or {
            row.get("seed") for row in per_seed if isinstance(row, Mapping)
        } != set(TRAINING_SEEDS):
            yellow_reasons.append("offline_per_seed_results_incomplete")
    if isinstance(closed, Mapping) and not information_gate_red:
        methods = closed.get("methods")
        required_methods = {
            "default_h1",
            "dual_budget_default_h1",
            "rule_guided_h1",
            "metrics_mlp_meta_h1",
            "oracle_guided_h1",
            "random_or_frequency_meta_h1",
            *(f"qwen_guided_h1_seed_{seed}" for seed in TRAINING_SEEDS),
        }
        if not isinstance(methods, Mapping) or not required_methods.issubset(methods):
            yellow_reasons.append("closed_loop_controller_comparison_incomplete")
        elif any(
            methods[name].get("method_input_complete") is not True
            for name in required_methods
            if isinstance(methods.get(name), Mapping)
        ):
            yellow_reasons.append("closed_loop_method_inputs_incomplete")

    gain = _qwen_candidate_gain(closed if isinstance(closed, Mapping) else None)
    reasoning = _ablation_reasoning_support(
        ablation if isinstance(ablation, Mapping) else None,
        closed_loop_gain_supported=gain,
        regression=regression if isinstance(regression, Mapping) else None,
    )
    if not gain and not information_gate_red:
        yellow_reasons.append("stable_qwen_closed_loop_gain_not_confirmed")
    if not reasoning and not information_gate_red:
        yellow_reasons.append("candidate_reasoning_criteria_not_all_met")

    downstream_status = {
        "training": (
            "not_run_due_to_information_sufficiency_gate"
            if information_gate_red and not training_runs
            else "evidence_supplied"
            if training_runs
            else "not_supplied"
        ),
        "offline": (
            "not_run_due_to_information_sufficiency_gate"
            if information_gate_red and (offline is None or _declared_not_run(offline))
            else "evidence_supplied"
            if offline is not None
            else "not_supplied"
        ),
        "closed_loop": (
            "not_run_due_to_information_sufficiency_gate"
            if information_gate_red and (closed is None or _declared_not_run(closed))
            else "evidence_supplied"
            if closed is not None
            else "not_supplied"
        ),
        "ablation": (
            "not_run_due_to_information_sufficiency_gate"
            if information_gate_red and (ablation is None or _declared_not_run(ablation))
            else "evidence_supplied"
            if ablation is not None
            else "not_supplied"
        ),
    }
    if information_gate_red and any(
        status == "evidence_supplied" for status in downstream_status.values()
    ):
        red_reasons.append("downstream_execution_evidence_present_after_red_information_gate")
    if information_gate_red and isinstance(execution, Mapping):
        stopped = execution.get("stopped_operations")
        expected_stops = (
            "qwen_training",
            "dev_generation",
            "offline_performance_evaluation",
            "candidate_closed_loop_evaluation",
            "reasoning_ablations",
        )
        if not isinstance(stopped, Mapping) or any(
            stopped.get(name) != "not_run_due_to_information_sufficiency_gate"
            for name in expected_stops
        ):
            red_reasons.append("execution_evidence_does_not_confirm_required_red_gate_stop")

    red_reasons = sorted(set(red_reasons))
    yellow_reasons = sorted(set(yellow_reasons))
    status = "RED" if red_reasons else "YELLOW" if yellow_reasons else "GO"
    return {
        "status": status,
        "red_reasons": red_reasons,
        "yellow_reasons": yellow_reasons,
        "qwen_closed_loop_gain_supported": gain,
        "candidate_reasoning_supported": reasoning,
        "downstream_status": downstream_status,
        "scientific_conclusion": False,
    }


def _fmt(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(item, digits=digits) for item in value) or "none"
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return str(value)


def _md_cell(value: Any) -> str:
    return _fmt(value).replace("|", "\\|").replace("\n", " ")


def _json_literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _training_offline_by_seed(offline: Mapping[str, Any] | None) -> dict[int, Mapping[str, Any]]:
    offline = _full_offline_evaluation(offline)
    if not isinstance(offline, Mapping) or not isinstance(offline.get("per_seed"), list):
        return {}
    return {
        int(row["seed"]): row
        for row in offline["per_seed"]
        if isinstance(row, Mapping)
        and isinstance(row.get("seed"), int)
        and not isinstance(row.get("seed"), bool)
    }


def _full_offline_evaluation(offline: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(offline, Mapping):
        return None
    nested = offline.get("full_input_offline_evaluation")
    return nested if isinstance(nested, Mapping) else offline


def _render_training_report(
    *, protocol: Mapping[str, Any], evidence: Mapping[str, Any], training_runs: Sequence[Mapping[str, Any]], verdict: Mapping[str, Any]
) -> str:
    offline = evidence.get("offline")
    execution = evidence.get("execution")
    normalized_offline = _full_offline_evaluation(
        offline if isinstance(offline, Mapping) else None
    )
    offline_by_seed = _training_offline_by_seed(offline)
    run_by_seed = {int(run["training"]["seed"]): run for run in training_runs}
    lines = [
        "# Qwen-H1 meta-controller candidate training report",
        "",
        STATUS_LINE,
        "",
        f"Reporting gate: **{verdict['status']}**. This is engineering evidence from candidate train/dev only, not a scientific or frozen-evaluation conclusion.",
        f"Downstream execution state: `{_fmt(verdict['downstream_status'])}`.",
        "",
        "## Preregistered training contract",
        "",
        f"- Adapter: `{protocol['training']['adapter_name']}` (independent from the anomaly-supervisor adapter).",
        f"- Seeds: `{', '.join(str(seed) for seed in TRAINING_SEEDS)}`.",
        f"- Steps/eval/save: `{protocol['training']['max_steps']}/{protocol['training']['eval_steps']}/{protocol['training']['save_steps']}`.",
        f"- Batch/gradient accumulation: `{protocol['training']['batch_size']}/{protocol['training']['gradient_accumulation_steps']}`.",
        f"- QLoRA: rank `{protocol['training']['lora_rank']}`, alpha `{protocol['training']['lora_alpha']}`, dropout `{protocol['training']['lora_dropout']}`, {protocol['training']['quantization']}, gradient checkpointing `{protocol['training']['gradient_checkpointing']}`.",
        f"- Checkpoint selection: {protocol['training']['checkpoint_selection']}.",
        "",
        "## Per-seed results",
        "",
        "| Seed | Run | Steps | Train loss | Best dev loss | Final dev loss | Valid JSON | Decision macro-F1 | Config exact | Direction acc/F1 | Compiled valid | Wall s | Peak allocated/reserved bytes | Best checkpoint |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in TRAINING_SEEDS:
        run = run_by_seed.get(seed, {})
        off = offline_by_seed.get(seed, {})
        training_result = run.get("training_result", {}) if isinstance(run, Mapping) else {}
        metrics = training_result.get("metrics", {}) if isinstance(training_result, Mapping) else {}
        gpu = run.get("gpu", {}) if isinstance(run, Mapping) else {}
        training_meta = off.get("training_run", {}) if isinstance(off, Mapping) else {}
        direction_pair = (
            f"{_fmt(off.get('direction_accuracy'))}/{_fmt(off.get('direction_macro_f1'))}"
            if off
            else "not available"
        )
        peak = (
            f"{_fmt(gpu.get('peak_allocated_bytes'))}/{_fmt(gpu.get('peak_reserved_bytes'))}"
            if gpu
            else "not available"
        )
        lines.append(
            "| "
            + " | ".join(
                _md_cell(value)
                for value in (
                    seed,
                    run.get("status") if run else None,
                    training_result.get("global_step") if training_result else None,
                    metrics.get("train_loss") if metrics else None,
                    training_meta.get("best_dev_loss") if training_meta else None,
                    training_meta.get("final_dev_loss") if training_meta else None,
                    off.get("valid_json_rate") if off else None,
                    off.get("decision_macro_f1") if off else None,
                    off.get("full_configuration_exact_match") if off else None,
                    direction_pair,
                    off.get("compiled_guidance_validity_rate") if off else None,
                    training_result.get("runtime_wall_seconds")
                    if training_result
                    else training_meta.get("wall_time_seconds")
                    if training_meta
                    else None,
                    peak,
                    training_result.get("best_dev_checkpoint") if training_result else None,
                )
            )
            + " |"
        )
    data = evidence.get("data_audit")
    aggregate = (
        normalized_offline.get("aggregate", {})
        if isinstance(normalized_offline, Mapping)
        else {}
    )
    lines.extend(
        [
            "",
            "## Data and stability audit",
            "",
            f"- Information gate: `{_fmt(_lookup(data, 'gate'))}`; audited records: `{_fmt(_lookup(data, 'record_count'))}`; export permitted: `{_fmt(_lookup(data, 'sft_export_permitted'))}`.",
            f"- Exact visible-label conflicts: `{_fmt(_lookup(data, 'exact_visible_collision.conflicting_collision_group_count'))}`; rounded conflicts: `{_fmt(_lookup(data, 'rounded_4dp_visible_collision.conflicting_collision_group_count'))}`.",
            f"- Near-visible conflicting-pair rate: `{_fmt(_lookup(data, 'near_visible_collision.conflicting_near_pair_rate'))}` (maximum `{_fmt(_lookup(data, 'thresholds.near_conflict_rate_max'))}`); visible-only grouped-classifier macro-F1: `{_fmt(_lookup(data, 'visible_only_classifier.macro_f1'))}` (minimum `{_fmt(_lookup(data, 'thresholds.visible_classifier_macro_f1_min'))}`).",
            f"- Known identity overlap: `{_fmt(_lookup(data, 'identity_overlap.known_overlap_count'))}`; candidate cross-split overlap: `{_fmt(_lookup(data, 'identity_overlap.cross_split_overlap_count'))}`.",
            f"- Data generation exit/elapsed/max RSS: `{_fmt(_lookup(execution, 'data_generation.exit_status'))}` / `{_fmt(_lookup(execution, 'data_generation.pipeline_elapsed_seconds'))}` seconds / `{_fmt(_lookup(execution, 'data_generation.maximum_resident_set_kib'))}` KiB.",
            f"- Three-seed valid-JSON mean/std: `{_fmt(_lookup(aggregate, 'valid_json_rate.mean'))}` / `{_fmt(_lookup(aggregate, 'valid_json_rate.std_population'))}`.",
            f"- Three-seed configuration-regret mean/std: `{_fmt(_lookup(aggregate, 'configuration_regret_mean.mean'))}` / `{_fmt(_lookup(aggregate, 'configuration_regret_mean.std_population'))}`.",
            "",
            "Offline JSON and field accuracy are diagnostics only. They are not closed-loop success and do not establish frozen generalization.",
            "When the information-sufficiency gate is RED, absent training/offline/closed-loop/ablation artifacts mean `not_run_due_to_information_sufficiency_gate`, not an unreported run.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _regression_checks(regression: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(regression, Mapping):
        return []
    checks = regression.get("checks", regression.get("test_runs", []))
    if isinstance(checks, Mapping):
        return [
            {"name": name, **(dict(value) if isinstance(value, Mapping) else {"status": value})}
            for name, value in sorted(checks.items())
        ]
    return [value for value in checks if isinstance(value, Mapping)] if isinstance(checks, list) else []


def _render_regression_report(
    *, evidence: Mapping[str, Any], verdict: Mapping[str, Any]
) -> str:
    regression = evidence.get("regression")
    lines = [
        "# Regression and safety report",
        "",
        STATUS_LINE,
        "",
        f"Reporting gate: **{verdict['status']}**.",
        "",
        "## Hard-stop assessment",
        "",
        f"- RED reasons: `{_fmt(verdict['red_reasons'])}`.",
        f"- YELLOW reasons: `{_fmt(verdict['yellow_reasons'])}`.",
        "- H3 remains disabled and forbidden; H1 is the only continuous controller.",
        "- Qwen may only emit strict discrete configuration JSON. It cannot emit or dispatch continuous actuator actions, enlarge bounds, change the target, or bypass uncertainty/budget/dispatch checks.",
        "- Safety in the existing repository is distributed across contracts, bounds/projection, uncertainty handling, budget/state-machine logic and simulator validity. The candidate adds a reject-only pre-dispatch gate; this report does not invent an unchanged standalone SafetyGate that did not exist.",
        "- Formal frozen evaluation remains disabled. No candidate report is evidence about frozen IID/OOD/protected generalization or real hardware.",
        "- The candidate closed-loop harness consumes manifest/synthetic supervisor-validity state; it does not execute the existing Qwen anomaly supervisor. Sequential reobserve recovery has no verified repository backend. Therefore these artifacts are not full-stack or end-to-end validation.",
        "",
        "## Supplied test evidence",
        "",
    ]
    checks = _regression_checks(regression if isinstance(regression, Mapping) else None)
    if checks:
        lines.extend(
            [
                "| Check | Status | Passed | Failed | Notes |",
                "|---|---|---:|---:|---|",
            ]
        )
        for check in checks:
            lines.append(
                "| "
                + " | ".join(
                    _md_cell(value)
                    for value in (
                        check.get("name", "unnamed"),
                        check.get("status"),
                        check.get("passed", check.get("passed_count")),
                        check.get("failed", check.get("failed_count")),
                        check.get("notes", check.get("summary")),
                    )
                )
                + " |"
            )
    else:
        lines.append("No structured regression test evidence was supplied; compatibility remains unverified in this aggregate.")
    flags = (
        ("Feature flag off reproduces unchanged default H1", "safety.feature_flag_off_equivalent", "off_equivalence"),
        ("Compiler preserves or shrinks bounds", "safety.compiler_bounds_preserved", "compiler_bounds_preserved"),
        ("Continuous action/mu/sigma/bounds injection rejected", "safety.continuous_action_injection_rejected", "continuous_action_injection_rejected"),
        ("Dispatch gate enforced", "safety.dispatch_gate_enforced", "dispatch_gate_enforced"),
        ("Existing state-machine tests pass", "safety.existing_state_machine_tests_passed", "existing_state_machine_tests_passed"),
        ("Canonical objective and strict-success semantics preserved", "safety.canonical_objective_and_strict_success_preserved", "canonical_objective_and_strict_success_preserved"),
        ("Frozen/protected content untouched", "safety.frozen_or_protected_content_untouched", "frozen_or_protected_content_untouched"),
        ("H3 disabled", "safety.h3_disabled", "h3_disabled"),
    )
    lines.extend(["", "## Compatibility flags", ""])
    for label, *paths in flags:
        lines.append(f"- {label}: `{_fmt(_lookup(regression, *paths))}`.")
    preexisting = _lookup(regression, "known_preexisting_failures")
    lines.extend(
        [
            f"- Known pre-existing failures (not silently waived): `{_fmt(preexisting)}`.",
            "",
            "A failed hard-stop check is RED. Missing evidence is YELLOW; it is never silently treated as a pass.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _render_commands(
    *, commands: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]
) -> str:
    lines = [
        "# Candidate command ledger",
        "",
        STATUS_LINE,
        "",
        "Only commands explicitly supplied in the candidate command ledger are reproduced here. This renderer does not infer shell history and executes none of these commands.",
        "",
    ]
    if not commands:
        lines.append("No actual command ledger was supplied.")
    for index, entry in enumerate(commands, start=1):
        lines.extend(
            [
                f"## {index}. {_fmt(entry.get('purpose', 'Recorded command'))}",
                "",
                "```bash",
                str(entry["command"]),
                "```",
                "",
                f"- Status/exit: `{_fmt(entry.get('status', entry.get('exit_code')))}`.",
                f"- Wall time seconds: `{_fmt(entry.get('wall_time_seconds'))}`.",
                f"- Peak memory bytes: `{_fmt(entry.get('peak_memory_bytes'))}`.",
                f"- Notes: {_fmt(entry.get('notes'))}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Evidence ledger",
            "",
            "| Role | Candidate-relative path | SHA-256 |",
            "|---|---|---|",
        ]
    )
    for role, item in sorted(provenance.items()):
        lines.append(
            f"| {_md_cell(role)} | `{_md_cell(item.get('path'))}` | `{_md_cell(item.get('sha256'))}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _method_summary_rows(closed: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(closed, Mapping) or not isinstance(closed.get("methods"), Mapping):
        return ["| not available | no | not available | not available | not available | not available |"]
    rows: list[str] = []
    for name, method in sorted(closed["methods"].items()):
        if not isinstance(method, Mapping):
            continue
        performance = method.get("performance", {})
        audit = method.get("audit", {})
        values = (
            name,
            method.get("method_input_complete"),
            _lookup(performance, "strict_all_five_success"),
            _lookup(performance, "final_normalized_error"),
            _lookup(audit, "configuration_regret"),
            _lookup(audit, "wall_clock_compute_seconds"),
        )
        normalized = [
            value.get("value") if isinstance(value, Mapping) and "value" in value else value
            for value in values
        ]
        rows.append("| " + " | ".join(_md_cell(value) for value in normalized) + " |")
    return rows or ["| not available | no | not available | not available | not available | not available |"]


def _render_final_report(
    *, protocol: Mapping[str, Any], evidence: Mapping[str, Any], training_runs: Sequence[Mapping[str, Any]], commands: Sequence[Mapping[str, Any]], verdict: Mapping[str, Any], machine: Mapping[str, Any]
) -> str:
    data = evidence.get("data_audit")
    offline = evidence.get("offline")
    normalized_offline = _full_offline_evaluation(
        offline if isinstance(offline, Mapping) else None
    )
    closed = evidence.get("closed_loop")
    ablation = evidence.get("ablation")
    regression = evidence.get("regression")
    execution = evidence.get("execution")
    methods = _method_summary_rows(closed if isinstance(closed, Mapping) else None)
    total_training_wall = sum(
        float(_lookup(run, "training_result.runtime_wall_seconds") or 0.0)
        for run in training_runs
    )
    peaks = [
        int(value)
        for run in training_runs
        for value in (_lookup(run, "gpu.peak_allocated_bytes"),)
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    recorded_training_peak_mib = (
        max(peaks) / (1024 * 1024)
        if peaks
        else _lookup(execution, "hardware_snapshot_after_generation.gpu_peak_for_training_mib")
    )
    changes = _lookup(regression, "repository_changes")
    checkpoint_statement = (
        "No QLoRA checkpoint was created because training was not run after the RED data gate."
        if verdict["downstream_status"]["training"]
        == "not_run_due_to_information_sufficiency_gate"
        else "Training config and checkpoint paths are recorded in supplied run manifests."
    )
    if verdict["status"] == "RED":
        claim = (
            "The candidate implementation reached its preregistered information audit, "
            "which triggered a hard stop. Qwen training, inference, closed-loop evaluation "
            "and reasoning ablations were not run, so no Qwen reasoning or control-performance "
            "claim is supported."
        )
    elif verdict["candidate_reasoning_supported"] and verdict["qwen_closed_loop_gain_supported"]:
        claim = (
            "Candidate-only paired evidence and preregistered ablations support a bounded "
            "optical-control reasoning claim."
        )
    else:
        claim = (
            "Qwen optical meta-controller has been integrated as an optional constrained H1 "
            "configuration layer with fallback to unchanged default H1, but a stable closed-loop "
            "advantage over rule, MLP and default controls is not confirmed."
        )
    lines = [
        "# Qwen-H1 meta-controller candidate final report",
        "",
        STATUS_LINE,
        "",
        "## 1. Gate",
        "",
        f"**{verdict['status']}**. RED reasons: `{_fmt(verdict['red_reasons'])}`. YELLOW reasons: `{_fmt(verdict['yellow_reasons'])}`.",
        "",
        "## 2. Files changed and added",
        "",
        f"Structured repository-change evidence: `{_fmt(changes)}`. All candidate outputs are confined to `qwen_h1_meta_v0_candidate`; this aggregate does not claim that an unspecified existing file was unchanged.",
        "",
        "## 3. Preservation of the existing structure",
        "",
        f"Existing-structure preservation evidence: `{_fmt(_lookup(regression, 'existing_structure_preserved'))}`. The intended additive path keeps the anomaly supervisor, learned forward ensemble, default Learned-H1 and state-machine authority separate.",
        "",
        "## 4. Feature-flag-off equivalence",
        "",
        f"Off-mode equivalence: `{_fmt(_lookup(regression, 'safety.feature_flag_off_equivalent', 'off_equivalence'))}`. Missing evidence is not a pass.",
        "",
        "## 5. Qwen schema and authority boundary",
        "",
        "Qwen emits only the frozen discrete JSON schema: decision, observation request, objective profile, allowlisted mask, per-canonical-actuator direction categories, step scale, risk mode, confidence and reason codes. Qwen never owns continuous actuator values, legal bounds, targets, safety decisions or dispatch.",
        "",
        "## 6. Deterministic compiler",
        "",
        f"Compiler mapping: `{protocol['contracts']['compiler_mapping']}`. Balanced/default identity, trust-region shrinkage, fixed masks/directions, confidence fallback and H1-only dispatch remain deterministic and hash-pinned.",
        "",
        "## 7. Candidate data",
        "",
        f"Audited records: `{_fmt(_lookup(data, 'record_count'))}`; preregistered setups are train/dev/eval `{_fmt(_lookup(protocol, 'data.train_setups'))}/{_fmt(_lookup(protocol, 'data.dev_setups'))}/{_fmt(_lookup(protocol, 'data.candidate_eval_setups'))}`, with three target counterfactuals per setup. Known overlap: `{_fmt(_lookup(data, 'identity_overlap.known_overlap_count'))}`; cross-split overlap: `{_fmt(_lookup(data, 'identity_overlap.cross_split_overlap_count'))}`.",
        "",
        "## 8. Information sufficiency",
        "",
        f"Gate: `{_fmt(_lookup(data, 'gate'))}`. Exact/rounded conflicting collision groups: `{_fmt(_lookup(data, 'exact_visible_collision.conflicting_collision_group_count'))}` / `{_fmt(_lookup(data, 'rounded_4dp_visible_collision.conflicting_collision_group_count'))}`. Near-visible conflicts: `{_fmt(_lookup(data, 'near_visible_collision.conflicting_near_pair_count'))}/{_fmt(_lookup(data, 'near_visible_collision.near_pair_count'))} = {_fmt(_lookup(data, 'near_visible_collision.conflicting_near_pair_rate'), digits=6)}` versus maximum `{_fmt(_lookup(data, 'thresholds.near_conflict_rate_max'), digits=6)}`. Visible-only grouped classifier macro-F1: `{_fmt(_lookup(data, 'visible_only_classifier.macro_f1'), digits=6)}` versus minimum `{_fmt(_lookup(data, 'thresholds.visible_classifier_macro_f1_min'), digits=6)}`. Hidden-setup macro-F1 gain: `{_fmt(_lookup(data, 'hidden_setup_macro_f1_gain'), digits=6)}`. Known/cross-split identity overlap: `{_fmt(_lookup(data, 'identity_overlap.known_overlap_count'))}` / `{_fmt(_lookup(data, 'identity_overlap.cross_split_overlap_count'))}`.",
        "",
        "## 9. Three-seed training stability",
        "",
        f"Completed preregistered seeds: `{_fmt([run.get('training', {}).get('seed') for run in training_runs if run.get('status') == 'completed'])}`. Offline valid-JSON mean/std: `{_fmt(_lookup(normalized_offline, 'aggregate.valid_json_rate.mean'))}` / `{_fmt(_lookup(normalized_offline, 'aggregate.valid_json_rate.std_population'))}`. Configuration-regret mean/std: `{_fmt(_lookup(normalized_offline, 'aggregate.configuration_regret_mean.mean'))}` / `{_fmt(_lookup(normalized_offline, 'aggregate.configuration_regret_mean.std_population'))}`.",
        f"Execution state: `{_fmt(verdict['downstream_status']['training'])}` / `{_fmt(verdict['downstream_status']['offline'])}`. A RED information-sufficiency gate forbids starting QLoRA or downstream evaluation.",
        "",
        "## 10. Controller comparisons",
        "",
        "| Method | Complete | Strict success | Final normalized error | Config regret | Wall-clock s |",
        "|---|---|---:|---:|---:|---:|",
        *methods,
        "",
        "Default H1 is the original compute budget. Dual-budget default is the required compute-fair control. If Qwen only beats original default and not dual-budget default, any gain may be extra search computation rather than Qwen reasoning.",
        "",
        "## 11. Paired closed-loop evidence",
        "",
        f"Episodes/setups: `{_fmt(_lookup(closed, 'manifest_record_count'))}` / `{_fmt(_lookup(closed, 'setup_count'))}`. Qwen stable candidate gain supported: `{_fmt(verdict['qwen_closed_loop_gain_supported'])}`. Paired bootstrap evidence is preserved verbatim in `closed_loop_results.json`; no missing Qwen trace is relabeled as Qwen performance.",
        f"Closed-loop execution state: `{_fmt(verdict['downstream_status']['closed_loop'])}`.",
        "The closed-loop harness uses candidate-manifest/synthetic measurement-validity and supervisor state; it does not run the actual existing Qwen anomaly-supervisor inference path. The sequential reobserve recovery backend is unverified. This is neither full-stack nor end-to-end validation.",
        "",
        "## 12. Reasoning ablations",
        "",
        f"Candidate reasoning criteria all supported: `{_fmt(verdict['candidate_reasoning_supported'])}`. Ablation seed count: `{_fmt(len(ablation.get('per_seed', [])) if isinstance(ablation, Mapping) and isinstance(ablation.get('per_seed'), list) else None)}`. Required target, history, image, metrics, actuator-semantics, uncertainty and reason-code ablations remain candidate simulator/shadow evidence only.",
        f"Ablation execution state: `{_fmt(verdict['downstream_status']['ablation'])}`.",
        "",
        "## 13. Safety, fallback and state machine",
        "",
        f"Regression status: `{_fmt(_lookup(regression, 'status'))}`; dispatch gate: `{_fmt(_lookup(regression, 'safety.dispatch_gate_enforced', 'dispatch_gate_enforced'))}`; state-machine tests: `{_fmt(_lookup(regression, 'safety.existing_state_machine_tests_passed', 'existing_state_machine_tests_passed'))}`; H3 disabled: `{_fmt(_lookup(regression, 'safety.h3_disabled', 'h3_disabled'))}`. Reobserve/stop are non-dispatch outcomes, anomaly recovery has priority, and fallback still passes the existing bounds/budget/validity path.",
        "",
        "## 14. Remaining blockers",
        "",
        f"RED: `{_fmt(verdict['red_reasons'])}`. YELLOW/negative: `{_fmt(verdict['yellow_reasons'])}`.",
        "Old-source partial generation files, if present, remain isolated under `artifacts/interrupted_generation_old_import_20260802T143042` and are not treated as legal current candidate data or downstream evidence.",
        "",
        "## 15. Claim boundary",
        "",
        claim,
        "",
        "This is not evidence that Qwen directly controls actuators, learned complete optical physics, replaces the forward model, converts offline accuracy into closed-loop success, generalizes to frozen data, validates temporal recovery, validates a full stack/end-to-end system or hardware, or enables H3.",
        "",
        "The anomaly-supervisor dev diagnosis result (96.91% ± 1.67 pp) and the separate Learned-H1 candidate result (37/48, 77.1%) come from different experiments. They must not be multiplied and are not an end-to-end system success rate.",
        "",
        "## 16. Commands, configs, checkpoints, manifests and hashes",
        "",
        f"Recorded commands: `{len(commands)}`. See `commands.md`. {checkpoint_statement} Protocol/config/manifest identities remain hash-pinned. Candidate-relative evidence SHA-256 values are in `machine_summary.json`; the complete deterministic inventory is in `artifact_hashes.json`, whose explicit rule excludes the hash manifest itself.",
        "",
        "## 17. Runtime and memory",
        "",
        f"Data generation/audit pipeline elapsed: `{_fmt(_lookup(execution, 'data_generation.pipeline_elapsed_seconds'))}` seconds (`/usr/bin/time` wall `{_fmt(_lookup(execution, 'data_generation.time_wall_clock'))}`), maximum resident set `{_fmt(_lookup(execution, 'data_generation.maximum_resident_set_kib'))}` KiB. Training wall time across supplied completed manifests: `{_fmt(total_training_wall if training_runs else None)}` seconds. GPU training peak MiB (recorded JSON): `{_json_literal(recorded_training_peak_mib)}`; QLoRA/inference start state is explained by `{_fmt(_lookup(execution, 'hardware_snapshot_after_generation.gpu_peak_missing_reason'))}`. Closed-loop wall time: `{_fmt(_lookup(closed, 'wall_clock_seconds'))}` seconds. Recorded GPU identity: `{_fmt(_lookup(execution, 'hardware_snapshot_after_generation.gpu'))}`; reporter machine query: `{_fmt(_lookup(machine, 'gpu.devices'))}`.",
        "",
        "Formal frozen evaluation was not run by this reporter, the protocol remains unsealed, no scientific conclusion is made, and no commit or push is performed.",
        "",
        TERMINAL_LINE,
    ]
    return "\n".join(lines).rstrip() + "\n"


def _skip_hash_relative(relative: Path) -> bool:
    if relative.as_posix() == HASH_MANIFEST_FILENAME:
        return True
    if any(part in HASH_EXCLUDED_DIR_NAMES for part in relative.parts[:-1]):
        return True
    if any(part.startswith("interrupted_") for part in relative.parts[:-1]):
        return True
    if relative.suffix.lower() in HASH_EXCLUDED_FILE_SUFFIXES:
        return True
    return any(_path_tokens(part) & FORBIDDEN_ARTIFACT_TOKENS for part in relative.parts)


def build_hash_manifest(output_root: Path) -> dict[str, Any]:
    """Hash every stable regular file under output_root except this manifest."""

    root = output_root.resolve()
    files: dict[str, dict[str, Any]] = {}
    excluded_symlinks: list[str] = []
    excluded_policy_paths: list[str] = []
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        kept_names: list[str] = []
        for name in sorted(names):
            child = directory_path / name
            relative = child.relative_to(root)
            if child.is_symlink():
                excluded_symlinks.append(relative.as_posix())
            elif _skip_hash_relative(relative / "placeholder"):
                excluded_policy_paths.append(relative.as_posix() + "/")
            else:
                kept_names.append(name)
        names[:] = kept_names
        for filename in sorted(filenames):
            path = directory_path / filename
            relative = path.relative_to(root)
            if relative.as_posix() == HASH_MANIFEST_FILENAME:
                # The explicit self-hash rule is stable whether this is the
                # first build or a rebuild, so do not add it to dynamic skips.
                continue
            if path.is_symlink():
                excluded_symlinks.append(relative.as_posix())
                continue
            if _skip_hash_relative(relative):
                excluded_policy_paths.append(relative.as_posix())
                continue
            if not path.is_file():
                excluded_policy_paths.append(relative.as_posix())
                continue
            files[relative.as_posix()] = {
                "sha256": _sha256_path(path),
                "bytes": path.stat().st_size,
            }
    return {
        "schema_version": "qwen_h1_meta_v0_artifact_hashes_v1",
        "status": STATUS_LINE,
        "candidate_only": True,
        "formal_frozen_evaluation_enabled": False,
        "root": ".",
        "algorithm": "sha256",
        "self_hash_rule": (
            "artifact_hashes.json is excluded because a cryptographic manifest cannot "
            "contain its own stable digest"
        ),
        "inventory_policy": {
            "include": "all stable regular files recursively under root",
            "exclude_filename": HASH_MANIFEST_FILENAME,
            "exclude_directory_names": sorted(HASH_EXCLUDED_DIR_NAMES),
            "exclude_directory_prefixes": ["interrupted_"],
            "exclude_file_suffixes": sorted(HASH_EXCLUDED_FILE_SUFFIXES),
            "exclude_forbidden_split_tokens_without_opening": sorted(
                FORBIDDEN_EVIDENCE_TOKENS - {"test", "tests"}
            ),
            "symlinks": "excluded without opening targets",
        },
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "files": dict(sorted(files.items())),
        "excluded_symlinks": sorted(excluded_symlinks),
        "excluded_policy_paths": sorted(excluded_policy_paths),
    }


def build_reports(
    sources: ReportSources,
    *,
    output_root: Path = PACKAGE_ROOT,
    namespace_root: Path = PACKAGE_ROOT,
) -> dict[str, Any]:
    """Build all final aggregate artifacts with deterministic serialization."""

    output = output_root.expanduser().resolve()
    namespace = namespace_root.expanduser().resolve()
    try:
        output.relative_to(namespace)
    except ValueError as exc:
        raise ReportingError("report output root must remain inside candidate namespace") from exc
    if _contains_forbidden_marker(output.relative_to(namespace)):
        raise ReportingError("report output root contains a forbidden split marker")
    output.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(namespace)
    evidence, provenance, commands, training_runs = _validate_inputs(
        sources, namespace_root=namespace
    )
    verdict = compute_verdict(
        evidence=evidence, training_runs=training_runs, commands=commands
    )
    machine = collect_machine_summary(provenance=provenance, verdict=verdict)
    execution = evidence.get("execution")
    if isinstance(execution, Mapping):
        machine["recorded_execution_environment"] = {
            "hardware_snapshot_after_generation": execution.get(
                "hardware_snapshot_after_generation"
            ),
            "environments": execution.get("environments"),
            "stopped_operations": execution.get("stopped_operations"),
        }
    _atomic_write_text(
        output / "training_report.md",
        _render_training_report(
            protocol=protocol,
            evidence=evidence,
            training_runs=training_runs,
            verdict=verdict,
        ),
    )
    _atomic_write_text(
        output / "regression_and_safety_report.md",
        _render_regression_report(evidence=evidence, verdict=verdict),
    )
    _atomic_write_text(
        output / "commands.md",
        _render_commands(commands=commands, provenance=provenance),
    )
    _atomic_write_json(output / "machine_summary.json", machine)
    _atomic_write_text(
        output / "final_report.md",
        _render_final_report(
            protocol=protocol,
            evidence=evidence,
            training_runs=training_runs,
            commands=commands,
            verdict=verdict,
            machine=machine,
        ),
    )
    hash_manifest = build_hash_manifest(output)
    _atomic_write_json(output / HASH_MANIFEST_FILENAME, hash_manifest)
    return {
        "status": verdict["status"],
        "output_root": str(output),
        "files": [str(output / name) for name in REPORT_FILENAMES],
        "hash_manifest_file_count": hash_manifest["file_count"],
    }


def validate_generated_artifacts(
    output_root: Path, *, namespace_root: Path = PACKAGE_ROOT
) -> dict[str, Any]:
    """Validate report presence, fixed boundaries and the complete hash inventory."""

    root = output_root.expanduser().resolve()
    namespace = namespace_root.expanduser().resolve()
    try:
        root.relative_to(namespace)
    except ValueError as exc:
        raise ArtifactValidationError(
            "report root must remain inside candidate namespace"
        ) from exc
    for name in REPORT_FILENAMES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ArtifactValidationError(f"missing regular report artifact: {path}")
    final_text = (root / "final_report.md").read_text(encoding="utf-8")
    if final_text.rstrip().splitlines()[-1] != TERMINAL_LINE:
        raise ArtifactValidationError("final_report.md has an invalid terminal line")
    required_warnings = (
        STATUS_LINE,
        "96.91% ± 1.67 pp",
        "37/48, 77.1%",
        "must not be multiplied",
        "H3",
        "Formal frozen evaluation",
    )
    for warning in required_warnings:
        if warning not in final_text:
            raise ArtifactValidationError(
                f"final_report.md is missing required boundary text: {warning}"
            )
    machine = json.loads((root / "machine_summary.json").read_text(encoding="utf-8"))
    if machine.get("status") != STATUS_LINE or machine.get(
        "formal_frozen_evaluation_enabled"
    ) is not False:
        raise ArtifactValidationError("machine summary candidate boundary is invalid")
    expected = json.loads((root / HASH_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if expected.get("schema_version") != "qwen_h1_meta_v0_artifact_hashes_v1":
        raise ArtifactValidationError("unsupported artifact hash manifest schema")
    if HASH_MANIFEST_FILENAME in expected.get("files", {}):
        raise ArtifactValidationError("artifact hash manifest illegally includes itself")
    actual = build_hash_manifest(root)
    keys = ("file_count", "total_bytes", "files", "excluded_symlinks", "excluded_policy_paths")
    mismatches = [key for key in keys if actual.get(key) != expected.get(key)]
    if mismatches:
        raise ArtifactValidationError(
            "artifact inventory or digest mismatch: " + ", ".join(mismatches)
        )
    return {
        "status": "PASS",
        "candidate_only": True,
        "output_root": str(root),
        "file_count": actual["file_count"],
        "self_hash_excluded": True,
    }


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-audit", type=Path)
    parser.add_argument("--training-run", type=Path, action="append", default=[])
    parser.add_argument("--offline", type=Path)
    parser.add_argument("--closed-loop", type=Path)
    parser.add_argument("--ablation", type=Path)
    parser.add_argument("--regression", type=Path)
    parser.add_argument("--execution-evidence", type=Path)
    parser.add_argument("--command-log", type=Path)
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    build = subparsers.add_parser("build", help="render reports and artifact hashes")
    _add_source_arguments(build)
    validate = subparsers.add_parser("validate", help="validate generated reports/hashes")
    validate.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.operation == "build":
        result = build_reports(
            ReportSources(
                data_audit=args.data_audit,
                training_runs=tuple(args.training_run),
                offline=args.offline,
                closed_loop=args.closed_loop,
                ablation=args.ablation,
                regression=args.regression,
                execution_evidence=args.execution_evidence,
                command_log=args.command_log,
            ),
            output_root=args.output_root,
        )
    else:
        result = validate_generated_artifacts(args.output_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArtifactValidationError",
    "ReportSources",
    "ReportingError",
    "STATUS_LINE",
    "TERMINAL_LINE",
    "build_hash_manifest",
    "build_reports",
    "collect_machine_summary",
    "compute_verdict",
    "guard_candidate_path",
    "validate_generated_artifacts",
]
