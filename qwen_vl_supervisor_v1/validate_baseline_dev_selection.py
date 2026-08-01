#!/usr/bin/env python3
"""Fail-closed validator for future dev-only baseline selection evidence.

This utility does not train a baseline or choose thresholds.  It validates a
future artifact produced from legal train/development data, recomputes every A
candidate report from raw predictions, recomputes every B threshold-grid row
from raw specialist probabilities, and verifies that the recorded winners are
the deterministic rank-one choices.  Frozen/protected use cannot be represented
as a successful artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .evaluate_offline import evaluate_records


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "qwen_vl_supervisor_baseline_dev_selection_v1.0.0"
FILE_HASH_ALGORITHM = "sha256_file_v1"
TREE_HASH_ALGORITHM = "sha256_tree_relative_path_nul_file_sha256_lf_v1"
REFERENCE_SEED = "deterministic_reference"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PROTECTED_PATH_TOKEN = re.compile(
    r"(?:^|[_.-])(?:frozen|heldout|held_out|severity_ood|test|tests)(?:$|[_.-])"
)

A_CANDIDATES = (
    "deterministic_frozen_rules",
    "standardized_multinomial_logistic_regression",
    "standardized_one_hidden_layer_mlp",
)
A_COMPLEXITY_RANK = {
    "deterministic_frozen_rules": 0,
    "standardized_multinomial_logistic_regression": 1,
    "standardized_one_hidden_layer_mlp": 2,
}
A_SELECTION_RULE = (
    "highest_joint_diagnosis_policy_action_exact_accuracy",
    "highest_diagnosis_macro_f1",
    "highest_valid_json_rate",
    "lowest_model_complexity",
    "lexicographically_lowest_candidate_name",
)
B_SELECTION_RULE = (
    "highest_joint_diagnosis_policy_action_exact_accuracy",
    "highest_diagnosis_macro_f1",
    "highest_valid_json_rate",
    "lowest_sensor_saturation_threshold",
    "lowest_secondary_reflection_threshold",
)

OUTPUT_MAPPING: dict[str, dict[str, str]] = {
    "nominal": {
        "diagnosis": "nominal",
        "measurement_policy": "standard",
        "supervisor_action": "execute",
    },
    "sensor_saturation": {
        "diagnosis": "sensor_saturation",
        "measurement_policy": "lower_exposure_reacquire",
        "supervisor_action": "reacquire",
    },
    "secondary_reflection": {
        "diagnosis": "secondary_reflection",
        "measurement_policy": "primary_spot",
        "supervisor_action": "switch_measurement",
    },
}

B_ARBITRATION: dict[str, Any] = {
    "score_domain": "calibrated_probability_closed_unit_interval",
    "nominal_rule": "both_scores_strictly_below_their_selected_thresholds",
    "anomaly_rule": "otherwise_choose_larger_normalized_margin_above_threshold",
    "normalized_margin_formula": "(score-threshold)/max(abs(threshold),1e-12)",
    "threshold_comparator_for_anomaly": "greater_than_or_equal",
    "tie_order": ["nominal", "sensor_saturation", "secondary_reflection"],
}

FILE_ENTRY_KEYS = {"path", "hash_algorithm", "sha256"}
DATA_IDENTITY_KEYS = {
    "source_manifest",
    "record_count",
    "sample_id_set_sha256",
}
RANKING_METRIC_KEYS = {
    "joint_exact_accuracy",
    "diagnosis_macro_f1",
    "valid_json_rate",
}
COVERAGE_KEYS = {
    "manifest_records",
    "supplied_predictions",
    "missing_predictions",
    "coverage_rate",
    "expected_seed_enforced",
}
A_CANDIDATE_KEYS = {
    "candidate_name",
    "model_complexity_rank",
    "selection_rank",
    "model_artifact",
    "training_report",
    "predictions",
    "offline_report",
    "dev_coverage",
    "ranking_metrics",
    "eligible",
    "protected_or_frozen_data_used",
    "frozen_predictions_opened",
}
TOP_KEYS = {
    "schema_version",
    "artifact_kind",
    "selection_split",
    "train_data",
    "dev_data",
    "arm_A",
    "arm_B",
    "protected_or_frozen_data_used",
    "frozen_predictions_opened",
    "timestamp_in_artifact",
    "validator",
}


class BaselineSelectionError(ValueError):
    """Raised when baseline selection evidence violates the frozen contract."""


def _exact_keys(value: Any, expected: set[str], location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineSelectionError(f"{location} must be an object")
    actual = set(value)
    if actual != expected:
        raise BaselineSelectionError(
            f"{location} keys differ; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    return value


def _reject_constant(value: str) -> None:
    raise BaselineSelectionError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise BaselineSelectionError(f"duplicate JSON key {key!r}")
        value[key] = child
    return value


def _read_json(path: Path, *, location: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BaselineSelectionError(f"{location} is not a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineSelectionError(f"cannot read strict {location} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineSelectionError(f"{location} must contain one JSON object")
    return value


def _read_jsonl(path: Path, *, location: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise BaselineSelectionError(f"{location} is not a regular file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except (json.JSONDecodeError, BaselineSelectionError) as exc:
                raise BaselineSelectionError(
                    f"invalid {location} JSONL at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise BaselineSelectionError(
                    f"{location}:{line_number} must contain one object"
                )
            rows.append(row)
    if not rows:
        raise BaselineSelectionError(f"{location} is empty: {path}")
    return rows


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise BaselineSelectionError("PyYAML is required for the evaluation config") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise BaselineSelectionError(f"cannot read evaluation config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineSelectionError("evaluation config must contain one object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_path(path: Path) -> tuple[str, str]:
    if not path.exists():
        raise BaselineSelectionError(f"required evidence path does not exist: {path}")
    if path.is_symlink():
        raise BaselineSelectionError(f"symbolic evidence path is forbidden: {path}")
    if path.is_file():
        return FILE_HASH_ALGORITHM, _sha256_file(path)
    if not path.is_dir():
        raise BaselineSelectionError(f"evidence path is neither file nor directory: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file() or item.is_symlink())
    if not files:
        raise BaselineSelectionError(f"evidence tree is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise BaselineSelectionError(f"evidence tree contains a symbolic link: {item}")
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return TREE_HASH_ALGORITHM, digest.hexdigest()


def _resolve_inside(root: Path, raw: str, *, location: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise BaselineSelectionError(f"{location} must be a repository-relative path")
    lexical = root / path
    cursor = root.resolve()
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BaselineSelectionError(f"{location} traverses a symbolic link: {cursor}")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise BaselineSelectionError(f"{location} escapes the repository root") from exc
    return resolved


def _validate_file_entry(
    value: Any,
    *,
    root: Path,
    location: str,
    require_file: bool = False,
    prohibit_protected_path: bool = False,
) -> dict[str, str]:
    entry = _exact_keys(value, FILE_ENTRY_KEYS, location)
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise BaselineSelectionError(f"{location}.path must be a nonempty string")
    path_components = tuple(part.lower() for part in Path(raw_path).parts)
    path_looks_protected = any(
        component != "deterministic_frozen_rules"
        and PROTECTED_PATH_TOKEN.search(component) is not None
        for component in path_components
    )
    if prohibit_protected_path and path_looks_protected:
        raise BaselineSelectionError(f"{location} uses a protected/frozen path: {raw_path}")
    expected = entry.get("sha256")
    if not isinstance(expected, str) or not HEX64.fullmatch(expected):
        raise BaselineSelectionError(f"{location}.sha256 must be lowercase SHA-256")
    path = _resolve_inside(root, raw_path, location=location)
    algorithm, actual = _hash_path(path)
    if require_file and algorithm != FILE_HASH_ALGORITHM:
        raise BaselineSelectionError(f"{location} must identify one regular file")
    if entry.get("hash_algorithm") != algorithm or expected != actual:
        raise BaselineSelectionError(f"{location} byte identity differs for {path}")
    return {"path": raw_path, "hash_algorithm": algorithm, "sha256": actual}


def _config_entry(
    root: Path,
    raw: Any,
    *,
    location: str,
    require_declared_sha256: bool = True,
) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise BaselineSelectionError(f"{location} is missing from the evaluation config")
    path_value = raw.get("path")
    digest = raw.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise BaselineSelectionError(f"{location}.path must be a nonempty string")
    path = _resolve_inside(root, path_value, location=location)
    algorithm, actual = _hash_path(path)
    if digest is None and not require_declared_sha256:
        digest = actual
    if not isinstance(digest, str) or not HEX64.fullmatch(digest):
        raise BaselineSelectionError(f"{location}.sha256 must be lowercase SHA-256")
    if actual != digest:
        raise BaselineSelectionError(f"{location} hash differs from evaluation config")
    return {"path": path_value, "hash_algorithm": algorithm, "sha256": digest}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_manifest_identity(
    value: Any,
    *,
    root: Path,
    configured: Mapping[str, Any],
    expected_split: str,
    location: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    identity = _exact_keys(value, DATA_IDENTITY_KEYS, location)
    # The evaluation config pins manifest roles and paths while the pre-server
    # freeze records their current byte hashes. Recompute that path here; an
    # optional declared config hash, when present, remains mandatory.
    expected_entry = _config_entry(
        root,
        configured,
        location=f"config.manifests.{expected_split}",
        require_declared_sha256=False,
    )
    observed_entry = _validate_file_entry(
        identity["source_manifest"],
        root=root,
        location=f"{location}.source_manifest",
        require_file=True,
    )
    if observed_entry != expected_entry:
        raise BaselineSelectionError(f"{location} does not match the configured manifest")
    rows = _read_jsonl(
        _resolve_inside(root, observed_entry["path"], location=f"{location}.source_manifest"),
        location=f"{location}.source_manifest",
    )
    ids: list[str] = []
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise BaselineSelectionError(f"{location} manifest row {index} has no sample_id")
        if row.get("split") != expected_split:
            raise BaselineSelectionError(
                f"{location} manifest row {sample_id} is not split {expected_split!r}"
            )
        ids.append(sample_id)
    if len(ids) != len(set(ids)):
        raise BaselineSelectionError(f"{location} manifest has duplicate sample IDs")
    if identity.get("record_count") != len(rows):
        raise BaselineSelectionError(f"{location}.record_count differs from manifest")
    if identity.get("sample_id_set_sha256") != _canonical_sha256(sorted(ids)):
        raise BaselineSelectionError(f"{location}.sample_id_set_sha256 differs from manifest")
    return rows, ids


def _unit_float(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineSelectionError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise BaselineSelectionError(f"{location} must be finite in [0,1]")
    return result


def _validate_ranking_metrics(value: Any, *, location: str) -> dict[str, float]:
    metrics = _exact_keys(value, RANKING_METRIC_KEYS, location)
    return {
        name: _unit_float(metrics[name], location=f"{location}.{name}")
        for name in sorted(RANKING_METRIC_KEYS)
    }


def _report_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    per_seed = report.get("per_seed")
    if not isinstance(per_seed, list) or len(per_seed) != 1:
        raise BaselineSelectionError("offline report must contain exactly one reference seed")
    row = per_seed[0]
    if not isinstance(row, Mapping) or row.get("seed") != REFERENCE_SEED:
        raise BaselineSelectionError("offline report seed must be deterministic_reference")
    return {
        "diagnosis_macro_f1": _unit_float(
            row.get("diagnosis_macro_f1"), location="offline diagnosis_macro_f1"
        ),
        "joint_exact_accuracy": _unit_float(
            row.get("joint_exact_accuracy"), location="offline joint_exact_accuracy"
        ),
        "valid_json_rate": _unit_float(
            row.get("valid_json_rate"), location="offline valid_json_rate"
        ),
    }


def _coverage(report: Mapping[str, Any], *, manifest_size: int) -> dict[str, Any]:
    per_seed = report["per_seed"][0]
    return {
        "manifest_records": manifest_size,
        "supplied_predictions": per_seed["supplied_prediction_count"],
        "missing_predictions": per_seed["missing_prediction_count"],
        "coverage_rate": per_seed["coverage_rate"],
        "expected_seed_enforced": report["seed_enforcement"] == {
            "enabled": True,
            "expected_seeds": [REFERENCE_SEED],
        },
    }


def _validate_predictions(
    entry: Any,
    *,
    root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    manifest_ids: Sequence[str],
    location: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    identity = _validate_file_entry(
        entry,
        root=root,
        location=location,
        require_file=True,
        prohibit_protected_path=True,
    )
    rows = _read_jsonl(
        _resolve_inside(root, identity["path"], location=location), location=location
    )
    for index, row in enumerate(rows):
        _exact_keys(row, {"sample_id", "prediction", "seed"}, f"{location}[{index}]")
        if row.get("seed") != REFERENCE_SEED:
            raise BaselineSelectionError(f"{location}[{index}] uses a non-reference seed")
        if not isinstance(row.get("prediction"), str):
            raise BaselineSelectionError(f"{location}[{index}].prediction must be a string")
    if [row.get("sample_id") for row in rows] != list(manifest_ids):
        raise BaselineSelectionError(
            f"{location} must contain exactly one row per dev sample in manifest order"
        )
    report = evaluate_records(
        manifest_rows,
        rows,
        expected_seeds=[REFERENCE_SEED],
    )
    if _coverage(report, manifest_size=len(manifest_rows)) != {
        "manifest_records": len(manifest_rows),
        "supplied_predictions": len(manifest_rows),
        "missing_predictions": 0,
        "coverage_rate": 1.0,
        "expected_seed_enforced": True,
    }:
        raise BaselineSelectionError(f"{location} does not have complete dev coverage")
    return rows, report, identity


def _validate_offline_report(
    entry: Any,
    *,
    root: Path,
    expected: Mapping[str, Any],
    location: str,
) -> dict[str, str]:
    identity = _validate_file_entry(
        entry,
        root=root,
        location=location,
        require_file=True,
        prohibit_protected_path=True,
    )
    observed = _read_json(
        _resolve_inside(root, identity["path"], location=location), location=location
    )
    if observed != expected:
        raise BaselineSelectionError(f"{location} differs from current offline recomputation")
    return identity


def _a_rank_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["ranking_metrics"]
    return (
        -float(metrics["joint_exact_accuracy"]),
        -float(metrics["diagnosis_macro_f1"]),
        -float(metrics["valid_json_rate"]),
        int(candidate["model_complexity_rank"]),
        str(candidate["candidate_name"]),
    )


def _validate_arm_a(
    value: Any,
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    manifest_ids: Sequence[str],
) -> dict[str, Any]:
    arm = _exact_keys(
        value,
        {
            "selection_rule",
            "candidates",
            "selected",
            "protected_or_frozen_data_used",
            "frozen_predictions_opened",
        },
        "arm_A",
    )
    if arm.get("selection_rule") != list(A_SELECTION_RULE):
        raise BaselineSelectionError("arm_A selection rule differs from the frozen rule")
    matrix_a = config.get("comparison_matrix", {}).get("A", {})
    if not isinstance(matrix_a, Mapping) or tuple(matrix_a.get("candidates", ())) != A_CANDIDATES:
        raise BaselineSelectionError("evaluation config A candidate registry differs")
    candidates = arm.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(A_CANDIDATES):
        raise BaselineSelectionError("arm_A must contain every configured candidate exactly once")
    validated: list[dict[str, Any]] = []
    for index, raw_candidate in enumerate(candidates):
        candidate = _exact_keys(raw_candidate, A_CANDIDATE_KEYS, f"arm_A.candidates[{index}]")
        name = candidate.get("candidate_name")
        if name not in A_COMPLEXITY_RANK:
            raise BaselineSelectionError(f"unknown arm_A candidate {name!r}")
        if candidate.get("model_complexity_rank") != A_COMPLEXITY_RANK[name]:
            raise BaselineSelectionError(f"arm_A candidate {name} complexity rank changed")
        if (
            candidate.get("eligible") is not True
            or candidate.get("protected_or_frozen_data_used") is not False
            or candidate.get("frozen_predictions_opened") is not False
        ):
            raise BaselineSelectionError(f"arm_A candidate {name} is ineligible or unsafe")
        model = _validate_file_entry(
            candidate["model_artifact"],
            root=root,
            location=f"arm_A candidate {name} model_artifact",
            prohibit_protected_path=True,
        )
        training_report = _validate_file_entry(
            candidate["training_report"],
            root=root,
            location=f"arm_A candidate {name} training_report",
            require_file=True,
            prohibit_protected_path=True,
        )
        _, report, predictions = _validate_predictions(
            candidate["predictions"],
            root=root,
            manifest_rows=manifest_rows,
            manifest_ids=manifest_ids,
            location=f"arm_A candidate {name} predictions",
        )
        offline = _validate_offline_report(
            candidate["offline_report"],
            root=root,
            expected=report,
            location=f"arm_A candidate {name} offline_report",
        )
        metrics = _validate_ranking_metrics(
            candidate["ranking_metrics"],
            location=f"arm_A candidate {name} ranking_metrics",
        )
        if metrics != _report_metrics(report) or metrics["valid_json_rate"] != 1.0:
            raise BaselineSelectionError(f"arm_A candidate {name} ranking metrics differ")
        coverage = _exact_keys(
            candidate["dev_coverage"], COVERAGE_KEYS, f"arm_A candidate {name} dev_coverage"
        )
        if dict(coverage) != _coverage(report, manifest_size=len(manifest_rows)):
            raise BaselineSelectionError(f"arm_A candidate {name} coverage differs")
        rank = candidate.get("selection_rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise BaselineSelectionError(f"arm_A candidate {name} selection_rank is invalid")
        validated.append(
            {
                **dict(candidate),
                "model_artifact": model,
                "training_report": training_report,
                "predictions": predictions,
                "offline_report": offline,
                "ranking_metrics": metrics,
            }
        )
    if {candidate["candidate_name"] for candidate in validated} != set(A_CANDIDATES):
        raise BaselineSelectionError("arm_A candidate registry is incomplete or duplicated")
    if validated != sorted(validated, key=_a_rank_key):
        raise BaselineSelectionError("arm_A candidates are not ordered by the frozen rule")
    if [candidate["selection_rank"] for candidate in validated] != list(
        range(1, len(validated) + 1)
    ):
        raise BaselineSelectionError("arm_A selection ranks are incomplete")
    winner = validated[0]
    expected_selected = {
        "candidate_name": winner["candidate_name"],
        "selection_rank": 1,
        "model_artifact": winner["model_artifact"],
        "ranking_metrics": winner["ranking_metrics"],
        "output_mapping": OUTPUT_MAPPING,
    }
    if arm.get("selected") != expected_selected:
        raise BaselineSelectionError("arm_A selected entry is not the rank-one candidate")
    if (
        arm.get("protected_or_frozen_data_used") is not False
        or arm.get("frozen_predictions_opened") is not False
    ):
        raise BaselineSelectionError("arm_A reports protected/frozen use")
    return {
        "selected_candidate": winner["candidate_name"],
        "selected_model_artifact": winner["model_artifact"],
        "ranking_metrics": winner["ranking_metrics"],
    }


def _canonical_prediction(diagnosis: str) -> str:
    return json.dumps(
        OUTPUT_MAPPING[diagnosis],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _b_diagnosis(
    saturation_score: float,
    reflection_score: float,
    saturation_threshold: float,
    reflection_threshold: float,
) -> str:
    if saturation_score < saturation_threshold and reflection_score < reflection_threshold:
        return "nominal"
    saturation_margin = (saturation_score - saturation_threshold) / max(
        abs(saturation_threshold), 1e-12
    )
    reflection_margin = (reflection_score - reflection_threshold) / max(
        abs(reflection_threshold), 1e-12
    )
    # The frozen tie order places saturation before reflection once nominal is
    # ruled out by the strict-below condition.
    return "sensor_saturation" if saturation_margin >= reflection_margin else "secondary_reflection"


def _b_prediction_rows(
    scores: Sequence[Mapping[str, Any]],
    *,
    saturation_threshold: float,
    reflection_threshold: float,
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row["sample_id"],
            "prediction": _canonical_prediction(
                _b_diagnosis(
                    float(row["sensor_saturation_score"]),
                    float(row["secondary_reflection_score"]),
                    saturation_threshold,
                    reflection_threshold,
                )
            ),
            "seed": REFERENCE_SEED,
        }
        for row in scores
    ]


def _b_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["ranking_metrics"]
    return (
        -float(metrics["joint_exact_accuracy"]),
        -float(metrics["diagnosis_macro_f1"]),
        -float(metrics["valid_json_rate"]),
        float(row["sensor_saturation_threshold"]),
        float(row["secondary_reflection_threshold"]),
    )


def _validate_threshold_values(value: Any, *, location: str) -> list[float]:
    if not isinstance(value, list) or len(value) < 2:
        raise BaselineSelectionError(f"{location} must contain at least two thresholds")
    values = [_unit_float(item, location=f"{location}[{index}]") for index, item in enumerate(value)]
    if values != sorted(set(values)):
        raise BaselineSelectionError(f"{location} must be strictly increasing and unique")
    return values


def _validate_arm_b(
    value: Any,
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    manifest_ids: Sequence[str],
) -> dict[str, Any]:
    arm = _exact_keys(
        value,
        {
            "fixed_specialists",
            "score_evidence",
            "threshold_grid",
            "grid_definition",
            "selection_rule",
            "predictions",
            "offline_report",
            "dev_coverage",
            "selected",
            "protected_or_frozen_data_used",
            "frozen_predictions_opened",
        },
        "arm_B",
    )
    if arm.get("selection_rule") != list(B_SELECTION_RULE):
        raise BaselineSelectionError("arm_B selection rule differs from the frozen rule")
    specialists = _exact_keys(
        arm["fixed_specialists"],
        {"saturation_multimodal_small_model", "width_relative_reflection_selected_diagnostic"},
        "arm_B.fixed_specialists",
    )
    configured = config.get("comparison_matrix", {}).get("B", {}).get("fixed_specialists", {})
    if not isinstance(configured, Mapping):
        raise BaselineSelectionError("evaluation config omits B specialist identities")
    for name in specialists:
        observed = _validate_file_entry(
            specialists[name], root=root, location=f"arm_B.fixed_specialists.{name}"
        )
        expected = _config_entry(
            root, configured.get(name), location=f"config comparison_matrix.B.{name}"
        )
        if observed != expected:
            raise BaselineSelectionError(f"arm_B specialist {name} differs from config")

    score_identity = _validate_file_entry(
        arm["score_evidence"],
        root=root,
        location="arm_B.score_evidence",
        require_file=True,
        prohibit_protected_path=True,
    )
    scores = _read_jsonl(
        _resolve_inside(root, score_identity["path"], location="arm_B.score_evidence"),
        location="arm_B.score_evidence",
    )
    for index, row in enumerate(scores):
        _exact_keys(
            row,
            {"sample_id", "sensor_saturation_score", "secondary_reflection_score"},
            f"arm_B.score_evidence[{index}]",
        )
        _unit_float(
            row["sensor_saturation_score"],
            location=f"arm_B.score_evidence[{index}].sensor_saturation_score",
        )
        _unit_float(
            row["secondary_reflection_score"],
            location=f"arm_B.score_evidence[{index}].secondary_reflection_score",
        )
    if [row.get("sample_id") for row in scores] != list(manifest_ids):
        raise BaselineSelectionError(
            "arm_B score evidence must contain exactly one row per dev sample in manifest order"
        )

    definition = _exact_keys(
        arm["grid_definition"],
        {
            "sensor_saturation_thresholds",
            "secondary_reflection_thresholds",
            "cartesian_product_complete",
        },
        "arm_B.grid_definition",
    )
    saturation_values = _validate_threshold_values(
        definition["sensor_saturation_thresholds"],
        location="arm_B.grid_definition.sensor_saturation_thresholds",
    )
    reflection_values = _validate_threshold_values(
        definition["secondary_reflection_thresholds"],
        location="arm_B.grid_definition.secondary_reflection_thresholds",
    )
    if definition.get("cartesian_product_complete") is not True:
        raise BaselineSelectionError("arm_B threshold grid must be the complete Cartesian product")

    grid_identity = _validate_file_entry(
        arm["threshold_grid"],
        root=root,
        location="arm_B.threshold_grid",
        require_file=True,
        prohibit_protected_path=True,
    )
    grid = _read_jsonl(
        _resolve_inside(root, grid_identity["path"], location="arm_B.threshold_grid"),
        location="arm_B.threshold_grid",
    )
    expected_pairs = [
        (sat, reflection)
        for sat in saturation_values
        for reflection in reflection_values
    ]
    if len(grid) != len(expected_pairs):
        raise BaselineSelectionError("arm_B threshold grid is not the complete Cartesian product")
    validated_grid: list[dict[str, Any]] = []
    observed_pairs: set[tuple[float, float]] = set()
    for index, raw_row in enumerate(grid):
        row = _exact_keys(
            raw_row,
            {
                "sensor_saturation_threshold",
                "secondary_reflection_threshold",
                "selection_rank",
                "ranking_metrics",
            },
            f"arm_B.threshold_grid[{index}]",
        )
        sat = _unit_float(
            row["sensor_saturation_threshold"],
            location=f"arm_B.threshold_grid[{index}].sensor_saturation_threshold",
        )
        reflection = _unit_float(
            row["secondary_reflection_threshold"],
            location=f"arm_B.threshold_grid[{index}].secondary_reflection_threshold",
        )
        pair = (sat, reflection)
        if pair in observed_pairs or pair not in set(expected_pairs):
            raise BaselineSelectionError(f"arm_B threshold grid has duplicate/unregistered pair {pair}")
        observed_pairs.add(pair)
        derived = _b_prediction_rows(
            scores,
            saturation_threshold=sat,
            reflection_threshold=reflection,
        )
        report = evaluate_records(
            manifest_rows,
            derived,
            expected_seeds=[REFERENCE_SEED],
        )
        metrics = _validate_ranking_metrics(
            row["ranking_metrics"], location=f"arm_B.threshold_grid[{index}].ranking_metrics"
        )
        if metrics != _report_metrics(report) or metrics["valid_json_rate"] != 1.0:
            raise BaselineSelectionError(
                f"arm_B threshold grid row {pair} differs from raw-score recomputation"
            )
        rank = row.get("selection_rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise BaselineSelectionError(f"arm_B threshold grid row {pair} has invalid rank")
        validated_grid.append(
            {
                "sensor_saturation_threshold": sat,
                "secondary_reflection_threshold": reflection,
                "selection_rank": rank,
                "ranking_metrics": metrics,
            }
        )
    if observed_pairs != set(expected_pairs):
        raise BaselineSelectionError("arm_B threshold grid omits configured pairs")
    if validated_grid != sorted(validated_grid, key=_b_rank_key):
        raise BaselineSelectionError("arm_B threshold grid is not ordered by the frozen rule")
    if [row["selection_rank"] for row in validated_grid] != list(
        range(1, len(validated_grid) + 1)
    ):
        raise BaselineSelectionError("arm_B threshold ranks are incomplete")

    winner = validated_grid[0]
    selected = _exact_keys(
        arm["selected"],
        {"thresholds", "selection_rank", "ranking_metrics", "arbitration", "output_mapping"},
        "arm_B.selected",
    )
    thresholds = _exact_keys(
        selected["thresholds"],
        {"sensor_saturation", "secondary_reflection"},
        "arm_B.selected.thresholds",
    )
    expected_selected = {
        "thresholds": {
            "sensor_saturation": winner["sensor_saturation_threshold"],
            "secondary_reflection": winner["secondary_reflection_threshold"],
        },
        "selection_rank": 1,
        "ranking_metrics": winner["ranking_metrics"],
        "arbitration": B_ARBITRATION,
        "output_mapping": OUTPUT_MAPPING,
    }
    if dict(selected) != expected_selected or dict(thresholds) != expected_selected["thresholds"]:
        raise BaselineSelectionError("arm_B selected thresholds/arbitration are not rank one")

    observed_predictions, report, predictions_identity = _validate_predictions(
        arm["predictions"],
        root=root,
        manifest_rows=manifest_rows,
        manifest_ids=manifest_ids,
        location="arm_B.predictions",
    )
    expected_predictions = _b_prediction_rows(
        scores,
        saturation_threshold=winner["sensor_saturation_threshold"],
        reflection_threshold=winner["secondary_reflection_threshold"],
    )
    if observed_predictions != expected_predictions:
        raise BaselineSelectionError(
            "arm_B predictions differ from selected thresholds and frozen arbitration"
        )
    offline_identity = _validate_offline_report(
        arm["offline_report"],
        root=root,
        expected=report,
        location="arm_B.offline_report",
    )
    if _report_metrics(report) != winner["ranking_metrics"]:
        raise BaselineSelectionError("arm_B selected offline metrics differ from threshold grid")
    coverage = _exact_keys(arm["dev_coverage"], COVERAGE_KEYS, "arm_B.dev_coverage")
    if dict(coverage) != _coverage(report, manifest_size=len(manifest_rows)):
        raise BaselineSelectionError("arm_B development coverage differs")
    if (
        arm.get("protected_or_frozen_data_used") is not False
        or arm.get("frozen_predictions_opened") is not False
    ):
        raise BaselineSelectionError("arm_B reports protected/frozen use")
    return {
        "selected_thresholds": expected_selected["thresholds"],
        "ranking_metrics": winner["ranking_metrics"],
        "score_evidence": score_identity,
        "threshold_grid": grid_identity,
        "predictions": predictions_identity,
        "offline_report": offline_identity,
    }


def validate_baseline_selection_artifact(
    artifact: Mapping[str, Any],
    *,
    repository_root: Path,
    evaluation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and recompute one future A/B dev-selection artifact."""

    root = repository_root.resolve()
    artifact = _exact_keys(artifact, TOP_KEYS, "artifact")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise BaselineSelectionError("baseline selection schema_version differs")
    if artifact.get("artifact_kind") != "dev_only_A_and_B_baseline_selection":
        raise BaselineSelectionError("baseline selection artifact_kind differs")
    if artifact.get("selection_split") != "dev":
        raise BaselineSelectionError("baseline selection must use dev only")
    if (
        artifact.get("protected_or_frozen_data_used") is not False
        or artifact.get("frozen_predictions_opened") is not False
        or artifact.get("timestamp_in_artifact") is not False
    ):
        raise BaselineSelectionError("artifact protection/timestamp flags differ")

    manifests = evaluation_config.get("manifests")
    if not isinstance(manifests, Mapping):
        raise BaselineSelectionError("evaluation config omits manifests")
    train_rows, _ = _validate_manifest_identity(
        artifact["train_data"],
        root=root,
        configured=manifests.get("train", {}),
        expected_split="train",
        location="train_data",
    )
    dev_rows, dev_ids = _validate_manifest_identity(
        artifact["dev_data"],
        root=root,
        configured=manifests.get("dev", {}),
        expected_split="dev",
        location="dev_data",
    )

    immutable = evaluation_config.get("immutable_artifacts")
    if not isinstance(immutable, Mapping):
        raise BaselineSelectionError("evaluation config omits immutable_artifacts")
    validator = _exact_keys(
        artifact["validator"],
        {
            "implementation",
            "json_schema",
            "offline_reports_recomputed_from_raw_predictions",
            "B_grid_recomputed_from_raw_scores",
        },
        "validator",
    )
    expected_identities = {
        "implementation": "baseline_dev_selection_validator",
        "json_schema": "baseline_dev_selection_artifact_schema",
    }
    for artifact_key, config_key in expected_identities.items():
        observed = _validate_file_entry(
            validator[artifact_key], root=root, location=f"validator.{artifact_key}"
        )
        expected = _config_entry(
            root, immutable.get(config_key), location=f"immutable_artifacts.{config_key}"
        )
        if observed != expected:
            raise BaselineSelectionError(f"validator.{artifact_key} differs from config")
    if (
        validator.get("offline_reports_recomputed_from_raw_predictions") is not True
        or validator.get("B_grid_recomputed_from_raw_scores") is not True
    ):
        raise BaselineSelectionError("baseline validator audit flags differ")

    arm_a = _validate_arm_a(
        artifact["arm_A"],
        root=root,
        config=evaluation_config,
        manifest_rows=dev_rows,
        manifest_ids=dev_ids,
    )
    arm_b = _validate_arm_b(
        artifact["arm_B"],
        root=root,
        config=evaluation_config,
        manifest_rows=dev_rows,
        manifest_ids=dev_ids,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_split": "dev",
        "train_records": len(train_rows),
        "dev_records": len(dev_rows),
        "arm_A": arm_a,
        "arm_B": arm_b,
        "protected_or_frozen_data_used": False,
        "frozen_predictions_opened": False,
        "verification": "passed_exact_raw_evidence_recomputation",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--evaluation-config", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args(argv)
    root = args.repository_root.resolve()
    config_path = _resolve_inside(root, args.evaluation_config, location="evaluation config")
    artifact_path = _resolve_inside(root, args.artifact, location="baseline selection artifact")
    artifact = _read_json(artifact_path, location="baseline selection artifact")
    result = validate_baseline_selection_artifact(
        artifact,
        repository_root=root,
        evaluation_config=_load_yaml(config_path),
    )
    output = {
        "artifact": artifact_path.relative_to(root).as_posix(),
        "sha256": _sha256_file(artifact_path),
        **result,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
