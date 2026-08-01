#!/usr/bin/env python3
"""Prepare, seal, or verify the Qwen-VL evaluation gate without inference.

``prepare`` creates the deterministic pre-server protocol artifact and hashes
the actual current-image bytes named by all four manifests.  The utility has
no prediction input and no model/controller execution path.  Later, ``freeze``
refuses to seal a final artifact unless that exact preparation artifact and its
externally recorded SHA-256 still verify, in addition to the final checkpoint,
hashed server-training configuration, strict Qwen and A/B dev-selection artifacts, and
concrete prediction/closed-loop runners.  ``verify`` is the mandatory preflight
immediately before a future formal evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_ARMS = ("A", "B", "C", "D")
EXPECTED_SPLITS = ("train", "dev", "frozen_iid", "frozen_ood")
EXPECTED_A_SELECTION_RULE = (
    "highest_joint_diagnosis_policy_action_exact_accuracy",
    "highest_diagnosis_macro_f1",
    "highest_valid_json_rate",
    "lowest_model_complexity",
    "lexicographically_lowest_candidate_name",
)
EXPECTED_B_ARBITRATION = {
    "score_domain": "calibrated_probability_closed_unit_interval",
    "nominal_rule": "both_scores_strictly_below_their_selected_thresholds",
    "anomaly_rule": "otherwise_choose_larger_normalized_margin_above_threshold",
    "normalized_margin_formula": "(score-threshold)/max(abs(threshold),1e-12)",
    "threshold_comparator_for_anomaly": "greater_than_or_equal",
    "tie_order": ["nominal", "sensor_saturation", "secondary_reflection"],
}
REQUIRED_IMMUTABLE_ARTIFACTS = {
    "server_training_commands",
    "dev_checkpoint_selector",
    "dev_selection_artifact_schema",
    "server_selection_commands",
    "baseline_dev_selection_validator",
    "baseline_dev_selection_artifact_schema",
    "server_baseline_selection_commands",
    "reproduction_commands",
    "saturation_generator_behavior",
    "width_relative_reflection_generator",
    "saturation_generator_runtime",
    "width_relative_reflection_generator_runtime",
    "forward_ensemble",
    "corrected_v12_config",
    "v13_config",
    "cem_runtime",
    "gain_probe_model",
    "sequential_rule_report",
    "visual_controller_runtime",
}
EXPECTED_OFFLINE_METRICS = {
    "valid_json_rate",
    "diagnosis_balanced_accuracy",
    "diagnosis_macro_f1",
    "diagnosis_per_class_precision",
    "diagnosis_per_class_recall",
    "measurement_policy_accuracy",
    "supervisor_action_macro_f1",
    "joint_diagnosis_policy_action_exact_accuracy",
    "diagnosis_confusion_matrix",
    "measurement_policy_confusion_matrix",
    "supervisor_action_confusion_matrix",
    "performance_by_anomaly_family",
    "reflection_performance_by_beam_width_quartile",
    "boundary_versus_non_boundary_performance",
    "performance_by_severity",
}
EXPECTED_CLOSED_LOOP_METRICS = {
    "strict_all_five_success",
    "matched_recoveries",
    "matched_regressions",
    "final_normalized_target_distance",
    "executed_steps",
    "actuator_saturation",
    "hard_constraint_violations",
    "results_by_family",
    "results_by_width_quartile",
    "results_by_boundary_status",
    "exact_qwen_versus_oracle_disagreements",
    "diagnosis_correct_but_control_failed_cases",
    "diagnosis_wrong_but_accidentally_recovered_cases",
}


class FreezeError(RuntimeError):
    """Raised when a formal-evaluation invariant is not frozen."""


def _reject_constant(value: str) -> None:
    raise FreezeError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FreezeError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"cannot read strict JSON {path}: {exc}") from exc


def _assert_finite(value: Any, location: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FreezeError(f"non-finite value at {location}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_finite(child, f"{location}[{index}]")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment error
        raise FreezeError("PyYAML is required to read evaluation_frozen.yaml") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise FreezeError(f"cannot read YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FreezeError("evaluation config must be one YAML object")
    _assert_finite(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FreezeError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _hash_path(path: Path) -> tuple[str, str]:
    """Return (algorithm, digest) for a regular file or deterministic tree."""

    if not path.exists():
        raise FreezeError(f"required path does not exist: {path}")
    if path.is_symlink():
        raise FreezeError(f"symbolic links are not accepted as frozen artifacts: {path}")
    if path.is_file():
        return "sha256_file_v1", _sha256_file(path)
    if not path.is_dir():
        raise FreezeError(f"required path is neither a file nor directory: {path}")

    files = sorted(item for item in path.rglob("*") if item.is_file() or item.is_symlink())
    if not files:
        raise FreezeError(f"cannot freeze an empty checkpoint directory: {path}")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise FreezeError(f"checkpoint tree contains a symbolic link: {item}")
        relative = item.relative_to(path).as_posix()
        file_hash = _sha256_file(item)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return "sha256_tree_relative_path_nul_file_sha256_lf_v1", digest.hexdigest()


def _require_sha256(name: str, value: str) -> str:
    normalized = str(value).lower()
    if not HEX_SHA256.fullmatch(normalized):
        raise FreezeError(f"{name} must be an explicit lowercase SHA-256, got {value!r}")
    if len(set(normalized)) == 1:
        raise FreezeError(f"{name} looks like a placeholder, not an artifact hash")
    return normalized


def _verify_expected_hash(name: str, path: Path, expected: str) -> tuple[str, str]:
    expected = _require_sha256(name, expected)
    algorithm, actual = _hash_path(path)
    if actual != expected:
        raise FreezeError(f"{name} mismatch for {path}: expected {expected}, got {actual}")
    return algorithm, actual


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _require_inside_root(root: Path, path: Path, *, name: str) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise FreezeError(f"{name} escapes repository root: {resolved}") from exc
    return resolved


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise FreezeError(f"evaluation config field {key!r} must be an object")
    return value


def _validate_protocol(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "qwen_vl_supervisor_evaluation_v1.0.0":
        raise FreezeError("unexpected evaluation protocol schema_version")
    if config.get("protocol_state") != "frozen_before_formal_training_or_prediction":
        raise FreezeError("protocol must be frozen before formal training or prediction")

    guard = _require_mapping(config, "scope_guard")
    if guard.get("pre_server_protocol_freeze_artifact") != (
        "required_before_server_training"
    ):
        raise FreezeError("pre-server protocol freeze must be required before training")
    if (
        guard.get("later_final_freeze_must_verify_pre_server_artifact_and_external_sha256")
        is not True
    ):
        raise FreezeError("final freeze must verify the pre-server artifact and SHA-256")
    if guard.get("formal_evaluation_during_pipeline_preparation") != "forbidden":
        raise FreezeError("formal evaluation must remain forbidden during preparation")
    if guard.get("frozen_prediction_access_during_freeze") != "forbidden":
        raise FreezeError("freeze utility may not access frozen predictions")
    if guard.get("continuous_actuator_output") != "forbidden":
        raise FreezeError("Qwen continuous actuator output must be forbidden")

    finalization = _require_mapping(config, "finalization")
    _require_sha256(
        "finalization.training_config_sha256",
        str(finalization.get("training_config_sha256", "")),
    )
    frozen_training_seeds = finalization.get("training_seeds")
    if (
        not isinstance(frozen_training_seeds, list)
        or not frozen_training_seeds
        or any(
            not isinstance(seed, int) or isinstance(seed, bool)
            for seed in frozen_training_seeds
        )
        or len(frozen_training_seeds) != len(set(frozen_training_seeds))
    ):
        raise FreezeError("finalization.training_seeds must be a nonempty unique integer list")

    matrix = _require_mapping(config, "comparison_matrix")
    if tuple(matrix.keys()) != EXPECTED_ARMS:
        raise FreezeError(f"comparison matrix must be ordered exactly {EXPECTED_ARMS}")
    arm_a = _require_mapping(matrix, "A")
    if tuple(arm_a.get("selection_rule", [])) != EXPECTED_A_SELECTION_RULE:
        raise FreezeError("arm A dev-selection rule differs from the frozen rank order")
    arm_b = _require_mapping(matrix, "B")
    b_arbitration = _require_mapping(arm_b, "unified_inference_rule")
    for field, expected in EXPECTED_B_ARBITRATION.items():
        if b_arbitration.get(field) != expected:
            raise FreezeError(f"arm B frozen arbitration field {field} differs")

    manifests = _require_mapping(config, "manifests")
    if tuple(manifests.keys()) != EXPECTED_SPLITS:
        raise FreezeError(f"manifest registry must be ordered exactly {EXPECTED_SPLITS}")

    immutable = _require_mapping(config, "immutable_artifacts")
    missing_immutable = REQUIRED_IMMUTABLE_ARTIFACTS - set(immutable)
    if missing_immutable:
        raise FreezeError(
            f"immutable artifact registry is missing {sorted(missing_immutable)}"
        )

    offline = _require_mapping(config, "offline_evaluation")
    if set(offline.get("required_metrics", [])) != EXPECTED_OFFLINE_METRICS:
        raise FreezeError("offline required metrics differ from the frozen Phase-5 set")
    masking = _require_mapping(offline, "continue_stop_masking")
    if masking.get("per_class_continue_stop_scores_on_static_records") != "forbidden":
        raise FreezeError("continue/stop must not be scored as static target classes")
    arm_seed_semantics = _require_mapping(offline, "offline_arm_seed_semantics")
    c_seed_semantics = _require_mapping(arm_seed_semantics, "C")
    deterministic_seed_semantics = _require_mapping(arm_seed_semantics, "A_B_D")
    if (
        c_seed_semantics.get("reducer_seed_gate") != "evaluation_config"
        or c_seed_semantics.get("duplication")
        != "forbidden_beyond_one_row_per_sample_per_training_seed"
    ):
        raise FreezeError("arm C must use the exact configured training-seed set")
    if (
        deterministic_seed_semantics.get("reducer_seed_gate")
        != "expected_seeds_deterministic_reference"
        or deterministic_seed_semantics.get("duplicate_across_C_training_seeds")
        != "forbidden"
    ):
        raise FreezeError("A/B/D must remain single deterministic-reference reductions")

    closed = _require_mapping(config, "closed_loop_evaluation")
    if tuple(closed.get("arms", [])) != EXPECTED_ARMS:
        raise FreezeError("closed-loop arms must be A/B/C/D")
    if set(closed.get("required_metrics", [])) != EXPECTED_CLOSED_LOOP_METRICS:
        raise FreezeError("closed-loop required metrics differ from the frozen Phase-5 set")
    controller = _require_mapping(closed, "controller")
    required_controller_values = {
        "imagined_model_horizon": 1,
        "maximum_horizon": 8,
        "population": 24,
        "elites": 6,
        "cem_iterations": 3,
    }
    for field, expected in required_controller_values.items():
        if controller.get(field) != expected:
            raise FreezeError(f"frozen controller {field} must equal {expected}")
    if "real_observation" not in str(controller.get("planner", "")):
        raise FreezeError("planner must replan from real observations")

    paired = _require_mapping(closed, "paired_invariants")
    if not all(value is True for value in paired.values()):
        raise FreezeError("every closed-loop paired invariant must be true")

    limitations = _require_mapping(config, "known_reflection_limitations")
    expected_limitations = {
        "Q1_narrow_beam_accuracy_percent": 68.75,
        "width_quartile_gap_percentage_points": 31.25,
        "boundary_accuracy_percent": 75.0,
        "valid_width_relative_severity_OOD_conclusion": False,
    }
    for field, expected in expected_limitations.items():
        if limitations.get(field) != expected:
            raise FreezeError(f"reflection limitation {field} must remain {expected!r}")
    if limitations.get("strict_preregistered_pass_claim") != "forbidden":
        raise FreezeError("reflection may not be relabeled as a strict preregistered pass")

    claims = _require_mapping(config, "claims")
    if claims.get("formal_frozen_evaluation_completed") is not False:
        raise FreezeError("config must state that formal frozen evaluation is not complete")
    if claims.get("smoke_test_can_validate_scientific_performance") is not False:
        raise FreezeError("smoke results may not validate scientific performance")

    commands = _require_mapping(config, "exact_commands")
    prepare_command = str(commands.get("prepare_protocol_artifact", ""))
    if (
        "freeze_evaluation.py prepare" not in prepare_command
        or "--authorize-pre-server-protocol-freeze" not in prepare_command
    ):
        raise FreezeError("exact prepare command does not seal the pre-server protocol")
    final_freeze_command = str(commands.get("create_freeze_artifact", ""))
    if (
        "--preparation-artifact" not in final_freeze_command
        or "--preparation-artifact-sha256" not in final_freeze_command
    ):
        raise FreezeError("exact final freeze command omits the pre-server artifact or SHA-256")
    if (
        "--baseline-dev-selection-artifact" not in final_freeze_command
        or "--baseline-dev-selection-artifact-sha256" not in final_freeze_command
    ):
        raise FreezeError("exact final freeze command omits validated A/B dev selection")
    matrix_command = str(commands.get("offline_reduce_formal_matrix", ""))
    required_matrix_fragments = (
        "for SPLIT in frozen_iid frozen_ood",
        "for ARM in A B D",
        "--expected-seeds deterministic_reference",
        "arm_C_predictions_all_training_seeds.jsonl",
        "--evaluation-config qwen_vl_supervisor_v1/configs/evaluation_frozen.yaml",
    )
    missing_matrix_fragments = [
        fragment for fragment in required_matrix_fragments if fragment not in matrix_command
    ]
    if missing_matrix_fragments:
        raise FreezeError(
            "exact formal offline matrix command omits seed/arm enforcement: "
            f"{missing_matrix_fragments}"
        )


def _read_manifest(path: Path, expected_split: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FreezeError(f"manifest does not exist: {path}")
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise FreezeError(f"cannot read manifest {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, FreezeError) as exc:
            raise FreezeError(f"invalid manifest JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise FreezeError(f"manifest row {path}:{line_number} is not an object")
        if row.get("split") != expected_split:
            raise FreezeError(
                f"manifest {path}:{line_number} declares split {row.get('split')!r}, "
                f"expected {expected_split!r}"
            )
        records.append(row)
    if not records:
        raise FreezeError(f"manifest is empty: {path}")
    return records


def _verified_manifest_image(
    root: Path,
    *,
    split: str,
    sample_id: str,
    assets: Mapping[str, Any],
) -> dict[str, str]:
    raw_path = assets.get("current_image_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise FreezeError(f"sample {sample_id} has no current_image_path")
    unresolved = Path(raw_path)
    unresolved = unresolved if unresolved.is_absolute() else root / unresolved
    lexical_path = Path(os.path.abspath(unresolved))
    try:
        relative_parts = lexical_path.relative_to(root.resolve()).parts
    except ValueError as exc:
        raise FreezeError(
            f"sample {sample_id} current image escapes repository root: {lexical_path}"
        ) from exc
    cursor = root.resolve()
    for part in relative_parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise FreezeError(
                f"sample {sample_id} current image path contains a symbolic link: {cursor}"
            )
    image_path = _require_inside_root(
        root,
        lexical_path,
        name=f"sample {sample_id} current image",
    )
    expected = _require_sha256(
        f"sample {sample_id} image hash",
        str(assets.get("current_image_sha256", "")),
    )
    algorithm, actual = _hash_path(image_path)
    if algorithm != "sha256_file_v1":
        raise FreezeError(f"sample {sample_id} current image is not a regular file: {image_path}")
    if actual != expected:
        raise FreezeError(
            f"sample {sample_id} image-byte SHA-256 mismatch for {image_path}: "
            f"expected {expected}, got {actual}"
        )
    return {
        "sample_id": sample_id,
        "split": split,
        "path": _display_path(root, image_path),
        "hash_algorithm": algorithm,
        "sha256": actual,
        "manifest_current_image_sha256": expected,
    }


def _manifest_snapshot(
    root: Path,
    registry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, set[str]], dict[str, set[str]], dict[str, dict[str, str]]]:
    snapshot: dict[str, Any] = {}
    setup_sets: dict[str, set[str]] = {}
    image_sets: dict[str, set[str]] = {}
    controller_sets: dict[str, dict[str, str]] = {}
    all_sample_ids: set[str] = set()
    pair_owner: dict[str, str] = {}
    episode_owner: dict[str, str] = {}

    for split in EXPECTED_SPLITS:
        entry = registry[split]
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise FreezeError(f"manifest registry {split} must contain a path")
        path = _resolve(root, entry["path"])
        records = _read_manifest(path, split)
        image_hashes: list[str] = []
        images: list[dict[str, str]] = []
        setups: set[str] = set()
        generators: dict[str, set[str]] = {}
        controllers: dict[str, set[str]] = {}
        pairs: set[str] = set()
        episodes: set[str] = set()

        for row_index, row in enumerate(records):
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise FreezeError(f"{path}: record {row_index} has no sample_id")
            if sample_id in all_sample_ids:
                raise FreezeError(f"duplicate sample_id across manifests: {sample_id}")
            all_sample_ids.add(sample_id)

            setup_hash = row.get("setup_hash")
            if not isinstance(setup_hash, str) or not setup_hash:
                raise FreezeError(f"sample {sample_id} has no irreversible setup hash")
            setups.add(setup_hash)

            assets = row.get("assets")
            if not isinstance(assets, Mapping):
                raise FreezeError(f"sample {sample_id} has no assets object")
            image = _verified_manifest_image(
                root,
                split=split,
                sample_id=sample_id,
                assets=assets,
            )
            image_hash = image["sha256"]
            image_hashes.append(image_hash)
            images.append(image)

            pair_id = row.get("counterfactual_pair_id")
            if isinstance(pair_id, str) and pair_id:
                previous = pair_owner.setdefault(pair_id, split)
                if previous != split:
                    raise FreezeError(f"counterfactual pair {pair_id} spans {previous} and {split}")
                pairs.add(pair_id)
            episode_id = row.get("episode_hash")
            if isinstance(episode_id, str) and episode_id:
                previous = episode_owner.setdefault(episode_id, split)
                if previous != split:
                    raise FreezeError(f"episode {episode_id} spans {previous} and {split}")
                episodes.add(episode_id)

            provenance = row.get("provenance")
            if not isinstance(provenance, Mapping):
                raise FreezeError(f"sample {sample_id} has no provenance object")
            family = provenance.get("anomaly_family")
            generator_hash = provenance.get("generator_sha256")
            if not isinstance(family, str):
                raise FreezeError(f"sample {sample_id} has no provenance anomaly_family")
            generator_hash = _require_sha256(
                f"sample {sample_id} generator hash", str(generator_hash or "")
            )
            generators.setdefault(family, set()).add(generator_hash)
            raw_controller = provenance.get("controller_hashes")
            if not isinstance(raw_controller, Mapping):
                raise FreezeError(f"sample {sample_id} has no controller_hashes")
            for name, value in raw_controller.items():
                controllers.setdefault(str(name), set()).add(
                    _require_sha256(f"sample {sample_id} controller hash {name}", str(value))
                )

        non_unique_generators = {key: sorted(value) for key, value in generators.items() if len(value) != 1}
        non_unique_controllers = {key: sorted(value) for key, value in controllers.items() if len(value) != 1}
        if non_unique_generators or non_unique_controllers:
            raise FreezeError(
                f"inconsistent provenance hashes in {split}: "
                f"generators={non_unique_generators}, controllers={non_unique_controllers}"
            )
        generator_map = {key: next(iter(value)) for key, value in sorted(generators.items())}
        controller_map = {key: next(iter(value)) for key, value in sorted(controllers.items())}
        setup_sets[split] = setups
        image_sets[split] = set(image_hashes)
        controller_sets[split] = controller_map
        snapshot[split] = {
            "path": _display_path(root, path),
            "sha256": _sha256_file(path),
            "records": len(records),
            "setup_hash_count": len(setups),
            "counterfactual_pair_count": len(pairs),
            "episode_hash_count": len(episodes),
            "image_hash_source": (
                "actual_current_image_bytes_verified_against_"
                "manifest_assets_current_image_sha256"
            ),
            "actual_image_byte_count": len(images),
            "images": sorted(images, key=lambda item: item["sample_id"]),
            "image_hashes": sorted(image_hashes),
            "ordered_image_hash_list_sha256": _canonical_sha256(image_hashes),
            "unique_sorted_image_hash_set_sha256": _canonical_sha256(sorted(set(image_hashes))),
            "generator_hashes": generator_map,
            "controller_hashes": controller_map,
        }

    for index, left in enumerate(EXPECTED_SPLITS):
        for right in EXPECTED_SPLITS[index + 1 :]:
            setup_overlap = setup_sets[left] & setup_sets[right]
            image_overlap = image_sets[left] & image_sets[right]
            if setup_overlap:
                raise FreezeError(f"setup overlap between {left} and {right}: {len(setup_overlap)}")
            if image_overlap:
                raise FreezeError(f"image-hash overlap between {left} and {right}: {len(image_overlap)}")
    return snapshot, setup_sets, image_sets, controller_sets


def _manifest_file_entries(manifests: Mapping[str, Any]) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    for split, manifest in manifests.items():
        files.append(
            {
                "role": f"manifest:{split}",
                "path": manifest["path"],
                "hash_algorithm": "sha256_file_v1",
                "sha256": manifest["sha256"],
            }
        )
        for image in manifest["images"]:
            files.append(
                {
                    "role": f"manifest_image:{split}:{image['sample_id']}",
                    "path": image["path"],
                    "hash_algorithm": image["hash_algorithm"],
                    "sha256": image["sha256"],
                }
            )
    return files


def _file_entry(root: Path, path: Path, algorithm: str, digest: str, role: str) -> dict[str, str]:
    return {
        "role": role,
        "path": _display_path(root, path),
        "hash_algorithm": algorithm,
        "sha256": digest,
    }


def _verify_config_file_registry(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    output: dict[str, Any] = {}
    files: list[dict[str, str]] = []

    immutable = _require_mapping(config, "immutable_artifacts")
    output["immutable_artifacts"] = {}
    for name, raw in immutable.items():
        if not isinstance(raw, Mapping):
            raise FreezeError(f"immutable_artifacts.{name} must be an object")
        if "path" not in raw:
            provenance_hash = _require_sha256(
                f"immutable_artifacts.{name}.provenance_sha256",
                str(raw.get("provenance_sha256", "")),
            )
            output["immutable_artifacts"][name] = {
                "provenance_sha256": provenance_hash,
                "verification": raw.get("verification"),
            }
            continue
        path = _resolve(root, str(raw["path"]))
        algorithm, digest = _verify_expected_hash(
            f"immutable_artifacts.{name}.sha256", path, str(raw.get("sha256", ""))
        )
        output["immutable_artifacts"][name] = _file_entry(root, path, algorithm, digest, name)
        files.append(_file_entry(root, path, algorithm, digest, f"immutable:{name}"))

    matrix = _require_mapping(config, "comparison_matrix")
    b_arm = _require_mapping(matrix, "B")
    specialists = _require_mapping(b_arm, "fixed_specialists")
    output["fixed_small_image_specialists"] = {}
    for name, raw in specialists.items():
        if not isinstance(raw, Mapping):
            raise FreezeError(f"comparison_matrix.B.fixed_specialists.{name} must be an object")
        path = _resolve(root, str(raw.get("path", "")))
        algorithm, digest = _verify_expected_hash(
            f"comparison_matrix.B.fixed_specialists.{name}.sha256",
            path,
            str(raw.get("sha256", "")),
        )
        output["fixed_small_image_specialists"][name] = _file_entry(
            root, path, algorithm, digest, name
        )
        files.append(_file_entry(root, path, algorithm, digest, f"small_image:{name}"))

    closed = _require_mapping(config, "closed_loop_evaluation")
    suites = _require_mapping(closed, "suite_registry")
    output["closed_loop_suites"] = {}
    for name, raw in suites.items():
        if not isinstance(raw, Mapping):
            raise FreezeError(f"closed_loop_evaluation.suite_registry.{name} must be an object")
        path = _resolve(root, str(raw.get("path", "")))
        algorithm, digest = _verify_expected_hash(
            f"closed_loop_evaluation.suite_registry.{name}.sha256",
            path,
            str(raw.get("sha256", "")),
        )
        output["closed_loop_suites"][name] = _file_entry(root, path, algorithm, digest, name)
        files.append(_file_entry(root, path, algorithm, digest, f"closed_loop_suite:{name}"))
    return output, files


def _validate_manifest_provenance(
    config_snapshot: Mapping[str, Any], manifests: Mapping[str, Any]
) -> None:
    immutable = config_snapshot["immutable_artifacts"]
    expected_generators = {
        "sensor_saturation": immutable["saturation_generator_behavior"]["provenance_sha256"],
        "secondary_reflection": immutable["width_relative_reflection_generator"]["provenance_sha256"],
    }
    expected_controllers = {
        "forward_ensemble": immutable["forward_ensemble"]["sha256"],
        "v12_config": immutable["corrected_v12_config"]["sha256"],
        "v13_config": immutable["v13_config"]["sha256"],
        "visual_controller": immutable["visual_controller_runtime"]["sha256"],
    }
    for split, manifest in manifests.items():
        for family, digest in manifest["generator_hashes"].items():
            if family not in expected_generators or digest != expected_generators[family]:
                raise FreezeError(f"{split} has unexpected generator provenance {family}={digest}")
        for name, expected in expected_controllers.items():
            actual = manifest["controller_hashes"].get(name)
            if actual != expected:
                raise FreezeError(
                    f"{split} controller provenance {name} mismatch: expected {expected}, got {actual}"
                )


def _collect_seed_candidates(value: Any, path: tuple[str, ...] = ()) -> list[tuple[str, list[int]]]:
    candidates: list[tuple[str, list[int]]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = path + (str(key),)
            lower = str(key).lower()
            if lower in {"training_seeds", "train_seeds"} and isinstance(child, list):
                if all(isinstance(item, int) and not isinstance(item, bool) for item in child):
                    candidates.append((".".join(child_path), list(child)))
            elif lower == "seeds" and "training" in ".".join(path).lower() and isinstance(child, list):
                if all(isinstance(item, int) and not isinstance(item, bool) for item in child):
                    candidates.append((".".join(child_path), list(child)))
            candidates.extend(_collect_seed_candidates(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            candidates.extend(_collect_seed_candidates(child, path + (str(index),)))
    return candidates


def _extract_training_seeds(training_config: Path) -> tuple[str, list[int]]:
    suffix = training_config.suffix.lower()
    if suffix == ".json":
        content = _load_json(training_config)
    else:
        content = _load_yaml(training_config)
    candidates = _collect_seed_candidates(content)
    unique = {(path, tuple(seeds)) for path, seeds in candidates if seeds}
    if not unique and isinstance(content, Mapping):
        training = content.get("training")
        if isinstance(training, Mapping):
            seed = training.get("seed")
            if isinstance(seed, int) and not isinstance(seed, bool):
                unique.add(("training.seed", (seed,)))
    if len(unique) != 1:
        raise FreezeError(
            "hashed training config must declare exactly one nonempty training_seeds list "
            "(a sole training.seed is accepted as a one-seed schedule); "
            f"found {sorted(unique)}"
        )
    path, seeds_tuple = next(iter(unique))
    seeds = list(seeds_tuple)
    if len(seeds) != len(set(seeds)):
        raise FreezeError("training seed list contains duplicates")
    return path, seeds


def _require_nonplaceholder_command(name: str, value: str) -> str:
    command = str(value).strip()
    if not command:
        raise FreezeError(f"{name} must be a concrete future formal-evaluation command")
    upper = command.upper()
    if "REQUIRED_" in upper or "TODO" in upper or "PLACEHOLDER" in upper:
        raise FreezeError(f"{name} is still a placeholder")
    return command


def _configured_training_identity(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[Path, str, str, str, list[int]]:
    finalization = _require_mapping(config, "finalization")
    raw_path = finalization.get("training_config_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise FreezeError("finalization.training_config_path must be a nonempty path")
    path = _resolve(root, raw_path)
    algorithm, digest = _verify_expected_hash(
        "finalization.training_config_sha256",
        path,
        str(finalization.get("training_config_sha256", "")),
    )
    seed_path, seeds = _extract_training_seeds(path)
    if seeds != list(finalization.get("training_seeds", [])):
        raise FreezeError(
            "training seed schedule differs from finalization.training_seeds in the evaluation protocol"
        )
    return path, algorithm, digest, seed_path, seeds


def _runtime_file_entries(root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    runtime_paths = {
        "protocol_freeze_utility": Path(__file__).resolve(),
        "evaluation_protocol_document": root / "qwen_vl_supervisor_v1/evaluation_protocol.md",
        "offline_evaluator": root / "qwen_vl_supervisor_v1/evaluate_offline.py",
        "supervisor_contract": root / "qwen_vl_supervisor_v1/contracts.py",
        "development_prediction_generator": root / "qwen_vl_supervisor_v1/generate.py",
        "closed_loop_adapter": root / "qwen_vl_supervisor_v1/closed_loop_adapter.py",
        "server_training_entrypoint": root / "qwen_vl_supervisor_v1/train_qlora.py",
        "model_snapshot_identity": root / "qwen_vl_supervisor_v1/model_snapshot.py",
        "resume_replay_comparator": root
        / "qwen_vl_supervisor_v1/compare_resume_replays.py",
        "resume_replay_proof": root
        / (
            "qwen_vl_supervisor_v1/artifacts/training/"
            "resume_replay_comparison.step11.deterministic_pinned.json"
        ),
    }
    identities: dict[str, Any] = {}
    files: list[dict[str, str]] = []
    for name, raw_path in runtime_paths.items():
        path = _require_inside_root(root, raw_path, name=f"runtime {name}")
        algorithm, digest = _hash_path(path)
        identities[name] = _file_entry(root, path, algorithm, digest, name)
        files.append(_file_entry(root, path, algorithm, digest, f"runtime:{name}"))
    return identities, files


def _training_data_file_entries(
    root: Path,
    training_config_path: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    training_config = _load_yaml(training_config_path)
    data = _require_mapping(training_config, "data")
    specs = {
        "train_jsonl": ("train_jsonl", "train_sha256"),
        "dev_jsonl": ("dev_jsonl", "dev_sha256"),
        "export_report": ("export_report", "export_report_sha256"),
    }
    identities: dict[str, Any] = {}
    files: list[dict[str, str]] = []
    for name, (path_key, hash_key) in specs.items():
        raw_path = data.get(path_key)
        if not isinstance(raw_path, str) or not raw_path:
            raise FreezeError(f"training config data.{path_key} must be a nonempty path")
        path = _resolve(root, raw_path)
        algorithm, digest = _verify_expected_hash(
            f"training config data.{hash_key}",
            path,
            str(data.get(hash_key, "")),
        )
        identities[name] = _file_entry(root, path, algorithm, digest, name)
        files.append(_file_entry(root, path, algorithm, digest, f"training_data:{name}"))
    return identities, files


def _verify_file_entries(
    root: Path,
    raw_files: Any,
    *,
    registry_name: str,
) -> list[dict[str, str]]:
    if not isinstance(raw_files, list) or not raw_files:
        raise FreezeError(f"{registry_name} is empty or missing")
    checked: list[dict[str, str]] = []
    seen_roles: set[str] = set()
    for index, entry in enumerate(raw_files):
        if not isinstance(entry, Mapping):
            raise FreezeError(f"invalid {registry_name} entry {index}")
        role = entry.get("role")
        raw_path = entry.get("path")
        expected_algorithm = entry.get("hash_algorithm")
        expected_digest = entry.get("sha256")
        if not isinstance(role, str) or not role or role in seen_roles:
            raise FreezeError(f"invalid or duplicate role in {registry_name}: {role!r}")
        if not isinstance(raw_path, str) or not raw_path:
            raise FreezeError(f"{registry_name} role {role} has no path")
        if expected_algorithm not in {
            "sha256_file_v1",
            "sha256_tree_relative_path_nul_file_sha256_lf_v1",
        }:
            raise FreezeError(
                f"{registry_name} role {role} has unsupported hash algorithm "
                f"{expected_algorithm!r}"
            )
        expected_digest = _require_sha256(
            f"{registry_name} role {role} SHA-256",
            str(expected_digest or ""),
        )
        path = _resolve(root, raw_path)
        algorithm, digest = _hash_path(path)
        if algorithm != expected_algorithm or digest != expected_digest:
            raise FreezeError(f"frozen file drift for {role}: {path}")
        seen_roles.add(role)
        checked.append(
            {
                "role": role,
                "path": raw_path,
                "hash_algorithm": str(expected_algorithm),
                "sha256": expected_digest,
            }
        )
    return checked


def _write_new_json_artifact(
    root: Path,
    output_value: str | Path,
    artifact: Mapping[str, Any],
    *,
    description: str,
) -> tuple[Path, str]:
    output = _resolve(root, output_value)
    if output.exists():
        raise FreezeError(f"refusing to overwrite {description}: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        artifact,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    output.write_text(serialized, encoding="utf-8")
    return output, _sha256_file(output)


def _prepare(args: argparse.Namespace) -> int:
    if not args.authorize_pre_server_protocol_freeze:
        raise FreezeError(
            "refusing to prepare without --authorize-pre-server-protocol-freeze"
        )

    root = args.repository_root.resolve()
    config_path = _resolve(root, args.config)
    config = _load_yaml(config_path)
    _validate_protocol(config)
    config_algorithm, config_hash = _hash_path(config_path)

    training_path, training_algorithm, training_hash, seed_path, training_seeds = (
        _configured_training_identity(root, config)
    )
    manifests, _, _, _ = _manifest_snapshot(root, _require_mapping(config, "manifests"))
    config_snapshot, immutable_files = _verify_config_file_registry(root, config)
    _validate_manifest_provenance(config_snapshot, manifests)
    runtime_identities, runtime_files = _runtime_file_entries(root)
    training_data_identities, training_data_files = _training_data_file_entries(
        root,
        training_path,
    )

    files_to_verify = [
        _file_entry(
            root,
            config_path,
            config_algorithm,
            config_hash,
            "evaluation_protocol_config",
        ),
        _file_entry(
            root,
            training_path,
            training_algorithm,
            training_hash,
            "training_config",
        ),
        *immutable_files,
        *_manifest_file_entries(manifests),
        *runtime_files,
        *training_data_files,
    ]
    # Role uniqueness makes registry substitution/ambiguity a hard error both
    # now and when this artifact is consumed by the later final freeze.
    files_to_verify = sorted(files_to_verify, key=lambda row: (row["role"], row["path"]))
    if len({row["role"] for row in files_to_verify}) != len(files_to_verify):
        raise FreezeError("pre-server files-to-verify registry has duplicate roles")

    closed = _require_mapping(config, "closed_loop_evaluation")
    offline = _require_mapping(config, "offline_evaluation")
    finalization = _require_mapping(config, "finalization")
    artifact = {
        "schema_version": "qwen_vl_supervisor_evaluation_protocol_freeze_v1.0.0",
        "artifact_semantics": (
            "deterministic_pre_server_no_timestamp_no_predictions_opened_"
            "actual_image_bytes_hashed"
        ),
        "protocol_commitment": {
            "sealed_before_server_training": True,
            "required_by_later_final_freeze": True,
            "formal_evaluation_executed_by_this_utility": False,
            "prediction_files_opened_by_this_utility": False,
            "model_or_controller_executed_by_this_utility": False,
        },
        "protocol_config": {
            "path": _display_path(root, config_path),
            "hash_algorithm": config_algorithm,
            "byte_sha256": config_hash,
            "canonical_content_sha256": _canonical_sha256(config),
        },
        "training_config": {
            **_file_entry(
                root,
                training_path,
                training_algorithm,
                training_hash,
                "training_config",
            ),
            "training_seed_field": seed_path,
            "training_seeds": training_seeds,
        },
        "training_data_identities": training_data_identities,
        "manifests": manifests,
        "source_generator_controller_baseline_and_suite_identities": config_snapshot,
        "evaluation_runtime_identities": runtime_identities,
        "evaluation_seeds": {
            "training_seeds": training_seeds,
            **dict(_require_mapping(closed, "evaluation_seeds")),
        },
        "scope_and_split_rules": {
            "scope_guard": config["scope_guard"],
            "manifests": config["manifests"],
        },
        "comparison_matrix": config["comparison_matrix"],
        "required_metrics": {
            "offline": offline["required_metrics"],
            "closed_loop": closed["required_metrics"],
        },
        "subgroup_definitions": config["subgroups"],
        "offline_rules": {
            "splits": offline["splits"],
            "primary_split": offline["primary_split"],
            "prediction_join_key": offline["prediction_join_key"],
            "invalid_or_missing_predictions": offline[
                "invalid_or_missing_predictions"
            ],
            "denominators": offline["denominators"],
            "training_seed_reporting": offline["training_seed_reporting"],
            "arm_seed_semantics": offline["offline_arm_seed_semantics"],
        },
        "closed_loop_rules": {
            "arms": closed["arms"],
            "paired_invariants": closed["paired_invariants"],
            "controller": closed["controller"],
            "reporting": closed["reporting"],
        },
        "selection_rules": {
            "checkpoint": finalization["checkpoint_selection_rule"],
            "threshold": finalization["threshold_selection_rule"],
        },
        "primary_research_tests_and_pass_fail_rules": config["primary_research_tests"],
        "known_reflection_limitations": config["known_reflection_limitations"],
        "continue_stop_masking": offline["continue_stop_masking"],
        "exact_commands": config["exact_commands"],
        "later_final_freeze_requirements": {
            "this_preparation_artifact_and_external_sha256": "required",
            "final_checkpoint_bundle_and_sha256": "required",
            "dev_only_selection_artifact_and_sha256": "required",
            "baseline_A_B_dev_selection_artifact_and_sha256": "required",
            "concrete_prediction_runner_command_and_sha256": "required",
            "concrete_closed_loop_runner_command_and_sha256": "required",
        },
        "files_to_verify_before_final_freeze": files_to_verify,
    }
    output, digest = _write_new_json_artifact(
        root,
        args.output,
        artifact,
        description="pre-server evaluation protocol freeze artifact",
    )
    print(
        json.dumps(
            {
                "status": "PRE_SERVER_EVALUATION_PROTOCOL_FROZEN_NO_EVALUATION_RUN",
                "output": _display_path(root, output),
                "sha256": digest,
                "training_seeds": training_seeds,
                "manifest_image_bytes_verified": sum(
                    int(manifest["actual_image_byte_count"])
                    for manifest in manifests.values()
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _load_verified_preparation(
    root: Path,
    path_value: str | Path,
    expected_sha256: str,
) -> tuple[Path, str, Mapping[str, Any], list[dict[str, str]]]:
    path = _resolve(root, path_value)
    algorithm, digest = _verify_expected_hash(
        "pre-server protocol freeze artifact SHA-256",
        path,
        expected_sha256,
    )
    if algorithm != "sha256_file_v1":
        raise FreezeError("pre-server protocol freeze artifact must be a regular JSON file")
    artifact = _load_json(path)
    if not isinstance(artifact, Mapping):
        raise FreezeError("pre-server protocol freeze artifact must be a JSON object")
    if artifact.get("schema_version") != (
        "qwen_vl_supervisor_evaluation_protocol_freeze_v1.0.0"
    ):
        raise FreezeError("unsupported pre-server protocol freeze artifact schema")
    commitment = artifact.get("protocol_commitment")
    if (
        not isinstance(commitment, Mapping)
        or commitment.get("sealed_before_server_training") is not True
        or commitment.get("required_by_later_final_freeze") is not True
        or commitment.get("formal_evaluation_executed_by_this_utility") is not False
        or commitment.get("prediction_files_opened_by_this_utility") is not False
    ):
        raise FreezeError("pre-server protocol artifact has no valid sealed commitment")
    files = _verify_file_entries(
        root,
        artifact.get("files_to_verify_before_final_freeze"),
        registry_name="pre-server files-to-verify registry",
    )
    return path, digest, artifact, files


def _validate_dev_selection_for_freeze(
    root: Path,
    *,
    artifact: Mapping[str, Any],
    config: Mapping[str, Any],
    preparation: Mapping[str, Any],
    training_config: Path,
    training_algorithm: str,
    training_hash: str,
    training_seeds: Sequence[int],
    final_checkpoint_root: Path,
) -> list[dict[str, Any]]:
    """Validate the dev-selection schema and bind its winners to the final root."""

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from qwen_vl_supervisor_v1.select_dev_checkpoints import (
            validate_selection_artifact,
        )

        validate_selection_artifact(artifact)
    except Exception as exc:
        raise FreezeError(f"dev selection artifact failed strict validation: {exc}") from exc

    if artifact.get("training_seeds") != list(training_seeds):
        raise FreezeError("dev selection artifact seeds differ from the frozen training seeds")
    if artifact.get("protected_or_frozen_data_used") is not False:
        raise FreezeError("dev selection artifact reports protected/frozen data use")
    if artifact.get("frozen_predictions_opened") is not False:
        raise FreezeError("dev selection artifact reports frozen prediction access")
    coverage = artifact.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("complete") is not True:
        raise FreezeError("dev selection artifact does not prove complete checkpoint coverage")

    selected_training = artifact.get("training_config")
    expected_training = {
        "path": _display_path(root, training_config),
        "hash_algorithm": training_algorithm,
        "sha256": training_hash,
    }
    if selected_training != expected_training:
        raise FreezeError("dev selection artifact used a different training config")

    prepared_training_data = preparation.get("training_data_identities")
    prepared_manifests = preparation.get("manifests")
    dev_data = artifact.get("dev_data")
    if (
        not isinstance(prepared_training_data, Mapping)
        or not isinstance(prepared_manifests, Mapping)
        or not isinstance(dev_data, Mapping)
    ):
        raise FreezeError("dev selection or pre-server artifact omits dev-data identities")
    expected_dev_sft = prepared_training_data.get("dev_jsonl")
    prepared_dev_manifest = prepared_manifests.get("dev")
    if not isinstance(prepared_dev_manifest, Mapping):
        raise FreezeError("pre-server artifact omits the dev manifest identity")
    expected_dev_manifest = {
        "path": prepared_dev_manifest.get("path"),
        "hash_algorithm": "sha256_file_v1",
        "sha256": prepared_dev_manifest.get("sha256"),
    }
    if dev_data.get("sft") != expected_dev_sft:
        raise FreezeError("dev selection artifact used a different dev SFT export")
    if dev_data.get("source_manifest") != expected_dev_manifest:
        raise FreezeError("dev selection artifact used a different dev source manifest")

    immutable = _require_mapping(config, "immutable_artifacts")
    selector = artifact.get("selector")
    if not isinstance(selector, Mapping):
        raise FreezeError("dev selection artifact omits selector identities")
    expected_selector_identities = {
        "implementation": "dev_checkpoint_selector",
        "json_schema": "dev_selection_artifact_schema",
    }
    for artifact_field, config_name in expected_selector_identities.items():
        configured = immutable.get(config_name)
        if not isinstance(configured, Mapping):
            raise FreezeError(f"evaluation config omits immutable {config_name}")
        expected = {
            "path": _display_path(root, _resolve(root, str(configured.get("path", "")))),
            "hash_algorithm": "sha256_file_v1",
            "sha256": _require_sha256(
                f"immutable_artifacts.{config_name}.sha256",
                str(configured.get("sha256", "")),
            ),
        }
        if selector.get(artifact_field) != expected:
            raise FreezeError(
                f"dev selection artifact {artifact_field} is not the pre-server-pinned artifact"
            )

    if not final_checkpoint_root.is_dir():
        raise FreezeError("final checkpoint bundle/root must be a directory")
    resolved_final_root = final_checkpoint_root.resolve()
    seed_results = artifact.get("seed_results")
    if not isinstance(seed_results, list) or len(seed_results) != len(training_seeds):
        raise FreezeError("dev selection artifact does not select exactly one checkpoint per seed")
    links: list[dict[str, Any]] = []
    for expected_seed, result in zip(training_seeds, seed_results, strict=True):
        if not isinstance(result, Mapping) or result.get("training_seed") != expected_seed:
            raise FreezeError("dev selection results do not follow the frozen seed order")
        selected = result.get("selected")
        checkpoint_entry = selected.get("checkpoint") if isinstance(selected, Mapping) else None
        if not isinstance(checkpoint_entry, Mapping):
            raise FreezeError(f"seed {expected_seed} has no selected checkpoint identity")
        raw_selected_path = checkpoint_entry.get("path")
        if not isinstance(raw_selected_path, str) or not raw_selected_path:
            raise FreezeError(f"seed {expected_seed} selected checkpoint has no path")
        selected_path = _resolve(root, raw_selected_path)
        try:
            relative = selected_path.relative_to(resolved_final_root)
        except ValueError as exc:
            raise FreezeError(
                f"seed {expected_seed} selected checkpoint escapes final checkpoint root: "
                f"{selected_path}"
            ) from exc
        algorithm, digest = _hash_path(selected_path)
        expected_digest = _require_sha256(
            f"seed {expected_seed} selected checkpoint SHA-256",
            str(checkpoint_entry.get("sha256", "")),
        )
        if (
            checkpoint_entry.get("hash_algorithm")
            != "sha256_tree_relative_path_nul_file_sha256_lf_v1"
            or algorithm != checkpoint_entry.get("hash_algorithm")
            or digest != expected_digest
        ):
            raise FreezeError(
                f"seed {expected_seed} selected checkpoint tree differs from dev selection"
            )
        links.append(
            {
                "training_seed": expected_seed,
                "optimizer_step": selected.get("optimizer_step"),
                "selection_rank": selected.get("selection_rank"),
                "selected_checkpoint_path": _display_path(root, selected_path),
                "relative_path_within_final_checkpoint_root": relative.as_posix(),
                "hash_algorithm": algorithm,
                "sha256": digest,
            }
        )
    return links


def _validate_baseline_selection_for_freeze(
    root: Path,
    *,
    artifact: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from qwen_vl_supervisor_v1.validate_baseline_dev_selection import (
            validate_baseline_selection_artifact,
        )

        result = validate_baseline_selection_artifact(
            artifact,
            repository_root=root,
            evaluation_config=config,
        )
    except Exception as exc:
        raise FreezeError(
            f"baseline dev-selection artifact failed exact validation: {exc}"
        ) from exc
    if not isinstance(result, dict):
        raise FreezeError("baseline dev-selection validator returned no audit result")
    if (
        result.get("selection_split") != "dev"
        or result.get("protected_or_frozen_data_used") is not False
        or result.get("frozen_predictions_opened") is not False
        or result.get("verification") != "passed_exact_raw_evidence_recomputation"
    ):
        raise FreezeError("baseline dev-selection audit result is not a clean dev-only pass")
    return result


def _freeze(args: argparse.Namespace) -> int:
    if not args.authorize_formal_evaluation_freeze:
        raise FreezeError("refusing to freeze without --authorize-formal-evaluation-freeze")

    root = args.repository_root.resolve()
    (
        preparation_path,
        preparation_hash,
        preparation,
        prepared_files,
    ) = _load_verified_preparation(
        root,
        args.preparation_artifact,
        args.preparation_artifact_sha256,
    )
    config_path = _resolve(root, args.config)
    config = _load_yaml(config_path)
    _validate_protocol(config)
    config_algorithm, config_hash = _hash_path(config_path)
    prepared_protocol = preparation.get("protocol_config")
    if not isinstance(prepared_protocol, Mapping):
        raise FreezeError("pre-server artifact omits protocol_config")
    if (
        prepared_protocol.get("path") != _display_path(root, config_path)
        or prepared_protocol.get("hash_algorithm") != config_algorithm
        or prepared_protocol.get("byte_sha256") != config_hash
        or prepared_protocol.get("canonical_content_sha256") != _canonical_sha256(config)
    ):
        raise FreezeError(
            "evaluation config is not the exact config sealed by the pre-server artifact"
        )

    checkpoint = _resolve(root, args.final_checkpoint)
    checkpoint_algorithm, checkpoint_hash = _verify_expected_hash(
        "final checkpoint SHA-256", checkpoint, args.final_checkpoint_sha256
    )
    training_config = _resolve(root, args.training_config)
    training_algorithm, training_hash = _verify_expected_hash(
        "training config SHA-256", training_config, args.training_config_sha256
    )
    finalization = _require_mapping(config, "finalization")
    configured_training_path = str(finalization.get("training_config_path"))
    if _resolve(root, configured_training_path) != training_config:
        raise FreezeError("supplied training config does not match finalization.training_config_path")
    pinned_training_hash = _require_sha256(
        "finalization.training_config_sha256",
        str(finalization.get("training_config_sha256", "")),
    )
    if training_hash != pinned_training_hash:
        raise FreezeError(
            "supplied training config differs from the SHA-256 pinned in the evaluation protocol"
        )
    seed_path, training_seeds = _extract_training_seeds(training_config)
    if training_seeds != list(finalization.get("training_seeds", [])):
        raise FreezeError(
            "training seed schedule differs from finalization.training_seeds in the evaluation protocol"
        )
    prepared_training = preparation.get("training_config")
    if not isinstance(prepared_training, Mapping):
        raise FreezeError("pre-server artifact omits training_config")
    if (
        prepared_training.get("path") != _display_path(root, training_config)
        or prepared_training.get("hash_algorithm") != training_algorithm
        or prepared_training.get("sha256") != training_hash
        or prepared_training.get("training_seed_field") != seed_path
        or prepared_training.get("training_seeds") != training_seeds
    ):
        raise FreezeError(
            "supplied training config is not the config sealed before server training"
        )

    selection_artifact = _resolve(root, args.dev_selection_artifact)
    selection_algorithm, selection_hash = _verify_expected_hash(
        "dev selection artifact SHA-256",
        selection_artifact,
        args.dev_selection_artifact_sha256,
    )
    raw_selection = _load_json(selection_artifact)
    if not isinstance(raw_selection, Mapping):
        raise FreezeError("dev selection artifact must be one strict JSON object")
    selected_checkpoint_links = _validate_dev_selection_for_freeze(
        root,
        artifact=raw_selection,
        config=config,
        preparation=preparation,
        training_config=training_config,
        training_algorithm=training_algorithm,
        training_hash=training_hash,
        training_seeds=training_seeds,
        final_checkpoint_root=checkpoint,
    )
    baseline_selection_artifact = _resolve(root, args.baseline_dev_selection_artifact)
    baseline_selection_algorithm, baseline_selection_hash = _verify_expected_hash(
        "baseline dev-selection artifact SHA-256",
        baseline_selection_artifact,
        args.baseline_dev_selection_artifact_sha256,
    )
    raw_baseline_selection = _load_json(baseline_selection_artifact)
    if not isinstance(raw_baseline_selection, Mapping):
        raise FreezeError("baseline dev-selection artifact must be one strict JSON object")
    baseline_selection_validation = _validate_baseline_selection_for_freeze(
        root,
        artifact=raw_baseline_selection,
        config=config,
    )
    prediction_runner = _resolve(root, args.prediction_runner)
    prediction_algorithm, prediction_hash = _verify_expected_hash(
        "prediction runner SHA-256", prediction_runner, args.prediction_runner_sha256
    )
    closed_loop_runner = _resolve(root, args.closed_loop_runner)
    closed_loop_algorithm, closed_loop_hash = _verify_expected_hash(
        "closed-loop runner SHA-256", closed_loop_runner, args.closed_loop_runner_sha256
    )
    prediction_command = _require_nonplaceholder_command(
        "prediction command", args.prediction_command
    )
    closed_loop_command = _require_nonplaceholder_command(
        "closed-loop command", args.closed_loop_command
    )

    manifests = preparation.get("manifests")
    config_snapshot = preparation.get(
        "source_generator_controller_baseline_and_suite_identities"
    )
    if not isinstance(manifests, Mapping) or tuple(manifests.keys()) != EXPECTED_SPLITS:
        raise FreezeError("pre-server artifact omits the four frozen manifest snapshots")
    if not isinstance(config_snapshot, Mapping):
        raise FreezeError("pre-server artifact omits source/baseline identities")

    files_to_verify = [
        *prepared_files,
        _file_entry(
            root,
            preparation_path,
            "sha256_file_v1",
            preparation_hash,
            "pre_server_protocol_freeze_artifact",
        ),
        _file_entry(root, checkpoint, checkpoint_algorithm, checkpoint_hash, "final_checkpoint"),
        _file_entry(root, selection_artifact, selection_algorithm, selection_hash, "dev_selection_artifact"),
        _file_entry(
            root,
            baseline_selection_artifact,
            baseline_selection_algorithm,
            baseline_selection_hash,
            "baseline_dev_selection_artifact",
        ),
        _file_entry(root, prediction_runner, prediction_algorithm, prediction_hash, "prediction_runner"),
        _file_entry(root, closed_loop_runner, closed_loop_algorithm, closed_loop_hash, "closed_loop_runner"),
    ]
    if len({row["role"] for row in files_to_verify}) != len(files_to_verify):
        raise FreezeError("final files-to-verify registry has duplicate roles")

    closed = _require_mapping(config, "closed_loop_evaluation")
    offline = _require_mapping(config, "offline_evaluation")
    exact_commands = dict(_require_mapping(config, "exact_commands"))
    exact_commands.pop("prediction_and_closed_loop_runner", None)
    exact_commands["prediction_generation"] = prediction_command
    exact_commands["closed_loop_evaluation"] = closed_loop_command
    artifact = {
        "schema_version": "qwen_vl_supervisor_evaluation_freeze_v1.1.0",
        "artifact_semantics": (
            "deterministic_no_timestamp_no_predictions_opened_"
            "requires_pre_server_protocol_commitment"
        ),
        "protocol_config": {
            "path": _display_path(root, config_path),
            "byte_sha256": config_hash,
            "canonical_content_sha256": _canonical_sha256(config),
        },
        "formal_gate": {
            "sealed": True,
            "formal_evaluation_executed_by_this_utility": False,
            "prediction_files_opened_by_this_utility": False,
            "required_preflight": "verify_subcommand_with_--formal-evaluation",
        },
        "pre_server_protocol_freeze_artifact": _file_entry(
            root,
            preparation_path,
            "sha256_file_v1",
            preparation_hash,
            "pre_server_protocol_freeze_artifact",
        ),
        "final_checkpoint": _file_entry(
            root, checkpoint, checkpoint_algorithm, checkpoint_hash, "final_checkpoint"
        ),
        "training_config": {
            **_file_entry(root, training_config, training_algorithm, training_hash, "training_config"),
            "training_seed_field": seed_path,
            "training_seeds": training_seeds,
        },
        "dev_only_selection_artifact": _file_entry(
            root, selection_artifact, selection_algorithm, selection_hash, "dev_selection_artifact"
        ),
        "baseline_dev_selection_artifact": _file_entry(
            root,
            baseline_selection_artifact,
            baseline_selection_algorithm,
            baseline_selection_hash,
            "baseline_dev_selection_artifact",
        ),
        "baseline_dev_selection_validation": baseline_selection_validation,
        "selected_checkpoint_bundle_linkage": {
            "final_checkpoint_root": _file_entry(
                root,
                checkpoint,
                checkpoint_algorithm,
                checkpoint_hash,
                "final_checkpoint",
            ),
            "one_selected_checkpoint_per_training_seed": True,
            "selected_checkpoints": selected_checkpoint_links,
        },
        "manifests": manifests,
        "source_and_baseline_identities": config_snapshot,
        "evaluation_seeds": closed["evaluation_seeds"],
        "comparison_matrix": config["comparison_matrix"],
        "offline_metrics": offline["required_metrics"],
        "closed_loop_metrics": closed["required_metrics"],
        "subgroup_definitions": config["subgroups"],
        "checkpoint_selection_rule": config["finalization"]["checkpoint_selection_rule"],
        "threshold_selection_rule": config["finalization"]["threshold_selection_rule"],
        "primary_research_tests_and_pass_fail_rules": config["primary_research_tests"],
        "known_reflection_limitations": config["known_reflection_limitations"],
        "continue_stop_masking": offline["continue_stop_masking"],
        "exact_commands": exact_commands,
        "runner_identities": {
            "prediction": _file_entry(
                root, prediction_runner, prediction_algorithm, prediction_hash, "prediction_runner"
            ),
            "closed_loop": _file_entry(
                root, closed_loop_runner, closed_loop_algorithm, closed_loop_hash, "closed_loop_runner"
            ),
        },
        "files_to_verify_before_formal_evaluation": sorted(
            files_to_verify, key=lambda row: (row["role"], row["path"])
        ),
    }

    output, output_hash = _write_new_json_artifact(
        root,
        args.output,
        artifact,
        description="evaluation freeze artifact",
    )
    print(
        json.dumps(
            {
                "status": "FORMAL_EVALUATION_FREEZE_SEALED_NO_EVALUATION_RUN",
                "output": _display_path(root, output),
                "sha256": output_hash,
                "pre_server_protocol_freeze_sha256": preparation_hash,
                "training_seeds": training_seeds,
            },
            sort_keys=True,
        )
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    if not args.formal_evaluation:
        raise FreezeError("refusing formal gate verification without --formal-evaluation")
    root = args.repository_root.resolve()
    artifact_path = _resolve(root, args.freeze_artifact)
    _, artifact_hash = _verify_expected_hash(
        "evaluation freeze artifact SHA-256",
        artifact_path,
        args.freeze_artifact_sha256,
    )
    artifact = _load_json(artifact_path)
    if not isinstance(artifact, Mapping):
        raise FreezeError("evaluation freeze artifact must be a JSON object")
    if artifact.get("schema_version") != "qwen_vl_supervisor_evaluation_freeze_v1.1.0":
        raise FreezeError("unsupported evaluation freeze artifact schema")
    gate = artifact.get("formal_gate")
    if not isinstance(gate, Mapping) or gate.get("sealed") is not True:
        raise FreezeError("formal evaluation gate is not sealed")
    preparation = artifact.get("pre_server_protocol_freeze_artifact")
    if not isinstance(preparation, Mapping):
        raise FreezeError("final freeze artifact omits the pre-server protocol commitment")

    checkpoint = _resolve(root, args.final_checkpoint)
    checkpoint_algorithm, checkpoint_hash = _verify_expected_hash(
        "final checkpoint SHA-256", checkpoint, args.final_checkpoint_sha256
    )
    training_config = _resolve(root, args.training_config)
    training_algorithm, training_hash = _verify_expected_hash(
        "training config SHA-256", training_config, args.training_config_sha256
    )
    stored_checkpoint = artifact.get("final_checkpoint")
    stored_training = artifact.get("training_config")
    if not isinstance(stored_checkpoint, Mapping) or not isinstance(stored_training, Mapping):
        raise FreezeError("freeze artifact omits checkpoint or training config")
    if (
        checkpoint_hash != stored_checkpoint.get("sha256")
        or checkpoint_algorithm != stored_checkpoint.get("hash_algorithm")
        or _display_path(root, checkpoint) != stored_checkpoint.get("path")
    ):
        raise FreezeError("supplied final checkpoint is not the frozen checkpoint")
    if (
        training_hash != stored_training.get("sha256")
        or training_algorithm != stored_training.get("hash_algorithm")
        or _display_path(root, training_config) != stored_training.get("path")
    ):
        raise FreezeError("supplied training config is not the frozen training config")

    _verify_file_entries(
        root,
        artifact.get("files_to_verify_before_formal_evaluation"),
        registry_name="formal files-to-verify registry",
    )

    print(
        json.dumps(
            {
                "status": "FORMAL_EVALUATION_GATE_VERIFIED",
                "freeze_artifact_sha256": artifact_hash,
                "final_checkpoint_sha256": checkpoint_hash,
                "training_config_sha256": training_hash,
                "note": "preflight_only_no_prediction_or_controller_execution",
            },
            sort_keys=True,
        )
    )
    return 0


def _hash_command(args: argparse.Namespace) -> int:
    """Expose the exact file/tree digest algorithm used by freeze and verify."""

    root = args.repository_root.resolve()
    path = _resolve(root, args.path)
    algorithm, digest = _hash_path(path)
    if args.digest_only:
        print(digest)
    else:
        print(
            json.dumps(
                {
                    "path": _display_path(root, path),
                    "hash_algorithm": algorithm,
                    "sha256": digest,
                },
                sort_keys=True,
            )
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root used to resolve recorded relative paths",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="seal the deterministic protocol and actual image bytes before server training",
    )
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--authorize-pre-server-protocol-freeze", action="store_true")
    prepare.set_defaults(handler=_prepare)

    freeze = subparsers.add_parser("freeze", help="seal a deterministic evaluation artifact")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--preparation-artifact", required=True)
    freeze.add_argument("--preparation-artifact-sha256", required=True)
    freeze.add_argument("--final-checkpoint", required=True)
    freeze.add_argument("--final-checkpoint-sha256", required=True)
    freeze.add_argument("--training-config", required=True)
    freeze.add_argument("--training-config-sha256", required=True)
    freeze.add_argument("--dev-selection-artifact", required=True)
    freeze.add_argument("--dev-selection-artifact-sha256", required=True)
    freeze.add_argument("--baseline-dev-selection-artifact", required=True)
    freeze.add_argument("--baseline-dev-selection-artifact-sha256", required=True)
    freeze.add_argument("--prediction-runner", required=True)
    freeze.add_argument("--prediction-runner-sha256", required=True)
    freeze.add_argument("--prediction-command", required=True)
    freeze.add_argument("--closed-loop-runner", required=True)
    freeze.add_argument("--closed-loop-runner-sha256", required=True)
    freeze.add_argument("--closed-loop-command", required=True)
    freeze.add_argument("--authorize-formal-evaluation-freeze", action="store_true")
    freeze.set_defaults(handler=_freeze)

    verify = subparsers.add_parser("verify", help="verify the sealed gate before a formal run")
    verify.add_argument("--freeze-artifact", required=True)
    verify.add_argument("--freeze-artifact-sha256", required=True)
    verify.add_argument("--final-checkpoint", required=True)
    verify.add_argument("--final-checkpoint-sha256", required=True)
    verify.add_argument("--training-config", required=True)
    verify.add_argument("--training-config-sha256", required=True)
    verify.add_argument("--formal-evaluation", action="store_true")
    verify.set_defaults(handler=_verify)

    hash_parser = subparsers.add_parser(
        "hash", help="print the exact deterministic file or checkpoint-tree hash"
    )
    hash_parser.add_argument("--path", required=True)
    hash_parser.add_argument("--digest-only", action="store_true")
    hash_parser.set_defaults(handler=_hash_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except FreezeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
