"""Integrity checks for profile2setup v2 workflow artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from profile2setup.schema import VARIABLE_ORDER, validate_dataset_record


CANONICAL_VARIABLE_ORDER = [
    "source_to_lens",
    "lens_to_camera",
    "focal_length",
    "lens_x",
    "lens_y",
    "camera_x",
    "camera_y",
]

_FORBIDDEN_ROOT = "alignment"
FORBIDDEN_KEYS = {
    _FORBIDDEN_ROOT,
    f"{_FORBIDDEN_ROOT}_x",
    f"{_FORBIDDEN_ROOT}_y",
}
FORBIDDEN_EXACT_CODE_TOKENS = {f"{_FORBIDDEN_ROOT}_x", f"{_FORBIDDEN_ROOT}_y"}
SETUP_FIELDS = ("current_setup", "target_setup", "target_delta")
CODE_SUFFIXES = {".py"}
CONFIG_SUFFIXES = {".yaml", ".yml"}


@dataclass
class CheckResult:
    name: str
    checked: int = 0
    warnings: list[str] | None = None
    errors: list[str] | None = None
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
        if self.details is None:
            self.details = {}

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check profile2setup v2 integrity")
    parser.add_argument("--root", default="profile2setup", help="profile2setup package root")
    parser.add_argument("--data-dir", default=None, help="Optional data directory containing JSONL files")
    parser.add_argument("--results-dir", default=None, help="Optional results directory containing JSON files")

    code_group = parser.add_mutually_exclusive_group()
    code_group.add_argument("--check-code", dest="check_code", action="store_true")
    code_group.add_argument("--no-check-code", dest="check_code", action="store_false")
    parser.set_defaults(check_code=True)

    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--check-data", dest="check_data", action="store_true")
    data_group.add_argument("--no-check-data", dest="check_data", action="store_false")
    parser.set_defaults(check_data=True)

    results_group = parser.add_mutually_exclusive_group()
    results_group.add_argument("--check-results", dest="check_results", action="store_true")
    results_group.add_argument("--no-check-results", dest="check_results", action="store_false")
    parser.set_defaults(check_results=True)

    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict", dest="strict", action="store_true")
    strict_group.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def _add_error(result: CheckResult, message: str) -> None:
    result.errors.append(message)


def _add_warning(result: CheckResult, message: str) -> None:
    result.warnings.append(message)


def _is_hidden_or_cache(path: Path) -> bool:
    return any(part.startswith(".") or part == "__pycache__" for part in path.parts)


def _json_location(path: Path, line_number: int | None = None) -> str:
    if line_number is None:
        return str(path)
    return f"{path}:{line_number}"


def _find_forbidden_key_paths(obj: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            path = f"{prefix}.{key_str}" if prefix else key_str
            if key_str in FORBIDDEN_KEYS:
                paths.append(path)
            paths.extend(_find_forbidden_key_paths(value, path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            paths.extend(_find_forbidden_key_paths(item, path))
    return paths


def _check_setup_field_keys(record: dict[str, Any], path: Path, line_number: int, result: CheckResult) -> None:
    for field in SETUP_FIELDS:
        value = record.get(field)
        if value is None:
            continue
        if not isinstance(value, dict):
            _add_error(
                result,
                f"{_json_location(path, line_number)} field {field} must be an object or null",
            )
            continue
        keys = list(value.keys())
        if keys != CANONICAL_VARIABLE_ORDER:
            _add_error(
                result,
                f"{_json_location(path, line_number)} field {field} must use canonical variable order "
                f"{CANONICAL_VARIABLE_ORDER}; got {keys}",
            )


def _load_json(path: Path, result: CheckResult) -> Any:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        _add_error(result, f"{path} is not valid JSON: {exc}")
        return None


def _load_yaml(path: Path, result: CheckResult) -> Any:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        _add_error(result, f"{path} is not valid YAML: {exc}")
        return None


def check_canonical_schema(root: Path) -> CheckResult:
    result = CheckResult("canonical schema", checked=1)
    if not root.exists():
        _add_error(result, f"root does not exist: {root}")
        return result
    if list(VARIABLE_ORDER) != CANONICAL_VARIABLE_ORDER:
        _add_error(
            result,
            f"profile2setup.schema.VARIABLE_ORDER must be {CANONICAL_VARIABLE_ORDER}; got {list(VARIABLE_ORDER)}",
        )
    return result


def _is_allowed_defensive_code_line(line: str) -> bool:
    lowered = line.lower()
    allowed_markers = (
        "forbidden",
        "not allowed",
        "not use",
        "does not use",
        "legacy",
        "use camera_x/camera_y",
    )
    return any(marker in lowered for marker in allowed_markers)


def check_code_files(root: Path) -> CheckResult:
    result = CheckResult("code token scan")
    if not root.exists():
        _add_warning(result, f"code root does not exist: {root}")
        return result

    for path in sorted(root.rglob("*")):
        if _is_hidden_or_cache(path) or not path.is_file() or path.suffix not in CODE_SUFFIXES:
            continue
        result.checked += 1
        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception as exc:  # noqa: BLE001
            _add_error(result, f"could not read code file {path}: {exc}")
            continue
        for line_number, line in enumerate(lines, start=1):
            hits = sorted(token for token in FORBIDDEN_EXACT_CODE_TOKENS if token in line)
            if not hits:
                continue
            if _is_allowed_defensive_code_line(line):
                continue
            _add_error(result, f"{path}:{line_number} contains forbidden v2 token(s): {', '.join(hits)}")

    return result


def _check_json_or_yaml_object(path: Path, obj: Any, result: CheckResult) -> None:
    bad_paths = _find_forbidden_key_paths(obj)
    for bad_path in bad_paths:
        _add_error(result, f"{path} contains forbidden key: {bad_path}")


def check_dataset_jsonl(data_dir: Path | None) -> CheckResult:
    result = CheckResult("dataset JSONL validation", details={"task_type_counts": {}})
    if data_dir is None:
        _add_warning(result, "no data directory provided")
        return result
    if not data_dir.exists():
        _add_warning(result, f"data directory does not exist: {data_dir}")
        return result

    counter: Counter[str] = Counter()
    files_checked = 0
    records_checked = 0
    for path in sorted(data_dir.rglob("*.jsonl")):
        if _is_hidden_or_cache(path):
            continue
        files_checked += 1
        with open(path, "r") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    _add_error(result, f"{_json_location(path, line_number)} is not valid JSON: {exc}")
                    continue
                records_checked += 1
                bad_paths = _find_forbidden_key_paths(record)
                for bad_path in bad_paths:
                    _add_error(result, f"{_json_location(path, line_number)} contains forbidden key: {bad_path}")
                if isinstance(record, dict):
                    task_type = record.get("task_type")
                    if isinstance(task_type, str):
                        counter[task_type] += 1
                    _check_setup_field_keys(record, path, line_number, result)
                    try:
                        validate_dataset_record(record, strict=True)
                    except Exception as exc:  # noqa: BLE001
                        _add_error(result, f"{_json_location(path, line_number)} invalid dataset record: {exc}")
                else:
                    _add_error(result, f"{_json_location(path, line_number)} record must be a JSON object")

    result.checked = records_checked
    result.details["files_checked"] = files_checked
    result.details["records_checked"] = records_checked
    result.details["task_type_counts"] = dict(sorted(counter.items()))
    return result


def check_results_json(results_dir: Path | None) -> CheckResult:
    result = CheckResult("results JSON validation")
    if results_dir is None:
        _add_warning(result, "no results directory provided")
        return result
    if not results_dir.exists():
        _add_warning(result, f"results directory does not exist: {results_dir}")
        return result

    for path in sorted(results_dir.rglob("*.json")):
        if _is_hidden_or_cache(path):
            continue
        result.checked += 1
        obj = _load_json(path, result)
        if obj is not None:
            _check_json_or_yaml_object(path, obj, result)
    result.details["files_checked"] = result.checked
    return result


def check_configs(root: Path) -> CheckResult:
    result = CheckResult("config validation")
    variables_path = root / "configs" / "variables.yaml"
    train_path = root / "configs" / "train.yaml"

    if not variables_path.exists():
        _add_error(result, f"missing variables config: {variables_path}")
    else:
        result.checked += 1
        variables_cfg = _load_yaml(variables_path, result)
        if isinstance(variables_cfg, dict):
            order = variables_cfg.get("variable_order")
            if order != CANONICAL_VARIABLE_ORDER:
                _add_error(
                    result,
                    f"{variables_path} variable_order must be {CANONICAL_VARIABLE_ORDER}; got {order}",
                )
            _check_json_or_yaml_object(variables_path, variables_cfg, result)

    if not train_path.exists():
        _add_error(result, f"missing train config: {train_path}")
    else:
        result.checked += 1
        train_cfg = _load_yaml(train_path, result)
        if train_cfg is not None:
            _check_json_or_yaml_object(train_path, train_cfg, result)

    for path in sorted((root / "configs").glob("*")) if (root / "configs").exists() else []:
        if path in {variables_path, train_path} or path.suffix not in CONFIG_SUFFIXES:
            continue
        result.checked += 1
        cfg = _load_yaml(path, result)
        if cfg is not None:
            _check_json_or_yaml_object(path, cfg, result)
    return result


def _print_result(result: CheckResult, *, strict: bool) -> None:
    status = "PASS" if result.ok else ("FAIL" if strict else "WARN")
    print(f"[{status}] {result.name}: checked={result.checked}")
    for key, value in sorted(result.details.items()):
        print(f"  {key}: {value}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    for error in result.errors:
        prefix = "error" if strict else "warning"
        print(f"  {prefix}: {error}")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    data_dir = Path(args.data_dir) if args.data_dir else None
    results_dir = Path(args.results_dir) if args.results_dir else None

    checks = [
        check_canonical_schema(root),
        check_configs(root),
    ]
    if args.check_code:
        checks.append(check_code_files(root))
    if args.check_data:
        checks.append(check_dataset_jsonl(data_dir))
    if args.check_results:
        checks.append(check_results_json(results_dir))

    for result in checks:
        _print_result(result, strict=args.strict)

    total_errors = sum(len(result.errors) for result in checks)
    total_warnings = sum(len(result.warnings) for result in checks)
    if total_errors:
        print(f"v2 integrity check completed with {total_errors} error(s) and {total_warnings} warning(s)")
        if args.strict:
            sys.exit(1)
    else:
        print(f"v2 integrity check passed with {total_warnings} warning(s)")


if __name__ == "__main__":
    main()
