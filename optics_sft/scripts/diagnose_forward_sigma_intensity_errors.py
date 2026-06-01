#!/usr/bin/env python3
"""Diagnose sigma and peak-intensity errors for forward_transition predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover - script reports this in image diagnostics.
    np = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]


STATE_FIELDS = (
    "centroid_x_px",
    "centroid_y_px",
    "sigma_x_px",
    "sigma_y_px",
    "peak_intensity",
)
SIGMA_PEAK_FIELDS = ("sigma_x_px", "sigma_y_px", "peak_intensity")
ALIAS_KEYS = (
    "sigma_x",
    "sigma_y",
    "sigma_x_px",
    "sigma_y_px",
    "beam_sigma_x_px",
    "beam_sigma_y_px",
    "d4sigma_x_px",
    "d4sigma_y_px",
    "width_4sigma_x",
    "width_4sigma_y",
    "peak",
    "peak_intensity",
    "max_intensity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose forward_transition sigma and peak-intensity errors."
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=Path("../VLM_data/physics_sft_forward_transition_v1/val.jsonl"),
    )
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        default=Path("../VLM_runs/qwen25vl_3b_qlora_forward_only_v1/val_predictions.jsonl"),
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=Path("../VLM_runs/qwen25vl_3b_qlora_forward_only_v1/val_forward_eval.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../VLM_runs/qwen25vl_3b_qlora_forward_only_v1/diagnostics_sigma_intensity"),
    )
    parser.add_argument("--num-examples", type=int, default=10)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            rows.append(row)
    return rows


def read_eval_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            row["sample_id"]: row
            for row in csv.DictReader(f)
            if row.get("sample_id")
        }


def first_json_object_text(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parsed_prediction(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(row, Mapping):
        return None
    parsed = row.get("parsed_json")
    if isinstance(parsed, dict):
        return parsed
    for key in ("prediction", "raw_prediction_text", "raw_text"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                candidate = json.loads(value)
            except json.JSONDecodeError:
                block = first_json_object_text(value)
                if block is None:
                    continue
                try:
                    candidate = json.loads(block)
                except json.JSONDecodeError:
                    continue
            if isinstance(candidate, dict):
                return candidate
    return None


def prediction_index(predictions: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(predictions):
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str):
            indexed[sample_id] = row
        elif "sample_index" in row:
            indexed[str(row["sample_index"])] = row
        else:
            indexed[str(index)] = row
    return indexed


def nested_mapping(obj: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = obj.get(key)
    return value if isinstance(value, Mapping) else {}


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
    elif isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return value if math.isfinite(value) else None


def state_numbers(state: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for field in STATE_FIELDS:
        parsed = number(state.get(field))
        if parsed is not None:
            values[field] = parsed
    return values


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if values else None


def stats(values: Iterable[float]) -> dict[str, Any]:
    parsed = [float(value) for value in values if math.isfinite(float(value))]
    if not parsed:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": len(parsed),
        "min": min(parsed),
        "max": max(parsed),
        "mean": mean(parsed),
        "std": std(parsed),
        "unique_rounded_3": len({round(value, 3) for value in parsed}),
        "unique_rounded_6": len({round(value, 6) for value in parsed}),
    }


def abs_error(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(a - b)


def ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def finite(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def scale_flags(name: str, left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    left_max = left.get("max")
    right_max = right.get("max")
    left_mean = left.get("mean")
    right_mean = right.get("mean")
    if not isinstance(left_max, (int, float)) or not isinstance(right_max, (int, float)):
        return flags
    bigger = max(abs(left_max), abs(right_max))
    smaller = min(abs(left_max), abs(right_max))
    if smaller > 0:
        max_ratio = bigger / smaller
        if 180.0 <= max_ratio <= 320.0:
            flags.append(f"{name}: max ratio {max_ratio:.2f}, likely 0-1 vs 0-255 scale mismatch")
        if 1.8 <= max_ratio <= 2.2:
            flags.append(f"{name}: max ratio {max_ratio:.2f}, possible radius/diameter mismatch")
        if 3.6 <= max_ratio <= 4.4:
            flags.append(f"{name}: max ratio {max_ratio:.2f}, possible sigma vs D4sigma mismatch")
    if isinstance(left_mean, (int, float)) and isinstance(right_mean, (int, float)):
        mean_ratio = ratio(max(abs(left_mean), abs(right_mean)), min(abs(left_mean), abs(right_mean)))
        if mean_ratio is not None:
            if 180.0 <= mean_ratio <= 320.0:
                flags.append(f"{name}: mean ratio {mean_ratio:.2f}, likely 0-1 vs 0-255 scale mismatch")
            if 1.8 <= mean_ratio <= 2.2:
                flags.append(f"{name}: mean ratio {mean_ratio:.2f}, possible radius/diameter mismatch")
            if 3.6 <= mean_ratio <= 4.4:
                flags.append(f"{name}: mean ratio {mean_ratio:.2f}, possible sigma vs D4sigma mismatch")
    return flags


def collapse_flags(prefix: str, field: str, stat: dict[str, Any], n: int) -> list[str]:
    flags: list[str] = []
    field_std = stat.get("std")
    unique_3 = stat.get("unique_rounded_3")
    if isinstance(field_std, (int, float)) and field_std < 1e-6:
        flags.append(f"{prefix}.{field}: near-constant output, std={field_std:.3g}")
    if isinstance(unique_3, int) and n >= 20 and unique_3 <= max(2, n // 20):
        flags.append(f"{prefix}.{field}: low unique rounded values ({unique_3}/{n}), possible output collapse")
    return flags


def range_compression_flags(
    field: str,
    predicted_stat: Mapping[str, Any],
    true_stat: Mapping[str, Any],
) -> list[str]:
    flags: list[str] = []
    pred_std = predicted_stat.get("std")
    true_std = true_stat.get("std")
    pred_min = predicted_stat.get("min")
    pred_max = predicted_stat.get("max")
    true_min = true_stat.get("min")
    true_max = true_stat.get("max")
    if isinstance(pred_std, (int, float)) and isinstance(true_std, (int, float)) and true_std > 0:
        std_ratio = pred_std / true_std
        if std_ratio < 0.6:
            flags.append(
                f"predicted.{field}: compressed variation, std ratio predicted/true={std_ratio:.3f}"
            )
    if all(isinstance(value, (int, float)) for value in (pred_min, pred_max, true_min, true_max)):
        pred_range = float(pred_max) - float(pred_min)
        true_range = float(true_max) - float(true_min)
        if true_range > 0:
            range_ratio = pred_range / true_range
            if range_ratio < 0.6:
                flags.append(
                    f"predicted.{field}: compressed range, range ratio predicted/true={range_ratio:.3f}"
                )
    return flags


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    if left_mean is None or right_mean is None:
        return None
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


def image_metrics(path: Path) -> dict[str, float] | None:
    if np is None or Image is None or not path.exists():
        return None
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float64)
    gray = arr.mean(axis=2)
    total = float(gray.sum())
    if total <= 0.0:
        return None
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float((gray * xx).sum() / total)
    cy = float((gray * yy).sum() / total)
    sx = float(np.sqrt(max(float((gray * (xx - cx) ** 2).sum() / total), 0.0)))
    sy = float(np.sqrt(max(float((gray * (yy - cy) ** 2).sum() / total), 0.0)))
    return {
        "centroid_x_px": cx,
        "centroid_y_px": cy,
        "sigma_x_px": sx,
        "sigma_y_px": sy,
        "peak_uint8": float(gray.max()),
        "peak_0_1": float(gray.max() / 255.0),
        "width": float(w),
        "height": float(h),
    }


def sample_context(row: Mapping[str, Any]) -> dict[str, Any]:
    prompt_inputs = nested_mapping(row, "prompt_inputs")
    return {
        "sample_id": row.get("sample_id"),
        "before_state": nested_mapping(nested_mapping(row, "private_eval"), "before_state"),
        "action": nested_mapping(prompt_inputs, "action"),
        "safe_setup_metadata": nested_mapping(prompt_inputs, "safe_setup_metadata"),
    }


def row_for_csv(record: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {field: record.get(field) for field in fields}


def write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(row_for_csv(record, fieldnames))


def collect_aliases(state: Mapping[str, Any]) -> list[str]:
    return [key for key in ALIAS_KEYS if key in state]


def format_stat(stat: Mapping[str, Any]) -> str:
    return (
        f"min={stat.get('min'):.6g}, max={stat.get('max'):.6g}, "
        f"mean={stat.get('mean'):.6g}, std={stat.get('std'):.6g}"
        if stat.get("count")
        else "no values"
    )


def build_report(summary: dict[str, Any], worst_sigma: list[dict[str, Any]], worst_peak: list[dict[str, Any]]) -> str:
    lines = [
        "# Forward Sigma/Intensity Diagnostics",
        "",
        "## Summary",
        "",
    ]
    for flag in summary["flags"]:
        lines.append(f"- {flag}")
    if not summary["flags"]:
        lines.append("- No automatic scale/collapse flags were triggered.")
    lines.extend(["", "## Distributions", ""])
    for source in ("private_eval", "target", "predicted"):
        lines.append(f"### {source}")
        for field in SIGMA_PEAK_FIELDS:
            lines.append(f"- `{field}`: {format_stat(summary['statistics'][source][field])}")
        lines.append("")
    lines.extend(
        [
            "## Target vs Private Eval",
            "",
            f"- Max abs diff sigma_x_px: `{summary['target_private_max_abs_diff']['sigma_x_px']}`",
            f"- Max abs diff sigma_y_px: `{summary['target_private_max_abs_diff']['sigma_y_px']}`",
            f"- Max abs diff peak_intensity: `{summary['target_private_max_abs_diff']['peak_intensity']}`",
            "",
            "## Errors",
            "",
        ]
    )
    for key, value in summary["error_summary"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Baselines", ""])
    for key, value in summary["baseline_summary"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Image Diagnostics", ""])
    image_summary = summary.get("image_summary", {})
    if image_summary:
        for key, value in image_summary.items():
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- Image diagnostics unavailable.")
    lines.extend(["", "## Worst Sigma Examples", ""])
    for record in worst_sigma[:5]:
        lines.append(
            f"- `{record['sample_id']}`: sigma_error_max={record['sigma_error_max']:.6g}, "
            f"true=({record.get('true_sigma_x_px'):.6g}, {record.get('true_sigma_y_px'):.6g}), "
            f"pred=({record.get('pred_sigma_x_px'):.6g}, {record.get('pred_sigma_y_px'):.6g})"
        )
    lines.extend(["", "## Worst Peak Examples", ""])
    for record in worst_peak[:5]:
        lines.append(
            f"- `{record['sample_id']}`: peak_error={record['peak_error']:.6g}, "
            f"true={record.get('true_peak_intensity'):.6g}, pred={record.get('pred_peak_intensity'):.6g}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive")

    rows = [row for row in read_jsonl(args.val_jsonl) if row.get("sample_type") == "forward_transition"]
    predictions = prediction_index(read_jsonl(args.predictions_jsonl))
    eval_rows = read_eval_csv(args.eval_csv)
    image_root = args.val_jsonl.parent / "images"

    records: list[dict[str, Any]] = []
    keys_by_source: dict[str, set[str]] = {"target": set(), "private_eval": set(), "predicted": set()}
    aliases_by_source: dict[str, set[str]] = {"target": set(), "private_eval": set(), "predicted": set()}
    image_records: list[dict[str, Any]] = []

    for row in rows:
        sample_id = str(row.get("sample_id"))
        target_state = nested_mapping(nested_mapping(row, "target"), "predicted_after_state")
        target_change = nested_mapping(nested_mapping(row, "target"), "predicted_change")
        private_eval = nested_mapping(row, "private_eval")
        before_state = nested_mapping(private_eval, "before_state")
        after_state = nested_mapping(private_eval, "after_state")
        prediction = parsed_prediction(predictions.get(sample_id)) or {}
        predicted_state = nested_mapping(prediction, "predicted_after_state")
        predicted_change = nested_mapping(prediction, "predicted_change")

        keys_by_source["target"].update(target_state.keys())
        keys_by_source["private_eval"].update(after_state.keys())
        keys_by_source["predicted"].update(predicted_state.keys())
        aliases_by_source["target"].update(collect_aliases(target_state))
        aliases_by_source["private_eval"].update(collect_aliases(after_state))
        aliases_by_source["predicted"].update(collect_aliases(predicted_state))

        target = state_numbers(target_state)
        true = state_numbers(after_state)
        pred = state_numbers(predicted_state)

        record: dict[str, Any] = {
            "sample_id": sample_id,
            "target_keys": sorted(target_state.keys()),
            "private_eval_keys": sorted(after_state.keys()),
            "predicted_keys": sorted(predicted_state.keys()),
            "target_change_keys": sorted(target_change.keys()),
            "predicted_change_keys": sorted(predicted_change.keys()),
        }
        for field in STATE_FIELDS:
            record[f"target_{field}"] = target.get(field)
            record[f"true_{field}"] = true.get(field)
            record[f"pred_{field}"] = pred.get(field)
            record[f"target_private_{field}_abs_diff"] = abs_error(target.get(field), true.get(field))
            record[f"pred_true_{field}_abs_error"] = abs_error(pred.get(field), true.get(field))
            record[f"pred_target_{field}_abs_error"] = abs_error(pred.get(field), target.get(field))

        record["sigma_error_max"] = max(
            finite([
                record.get("pred_true_sigma_x_px_abs_error"),
                record.get("pred_true_sigma_y_px_abs_error"),
            ])
            or [0.0]
        )
        record["peak_error"] = record.get("pred_true_peak_intensity_abs_error")

        csv_eval = eval_rows.get(sample_id, {})
        record["eval_csv_sigma_x_abs_error"] = csv_eval.get("sigma_x_px_abs_error")
        record["eval_csv_sigma_y_abs_error"] = csv_eval.get("sigma_y_px_abs_error")
        record["eval_csv_peak_abs_error"] = csv_eval.get("peak_intensity_abs_error")
        record.update(sample_context(row))
        records.append(record)

    statistics_by_source: dict[str, dict[str, dict[str, Any]]] = {}
    for source, prefix in (
        ("private_eval", "true"),
        ("target", "target"),
        ("predicted", "pred"),
    ):
        statistics_by_source[source] = {}
        for field in SIGMA_PEAK_FIELDS:
            statistics_by_source[source][field] = stats(
                value
                for value in (record.get(f"{prefix}_{field}") for record in records)
                if value is not None
            )

    target_private_max_abs_diff = {
        field: max(finite(record.get(f"target_private_{field}_abs_diff") for record in records) or [0.0])
        for field in SIGMA_PEAK_FIELDS
    }
    error_summary = {
        "pred_vs_true_sigma_x_mae_px": mean(finite(record.get("pred_true_sigma_x_px_abs_error") for record in records)),
        "pred_vs_true_sigma_y_mae_px": mean(finite(record.get("pred_true_sigma_y_px_abs_error") for record in records)),
        "pred_vs_true_peak_intensity_mae": mean(finite(record.get("pred_true_peak_intensity_abs_error") for record in records)),
        "pred_vs_target_sigma_x_mae_px": mean(finite(record.get("pred_target_sigma_x_px_abs_error") for record in records)),
        "pred_vs_target_sigma_y_mae_px": mean(finite(record.get("pred_target_sigma_y_px_abs_error") for record in records)),
        "pred_vs_target_peak_intensity_mae": mean(finite(record.get("pred_target_peak_intensity_abs_error") for record in records)),
    }
    baseline_summary: dict[str, Any] = {}
    for field in SIGMA_PEAK_FIELDS:
        true_values = finite(record.get(f"true_{field}") for record in records)
        pred_values = finite(record.get(f"pred_{field}") for record in records)
        before_values = finite(record.get("before_state", {}).get(field) for record in records)
        if len(true_values) == len(records):
            true_mean = mean(true_values)
            baseline_summary[f"{field}_true_mean"] = true_mean
            if true_mean is not None:
                baseline_summary[f"{field}_true_mean_baseline_mae"] = mean(
                    [abs(value - true_mean) for value in true_values]
                )
        if len(pred_values) == len(true_values):
            baseline_summary[f"{field}_pred_true_corr"] = correlation(pred_values, true_values)
        if len(before_values) == len(true_values):
            baseline_summary[f"{field}_copy_before_mae"] = mean(
                [abs(before - true) for before, true in zip(before_values, true_values)]
            )
            baseline_summary[f"{field}_before_true_corr"] = correlation(before_values, true_values)

    flags: list[str] = []
    n = len(records)
    for field in SIGMA_PEAK_FIELDS:
        flags.extend(
            scale_flags(
                f"predicted vs private_eval {field}",
                statistics_by_source["predicted"][field],
                statistics_by_source["private_eval"][field],
            )
        )
        flags.extend(
            scale_flags(
                f"target vs private_eval {field}",
                statistics_by_source["target"][field],
                statistics_by_source["private_eval"][field],
            )
        )
        flags.extend(
            range_compression_flags(
                field,
                statistics_by_source["predicted"][field],
                statistics_by_source["private_eval"][field],
            )
        )
        flags.extend(collapse_flags("predicted", field, statistics_by_source["predicted"][field], n))
    for field, max_diff in target_private_max_abs_diff.items():
        if max_diff > 1e-6:
            flags.append(f"target vs private_eval {field}: max abs diff {max_diff:.6g}")

    worst_sigma = sorted(records, key=lambda r: float(r.get("sigma_error_max") or 0.0), reverse=True)[
        : args.num_examples
    ]
    worst_peak = sorted(records, key=lambda r: float(r.get("peak_error") or 0.0), reverse=True)[
        : args.num_examples
    ]

    for record in worst_sigma[: max(1, min(args.num_examples, 5))]:
        sample_id = record["sample_id"]
        row = next(item for item in rows if str(item.get("sample_id")) == sample_id)
        private_eval = nested_mapping(row, "private_eval")
        prompt_images = nested_mapping(nested_mapping(row, "prompt_inputs"), "images")
        before_path = prompt_images.get("before_image_path")
        after_path = private_eval.get("after_image_path")
        image_record: dict[str, Any] = {"sample_id": sample_id}
        if isinstance(before_path, str):
            before_metrics = image_metrics(image_root / before_path)
            if before_metrics:
                for key, value in before_metrics.items():
                    image_record[f"before_png_{key}"] = value
        if isinstance(after_path, str):
            after_metrics = image_metrics(image_root / after_path)
            if after_metrics:
                for key, value in after_metrics.items():
                    image_record[f"after_png_{key}"] = value
        image_record["true_after_sigma_x_px"] = record.get("true_sigma_x_px")
        image_record["true_after_sigma_y_px"] = record.get("true_sigma_y_px")
        image_record["true_after_peak_intensity"] = record.get("true_peak_intensity")
        image_records.append(image_record)

    image_summary: dict[str, Any] = {"checked_examples": len(image_records)}
    if image_records:
        image_summary.update(
            {
                "after_png_sigma_x_mae_vs_private_px": mean(
                    finite(
                        abs_error(
                            record.get("after_png_sigma_x_px"),
                            record.get("true_after_sigma_x_px"),
                        )
                        for record in image_records
                    )
                ),
                "after_png_sigma_y_mae_vs_private_px": mean(
                    finite(
                        abs_error(
                            record.get("after_png_sigma_y_px"),
                            record.get("true_after_sigma_y_px"),
                        )
                        for record in image_records
                    )
                ),
                "after_png_peak_uint8_mean": mean(finite(record.get("after_png_peak_uint8") for record in image_records)),
                "after_png_peak_0_1_mean": mean(finite(record.get("after_png_peak_0_1") for record in image_records)),
                "true_after_peak_mean_checked": mean(
                    finite(record.get("true_after_peak_intensity") for record in image_records)
                ),
            }
        )

    summary = {
        "val_jsonl": str(args.val_jsonl),
        "predictions_jsonl": str(args.predictions_jsonl),
        "eval_csv": str(args.eval_csv),
        "num_rows": len(records),
        "field_keys": {source: sorted(keys) for source, keys in keys_by_source.items()},
        "alias_keys": {source: sorted(keys) for source, keys in aliases_by_source.items()},
        "statistics": statistics_by_source,
        "target_private_max_abs_diff": target_private_max_abs_diff,
        "error_summary": error_summary,
        "baseline_summary": baseline_summary,
        "flags": flags,
        "image_summary": image_summary,
        "image_examples": image_records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "diagnostics_report.md").write_text(
        build_report(summary, worst_sigma, worst_peak),
        encoding="utf-8",
    )
    common_fields = [
        "sample_id",
        "sigma_error_max",
        "peak_error",
        "true_sigma_x_px",
        "true_sigma_y_px",
        "pred_sigma_x_px",
        "pred_sigma_y_px",
        "target_sigma_x_px",
        "target_sigma_y_px",
        "true_peak_intensity",
        "pred_peak_intensity",
        "target_peak_intensity",
        "before_state",
        "action",
        "safe_setup_metadata",
    ]
    write_csv(args.output_dir / "worst_sigma_examples.csv", worst_sigma, common_fields)
    write_csv(args.output_dir / "worst_peak_examples.csv", worst_peak, common_fields)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
