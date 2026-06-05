#!/usr/bin/env python3
"""Diagnose lens sign errors: inversion, perception mismatch, naive heuristics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optics_sft.eval.control_metrics import (
    CONTROL_KEYS,
    LENS_KEYS,
    control_plan,
    extract_prediction,
    overall_lens_sign_accuracy,
    sign,
    sign_accuracy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose sign conventions on labels and/or predictions.")
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        default=None,
        help="Optional rows with generated_text or parsed_json from eval.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_perception_y(err_y: float) -> str:
    if abs(err_y) < 1.0:
        return "near_center"
    return "below" if err_y > 0 else "above"


def expected_perception_x(err_x: float) -> str:
    if abs(err_x) < 1.0:
        return "near_center"
    return "right" if err_x > 0 else "left"


def negate_plan(plan: dict[str, Any]) -> dict[str, float]:
    return {key: -float(plan[key]) for key in CONTROL_KEYS if isinstance(plan.get(key), (int, float))}


def sign_report(name: str, predictions: list[Any], labels: list[Any]) -> dict[str, Any]:
    raw = sign_accuracy(predictions, labels)
    flipped_preds = []
    for row in predictions:
        plan = control_plan(row)
        if plan is None:
            flipped_preds.append(row)
            continue
        negated = negate_plan(plan)
        flipped_preds.append({"parsed_json": {"control_plan": negated}})
    flipped = sign_accuracy(flipped_preds, labels)
    return {
        "name": name,
        "overall_lens_sign_accuracy": overall_lens_sign_accuracy(predictions, labels),
        "overall_lens_sign_accuracy_if_negate_pred": overall_lens_sign_accuracy(flipped_preds, labels),
        "per_axis": {key: {"raw": raw.get(key), "negated_pred": flipped.get(key)} for key in LENS_KEYS},
    }


def label_heuristics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = {key: {"agree": 0, "total": 0} for key in LENS_KEYS}
    perception_y_mismatch = 0
    for row in rows:
        pe = row.get("private_eval")
        target = row.get("target")
        if not isinstance(pe, Mapping) or not isinstance(target, Mapping):
            continue
        current = pe.get("current_state")
        tgt = pe.get("target_state")
        plan = target.get("control_plan")
        perception = target.get("perception")
        if not all(isinstance(v, dict) for v in (current, tgt, plan, perception)):
            continue
        err_x = float(current["centroid_x_px"]) - float(tgt["centroid_x_px"])
        err_y = float(current["centroid_y_px"]) - float(tgt["centroid_y_px"])
        if perception.get("current_relative_to_target_y") != expected_perception_y(err_y):
            perception_y_mismatch += 1
        shift_x = float(tgt["centroid_x_px"]) - float(current["centroid_x_px"])
        shift_y = float(tgt["centroid_y_px"]) - float(current["centroid_y_px"])
        for axis_key, delta, shift in (
            ("lens_x_delta_mm", err_x, shift_x),
            ("lens_y_delta_mm", err_y, shift_y),
        ):
            label_value = float(plan[axis_key])
            label_sign = sign(label_value)
            if label_sign == 0:
                continue
            for heuristic_name, reference in (("vs_centroid_shift", shift), ("vs_neg_centroid_error", -delta)):
                ref_sign = sign(reference)
                if ref_sign == 0:
                    continue
                bucket = stats[axis_key]
                bucket.setdefault(heuristic_name, {"agree": 0, "total": 0})
                bucket[heuristic_name]["total"] += 1
                if label_sign == ref_sign:
                    bucket[heuristic_name]["agree"] += 1
    return {
        "perception_y_mismatch_count": perception_y_mismatch,
        "label_sign_heuristics": stats,
    }


def main() -> None:
    args = parse_args()
    labels = read_jsonl(args.eval_jsonl)
    print("=== Label / dataset conventions ===")
    print(json.dumps(label_heuristics(labels), indent=2))

    if args.predictions_jsonl is None:
        print("\n(No --predictions-jsonl; skip model flip analysis.)")
        return

    pred_rows = read_jsonl(args.predictions_jsonl)
    predictions = []
    for row in pred_rows:
        parsed = extract_prediction(row)
        predictions.append(
            {
                "parsed_json": parsed,
                "generated_text": row.get("generated_text"),
            }
        )
    print("\n=== Model sign (raw vs negate prediction) ===")
    print(json.dumps(sign_report("model", predictions, labels), indent=2))


if __name__ == "__main__":
    main()
