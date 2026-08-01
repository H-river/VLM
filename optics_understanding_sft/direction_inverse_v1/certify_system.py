#!/usr/bin/env python3
"""Aggregate frozen evaluations into explicit component promotion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from optics_understanding_sft.core import read_jsonl, stable_json_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--llm-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def gate(passed: bool, evidence: Mapping[str, Any], rule: str) -> dict[str, Any]:
    return {"passed": bool(passed), "rule": rule, "evidence": dict(evidence)}


def main() -> None:
    args = parse_args(); data = args.package_dir / "data/v1"; results = args.package_dir / "results"
    direction_audit = load(data / "direction/audit_report.json")
    inverse_audit = load(data / "inverse/audit_report.json")
    visual_audit = load(data / "inverse/visual_sensor_frame_audit.json")
    hash_ok = all(
        inverse_audit["record_hashes"][split]
        == stable_json_hash(read_jsonl(data / "inverse/canonical" / f"{split}.jsonl"))
        for split in ("train", "val", "eval_iid", "eval_ood")
    )
    image_errors = []
    for split in ("train", "val", "eval_iid", "eval_ood"):
        for row in read_jsonl(data / "inverse/canonical" / f"{split}.jsonl"):
            for relative in row["prompt_inputs"].get("images", []):
                path = data / relative
                if not path.exists(): image_errors.append(f"missing:{relative}"); continue
                with Image.open(path) as image:
                    if image.size != (384, 384): image_errors.append(f"size:{relative}:{image.size}")

    small = load(results / "direction_small_v1/summary.json")
    llm = load(args.llm_evaluation)
    forward = load(results / "forward_grid_hybrid_v1/summary.json")
    inverse_candidates = {
        name: load(results / path / "summary.json") for name, path in {
            "general_forward_search": "inverse_controller_v1",
            "grid_forward_search": "inverse_controller_grid_v1",
            "direct_inverse": "inverse_direct_v1", "probability_residual_ensemble": "inverse_ensemble_v1",
        }.items()
    }
    selected_inverse_name = max(inverse_candidates, key=lambda name: (
        inverse_candidates[name]["val"]["selected_action_target_success_feasible"], name))
    selected_inverse = inverse_candidates[selected_inverse_name]
    visual = load(results / "visual_pipeline_sensor_v1/summary.json")
    gates = {}
    gates["dataset_integrity"] = gate(
        direction_audit["passed"] and inverse_audit["passed"] and visual_audit["passed"]
        and hash_ok and not image_errors,
        {"direction_audit": direction_audit["passed"], "inverse_audit": inverse_audit["passed"],
         "visual_frame_audit": visual_audit["passed"], "canonical_hashes_match": hash_ok,
         "image_error_count": len(image_errors)}, "all audits pass; hashes match; every image is 384x384")
    iid_small = small["eval_iid"]["shared_mlp"]["equal_field_macro_f1"]
    ood_small = small["eval_ood"]["shared_mlp"]["equal_field_macro_f1"]
    gates["small_direction_model"] = gate(iid_small >= 0.45 and ood_small >= 0.45,
        {"iid_macro_f1": iid_small, "ood_macro_f1": ood_small}, "IID and OOD macro-F1 >= 0.45")
    iid_llm = llm["eval_iid"]["metrics"]["equal_field_macro_f1"]
    ood_llm = llm["eval_ood"]["metrics"]["equal_field_macro_f1"]
    gates["dedicated_direction_llm"] = gate(iid_llm >= iid_small and ood_llm >= ood_small,
        {"iid_macro_f1": iid_llm, "ood_macro_f1": ood_llm,
         "small_iid_macro_f1": iid_small, "small_ood_macro_f1": ood_small},
        "LLM must match or beat the small direction model on both test splits")
    forward_evidence = {}
    forward_pass = True
    for split in ("eval_iid", "eval_ood"):
        model_rate = forward[split]["strict_all_five_success"]
        zero_rate = forward[split]["zero_baseline_strict_all_five_success"]
        forward_evidence[f"{split}_model_strict"] = model_rate
        forward_evidence[f"{split}_zero_strict"] = zero_rate
        forward_evidence[f"{split}_skill_over_zero"] = forward[split]["skill_over_zero"]
        forward_pass &= model_rate - zero_rate >= 0.05 and forward[split]["skill_over_zero"] > 0.0
    gates["quantitative_forward_model"] = gate(forward_pass, forward_evidence,
        "strict all-five success exceeds zero by >= 0.05 and mean-error skill is positive on IID and OOD")
    inverse_evidence = {"validation_selected_model": selected_inverse_name,
                        "iid_target_success": selected_inverse["eval_iid"]["selected_action_target_success_feasible"],
                        "ood_target_success": selected_inverse["eval_ood"]["selected_action_target_success_feasible"]}
    gates["numeric_inverse_control"] = gate(
        inverse_evidence["iid_target_success"] >= 0.5 and inverse_evidence["ood_target_success"] >= 0.5,
        inverse_evidence, "validation-selected controller reaches target on >= 50% of feasible IID and OOD cases")
    visual_iid = visual["eval_iid"]["measurement"]["strict_all_five_state_success"]
    visual_ood = visual["eval_ood"]["measurement"]["strict_all_five_state_success"]
    gates["visual_measurement"] = gate(visual_iid >= 0.8 and visual_ood >= 0.8,
        {"iid_all_five_state_success": visual_iid, "ood_all_five_state_success": visual_ood},
        "calibrated image meter all-five success >= 80% on IID and OOD")
    visual_control_iid = visual["eval_iid"]["controller"]["selected_action_target_success_feasible"]
    visual_control_ood = visual["eval_ood"]["controller"]["selected_action_target_success_feasible"]
    gates["visual_inverse_control"] = gate(visual_control_iid >= 0.5 and visual_control_ood >= 0.5,
        {"iid_target_success": visual_control_iid, "ood_target_success": visual_control_ood},
        "end-to-end image A-to-B target success >= 50% on IID and OOD")
    full = all(item["passed"] for item in gates.values())
    promoted = [name for name, item in gates.items() if item["passed"]]
    summary = {"version": "direction_inverse_v1_final_certification",
               "full_autonomous_system_promoted": full,
               "promotion_status": "full" if full else "component_only",
               "promoted_components": promoted, "gates": gates,
               "test_policy": "all routing and checkpoint choices fixed on validation before IID/OOD scoring"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Direction and inverse optics v1: final certification", "",
             f"**Promotion status: {summary['promotion_status']}**", "",
             "| Gate | Result | Key evidence |", "|---|---|---|"]
    for name, item in gates.items():
        evidence = "; ".join(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                             for key, value in item["evidence"].items())
        lines.append(f"| {name} | {'PASS' if item['passed'] else 'FAIL'} | {evidence} |")
    lines += ["", "The full autonomous system is promoted only if every gate passes. Component-only status means passing modules may be used with explicit supervision, while failed control or LLM modules remain experimental."]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
