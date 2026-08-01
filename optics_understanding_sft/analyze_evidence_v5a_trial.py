#!/usr/bin/env python3
"""Consolidate the frozen base, step-25, and step-50 v5A development results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNS = {
    "base": "evidence_v5a_base_dev",
    "checkpoint_25": "evidence_v5a_seed42_ckpt25_dev",
    "checkpoint_50": "evidence_v5a_seed42_ckpt50_dev",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_run(directory: Path) -> dict[str, Any]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((directory / "gates.json").read_text(encoding="utf-8"))
    metrics = gates["metrics"]
    return {
        "passed": gates["passed"],
        "passed_gate_count": sum(bool(value) for value in gates["checks"].values()),
        "gate_checks": gates["checks"],
        "macro_task_score": summary["macro_task_score"],
        "schema_valid_rate": metrics["schema_valid_rate"],
        "control_status_macro_f1": metrics["per_task_status_macro_f1"]["constrained_intervention"],
        "sufficiency_status_macro_f1": metrics["per_task_status_macro_f1"]["information_sufficiency"],
        "status_recalls": metrics["per_status_recall"],
        "control_pair_joint": metrics["pair_metrics"]["constrained_intervention"]["both_statuses_correct_rate"],
        "sufficiency_pair_joint": metrics["pair_metrics"]["information_sufficiency"]["both_statuses_correct_rate"],
        "control_action_exact_match": metrics["control_action_exact_match"],
        "control_simulator_success": metrics["control_simulator_success"],
        "control_minimum_motion_correct": metrics["control_minimum_motion_correct"],
        "sufficiency_answerable_direction_accuracy": metrics["sufficiency_answerable_direction_accuracy"],
        "sufficiency_witness_validity": metrics["sufficiency_witness_validity"],
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Evidence-grounded v5A seed-42 decision",
        "",
        "The single predeclared run stopped at 50 optimizer steps. Both saved checkpoints were evaluated on all 200 scenario-disjoint development records. Neither passed any promotion gate, so the 200-record confirmation split remains sealed.",
        "",
        "| Run | Macro | Schema | Control F1 | Feasible recall | Control pair-joint | Action exact | Sufficiency F1 | Insufficient recall | Sufficiency pair-joint | Witness valid | Gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, run in report["runs"].items():
        lines.append(
            "| {name} | {macro:.3f} | {schema:.3f} | {control:.3f} | {feasible:.3f} | {control_pair:.3f} | {action:.3f} | {suff:.3f} | {insufficient:.3f} | {suff_pair:.3f} | {witness:.3f} | {passed}/9 |".format(
                name=name.replace("_", " "),
                macro=run["macro_task_score"],
                schema=run["schema_valid_rate"],
                control=run["control_status_macro_f1"],
                feasible=run["status_recalls"]["feasible"],
                control_pair=run["control_pair_joint"],
                action=run["control_action_exact_match"],
                suff=run["sufficiency_status_macro_f1"],
                insufficient=run["status_recalls"]["insufficient_information"],
                suff_pair=run["sufficiency_pair_joint"],
                witness=run["sufficiency_witness_validity"],
                passed=run["passed_gate_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Do not promote either checkpoint and do not open confirmation or the sealed pilot test.",
            "- Do not extend the run or try another seed under this protocol. The point estimates move monotonically toward the conservative infeasible control branch while insufficiency remains completely unlearned.",
            "- Lower completion loss mainly improves JSON/schema imitation. It does not improve pair-sensitive evidence aggregation or executable action selection.",
            "",
            "## Failure interpretation",
            "",
            "The v4 failure was partly missing evidence. V5A removes that problem: the prompt explicitly provides residuals and thresholded directions, and the deterministic visible-evidence solver is perfect. The remaining failure is therefore at the instruction/aggregation interface. Ordinary token-level SFT rewards many predictable JSON and copied numeric tokens, while the small set-valued decision and action-selection fields contribute little loss. One balanced pass over 200 records changes formatting more readily than it changes those decisions.",
            "",
            "The next design should not simply add epochs or seeds. It should make prompt-visible evidence use an explicit supervised object—for example `observed_direction_set`, `successful_action_indices`, and `selected_index`—and score those fields before the final status. A deterministic tool should remain the reference and may be the production path for thresholding and exhaustive selection; the LLM can be evaluated on choosing the tool, supplying its inputs, and interpreting the result.",
            "",
            "Passing such a decomposed tier would still establish evidence aggregation only, not internal Fresnel simulation or laboratory validity.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    runs = {name: load_run(args.results_root / directory) for name, directory in RUNS.items()}
    report = {
        "protocol": "evidence_grounded_v5a_seed42_trial50",
        "training_steps": 50,
        "seed": 42,
        "paid_api_calls": 0,
        "selected_checkpoint": None,
        "confirmation_opened": False,
        "sealed_pilot_test_opened": False,
        "decision": "stop_no_promotion",
        "runs": runs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "runs": list(runs)}, indent=2))


if __name__ == "__main__":
    main()
