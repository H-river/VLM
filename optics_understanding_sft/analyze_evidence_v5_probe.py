#!/usr/bin/env python3
"""Consolidate the local, no-training evidence-v5 learnability probes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .core import read_jsonl


VARIANTS = {
    "evidence_v5_probe_base28": "verbose raw centroids, base model",
    "evidence_v5_probe_compact_base28": "compact raw centroids, base model",
    "evidence_v5_probe_compact_mixed100_28": "compact raw centroids, v4 mixed-100 adapter",
    "evidence_v5_probe_scaffold_base28": "numeric residual/delta scaffold, base model",
    "evidence_v5_probe_symbolic_base28": "residual plus direction scaffold, base model",
    "evidence_v5_probe_symbolic_recency_base28": "symbolic scaffold with repeated decision rule, base model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def status_from_prediction(prediction: dict[str, Any]) -> str | None:
    parsed = prediction.get("parsed_json")
    if not isinstance(parsed, dict):
        return None
    status = parsed.get("status")
    return str(status) if status is not None else None


def pair_metrics(
    rows: list[dict[str, Any]], records: dict[str, dict[str, Any]], task: str
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for prediction in rows:
        record = records[str(prediction["example_id"])]
        if record["task_type"] != task:
            continue
        grouped[str(record["provenance"]["match_group_id"])].append(
            (str(record["target"]["status"]), status_from_prediction(prediction))
        )
    complete = [pair for pair in grouped.values() if len(pair) == 2]
    both_correct = sum(all(target == predicted for target, predicted in pair) for pair in complete)
    changed = sum(len({predicted for _, predicted in pair}) == 2 for pair in complete)
    return {
        "complete_pair_count": len(complete),
        "both_statuses_correct_rate": both_correct / len(complete) if complete else None,
        "prediction_changes_with_pair_rate": changed / len(complete) if complete else None,
    }


def analyze_variant(
    result_dir: Path, records: dict[str, dict[str, Any]], description: str
) -> dict[str, Any]:
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(result_dir / "predictions.jsonl")
    task_statuses: dict[str, Counter[str]] = defaultdict(Counter)
    for prediction in predictions:
        task = str(records[str(prediction["example_id"])]["task_type"])
        task_statuses[task][str(status_from_prediction(prediction))] += 1
    per_task = {}
    for task in ("constrained_intervention", "information_sufficiency"):
        task_summary = summary["per_task"][task]
        per_task[task] = {
            "status_accuracy": task_summary.get("status_exact"),
            "status_macro_f1": task_summary.get("status_macro_f1"),
            "schema_valid_rate": task_summary.get("schema_valid_rate"),
            "task_score": task_summary.get("task_score"),
            "predicted_status_counts": dict(sorted(task_statuses[task].items())),
            "pair_metrics": pair_metrics(predictions, records, task),
        }
    run_path = result_dir / "predictions.run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    return {
        "description": description,
        "model_kind": run.get("model_kind"),
        "adapter_path": run.get("adapter_path"),
        "input_records_hash": run.get("input_records_hash"),
        "evaluated_records": summary["evaluated_records"],
        "macro_task_score": summary["macro_task_score"],
        "schema_valid_rate": summary["schema_valid_rate"],
        "per_task": per_task,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Evidence-grounded v5 learnability probe",
        "",
        "All runs use the same 28 target records (seven complete physical scenarios). They are a design probe built from already-used v4 training groups, not a future evaluation split. No additional training or paid API call was used.",
        "",
        "| Representation | Model | Macro | Schema | Control F1 | Control pair-joint | Sufficiency F1 | Sufficiency pair-joint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in report["variants"].values():
        control = variant["per_task"]["constrained_intervention"]
        sufficiency = variant["per_task"]["information_sufficiency"]
        model = "v4 adapter" if variant["adapter_path"] else "base"
        lines.append(
            "| {description} | {model} | {macro:.3f} | {schema:.3f} | {control_f1:.3f} | {control_pair:.3f} | {suff_f1:.3f} | {suff_pair:.3f} |".format(
                description=variant["description"],
                model=model,
                macro=variant["macro_task_score"],
                schema=variant["schema_valid_rate"],
                control_f1=control["status_macro_f1"],
                control_pair=control["pair_metrics"]["both_statuses_correct_rate"],
                suff_f1=sufficiency["status_macro_f1"],
                suff_pair=sufficiency["pair_metrics"]["both_statuses_correct_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The numeric residual scaffold is the first representation that makes control decisions pair-sensitive: the base model reaches 0.708 status macro-F1 and 0.429 pair-joint accuracy on that small slice.",
            "- The old v4 adapter erases that improvement and returns the conservative control class for every row, consistent with the v4 collapse analysis.",
            "- Sufficiency remains a constant `answerable` classifier even when each row contains an explicit thresholded direction and the decision rule is repeated next to the output contract. Its pair-joint accuracy remains zero.",
            "- Repeating the decision rule improves schema validity to 1.0 but collapses control status to the constant infeasible class. Prompt wording alone is therefore unstable and is not a promotion signal.",
            "- The next justified experiment is a fresh, scenario-disjoint v5A dataset followed by one predeclared 50-step seed-42 format-and-aggregation trial. It should be stopped unless both task F1 and pair-joint gates improve; raw-centroid v5B/v5C transfer is evaluated only after that.",
            "",
            "The 28-record numbers are diagnostic estimates, not benchmark claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    records = {str(record["example_id"]): record for record in read_jsonl(args.records_jsonl)}
    variants = {}
    for name, description in VARIANTS.items():
        result_dir = args.results_root / name
        if (result_dir / "summary.json").exists() and (result_dir / "predictions.jsonl").exists():
            variants[name] = analyze_variant(result_dir, records, description)
    report = {
        "protocol": "evidence_grounded_v5_design_probe",
        "design_only": True,
        "future_evaluation_eligible": False,
        "additional_training_steps": 0,
        "paid_api_calls": 0,
        "variant_count": len(variants),
        "variants": variants,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"variant_count": len(variants), "output": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
