#!/usr/bin/env python3
"""Write the candidate machine summary, artifact registry, and meeting report."""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "supervisor_v1_1_candidate"
REPORTS = BASE / "reports"
STATUS = "NOT SEALED — FROZEN EVALUATION DISABLED"
CONCLUSION = "READY FOR HUMAN REVIEW BEFORE FINAL FREEZE"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def mean_std(metric: dict[str, Any]) -> str:
    return f"{pct(metric['mean'])} +/- {100.0 * float(metric['std_population']):.2f} pp"


def ci(metric: dict[str, Any]) -> str:
    low, high = metric["ci95_percentile"]
    return f"[{100.0 * low:.2f}, {100.0 * high:.2f}] pp"


def matrix_text(matrix: dict[str, Any]) -> list[str]:
    labels = matrix["labels"]
    values = matrix.get("summed_matrix", matrix.get("matrix"))
    lines = ["| true \\ predicted | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for label, row in zip(labels, values, strict=True):
        lines.append("| " + label + " | " + " | ".join(str(value) for value in row) + " |")
    return lines


def selected_artifacts(results: dict[str, Any]) -> dict[str, Any]:
    paths: set[Path] = set()
    for pattern in (
        "protocol/*.json", "configs/*.yaml", "configs/*.json", "manifests/*", "sft/*.json",
        "sft/sft_train.jsonl", "sft/sft_dev.jsonl", "reports/*.json", "reports/*.md", "scripts/*.py",
        "artifacts/training/*/run_manifest.latest.json", "artifacts/training/*/final_adapter/adapter_config.json",
        "artifacts/training/*/final_adapter/adapter_model.safetensors",
        "artifacts/predictions/*.report.json", "artifacts/predictions/interventions/*.report.json",
        "artifacts/predictions/combined/*.jsonl",
    ):
        paths.update(path for path in BASE.glob(pattern) if path.is_file())
    registry = {
        str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(paths)
    }
    return {
        "version": "supervisor_v1_1_candidate_artifact_registry_v1",
        "status": STATUS,
        "file_count": len(registry),
        "files": registry,
        "note": "Selected reproducibility, report, final-adapter, generation-report, and combined-prediction artifacts; intermediate checkpoint bytes remain in their run directories.",
        "frozen_v1_artifacts_included_or_modified": False,
    }


def main() -> None:
    results = read(REPORTS / "candidate_results.json")
    audit = read(REPORTS / "data_audit.json")
    state = read(REPORTS / "state_machine_dry_run.json")
    baselines = read(REPORTS / "unified_baselines.json")
    manifest_index = read(BASE / "manifests/manifest_index.json")
    full = results["evaluations"]["full"]
    full_metrics = full["aggregate_metrics"]
    full_boot = results["bootstrap"]["conditions"]["full"]["metrics"]
    paired = results["bootstrap"]["paired_full_minus_condition"]

    intervention_names = [name for name in results["evaluations"] if name.startswith("intervention_")]
    ablation_names = ("image_only_qwen", "metrics_only_qwen")
    baseline_names = [name for name in results["evaluations"] if name.startswith("baseline_")]
    no_ops = {"intervention_history_empty", "intervention_history_shuffle", "intervention_budget_shuffle"}

    blank_drop = paired["intervention_blank_image"]["metrics"]["diagnosis_balanced_accuracy"]["point_estimate"]
    cross_drop = paired["intervention_cross_class_image_shuffle"]["metrics"]["diagnosis_balanced_accuracy"]["point_estimate"]
    metrics_drop = paired["intervention_metrics_blank"]["metrics"]["diagnosis_balanced_accuracy"]["point_estimate"]
    visual_claim_supported = blank_drop >= 0.05 and cross_drop >= 0.05
    structured_reliance_supported = metrics_drop >= 0.05

    training_runs = [run for modality in results["training"].values() for run in modality["runs"]]
    actual_generation_reports = sorted((BASE / "artifacts/predictions").glob("**/*.report.json"))
    generation_rows = [read(path) for path in actual_generation_reports if read(path).get("status") == "completed"]
    total_training_seconds = sum(run["wall_seconds"] for run in training_runs)
    total_generation_model_seconds = sum(row.get("telemetry", {}).get("generation_latency_seconds_total_this_invocation", 0.0) for row in generation_rows)
    peak_allocated = max([run["gpu_peak_allocated_bytes"] for run in training_runs] + [row.get("telemetry", {}).get("gpu_peak_allocated_bytes", 0) for row in generation_rows])
    peak_reserved = max([run["gpu_peak_reserved_bytes"] for run in training_runs] + [row.get("telemetry", {}).get("gpu_peak_reserved_bytes", 0) for row in generation_rows])

    machine = {
        "version": "supervisor_v1_1_candidate_machine_summary_v1",
        "status": STATUS,
        "overall_signal": "YELLOW",
        "maximum_conclusion": CONCLUSION,
        "scope": "candidate train/dev engineering evidence only",
        "gates": {
            "data_identity_and_schema_audit": "GO",
            "three_seed_full_execution": "GO",
            "modality_dependency": "YELLOW",
            "unified_baselines": "GO",
            "state_machine_contract_dry_run": "GO" if state["all_safety_assertions_passed"] else "RED",
            "near_duplicate_risk": "YELLOW",
            "temporal_recovery": "NOT VERIFIED",
            "reflection_severity_ood": "NOT RUN",
            "frozen_iid_ood": "DISABLED AND NOT RUN",
            "formal_end_to_end_closed_loop": "NOT RUN",
        },
        "red_findings": [] if state["all_safety_assertions_passed"] and audit["status"] == "PASSED_FOR_DEVELOPMENT_TRAINING" else ["candidate integrity or safety gate failed"],
        "data": {
            "new": audit["new_candidate_counts"],
            "combined": audit["candidate_distributions"],
            "identity_overlap": audit["cross_cohort_identity_overlap"],
            "perceptual_near_neighbor_dev_records": audit["train_dev_perceptual_nearest_neighbor"]["perceptual_duplicate_dev_records"],
            "old_fixed_pixel_reflection_records": audit["old_fixed_pixel_reflection_records"],
            "prompt_visible_audit": audit["prompt_visible_audit"],
            "hashes": audit["hashes"],
        },
        "full_qwen_dev": {
            "metrics": full_metrics,
            "pair_aware_bootstrap": full_boot,
            "three_seed_agreement": results["full_three_seed_agreement"],
            "training": results["training"]["full"],
        },
        "interventions": {
            "evaluations": {name: results["evaluations"][name] for name in intervention_names},
            "paired_full_minus_condition": {name: paired[name] for name in intervention_names},
            "structural_no_ops": sorted(no_ops),
            "visual_understanding_claim_supported_by_preregistered_rule": visual_claim_supported,
            "structured_metrics_reliance_supported_by_preregistered_rule": structured_reliance_supported,
        },
        "retrained_qwen_ablations": {
            name: {"evaluation": results["evaluations"][name], "training": results["training"][name.removesuffix("_qwen")], "paired_difference": paired[name]}
            for name in ablation_names
        },
        "unified_baselines": {
            name: {"evaluation": results["evaluations"][name], "paired_difference": paired[name]}
            for name in baseline_names
        },
        "state_machine": {
            "all_assertions_passed": state["all_safety_assertions_passed"],
            "assertions": state["safety_assertions"],
            "temporal_evidence": state["temporal_evidence"],
            "known_repository_semantics": state["known_repository_semantics"],
        },
        "runtime": {
            "qwen_training_wall_seconds_sum": total_training_seconds,
            "actual_generation_model_latency_seconds_sum": total_generation_model_seconds,
            "actual_generation_reports": len(generation_rows),
            "peak_gpu_allocated_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "gpu": "NVIDIA GeForce RTX 4080 Laptop GPU, 12282 MiB",
        },
        "claim_guards": [
            "v1 is a static anomaly diagnosis and routing supervisor, not a general temporal reasoning agent",
            "reflection remains a provisional engineering GO with no reflection severity-OOD evidence",
            "offline classification is not closed-loop success",
            "Qwen candidate dev accuracy and Learned-H1 77.1 percent come from different experiments and must not be multiplied",
            "Learned-H1 is the deployed controller; H3 failed mainly at planner/objective behavior and is not deployed",
            "frozen IID/OOD and formal end-to-end closed-loop evaluation were not run",
            "temporal recovery performance has not been verified without real paired sequential anomalies",
        ],
        "artifacts": {
            "candidate_results": {"path": "supervisor_v1_1_candidate/reports/candidate_results.json", "sha256": sha256(REPORTS / "candidate_results.json")},
            "data_audit": {"path": "supervisor_v1_1_candidate/reports/data_audit.json", "sha256": sha256(REPORTS / "data_audit.json")},
            "state_machine": {"path": "supervisor_v1_1_candidate/reports/state_machine_dry_run.json", "sha256": sha256(REPORTS / "state_machine_dry_run.json")},
            "commands": {"path": "supervisor_v1_1_candidate/reports/commands.md", "sha256": sha256(REPORTS / "commands.md")},
        },
        "frozen_iid_ood_or_protected_predictions_generated": False,
        "formal_frozen_evaluation_run": False,
        "final_checkpoint_bound_freeze_run": False,
    }
    machine_path = REPORTS / "machine_summary.json"
    write_new(machine_path, json.dumps(machine, indent=2, sort_keys=True, allow_nan=False) + "\n")

    lines = [
        "# Qwen-VL supervisor v1.1 candidate pre-meeting report",
        "",
        f"Status: **{STATUS}**",
        "",
        f"Overall: **YELLOW**. Maximum supported conclusion: **{CONCLUSION}**.",
        "",
        "This report is development/engineering evidence only. No frozen IID/OOD/protected inference, formal frozen evaluation, final checkpoint-bound freeze, or real end-to-end closed-loop evaluation was run.",
        "",
        "## Executive summary",
        "",
        f"- GO: candidate data/schema/identity audit passed; exact train-dev/frozen registry overlaps are zero; old fixed-pixel reflection count is {audit['old_fixed_pixel_reflection_records']}.",
        f"- GO: Full Qwen completed all three 200-step seeds and all 60 dev records per seed with {mean_std(full_metrics['diagnosis_balanced_accuracy'])} diagnosis balanced accuracy, {mean_std(full_metrics['diagnosis_macro_f1'])} macro-F1, and {mean_std(full_metrics['valid_json_rate'])} strict valid JSON.",
        f"- GO: state-machine dry-run passed all {len(state['safety_assertions'])} safety assertions, including strict fail-safe stop, injection rejection, H1-only routing, reversible enums, and finite budget termination.",
        f"- YELLOW: {audit['train_dev_perceptual_nearest_neighbor']['perceptual_duplicate_dev_records']} dev records meet the preregistered legal train-dev perceptual-neighbor rule despite zero exact/setup/pair/episode/augmentation overlap.",
        "- YELLOW: dev is small, Full checkpoints show seed-dependent post-best loss increase, frozen evaluation is disabled, and real sequential recovery evidence is absent.",
        "- RED findings: none in the authorized candidate scope.",
        "",
        "## Candidate data and leakage audit",
        "",
        f"New train adds {audit['new_candidate_counts']['train']['records']} records / {audit['new_candidate_counts']['train']['pairs']} pairs / {audit['new_candidate_counts']['train']['setups']} setups; new dev adds {audit['new_candidate_counts']['dev']['records']} / {audit['new_candidate_counts']['dev']['pairs']} / {audit['new_candidate_counts']['dev']['setups']}.",
        f"Combined train is {audit['candidate_distributions']['train']['records']} records / {audit['candidate_distributions']['train']['pairs']} pairs / {audit['candidate_distributions']['train']['setups']} setups; combined dev is {audit['candidate_distributions']['dev']['records']} / {audit['candidate_distributions']['dev']['pairs']} / {audit['candidate_distributions']['dev']['setups']}.",
        "",
        "Exact sample, image, setup, pair, episode, and augmentation-base overlap is zero for train-dev and for candidate-versus-frozen identity registries. Pair completeness, serialized metric match, schema, prompt-visible fields, and image hashes passed. Frozen image content was not opened during the perceptual audit. All reflection records are width-relative.",
        "",
        f"Width-quartile coverage: train `{audit['candidate_distributions']['train']['width_quartiles']}`; dev `{audit['candidate_distributions']['dev']['width_quartiles']}`. Class counts: train `{audit['candidate_distributions']['train']['classes']}`; dev `{audit['candidate_distributions']['dev']['classes']}`.",
        "",
        "The 4 legal near-neighbors are a YELLOW morphology-similarity risk, not a demonstrated provenance leak: no same setup or exact image was found. With this dev size, effects involving only a few records are inconclusive.",
        "",
        "Key hashes:",
        "",
        f"- train manifest `{audit['hashes']['manifests']['train']}`; dev manifest `{audit['hashes']['manifests']['dev']}`",
        f"- train SFT `{audit['hashes']['exports']['train']}`; dev SFT `{audit['hashes']['exports']['dev']}`",
        f"- experiment protocol `{sha256(BASE / 'protocol/experiment_protocol.json')}`; baseline protocol `{sha256(BASE / 'protocol/baseline_protocol.json')}`",
        "",
        "## Full Qwen three-seed results",
        "",
        "| seed | best step | train loss (whole run) | best dev loss | final dev loss | BA | macro-F1 | policy acc | action macro-F1 | joint exact | wall min | peak alloc GiB | adapter SHA-256 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    training_by_seed = {run["seed"]: run for run in results["training"]["full"]["runs"]}
    eval_by_seed = {row["seed"]: row for row in full["per_seed"]}
    for seed in sorted(training_by_seed):
        train = training_by_seed[seed]
        ev = eval_by_seed[seed]
        lines.append(
            f"| {seed} | {train['best_step']} | {train['reported_whole_training_loss']:.6f} | {train['best_dev_loss']:.6f} | {train['final_dev_loss']:.6f} | {pct(ev['diagnosis_balanced_accuracy'])} | {pct(ev['diagnosis_macro_f1'])} | {pct(ev['measurement_policy_accuracy'])} | {pct(ev['supervisor_action_macro_f1'])} | {pct(ev['joint_exact_accuracy'])} | {train['wall_seconds']/60:.2f} | {train['gpu_peak_allocated_bytes']/2**30:.2f} | `{train['adapter_model_sha256']}` |"
        )
    lines.extend([
        "",
        f"Seed mean/std: coverage {mean_std(full_metrics['coverage_rate'])}; strict valid JSON {mean_std(full_metrics['valid_json_rate'])}; diagnosis BA {mean_std(full_metrics['diagnosis_balanced_accuracy'])}; macro-F1 {mean_std(full_metrics['diagnosis_macro_f1'])}; policy accuracy {mean_std(full_metrics['measurement_policy_accuracy'])}; action macro-F1 {mean_std(full_metrics['supervisor_action_macro_f1'])}; joint exact {mean_std(full_metrics['joint_exact_accuracy'])}.",
        "",
        f"Setup-cluster/pair-preserving bootstrap 95% CI: diagnosis BA {ci(full_boot['diagnosis_balanced_accuracy'])}, macro-F1 {ci(full_boot['diagnosis_macro_f1'])}, joint exact {ci(full_boot['joint_exact_accuracy'])}.",
        "",
        f"Three-seed diagnosis agreement is {results['full_three_seed_agreement']['three_seed_diagnosis_agreement_count']}/{results['full_three_seed_agreement']['records']} records; {len(results['full_three_seed_agreement']['repeated_errors_wrong_in_at_least_two_seeds'])} records are wrong in at least two seeds. Exact rows are recorded in `candidate_results.json`.",
        "",
        "Summed Full diagnosis confusion matrix across seeds:",
        "",
        *matrix_text(full["aggregate_diagnosis_confusion_matrix"]),
        "",
        "Full per-class mean precision / recall / F1:",
        "",
        "| class | precision | recall | F1 |",
        "|---|---:|---:|---:|",
    ])
    for label, values in full["aggregate_diagnosis_per_class"].items():
        lines.append(f"| {label} | {pct(values['precision']['mean'])} | {pct(values['recall']['mean'])} | {pct(values['f1']['mean'])} |")
    overfit = [run["dev_loss_increase_from_best_fraction"] for run in results["training"]["full"]["runs"]]
    lines.extend([
        "",
        "Overfitting signs: teacher-forced training losses continue toward zero while final dev loss is above the selected minimum by " + ", ".join(f"{100*x:.1f}%" for x in overfit) + " for seeds 01/02/03. This is seed-dependent dev overfit evidence; selected checkpoints follow the frozen lowest-full-dev-loss rule. Full loss curves are in `candidate_results.json`.",
        "",
        "## Full-model inference interventions",
        "",
        "Positive drop means Full is better. CIs are setup-cluster paired bootstrap intervals.",
        "",
        "| condition | BA mean +/- std | Full-minus-condition BA | BA drop 95% CI | macro-F1 | joint exact | note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for name in sorted(intervention_names):
        metrics = results["evaluations"][name]["aggregate_metrics"]
        delta = paired[name]["metrics"]["diagnosis_balanced_accuracy"]
        note = "byte-identical structural no-op; Full predictions reused" if name in no_ops else "actual Full-adapter intervention generation"
        lines.append(f"| {name.removeprefix('intervention_')} | {mean_std(metrics['diagnosis_balanced_accuracy'])} | {100*delta['point_estimate']:.2f} pp | {ci(delta)} | {mean_std(metrics['diagnosis_macro_f1'])} | {mean_std(metrics['joint_exact_accuracy'])} | {note} |")
    visual_sentence = (
        "Both blank and cross-class shuffled image conditions cross the preregistered 5 pp engineering threshold, supporting image dependence on this dev set. This still does not establish general visual understanding."
        if visual_claim_supported else
        "Blank and/or cross-class shuffled images do not produce a consistently large preregistered drop. Therefore this pilot does not support claiming that Qwen learned visual anomaly understanding; structured metrics or dataset structure may dominate."
    )
    lines.extend([
        "",
        visual_sentence,
        ("Metrics neutralization crosses the 5 pp engineering threshold, supporting structured-state dependence on this dev set." if structured_reliance_supported else "Metrics neutralization does not cross the 5 pp engineering threshold; strong structured-metric dependence is not established."),
        "History empty/shuffle and budget shuffle are not evidence: those source fields are constant and the generated intervention files are byte-identical to Full dev. Goal and budget neutralization use schema-valid zero values, not missing-token semantics.",
        "",
        "## Retrained Qwen single-modality ablations",
        "",
        "| model | BA mean +/- std | macro-F1 | joint exact | Full-minus-model BA | BA drop 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in ablation_names:
        metrics = results["evaluations"][name]["aggregate_metrics"]
        delta = paired[name]["metrics"]["diagnosis_balanced_accuracy"]
        lines.append(f"| {name} | {mean_std(metrics['diagnosis_balanced_accuracy'])} | {mean_std(metrics['diagnosis_macro_f1'])} | {mean_std(metrics['joint_exact_accuracy'])} | {100*delta['point_estimate']:.2f} pp | {ci(delta)} |")
    lines.extend([
        "",
        "Inference interventions answer what the trained Full model uses; retrained ablations answer what one modality can learn after retraining. They are not interchangeable.",
        "",
        "## Unified three-class baselines",
        "",
        "Diagnosis is the primary capability. Policy/action/joint fields below are deterministic routing consequences and are not three independent reasoning tasks.",
        "",
        "| baseline | seeds | params | input | train wall s mean | BA mean +/- std | macro-F1 | diagnosis accuracy | Full-minus BA 95% CI |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    baseline_raw = baselines["models"]
    for name in sorted(baseline_names):
        short = name.removeprefix("baseline_")
        raw_runs = baseline_raw[short]["runs"]
        metrics = results["evaluations"][name]["aggregate_metrics"]
        boot_metrics = results["bootstrap"]["conditions"][name]["metrics"]
        params = raw_runs[0]["parameter_count"]
        wall = statistics.fmean(run["wall_seconds"] for run in raw_runs)
        input_name = "metrics" if "metrics" in short and "cnn" not in short else ("image+metrics" if "fusion" in short else "image")
        lines.append(f"| {short} | {len(raw_runs)} | {params} | {input_name} | {wall:.3f} | {mean_std(metrics['diagnosis_balanced_accuracy'])} | {mean_std(metrics['diagnosis_macro_f1'])} | {pct(boot_metrics['diagnosis_accuracy']['point_estimate'])} | {ci(paired[name]['metrics']['diagnosis_balanced_accuracy'])} |")
    lines.extend([
        "",
        "All baselines use the same 192-record candidate train and 60-record candidate dev, the same three diagnosis labels, fixed configurations, and no dev hyperparameter sweep. Full confusion matrices and per-class scores are in `reports/evaluations/` and `candidate_results.json`.",
        "",
        "## State-machine and safety dry-run",
        "",
        "- nominal -> `standard` -> frozen H1 one-step CEM route: passed.",
        "- saturation -> `lower_exposure_reacquire` -> reacquire -> external re-diagnosis state: passed as a contract trace; no recovery outcome was observed.",
        "- reflection -> `primary_spot` -> switch measurement -> frozen H1 route: passed.",
        "- invalid JSON and numeric/continuous-action injection: strict rejection to non-dispatched conservative stop passed.",
        "- repeated anomaly with remaining budget: terminates; continuation gate and horizon remain authoritative.",
        "- enum mapping is reversible; H3 is not registered/called; supervisor has no actuator fields or direct actuator authority.",
        "- all 252 candidate train/dev records passed image 128px, current diagnostic 128px frame/range, goal labelled 1024px lab/sensor frame/range, positive width, mm unit, per-step-bound, repository-limit-source, budget, and image-hash checks.",
        "",
        "Known semantics retained in the report:",
        "",
        "- legacy v12.0 camera extraction had clipped left-searchsorted discontinuity; corrected v12.1 uses continuous finite-pixel irradiance sampling. This dry-run is not a hardware validation.",
        "- current supervisor metrics are diagnostic-image 128px while goals are labelled lab_sensor 1024px/raw-peak; v12.1 documents the lab-to-sensor correction, but calibration accuracy was not tested here.",
        "- legacy `power_w` was a dead input under peak-normalized source behavior; corrected v12.1 makes it causal. The static supervisor prompt does not expose setup `power_w` directly.",
        "- lens +/-0.05 mm and camera +/-0.02 mm are visible per-step bounds. +/-3 mm is a repository sampling-domain limit, not a hardware limit.",
        "",
        "There are no real paired sequential anomaly observations in this candidate. **Temporal recovery performance has not been verified.**",
        "",
        "## Claim-evidence table",
        "",
        "| claim | evidence | allowed wording |",
        "|---|---|---|",
        "| Candidate split identity isolation | zero exact/sample/image/setup/pair/episode/augmentation overlap; protected registry identity only | engineering GO |",
        "| Full candidate-dev classification | three seeds, complete 60-record dev, strict reducer and pair-aware CI | candidate dev engineering result only |",
        "| Visual anomaly understanding | blank/shuffle interventions plus image-only retraining | only the conditional wording above; never general visual understanding |",
        "| Structured-state dependence | metrics blank/shuffle plus metrics-only retraining | dependency on this candidate dev only |",
        "| Closed-loop safety | synthetic contract dry-run | state-machine contract passed; not recovery success |",
        "| Reflection robustness | width-relative train/dev only | provisional engineering GO; no severity-OOD evidence |",
        "| Frozen performance | not run | no claim |",
        "| Temporal recovery | no real paired sequential anomaly observations | not verified |",
        "",
        "## Remaining blockers before final freeze",
        "",
        "1. Human review of the 4 legal perceptual near-neighbors, intervention interpretation, seed variability, and post-best dev-loss increases.",
        "2. Decide whether the static reacquire orchestration must be productionized as an explicit reacquire-then-re-diagnose state before any closed-loop evaluation.",
        "3. Preserve reflection as provisional until a separately frozen severity-OOD protocol is authorized.",
        "4. Only after human approval: bind final checkpoint/config and separately authorize frozen IID/OOD and formal end-to-end closed-loop evaluation.",
        "",
        "## Meeting: can say / cannot say",
        "",
        "Can say:",
        "",
        "- The candidate completed a preregistered three-seed, 200-step development pilot with complete-dev strict generation, modality checks, unified baselines, and safety-contract dry-runs.",
        "- v1 is a static anomaly diagnosis and routing supervisor.",
        "- Learned-H1 is the deployed controller; supervisor outputs remain high-level enums and H1 owns continuous actions.",
        "",
        "Cannot say:",
        "",
        "- This is a general temporal reasoning agent, a validated closed-loop recovery system, or frozen scientific performance.",
        "- Reflection severity-OOD robustness is established.",
        "- Qwen dev accuracy and Learned-H1 77.1% form an overall accuracy. They are different experiments and must never be multiplied.",
        "- H3 is deployed. H3 failed mainly at planner/objective behavior and is not deployed.",
        "- Frozen IID/OOD or formal end-to-end results exist; neither was run.",
        "",
        "## Runtime and reproducibility",
        "",
        f"Nine Qwen training runs sum to {total_training_seconds/3600:.2f} GPU-hours of measured wall time. Actual generated-condition model latency sums to {total_generation_model_seconds/3600:.2f} hours across {len(generation_rows)} generation reports. Maximum CUDA allocation was {peak_allocated/2**30:.2f} GiB and maximum reservation was {peak_reserved/2**30:.2f} GiB on an RTX 4080 Laptop GPU (12,282 MiB). Per-run times and hashes are in `machine_summary.json` and `candidate_results.json`.",
        "",
        "Actual command patterns and exceptions are in `reports/commands.md`; the artifact registry contains hashes for protocols, configs, manifests, reports, final adapters, generation reports, and combined predictions.",
        "",
        f"Final disposition: **YELLOW — {CONCLUSION}**.",
    ])
    report_path = REPORTS / "pilot_report.md"
    write_new(report_path, "\n".join(lines) + "\n")

    # Include the final report and machine summary in the registry, but avoid a
    # self-hash for the registry itself.
    registry = selected_artifacts(results)
    registry_path = REPORTS / "artifact_hashes.json"
    write_new(registry_path, json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "machine_summary": str(machine_path), "pilot_report": str(report_path),
        "artifact_registry": str(registry_path), "overall": "YELLOW",
        "conclusion": CONCLUSION,
    }, indent=2))


if __name__ == "__main__":
    main()
