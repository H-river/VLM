#!/usr/bin/env python3
"""Build the evidence-led final candidate report and reproducibility manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/qwen_reasoning_plan_selector"
SEEDS = (2026080401, 2026080402, 2026080403)


def load(name: str) -> dict[str, Any]:
    return json.loads((ARTIFACT / name).read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_line(name: str, value: dict[str, Any]) -> str:
    return f"| {name} | {value['strict_success_rate']:.4f} | {value['terminal_normalized_error_mean']:.4f} | {value['steps_mean']:.3f} | {value['boundary_risk_cost_mean']:.4f} |"


def invocation_runtime_audit(seed_dir: Path) -> dict[str, Any]:
    """Count each persisted trainer invocation once, including safe resumes.

    Completed invocations use the script-recorded wall time.  An invocation
    terminated after a safe checkpoint has no final manifest, so its
    start-to-last-persisted-log interval is retained as an explicit lower
    bound instead of silently dropping that compute from the report.
    """

    invocations = []
    for log_path in sorted(seed_dir.glob("trainer_log.*.jsonl")):
        stamp = log_path.name.removeprefix("trainer_log.").removesuffix(".jsonl")
        start_path = seed_dir / f"run_manifest.{stamp}.start.json"
        final_path = seed_dir / f"run_manifest.{stamp}.json"
        start_manifest = json.loads(start_path.read_text())
        started = datetime.fromisoformat(start_manifest["started_at_utc"])
        final_manifest = json.loads(final_path.read_text()) if final_path.exists() else None
        if final_manifest and final_manifest.get("status") == "completed":
            seconds = float(final_manifest["training_result"]["runtime_wall_seconds"])
            source = "completed_manifest_wall_seconds"
        else:
            timestamps = [
                datetime.fromisoformat(json.loads(line)["logged_at_utc"])
                for line in log_path.read_text().splitlines()
                if line.strip()
            ]
            if not timestamps:
                raise RuntimeError(f"trainer invocation has no persisted timing evidence: {log_path}")
            seconds = max(0.0, (max(timestamps) - started).total_seconds())
            source = "start_to_last_persisted_log_lower_bound"
        invocations.append({"stamp": stamp, "seconds": seconds, "source": source})
    return {
        "invocation_count": len(invocations),
        "effective_wall_seconds_lower_bound": sum(row["seconds"] for row in invocations),
        "invocations": invocations,
    }


def main() -> None:
    audit, equality, status = load("anti_collapse_audit.json"), load("gain_equality_audit.json"), load("rollout_run_status.json")
    identity, rollout_config = load("manifest.json"), load("rollout_config.json")
    training_protocol = json.loads((ROOT / "qwen_reasoning_plan_selector_candidate/training_protocol.json").read_text())
    evaluation_path = ARTIFACT / "confirmation_evaluation.json"
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
    robust_path = ARTIFACT / "confirmation_13seed_evaluation.json"
    robust = json.loads(robust_path.read_text()) if robust_path.exists() else None
    ablation_path = ARTIFACT / "intervention_ablation_audit.json"
    ablation = json.loads(ablation_path.read_text()) if ablation_path.exists() else None
    probe_audit_path = ARTIFACT / "probe_coupling_intervention_audit.json"
    probe_audit = json.loads(probe_audit_path.read_text()) if probe_audit_path.exists() else None
    training = {}
    for seed in SEEDS:
        path = ARTIFACT / f"training/seed_{seed}/run_manifest.latest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            seed_dir = path.parent
            training[str(seed)] = {
                "status": manifest.get("status"),
                "successful_invocation_wall_seconds": manifest.get("training_result", {}).get("runtime_wall_seconds"),
                "best_dev_checkpoint": manifest.get("training_result", {}).get("best_dev_checkpoint"),
                "final_adapter_sha256": manifest.get("adapter_dtype_invariant", {}).get("final_adapter", {}).get("sha256"),
                **invocation_runtime_audit(seed_dir),
            }
    training_runtime = sum(float(row["effective_wall_seconds_lower_bound"]) for row in training.values())
    if not audit["gate_passed"]:
        decision = "NO-GO"
        rationale = "The real-rollout information-sufficiency gate failed, so SFT was not permitted."
    elif evaluation is None:
        decision = "YELLOW"
        rationale = "The plan labels passed the gate, but learned-selector confirmation evaluation is incomplete."
    else:
        qwen = robust["selectors"]["qwen_borda_3seed"] if robust else evaluation["qwen_borda_3seed"]
        improvement_flags = [qwen["beats_best_fixed_strict_success"], qwen["beats_visual_lookup_strict_success"], qwen["beats_numerical_only_strict_success"]]
        no_collapse = len(qwen["plan_distribution"]) >= 2 and qwen.get("plan_selection_entropy_bits", 0.0) >= 0.5
        regular_causal = False
        if ablation:
            required = [ablation["matched_physical_interventions"].get(name, {}) for name in ("target_swap", "boundary_swap", "image_family_swap")]
            regular_causal = all(row.get("pairs", 0) > 0 and (row.get("flip_presence_agreement_rate") or 0.0) >= 2.0 / 3.0 and (row.get("exact_plan_transition_agreement_rate") or 0.0) >= 1.0 / 3.0 for row in required)
        probe_causal = bool(probe_audit and probe_audit.get("same_dominant_error_component_verified") and probe_audit.get("pairs", 0) >= 3 and probe_audit.get("flip_presence_agreement_rate", 0.0) >= 2.0 / 3.0 and probe_audit.get("exact_plan_transition_agreement_rate", 0.0) >= 1.0 / 3.0)
        causal_pass = regular_causal and probe_causal
        if all(improvement_flags) and no_collapse and equality["passed"] and causal_pass:
            decision = "GO"
            rationale = "The three-seed Qwen selector exceeded all preregistered baselines without collapse under fixed gain and passed target, boundary, image, and physical probe-coupling causal checks."
        elif any(improvement_flags) or (all(improvement_flags) and not causal_pass):
            decision = "YELLOW"
            rationale = "The Qwen selector showed some closed-loop plan value, but did not satisfy every baseline, collapse, and causal-intervention requirement for a positive conclusion."
        else:
            decision = "NO-GO"
            rationale = "The Qwen selector did not exceed the preregistered selector baselines on untouched candidate confirmation."
    revision = (ARTIFACT / "plan_bank_revision.json")
    revision_data = json.loads(revision.read_text()) if revision.exists() else None
    total_formal = int(status["completed_unique_episodes_in_output"])
    archived_status = ROOT / "artifacts/qwen_reasoning_plan_selector_revision0/rollout_run_status.json"
    archived_formal = int(json.loads(archived_status.read_text())["completed_unique_episodes_in_output"]) if archived_status.exists() else 0
    initial60_status = ARTIFACT / "rollout_run_status_initial60.json"
    extra_status = ARTIFACT / "confirmation_extra_10seed_status.json"
    extra_formal = int(json.loads(extra_status.read_text())["completed_episodes"]) if extra_status.exists() else 0
    probe_status_path = ARTIFACT / "probe_coupling_confirmation/status.json"
    probe_formal = int(json.loads(probe_status_path.read_text())["completed_episodes"]) if probe_status_path.exists() else 0
    total_runtime = float(status["invocation_wall_seconds"])
    if initial60_status.exists():
        total_runtime += float(json.loads(initial60_status.read_text())["invocation_wall_seconds"])
    if extra_status.exists():
        total_runtime += float(json.loads(extra_status.read_text())["wall_seconds"])
    if archived_status.exists():
        total_runtime += float(json.loads(archived_status.read_text())["invocation_wall_seconds"])
    if probe_status_path.exists():
        total_runtime += float(json.loads(probe_status_path.read_text())["wall_seconds"])
    phase_times = {
        "group_generation": float(rollout_config.get("generation_elapsed_seconds", 0.0)),
        "revision0_rollouts": float(json.loads(archived_status.read_text())["invocation_wall_seconds"]) if archived_status.exists() else 0.0,
        "revision1_initial60_rollouts": float(json.loads(initial60_status.read_text())["invocation_wall_seconds"]) if initial60_status.exists() else 0.0,
        "revision1_supplemental_rollouts": float(status["invocation_wall_seconds"]),
        "extra_confirmation_rollouts": float(json.loads(extra_status.read_text())["wall_seconds"]) if extra_status.exists() else 0.0,
        "probe_coupling_confirmation_rollouts": float(json.loads(probe_status_path.read_text())["wall_seconds"]) if probe_status_path.exists() else 0.0,
        "qwen_sft": training_runtime,
    }
    lines = [
        "# Qwen reasoning-based fixed-gain meta-controller candidate report",
        "",
        f"**Decision: {decision}.** {rationale}",
        "",
        "## Scope and invariants",
        "",
        f"- Candidate-only evaluation; frozen/protected enabled: **false**.",
        f"- Real corrected-simulator formal episodes: **{total_formal + archived_formal + extra_formal + probe_formal}** ({total_formal + extra_formal + probe_formal} final plan bank/interventions; {archived_formal} pre-revision evidence).",
        f"- Recorded formal rollout wall time: **{total_runtime:.1f} s**; final mean episode: **{status['mean_episode_seconds_this_invocation']:.3f} s**.",
        f"- Recorded three-seed Qwen SFT wall time: **{training_runtime:.1f} s**; rollout plus SFT wall time: **{total_runtime + training_runtime:.1f} s** (excludes lightweight export/analysis).",
        f"- Same gain/bounds/CEM budget/horizon/max steps/tolerance/checkpoint/simulator audit: **{'PASS' if equality['passed'] else 'FAIL'}**.",
        f"- Learned-H1 checkpoint SHA-256: `{equality['checkpoint_sha256']}`.",
        f"- Git commit/branch: `{identity['git_commit']}` / `{identity['git_branch']}`; commit or push performed: **{identity['commit_or_push_performed']}**.",
        f"- Group/confirmation manifest SHA-256: `{identity['artifact_files']['groups_sha256']}` / `{identity['artifact_files']['confirmation_manifest_sha256']}`.",
        f"- Source train/dev/full suite SHA-256: `{rollout_config['source_files']['source_train_sha256']}` / `{rollout_config['source_files']['source_dev_sha256']}` / `{rollout_config['source_files']['source_suite_sha256']}`.",
        f"- Fixed controller/base simulator/v12 config hashes: `{rollout_config['fixed_controller_config_hash']}` / `{rollout_config['source_files']['base_simulator_config_sha256']}` / `{rollout_config['source_files']['v12_config_sha256']}`.",
        f"- Major measured phase wall times (s): `{json.dumps(phase_times, sort_keys=True)}`.",
        "- Qwen never emits actuator deltas, gain, bounds, horizon, or CEM settings; it emits only a complete plan ranking and selected plan.",
        "",
        "## Independent plan value and anti-collapse gate",
        "",
        f"- Gate: **{'PASS' if audit['gate_passed'] else 'FAIL'}**; decisive/ambiguous train+dev groups: **{audit['decisive_groups']}/{audit['ambiguous_groups']}**.",
        f"- Decisive winner counts: `{json.dumps(audit['winner_counts'], sort_keys=True)}`.",
        f"- Plan-label entropy: **{audit['plan_label_entropy_bits']:.4f} bits**; matched-intervention oracle flip rate: **{audit['effective_matched_intervention_flip_rate']}**.",
        f"- Best fixed plan: **{audit['selector_closed_loop_value']['best_fixed']['plan']}**.",
        f"- Oracle plan selection strict success/error: **{audit['selector_closed_loop_value']['oracle']['strict_success_rate']:.4f} / {audit['selector_closed_loop_value']['oracle']['terminal_normalized_error_mean']:.4f}**.",
        f"- Best fixed strict success/error: **{audit['selector_closed_loop_value']['best_fixed']['strict_success_rate']:.4f} / {audit['selector_closed_loop_value']['best_fixed']['terminal_normalized_error_mean']:.4f}**.",
        f"- Visual lookup strict success/error: **{audit['selector_closed_loop_value']['visual_diagnosis_lookup_lobo']['strict_success_rate']:.4f} / {audit['selector_closed_loop_value']['visual_diagnosis_lookup_lobo']['terminal_normalized_error_mean']:.4f}**.",
        f"- Numerical-only strict success/error: **{audit['selector_closed_loop_value']['numerical_only_mlp_lobo']['strict_success_rate']:.4f} / {audit['selector_closed_loop_value']['numerical_only_mlp_lobo']['terminal_normalized_error_mean']:.4f}**.",
        "",
        "| Fixed plan / oracle | Strict success | Terminal max normalized error | Steps | Boundary risk |",
        "|---|---:|---:|---:|---:|",
        *[metric_line(name, value) for name, value in audit["selector_closed_loop_value"]["all_fixed_plans"].items()],
        metric_line("oracle selector", audit["selector_closed_loop_value"]["oracle"]),
        "",
        f"Real execution contract: `{json.dumps(audit['execution_contract'], sort_keys=True)}`.",
    ]
    if revision_data:
        lines += ["", "## Single evidence-backed plan-bank revision", "", f"- Trigger: {revision_data['trigger']}", f"- Change: {revision_data['change']}", f"- Confirmation was not inspected or used: **{revision_data['confirmation_not_used']}**."]
    if evaluation:
        base = robust["selectors"] if robust else evaluation["baselines"]
        qwen = base["qwen_borda_3seed"] if robust else evaluation["qwen_borda_3seed"]
        oracle_key = "oracle_13seed" if robust else "oracle"
        seed_label = "13-seed confirmation" if robust else "3-seed confirmation"
        lines += ["", f"## Untouched candidate-confirmation closed loop ({seed_label})", "", "| Selector | Strict success | Terminal max normalized error | Steps | Boundary risk |", "|---|---:|---:|---:|---:|", metric_line("Qwen 3-training-seed Borda", qwen), metric_line(f"Best fixed ({evaluation['best_fixed_chosen_without_confirmation']})", base["best_fixed"]), metric_line("Direct all-five", base["direct_all_five"]), metric_line("Frequency-matched random", base["frequency_matched_random"]), metric_line("Visual diagnosis lookup", base["visual_diagnosis_lookup"]), metric_line("Numerical-only MLP", base["numerical_only_mlp"]), metric_line("Oracle", base[oracle_key]), "", f"Qwen plan distribution/entropy: `{json.dumps(qwen['plan_distribution'], sort_keys=True)}` / **{qwen['plan_selection_entropy_bits']:.4f} bits**.", f"Qwen oracle strict-success/error regret: **{qwen['oracle_strict_success_regret']:.4f} / {qwen['oracle_terminal_error_regret']:.4f}**.", f"Per-visual-family selector results: `{json.dumps((robust or evaluation)['per_visual_family'], sort_keys=True)}`."]
        qwen_selection = evaluation["selector_selections"]["qwen_borda_3seed"]
        oracle_selection = robust["oracle_13seed_selection"] if robust else evaluation["oracle_selection"]
        failures = [{"group_id": group_id, "qwen": qwen_selection[group_id], "oracle": oracle_selection[group_id]} for group_id in sorted(qwen_selection) if qwen_selection[group_id] != oracle_selection[group_id]]
        lines += ["", "## Failure cases", "", f"Qwen/oracle plan mismatches ({len(failures)}/{len(qwen_selection)} groups): `{json.dumps(failures, sort_keys=True)}`."]
    if ablation:
        lines += ["", "## Intervention and input ablations", "", f"Image/probe ablations: `{json.dumps(ablation['image_and_probe_ablation'], sort_keys=True)}`.", f"Matched target/boundary/image-family interventions: `{json.dumps(ablation['matched_physical_interventions'], sort_keys=True)}`."]
    if probe_audit:
        lines += [f"Physical same-dominant-error probe-coupling interventions: `{json.dumps(probe_audit, sort_keys=True)}`."]
    lines += ["", "## Training", "", f"Frozen QLoRA protocol and seeds: `{json.dumps(training_protocol, sort_keys=True)}`.", f"Three-seed training records: `{json.dumps(training, sort_keys=True)}`." if training else "SFT was not executed because the gate did not permit it.", "", "## Limitations", "", "- Candidate-only corrected-simulator evidence; no frozen/protected split was accessed, so external or deployment generalization is not claimed.", "- Only 20 decisive train and 7 decisive dev groups supervise Qwen; strong late-training overfit makes best-dev checkpoint selection essential.", "- Confirmation contains 15 setup-disjoint groups. Thirteen physical CEM seeds reduce optimizer noise but do not increase the number of independent optical setups.", "- Closed-loop selector values replay the selected plans' already executed, common-random-number real rollouts; the selector cannot alter gain, bounds, CEM budget, horizon, or continuous actions.", "- Intervention flips are evidence of causal consistency only when they agree with paired physical oracle outcomes; neither explanations nor performance drops alone count as reasoning evidence.", "", "## Artifact index", "", "- `system_audit.md`: system semantics and output-to-execution audit", "- `plan_contracts.json`: executable plan definitions and trajectory evidence", "- `gain_equality_audit.json`: same-gain/bound/budget proof", "- `anti_collapse_audit.json`: plan value, baselines, ambiguity, intervention flips, and gate", "- `final_outcome_audit.json`: post-training confirmation-inclusive outcomes", "- `plan_outcomes.csv`: per-group real-rollout rankings", "- `rollout_results.jsonl`: complete actions, measurements, phases, and physical outcomes", "- `confirmation_evaluation.json`: untouched-confirmation three-seed selector comparison", "- `confirmation_13seed_evaluation.json`: high-power thirteen-CEM-seed confirmation comparison", "- `intervention_ablation_audit.json`: image/probe ablations and physical matched-pair flips", "- `probe_coupling_intervention_audit.json`: same-dominant-error physical probe-coupling results", ""]
    (ARTIFACT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    artifacts = {}
    for path in sorted(ARTIFACT.rglob("*")):
        if path.is_file() and path.name not in {"reproducibility_manifest.json", "summary.json"} and "checkpoint-" not in str(path) and "final_adapter" not in str(path):
            artifacts[path.relative_to(ARTIFACT).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    repro = {"candidate_only": True, "decision": decision, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "commands": ["python -m qwen_reasoning_plan_selector_candidate.generate_groups", "python -m qwen_reasoning_plan_selector_candidate.run_rollouts", "python -m qwen_reasoning_plan_selector_candidate.analyze_rollouts", "python -m qwen_reasoning_plan_selector_candidate.export_sft", "python -m qwen_reasoning_plan_selector_candidate.make_training_configs", "python -m qwen_reasoning_plan_selector_candidate.training --config <seed-config>", "python -m qwen_reasoning_plan_selector_candidate.evaluate_selector", "python -m qwen_reasoning_plan_selector_candidate.build_report"], "artifacts": artifacts, "commit_or_push_performed": False, "frozen_or_protected_enabled": False}
    (ARTIFACT / "reproducibility_manifest.json").write_text(json.dumps(repro, indent=2, sort_keys=True) + "\n")
    (ARTIFACT / "summary.json").write_text(json.dumps({"decision": decision, "rationale": rationale, "formal_episodes": total_formal + archived_formal + extra_formal + probe_formal, "final_plan_bank_episodes": total_formal + extra_formal + probe_formal, "archived_pre_revision_episodes": archived_formal, "extra_confirmation_episodes": extra_formal, "probe_coupling_confirmation_episodes": probe_formal, "formal_rollout_wall_seconds": total_runtime, "qwen_sft_wall_seconds": training_runtime, "recorded_rollout_plus_sft_wall_seconds": total_runtime + training_runtime, "gate_passed": audit["gate_passed"], "training": training}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "report": str(ARTIFACT / 'report.md')}, sort_keys=True))


if __name__ == "__main__":
    main()
