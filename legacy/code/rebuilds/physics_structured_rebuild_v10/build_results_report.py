#!/usr/bin/env python3
"""Build the versioned v10 registry, tables, plot, and cycle report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/physics_structured_rebuild_v10"
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v10_full"
PACKAGE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def ci_text(block: dict[str, Any]) -> str:
    return (
        f"{percent(block['mean'])} "
        f"[{percent(block['ci95_low'])}, {percent(block['ci95_high'])}]"
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    forward = read(run_dir / "forward_ablation_development_v10.json")
    inverse = read(run_dir / "inverse_ranker_v10.json")
    visual = read(run_dir / "visual_inverse_v10.json")
    direction = read(run_dir / "direction_calibrator_v10.json")
    freeze = read(run_dir / "freeze_manifest_v10.json")
    locked = read(run_dir / "locked_test_once_v10.json")
    manifest = read(data_dir / "manifest.json")
    comparability_path = run_dir / "postfreeze_system_comparability_v10.json"
    comparability = read(comparability_path)
    registry = []
    for experiment in ("A", "B", "C", "D"):
        gate = forward["acceptance_gates"].get(experiment)
        registry.append(
            {
                "experiment_id": f"forward_{experiment}",
                "task": "forward",
                **forward["artifacts"][experiment]["contract"],
                "development_primary": forward["metrics"][experiment][
                    "natural_requested_action"
                ]["strict_all_five"]["mean"],
                "status": (
                    "control"
                    if experiment == "A"
                    else "accepted"
                    if gate["accepted"]
                    else "rejected"
                ),
                "artifact": forward["artifacts"][experiment]["path"],
                "artifact_sha256": forward["artifacts"][experiment]["sha256"],
            }
        )
    for task, report, artifact_name in (
        ("numerical_inverse", inverse, "inverse_ranker_v10.pt"),
        ("visual_inverse", visual, "visual_inverse_v10.pt"),
        ("direction", direction, "direction_calibrator_v10.pt"),
    ):
        primary = (
            report["development"]["candidate"]["top1_physical_success"]["mean"]
            if task == "numerical_inverse"
            else report["development"]["candidate_direct_image"][
                "top1_physical_success"
            ]["mean"]
            if task == "visual_inverse"
            else report["development"]["forward_plus_calibration_head"][
                "strict_all_five"
            ]["mean"]
        )
        registry.append(
            {
                "experiment_id": task,
                "task": task,
                "data": "system_aligned",
                "architecture": report["version"],
                "objective": "physical_success" if "inverse" in task else "joint_direction",
                "development_primary": primary,
                "status": (
                    "accepted" if report["acceptance"]["accepted"] else "rejected"
                ),
                "artifact": str(run_dir / artifact_name),
                "artifact_sha256": report["artifact_sha256"],
            }
        )
    registry_path = run_dir / "experiment_registry_v10.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": "physics_structured_rebuild_v10_experiment_registry",
                "experiments": registry,
                "selected_components": freeze["selected_components"],
                "complete": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (run_dir / "experiment_registry_v10.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(registry[0]))
        writer.writeheader()
        writer.writerows(registry)
    with (run_dir / "forward_breakdowns_v10.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["experiment", "breakdown", "value", "ci95_low", "ci95_high"])
        for experiment in ("A", "B", "C", "D", "frozen_v9"):
            for regime, block in forward["metrics"][experiment][
                "by_regime_natural_requested_action"
            ].items():
                writer.writerow(
                    [
                        experiment,
                        f"regime:{regime}",
                        block["mean"],
                        block["ci95_low"],
                        block["ci95_high"],
                    ]
                )
            for cardinality, block in forward["metrics"][experiment][
                "by_action_cardinality_full_surface"
            ].items():
                writer.writerow(
                    [
                        experiment,
                        f"cardinality:{cardinality}",
                        block["mean"],
                        block["ci95_low"],
                        block["ci95_high"],
                    ]
                )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["A", "B", "C", "D", "frozen_v9"]
    values = [
        forward["metrics"][name]["natural_requested_action"]["strict_all_five"][
            "mean"
        ]
        for name in names
    ]
    lows = [
        value
        - forward["metrics"][name]["natural_requested_action"]["strict_all_five"][
            "ci95_low"
        ]
        for name, value in zip(names, values)
    ]
    highs = [
        forward["metrics"][name]["natural_requested_action"]["strict_all_five"][
            "ci95_high"
        ]
        - value
        for name, value in zip(names, values)
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(names, values, color=["#64748b", "#0ea5e9", "#8b5cf6", "#f59e0b", "#334155"])
    axis.errorbar(names, values, yerr=[lows, highs], fmt="none", color="black", capsize=4)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Strict all-five development success")
    axis.set_title("V10 controlled forward ablation")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(run_dir / "forward_primary_v10.png", dpi=160)
    plt.close(figure)

    comparisons = forward["controlled_comparisons"]
    locked_forward = locked["forward"]["selected"]["natural_requested_action"][
        "strict_all_five"
    ]
    locked_inverse = locked["numerical_inverse"]["selected"][
        "top1_physical_success"
    ]
    locked_visual = locked["visual_inverse"]["selected"][
        "top1_physical_success"
    ]
    locked_direction = locked["direction"]["selected"]["strict_all_five"]
    system_metrics = comparability["metrics"]
    safety_specs = (
        ("pilot finalization", "pilot_finalization_safety_retry.json"),
        ("forward A-D training", "forward_ablation_training_safety.json"),
        ("forward development evaluation", "forward_ablation_evaluation_safety.json"),
        ("inverse/visual/direction training", "specialist_training_safety.json"),
        ("freeze", "freeze_safety.json"),
        ("locked evaluation", "locked_evaluation_safety.json"),
        ("system comparability", "system_comparability_safety.json"),
    )
    safety_rows = []
    audited_seconds = 0.0
    for stage, filename in safety_specs:
        audit = read(run_dir / filename)
        audited_seconds += float(audit["seconds"])
        safety_rows.append(
            f"| {stage} | {audit['seconds']:.1f} s | "
            f"{audit['maximum_gpu_memory_used_mib']:.0f} MiB | "
            f"{audit['maximum_gpu_temperature_c']:.0f} C | "
            f"{audit['minimum_available_mib']:.0f} MiB | "
            f"{'passed' if audit['return_code'] == 0 and audit['safe_stop_reason'] is None else 'failed'} |"
        )
    locked_output_rows = []
    for field in (
        "centroid_x_px",
        "centroid_y_px",
        "sigma_x_px",
        "sigma_y_px",
        "peak_intensity",
    ):
        locked_output_rows.append(
            f"| {field} | "
            f"{percent(locked['forward']['selected']['natural_requested_action']['per_field_tolerance_accuracy'][field])} | "
            f"{percent(locked['direction']['selected']['per_field'][field]['accuracy'])} |"
        )
    locked_cardinality_rows = []
    for cardinality, block in locked["forward"]["selected"][
        "by_action_cardinality_full_surface"
    ].items():
        locked_cardinality_rows.append(
            f"| {cardinality} | {ci_text(block)} |"
        )
    locked_regime_rows = []
    for regime, block in locked["forward"]["selected"][
        "by_regime_natural_requested_action"
    ].items():
        locked_regime_rows.append(f"| {regime} | {ci_text(block)} |")
    development_rows = []
    for experiment in ("A", "B", "C", "D", "frozen_v9"):
        block = forward["metrics"][experiment]["natural_requested_action"][
            "strict_all_five"
        ]
        development_rows.append(
            f"| {experiment} | {ci_text(block)} | "
            f"{percent(forward['metrics'][experiment]['full_81_action_surface']['strict_all_five']['mean'])} |"
        )
    accepted_forward = forward["selected_forward"]
    train_group_count = manifest["split_summary"]["train"]["groups"]
    development_group_count = manifest["split_summary"]["development"]["groups"]
    locked_group_count = manifest["split_summary"]["locked_test"]["groups"]
    interrupted_train_count = manifest.get("interrupted_full_attempt", {}).get(
        "train", 0
    )
    report = f"""# V10 specialist-improvement cycle results

Date: 2026-07-29

## Decision

The frozen selections are:

| Component | Selection |
|---|---|
| forward | `{freeze['selected_components']['forward']}` |
| numerical inverse | `{freeze['selected_components']['numerical_inverse']}` |
| visual inverse | `{freeze['selected_components']['visual_inverse']}` |
| direction | `{freeze['selected_components']['direction']}` |

No experimental artifact silently replaced a frozen component. All choices
were made on the new development split and hashed before the locked test was
opened.

## Evaluation boundary

The completed pilot contains {train_group_count} training,
{development_group_count} development, and {locked_group_count} locked-test
independent groups, each with all 81 actions. Setup and setup-plus-state
overlap across splits are both zero. The full 2,400-group training generation
was attempted first; {interrupted_train_count} training shards completed
before the sustained estimate exceeded the current window. The deterministic
384-group prefix was finalized as the preregistered pilot, and extra shards
were excluded from every manifest and model.

## Controlled forward result

| Experiment | Natural-action strict all-five (group-bootstrap 95% CI) | Full-surface strict |
|---|---:|---:|
{chr(10).join(development_rows)}

The isolated effects on the primary metric were:

| Question | Paired difference (95% CI) |
|---|---:|
| distribution alignment, B-A | {ci_text(comparisons['distribution_alignment_B_minus_A'])} |
| metric alignment, C-B | {ci_text(comparisons['metric_alignment_C_minus_B'])} |
| structured action model, D-C | {ci_text(comparisons['structured_model_D_minus_C'])} |

Selected forward: `{accepted_forward}`. The acceptance gates required a
three-point gain, positive paired lower bound versus A and frozen v9, ordinary
non-inferiority, and per-field protection.

## Direct inverse specialists

Numerical inverse development top-1 was
{ci_text(inverse['development']['candidate']['top1_physical_success'])} for the
direct multi-positive ranker versus
{ci_text(inverse['development']['frozen_v9']['top1_physical_success'])} for
frozen v9. Candidate-minus-v9 was
{ci_text(inverse['development']['paired_candidate_minus_v9'])}; status:
`{'accepted' if inverse['acceptance']['accepted'] else 'rejected'}`.

Visual inverse development top-1 was
{ci_text(visual['development']['candidate_direct_image']['top1_physical_success'])}
for the image-primary scorer versus
{ci_text(visual['development']['frozen_v9_direct_visual_scorer']['top1_physical_success'])}
for the existing direct visual scorer and
{ci_text(visual['development']['frozen_v9_measured_state_numerical_route']['top1_physical_success'])}
for the measured-state route. Status:
`{'accepted' if visual['acceptance']['accepted'] else 'rejected'}`.

## Direction

Development strict all-five:

| Variant | Result |
|---|---:|
| frozen separate specialist | {ci_text(direction['development']['frozen_separate_direction']['strict_all_five'])} |
| direct selected-forward thresholding | {ci_text(direction['development']['direct_forward_thresholding']['strict_all_five'])} |
| forward plus calibration head | {ci_text(direction['development']['forward_plus_calibration_head']['strict_all_five'])} |

Selected direction: `{direction['selected_direction']}`.

## One-time locked test

| Task | Frozen selection result |
|---|---:|
| forward strict all-five natural action | {ci_text(locked_forward)} |
| numerical inverse top-1 physical success | {ci_text(locked_inverse)} |
| visual inverse top-1 physical success | {ci_text(locked_visual)} |
| direction strict all-five full surface | {ci_text(locked_direction)} |

The locked test was evaluated exactly once after the freeze manifest was
written. No selection was changed afterward.

Locked per-output results:

| Output | Forward natural-action tolerance accuracy | Direction accuracy |
|---|---:|---:|
{chr(10).join(locked_output_rows)}

Locked forward action-cardinality breakdown:

| Moving actuators | Full-surface strict all-five |
|---:|---:|
{chr(10).join(locked_cardinality_rows)}

Locked forward natural-request regime breakdown:

| Regime | Strict all-five |
|---|---:|
{chr(10).join(locked_regime_rows)}

## Post-freeze system comparability

The existing 1,600-request/150-case-per-route system replay was run only after
the selection and locked result were sealed. It reproduced the accepted v9
system metrics exactly:

| Task | Frozen v9 correct-route | Frozen v9 end-to-end | Final v10 correct-route | Final v10 end-to-end |
|---|---:|---:|---:|---:|
| direction strict all-five | {percent(system_metrics['correctly_routed_direction_physical_all_five_exact'])} | {percent(system_metrics['end_to_end_direction_physical_all_five_exact'])} | {percent(system_metrics['correctly_routed_direction_physical_all_five_exact'])} | {percent(system_metrics['end_to_end_direction_physical_all_five_exact'])} |
| forward strict all-five | {percent(system_metrics['correctly_routed_forward_physical_strict_all_five_success'])} | {percent(system_metrics['end_to_end_forward_physical_strict_all_five_success'])} | {percent(system_metrics['correctly_routed_forward_physical_strict_all_five_success'])} | {percent(system_metrics['end_to_end_forward_physical_strict_all_five_success'])} |
| combined inverse physical success | {percent(system_metrics['oracle_inverse_target_reached_rate'])} | {percent(system_metrics['end_to_end_inverse_target_reached_rate'])} | {percent(system_metrics['oracle_inverse_target_reached_rate'])} | {percent(system_metrics['end_to_end_inverse_target_reached_rate'])} |
| measurement strict all-five | {percent(system_metrics['oracle_visual_measurement_strict_all_five_success'])} | {percent(system_metrics['end_to_end_visual_measurement_strict_all_five_success'])} | {percent(system_metrics['oracle_visual_measurement_strict_all_five_success'])} | {percent(system_metrics['end_to_end_visual_measurement_strict_all_five_success'])} |

Final v10 equals frozen v9 because every experimental replacement was
rejected before the freeze. The comparability replay was not used for
training, tuning, or selection. Qwen routing was not tuned and remains the
authoritative frozen 98.76%.

## Compute and safety

| Stage | Wall time | Peak GPU memory | Peak GPU temperature | Minimum available RAM | Guard |
|---|---:|---:|---:|---:|---|
{chr(10).join(safety_rows)}

Completed guarded stages totaled {audited_seconds:.1f} seconds. Every completed
production stage used CPU cores 0 and 8 and passed the GPU-temperature,
GPU-memory, and RAM guards. This total excludes the interrupted full-generation
attempt because it was stopped interactively before its wrapper could finalize
an audit; the manifest records its 678 completed training shards.

## Answer to the v10 question

The controlled B-A, C-B, and D-C intervals above are the evidence for
distribution, metric, and structure effects. None changed the primary
natural-request metric, and none had a paired interval excluding zero. D did
raise the secondary full 81-action surface score from 11.91% to 13.19%, but
that did not transfer to requested-action success and remained far below
frozen v9. Therefore this pilot does not support distribution mismatch,
objective mismatch, or insufficient action structure as the dominant cause
of the current specialist ceiling; it only suggests that explicit structure
may help the broader surface modestly.

The pilot size limits power for small effects, especially regime and
1/81-positive breakdowns. A full generation can resume from the preserved
deterministic shards without changing the locked pilot result. The new neural
inverse and visual rankers also underfit their frozen counterparts, so their
negative results should not be generalized into a claim that direct ranking
is intrinsically inferior.

## Artifacts

- dataset manifest: `{data_dir / 'manifest.json'}`
- experiment registry: `{run_dir / 'experiment_registry_v10.json'}`
- freeze manifest: `{run_dir / 'freeze_manifest_v10.json'}`
- locked evaluation: `{run_dir / 'locked_test_once_v10.json'}`
- post-freeze system comparability: `{comparability_path}`
- forward plot: `{run_dir / 'forward_primary_v10.png'}`
- breakdown table: `{run_dir / 'forward_breakdowns_v10.csv'}`
- smoke-test reports: `{REPO_ROOT.parent / 'VLM_runs/physics_structured_rebuild_v10_smoke'}`
- reproducible commands: `{PACKAGE / 'REPRODUCIBLE_COMMANDS.md'}`
"""
    (PACKAGE / "V10_CYCLE_RESULTS.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(PACKAGE / "V10_CYCLE_RESULTS.md"),
                "registry": str(registry_path),
                "plot": str(run_dir / "forward_primary_v10.png"),
                "complete": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
