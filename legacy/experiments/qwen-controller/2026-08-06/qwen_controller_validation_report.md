# Qwen Controller Overnight Validation v1

## 1. Disposition

**Do not score Qwen controller selection.** Two preregistered gates block it: (1) the current trained Qwen adapter is diagnosis/routing-only and has no legal mapping to controller strategies; (2) the repaired candidate-only controller acceptable-set audit is RED. This is an engineering validation report, not a scientific conclusion or frozen evaluation.

## 2. Four independent sub-gates

| Gate | Status | Reason |
|---|---|---|
| `INTERFACE_GATE` | RED | Available trained Qwen predicts diagnosis/routing, not controller fields; native timeout also has no production exception-to-fallback wrapper. |
| `LABEL_GATE` | RED | Repaired candidate-only acceptable set has 25.76% K5/K10 Jaccard <0.5 (maximum 10%) and 75% increased-budget material change. |
| `SELECTOR_VALUE_GATE` | NOT_EVALUABLE | It requires a non-RED reliable label target and confirmation-root oracle/best-fixed comparison. |
| `QWEN_PERFORMANCE_GATE` | NOT_EVALUABLE | No compatible trained controller selector exists; diagnosis results were not repurposed. |

## 3. P0 findings

* Reconstructed 36 H1 configurations. They share frozen H1 CEM compute (population 24, elites 6, iterations 3, 72 candidate evaluations) and differ only in proposal/search, scoring/risk, or mask fields.
* The available Qwen2.5-VL pilot adapter outputs `diagnosis`, `measurement_policy`, and `supervisor_action`. Its closed-loop adapter calls frozen CEM but deliberately cannot select CEM configuration, proposal scale, risk mode, or continuous action.
* `qwen_h1_meta_v0` has the intended strict controller-field schema, but its three preregistered controller checkpoints are all unavailable because its information-sufficiency gate stopped training. No compatible adapter was found.
* Strict fixture tests reached parser → compiler → guarded H1 CEM → bounded action; invalid output fell back to default H1. Native adapter timeout propagates without a production controller dispatcher, which remains an interface blocker. A matched-root sentinel also showed that `default` and neutral `unbiased_unknown/default/standard` yield exactly the same action and realized cost, so those two metadata labels are not behaviorally separate on the sentinel.

## 4. Pilot (candidate-dev only)

The P0 pilot ran 18 equal-compute one-step episodes: three preregistered dev records × three fixed real-codebook strategies × two paired roots. Mean realized cost: 8.110522; strict success: 27.8%. Exact per-strategy values are in `aggregated_metrics.json`; these point estimates are not a label-stability or selector-quality claim.

## 5. Label/stability gate

Existing repaired candidate-only evidence remains RED under its frozen criteria: K5/K10 Jaccard was below 0.5 on 25.76% of records (maximum 10%), and an increased-budget check materially changed 75% of its fixed subset. No thresholds, splits, or gates were changed. Therefore label-based Qwen accuracy, acceptable-set hit rate, oracle-headroom recovery, and comparisons against best-fixed/oracle are **not evaluable**.

## 6. Required next work

Create a controller-target Qwen checkpoint whose output is exactly the `qwen_h1_meta_v0` schema (or an explicitly versioned reduced-strategy field), add a production exception-to-fallback wrapper that logs events, and repair/revalidate the acceptable-set target before any full Qwen controller evaluation.

## 7. Commands, tests, scope, and artifacts

`run_overnight_validation.py --run-all` completed the fixture and P0 pilot with peak RSS 849,444 KiB (829.5 MiB; raw 18-episode pilot peak 848,908 KiB). Targeted existing contract/compiler/controller tests: **47 passed**. The first controlled runner attempt stopped before any rollout because of an audit-script prompt-contract loading error; the corrected run completed successfully. Only candidate train/dev manifests and prior candidate-only artifacts were read. No candidate-eval, frozen, protected, IID/OOD, or test record was read, predicted, or evaluated.

Produced artifacts: `RUN_STATUS.md`, `execution_plan.json`, `configuration_inventory.md`, `qwen_interface_inventory.md`, `qwen_interface_audit.md`, `qwen_fallback_tests.json`, `reduced_strategy_manifest.json`, `raw_episode_results.jsonl`, `raw_qwen_predictions.jsonl` (empty by Fallback C), `aggregated_metrics.json`, `controller_stability_metrics.json`, `tomorrow_update_cn.md`, `commands.md`, `input_hashes.json`, and `artifact_hashes.json`.

Incomplete by gate (not omission): Phase 5 multi-root stability, Phase 6 fixed/oracle necessity baselines, and Phase 8 Qwen controller scoring. They would violate the preregistered stop conditions rather than resolve uncertainty.
