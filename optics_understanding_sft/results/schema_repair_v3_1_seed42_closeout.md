# Schema-repair v3.1 seed-42 closeout

## Outcome

The 50-step schema-repair continuation completed, but it is not a promotion candidate. It repaired the output envelope without repairing the two discriminative decisions that motivated the run. Seeds 123 and 314 were therefore not started, the frozen thresholds were not changed, and the sealed pilot test remained untouched.

No paid API calls were made.

## Training and data verification

- Continued seed 42 for 50 optimizer steps from its corrective-v2 checkpoint.
- Saved checkpoints 25 and 50 under `/home/jiamo/VLM_runs/qwen25vl_3b_qlora_schema_repair_v3_1_seed42/`.
- Teacher-forced dev loss was 0.13736 at step 25 and 0.13797 at step 50. The lower loss did not establish generative gate performance.
- The 200-record curriculum audit passed: 100 constrained-intervention rows, 60 information-sufficiency rows, 40 preservation anchors, four visual rows, zero dev overlap, zero action-order failures, and zero sufficiency-evidence failures.
- All 32 project tests pass.

## Pre-promotion diagnostics

These are deterministic free-generation diagnostics on the beginning of the unchanged 600-record `dev_v2` ordering. They are deliberately labelled partial diagnostics, not formal all-dev promotion evaluations.

| Metric | Checkpoint 25 (122 rows) | Checkpoint 50 (120 rows) | Frozen gate |
|---|---:|---:|---:|
| Schema-valid rate | 0.992 | 0.992 | >= 0.93 |
| Sufficiency status macro-F1 | 0.403 | 0.333 | >= 0.60 |
| Diagnosis status macro-F1 | 0.835 | 0.822 | >= 0.60 |
| Control status macro-F1 | 0.460 | 0.403 | >= 0.60 |
| Feasible-control recall | 0.813 | 0.938 | >= 0.60 |
| Infeasible-control recall | 0.200 | 0.071 | >= 0.60 |
| Feasible simulator success | 0.500 | 0.562 | >= 0.40 |

Checkpoint 50 predicted all 15 insufficient-information examples as `answerable` and 13 of 14 infeasible control examples as `feasible`. Checkpoint 25 showed the same direction: 14 of 15 insufficient examples became `answerable`, and 12 of 15 infeasible controls became `feasible`.

The repair therefore solved syntax, not understanding. Running two more random seeds would test variance around a failed intervention and would not satisfy the protocol's requirement that seed 42 first show a viable repair. Full 600-row generation was stopped after the diagnostic gate failure to conserve local compute.

## Next experiment

The next dataset/training change should target decision-token learning rather than add more near-duplicate records:

1. Build a fresh, scenario-disjoint repair set with 1:1 feasible/infeasible control pairs, not the current 4:1 control ratio. Each pair should have closely matched visible errors but different correct feasibility.
2. Keep sufficiency at 1:1, but add hard minimal pairs in which one prompt-visible field alone flips the label. Audit that no lexical cue predicts the class.
3. Shorten targets and weight the decision-bearing tokens (`status`, action validity, and nonzero actuator choice) so long numeric evidence cannot dominate token loss.
4. Use a two-stage continuation: a short balanced decision warm-up followed by mixed seven-task preservation examples. Select only with deterministic free generation.
5. Repeat seed 42 first on a stratified diagnostic slice containing equal status classes. Run all 600 dev rows and seeds 123/314 only if every status gate is plausibly met without anchor regression.

This remains a multi-task causal-understanding benchmark: simulator replay, intervention success, counterfactual consistency, ambiguity, and held-out scenario groups stay in the evaluation. The proposed decision warm-up is not a replacement with scalar regression.
