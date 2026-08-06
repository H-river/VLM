# v12 Oracle versus learned MPC H1/H3 diagnosis

**Status:** final. The matched 48-case primary evaluation, the single labelled
oracle-only retry, aggregation, validation, and report are complete.

## Executive finding

The current learned forward model already supports useful closed-loop **H1
MPC**, but the current **H3 planner/objective does not**. On 48 fresh,
group-disjoint, corrected-simulator targets, learned H1 improved every case and
reached strict all-five tolerance in 37/48 (77.1%). Oracle H1 reached 41/48
(85.4%). In contrast, learned H3 reached 14/48 (29.2%) and Oracle H3 reached
15/48 (31.3%). Learned H3 is statistically indistinguishable from Oracle H3,
while both H3 controllers are decisively worse than their H1 counterparts.

This matched result rules out forward-model rollout error as the primary cause
of the H3 failure. The exact-simulator H3 planner shows the same failure, and
its one-step-reachable behavior shows receding-horizon deferral: at the first
decision Oracle H3 predicts terminal success in 9/16 cases but executes a first
action that succeeds in only 3/16. The terminal-heavy objective can plan to
arrive later, execute only the first partial move, then replan and defer again.

The learned model is not perfect. Its selected-sequence error increases with
depth and its H3 false-improvement rate is higher near actuator bounds.
Nevertheless, ranking remains strong, selected actions are not in the training
tail, terminal improvement remains highly correlated with simulator truth, and
the learned-versus-oracle H3 gap is negligible compared with the H3-versus-H1
gap.

## Locked evidence and validity

- Starting commit:
  `aa7e1f66cfcb497cd5c5ff39c394520668f75584`, branch
  `optics-sft-smoke-test`; the dirty user worktree was preserved.
- Simulator semantics: corrected, versioned
  `v12_sensor_power_semantics_v1` only.
- Primary config SHA-256:
  `2a9694eb493da27468f73fa0258705ce7d703842c2654a4240bffb2facf4c1cf`.
- Suite SHA-256:
  `c0c8d1dd9bfbde9ac5a23323ead231dd542ddc4dabae65cf05949efb6fbd49b5`.
- Checkpoint SHA-256:
  `d9b30627c80817f6ecade1959d8cc9e91e7a9de9cc51485153fdbfaa173aca2e`.
- Dataset-manifest SHA-256:
  `f4d6109cb386405434147e02bb97874f658ad3ae2d0d93bdd312e373d0d661fd`.
- Root seed `2026073107`; bootstrap seed `2026073109`, 4,000
  group-level bootstrap samples.
- Primary CEM: population 24, elites 6, three iterations, horizons 1/3,
  four maximum executed steps, identical per-case planner seeds.
- Suite: 48 independent groups and setup hashes; 16 one-step interior, 16
  multi-step interior, and 16 boundary/clipping; 16 each low/medium/high
  initial distance. There is zero overlap with prior train/dev/test or smoke
  groups.
- Every initial state fails strict tolerance. Every target is exactly
  reachable by corrected-simulator replay of evaluator-only `q_goal`.
- Validation passed for 192/192 primary records, 5,920 learned candidate
  audits, and 175 learned-H3 selected sequences counterfactually replayed at
  depths 1, 2, and 3.
- All controller records assert that `q_goal`, future simulator state,
  reachability labels, and evaluator candidate truth were not planner inputs.
- H3 always executes only its first action, observes the corrected simulator,
  and replans. Requested and effective actions, positions, states, bounds,
  prediction residuals, calls, and runtime are retained per step.
- Regression and deterministic smoke validation passed: 40 tests passed and
  three expected failures; the six-episode smoke rerun was byte/mtime
  resumable.

The prior artifacts were checked rather than accepted from the prompt. The
128-group checkpoint reports test normalized MAE `0.157875`, strict accuracy
`0.899457`, paired direction-sign accuracy `0.805310`, median relative
directional-Jacobian error `0.038916`, and rollout MAE H1/H3/H5
`0.162488/0.460972/0.702097`. Its three earlier oracle episodes all improved
and one reached strict success. Centroid Y was the weakest held-out output.

## Primary closed-loop result

Confidence intervals are independent-group bootstrap intervals, not
step-level intervals.

| Controller | Strict success (95% CI) | Any improvement | False improvement | Final distance mean / median | Mean final / initial | Median steps |
|---|---:|---:|---:|---:|---:|---:|
| Oracle H1 | 85.4% [75.0, 93.8] | 100.0% | 0.0% | 0.827 / 0.650 | 0.051 | 2 |
| Oracle H3 | 31.3% [18.8, 43.8] | 97.9% | 0.0% | 4.258 / 2.152 | 0.215 | 4 |
| Learned H1 | 77.1% [64.6, 87.5] | 100.0% | 0.8% | 1.703 / 0.600 | 0.087 | 2 |
| Learned H3 | 29.2% [16.7, 41.7] | 97.9% | 6.1% | 4.627 / 2.655 | 0.267 | 4 |

“False improvement” is conditional on the model predicting improvement:
1/121 learned-H1 decisions and 9/148 predicted-improving learned-H3 decisions.
Across all learned-H3 executed steps the latter is 9/175, or 5.1%.

Planner rollout cost differs sharply by backend: mean simulator calls are
173.4/759.5 for Oracle H1/H3, while learned H1/H3 use 181.5/787.5 model
rollout calls and 52.9/233.3 simulator calls (environment execution plus
evaluator audits). Mean planner runtime is 167.1/739.9 seconds for Oracle
H1/H3 and 0.127/0.509 seconds for learned H1/H3.

### Paired independent-group differences

Differences are named `right minus left`; positive final-distance difference
is worse for the right-hand controller.

| Pair | Strict-success difference (95% CI) | Mean final-distance difference (95% CI) |
|---|---:|---:|
| Oracle H3 − Oracle H1 | −54.2 pp [−68.8, −39.6] | +3.431 [+2.161, +4.954] |
| Learned H3 − Learned H1 | −47.9 pp [−62.5, −33.3] | +2.924 [+1.773, +4.320] |
| Learned H1 − Oracle H1 | −8.3 pp [−20.8, +2.1] | +0.875 [−0.054, +2.485] |
| Learned H3 − Oracle H3 | −2.1 pp [−12.5, +8.3] | +0.369 [−1.534, +2.370] |

The model gap is modest and statistically unresolved at each matched horizon.
The H3 horizon gap is large, consistent, and excludes zero for both backends.

### Strict success by target stratum and initial distance

| Split | Oracle H1 | Oracle H3 | Learned H1 | Learned H3 |
|---|---:|---:|---:|---:|
| One-step interior | 100.0% | 62.5% | 100.0% | 43.8% |
| Multi-step interior | 87.5% | 18.8% | 75.0% | 25.0% |
| Boundary/clipping | 68.8% | 12.5% | 56.3% | 18.8% |
| Low initial distance | 100.0% | 62.5% | 100.0% | 50.0% |
| Medium initial distance | 93.8% | 25.0% | 87.5% | 25.0% |
| High initial distance | 62.5% | 6.3% | 43.8% | 12.5% |

All four controllers degrade on boundary/high-distance cases, but H3 is worse
than H1 in every split. Full means, medians, ratios, motion, confidence
intervals, and per-output rates are in `aggregates/report_tables/`.

## Learned prediction and CEM diagnostics

### Selected actions and first-step prediction

- Selected-action H1 normalized prediction MAE is `0.358` for learned H1 and
  `0.287` for learned H3.
- Predicted-versus-actual improvement Spearman correlation is `0.976` and
  `0.972`, respectively (Pearson `0.978` and `0.984`).
- The model predicts improvement while the simulator worsens on 0.8% of H1
  predicted-improving decisions and 6.1% of H3 predicted-improving decisions.
- For learned H3, false improvement is more common on action-bound-adjacent
  steps (10.7%, 6/56) than ordinary steps (2.5%, 3/119). Learned H1 has
  0/51 adjacent and 1/70 ordinary false improvements.

### H3 recursive rollout

| Depth | Mean normalized MAE | Mean normalized L-infinity |
|---:|---:|---:|
| 1 | 0.287 | 0.900 |
| 2 | 0.367 | 1.136 |
| 3 | 0.492 | 1.482 |

Of 175 selected sequences, 80 (45.7%) are approximately linear, 39 (22.3%)
sublinear, and 56 (32.0%) superlinear under the preregistered second-difference
rule. The aggregate curve increases smoothly; it is not an exploding
trajectory. Depth-3 is 1.71 times depth-1 on these planner-selected
sequences.

The error is centroid-dominated. Centroid Y is the largest normalized
residual in 80/175 sequences at depth 1 and 86/175 at depth 3; centroid X is
largest in 74/175 at both depths. At depth 3 their mean residuals are
`0.937` and `0.875`; peak intensity is `0.435`, and both width outputs are
about `0.105`. Predicted and actual counterfactual terminal improvement have
Pearson correlation `0.989`; only 5/175 (2.9%) predict terminal improvement
but counterfactually worsen. Mean terminal optimism gap is `0.253`.

### Candidate ranking, optimism, and action distribution

Every learned decision audits ten predicted-best and ten deterministic
reference candidates against the corrected simulator.

| Controller | Mean candidate Spearman | Mean top optimism gap | Mean top regret | Mean predicted-improving / actual-worsening candidate fraction |
|---|---:|---:|---:|---:|
| Learned H1 | 0.897 | 0.252 | 0.847 | 7.7% |
| Learned H3 | 0.912 | 0.164 | 0.743 | 13.9% |

These are not signatures of systematic rare-error exploitation. There are
individual outliers, but rank calibration is high and selected decisions are
substantially safer than the full audited candidate set.

Selected actions outside the empirical 5th–95th percentile central range are
10.9% for learned H1 and 8.9% for learned H3; **none** is outside the
1st–99th percentile tail. OOD candidate actions therefore do not explain the
H3 result.

## Oracle planner diagnosis

Oracle H1 is reliable on a meaningful and varied set, though not perfect:
85.4% overall, 100% one-step, 87.5% multi-step, and 68.8% boundary/clipping.
Oracle H3 is unreliable despite exact dynamics.

The primary Oracle-H3 CEM objective improves with search iteration—mean best
score `8.251 -> 5.987 -> 4.307` and mean best terminal distance
`6.811 -> 4.794 -> 3.334`—but its mean best first-step distance remains
`12.480 -> 11.270 -> 9.901`. Its selected terminal distance is far lower than
its executed next distance (mean `3.068` versus `10.243`), and 14.3% of its
first steps worsen even with exact predictions.

The one-step-reachable stratum isolates the temporal mismatch. Oracle H1’s
first executed action has median distance `0.722`; Oracle H3’s is `1.916`,
even though Oracle H3’s median planned terminal distance is `0.943`. The H3
objective is terminal-dominated (terminal max error plus only `0.10` times
the horizon-average intermediate distance), so a plan can defer most progress
to later actions. Receding-horizon execution discards those later actions and
re-optimizes, enabling repeated procrastination until the four-step cap.

Action saturation and physical infeasibility are not general explanations:
targets replay exactly, no learned selected action is in the training tail,
and the same H3 deficit is large on interior targets. Boundary/clipping and
high distance amplify failure, but do not create it.

### One-time high-budget oracle retry

The retry is deliberately conditional and biased toward failures; it is not a
replacement success score. Six primary Oracle-H3 failures were selected
deterministically—two per stratum, all cases where primary Oracle-H1
succeeded—and both oracle horizons were rerun with population 48, 12 elites,
and five CEM iterations.

| Selected-case result | Primary | High-budget retry |
|---|---:|---:|
| Oracle H1 strict success | 6/6 | 6/6 |
| Oracle H3 strict success | 0/6 | 1/6 |
| Oracle H1 mean final distance | 0.415 | 0.080 |
| Oracle H3 mean final distance | 8.516 | 2.038 |
| Oracle H3 mean simulator calls | 868.0 | 2,763.8 |
| Oracle H3 mean planner runtime | 833.5 s | 2,432.9 s |

The larger budget lowers H3 final distance on all six cases by a mean `6.478`,
but rescues only one multi-step interior case. Retry success by stratum is
0/2 one-step, 1/2 multi-step, and 0/2 boundary/clipping. The two verified
one-step targets remain failures at distances `1.095` and `1.241`; the two
boundary targets remain failures at `1.112` and `1.897`.

More search materially improves terminal planning: across 23 retry H3
decisions, mean best terminal distance falls from `3.889` at CEM iteration 1
to `0.812` at iteration 5. But mean best first-step distance only falls from
`9.689` to `7.958`. This is the decisive result: H3 is search-budget
sensitive, but extra search mostly finds better later endpoints and does not
reliably force the first receding-horizon action to finish near the target.
Budget increase alone is therefore not the repair.

## Answers to the required questions

1. **Is Oracle reliable?** Oracle H1 is meaningfully reliable (41/48, 85.4%);
   Oracle H3 is not (15/48, 31.3%). Reliability is therefore
   horizon/objective-dependent, not a general simulator-control failure.
2. **Does learned H1 usually improve the real simulator?** Yes: all 48 cases
   improve, 37/48 reach strict success, and its paired success gap from Oracle
   H1 is −8.3 points with a confidence interval spanning zero.
3. **Does learned H3 outperform H1?** It decisively underperforms: −47.9
   success points and +2.924 mean final distance, both paired intervals
   excluding zero.
4. **How often is predicted improvement actually worsening?** 1/121 (0.8%)
   for learned H1 and 9/148 (6.1%) for predicted-improving learned-H3
   decisions.
5. **Ordinary accumulation or superlinear divergence?** Mostly smooth
   accumulation, not general divergence: 45.7% approximately linear, 22.3%
   sublinear, 32.0% superlinear; mean MAE is 0.287/0.367/0.492. Model error is
   secondary because Oracle H3 fails similarly.
6. **Is CEM exploiting rare optimistic errors?** Not as the primary failure:
   candidate rank correlation is about 0.9, optimism/regret are below one
   normalized tolerance on average, terminal improvement correlation is
   0.989, and learned H3 matches Oracle H3.
7. **Are selected actions OOD?** No selected learned action is outside the
   training 1st–99th percentile tail. Central-range exceedance is 10.9% H1 and
   8.9% H3.
8. **Which output/regime dominates?** Centroid X/Y dominate. Centroid Y is the
   largest recursive residual most often and the H3 final failure rate is
   62.5% for learned H3, but learned H1 final failures are more often centroid
   X (16.7%) than Y (8.3%). Boundary/clipping and high initial distance are the
   hardest regimes. Peak and beam widths are secondary.
9. **Was the previous rollout gate justified?** It was conservative as an H3
   safety screen, but not justified as a blanket blocker on learned MPC. The
   previous `H3/H1=2.84` ratio did not imply catastrophic divergence, and the
   current learned H1 controller is effective. H3 should remain blocked until
   the planner objective is repaired.
10. **What next?** **Planner/cost repair.** Run one oracle-only,
    receding-consistent H3 objective test on the same fixed suite before any
    256/512 scaling, rollout-aware training, targeted data, trust region, or
    learned-H3 rerun. The budget retry shows search quality is secondary, but
    the one-step failures prove that budget alone is insufficient.

## Next experiment (specified, not executed)

The exact proposed configuration is
`configs/recommended_next_planner_cost_experiment.json`. It replaces the
terminal-dominated objective with target-cost weights `[1.0, 0.5, 0.25]` at
depths 1/2/3, reuses the original population 24 / elites 6 / three iterations
to isolate cost rather than budget, uses Oracle H3 only, and predeclares the
promotion gate.

The next task should first implement and test the config’s
`discounted_stage_max_plus_mean_v1` objective in a dedicated driver, then use
this exact command:

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python \
  -m continuous_control_v12.run_oracle_h3_cost_repair \
  --config /home/jiamo/VLM/runs/v12_mpc_h1_h3_diagnosis_20260731_111829/configs/recommended_next_planner_cost_experiment.json \
  --suite /home/jiamo/VLM/runs/v12_mpc_h1_h3_diagnosis_20260731_111829/suite/evaluation_suite_manifest.json \
  --primary-baseline /home/jiamo/VLM/runs/v12_mpc_h1_h3_diagnosis_20260731_111829/episodes/primary \
  --output-dir /home/jiamo/VLM/runs/v12_oracle_h3_cost_repair_receding_v1
```

The named driver and objective were intentionally **not implemented or
executed here**; doing so would begin the next experiment and violate this
task’s boundary.

No next-stage experiment, new dataset generation, retraining, rollout-aware
loss, model-architecture change, v11 parity run, final-test tuning,
publication, push, or pull request was performed.

## Artifact map

- Fixed suite: `suite/evaluation_suite_manifest.json`
- Locked config: `configs/locked_primary_config.json`
- Primary episodes: `episodes/primary/`
- Non-primary oracle retry: `episodes/oracle_retry/`
- Full primary aggregate: `aggregates/primary/summary.json`
- Episodes and steps: `aggregates/primary/episodes.csv`,
  `aggregates/primary/steps.csv`
- H3 counterfactuals:
  `aggregates/primary/h3_counterfactual_rollouts.csv`
- Candidate audits:
  `aggregates/primary/candidate_ranking_audits.csv`
- Compact tables: `aggregates/report_tables/`
- Required plots: `aggregates/primary/*.png`
- Commands: `commands/exact_commands.md`
- Tests and run logs: `logs/`
- Incremental status: `diagnostic_status.md`
- File/diff inventory: `files_created_changed_inventory.md`
