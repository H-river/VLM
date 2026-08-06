# Qwen-VL supervisor v1.1 candidate pre-meeting report

Status: **NOT SEALED — FROZEN EVALUATION DISABLED**

Overall: **YELLOW**. Maximum supported conclusion: **READY FOR HUMAN REVIEW BEFORE FINAL FREEZE**.
阿
This report is development/engineering evidence only. No frozen IID/OOD/protected inference, formal frozen evaluation, final checkpoint-bound freeze, or real end-to-end closed-loop evaluation was run.

## Executive summary

- GO: candidate data/schema/identity audit passed; exact train-dev/frozen registry overlaps are zero; old fixed-pixel reflection count is 0.
- GO: Full Qwen completed all three 200-step seeds and all 60 dev records per seed with 96.91% +/- 1.67 pp diagnosis balanced accuracy, 96.59% +/- 1.80 pp macro-F1, and 100.00% +/- 0.00 pp strict valid JSON.
- GO: state-machine dry-run passed all 14 safety assertions, including strict fail-safe stop, injection rejection, H1-only routing, reversible enums, and finite budget termination.
- YELLOW: 4 dev records meet the preregistered legal train-dev perceptual-neighbor rule despite zero exact/setup/pair/episode/augmentation overlap.
- YELLOW: dev is small, Full checkpoints show seed-dependent post-best loss increase, frozen evaluation is disabled, and real sequential recovery evidence is absent.
- RED findings: none in the authorized candidate scope.

## Candidate data and leakage audit

New train adds 96 records / 48 pairs / 24 setups; new dev adds 24 / 12 / 6.
Combined train is 192 records / 96 pairs / 72 setups; combined dev is 60 / 30 / 24.

Exact sample, image, setup, pair, episode, and augmentation-base overlap is zero for train-dev and for candidate-versus-frozen identity registries. Pair completeness, serialized metric match, schema, prompt-visible fields, and image hashes passed. Frozen image content was not opened during the perceptual audit. All reflection records are width-relative.

Width-quartile coverage: train `{'Q1': 28, 'Q2': 26, 'Q3': 28, 'Q4': 26}`; dev `{'Q1': 10, 'Q2': 8, 'Q3': 10, 'Q4': 8}`. Class counts: train `{'nominal': 96, 'secondary_reflection': 54, 'sensor_saturation': 42}`; dev `{'nominal': 30, 'secondary_reflection': 18, 'sensor_saturation': 12}`.

The 4 legal near-neighbors are a YELLOW morphology-similarity risk, not a demonstrated provenance leak: no same setup or exact image was found. With this dev size, effects involving only a few records are inconclusive.

Key hashes:

- train manifest `237356417d1262d262f6007b5f12cd5195a0f90c04b1ffdf0b9e762e2b171528`; dev manifest `cccb6181235f450e617c30b9bb29683d9949a46c46b1a20154df57d298c273d2`
- train SFT `fc8d3f70af1f49d9273750bd616f470a280162c7340ea1f825ddb4b82d25299e`; dev SFT `94994e9b74d7255fb6d32cd56edc89848983e02ee6941a3884c5f69e41d59c10`
- experiment protocol `2de3a9ec905c028bf015e14e6479b2a32370ffa982e00faac0723c6dcf901eb5`; baseline protocol `9957fe4eb03a873acabe548f108fa6093cc2661eb1d2528f68905035925ee703`

## Full Qwen three-seed results

| seed | best step | train loss (whole run) | best dev loss | final dev loss | BA | macro-F1 | policy acc | action macro-F1 | joint exact | wall min | peak alloc GiB | adapter SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026080101 | 125 | 0.027251 | 0.002589 | 0.004881 | 98.89% | 98.53% | 98.33% | 98.53% | 98.33% | 31.62 | 4.88 | `acf6ba79d75d933586e8b23ee3884450fd59d6ed629619953499eae5487e8e3d` |
| 2026080102 | 175 | 0.035195 | 0.007056 | 0.007085 | 97.04% | 97.04% | 96.67% | 97.04% | 96.67% | 32.01 | 4.88 | `5f247b7c127aede29c3e1ae0fbca0f2dd74dc6a771bc471ee1291d0cf47cf6b3` |
| 2026080103 | 125 | 0.029278 | 0.009989 | 0.013021 | 94.81% | 94.19% | 93.33% | 94.19% | 93.33% | 32.41 | 4.88 | `42b7f67e4ef8e3bcb75b8eb28690c7401ff611e5a76bd64f6a0e621aa4334a7d` |

Seed mean/std: coverage 100.00% +/- 0.00 pp; strict valid JSON 100.00% +/- 0.00 pp; diagnosis BA 96.91% +/- 1.67 pp; macro-F1 96.59% +/- 1.80 pp; policy accuracy 96.11% +/- 2.08 pp; action macro-F1 96.59% +/- 1.80 pp; joint exact 96.11% +/- 2.08 pp.

Setup-cluster/pair-preserving bootstrap 95% CI: diagnosis BA [93.68, 99.31] pp, macro-F1 [93.21, 99.07] pp, joint exact [92.26, 98.96] pp.

Three-seed diagnosis agreement is 55/60 records; 2 records are wrong in at least two seeds. Exact rows are recorded in `candidate_results.json`.

Summed Full diagnosis confusion matrix across seeds:

| true \ predicted | nominal | sensor_saturation | secondary_reflection | <invalid> |
|---|---:|---:|---:|---:|
| nominal | 85 | 0 | 5 | 0 |
| sensor_saturation | 0 | 36 | 0 | 0 |
| secondary_reflection | 2 | 0 | 52 | 0 |
| <invalid> | 0 | 0 | 0 | 0 |

Full per-class mean precision / recall / F1:

| class | precision | recall | F1 |
|---|---:|---:|---:|
| nominal | 97.70% | 94.44% | 96.03% |
| secondary_reflection | 91.39% | 96.30% | 93.74% |
| sensor_saturation | 100.00% | 100.00% | 100.00% |

Overfitting signs: teacher-forced training losses continue toward zero while final dev loss is above the selected minimum by 88.5%, 0.4%, 30.4% for seeds 01/02/03. This is seed-dependent dev overfit evidence; selected checkpoints follow the frozen lowest-full-dev-loss rule. Full loss curves are in `candidate_results.json`.

## Full-model inference interventions

Positive drop means Full is better. CIs are setup-cluster paired bootstrap intervals.

| condition | BA mean +/- std | Full-minus-condition BA | BA drop 95% CI | macro-F1 | joint exact | note |
|---|---:|---:|---:|---:|---:|---|
| blank_image | 33.33% +/- 0.00 pp | 63.53 pp | [59.41, 67.09] pp | 21.86% +/- 5.15 pp | 42.22% +/- 8.75 pp | actual Full-adapter intervention generation |
| budget_empty | 96.54% +/- 2.15 pp | 0.38 pp | [0.00, 1.23] pp | 96.12% +/- 2.43 pp | 95.56% +/- 2.83 pp | actual Full-adapter intervention generation |
| budget_shuffle | 96.91% +/- 1.67 pp | 0.00 pp | [0.00, 0.00] pp | 96.59% +/- 1.80 pp | 96.11% +/- 2.08 pp | byte-identical structural no-op; Full predictions reused |
| cross_class_image_shuffle | 2.59% +/- 1.98 pp | 94.30 pp | [90.04, 97.84] pp | 2.54% +/- 1.93 pp | 2.78% +/- 2.08 pp | actual Full-adapter intervention generation |
| goal_empty | 96.30% +/- 2.48 pp | 0.60 pp | [0.00, 1.96] pp | 96.08% +/- 2.49 pp | 95.56% +/- 2.83 pp | actual Full-adapter intervention generation |
| goal_history_budget_empty | 96.91% +/- 1.67 pp | 0.00 pp | [0.00, 0.00] pp | 96.59% +/- 1.80 pp | 96.11% +/- 2.08 pp | actual Full-adapter intervention generation |
| goal_history_budget_shuffle | 96.91% +/- 1.67 pp | 0.00 pp | [0.00, 0.00] pp | 96.59% +/- 1.80 pp | 96.11% +/- 2.08 pp | actual Full-adapter intervention generation |
| goal_shuffle | 96.91% +/- 1.67 pp | 0.00 pp | [0.00, 0.00] pp | 96.59% +/- 1.80 pp | 96.11% +/- 2.08 pp | actual Full-adapter intervention generation |
| history_empty | 96.91% +/- 1.67 pp | 0.00 pp | [0.00, 0.00] pp | 96.59% +/- 1.80 pp | 96.11% +/- 2.08 pp | byte-identical structural no-op; Full predictions reused |
| history_shuffle | 96.91% +/- 1.67 pp | 0.00 pp | [0.00, 0.00] pp | 96.59% +/- 1.80 pp | 96.11% +/- 2.08 pp | byte-identical structural no-op; Full predictions reused |
| metrics_blank | 96.91% +/- 1.67 pp | 0.00 pp | [0.00, 0.00] pp | 96.59% +/- 1.80 pp | 96.11% +/- 2.08 pp | actual Full-adapter intervention generation |
| metrics_shuffle | 96.91% +/- 1.67 pp | 0.00 pp | [0.00, 0.00] pp | 96.59% +/- 1.80 pp | 96.11% +/- 2.08 pp | actual Full-adapter intervention generation |
| random_image_shuffle_2026084101 | 30.68% +/- 1.72 pp | 66.07 pp | [54.82, 77.35] pp | 30.61% +/- 1.73 pp | 36.11% +/- 2.08 pp | actual Full-adapter intervention generation |
| random_image_shuffle_2026084102 | 25.62% +/- 1.67 pp | 71.21 pp | [59.77, 80.92] pp | 25.65% +/- 1.64 pp | 31.11% +/- 2.08 pp | actual Full-adapter intervention generation |
| random_image_shuffle_2026084103 | 31.48% +/- 0.91 pp | 65.43 pp | [53.24, 76.89] pp | 31.51% +/- 0.89 pp | 36.67% +/- 1.36 pp | actual Full-adapter intervention generation |

Both blank and cross-class shuffled image conditions cross the preregistered 5 pp engineering threshold, supporting image dependence on this dev set. This still does not establish general visual understanding.
Metrics neutralization does not cross the 5 pp engineering threshold; strong structured-metric dependence is not established.
History empty/shuffle and budget shuffle are not evidence: those source fields are constant and the generated intervention files are byte-identical to Full dev. Goal and budget neutralization use schema-valid zero values, not missing-token semantics.

## Retrained Qwen single-modality ablations

| model | BA mean +/- std | macro-F1 | joint exact | Full-minus-model BA | BA drop 95% CI |
|---|---:|---:|---:|---:|---:|
| image_only_qwen | 90.37% +/- 1.29 pp | 90.44% +/- 0.86 pp | 90.56% +/- 0.79 pp | 6.52 pp | [1.98, 12.46] pp |
| metrics_only_qwen | 33.33% +/- 0.00 pp | 22.22% +/- 0.00 pp | 50.00% +/- 0.00 pp | 63.57 pp | [60.35, 65.97] pp |

Inference interventions answer what the trained Full model uses; retrained ablations answer what one modality can learn after retraining. They are not interchangeable.

## Unified three-class baselines

Diagnosis is the primary capability. Policy/action/joint fields below are deterministic routing consequences and are not three independent reasoning tasks.

| baseline | seeds | params | input | train wall s mean | BA mean +/- std | macro-F1 | diagnosis accuracy | Full-minus BA 95% CI |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| logistic_metrics | 1 | 18 | metrics | 0.006 | 38.89% +/- 0.00 pp | 32.33% +/- 0.00 pp | 31.73% | [48.15, 67.95] pp |
| metrics_mlp | 3 | 291 | metrics | 1.264 | 55.74% +/- 0.52 pp | 41.25% +/- 1.14 pp | 42.81% | [33.55, 49.10] pp |
| tiny_cnn_image | 3 | 38915 | image | 8.869 | 74.81% +/- 6.35 pp | 67.58% +/- 10.29 pp | 63.40% | [17.46, 26.14] pp |
| tiny_cnn_metrics_fusion | 3 | 39235 | image+metrics | 8.298 | 81.98% +/- 6.39 pp | 78.97% +/- 9.30 pp | 75.58% | [9.36, 20.17] pp |

All baselines use the same 192-record candidate train and 60-record candidate dev, the same three diagnosis labels, fixed configurations, and no dev hyperparameter sweep. Full confusion matrices and per-class scores are in `reports/evaluations/` and `candidate_results.json`.

## State-machine and safety dry-run

- nominal -> `standard` -> frozen H1 one-step CEM route: passed.
- saturation -> `lower_exposure_reacquire` -> reacquire -> external re-diagnosis state: passed as a contract trace; no recovery outcome was observed.
- reflection -> `primary_spot` -> switch measurement -> frozen H1 route: passed.
- invalid JSON and numeric/continuous-action injection: strict rejection to non-dispatched conservative stop passed.
- repeated anomaly with remaining budget: terminates; continuation gate and horizon remain authoritative.
- enum mapping is reversible; H3 is not registered/called; supervisor has no actuator fields or direct actuator authority.
- all 252 candidate train/dev records passed image 128px, current diagnostic 128px frame/range, goal labelled 1024px lab/sensor frame/range, positive width, mm unit, per-step-bound, repository-limit-source, budget, and image-hash checks.

Known semantics retained in the report:

- legacy v12.0 camera extraction had clipped left-searchsorted discontinuity; corrected v12.1 uses continuous finite-pixel irradiance sampling. This dry-run is not a hardware validation.
- current supervisor metrics are diagnostic-image 128px while goals are labelled lab_sensor 1024px/raw-peak; v12.1 documents the lab-to-sensor correction, but calibration accuracy was not tested here.
- legacy `power_w` was a dead input under peak-normalized source behavior; corrected v12.1 makes it causal. The static supervisor prompt does not expose setup `power_w` directly.
- lens +/-0.05 mm and camera +/-0.02 mm are visible per-step bounds. +/-3 mm is a repository sampling-domain limit, not a hardware limit.

There are no real paired sequential anomaly observations in this candidate. **Temporal recovery performance has not been verified.**

## Claim-evidence table

| claim | evidence | allowed wording |
|---|---|---|
| Candidate split identity isolation | zero exact/sample/image/setup/pair/episode/augmentation overlap; protected registry identity only | engineering GO |
| Full candidate-dev classification | three seeds, complete 60-record dev, strict reducer and pair-aware CI | candidate dev engineering result only |
| Visual anomaly understanding | blank/shuffle interventions plus image-only retraining | only the conditional wording above; never general visual understanding |
| Structured-state dependence | metrics blank/shuffle plus metrics-only retraining | dependency on this candidate dev only |
| Closed-loop safety | synthetic contract dry-run | state-machine contract passed; not recovery success |
| Reflection robustness | width-relative train/dev only | provisional engineering GO; no severity-OOD evidence |
| Frozen performance | not run | no claim |
| Temporal recovery | no real paired sequential anomaly observations | not verified |

## Remaining blockers before final freeze

1. Human review of the 4 legal perceptual near-neighbors, intervention interpretation, seed variability, and post-best dev-loss increases.
2. Decide whether the static reacquire orchestration must be productionized as an explicit reacquire-then-re-diagnose state before any closed-loop evaluation.
3. Preserve reflection as provisional until a separately frozen severity-OOD protocol is authorized.
4. Only after human approval: bind final checkpoint/config and separately authorize frozen IID/OOD and formal end-to-end closed-loop evaluation.

## Meeting: can say / cannot say

Can say:

- The candidate completed a preregistered three-seed, 200-step development pilot with complete-dev strict generation, modality checks, unified baselines, and safety-contract dry-runs.
- v1 is a static anomaly diagnosis and routing supervisor.
- Learned-H1 is the deployed controller; supervisor outputs remain high-level enums and H1 owns continuous actions.

Cannot say:

- This is a general temporal reasoning agent, a validated closed-loop recovery system, or frozen scientific performance.
- Reflection severity-OOD robustness is established.
- Qwen dev accuracy and Learned-H1 77.1% form an overall accuracy. They are different experiments and must never be multiplied.
- H3 is deployed. H3 failed mainly at planner/objective behavior and is not deployed.
- Frozen IID/OOD or formal end-to-end results exist; neither was run.

## Runtime and reproducibility

The observed candidate artifact-to-report window was approximately 7.02 hours (2026-08-02 00:30:39 to 07:32:01 SGT). Nine Qwen training runs sum to 4.79 single-GPU hours of measured wall time. Actual generated-condition model latency sums to 1.47 hours across 45 generation reports. Maximum CUDA allocation was 4.88 GiB and maximum reservation was 7.51 GiB on an RTX 4080 Laptop GPU (12,282 MiB). Per-run times and hashes are in `machine_summary.json` and `candidate_results.json`.

Actual command patterns and exceptions are in `reports/commands.md`; the artifact registry contains hashes for protocols, configs, manifests, reports, final adapters, generation reports, and combined predictions.

Final disposition: **YELLOW — READY FOR HUMAN REVIEW BEFORE FINAL FREEZE**.
