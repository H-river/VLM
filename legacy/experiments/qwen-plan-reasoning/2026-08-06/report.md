# Overnight VLA+Optics plan-reasoning experiment

## Main status: PLAN-BANK VALID, QWEN NEGATIVE

## Direct answers

1. The pre-fix current oracle is not a valid old-H1 comparison because of a wrapper regression. After the minimal equivalence fix, the candidate pilot oracle gains **5.64 pp** over old-H1/direct.
2. On independent confirmation, the frozen plan oracle is 68.33%, old-H1/direct is 59.74%, and best global single is 61.15%; oracle gains are 8.59 pp vs old H1 and 7.18 pp vs best fixed.
3. Family results and unique wins are in the accompanying summaries. Complementarity comes from staged centroid/shape correction, primary-spot reflection handling, and legal-boundary-safe control; each is budget matched to direct control.
4. Qwen full is 53.59%; image/diagnosis ablation 53.59%; goal ablation 53.59%.
5. Frozen rule is 65.26%, majority 56.03%, random 57.56%, oracle 68.33%. See paired_tests.json for group-level differences.
6. Captured oracle gap for Qwen full is -0.716417910447761; counterfactual oracle-flip results are in counterfactual_results.csv. The reasoning conclusion follows closed-loop benefit and ablations, not label/JSON accuracy alone.

## Legacy and metric audit

- Learned-H1 reproduced EXACT: 37/48 = 77.08%. Oracle-H1 is hash-verified reuse: 41/48 = 85.42%.
- 47.69% means 93/195 successful executions after selecting one plan per each of 15 independent groups by its expected utility over 13 matched seeds. All-plan episode success is 422/975 = 43.28%; episode hindsight is 94/195 = 48.21%.
- No aggregation/evaluator discrepancy explains the decline. The legacy 48-suite and old 15-group diagnostic suite are setup-disjoint; matched interventions show reflection and saturation dominate failures, while their effects are nonadditive and the remaining distribution shift is unresolved rather than force-attributed.

## Plan programs

- `direct_all_five`: joint five-metric H1 replanning.
- `centroid_then_full`: centroid-focused first stage, then all-five refinement.
- `shape_then_full`: width/intensity-focused first stage, then all-five refinement.
- `primary_spot_then_full`: primary-spot measurement recovery for reflection, then all-five refinement.
- `boundary_safe_then_full`: boundary-safe first stage, then all-five refinement.

All use population 24, 3 CEM iterations, horizon 1, max 4 total steps, identical action bounds and strict tolerances. The wrapper fix changed no budget or task definition.

## Matched candidate comparison

| Candidate method | Strict success | Terminal normalized error |
|---|---:|---:|
| boundary_safe_then_full | 56.32% | 6.737 |
| centroid_then_full | 56.67% | 6.672 |
| direct_all_five | 55.98% | 6.795 |
| primary_spot_then_full | 42.14% | 8.544 |
| shape_then_full | 53.59% | 7.069 |
| expected-utility oracle | 61.62% | 2.766 |

Unique wins, dominance, seed stability, and first-action diversity are stored in `redesigned_pilot/`.

## Independent confirmation selectors

| Confirmation selector | Strict success | Group bootstrap 95% CI | Gain vs direct | Error |
|---|---:|---:|---:|---:|
| direct_all_five | 59.74% | [48.59%, 70.26%] | 0.00 pp | 5.618 |
| best_global_single | 61.15% | [50.00%, 71.54%] | 1.41 pp | 5.598 |
| majority | 56.03% | [44.74%, 67.18%] | -3.72 pp | 5.926 |
| random | 57.56% | [46.92%, 68.33%] | -2.18 pp | 5.935 |
| rule_based_frozen | 65.26% | [54.87%, 75.13%] | 5.51 pp | 5.028 |
| qwen_full | 53.59% | [42.69%, 64.87%] | -6.15 pp | 8.646 |
| qwen_without_image_diagnosis | 53.59% | [42.69%, 64.87%] | -6.15 pp | 8.646 |
| qwen_without_goal | 53.59% | [42.69%, 64.87%] | -6.15 pp | 8.646 |
| oracle | 68.33% | [58.08%, 77.82%] | 8.59 pp | 3.596 |

## Intervention strata

| Intervention | Direct | Oracle | Qwen full | Frozen rule |
|---|---:|---:|---:|---:|
| boundary_swap | 80.77% | 83.97% | 80.13% | 81.41% |
| nominal | 76.92% | 81.41% | 76.92% | 76.92% |
| reflection | 0.00% | 26.28% | 25.64% | 25.64% |
| saturation | 57.69% | 64.10% | 1.28% | 56.41% |
| target_swap | 83.33% | 85.90% | 83.97% | 85.90% |

## Reproducibility and compute

- Git commit: `e6f18d4d5f513f912e73317aaaa66fdc3c479edf`; the pre-existing dirty worktree was preserved; no commit or push occurred.
- Confirmation groups hash: `b8e1240f91253bdb017047913df710ab025782905329a8994e519f4b23b589f9`.
- Learned-H1 checkpoint hash: `d9b30627c80817f6ecade1959d8cc9e91e7a9de9cc51485153fdbfaa173aca2e`.
- Plan-controller source hash after the minimal repair: `6e5448ae1d8806c2b706de3fc140c3e00c2c197be3bc74583c306907bb98e24b`; preserved pre-fix hash: `86004ab366bc291a18f7710f0e276d1657e0a9e50ebcaa4a696157682f73de13`.
- Unit/integration tests after the repair: 39/39 passed; plan smoke: 5/5 episodes executed with no infrastructure failure.
- Wall time: 10.47 h. Conservative effective real experiment/inference time: 10.17 h.
- New/reused/cached/failed complete episodes: 15743/1071/0/0. Separately, 450 first-action-only wrapper equivalence comparisons were executed and are not counted as episodes.
- Formal frozen/protected evaluation remained disabled.

## Limitations and recommendation

The independent confirmation contains 12 independent base states with five paired interventions each, so family estimates remain preliminary despite 25 matched controller seeds. Existing Qwen checkpoints were trained on candidate data under the earlier wrapper behavior; they were reused because the discrete schema is unchanged, but this mismatch is a limitation. Continue Qwen training only if the frozen plan oracle has a meaningful group-level gain; otherwise redesign or suspend the selector direction rather than optimize label accuracy.
