# Native hybrid direction + magnitude: first round

**Promotion decision: REJECT**

This experiment gives both stages the complete visible setup, current measured beam state, and action. The optical simulator was used only to create and score labels; it was not callable at inference time.

The LLM is the best pre-tool native checkpoint (`corrective_v2_seed314`). The numerical component is a three-seed ensemble of small 128-64 MLPs. This first round is tabular/text-only; visual-only rows were excluded rather than supplying hidden numerical state.

## Results

| Split | System | Skill vs zero | Strict joint pass | Centroid-x MAE | Centroid-y MAE | Sigma-x MAE | Sigma-y MAE | Peak MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IID (54) | zero_change | 0.000 | 0.056 | 1.996 | 2.021 | 1.164 | 1.001 | 5.130 |
| IID (54) | direct_mlp | 0.072 | 0.000 | 1.805 | 1.733 | 1.205 | 0.974 | 5.254 |
| IID (54) | oracle_direction_plus_magnitude | 0.369 | 0.185 | 1.008 | 1.090 | 0.829 | 0.813 | 4.088 |
| IID (54) | llm_direction_plus_magnitude | -0.358 | 0.000 | 2.519 | 2.502 | 1.249 | 1.218 | 6.572 |
| OOD (60) | zero_change | 0.000 | 0.167 | 1.738 | 2.102 | 0.896 | 1.145 | 4.483 |
| OOD (60) | direct_mlp | 0.251 | 0.033 | 1.555 | 1.719 | 1.000 | 1.069 | 4.676 |
| OOD (60) | oracle_direction_plus_magnitude | 0.674 | 0.283 | 0.938 | 1.119 | 0.689 | 0.797 | 2.659 |
| OOD (60) | llm_direction_plus_magnitude | -0.533 | 0.067 | 2.172 | 2.584 | 0.953 | 1.413 | 6.768 |

## Direction stage

- IID direct_mlp: equal-field macro-F1 0.377, all-five direction exact 0.074.
- IID llm: equal-field macro-F1 0.255, all-five direction exact 0.019, schema-valid 1.000.
- OOD direct_mlp: equal-field macro-F1 0.422, all-five direction exact 0.033.
- OOD llm: equal-field macro-F1 0.256, all-five direction exact 0.000, schema-valid 1.000.

The LLM collapsed to strong defaults: it predicted `centroid_x=decrease` for 54/54 IID and 60/60 OOD cases; it similarly predicted decreasing peak intensity for 50/54 IID and 58/60 OOD cases.

## Interpretation rules

- `strict_joint_pass` requires centroid vector error <= 1 px, both width errors <= 2 px, and peak error <= 5% of the initial peak simultaneously.
- `skill_vs_zero` is computed from tolerance-normalized squared error; positive is better than predicting no change, zero is tied, and negative is worse.
- `oracle_direction_plus_magnitude` is an upper bound for this factorization, not a deployable result.
- A hybrid should be promoted only if the real LLM-direction system beats both zero-change and direct signed regression on fresh IID and OOD setups.

## Decision

- IID: LLM hybrid skill versus zero was -0.358, not positive.
- IID: strict joint pass 0.000 did not beat both zero-change (0.056) and direct MLP (0.000).
- OOD: LLM hybrid skill versus zero was -0.533, not positive.
- OOD: strict joint pass 0.067 did not beat both zero-change (0.167) and direct MLP (0.033).
