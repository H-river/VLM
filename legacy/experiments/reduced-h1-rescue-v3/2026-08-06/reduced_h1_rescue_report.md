# Reduced Learned-H1 Controller Audit v3

Candidate-dev-only, H1-only evidence-reuse audit over 24 records. No Qwen training or scoring was run.

## Frozen equal-compute strategies

| Strategy | Original configuration | Compiled bound scale | CEM evaluations |
| --- | --- | --- | --- |
| default | default | 1.0 | 72 |
| strategy_b | guided__unbiased_unknown__objective-balanced__mask-default__scale-fine__risk-standard | 0.35 | 72 |
| strategy_c | guided__unbiased_unknown__objective-balanced__mask-default__scale-medium__risk-standard | 0.65 | 72 |

## Coverage

| Expected | Reused | New | Completed | Failed | Missing |
| --- | --- | --- | --- | --- | --- |
| 1440 | 678 | 762 | 1440 | 0 | 0 |

## Reduced acceptable-set stability

| Comparison | Median Jaccard | Fraction <0.5 | Top-1 agreement |
| --- | --- | --- | --- |
| K3_K5 | 1.000 | 0.000 | 1.000 |
| K3_K10 | 1.000 | 0.000 | 0.958 |
| K5_K10 | 1.000 | 0.000 | 0.958 |
| K10_CONFIRM_K10 | 1.000 | 0.000 | 0.958 |

## Confirmation-root baselines

| Policy | Strict success | Mean cost | P90 cost |
| --- | --- | --- | --- |
| actual_default_fixed | 0.188 | 6.8699 | 17.3973 |
| best_global_fixed | 0.292 | 11.6484 | 28.5451 |
| held_out_per_record_oracle | 0.375 | 6.5092 | 17.3973 |
| confirmation_clairvoyant_oracle_descriptive_only | 0.375 | 6.5092 | 17.3973 |

## Discovery selector frequency

| Strategy | Records selected |
| --- | --- |
| default | 16 |
| strategy_b | 8 |

## Strategy-pair crossover attribution

| Pair | Ordering reversals | Disjoint set flips | Bilateral regret |
| --- | --- | --- | --- |
| default vs strategy_b | 8 | 0 | 0.0000 |
| default vs strategy_c | 13 | 0 | 0.0246 |
| strategy_b vs strategy_c | 8 | 0 | 0.0000 |

## Unstable-record categories

| Category | Value | Records | Unstable | Overrepresentation |
| --- | --- | --- | --- | --- |
| difficulty | difficult | 8 | 0 | 0.00 |
| difficulty | easy | 8 | 0 | 0.00 |
| difficulty | medium | 8 | 1 | 3.00 |
| error_type | mixed | 24 | 1 | 1.00 |
| boundary | boundary_near | 6 | 1 | 4.00 |
| boundary | interior | 18 | 0 | 0.00 |
| uncertainty | high | 12 | 1 | 2.00 |
| uncertainty | low | 12 | 0 | 0.00 |
| realized_model_error | high | 12 | 1 | 2.00 |
| realized_model_error | low | 12 | 0 | 0.00 |
| strategy_margin | clearly_separated | 12 | 0 | 0.00 |
| strategy_margin | near_tied | 12 | 1 | 2.00 |

## Ten most unstable records

| Record | Score | Difficulty | Boundary | Margin |
| --- | --- | --- | --- | --- |
| qh1meta_dev_0004__target-00-one_step_mixed | 2.00 | medium | boundary_near | near_tied |
| qh1meta_dev_0000__target-00-one_step_mixed | 0.00 | medium | interior | clearly_separated |
| qh1meta_dev_0000__target-01-multi_step_opposite | 0.00 | difficult | interior | clearly_separated |
| qh1meta_dev_0000__target-02-near_target | 0.00 | easy | interior | near_tied |
| qh1meta_dev_0001__target-00-one_step_mixed | 0.00 | difficult | interior | clearly_separated |
| qh1meta_dev_0001__target-01-multi_step_opposite | 0.00 | difficult | interior | clearly_separated |
| qh1meta_dev_0001__target-02-near_target | 0.00 | easy | interior | near_tied |
| qh1meta_dev_0002__target-00-one_step_mixed | 0.00 | medium | interior | near_tied |
| qh1meta_dev_0002__target-01-multi_step_opposite | 0.00 | medium | interior | clearly_separated |
| qh1meta_dev_0002__target-02-near_target | 0.00 | easy | interior | near_tied |

## Most destabilizing roots

| Root | Group | LOO set changes | LOO top changes | Action dispersion |
| --- | --- | --- | --- | --- |
| 2026080802 | confirmation | 1 | 1 | 0.0110 |
| 2026080805 | confirmation | 1 | 1 | 0.0108 |
| 2026080808 | confirmation | 1 | 1 | 0.0100 |
| 2026080406 | discovery | 1 | 1 | 0.0100 |
| 2026080405 | discovery | 1 | 1 | 0.0097 |
| 2026080401 | discovery | 1 | 0 | 0.0103 |
| 2026080302 | discovery | 1 | 0 | 0.0094 |
| 2026080301 | discovery | 0 | 0 | 0.0115 |

## Gates

| Gate | Status | Reason |
| --- | --- | --- |
| LABEL_STABILITY_GATE | GREEN | reduced K5/K10 and discovery/confirmation criteria meet the frozen Jaccard and top-1 conditions |
| SELECTOR_VALUE_GATE | GREEN | held-out confirmation improvement meets the predeclared success/cost criterion with nontrivial crossover |

**Disposition:** Qwen controller-selector target scientifically plausible

Instability categories are associations, not causal claims. Cached roots without stored CEM traces are explicitly marked unresolved in the root attribution.
