# Corrective v2 larger-data round — final analysis

## Decision

**No adapter is promoted.** All three corrective-v2 seeds make a large and statistically clear improvement in equal-task macro score on the independent development set, but none passes the frozen feasible-control, simulator-success, control status-F1, or information-sufficiency status-F1 gates. The original step-40 adapter remains the retained reference, and the sealed pilot test was not evaluated.

No paid API calls were used.

## Data completed

| Artifact | Scenarios | Records | Visual records | Seed |
|---|---:|---:|---:|---:|
| Corrective training set | 500 | 2,000 | 100 (5%) | 2718 |
| Independent `dev_v2` | 150 | 600 | 30 (5%) | 1618 |
| Final training curriculum | — | 2,462 | 165 | 202 |

The curriculum contains all 2,000 corrective records and 462 clean pilot anchors. It has 2,462 unique source examples, no repetition, 1,798 sampling units, and keeps all 532 matched corrective units adjacent.

The corrective data exactly balances 200 feasible / 200 infeasible control records and 200 answerable / 200 insufficient sufficiency records. Diagnosis has 132 unique, 132 ambiguous, and 132 unsupported records. `dev_v2` independently balances 60/60 control, 60/60 sufficiency, and 40/40/40 diagnosis statuses.

## Data audit

- Strict structural and checksum audits passed for both datasets.
- Simulator replay passed 10,585/10,585 corrective states and 3,180/3,180 dev states. Maximum absolute replay errors were below 0.00005.
- Corrective matched-control baseline-error range: mean 1.020 px, p95 2.046 px.
- Dev matched-control baseline-error range: mean 1.112 px, p95 2.187 px.
- Fixed-gain control success: 32.0% on corrective data and 36.7% on dev, both below the frozen 60% shortcut ceiling.
- There is zero example-ID, scenario-group, scenario-seed, image, or matched-group overlap among pilot, targeted-v1.1, corrective-v2, and dev-v2.

## Training

All runs continued from the original step-40 adapter for 200 optimizer steps with learning rate `3e-5`, gradient accumulation 4, and checkpoints at steps 100 and 200.

| Seed | Final teacher-forced loss | Mean token accuracy | Adapter |
|---:|---:|---:|---|
| 42 | 0.13297 | 0.95145 | `/home/jiamo/VLM_runs/qwen25vl_3b_qlora_corrective_v2_seed42` |
| 123 | 0.13290 | 0.95173 | `/home/jiamo/VLM_runs/qwen25vl_3b_qlora_corrective_v2_seed123` |
| 314 | 0.13294 | 0.95150 | `/home/jiamo/VLM_runs/qwen25vl_3b_qlora_corrective_v2_seed314` |

The near-identical teacher-forced metrics make clear that checkpoint loss alone cannot select a useful control model; free generation and simulator replay remain necessary.

## Independent-dev results

Strict rubric v2 scores on all 600 records:

| Run | Setup | Sufficiency | Causal | Forward | Diagnosis | Control | Counterfactual | Macro | JSON | Schema |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Step-40 reference | 0.350 | 0.410 | 0.780 | 0.383 | 0.356 | 0.455 | 0.543 | 0.468 | 0.988 | 0.938 |
| Seed 42 | 0.894 | 0.505 | 0.897 | 0.378 | 0.896 | 0.503 | 0.750 | 0.689 | 1.000 | 0.983 |
| Seed 123 | 0.900 | 0.492 | 0.887 | 0.383 | 0.920 | 0.503 | 0.750 | 0.691 | 1.000 | 0.983 |
| Seed 314 | 0.900 | 0.522 | 0.890 | 0.383 | 0.903 | 0.512 | 0.750 | 0.694 | 0.998 | 0.982 |

Candidate macro mean is **0.6913** with population standard deviation **0.0023**. The paired 10,000-replicate scenario-group bootstrap gives:

| Seed | Macro improvement vs reference | 95% CI | P(candidate better) |
|---:|---:|---:|---:|
| 42 | +0.2207 | [0.1899, 0.2514] | 1.000 |
| 123 | +0.2226 | [0.1930, 0.2526] | 1.000 |
| 314 | +0.2262 | [0.1962, 0.2565] | 1.000 |

The overall gain is therefore real and seed-stable. It is driven mainly by setup interpretation, diagnosis, causal effects, counterfactual reasoning, and output structure. Forward prediction is unchanged.

## Frozen promotion gates

| Gate | Required | Seed 42 | Seed 123 | Seed 314 |
|---|---:|---:|---:|---:|
| Control feasible recall | >=0.60 | 0.117 | 0.083 | 0.100 |
| Control infeasible recall | >=0.60 | 0.917 | 0.967 | 0.967 |
| Feasible simulator success | >=0.40 | 0.117 | 0.067 | 0.083 |
| Sufficiency status macro-F1 | >=0.60 | 0.519 | 0.501 | 0.532 |
| Diagnosis status macro-F1 | >=0.60 | 0.856 | 0.889 | 0.865 |
| Control status macro-F1 | >=0.60 | 0.425 | 0.410 | 0.425 |
| Schema validity | >=0.93 | 0.983 | 0.983 | 0.982 |
| Anchor regression | no task below -0.05 | pass | pass | pass |

Passing seeds: **0 of 3**; required: **2 of 3**. Promotion status: **no promotion**.

## Failure diagnosis and next round

The larger balanced dataset fixed the earlier setup and diagnosis collapse, improved schema compliance, and produced seed-stable macro gains. It did not teach reliable feasible control. All seeds strongly favor `infeasible_within_limits`; even when they declare feasibility, only 7–12% of feasible cases succeed under simulator replay. Sufficiency also remains below the status gate, despite exact class balance.

The next corrective round should therefore change supervision quality rather than merely add more records:

1. Represent control answers action-first: actuator, signed movement, predicted residual, and executable validity before the status label.
2. Add matched feasible pairs sharing the same baseline error and actuator but requiring different signed actions, including near-boundary and minimum-motion alternatives.
3. Oversample action diversity within the feasible branch without changing the independent dev distribution; keep infeasible examples matched but do not increase their frequency.
4. Add sufficiency pairs with the same masked field but different completion sensitivity, and require an explicit set of answer-changing completions.
5. Run a small curriculum ablation first. Do not reopen the sealed test until at least two seeds pass every existing gate.

## Artifacts

- Frozen protocol: `results/corrective_v2_protocol.md`
- Promotion decision: `results/corrective_v2_selection.json`
- Paired bootstrap: `results/corrective_v2_bootstrap.json`
- Error taxonomy: `results/corrective_v2_error_analysis/error_analysis.md`
- Shortcut audits: `results/corrective_v2_shortcut_audit.json`, `results/dev_v2_shortcut_audit.json`
- Cross-dataset overlap audit: `results/corrective_v2_overlap_audit.json`

