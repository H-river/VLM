# Optics understanding evaluation

Rubric: **v2**.

Evaluated **28 / 400** records.
Macro task score: **0.325**
Valid JSON: **100.0%**; valid schema: **50.0%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 0 | n/a | n/a | n/a |
| information_sufficiency | 14 | 0.400 | 100.0% | 100.0% |
| causal_effects | 0 | n/a | n/a | n/a |
| forward_prediction | 0 | n/a | n/a | n/a |
| diagnosis | 0 | n/a | n/a | n/a |
| constrained_intervention | 14 | 0.250 | 100.0% | 0.0% |
| counterfactual_reasoning | 0 | n/a | n/a | n/a |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 28 | 0.325 | 100.0% | 50.0% |

Total generation latency: **64.3 s**; mean: **2.30 s/record**.
Mean tokens: **1615.5 input / 78.6 output**. Peak allocated CUDA memory: **2.52 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
