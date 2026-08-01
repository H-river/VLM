# Optics understanding evaluation

Rubric: **v2**.

Evaluated **28 / 400** records.
Macro task score: **0.505**
Valid JSON: **100.0%**; valid schema: **85.7%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 0 | n/a | n/a | n/a |
| information_sufficiency | 14 | 0.457 | 100.0% | 100.0% |
| causal_effects | 0 | n/a | n/a | n/a |
| forward_prediction | 0 | n/a | n/a | n/a |
| diagnosis | 0 | n/a | n/a | n/a |
| constrained_intervention | 14 | 0.554 | 100.0% | 71.4% |
| counterfactual_reasoning | 0 | n/a | n/a | n/a |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 28 | 0.505 | 100.0% | 85.7% |

Total generation latency: **44.9 s**; mean: **1.60 s/record**.
Mean tokens: **1391.3 input / 54.9 output**. Peak allocated CUDA memory: **2.47 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
