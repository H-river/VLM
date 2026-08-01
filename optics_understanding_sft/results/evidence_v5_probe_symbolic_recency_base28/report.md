# Optics understanding evaluation

Rubric: **v2**.

Evaluated **28 / 400** records.
Macro task score: **0.493**
Valid JSON: **100.0%**; valid schema: **100.0%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 0 | n/a | n/a | n/a |
| information_sufficiency | 14 | 0.486 | 100.0% | 100.0% |
| causal_effects | 0 | n/a | n/a | n/a |
| forward_prediction | 0 | n/a | n/a | n/a |
| diagnosis | 0 | n/a | n/a | n/a |
| constrained_intervention | 14 | 0.500 | 100.0% | 100.0% |
| counterfactual_reasoning | 0 | n/a | n/a | n/a |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 28 | 0.493 | 100.0% | 100.0% |

Total generation latency: **45.0 s**; mean: **1.61 s/record**.
Mean tokens: **1432.8 input / 53.3 output**. Peak allocated CUDA memory: **2.47 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
