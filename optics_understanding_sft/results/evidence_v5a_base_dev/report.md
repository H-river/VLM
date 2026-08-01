# Optics understanding evaluation

Rubric: **v2**.

Evaluated **200 / 200** records.
Macro task score: **0.417**
Valid JSON: **100.0%**; valid schema: **76.0%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 0 | n/a | n/a | n/a |
| information_sufficiency | 100 | 0.418 | 100.0% | 100.0% |
| causal_effects | 0 | n/a | n/a | n/a |
| forward_prediction | 0 | n/a | n/a | n/a |
| diagnosis | 0 | n/a | n/a | n/a |
| constrained_intervention | 100 | 0.416 | 100.0% | 52.0% |
| counterfactual_reasoning | 0 | n/a | n/a | n/a |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 200 | 0.417 | 100.0% | 76.0% |

Total generation latency: **364.9 s**; mean: **1.82 s/record**.
Mean tokens: **1416.4 input / 59.1 output**. Peak allocated CUDA memory: **2.47 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
