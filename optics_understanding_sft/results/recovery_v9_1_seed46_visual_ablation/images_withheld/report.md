# Optics understanding evaluation

Rubric: **v2**.

Evaluated **30 / 30** records.
Macro task score: **0.632**
Valid JSON: **100.0%**; valid schema: **70.0%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 0 | n/a | n/a | n/a |
| information_sufficiency | 0 | n/a | n/a | n/a |
| causal_effects | 4 | 0.950 | 100.0% | 100.0% |
| forward_prediction | 6 | 0.222 | 100.0% | 100.0% |
| diagnosis | 11 | 1.000 | 100.0% | 100.0% |
| constrained_intervention | 0 | n/a | n/a | n/a |
| counterfactual_reasoning | 9 | 0.356 | 100.0% | 0.0% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| visual | 30 | 0.644 | 100.0% | 70.0% |

Total generation latency: **132.3 s**; mean: **4.41 s/record**.
Mean tokens: **636.3 input / 80.1 output**. Peak allocated CUDA memory: **2.45 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
