# Optics understanding evaluation

Rubric: **v2**.

Evaluated **30 / 30** records.
Macro task score: **0.936**
Valid JSON: **100.0%**; valid schema: **100.0%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 10 | 0.933 | 100.0% | 100.0% |
| information_sufficiency | 0 | n/a | n/a | n/a |
| causal_effects | 10 | 0.940 | 100.0% | 100.0% |
| forward_prediction | 0 | n/a | n/a | n/a |
| diagnosis | 10 | 0.933 | 100.0% | 100.0% |
| constrained_intervention | 0 | n/a | n/a | n/a |
| counterfactual_reasoning | 0 | n/a | n/a | n/a |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 28 | 0.931 | 100.0% | 100.0% |
| visual | 2 | 1.000 | 100.0% | 100.0% |

Total generation latency: **88.7 s**; mean: **2.96 s/record**.
Mean tokens: **655.9 input / 55.1 output**. Peak allocated CUDA memory: **2.53 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
