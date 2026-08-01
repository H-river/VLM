# Optics understanding evaluation

Rubric: **v2**.

Evaluated **122 / 600** records.
Macro task score: **0.653**
Valid JSON: **100.0%**; valid schema: **99.2%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 7 | 0.905 | 100.0% | 100.0% |
| information_sufficiency | 30 | 0.473 | 100.0% | 100.0% |
| causal_effects | 8 | 0.900 | 100.0% | 100.0% |
| forward_prediction | 8 | 0.250 | 100.0% | 100.0% |
| diagnosis | 31 | 0.875 | 100.0% | 100.0% |
| constrained_intervention | 31 | 0.427 | 100.0% | 100.0% |
| counterfactual_reasoning | 7 | 0.743 | 100.0% | 85.7% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 118 | 0.617 | 100.0% | 100.0% |
| visual | 4 | 0.633 | 100.0% | 75.0% |

Total generation latency: **390.8 s**; mean: **3.20 s/record**.
Mean tokens: **686.6 input / 59.5 output**. Peak allocated CUDA memory: **2.53 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
