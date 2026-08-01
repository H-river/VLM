# Optics understanding evaluation

Evaluated **120 / 120** records.
Macro task score: **0.518**
Valid JSON: **99.2%**; valid schema: **90.8%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 12 | 0.167 | 100.0% | 100.0% |
| information_sufficiency | 12 | 0.500 | 100.0% | 75.0% |
| causal_effects | 18 | 0.856 | 100.0% | 100.0% |
| forward_prediction | 24 | 0.431 | 100.0% | 95.8% |
| diagnosis | 18 | 0.432 | 100.0% | 100.0% |
| constrained_intervention | 24 | 0.600 | 100.0% | 100.0% |
| counterfactual_reasoning | 12 | 0.639 | 91.7% | 41.7% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 108 | 0.540 | 100.0% | 91.7% |
| visual | 12 | 0.438 | 91.7% | 83.3% |

Total generation latency: **491.4 s**; mean: **4.09 s/record**.
Mean tokens: **663.6 input / 78.3 output**. Peak allocated CUDA memory: **2.53 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
