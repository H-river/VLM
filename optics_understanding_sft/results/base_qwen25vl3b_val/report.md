# Optics understanding evaluation

Evaluated **120 / 120** records.
Macro task score: **0.416**
Valid JSON: **94.2%**; valid schema: **58.3%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 12 | 0.000 | 100.0% | 8.3% |
| information_sufficiency | 12 | 0.450 | 100.0% | 100.0% |
| causal_effects | 18 | 0.644 | 100.0% | 100.0% |
| forward_prediction | 24 | 0.278 | 79.2% | 45.8% |
| diagnosis | 18 | 0.402 | 100.0% | 100.0% |
| constrained_intervention | 24 | 0.487 | 100.0% | 0.0% |
| counterfactual_reasoning | 12 | 0.648 | 83.3% | 83.3% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 108 | 0.421 | 94.4% | 54.6% |
| visual | 12 | 0.410 | 91.7% | 91.7% |

Total generation latency: **389.4 s**; mean: **3.24 s/record**.
Mean tokens: **663.6 input / 111.6 output**. Peak allocated CUDA memory: **2.41 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
