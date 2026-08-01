# Optics understanding evaluation

Rubric: **v2**.

Evaluated **120 / 120** records.
Macro task score: **0.538**
Valid JSON: **100.0%**; valid schema: **90.0%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 12 | 0.611 | 100.0% | 100.0% |
| information_sufficiency | 12 | 0.450 | 100.0% | 100.0% |
| causal_effects | 18 | 0.856 | 100.0% | 100.0% |
| forward_prediction | 24 | 0.438 | 100.0% | 100.0% |
| diagnosis | 18 | 0.659 | 100.0% | 100.0% |
| constrained_intervention | 24 | 0.250 | 100.0% | 75.0% |
| counterfactual_reasoning | 12 | 0.500 | 100.0% | 50.0% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 108 | 0.521 | 100.0% | 89.8% |
| visual | 12 | 0.524 | 100.0% | 91.7% |

Total generation latency: **496.0 s**; mean: **4.13 s/record**.
Mean tokens: **663.6 input / 84.2 output**. Peak allocated CUDA memory: **2.53 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
