# Optics understanding evaluation

Rubric: **v2**.

Evaluated **120 / 120** records.
Macro task score: **0.528**
Valid JSON: **100.0%**; valid schema: **81.7%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 12 | 0.556 | 100.0% | 100.0% |
| information_sufficiency | 12 | 0.533 | 100.0% | 100.0% |
| causal_effects | 18 | 0.856 | 100.0% | 100.0% |
| forward_prediction | 24 | 0.444 | 100.0% | 100.0% |
| diagnosis | 18 | 0.630 | 100.0% | 100.0% |
| constrained_intervention | 24 | 0.177 | 100.0% | 25.0% |
| counterfactual_reasoning | 12 | 0.500 | 100.0% | 66.7% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 108 | 0.501 | 100.0% | 80.6% |
| visual | 12 | 0.554 | 100.0% | 91.7% |

Total generation latency: **555.1 s**; mean: **4.63 s/record**.
Mean tokens: **663.6 input / 82.7 output**. Peak allocated CUDA memory: **2.53 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
