# Optics understanding evaluation

Rubric: **v2**.

Evaluated **120 / 600** records.
Macro task score: **0.648**
Valid JSON: **100.0%**; valid schema: **99.2%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 7 | 0.905 | 100.0% | 100.0% |
| information_sufficiency | 30 | 0.440 | 100.0% | 100.0% |
| causal_effects | 8 | 0.900 | 100.0% | 100.0% |
| forward_prediction | 8 | 0.250 | 100.0% | 100.0% |
| diagnosis | 30 | 0.871 | 100.0% | 100.0% |
| constrained_intervention | 30 | 0.425 | 100.0% | 100.0% |
| counterfactual_reasoning | 7 | 0.743 | 100.0% | 85.7% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 116 | 0.606 | 100.0% | 100.0% |
| visual | 4 | 0.633 | 100.0% | 75.0% |

Total generation latency: **387.2 s**; mean: **3.23 s/record**.
Mean tokens: **685.4 input / 60.3 output**. Peak allocated CUDA memory: **2.53 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
