# Optics understanding evaluation

Rubric: **v2**.

Evaluated **70 / 70** records.
Macro task score: **0.693**
Valid JSON: **100.0%**; valid schema: **98.6%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 10 | 0.933 | 100.0% | 100.0% |
| information_sufficiency | 10 | 0.440 | 100.0% | 100.0% |
| causal_effects | 10 | 0.920 | 100.0% | 100.0% |
| forward_prediction | 10 | 0.383 | 100.0% | 100.0% |
| diagnosis | 10 | 0.933 | 100.0% | 100.0% |
| constrained_intervention | 10 | 0.460 | 100.0% | 100.0% |
| counterfactual_reasoning | 10 | 0.780 | 100.0% | 90.0% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 66 | 0.696 | 100.0% | 100.0% |
| visual | 4 | 0.642 | 100.0% | 75.0% |

Total generation latency: **306.8 s**; mean: **4.38 s/record**.
Mean tokens: **666.3 input / 80.6 output**. Peak allocated CUDA memory: **2.53 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
