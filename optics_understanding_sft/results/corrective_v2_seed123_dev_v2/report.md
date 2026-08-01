# Optics understanding evaluation

Rubric: **v2**.

Evaluated **600 / 600** records.
Macro task score: **0.691**
Valid JSON: **100.0%**; valid schema: **98.3%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 60 | 0.900 | 100.0% | 100.0% |
| information_sufficiency | 120 | 0.492 | 100.0% | 100.0% |
| causal_effects | 60 | 0.887 | 100.0% | 100.0% |
| forward_prediction | 60 | 0.383 | 100.0% | 98.3% |
| diagnosis | 120 | 0.920 | 100.0% | 100.0% |
| constrained_intervention | 120 | 0.503 | 100.0% | 100.0% |
| counterfactual_reasoning | 60 | 0.750 | 100.0% | 85.0% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 570 | 0.677 | 100.0% | 100.0% |
| visual | 30 | 0.633 | 100.0% | 66.7% |

Total generation latency: **2002.7 s**; mean: **3.34 s/record**.
Mean tokens: **678.1 input / 65.8 output**. Peak allocated CUDA memory: **2.52 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
