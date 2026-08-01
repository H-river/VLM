# Optics understanding evaluation

Rubric: **v2**.

Evaluated **600 / 600** records.
Macro task score: **0.468**
Valid JSON: **98.8%**; valid schema: **93.8%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 60 | 0.350 | 100.0% | 100.0% |
| information_sufficiency | 120 | 0.410 | 100.0% | 100.0% |
| causal_effects | 60 | 0.780 | 88.3% | 88.3% |
| forward_prediction | 60 | 0.383 | 100.0% | 100.0% |
| diagnosis | 120 | 0.356 | 100.0% | 100.0% |
| constrained_intervention | 120 | 0.455 | 100.0% | 98.3% |
| counterfactual_reasoning | 60 | 0.543 | 100.0% | 53.3% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 570 | 0.450 | 98.8% | 95.1% |
| visual | 30 | 0.434 | 100.0% | 70.0% |

Total generation latency: **2241.8 s**; mean: **3.74 s/record**.
Mean tokens: **678.1 input / 65.9 output**. Peak allocated CUDA memory: **2.52 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
