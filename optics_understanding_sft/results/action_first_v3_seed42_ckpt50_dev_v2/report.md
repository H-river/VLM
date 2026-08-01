# Optics understanding evaluation

Rubric: **v2**.

Evaluated **600 / 600** records.
Macro task score: **0.623**
Valid JSON: **99.5%**; valid schema: **84.2%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 60 | 0.911 | 100.0% | 100.0% |
| information_sufficiency | 120 | 0.265 | 100.0% | 67.5% |
| causal_effects | 60 | 0.880 | 96.7% | 96.7% |
| forward_prediction | 60 | 0.386 | 100.0% | 98.3% |
| diagnosis | 120 | 0.911 | 100.0% | 100.0% |
| constrained_intervention | 120 | 0.283 | 100.0% | 63.3% |
| counterfactual_reasoning | 60 | 0.727 | 98.3% | 85.0% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 570 | 0.585 | 99.6% | 85.1% |
| visual | 30 | 0.532 | 96.7% | 66.7% |

Total generation latency: **2196.8 s**; mean: **3.66 s/record**.
Mean tokens: **678.1 input / 69.8 output**. Peak allocated CUDA memory: **2.53 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
