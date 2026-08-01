# Optics understanding evaluation

Rubric: **v2**.

Evaluated **101 / 600** records.
Macro task score: **0.595**
Valid JSON: **99.0%**; valid schema: **79.2%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 6 | 0.833 | 100.0% | 100.0% |
| information_sufficiency | 25 | 0.280 | 100.0% | 64.0% |
| causal_effects | 7 | 0.914 | 100.0% | 100.0% |
| forward_prediction | 6 | 0.194 | 100.0% | 83.3% |
| diagnosis | 25 | 0.920 | 100.0% | 100.0% |
| constrained_intervention | 25 | 0.306 | 100.0% | 60.0% |
| counterfactual_reasoning | 7 | 0.714 | 85.7% | 85.7% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 98 | 0.552 | 100.0% | 80.6% |
| visual | 3 | 0.389 | 66.7% | 33.3% |

Total generation latency: **315.7 s**; mean: **3.13 s/record**.
Mean tokens: **686.0 input / 59.4 output**. Peak allocated CUDA memory: **2.53 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
