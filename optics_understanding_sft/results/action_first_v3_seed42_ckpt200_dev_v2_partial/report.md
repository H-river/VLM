# Optics understanding evaluation

Rubric: **v2**.

Evaluated **111 / 600** records.
Macro task score: **0.577**
Valid JSON: **100.0%**; valid schema: **65.8%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 7 | 0.857 | 100.0% | 100.0% |
| information_sufficiency | 27 | 0.222 | 100.0% | 51.9% |
| causal_effects | 7 | 0.943 | 100.0% | 100.0% |
| forward_prediction | 7 | 0.238 | 100.0% | 100.0% |
| diagnosis | 28 | 0.976 | 100.0% | 100.0% |
| constrained_intervention | 28 | 0.062 | 100.0% | 14.3% |
| counterfactual_reasoning | 7 | 0.743 | 100.0% | 85.7% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 108 | 0.491 | 100.0% | 65.7% |
| visual | 3 | 0.511 | 100.0% | 66.7% |

Total generation latency: **279.9 s**; mean: **2.52 s/record**.
Mean tokens: **685.3 input / 48.1 output**. Peak allocated CUDA memory: **2.53 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
