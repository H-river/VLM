# Optics understanding evaluation

Rubric: **v2**.

Evaluated **30 / 30** records.
Macro task score: **0.618**
Valid JSON: **100.0%**; valid schema: **70.0%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 0 | n/a | n/a | n/a |
| information_sufficiency | 0 | n/a | n/a | n/a |
| causal_effects | 4 | 0.950 | 100.0% | 100.0% |
| forward_prediction | 6 | 0.278 | 100.0% | 100.0% |
| diagnosis | 11 | 1.000 | 100.0% | 100.0% |
| constrained_intervention | 0 | n/a | n/a | n/a |
| counterfactual_reasoning | 9 | 0.244 | 100.0% | 0.0% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| visual | 30 | 0.622 | 100.0% | 70.0% |

Total generation latency: **159.3 s**; mean: **5.31 s/record**.
Mean tokens: **968.3 input / 96.3 output**. Peak allocated CUDA memory: **2.53 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
