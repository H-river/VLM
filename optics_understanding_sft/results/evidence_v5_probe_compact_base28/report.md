# Optics understanding evaluation

Rubric: **v2**.

Evaluated **28 / 400** records.
Macro task score: **0.307**
Valid JSON: **100.0%**; valid schema: **60.7%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 0 | n/a | n/a | n/a |
| information_sufficiency | 14 | 0.400 | 100.0% | 100.0% |
| causal_effects | 0 | n/a | n/a | n/a |
| forward_prediction | 0 | n/a | n/a | n/a |
| diagnosis | 0 | n/a | n/a | n/a |
| constrained_intervention | 14 | 0.214 | 100.0% | 21.4% |
| counterfactual_reasoning | 0 | n/a | n/a | n/a |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 28 | 0.307 | 100.0% | 60.7% |

Total generation latency: **46.4 s**; mean: **1.66 s/record**.
Mean tokens: **1285.7 input / 57.0 output**. Peak allocated CUDA memory: **2.46 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
