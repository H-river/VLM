# Optics understanding evaluation

Rubric: **v2**.

Evaluated **120 / 120** records.
Macro task score: **0.518**
Valid JSON: **98.3%**; valid schema: **93.3%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 12 | 0.611 | 100.0% | 100.0% |
| information_sufficiency | 12 | 0.350 | 100.0% | 100.0% |
| causal_effects | 18 | 0.789 | 88.9% | 88.9% |
| forward_prediction | 24 | 0.444 | 100.0% | 100.0% |
| diagnosis | 18 | 0.466 | 100.0% | 100.0% |
| constrained_intervention | 24 | 0.467 | 100.0% | 100.0% |
| counterfactual_reasoning | 12 | 0.500 | 100.0% | 50.0% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 108 | 0.512 | 98.1% | 93.5% |
| visual | 12 | 0.554 | 100.0% | 91.7% |

Total generation latency: **473.5 s**; mean: **3.95 s/record**.
Mean tokens: **663.6 input / 76.2 output**. Peak allocated CUDA memory: **2.53 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
