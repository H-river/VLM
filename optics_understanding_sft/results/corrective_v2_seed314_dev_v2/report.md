# Optics understanding evaluation

Rubric: **v2**.

Evaluated **600 / 600** records.
Macro task score: **0.694**
Valid JSON: **99.8%**; valid schema: **98.2%**.

| Task | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| setup_interpretation | 60 | 0.900 | 100.0% | 100.0% |
| information_sufficiency | 120 | 0.522 | 100.0% | 100.0% |
| causal_effects | 60 | 0.890 | 100.0% | 100.0% |
| forward_prediction | 60 | 0.383 | 98.3% | 96.7% |
| diagnosis | 120 | 0.903 | 100.0% | 100.0% |
| constrained_intervention | 120 | 0.512 | 100.0% | 100.0% |
| counterfactual_reasoning | 60 | 0.750 | 100.0% | 85.0% |

| Modality | N | Score | JSON | Schema |
|---|---:|---:|---:|---:|
| text | 570 | 0.684 | 99.8% | 99.8% |
| visual | 30 | 0.589 | 100.0% | 66.7% |

Total generation latency: **2022.8 s**; mean: **3.37 s/record**.
Mean tokens: **678.1 input / 65.8 output**. Peak allocated CUDA memory: **2.52 GiB**.

Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.
