# Seven-task optics system gate

Passed: **True**
Equal-task macro: **0.970** (direct-only reference: 0.691)

| Task | Route | Score | Direct-only reference |
|---|---|---:|---:|
| setup_interpretation | direct_structured_answer | 0.933 | 0.933 |
| information_sufficiency | enumerate_compatible_completions_then_compact_interpretation | 1.000 | 0.440 |
| causal_effects | direct_structured_answer | 0.940 | 0.940 |
| forward_prediction | registered_state_deterministic_simulator | 1.000 | 0.383 |
| diagnosis | direct_structured_answer | 0.920 | 0.920 |
| constrained_intervention | exhaustive_grid_tool_then_compact_interpretation | 1.000 | 0.460 |
| counterfactual_reasoning | registered_paired_state_deterministic_simulator | 1.000 | 0.760 |

Scores routed through deterministic tools measure the LLM's tool choice, source-path construction, and result interpretation; they do not claim native pixel-level metrology.
