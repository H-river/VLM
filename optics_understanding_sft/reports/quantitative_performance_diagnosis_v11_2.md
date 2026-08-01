# Quantitative Performance Diagnosis and Improvement Report (v11.2 confirmed)

## Outcome

The remaining direct-language errors were structural numerical-evidence failures, not evidence that a few more fine-tuning steps would reliably solve the tasks. The unchanged seed-49 checkpoint is now used for registered-tool choice, exact source mapping, and compact result interpretation. Deterministic tools perform the arithmetic, physical deadband classification, and simulator candidate replay.

| Metric | v10.21 | v11.2 confirmed |
|---|---:|---:|
| Setup interpretation | 0.967 | 1.000 |
| Causal effects | 0.940 | 1.000 |
| Diagnosis | 0.933 | 1.000 |
| Seven-task equal macro | 0.977 | 1.000 |
| Direct-tool development end-to-end | not used | 30/30 |
| Direct-tool confirmation end-to-end | not used | 90/90 |
| Confirmation physical-group end-to-end | not used | 74/74 |
| Confirmation group-level 95% Wilson lower bound | not used | 0.951 |

The confirmation panel is physically group-disjoint from the 30-source development panel. It contains 30 sources for each of setup interpretation, causal effects, and diagnosis, converted into 270 independently generated orchestration stages. All tool choices, source mappings, deterministic executions, and final interpretations were exact.

## Newly isolated causes

### 12. Token generation is unreliable for exact visible arithmetic

The setup failure exposed every required value, but the model rounded a focal length to `0.11836 m` instead of the dataset contract's `0.118357 m`. Token cross-entropy has no intrinsic representation of decimal distance or the evaluator's rounding contract.

Resolution: `normalize_optical_setup_summary_v1` receives an opaque registered setup handle plus visible adjustability information and performs exact conversion and rounding. The LLM maps visible roles and copies the validated result.

### 13. Causal labels depended on omitted deadband rules

The causal prompt exposed before and after observations but did not expose the exact 1 px centroid, 2 px width, and 5% peak thresholds. Near a boundary, more than one categorical answer was compatible with the visible prompt.

Resolution: `classify_registered_transition_v1` receives registered before/after state handles and explicit threshold roles, then applies the same deterministic rule used by ground-truth generation. This removes hidden-rubric guessing.

### 14. Diagnosis required simulator responses that were not prompt-visible

The diagnosis prompt listed candidate interventions but hid their simulated after-states. The model could not reliably distinguish unique from ambiguous explanations by replaying optics internally.

Resolution: `replay_registered_diagnosis_candidates_v1` replays every visible candidate against the registered observation and returns every matching index plus the derived status. The LLM is tested on choosing the tool, mapping the observation and candidates, and interpreting the complete validated result.

## Validation safeguards

- No sealed pilot test labels were opened.
- Tool materialization is asserted to reconstruct the original simulator target exactly during dataset build and audit.
- The confirmation selector excludes complete physical groups used by the development panel.
- Promotion requires both panels to score at least 0.95 end-to-end and requires the confirmation physical-group Wilson lower bound to be at least 0.95.
- Text-only tool records require no dummy images; the existing seed-49 checkpoint is unchanged and no paid API was used.
- Regression tests pass: 94 non-PyTorch tests plus 9 QLoRA/Torch tests.

## Claim boundary

The 1.000 seven-task result is a routed-system score. It does not show that the language model natively performs high-precision optics arithmetic, simulator replay, or image metrology. The LLM is promoted for registered-tool selection, input construction, and result interpretation. All current physics evidence is synthetic and limited to the single-source, single-lens, camera topology.

## Remaining weaknesses and next target

The main remaining quantitative limitation is visual measurement under degradation, especially paired width direction under sensor noise. Clean production visual macro-F1 is 0.927, while ordinary-noise production macro-F1 is 0.869 and its conservative lower bound is 0.821. On noisy pairs, sigma-x and sigma-y macro-F1 are 0.730 and 0.751; counterfactual pair joint exact accuracy is 0.683. The next iteration should improve the deterministic paired-width estimator using training-only corrupted calibration, then require preservation on clean, blur, dim-plus-noise, and saturation panels before promotion.

## Reproducible evidence paths

- Direct development dataset audit: `data/direct_tool_orchestration_v11_2/audit_report.json`
- Direct development result: `results/direct_tool_orchestration_v11_2_seed49/summary.json`
- Group-disjoint confirmation panel: `data/dev_v2/canonical/val_direct_confirmation90_v11.jsonl`
- Confirmation tool audit: `data/direct_tool_orchestration_confirmation_v11_2/audit_report.json`
- Confirmation result: `results/direct_tool_orchestration_confirmation_v11_2_seed49/summary.json`
- Confirmed unified gate: `results/unified_system_v11_2_confirmed_seed49_gate/summary.json`
- Earlier visual diagnosis and robustness report: `reports/quantitative_performance_diagnosis_v10_21.md`
