# Qwen Learned-H1 Controller Selector v1

## Decision

**Not ready for frozen evaluation.** The candidate-only Qwen learnability and closed-loop gates are RED. Frozen evaluation remains locked.

## Scope and provenance

- Candidate-train matrix: 48 records × 3 strategies × 20 roots = 2,880 episodes; 1,320 exact reusable outcomes and 1,560 new candidate-train CEM outcomes.
- Candidate-dev: no CEM was rerun. The 24 pre-CEM selector inputs were inferred after the candidate-train method freeze; confirmation retrieves roots from `artifacts/reduced_h1_rescue_v3/raw_episode_results.jsonl`.
- Binary action space was retained because strategy_c had zero discovery selections and dropping it changed candidate-train success and cost by 0.0.
- The deployed fail-safe is fixed `strategy_b` on invalid JSON or low confidence.

## Qwen grouped-CV

| Method | Balanced accuracy | Recall default | Recall strategy_b | Predictions |
|---|---:|---:|---:|---|
| ordinary_hard | 0.505 | 0.323 | 0.688 | {'default': 15, 'strategy_b': 33} |
| stability_filtered | 0.505 | 0.323 | 0.688 | {'default': 15, 'strategy_b': 33} |

The selected method (`ordinary_hard`) misses the learnability criteria: balanced accuracy 0.505 < 0.700, default recall 0.323 < 0.600, and 0/3 nontrivial seeds (minimum 2).

## Frozen candidate-dev inference

| Seed | Schema validity | Deterministic match | Deployed policies | Fallbacks |
|---:|---:|---:|---|---:|
| 2026080901 | 1.000 | 1.000 | {'strategy_b': 24} | 0 |
| 2026080902 | 1.000 | 1.000 | {'default': 24} | 0 |
| 2026080903 | 1.000 | 1.000 | {'strategy_b': 24} | 0 |

## Candidate-dev confirmation retrieval

Fixed strategy_b achieves 29.167% strict success; the held-out discovery oracle reaches 37.500%, leaving 8.333pp selector headroom.

| Seed | Gain vs fixed B (pp) | 95% CI | Headroom recovery |
|---:|---:|---|---:|
| 2026080901 | 0.00 | [0.00, 0.00] | 0.000 |
| 2026080902 | -10.42 | [-25.00, 4.58] | -1.250 |
| 2026080903 | 0.00 | [0.00, 0.00] | 0.000 |

## Gates and verification

- LABEL_QUALITY_GATE: **GREEN**
- INPUT_SUFFICIENCY_GATE: **GREEN**
- QWEN_LEARNABILITY_GATE: **RED**
- CLOSED_LOOP_GATE: **RED**
- Regression/integration: **PASS** (32 targeted tests passed; strict duplicate-key rejection and stored-output revalidation passed).

**Disposition:** NOT_READY_FOR_FROZEN_EVAL. Frozen evaluation remains locked.
