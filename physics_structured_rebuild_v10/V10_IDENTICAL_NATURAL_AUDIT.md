# V10 identical natural-action audit

This is a diagnostic-only audit. It did not train, select, or replace a model
and did not open the raw locked-test JSONL.

## Finding

The 18.12% equality is real at the aggregate level but does **not** come from
identical checkpoints or identical numerical predictions. A, B, and C have
the same 29/160 natural-action success mask. D also has 29/160 successes, but
it swaps two successes for two failures, so four group outcomes differ.

All four checkpoint hashes, metadata identities, prediction hashes, and model
states are distinct. Every A-D pair differs numerically on all 160 development
groups at the requested action:

| Pair | Mean absolute prediction difference | Maximum difference | Natural groups differing |
|---|---:|---:|---:|
| A-B | 0.05727 | 0.29512 | 160/160 |
| A-C | 0.06112 | 0.34335 | 160/160 |
| A-D | 0.53787 | 9.11132 | 160/160 |
| B-C | 0.02115 | 0.12396 | 160/160 |
| B-D | 0.53669 | 9.14515 | 160/160 |
| C-D | 0.53723 | 9.11697 | 160/160 |

Natural strict-mask comparison:

| Pair | Different groups | Left-only successes | Right-only successes |
|---|---:|---:|---:|
| A-B | 0 | 0 | 0 |
| A-C | 0 | 0 | 0 |
| A-D | 4 | 2 | 2 |
| B-C | 0 | 0 | 0 |
| B-D | 4 | 2 | 2 |
| C-D | 4 | 2 | 2 |

The saved evaluation masks were independently recomputed from the checkpoints;
all four had zero discrepancies.

## Normalized fit and development error

The fit column uses each candidate's 326 optimizer-fit groups; A's come from
the sampled legacy distribution and B-D's from v10. Each group contributes all
81 correlated actions.

| Model | Parameters | Best / inferred epochs run | Fit normalized MAE | Fit full-surface strict | Development normalized MAE | Development full-surface strict |
|---|---:|---:|---:|---:|---:|---:|
| A | 96,517 | 2 / 14 | 1.206 | 13.97% | 1.291 | 12.11% |
| B | 96,517 | 1 / 13 | 1.306 | 12.22% | 1.298 | 11.90% |
| C | 96,517 | 1 / 13 | 1.306 | 12.20% | 1.298 | 11.91% |
| D | 509,706 | 37 / 49 | 0.987 | 17.67% | 1.159 | 13.19% |

Frozen v9's reference normalized MAE is 0.915 on the common v10 training
distribution and 0.932 on development. Full-surface strict success is 36.30%
and 38.39%, respectively.

The high fit-set errors show that A-D underfit even their 326 optimizer-fit
groups. This is most acute for A-C, which selected epochs 2/1/1 and stopped
after approximately 14/13/13 epochs. D used 509,706 parameters and about 49
epochs, but still remained far from fitting the training surface.

## Baseline parity mismatch

- A-C: 96,517 parameters; D: 509,706 parameters.
- Each v10 model fit 326 groups = 26,406 transitions, with 58 groups reserved
  for internal checkpoint calibration.
- Frozen v9's shared neural foundation alone has 825,236 parameters and was
  trained for 13 epochs (best 7) on 10,500 groups = 850,500 transitions.
- Frozen v9 then adds protected expert pipelines and a 180-iteration,
  20-feature HGB selector trained on 1,600 groups and 11,675 exclusive
  disagreement transitions. Its capacity is therefore not represented by the
  825,236 neural parameter count alone.
- V10 uses a z-scored 17-value context and either an opaque action-ID embedding
  or explicit structured actions. Frozen v9 uses 46 per-transition features
  (17 context + 4 action + 25 engineered terms), five out-of-fold prior
  predictions, z-scoring, residual learning around that prior, and later
  protected selectors.

Thus A-D were neither data-parity nor architecture/preprocessing-parity
reproductions of frozen v9. Their negative results diagnose an inadequate
common baseline, not a fair ceiling on the proposed objectives or structure.

## Natural-action indexing

Indexing passed for all 160 development groups:

- canonical 81-action order failures: 0;
- deterministic stored-index mismatches: 0;
- indexed action versus canonical action mismatches: 0;
- stored-mask versus recomputed-checkpoint mismatches: 0.

The requested action is correctly gathered as
`prediction[group_index, natural_requested_action_index]`.

## Development versus locked composition

Regime composition is identical, so it does not explain frozen v9's
43.75% to 30.63% decline:

| Regime | Development groups | Locked groups | Development successes | Locked successes | Change |
|---|---:|---:|---:|---:|---:|
| camera_boundary | 24 | 24 | 4 | 0 | -4 |
| clipping | 24 | 24 | 2 | 2 | 0 |
| focusing | 24 | 24 | 16 | 11 | -5 |
| high_offset_interaction | 16 | 16 | 6 | 4 | -2 |
| ordinary | 56 | 56 | 35 | 29 | -6 |
| tolerance_boundary | 16 | 16 | 7 | 3 | -4 |

The loss is 21 successes across the same regime allocation: ordinary -6,
focusing -5, camera-boundary -4, tolerance-boundary -4, high-offset -2, and
clipping 0. Natural-action cardinality composition changed modestly
(development 36/49/37/38 versus locked 38/36/46/40 for cardinalities 1-4),
but the saved locked report does not contain a requested-cardinality success
breakdown. The supported explanation is a harder independently seeded draw
within the same regimes, especially camera and tolerance boundaries—not a
regime-mixture shift.

## Implication for v11

The proposed baseline-parity and scaling sequence is supported. Resume the
preserved deterministic generation, establish a new development boundary
while keeping the v10 locked data closed, and require a plain current-style
model to approach frozen v9 before repeating the loss/structure ablations.
