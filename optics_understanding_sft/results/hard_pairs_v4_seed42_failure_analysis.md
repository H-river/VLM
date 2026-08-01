# Hard-pairs v4 seed-42 failure analysis

## Frozen decision

All predeclared checkpoints failed the balanced 120-record diagnostic. The 240-record confirmation holdout, unchanged development set, additional seeds, and sealed test remain unevaluated.

| Checkpoint | Schema | Sufficiency F1 | Control F1 | Feasible recall | Pair joint | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| warmup100 | 0.992 | 0.333 | 0.333 | 0.000 | 0.000 | no |
| mixed50 | 1.000 | 0.333 | 0.333 | 0.000 | 0.000 | no |
| mixed100 | 1.000 | 0.333 | 0.333 | 0.000 | 0.000 | no |

## Observed failure mode

- The final checkpoint returned the same status for all 30 sufficiency pairs and all 30 control pairs.
- Sufficiency pair responses were byte-identical for 30/30 pairs.
- For control, 43/60 predicted best residuals exactly matched the no-action current-to-target distance. The mean absolute difference was 0.0204 px.
- Adapter hashes differ across checkpoints, so this is not a trainer no-op. Generated decisions nevertheless remained almost unchanged.

## Why balance was not enough

- Every feasible control target starts outside the declared 2 px tolerance (minimum 2.0416 px). Feasibility is visible only after replaying the allowed actions.
- The hidden nine-action response curves average 3.20 local slope reversals. Their outcomes are not present in the prompt.
- Sufficiency pairs replace a median of 2.0 compatible values; the median replacement magnitude is 0.020 mm. Insufficient cases average 2.27 direction changes after sorting those values, and the per-value measurements are also absent from the prompt.
- The dataset is label-correct and shortcut-controlled, but these records demand high-precision emulation of a non-monotonic Fresnel simulator. They do not isolate reasoning over observable experimental evidence.

## Recommended v5 experiment

1. Add a raw calibration table to control prompts: candidate action plus measured after-observation. Ask the model to compute residuals, enforce the tolerance, and choose the minimum-motion successful action.
2. Add raw completion measurements to sufficiency prompts: hidden-value completion plus centroid observation. Ask the model to derive directions and decide whether all compatible completions agree.
3. Keep labels balanced and pairs counterfactual, but vary grid size, ordering, decoys, tolerances, and response-curve shape so success cannot reduce to one threshold or slot.
4. Add a separate analytic-physics tier using paraxial, monotonic setups for genuine optics intuition. Reserve exact Fresnel behavior for simulator-tool-use evaluation rather than requiring a 3B model to internalize the solver.
5. Before any new 200-step run, test base and retained-reference models on a small frozen v5 diagnostic. Train only if the task is above chance yet below the desired gate, proving that the prompt supplies usable evidence and still has learning headroom.
