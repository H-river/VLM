# Evidence-grounded v5 design protocol

This is a design-stage protocol. It authorizes no additional fine-tuning and does not open the hard-pairs v4 confirmation holdout, unchanged development set, or sealed pilot test.

## Motivation

Hard-pairs v4 proved that exact balance and shortcut control are necessary but not sufficient. Its labels are replay-correct, yet the prompt omits the simulator outcomes needed to distinguish paired cases. The failed seed-42 checkpoints therefore learned conservative output templates rather than the hidden non-monotonic Fresnel response.

Version 5 will separate three capabilities that v4 conflated:

1. **Evidence-grounded experiment reasoning:** reason over raw calibration measurements supplied in the prompt.
2. **Analytic optics intuition:** predict qualitative behavior in a paraxial, monotonic regime from explicit equations and conventions.
3. **Exact simulator use:** invoke a propagation tool or consume its measurements; do not expect a 3B language model to reproduce an FFT solver internally.

The first small pilot covers only capability 1. The other two become separate evaluation tiers rather than mixed labels. Capability 1 itself has three explicit representation tiers:

- **v5A symbolic evidence aggregation:** the measurement adapter supplies numeric control residuals and thresholded sufficiency directions. It never supplies success/agreement flags, ranks, selected actions, witnesses, statuses, or labels. The model performs threshold comparison, set agreement, and constrained selection.
- **v5B numeric evidence aggregation:** the adapter supplies control residuals and sufficiency deltas, but no thresholded directions. The model must classify each delta before aggregating the evidence.
- **v5C raw measurement reasoning:** the prompt supplies raw centroids only, and the model must first compute residuals or deltas.

The failed zero-shot design probes showed that beginning directly with raw measurements asks the 3B model to learn arithmetic, evidence aggregation, and task semantics simultaneously. Any future short curriculum starts with v5A, then evaluates transfer to v5B and v5C. Each tier remains separately scored so scaffold use is never mistaken for raw simulator understanding.

## Prompt-visible evidence

### Constrained intervention

Each prompt contains:

- the setup, current observation, target observation, action constraints, and success tolerance;
- a shuffled table of candidate actions and raw measured centroids;
- in v5A and v5B, a measurement-adapter residual consistent with those centroids;
- no success flag, rank, best-action marker, or target label.

The model must compute centroid residuals, identify every successful candidate, and select the minimum-motion successful action with a declared deterministic tie-break. If no candidate succeeds, it must report the smallest achievable residual.

Paired feasible and infeasible questions share the physical setup, current state, candidate actions, measured action outcomes, table permutation, and output contract. Only the target observation changes.

### Information sufficiency

Each prompt contains:

- the setup, current observation, masked action field, and compatible hidden values;
- for every compatible completion, the hidden value and raw measured centroid;
- in v5A and v5B, a measurement-adapter centroid delta consistent with the current and measured centroids;
- in v5A only, the thresholded direction for each delta;
- the sensor threshold used to classify increase, decrease, or no change;
- no agreement flag, witness pair, status, or target label.

The model must derive the direction for every completion. It returns `answerable` only when all derived directions agree; otherwise it returns `insufficient_information` and identifies two completion values that produce different directions.

Paired questions share setup, current state, conventions, table permutation, and output contract. Their completion tables differ only in the counterbalanced raw trials required to change agreement status.

## Preventing a simple regression task

- Keep exact 1:1 status balance and same-setup minimal pairs.
- Make every control current-to-target error exceed tolerance, so a no-action distance threshold is exactly chance.
- Counterbalance action-table position, action sign, actuator, target direction, answerable direction, and witness positions.
- Vary candidate-grid size, spacing, tolerance, number of successful actions, curve shape, and presence of decoy intensity/width changes.
- Include feasible cases with multiple successes where minimum residual is not the minimum-motion answer.
- Include infeasible cases whose best residual overlaps the successful residual distribution of other tolerances.
- Hold out complete physical scenarios, response-curve shapes, table sizes, numeric bands, and prompt paraphrases.
- Score status, arithmetic, evidence selection, action constraints, and pair consistency separately. Do not reward status alone.

## Design probe before training

Build a local probe from already-used v4 training scenarios only; it is not a future evaluation split. Require all of the following before generating fresh v5 data:

- deterministic evidence-only solver reproduces 100% of targets;
- zero forbidden derived fields in prompts;
- exact 1:1 statuses and zero physical-group overlap between any future train/dev splits;
- current-error, action-position, action-sign, list-order, and fixed-action baselines each achieve at most 0.65 status accuracy;
- every control feasible answer is supported by a prompt-visible trial and every insufficiency label has a prompt-visible conflicting witness pair;
- changing only the declared paired evidence or target changes the correct result in every pair;
- all numerical values are finite and sensor-rounded.

Then run local deterministic zero-shot inference on a fixed 28-record subset of the used-training design probe, with no paid API calls. Use this only as a representation learnability check, never as a generalization estimate:

- if both tasks remain at constant-class F1 near 0.333 and pair-joint accuracy is zero, revise the prompt/evidence representation before training;
- if status F1 is already above 0.85 with strong evidence-field accuracy, do not fine-tune this tier merely to raise an easy benchmark;
- otherwise, freeze the probe and run one short seed-42 trial capped at 50 steps before considering a larger experiment.

## Metrics for a future v5 trial

Promotion requires all of these on a scenario-disjoint confirmation set:

- schema validity at least 0.98;
- status macro-F1 at least 0.75 for each task;
- recall at least 0.70 for every status class;
- at least 0.70 of minimal pairs have both statuses correct;
- control feasible action exact-match at least 0.60;
- control simulator-verified success at least 0.70;
- control minimum-motion/tie-break correctness at least 0.60;
- sufficiency answerable-direction accuracy at least 0.75;
- insufficiency witness-pair validity at least 0.70;
- no more than a 0.03 absolute decline on the five preserved optics tasks.

Thresholds are frozen before the first v5 training run. Failure stops later seeds and sealed-test evaluation.
