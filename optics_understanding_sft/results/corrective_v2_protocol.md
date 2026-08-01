# Corrective v2 frozen protocol

Frozen before full data generation and training. The sealed pilot test remains out of scope.

## Data

- Corrective training: 500 independently sampled scenarios and 2,000 records, seed 2718.
- Independent development: 150 scenarios and 600 records, seed 1618.
- Visual share: 5% in each dataset.
- IDs, scenario seeds, images, and match groups must be disjoint from pilot v1, targeted v1.1, and each other.
- Multi-status tasks use exact balance with nuisance counterbalancing by actuator or axis.
- Matched groups contain one record per status: pairs for sufficiency/control and triples for diagnosis.
- Before training, matched control pairs must have mean baseline-error difference at most 2 px and 95th-percentile difference at most 5 px; the fixed-gain controller must remain below 60% success.

Corrective training task counts: control 400, sufficiency 400, diagnosis 396, and 201 each for setup, causal, forward, and counterfactual tasks. Development counts: control 120, sufficiency 120, diagnosis 120, and 60 for each anchor task.

## Curriculum

Use every corrective record once. Add the 462 original pilot training records from the four anchor families only, producing a 2,462-record curriculum. Exclude old control, sufficiency, and diagnosis records because their task-index schedules are confounded with status. No repetition-based weighting is allowed.

Shuffle matched groups as units and use a sequential trainer sampler so pair/triple members remain adjacent, normally within the same four-microbatch gradient-accumulation window. Different seeds still vary model stochasticity; the curriculum unit order is frozen at seed 202.

## Training

- Start point: original QLoRA step-40 adapter.
- Seeds: 42, 123, and 314.
- 200 optimizer steps per seed; checkpoints and teacher-forced evaluation at steps 100 and 200.
- Learning rate 3e-5, gradient accumulation 4, greedy generation.
- All other model, quantization, and LoRA settings remain fixed.

The longer 200-step run replaces the earlier 30-step estimate because the approved dataset was expanded from 480 to 2,000 corrective training records.

## Selection

Primary evaluation uses rubric v2 on all 600 independent development records. Report v1 only for continuity. Compare runs with a 10,000-replicate paired bootstrap over the 150 physical scenario groups and report cross-seed variation.

A candidate is promotable only when at least two of three seeds satisfy all gates:

- feasible and infeasible control recall at least 0.60 each;
- feasible-control simulator success at least 0.40;
- status macro-F1 at least 0.60 for sufficiency, diagnosis, and control;
- schema validity at least 0.93;
- no anchor-task regression greater than 0.05 against original step 40 on the same dev data.

If no seed passes, record `no_promotion`, retain original step 40, diagnose the cross-seed pattern, and do not lower gates or inspect the sealed test.
