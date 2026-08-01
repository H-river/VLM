# Action-first v3 focused ablation protocol

This protocol is frozen before generation or training. It addresses the two failed corrective-v2 status gates without changing `data/dev_v2` or opening the sealed pilot test.

## Training-only data

- Seed 5772; 320 fresh physical scenarios; 640 records; no visual records.
- 320 constrained-intervention records: 240 feasible and 80 infeasible-within-limits.
- 320 information-sufficiency records: 160 answerable and 160 insufficient-information.
- No diagnosis or general-task records are generated in this focused set.
- Every control target serializes action evidence before the status: actuator, signed movement, predicted residual, and executable validity.
- Feasible control records form 120 matched action-diversity pairs on the two responsive lens actuators. Each pair shares the actuator and a closely matched prompt-visible baseline error, uses opposite movement signs, and contains one minimum-motion and one near-boundary action. Camera translation remains represented in the starting corrective-v2 adapter and unchanged dev; the simulator response inside its small declared grids is often sub-pixel and cannot ground nonzero minimum-motion labels reliably.
- All 80 infeasible records are attached to 80 of those action pairs, producing 80 matched triplets and 40 feasible-only pairs. Infeasible frequency is lower than corrective v2 and is not oversampled.
- Sufficiency pairs use the same masked action field. Answerable members have invariant completion outputs; insufficient members include an explicit minimal set of compatible completions that change the correct output.

## Small curriculum ablation

- Continue each seed from its corresponding corrective-v2 200-step adapter.
- First run seed 42 only on the focused curriculum; evaluate all 600 unchanged `dev_v2` records.
- The curriculum contains all 640 focused records plus 160 deterministic, non-repeated preservation anchors (32 each for setup, causal, forward, diagnosis, and counterfactual tasks) from the prior corrective training curriculum. It must not contain pilot validation or test records.
- One complete curriculum pass is 200 optimizer steps at gradient accumulation 4. Continue with learning rate 2e-5 and save/evaluate training loss every 50 steps; checkpoint choice is made only from unchanged-dev task metrics, never training loss.
- Only if the first ablation improves the failed control/sufficiency gates without breaking schema or anchor gates will seeds 123 and 314 be run.

## Frozen evaluation and promotion rule

- Use rubric v2, all 600 `dev_v2` records, deterministic free generation, and the existing promotion checker without threshold changes.
- Required per seed: feasible control recall >= 0.60; infeasible control recall >= 0.60; feasible simulator success >= 0.40; status macro-F1 >= 0.60 for sufficiency, diagnosis, and control; schema validity >= 0.93; no anchor task regression below -0.05 versus the original step-40 reference.
- At least two of three seeds must pass every gate before the sealed pilot test can be reopened.
- If fewer than two pass, no adapter is promoted and no sealed-test inference is run.
