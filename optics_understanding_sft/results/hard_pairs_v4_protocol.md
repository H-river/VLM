# Hard-pairs v4 protocol

This protocol is frozen before GPU training. It follows the failed schema-repair v3.1 run without changing the existing promotion thresholds or opening the sealed pilot test.

## Why this round exists

The prior repair recovered approximately 99.2% schema validity but collapsed to `answerable` and `feasible`. Inspection found three training/evaluation artifacts:

1. The compact repair curriculum still used a 4:1 feasible/infeasible control ratio.
2. Ordinary completion-token loss gave a small status value little influence relative to the full structured target.
3. In `dev_v2`, sorted compatible-value order predicts sufficiency status with 98.3% accuracy, and a one-dimensional target-error threshold predicts control status with 79.2% accuracy.

The historical dev set remains unchanged for comparison and promotion. A new shortcut-controlled, scenario-disjoint holdout is added so a candidate cannot qualify merely by exploiting those artifacts.

## Fresh simulator-grounded data

- Seed 8128; 240 independently sampled IID physical scenarios; 960 text records; no paid API calls.
- Each scenario produces four records as two same-setup minimal pairs:
  - one feasible and one infeasible constrained intervention sharing setup, current state, actuator, and exact action grid; only target observation changes;
  - one answerable and one insufficient-information question sharing setup, current state, masked action interface, and prompt template; only compatible hidden values change.
- Sufficiency uses lens-x, the reliably responsive hidden actuator in the current single-lens simulator. Camera-x was rejected during pre-generation testing because answer-changing cases depended on rare sampling/aliasing behavior.
- Both sufficiency members receive the same deterministic rank permutation and identical negative/zero/positive value counts, eliminating list order and sign crossing as class cues.
- Targets use the exact compact `dev_v2` response envelope and retain simulator replay provenance privately.

After a deterministic scenario shuffle, physical groups are assigned before training:

- 100 scenarios / 400 records: decision-weighted warm-up.
- 50 disjoint scenarios / 200 records: mixed with 200 preservation anchors.
- 30 disjoint scenarios / 120 records: balanced checkpoint diagnostic.
- 60 further disjoint scenarios / 240 records: final hard-pair confirmation holdout, never used for checkpoint selection.

## Training

- Start seed 42 from its corrective-v2 step-200 adapter, not from either collapsed repair adapter.
- Stage 1: 100 optimizer steps, one pass over 400 balanced decision rows, learning rate 1e-5. Each exact status-value span receives the same total loss weight of 12 regardless of tokenized label length; prompt and other completion tokens retain ordinary weight.
- Stage 2: continue for 100 optimizer steps on 200 fresh hard-pair rows plus 200 seven-task preservation anchors, learning rate 5e-6 and ordinary completion loss.
- Total continuation is capped at 200 optimizer steps. Teacher-forced loss is diagnostic only; deterministic free generation selects checkpoints.

## Gates and stopping rule

Seed 42 is evaluated in this order:

1. Use the 120-record diagnostic to compare the warm-up and mixed checkpoints. A checkpoint is viable only if it meets the status, recall, schema, simulator-success, and paired-accuracy thresholds below.
2. Confirm the selected checkpoint on the separate 240-record hard-pair holdout:
   - schema validity >= 0.93;
   - sufficiency status macro-F1 >= 0.60;
   - control status macro-F1 >= 0.60;
   - feasible and infeasible control recall each >= 0.60;
   - feasible simulator success >= 0.40;
   - at least 40% of minimal pairs have both statuses correct.
3. Only if the holdout is viable, run a balanced early diagnostic on unchanged `dev_v2`.
4. Only if that diagnostic is viable, generate all 600 unchanged `dev_v2` answers and apply the existing frozen gates, including diagnosis, schema, and anchor non-regression.
5. Run seeds 123 and 314 only if seed 42 passes every full-dev gate.
6. Require at least two of three seeds to pass before evaluating the sealed pilot test exactly once.

No threshold may be lowered after observing results. If a stage fails, preserve the result, diagnose the paired errors, and stop later compute for that intervention.
