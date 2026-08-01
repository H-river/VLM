# Autonomous optics SFT worklist — 2026-07-14

This was an eight-hour maximum worklist, completed early because the local 3B QLoRA runs and validation passes finished faster than the timebox. No paid API calls were made.

| Timebox | Work item | Result |
|---|---|---|
| Hour 0-1 | Paired error taxonomy for base and initial checkpoints | Completed; found unit-scale errors and constant feasibility predictions. |
| Hour 1-2 | Audit and freeze evaluator rubric v2 | Completed; corrected hidden setup field, control branch leakage, and counterfactual weighting while preserving v1. |
| Hour 2-4 | Generate fresh targeted training-only data and curriculum | Completed; 100 scenarios / 400 records, 10% visual, balanced key labels, no overlap with pilot IDs. |
| Hour 4-6 | Run two controlled QLoRA ablations | Completed; one 20-step continuation and one fresh 40-step adapter. |
| Hour 6-7 | Full 120-record deterministic validation and scoring | Completed for both candidates under v1 and v2. Test remained sealed. |
| Hour 7-8 | Scenario-group bootstrap, error audit, selection, and verification | Completed; protocol winner withheld from promotion because of three constant-class collapses. |

Verification completed:

- 21 unit tests pass and all Python modules compile.
- Targeted dataset strict audit passes.
- Simulator replay passes 1,860/1,860 cached states with maximum absolute error below 0.00005.
- Both candidate prediction files contain exactly 120 unique validation records.
- Both adapters and the original step-40 reference exist locally.
- No test inference or test scoring was performed.

## Larger corrective-v2 continuation

The follow-on larger-data round was also completed without paid API calls:

- Generated and audited 2,000 new records from 500 corrective scenarios and 600 records from 150 disjoint development scenarios, each with 5% visual examples.
- Built a 2,462-row curriculum from 2,000 corrective examples and 462 clean pilot anchors, with no repetition.
- Passed full simulator replay (10,585 corrective states and 3,180 dev states), shortcut audits, checksum/structure checks, balance checks, and cross-dataset overlap checks.
- Trained seeds 42, 123, and 314 for 200 optimizer steps from the original step-40 adapter.
- Generated and scored all 600 independent-dev answers for every seed under rubrics v1 and v2.
- Ran a 10,000-replicate paired scenario-group bootstrap and the frozen promotion checker.
- Withheld promotion: all three seeds improved strict macro score by more than 0.22, but none passed feasible-control recall, feasible simulator-success, control status-F1, or sufficiency status-F1.
- Kept the sealed pilot test untouched.

## Action-first and schema-repair follow-on — 2026-07-15

- Completed the seed-42 action-first run through its requested 200-step stopping point and evaluated the viable checkpoint on all 600 unchanged development records.
- Built and audited a 200-record schema-compatible repair curriculum after the action-first model exposed valid-ground-truth targets through an incompatible output envelope.
- Completed a 50-step seed-42 schema-repair continuation with checkpoints at steps 25 and 50.
- Confirmed on deterministic free-generation diagnostics that schema validity recovered to 99.2%, while sufficiency and control status macro-F1 remained below the frozen 0.60 gates.
- Did not launch seeds 123 and 314 because the seed-42 intervention was not viable; did not lower gates or evaluate the sealed test.
- Passed all 32 project tests. Detailed evidence is in `schema_repair_v3_1_seed42_closeout.md` and `schema_repair_v3_1_seed42_decision.json`.
