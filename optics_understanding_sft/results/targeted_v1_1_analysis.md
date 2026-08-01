# Targeted v1.1 QLoRA ablations

## Outcome

The fresh 40-step targeted run is the frozen-protocol winner at **0.538 v2 macro**, but it is **not promoted**. Its 0.0195 advantage over the original step-40 reference has a paired scenario-group bootstrap 95% interval of **-0.0774 to 0.1166** and wins 65.3% of 10,000 replicates. The validation set therefore does not establish a reliable improvement.

More importantly, the apparent aggregate gain coexists with three constant-class shortcuts:

- constrained intervention: 12/12 feasible cases called feasible, 0/12 infeasible cases called infeasible;
- information sufficiency: all 12 cases called insufficient;
- diagnosis: all 18 cases called unique.

This is evidence that the targeted repetition weights changed output priors more than they taught the model to distinguish physical cases. The sealed test remains untouched.

## Controlled comparison

| Model | v2 macro | v1 macro | JSON | Schema | Trainer val loss |
|---|---:|---:|---:|---:|---:|
| Base | 0.392 | 0.416 | 94.2% | 58.3% | n/a |
| Original step 20 | 0.510 | 0.518 | 99.2% | 90.8% | 0.3123 |
| Original step 40 | 0.518 | 0.514 | 98.3% | **93.3%** | 0.2703 |
| Targeted continue 20 | 0.528 | 0.512 | **100%** | 81.7% | 0.2808 |
| Targeted fresh 40 | **0.538** | **0.524** | **100%** | 90.0% | **0.2664** |

Teacher-forced loss and task-level behavior again disagree: the fresh run has the best loss and macro, yet fails three status decisions by predicting one label for every record.

## v2 per-task scores

| Task | Original step 40 | Continue 20 | Fresh 40 |
|---|---:|---:|---:|
| Setup interpretation | **0.611** | 0.556 | **0.611** |
| Information sufficiency | 0.350 | **0.533** | 0.450 |
| Causal effects | 0.789 | **0.856** | **0.856** |
| Forward prediction | **0.444** | **0.444** | 0.438 |
| Diagnosis | 0.466 | 0.630 | **0.659** |
| Constrained intervention | **0.467** | 0.177 | 0.250 |
| Counterfactual reasoning | **0.500** | **0.500** | **0.500** |

The diagnosis score illustrates why a composite task score cannot replace the confusion matrix: plausible-cause overlap and the validation class mix give partial credit even though the fresh model always chooses `unique`.

## Physical and formatting diagnostics

- Both targeted runs flip the earlier control shortcut from “always infeasible” to “always feasible.” The continued run succeeds in the simulator on 1/12 feasible controls; the fresh run succeeds on 3/12. Neither distinguishes feasibility.
- Fresh setup interpretation gets all 12 component orders semantically correct, but only 6/12 focal lengths have the correct unit scale; 5/12 remain approximately 10x too large.
- Fresh forward outputs are internally consistent (`after - current` agrees with `change`), but the physical tolerance pass rate remains low: only 5/24 after-state centroids are within 2 px.
- Both targeted runs produce extractable JSON for 120/120 records. The continued run loses schema validity mainly by emitting sparse, off-grid control plans; fresh reduces but does not eliminate this issue.
- The fresh run's row-mean text and visual scores are 0.521 and 0.524 under v2. Only 12 visual records are present, so this is not a visual-generalization conclusion.

## Selection and next experiment

The protocol winner is saved at `/home/jiamo/VLM_runs/qwen25vl_3b_qlora_targeted_fresh40_v1`, but its selection file marks it `not_promoted_due_task_collapse`. For corrective training, retain the original step-40 adapter as the reference/start point because it has the strongest schema validity and does not import the targeted runs' simultaneous sufficiency and diagnosis collapse. It still has an unresolved all-infeasible control shortcut.

The next dataset revision should replace repetition-based class weighting with explicitly paired discrimination examples:

1. Put matched feasible/infeasible controls with similar observed centroid error but different correct actions in the same training unit.
2. Sample sufficiency and diagnosis statuses uniformly by batch rather than repeating one status globally.
3. Add a separate feasibility/status decision target before the numerical control plan, while keeping the final SFT answer concise and simulator-verifiable.
4. Require all four action keys in every feasible training target and include hard negative off-grid plans.
5. Freeze these changes before another validation pass; do not tune further against these same 120 labels.

No sealed test inference should be run until a new candidate avoids constant-class behavior on a separately generated development split.
