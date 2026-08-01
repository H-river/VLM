# Control rebuild v4

This folder contains the new specialist-control work. The Qwen orchestrator,
its registry, all v2 datasets, and all v2/v3 checkpoints remain unchanged.

## Numerical contracts

The forward model receives one optical setup, one current beam state, and each
of the 81 permitted actuator actions.

- Setup: 12 physical numbers, such as wavelength, focal length, distances,
  offsets, and camera pixel size.
- Current state: 5 numbers: horizontal centroid, vertical centroid, horizontal
  width, vertical width, and peak intensity.
- Action: 4 numbers: two lens-position changes and two camera-position changes.
- Forward output: 5 normalized state changes per action. One output unit means
  one allowed error for that state field.

The inverse model receives the setup, current state, desired state, and the 81
states predicted by the forward model.

- Action output: one selected index from 0 through 80 and its 4-number action.
- Status output: one of `unique`, `ambiguous`, or
  `infeasible_within_limits`.
- `unique` means exactly one action reaches the target within the declared
  tolerances.
- `ambiguous` means more than one action reaches it.
- `infeasible_within_limits` means none of the 81 actions reaches it.

The image path first converts a current beam image and desired beam image into
the same 5-number state. It then runs a measurement calibrator, converts
coordinate frames analytically, predicts all actions, and uses a sensor-frame
scorer to select the final action.

`VisualInversePipelineV4.predict_from_images` is the deployment entry point for
`setup + current image + desired image + calibration`. It streams one image at
a time. On the clean validation fixture, its generic file path and the dataset
measurement path produce exactly identical five-number measurements.

The learned measurement correction was trained only for gamma values from 0.78
through 1.0. Direct images with gamma outside the guarded range 0.75 through
1.05 use calibrated analytic intensity moments instead. On the 150 clean Qwen
validation images at gamma 0.5, this fallback gives 115/150 strict five-field
success; forcing the v4 calibrator gives only 40/150. The guard is therefore a
validation-selected domain check, not an arbitrary runtime heuristic.

The synthetic all-condition measurement and visual-inverse tables provide the
known deterministic synthetic transform to the measurement model. They are
reported with that qualifier. The separate 150-image direct metric supplies
only the four calibration fields allowed by the frozen Qwen deployment
contract.

`OrchestratedSpecialistRuntimeV4` is the candidate system-level execution
overlay. Qwen still emits the frozen `qwen_orchestration_decision_v1` object,
and the frozen registry still validates its status, task type, argument groups,
and image-role bindings. The overlay then executes:

- v4 measurement for `measure_beam_profile_v1`;
- the frozen v1 direction model for numerical direction requests;
- v4 measurement followed by the frozen direction model for image direction
  requests;
- the v4 forward model for numerical or image forward requests;
- the v4 forward and inverse models for numerical inverse requests;
- the modular v4 measurement, forward, inverse, and sensor-frame scorer for
  image inverse requests.

Clarification and unsupported decisions do not load or execute a specialist.
All ready calls retain the original route names, and all v4 forward and inverse
execution remains restricted to the registered 81-action grid.

`candidate_overlay_manifest.json` pins the frozen registry, decision schema,
seven route-to-backend mappings, and every specialist artifact by absolute
path, size, and SHA-256 digest. `OrchestratedSpecialistRuntimeV4.from_manifest`
rejects missing or modified artifacts before loading a model.

## Training data

The server-scale numerical dataset is configured for 3,000 training setups and
450 validation setups. Every setup has all 81 simulator actions, for 243,000
training transitions and 36,450 validation transitions.

The category assignment probabilities are:

- 20% expanded in-distribution setups.
- 40% boundary setups outside or near the edges of previous training ranges.
- 40% high-nonlinearity setups with difficult combinations of focal length,
  beam waist, propagation distance, aperture, and offsets.

The seeded hash assignment gives exact training counts of 646 expanded
in-distribution, 1,189 boundary, and 1,165 high-nonlinearity setups. Validation
contains 78, 183, and 189 setups respectively. Reports use these exact
numerators rather than treating the assignment probabilities as exact counts.

No previous held-out test group is used to build or select these checkpoints.

### Twelve-hour laptop validation

The current laptop quickcheck is a separate validation-only run. It uses 2,000
unique training setups and 300 unique difficult-validation setups, or 162,000
training transitions and 24,300 validation transitions. Its seeded category
counts are:

- Training: 422 expanded in-distribution, 807 boundary, and 771
  high-nonlinearity setups.
- Validation: 52 expanded in-distribution, 128 boundary, and 120
  high-nonlinearity setups.

Each new numerical setup is used once. The run does not oversample or repeat
the difficult examples. Numerical inverse training therefore combines the
60,000 frozen v2 requests with 12,000 unique requests derived from the 2,000
new grids.

Only the numerical dataset size changes in this trial. The stored beam images
remain 512 by 512 pixels in the original 1024 by 1024 sensor coordinate
system. The seven image conditions, 2,500 visual-training groups, 52,500
visual-training pairs, Qwen checkpoint, model structures, 81-action grid,
training epochs, evaluation counts, tolerances, and all 26 validation bars are
unchanged.

The quickcheck writes to
`/home/jiamo/VLM_data/control_rebuild_v4_quickcheck` and
`/home/jiamo/VLM_runs/control_rebuild_v4_quickcheck_12h`, so it cannot
overwrite the server-scale dataset or the previous candidate run. Passing all
26 bars recommends full unique-data server training; it does not authorize or
open held-out test data.

The frozen raw measurement predictions for all 70,000 training-condition
records and 8,400 validation-condition records are checksum-verified and copied
from the earlier identical measurement pass. The v4 calibrator is still
applied after loading those raw predictions. This avoids repeating CNN
inference without changing any image, condition, measurement, or training
pair.

The generated report exposes the difficult 450-group selection validation
separately from both the old IID validation and the final held-out tests. It
reports forward prediction, forward-only action retrieval, inverse target
success, and inverse status accuracy before and after paired measurement-error
augmentation. This distribution is explicitly labeled as checkpoint-selection
evidence rather than held-out evidence.

`compare_v3_v4_selection_validation.py` also runs the frozen v3 and candidate
v4 forward/inverse pipelines on those identical 450 setups and their identical
derived inverse requests. It reports absolute v4-minus-v3 changes overall and
separately for expanded in-distribution, boundary, and high-nonlinearity
setups. This is the apples-to-apples evidence used to decide whether v4 is an
actual specialist improvement.

The same comparison stores concrete side-by-side diagnostics. For forward
prediction and inverse control, overall and within each source category, it
records the first real case where v4 fixes a v3 failure and the first case
where v4 regresses from a v3 success. Each record contains the group or request
identifier, action, target, both model outputs, normalized errors or matching
actions, both metric outcomes, and an explicit name and definition for the
metric that separates the improvement or regression.

Inverse training combines 60,000 frozen v2 training requests with six derived
v4 requests per new grid: four reachable requests and two guaranteed
infeasible requests. Half of the training requests receive independent
current-state and desired-state errors sampled from 70,000 calibrated
training-image measurements. Current and desired errors are sampled from the
same visual condition, matching paired-image evaluation. The physical labels
are never perturbed.

## Forward training dataflow

1. Build a 47-number feature vector for each setup, current state, and action.
2. Subtract the zero-action feature vector from every action feature.
3. Fit a linear ridge baseline to normalized simulator changes.
4. Train a 983,813-parameter residual network on the remaining error.
5. Subtract the network's zero-action output from every network output.
6. Optimize mean regression error, worst-field error, and action ranking.
7. At each epoch, evaluate both frozen IID validation and new expanded
   validation, including the exact six-request-per-grid retrieval calculation
   used by the final v3-versus-v4 comparison.
8. Select both the epoch and the neural-residual multiplier by the number of
   registered forward gates passed, then the worst gate margin, mean gate
   margin, and the original composite score.

Both the linear and neural paths are zero-anchored, so the no-movement action
returns exactly five zero changes.

## Inverse training dataflow

1. Run the selected forward checkpoint on clean current states.
2. Sample one empirical measurement error per training group and rerun the
   forward checkpoint on that observed current state.
3. Independently perturb half of the desired-state observations.
4. Build 23 candidate features for each of 81 actions and 12 request-level
   status features.
5. Train a 385,284-parameter ranker using reachable-action ranking loss,
   three-class status loss, and correction regularization.
6. Evaluate clean and measurement-perturbed requests on both IID and expanded
   validation.
7. Select one checkpoint and one correction multiplier by the number of
   registered inverse gates passed, then the worst gate margin, mean gate
   margin, and the original composite score.

## Visual-inverse training dataflow

1. Load the checksum-verified frozen measurement predictions and apply the v4
   measurement calibrator.
2. Convert measured sensor coordinates into the forward model's coordinate
   frame analytically.
3. Predict all 81 candidate states with the selected forward checkpoint.
4. Convert those candidates back into the camera-sensor coordinate frame.
5. Train the sensor-frame ranker on 52,500 measured current/desired pairs.
6. At every epoch, select the correction multiplier primarily by physical
   target success.
7. Select the final epoch by whether it passes the registered 44% physical
   target-success floor, then by its exact physical-success margin and only
   afterward by minimum movement, status quality, and correction size.

## Metrics

- `strict_all_five_success`: fraction of transitions where all five predicted
  changes are within one declared tolerance of simulator truth.
- `mae_in_tolerance_units`: mean absolute error after dividing each field by
  its tolerance.
- `target_success_feasible`: fraction of physically reachable requests where
  the selected action is one of the simulator-confirmed matching actions.
- `minimum_movement_exact_feasible`: fraction where the selected action is the
  least-moving simulator-confirmed match.
- `status_accuracy`: fraction with the correct three-class reachability status.
- `status_macro_f1`: mean F1 score across the three statuses, giving each class
  equal weight.
- `physical_target_success`: fraction where executing the selected action
  reaches the ground-truth target.
- `reached_by_step`: cumulative closed-loop physical success after one, two,
  or three re-measure-and-replan steps.

## Resource policy

The twelve-hour generator uses one worker process, at most 0.65 CPU core, a
2.5 GB soft memory limit, a 3 GB hard memory limit, and at most 256 MB swap.
Post-generation training uses at most one CPU core, a 3 GB soft memory limit,
a 4 GB hard memory limit, and at most 256 MB swap. Image evaluation streams
bounded batches and never stores decoded image tensors for the whole dataset.
Long jobs are pinned to CPU 8 on the cooler processor chiplet.

The quickcheck units `vlm-v4-quickdata-12h.service` and
`vlm-v4-quickpost-12h.path` are linked into the user `default.target`.
If the laptop reboots, dataset generation resumes from atomic per-group shards
after the next login. The manifest path then restarts the stage-marked training
and validation pipeline. The generator is skipped once dataset finalization has
passed, and the post-generation trigger is skipped once validation completes.
Both services retain the same CPU, memory, swap, processor-affinity, and
low-priority limits after a reboot. The held-out evaluation is intentionally
not enabled as a boot service.

The resource-capped post-generation pipeline runs:

1. Frozen Qwen baseline integrity verification.
2. Exact-key and checksum verification of the reusable frozen train/validation
   measurement predictions.
3. Dataset verification and stable checksums.
4. Full forward training and residual calibration.
5. Full inverse training with measurement-error augmentation.
6. Frozen-v3 versus candidate-v4 comparison on identical difficult validation.
7. Integrated sensor-frame scorer training.
8. Hash-pinned candidate overlay manifest creation.
9. All-condition controlled validation.
10. Fifty-request, three-step closed-loop validation.
11. One real canonical request through each of the seven Qwen-to-specialist
   routes.
12. All 150 direct Qwen measurement-validation images through the public
   image-plus-four-calibration-field deployment contract.
13. All 1,600 saved checkpoint-1000 Qwen decisions through the v4 overlay.
    The combined report includes 1,050 ready requests and uses the simulator
    only afterward to calculate 150 cached direction/forward ground truths and
    replay the 300 predicted plus 300 correctly routed inverse actions. The
    report separates physical accuracy from agreement with execution under the
    target Qwen decision.
14. Pre-registered validation-gate assessment, quantitative report generation,
    and a completion audit. In the twelve-hour run this records either a
    recommendation for server-scale training or the exact failed bars that
    require another specialist revision.

Held-out evaluation is deliberately not part of that automatic chain. It runs
once only after the validation artifacts are confirmed.

`assess_validation.py` applies pre-registered gates before any held-out file is
opened. The gates require positive v4-minus-v3 changes on the identical
difficult distribution, no regression in boundary or high-nonlinearity target
success, bounded old-IID regression, measurement and visual-control floors,
paired-error robustness, three-step closed-loop success, all seven execution
routes, and at most three percentage points of Qwen-induced physical loss. A
failed gate writes a diagnostic decision and blocks `run_final_evaluation.sh`;
the candidate must be revised and validated again.

The guarded final command is `run_final_evaluation.sh`. It refuses to run
before validation completes and refuses to overwrite an existing final
controlled evaluation. Both shell pipelines write a completion marker after
each stage. An interrupted controlled evaluation resumes from completed splits,
and an interrupted closed-loop evaluation resumes from completed requests.
Before the final completion marker is written, `audit_completion.py` verifies
the frozen Qwen hashes, dataset checksums and exact counts, training-only data
provenance, full checkpoint lineage, all validation and held-out outputs,
closed-loop request counts, report evidence scope, finite rates, and every
restart-safe stage marker. The candidate manifest must contain the exact six
specialist artifact pins and exact seven route/backend mappings, including
valid registry and schema pins. The final example report must contain a
non-null, metric-consistent success and failure from a declared held-out split
for every rebuilt task. Dataset checksum verification also requires a unique
entry for every generated file, no unlisted or extra targets, and an exact
checksum-manifest digest match. The combined-system details file must contain
one unique record for every frozen Qwen example and must reconcile its ready
route counts and physical direction/forward outcomes with the summary metrics.
Held-out authorization additionally requires the unchanged registered
threshold table and all 26 uniquely labeled validation-gate records to pass.

`FINAL_EXAMPLES.md` and `final_examples.json` contain one real held-out
success and failure where available for forward prediction, numerical inverse
control, image measurement, and visual inverse control. Each record includes
the exact input identifiers, selected action or numerical prediction,
ground-truth comparison, and the metric-specific success flag.
