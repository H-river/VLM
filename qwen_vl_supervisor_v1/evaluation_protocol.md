# Qwen-VL supervisor v1 frozen evaluation protocol

Protocol state: frozen before formal server training or any frozen-test
prediction. This document and `configs/evaluation_frozen.yaml` define the
evaluation; they do not authorize or run it. A deterministic pre-server
protocol artifact must seal their evaluation inputs before server training,
and the later final freeze must verify that artifact by its externally recorded
SHA-256. Neither freeze step may open predictions, run a model, or execute the
formal closed-loop comparison.

The research status is **PROVISIONALLY READY FOR QWEN-VL**. Sensor saturation
is formally validated. Width-relative secondary reflection is available for
engineering progress, but it is not a strict preregistered pass. Its prior Q1
accuracy was 68.75%, its width-quartile gap was 31.25 percentage points, its
boundary accuracy was 75%, and no valid width-relative severity-OOD conclusion
exists. All four facts remain mandatory report fields.

## Frozen comparison matrix

| Arm | Supervisor | Inputs | Selection and role |
| --- | --- | --- | --- |
| A | Strongest metrics-only rule/logistic/MLP | Current and goal metrics, legal past history, remaining budget, legal limits; no image | Train on `train`; select once on `dev` by joint exact, diagnosis macro F1, valid-JSON rate, lower complexity, then name. This is the primary deployable baseline. |
| B | Previously validated small-image diagnostic reference | Current 128x128 image and five metrics | Run the fixed saturation and width-relative specialists without evaluator-only family routing. Freeze their two thresholds and deterministic score arbitration on `dev`. Reflection remains only a provisional reference. |
| C | Qwen-VL high-level supervisor | Exactly one current image and the structured input allowlist | Select one checkpoint independently for every declared training seed, using `dev` only. Decode greedily to exactly one three-field JSON object. |
| D | Oracle diagnosis and high-level decision | Evaluator-only targets | Upper bound. It still cannot select continuous actions; the same numerical controller does that. |

Every arm produces the same canonical high-level fields: diagnosis,
measurement policy, and supervisor action. Qwen-VL never emits an actuator
coordinate, position, delta, or continuous action vector.

Arm B runs both fixed binary specialists on every record. Scores are calibrated
probabilities in `[0,1]`. It declares nominal only when both scores are strictly
below their dev-frozen thresholds; otherwise it selects the class with larger
normalized margin `(score-threshold)/max(abs(threshold),1e-12)`, treating a
score equal to threshold as anomalous. The fixed tie order is nominal, sensor
saturation, then secondary reflection. This prevents hidden family labels from
becoming a router.

## Split and selection discipline

- `train` may train A and C. It may not select a checkpoint or threshold.
- `dev` is the only checkpoint-, model-, threshold-, calibration-, and
  arbitration-selection split.
- `frozen_iid` is the setup-disjoint primary scientific test and is opened only
  after the final checkpoint, training config, Qwen checkpoint selection,
  validated A/B baseline selection, prediction runner, closed-loop runner, and
  this evaluation configuration are hashed.
- `frozen_ood` contains valid saturation severity-OOD records only. It cannot
  support a width-relative reflection severity-OOD conclusion.
- All members of a setup, counterfactual pair, episode, or augmented base stay
  in one split. Exact image hashes and setup hashes may not overlap splits.
- Frozen predictions do not participate in checkpoint or threshold selection,
  and no post-test retuning or favorable-seed selection is allowed.

For each training seed, rank complete saved checkpoints by: highest dev joint
exact accuracy, diagnosis macro F1, valid-JSON rate, lowest optimizer step, and
finally lexicographically lowest checkpoint-tree SHA-256. Keep one checkpoint
per declared seed in one final selected-checkpoint bundle. The bundle tree hash
is the `FINAL_CHECKPOINT_SHA256` used by the freeze gate. The dev-selection
artifact records the selected C checkpoint path/hash and all tie-break values
for each seed, plus explicit false flags for frozen/protected selection use.
The separate baseline dev-selection artifact records the selected A candidate,
both B thresholds, and the frozen arbitration contract. Formal reports show
every seed and the mean, population standard deviation, minimum, and maximum;
no single favorable seed replaces the complete result. The formal reducer
reads the exact expected seed list from the frozen evaluation config and
refuses missing, extra, or seedless prediction groups. Development smoke
reduction may omit this enforcement explicitly.

Arm A is selected separately on dev by highest joint exact accuracy, diagnosis
macro F1, valid-JSON rate, lowest model-complexity rank, and candidate name.
Arm B evaluates a complete dev threshold-grid Cartesian product under the exact
arbitration above, ranks by joint exact, diagnosis macro F1, valid-JSON rate,
then the two lower thresholds, and freezes the rank-one thresholds. A separate
baseline dev-selection artifact must recompute both choices from raw evidence,
show complete coverage, and record false frozen/protected-use flags before the
final gate can seal.

The dev selector enumerates the complete config-derived checkpoint schedule,
requires complete safe-state/run evidence and 100% dev coverage for every
candidate, recomputes the current offline reducer from raw dev predictions, and
applies the ranking above independently per seed. Its implementation, artifact
schema, and exact server selection commands are byte-pinned by the protocol
artifact; selection cannot silently substitute a different reducer or partial
checkpoint set.

The finalized server schedule is seeds 2026080101, 2026080102, and 2026080103.
The hashed server config is
`4354f527460e59f085589153e5b071a7e9a29fbd41e6ca4f625a826463f913e5`;
the exact three-seed launch document is independently pinned as
`40dd63a31fd0b7dc9f0d1eb0a0e4b1b4e0f07356d901e5a21112b3a987f19eca`.

## Offline evaluation

Join predictions to the source manifest by `sample_id`. Missing outputs and
outputs that are not one strict canonical JSON object count as invalid and as
incorrect for every supervised target field. The fixed output enums and strict
parser reject prose, markdown, extra keys, confidence, rationales, duplicate
keys, non-finite JSON, and continuous actions.

Report each deterministic A/B/D reference once, and report C for every declared
training seed:

- valid-JSON rate;
- diagnosis balanced accuracy and macro F1;
- diagnosis precision and recall for nominal, sensor saturation, and secondary
  reflection;
- measurement-policy accuracy;
- supervisor-action macro F1;
- joint diagnosis + policy + action exact accuracy;
- confusion matrices for diagnosis, measurement policy, and supervisor action;
- the same applicable metrics by anomaly family, reflection beam-width
  quartile, boundary status, and severity bucket.

Each subgroup includes its support. Q1, Q2, Q3, Q4, boundary, and non-boundary
rows are never hidden because support is small or performance is unfavorable.
Anomaly-family slices use evaluator-only provenance and include the nominal
member of each family-specific counterfactual pair; diagnosis-class slices are
reported separately. Nominal members of reflection counterfactual pairs inherit
the pair's quartile and boundary metadata for paired subgroup reporting.
For anomaly family, reflection width quartile, boundary status, and severity
bucket, the aggregate report includes across-seed metric summaries, support
summaries, per-class diagnosis summaries, and summed confusion matrices. Thus
Q1 and boundary results remain directly visible in both per-seed and aggregate
sections.

Arms A, B, and D each emit exactly one prediction per sample with seed label
`deterministic_reference`; their reducers require exactly that one seed and do
not copy their predictions across C's training seeds. Arm C emits one prediction
per sample for each of the three declared integer seeds; its reducer loads the
exact seed set from `evaluation_frozen.yaml`, reports every seed, and reports the
across-seed aggregate. The exact two-split, four-arm matrix command is frozen in
the YAML.

Target masks define denominators. A field is scored only when its mask is true;
joint exact requires all three masks. The current static anomaly-policy records
have valid `execute`, `reacquire`, or `switch_measurement` targets. They do not
provide temporal `continue` or `stop` supervision, so no per-class continue/stop
score is reported from them. A future record without valid temporal action
supervision must mask `supervisor_action`, removing it from both action and
joint denominators. If a model emits continue/stop instead of a supervised
static action, that output is simply wrong for the static action and joint
metrics; it is not treated as an invented continue/stop label.

The model-free reducer is:

```bash
python -m qwen_vl_supervisor_v1.evaluate_offline \
  --manifest qwen_vl_supervisor_v1/manifests/manifest_frozen_iid.jsonl \
  --predictions "$FORMAL_OUTPUT/frozen_iid/arm_C_predictions_all_training_seeds.jsonl" \
  --evaluation-config qwen_vl_supervisor_v1/configs/evaluation_frozen.yaml \
  --output "$FORMAL_OUTPUT/frozen_iid/offline_arm_C.json"
```

The analogous saturation-only OOD command is frozen in the YAML config. These
reducers consume already captured predictions; they do not authorize their
generation.

## Closed-loop evaluation

The four arms use exact paired cases. For each setup and planner root seed,
deep-copy the same initial simulator state and target into A/B/C/D. Keep setup
groups, target, CEM population and sampling budget, numerical action budget,
maximum step budget, and planner root seed identical. The frozen defaults are
planner seeds 2026081101 and 2026081102; the final list is sealed in the freeze
artifact and may not change after any prediction.

The numerical backbone is immutable:

- three-member learned H1 forward ensemble;
- one-step CEM with population 24, 6 elites, and 3 iterations;
- gain-aware symmetric probing and replanning where applicable;
- replanning after every real observation, never an imagined continuation;
- the frozen visible continuation rule, through at most eight real actuator
  steps;
- strict all-five success at maximum initial-tolerance-normalized distance
  no greater than 1.0.

At the current observation, the supervisor chooses only diagnosis, measurement
policy, and a high-level action. Lower-exposure reacquisition and primary-spot
switching are measurement operations and do not add actuator steps. The frozen
controller chooses each continuous action, the simulator returns a real next
observation, and the frozen continuation rule alone decides whether another
numerical step is permitted. A high-level stop is conservative; no model can
override an exhausted budget or a frozen-rule stop. The current static SFT
cohort does not make continue/stop an offline scientific target.

Report every training-seed × planner-seed cell and aggregate:

- strict all-five success, final normalized distance, executed actuator steps,
  actuator saturation, and hard-constraint violations;
- C-versus-A matched recoveries (A fails, C succeeds) and regressions (A
  succeeds, C fails) on identical paired episodes;
- all metrics by family, reflection width quartile, and boundary status;
- exact C-versus-D decision disagreements;
- diagnosis-correct but control-failed cases; and
- diagnosis-wrong but accidentally recovered cases.

Episode identity for pairing is `(setup_hash, anomaly condition, planner seed)`;
the C training seed is an additional result dimension. A/B/D results are paired
to every C seed rather than used to select a preferred C seed. Record the exact
setup hash, high-level decisions, strict outcomes, final distances, steps,
saturation, and constraint flags for every disagreement or matched change.

No formal closed-loop runner is invented in this preparation task. A concrete
runner that consumes the guarded adapter and the two frozen suite manifests
must be implemented, reviewed, supplied to the freeze utility with its SHA-256,
and recorded with its exact command before the freeze can seal. This is an
intentional authorization boundary, not permission to use an ad-hoc runner.

## Primary research tests

1. On frozen setup-disjoint IID, C passes the primary offline comparison only
   if its across-declared-seed mean diagnosis balanced accuracy is strictly
   greater than A's; every individual seed remains visible.
2. Closed-loop efficacy requires either `strict_success(C) -
   strict_success(A) >= 0.05` or at least five C-versus-A matched recoveries.
   It must also keep the actuator-saturation-rate increase at or below 0.05 and
   add zero hard-constraint violations.
3. Gaps from C to B and D are descriptive for every primary offline and
   closed-loop metric; this test has no pass threshold.
4. Reflection Q1/Q2/Q3/Q4 and boundary/non-boundary performance is always
   shown. This test is descriptive and can never retroactively relabel
   reflection as a strict pass.

## Deterministic freeze and formal-run gate

`scripts/freeze_evaluation.py` has no model, controller, or prediction-reading
code. During `prepare`, it opens every manifest-listed current image as bytes,
computes SHA-256, and refuses any mismatch with
`assets.current_image_sha256`. It records every actual image path/hash in the
later verification registry; manifest declarations alone are never treated as
proof of image identity. It also verifies zero cross-split setup/image-hash
overlap and pair/episode containment, verifies generator/controller provenance,
and hashes every frozen manifest, exported train/dev dataset, generator
runtime, controller, baseline, suite, training config, evaluator/runtime, and
command artifact. The output contains no timestamp, so identical inputs produce
byte-identical JSON.

Before server training, create the protocol commitment exactly once and record
the printed SHA-256 outside the artifact:

```bash
python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py prepare \
  --config qwen_vl_supervisor_v1/configs/evaluation_frozen.yaml \
  --output qwen_vl_supervisor_v1/artifacts/evaluation_protocol_freeze.json \
  --authorize-pre-server-protocol-freeze
```

The transfer, validation, training, interruption/resume, dev-selection, and
future formal-gate sequence is recorded in `reproduction_commands.md`; that
command document is itself byte-pinned by this preparation artifact.

`prepare` does not accept a prediction path and cannot run inference. The later
`freeze` command first verifies the exact preparation-artifact bytes against
`EVALUATION_PROTOCOL_FREEZE_SHA256`, then re-hashes every file and actual image
sealed by that artifact. It consumes the prepared snapshots instead of creating
a new protocol snapshot from whatever files happen to exist after training.
It also runs the strict dev-selection validator, requires the exact configured
seed order, complete checkpoint/dev coverage, false protected/frozen-use flags,
and the pinned selector/schema identities. Every rank-one selected checkpoint
path must resolve beneath the supplied final-checkpoint root and its live tree
hash must equal the selection record. The final artifact records those relative
paths/tree hashes and the whole-root tree hash; extra retained nonselected
checkpoints are permitted but cannot change afterward.

The final gate separately requires a hashed A/B baseline dev-selection artifact.
Its pinned validator re-reads the train/dev manifests, recomputes all arm-A
candidate reports from raw predictions, recomputes the complete arm-B threshold
grid from raw specialist scores under the exact arbitration above, and verifies
both rank-one selections plus false frozen/protected-use flags. No A candidate
or B threshold is fabricated during protocol preparation; absence of this
future dev evidence makes `freeze` refuse to seal.

Compute hashes with the utility's exact file/tree algorithm:

```bash
BASELINE_DEV_SELECTION_ARTIFACT=qwen_vl_supervisor_v1/artifacts/dev_baseline_selection/baseline_dev_selection.json
FINAL_CHECKPOINT_SHA256="$(python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py hash --path "$FINAL_CHECKPOINT" --digest-only)"
TRAINING_CONFIG_SHA256="$(python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py hash --path qwen_vl_supervisor_v1/configs/training_server.yaml --digest-only)"
DEV_SELECTION_ARTIFACT_SHA256="$(python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py hash --path "$DEV_SELECTION_ARTIFACT" --digest-only)"
BASELINE_DEV_SELECTION_ARTIFACT_SHA256="$(python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py hash --path "$BASELINE_DEV_SELECTION_ARTIFACT" --digest-only)"
PREDICTION_RUNNER_SHA256="$(python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py hash --path "$PREDICTION_RUNNER" --digest-only)"
CLOSED_LOOP_RUNNER_SHA256="$(python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py hash --path "$CLOSED_LOOP_RUNNER" --digest-only)"
```

After full server training and dev-only selection, seal the artifact:

```bash
python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py freeze \
  --config qwen_vl_supervisor_v1/configs/evaluation_frozen.yaml \
  --output qwen_vl_supervisor_v1/artifacts/evaluation_freeze.json \
  --preparation-artifact qwen_vl_supervisor_v1/artifacts/evaluation_protocol_freeze.json \
  --preparation-artifact-sha256 "$EVALUATION_PROTOCOL_FREEZE_SHA256" \
  --final-checkpoint "$FINAL_CHECKPOINT" \
  --final-checkpoint-sha256 "$FINAL_CHECKPOINT_SHA256" \
  --training-config qwen_vl_supervisor_v1/configs/training_server.yaml \
  --training-config-sha256 "$TRAINING_CONFIG_SHA256" \
  --dev-selection-artifact "$DEV_SELECTION_ARTIFACT" \
  --dev-selection-artifact-sha256 "$DEV_SELECTION_ARTIFACT_SHA256" \
  --baseline-dev-selection-artifact "$BASELINE_DEV_SELECTION_ARTIFACT" \
  --baseline-dev-selection-artifact-sha256 "$BASELINE_DEV_SELECTION_ARTIFACT_SHA256" \
  --prediction-runner "$PREDICTION_RUNNER" \
  --prediction-runner-sha256 "$PREDICTION_RUNNER_SHA256" \
  --prediction-command "$PREDICTION_COMMAND" \
  --closed-loop-runner "$CLOSED_LOOP_RUNNER" \
  --closed-loop-runner-sha256 "$CLOSED_LOOP_RUNNER_SHA256" \
  --closed-loop-command "$CLOSED_LOOP_COMMAND" \
  --authorize-formal-evaluation-freeze
```

Record the printed artifact hash. Immediately before any future frozen
prediction or closed-loop execution, run this preflight in the same shell and
continue only on exit status zero:

```bash
python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py verify \
  --freeze-artifact qwen_vl_supervisor_v1/artifacts/evaluation_freeze.json \
  --freeze-artifact-sha256 "$EVALUATION_FREEZE_SHA256" \
  --final-checkpoint "$FINAL_CHECKPOINT" \
  --final-checkpoint-sha256 "$FINAL_CHECKPOINT_SHA256" \
  --training-config qwen_vl_supervisor_v1/configs/training_server.yaml \
  --training-config-sha256 "$TRAINING_CONFIG_SHA256" \
  --formal-evaluation \
&& eval "$PREDICTION_COMMAND" \
&& eval "$CLOSED_LOOP_COMMAND"
```

The command strings themselves are frozen inside both the pre-server and final
artifacts. The explicit
`eval` is only for those reviewed, hashed, artifact-recorded commands; it is not
used during this preparation task. Omitting the pre-server artifact or its
external hash, the explicit authorization flag, checkpoint/config hashes, dev
checkpoint selection, validated A/B baseline selection, either runner, or
either concrete command makes `freeze` fail. Omitting `--formal-evaluation` or
changing any frozen manifest, actual
image, source/runtime artifact, config, runner, or checkpoint byte makes
`verify` fail.
