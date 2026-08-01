# Optics Understanding SFT Pilot

This package builds a simulator-grounded pilot dataset for testing several small forms of optics understanding instead of treating beam alignment as one fixed regression problem.

The v1 physics scope is deliberately narrow: one Gaussian source, one thin lens, and one camera. Ground truth is synthetic and is valid only with respect to `optical_sim`; it is not evidence of laboratory accuracy.

## Current promoted system (v11.3 confirmed)

The current best system is a hybrid rather than an unconstrained numerical regressor. The unchanged seed-49 QLoRA checkpoint chooses registered tools, constructs their visible inputs, and interprets validated results; deterministic simulator and image-analysis tools perform exact unit conversion, numerical thresholding, candidate replay, exhaustive action selection, and pixel measurement.

The unified development gate passes with:

- seven-task routed-system equal-weight macro score: 1.000;
- direct registered-tool end-to-end exact accuracy: 1.000 on both the 30-source development panel and a separate 90-source confirmation panel;
- confirmation physical-group exact accuracy: 74/74, with a two-sided 95% Wilson lower bound of 0.951;
- clean visual production macro-F1: 0.927;
- clean state/pair joint exact accuracy: 0.930/0.913;
- clean tool-orchestration end-to-end exact accuracy: 0.990;
- ordinary-noise state macro-F1/joint exact accuracy: 0.957/0.868;
- average and worst-condition visual macro-F1 under clean, synthetic noise, blur, dim-plus-noise, and saturation: 0.894/0.876;
- perturbed-image tool-orchestration end-to-end exact accuracy: 1.000.

The production mapping interface exposes only the two image roles accepted by the registered pair tool. Raw source-map and raw end-to-end exactness are 1.000 on both the 30-source noise/blur/dim-noise probe and the separate 10-source saturation probe. The older distractor-bearing stress probe remains reported: it scored 0.900 raw mapping before deterministic validation.

These scores do not mean that the language model natively predicts precise optics values. Exact numerical outputs remain tool-computed, and all evidence is simulator-derived. The visual diagnosis and robustness history are in `reports/quantitative_performance_diagnosis_v10_21.md`; the confirmed direct-reasoning extension is in `reports/quantitative_performance_diagnosis_v11_2.md`; and the noise-state improvement is in `reports/quantitative_performance_diagnosis_v11_3.md`. The machine-readable promotion decision is in `results/unified_system_v11_3_confirmed_seed49_gate/summary.json`.

## Build

From `/home/jiamo/VLM`:

```bash
python -m optics_understanding_sft.build_dataset \
  --config optics_understanding_sft/configs/pilot_v1.yaml \
  --output-dir optics_understanding_sft/data/pilot_v1
```

Use `--smoke` to build the tracked 28-record fixture instead. The full build creates 300 scenario groups and 1,200 question records with an exact 70/10/20 train/validation/test split and 10% visual records.

## Audit

```bash
python -m optics_understanding_sft.audit_dataset \
  --dataset-dir optics_understanding_sft/data/pilot_v1 \
  --replay --replay-workers 4
```

## Exports

- `canonical/` contains the dataset-native records.
- `exports/messages/` contains API-neutral chat messages.
- `exports/qwen/` contains prebuilt prompt/completion rows consumed by the repository's Qwen trainer with `dataset_format: prebuilt_chat`.
- `private/test_labels.jsonl` and `master/cases.jsonl` must not be supplied to the model.

No paid API is used in generation, wording, labelling, or auditing.

## Local QLoRA smoke

The repository's `optics_qlora` Conda environment contains the required CUDA training stack:

```bash
CUDA_VISIBLE_DEVICES=0 /home/jiamo/miniconda3/envs/optics_qlora/bin/python \
  optics_sft/scripts/train_qwen25vl_qlora.py \
  --config optics_understanding_sft/configs/qwen25vl_3b_qlora_pilot_v1.yaml \
  --smoke-test --smoke-max-samples 2
```

The smoke selector deliberately chooses one text-only and one visual row. The base shell Python is not the QLoRA environment and does not contain PyTorch or Transformers.

## Base-model baseline and evaluation

Run deterministic inference on the validation split without an adapter:

```bash
CUDA_VISIBLE_DEVICES=0 /home/jiamo/miniconda3/envs/optics_qlora/bin/python \
  -m optics_understanding_sft.run_inference \
  --config optics_understanding_sft/configs/qwen25vl_3b_qlora_pilot_v1.yaml \
  --input-jsonl optics_understanding_sft/data/pilot_v1/canonical/val.jsonl \
  --image-root optics_understanding_sft/data/pilot_v1 \
  --output-jsonl optics_understanding_sft/results/base_qwen25vl3b_val/predictions.jsonl \
  --resume
```

`--smoke-mixed` limits a run to one text and one visual record. Each prediction stores parse status, token counts, latency, peak allocated CUDA memory, and a prompt hash. `--resume` safely skips completed IDs.

Score all seven task families:

```bash
python -m optics_understanding_sft.evaluate \
  --records-jsonl optics_understanding_sft/data/pilot_v1/canonical/val.jsonl \
  --predictions-jsonl optics_understanding_sft/results/base_qwen25vl3b_val/predictions.jsonl \
  --master-jsonl optics_understanding_sft/data/pilot_v1/master/cases.jsonl \
  --output-dir optics_understanding_sft/results/base_qwen25vl3b_val
```

Pass `--rubric-version v2` for current checkpoint selection. Omitting it deliberately reproduces the legacy v1 pilot score. The exact differences are documented in `METRICS.md`.

The evaluator writes `details.jsonl`, `summary.json`, and `report.md`. It reports JSON/schema validity, equal-weight macro task score, task metrics, modality slices, latency/tokens, and status macro-F1. Numerical tasks use sensor-relevant tolerances; constrained interventions are verified against the simulator-cached exhaustive action grid rather than by target-string equality alone.

## Forty-step QLoRA trial (legacy v1 selection)

The tracked `qwen25vl_3b_qlora_trial40_v1.yaml` configuration runs a seed-42 trial with checkpoints and trainer evaluation at steps 20 and 40. Deterministic generative evaluation selected checkpoint 20 with validation macro score 0.518, compared with 0.514 at step 40 and 0.416 for the base model. See `results/qwen25vl3b_trial40_analysis.md` and `results/qwen25vl3b_trial40_selection.json` for the frozen comparison and adapter path.

The subsequent rubric audit produced v2 scores of 0.392 for the base model, 0.510 for step 20, and 0.518 for step 40. This does not rewrite the historical v1 decision; it establishes v2 as the frozen primary rubric for later ablations.

## Targeted v1.1 training-only augmentation

`configs/targeted_v1_1.yaml` generates 100 new seed-314 scenarios (400 records) with disjoint IDs and provenance. The data is training-only: it does not replace or modify the 120 pilot validation records or sealed test set. Its setup prompts expose actuator adjustability, and its balanced status examples target the setup, sufficiency, diagnosis, and constrained-control failures found in the audit.

Build the deterministic curriculum from the original 840 training records and the fresh augmentation:

```bash
python -m optics_understanding_sft.build_curriculum \
  --base-jsonl optics_understanding_sft/data/pilot_v1/exports/qwen/train.jsonl \
  --base-image-root optics_understanding_sft/data/pilot_v1 \
  --augmentation-jsonl optics_understanding_sft/data/targeted_v1_1/exports/qwen/train.jsonl \
  --augmentation-image-root optics_understanding_sft/data/targeted_v1_1 \
  --output-jsonl optics_understanding_sft/data/targeted_v1_1/exports/qwen/train_curriculum.jsonl \
  --seed 202
```

`compare_runs.py` performs paired bootstrap comparisons by resampling the 30 validation scenario groups, preserving the four related questions within each physical scenario.

Two frozen ablations were completed. The fresh 40-step run has the highest v2 point macro (0.538), but it is not promoted because it collapses to one status for every control, sufficiency, and diagnosis record; its bootstrap improvement over the original step-40 reference is also inconclusive. See `results/targeted_v1_1_analysis.md` and `results/targeted_v1_1_selection.json`. The sealed test remains unevaluated.

## Corrective v2 larger-data round

The corrective round replaces repetition-based status weighting with exact balance and matched opposite-label groups. It creates 2,000 new training records from 500 scenarios and a separate 600-record development set from 150 scenarios; both use 5% visual records.

```bash
python -m optics_understanding_sft.build_dataset \
  --config optics_understanding_sft/configs/corrective_v2.yaml \
  --output-dir optics_understanding_sft/data/corrective_v2

python -m optics_understanding_sft.build_dataset \
  --config optics_understanding_sft/configs/dev_v2.yaml \
  --output-dir optics_understanding_sft/data/dev_v2
```

The frozen protocol, promotion gates, three seeds, and no-promotion rule are documented in `results/corrective_v2_protocol.md`. The pilot test remains sealed during this round.

The completed curriculum contains 2,462 unique rows: 2,000 fresh corrective records plus 462 clean pilot anchors, with no repeated examples. Three 200-step continuation seeds were evaluated on every record in the independent 600-record dev set. Their strict v2 macro scores were 0.689, 0.691, and 0.694, compared with 0.468 for the original step-40 reference; paired scenario-group bootstrap intervals exclude zero for all three gains.

No seed is promoted. All three fail the frozen feasible-control recall, feasible simulator-success, control status-F1, and information-sufficiency status-F1 gates, despite passing schema, diagnosis, infeasible-control recall, and anchor-regression gates. The original step-40 adapter remains the retained reference and the sealed test remains unevaluated. See `results/corrective_v2_analysis.md` and `results/corrective_v2_selection.json`.

## Hard-pairs v4 decision repair

The v4 round builds 960 fresh records as 480 same-setup minimal pairs. It removes list-order leakage, uses exact 1:1 statuses, reserves 90 physical scenarios as an untouched hard-pair holdout, and caps seed-42 continuation at 200 total optimizer steps. The frozen design and stopping gates are in `results/hard_pairs_v4_protocol.md`.

```bash
PYTHONPATH=. /home/jiamo/miniconda3/envs/optical_sim/bin/python \
  -m optics_understanding_sft.build_hard_pairs_v4 \
  --config optics_understanding_sft/configs/hard_pairs_v4.yaml \
  --output-dir optics_understanding_sft/data/hard_pairs_v4 \
  --workers 4

PYTHONPATH=. /home/jiamo/miniconda3/envs/optical_sim/bin/python \
  -m optics_understanding_sft.audit_hard_pairs_v4 \
  --dataset-dir optics_understanding_sft/data/hard_pairs_v4 \
  --reference-dir optics_understanding_sft/data/action_first_v3 \
  --reference-dir optics_understanding_sft/data/dev_v2 \
  --output-json optics_understanding_sft/data/hard_pairs_v4/audit_report.json \
  --replay-workers 4

PYTHONPATH=. python -m optics_understanding_sft.build_hard_pair_curricula_v4 \
  --hard-jsonl optics_understanding_sft/data/hard_pairs_v4/exports/qwen/train.jsonl \
  --hard-canonical-jsonl optics_understanding_sft/data/hard_pairs_v4/canonical/train.jsonl \
  --anchor-jsonl optics_understanding_sft/data/corrective_v2/exports/qwen/train_curriculum.jsonl \
  --warmup-output-jsonl optics_understanding_sft/data/hard_pairs_v4/exports/qwen/train_decision_warmup_v4.jsonl \
  --mixed-output-jsonl optics_understanding_sft/data/hard_pairs_v4/exports/qwen/train_mixed_preservation_v4.jsonl \
  --diagnostic-output-jsonl optics_understanding_sft/data/hard_pairs_v4/canonical/diagnostic.jsonl \
  --holdout-output-jsonl optics_understanding_sft/data/hard_pairs_v4/canonical/holdout.jsonl
```

Stage 1 uses decision-token weighting only on the four exact status values. Stage 2 returns to ordinary completion loss while mixing fresh hard pairs with all five preservation task families. Run seed 42 first; later seeds and the sealed test remain conditional on the documented gates.

The seed-42 run stopped at the predeclared 200-step cap (100 decision-weighted warmup steps plus 100 mixed-preservation steps). Warmup step 100, mixed step 50, and mixed step 100 all failed the diagnostic gates in the same way: sufficiency and control status macro-F1 were both 0.333 and pair-joint accuracy was zero. Later seeds, the hard-pair confirmation holdout, unchanged development set, and sealed pilot test were not opened. The reproducible failure analysis is in `results/hard_pairs_v4_seed42_failure_analysis.md`.

## Evidence-grounded v5 design probe

The v4 failure analysis shows that its paired labels are simulator-correct but the decisive simulator response curves are hidden from the prompt. The design-only v5 probe tests a narrower capability: aggregating prompt-visible calibration evidence. It reuses 100 already-trained-on v4 physical groups, so it is not eligible for future evaluation and authorizes no additional fine-tuning.

```bash
PYTHONPATH=. /home/jiamo/miniconda3/envs/optical_sim/bin/python \
  -m optics_understanding_sft.build_evidence_v5_probe \
  --canonical-jsonl optics_understanding_sft/data/hard_pairs_v4/canonical/train.jsonl \
  --master-jsonl optics_understanding_sft/data/hard_pairs_v4/master/cases.jsonl \
  --used-curriculum-jsonl optics_understanding_sft/data/hard_pairs_v4/exports/qwen/train_decision_warmup_v4.jsonl \
  --output-dir optics_understanding_sft/data/evidence_v5_probe \
  --scenarios 100 --seed 15051

PYTHONPATH=. /home/jiamo/miniconda3/envs/optical_sim/bin/python \
  -m optics_understanding_sft.audit_evidence_v5_probe \
  --dataset-dir optics_understanding_sft/data/evidence_v5_probe \
  --output-json optics_understanding_sft/data/evidence_v5_probe/audit_report.json
```

The 400-record audit passes: the deterministic visible-evidence solver matches all targets, the 200 minimal pairs preserve their invariants, and the measured shortcut baselines remain below 0.60. Six local 28-record learnability probes show that numeric residuals make control partly pair-sensitive (0.708 status macro-F1 and 0.429 pair-joint), while sufficiency remains a constant-class predictor even with explicit thresholded directions. See `results/evidence_v5_probe_analysis.md` and the frozen tier definitions in `results/evidence_grounded_v5_protocol.md`.

The next recommended experiment is not another v4 seed. It is a fresh, scenario-disjoint v5A dataset followed by at most one predeclared 50-step seed-42 aggregation trial. Raw-centroid transfer is conditional on passing the v5A gates; the sealed pilot test stays closed.

## Evidence-grounded v5A trial

The fresh v5A round generates 300 new physical scenarios and 1,200 text-only records. Complete scenarios are split into 200 train, 50 development, and 50 target-free confirmation scenarios. The source audit replays 9,600 simulator states with zero failures; the final evidence audit reports zero solver, scaffold, leakage, overlap, or pair-invariant failures. See `results/evidence_v5a_trial_protocol.md` for the frozen data and promotion rules.

The single seed-42 QLoRA run consumed a balanced 200-record curriculum once and stopped at 50 optimizer steps. Checkpoints 25 and 50 were evaluated on all 200 development records. Neither passed any of the nine promotion gates. Checkpoint 50 improved schema validity from the base model's 0.760 to 0.885, but control status F1 fell from 0.650 to 0.605 and feasible recall fell from 0.66 to 0.38. Sufficiency stayed fully collapsed at 0.333 status F1, zero insufficient-information recall, and zero pair-joint accuracy at both checkpoints.

The confirmation split, v4 holdout, and sealed pilot test remain unopened. No checkpoint is promoted and no later seed or extra training step is authorized by this protocol. The comparison and next design recommendation are in `results/evidence_v5a_seed42_decision.md`.

## Intermediate-evidence v6 tool-use phase

V6 converts each fresh v5A example into three supervised stages: tool choice, exact tool-call construction, and tool-result interpretation. Numerical thresholding and exhaustive minimum-motion selection are implemented in `decision_tools.py`; the tools use only prompt-visible measurements and return explicit evidence objects such as `observed_direction_set`, `successful_action_indices`, and `selected_index`.

```bash
PYTHONPATH=. python -m optics_understanding_sft.build_intermediate_evidence_v6 \
  --config optics_understanding_sft/configs/intermediate_evidence_v6.yaml \
  --source-dir optics_understanding_sft/data/evidence_v5a \
  --output-dir optics_understanding_sft/data/intermediate_evidence_v6

PYTHONPATH=. python -m optics_understanding_sft.audit_intermediate_evidence_v6 \
  --dataset-dir optics_understanding_sft/data/intermediate_evidence_v6 \
  --source-dir optics_understanding_sft/data/evidence_v5a \
  --output-json optics_understanding_sft/data/intermediate_evidence_v6/audit_report.json
```

The build contains 3,600 records: 2,400 train, 600 development, and 600 target-free confirmation records. Its full reconstruction and tool-execution audit passes with zero failures. The frozen 120-step seed-42 protocol, stage-specific promotion gates, and pre-training diagnostic are documented in `results/intermediate_evidence_v6_protocol.md`.

The seed-42 run completed at the 120-step cap. Step 60 failed the schema gate after 13/13 targeted control calls produced invalid JSON. Step 120 produced valid outer JSON more often, but exact argument construction and executed-tool agreement were both 0/41; this makes the frozen 0.80 argument gate mathematically impossible even if every remaining call were correct. No checkpoint is promoted and confirmation remains sealed. See `results/intermediate_evidence_v6_seed42_decision.md`.
