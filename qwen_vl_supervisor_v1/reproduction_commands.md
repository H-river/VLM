# Qwen-VL supervisor v1 reproduction commands

Run every command from the repository root. The commands are deliberately
separated into three scopes:

1. **Completed local engineering smoke**: legal train/development data only.
2. **Future server training and development-only selection**: no frozen
   prediction is permitted.
3. **Future frozen scientific evaluation**: forbidden until the final freeze
   gate accepts concrete, reviewed prediction and closed-loop runners.

The local smoke demonstrates pipeline integrity only. It is not evidence that
Qwen-VL is scientifically accurate or that the overall benchmark has passed.

## Local environment and read-only integrity checks

The recorded workstation used these interpreters:

```bash
set -euo pipefail
cd /home/jiamo/VLM

PURE_PYTHON=/home/jiamo/miniconda3/bin/python
QLORA_PYTHON=/home/jiamo/miniconda3/envs/optics_qlora/bin/python
```

Validate all four source-of-truth manifests, without opening any prediction
file:

```bash
REPRO_ROOT="$(mktemp -d "$PWD/qwen_vl_supervisor_v1/artifacts/reproduction.XXXXXX")"

"$PURE_PYTHON" -m qwen_vl_supervisor_v1.validate_manifest \
  qwen_vl_supervisor_v1/manifests/manifest_train.jsonl \
  qwen_vl_supervisor_v1/manifests/manifest_dev.jsonl \
  qwen_vl_supervisor_v1/manifests/manifest_frozen_iid.jsonl \
  qwen_vl_supervisor_v1/manifests/manifest_frozen_ood.jsonl \
  --repository-root /home/jiamo/VLM \
  --output "$REPRO_ROOT/leakage_split_audit.json"
```

Regenerate manifests and SFT exports into a new scratch directory. This keeps
the previously recorded manifests and reports untouched:

```bash
"$PURE_PYTHON" -m qwen_vl_supervisor_v1.build_manifests \
  --repository-root /home/jiamo/VLM \
  --output-dir "$REPRO_ROOT/manifests"

"$PURE_PYTHON" -m qwen_vl_supervisor_v1.export_sft \
  --repository-root /home/jiamo/VLM \
  --train-manifest "$REPRO_ROOT/manifests/manifest_train.jsonl" \
  --dev-manifest "$REPRO_ROOT/manifests/manifest_dev.jsonl" \
  --output-dir "$REPRO_ROOT/sft" \
  --seed 2026080101

for SPLIT in train dev frozen_iid frozen_ood; do
  cmp "qwen_vl_supervisor_v1/manifests/manifest_${SPLIT}.jsonl" \
      "$REPRO_ROOT/manifests/manifest_${SPLIT}.jsonl"
done

for EXPORT in sft_train.jsonl sft_dev.jsonl sft_smoke_train.jsonl \
              sft_smoke_dev.jsonl sample_qwen_vl.jsonl export_report.json; do
  cmp "qwen_vl_supervisor_v1/sft/$EXPORT" "$REPRO_ROOT/sft/$EXPORT"
done

for VIEW in manifest_smoke_train_view.jsonl manifest_smoke_dev_view.jsonl; do
  cmp "qwen_vl_supervisor_v1/sft/manifest_views/$VIEW" \
      "$REPRO_ROOT/sft/manifest_views/$VIEW"
done

for FAMILY in nominal sensor_saturation secondary_reflection; do
  cmp "qwen_vl_supervisor_v1/sft/rendered_examples/$FAMILY.md" \
      "$REPRO_ROOT/sft/rendered_examples/$FAMILY.md"
done
```

Exercise the actual Qwen processor, image path, chat-template, visual-token,
loss-mask, and no-truncation contract over all 48 smoke rows:

```bash
"$QLORA_PYTHON" -m qwen_vl_supervisor_v1.verify_sft_contract \
  --repository-root /home/jiamo/VLM \
  --data qwen_vl_supervisor_v1/sft/sft_smoke_train.jsonl \
         qwen_vl_supervisor_v1/sft/sft_smoke_dev.jsonl \
  --model /home/jiamo/HF_models/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --min-pixels 3136 \
  --max-pixels 50176 \
  --output "$REPRO_ROOT/sft_contract_verification.json"
```

The dependency-light regression suite is:

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/pytest -q -rs -p no:cacheprovider \
  qwen_vl_supervisor_v1/tests
```

## Local approximately-20-step smoke and exact true-resume replay

These commands load the actual local Qwen2.5-VL 3B checkpoint and require a
CUDA GPU. They write only to the new scratch directory created above. The
21-step horizon deliberately puts the first replayed optimizer step after the
normal step-10 checkpoint while its learning rate is still nonzero. Both runs
retain that same scheduler horizon; `--stop-after-global-step 11` controls only
the audit run's execution and does not shorten its configured schedule.

```bash
SMOKE_CONFIG=qwen_vl_supervisor_v1/configs/training_smoke.yaml
REFERENCE_DIR="$REPRO_ROOT/training/uninterrupted_21step"
REPLAY_DIR="$REPRO_ROOT/training/resumed_step11"
LOCAL_MODEL_TREE_SHA256=2e1bd29589b91134a667572080bec76a5fb1446c49acddfa7f13049314bf3175

"$QLORA_PYTHON" -m qwen_vl_supervisor_v1.train_qlora \
  --config "$SMOKE_CONFIG" \
  --output-dir "$REFERENCE_DIR" \
  --expected-local-snapshot-tree-sha256 "$LOCAL_MODEL_TREE_SHA256" \
  --max-steps 21 \
  --save-total-limit 4 \
  --full-determinism \
  --save-at-global-step 11

"$QLORA_PYTHON" -m qwen_vl_supervisor_v1.train_qlora \
  --config "$SMOKE_CONFIG" \
  --output-dir "$REPLAY_DIR" \
  --expected-local-snapshot-tree-sha256 "$LOCAL_MODEL_TREE_SHA256" \
  --max-steps 21 \
  --save-total-limit 4 \
  --full-determinism \
  --resume-from-checkpoint "$REFERENCE_DIR/checkpoint-10" \
  --save-at-global-step 11 \
  --stop-after-global-step 11
```

Compare the uninterrupted and independently resumed step-11 checkpoint. The
comparator resolves both run manifests and logs, verifies the immediate
10-to-11 step relation and restore audit, and requires exact deterministic log
fields, BF16 adapter tensor bits, and optimizer/scheduler bytes. Evaluation
fields are compared only if both logs contain an evaluation row at step 11;
the optimizer replay proof does not invent one.

```bash
"$QLORA_PYTHON" -m qwen_vl_supervisor_v1.compare_resume_replays \
  --reference-checkpoint "$REFERENCE_DIR/checkpoint-11" \
  --candidate-checkpoint "$REPLAY_DIR/checkpoint-11" \
  --replay-step 11 \
  --output "$REPRO_ROOT/resume_replay_comparison.step11.json"
```

The checked-in engineering evidence produced by this exact shape of audit is
`artifacts/training/resume_replay_comparison.step11.deterministic_pinned.json`
(SHA-256
`632eed2293c42c4459922eb7aed31c5a7da07ff63a024450ae51ececfc4981ad`).
Both run manifests remain authoritative for parameter counts, LoRA delta
evidence, adapter dtype audits, per-step loss, runtime, throughput, GPU memory,
checkpoint paths, and resume sequencing.

Generate and score only the balanced 12-record development view, then exercise
the guarded controller boundary. The evaluator is given the matching 12-record
manifest view, so missing rows cannot be hidden by changing the denominator:

```bash
SMOKE_PREDICTIONS="$REPRO_ROOT/predictions_dev12_seed_2026080101.jsonl"

"$QLORA_PYTHON" -m qwen_vl_supervisor_v1.generate \
  --config "$SMOKE_CONFIG" \
  --data qwen_vl_supervisor_v1/sft/sft_smoke_dev.jsonl \
  --image-root /home/jiamo/VLM \
  --adapter "$REFERENCE_DIR/final_adapter" \
  --expected-local-snapshot-tree-sha256 "$LOCAL_MODEL_TREE_SHA256" \
  --seed 2026080101 \
  --max-new-tokens 128 \
  --output "$SMOKE_PREDICTIONS" \
  --report "$SMOKE_PREDICTIONS.report.json"

"$PURE_PYTHON" -m qwen_vl_supervisor_v1.evaluate_offline \
  --manifest qwen_vl_supervisor_v1/sft/manifest_views/manifest_smoke_dev_view.jsonl \
  --predictions "$SMOKE_PREDICTIONS" \
  --expected-seeds 2026080101 \
  --output "$REPRO_ROOT/offline_evaluation_dev12.json"

"$PURE_PYTHON" -m qwen_vl_supervisor_v1.run_adapter_smoke \
  --output "$REPRO_ROOT/controller_adapter_wiring.json"
```

None of the commands in this section accepts a frozen manifest or frozen
prediction path. Do not substitute one.

## Pre-server protocol commitment

This step is required before future full training. The utility reads and
hashes the frozen manifests and their actual image bytes, but it has no model,
controller, or prediction-reading path. It refuses to overwrite an existing
artifact. On a new server transfer, create it only if absent; otherwise compute
and externally record the hash of the existing committed bytes:

```bash
PROTOCOL_ARTIFACT=qwen_vl_supervisor_v1/artifacts/evaluation_protocol_freeze.json

if test ! -e "$PROTOCOL_ARTIFACT"; then
  "$PURE_PYTHON" qwen_vl_supervisor_v1/scripts/freeze_evaluation.py prepare \
    --config qwen_vl_supervisor_v1/configs/evaluation_frozen.yaml \
    --output "$PROTOCOL_ARTIFACT" \
    --authorize-pre-server-protocol-freeze
fi

EVALUATION_PROTOCOL_FREEZE_SHA256="$("$PURE_PYTHON" \
  qwen_vl_supervisor_v1/scripts/freeze_evaluation.py hash \
  --path "$PROTOCOL_ARTIFACT" --digest-only)"
printf '%s\n' "$EVALUATION_PROTOCOL_FREEZE_SHA256"
```

Creating this protocol artifact is not formal evaluation and does not
authorize frozen prediction.

## Future full server training — do not run locally in this task

Activate the server environment containing the pinned Qwen/Transformers/TRL/
PEFT/bitsandbytes stack, then run the exact three configured seeds from the
transferred repository root:

```bash
set -euo pipefail
cd /home/jiamo/VLM

CONFIG=qwen_vl_supervisor_v1/configs/training_server.yaml
OUTPUT_ROOT=qwen_vl_supervisor_v1/artifacts/training/server_qwen25vl_3b

for SEED in 2026080101 2026080102 2026080103; do
  torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    qwen_vl_supervisor_v1/train_qlora.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --data-seed "$SEED" \
    --output-dir "$OUTPUT_ROOT/seed_$SEED"
done
```

For a byte-identical locally staged base model, append
`--model-id /absolute/path/to/Qwen2.5-VL-3B-Instruct`; the trainer verifies its
complete snapshot-tree hash against the server config before loading it.

Resume one interrupted seed without changing its configured final step:

```bash
SEED=2026080101
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  qwen_vl_supervisor_v1/train_qlora.py \
  --config qwen_vl_supervisor_v1/configs/training_server.yaml \
  --seed "$SEED" \
  --data-seed "$SEED" \
  --output-dir "qwen_vl_supervisor_v1/artifacts/training/server_qwen25vl_3b/seed_$SEED" \
  --resume-from-checkpoint latest
```

Training-time evaluation is development-only. The trainer rejects frozen,
held-out, OOD, and test rows before model loading.

## Future development-only checkpoint selection

Run generation and reduction for every config-derived saved step and every
declared seed. The commands below intentionally use only `sft_dev.jsonl` and
`manifest_dev.jsonl`:

```bash
set -euo pipefail
cd /home/jiamo/VLM

CONFIG=qwen_vl_supervisor_v1/configs/training_server.yaml
TRAINING_ROOT=qwen_vl_supervisor_v1/artifacts/training/server_qwen25vl_3b
SELECTION_ROOT=qwen_vl_supervisor_v1/artifacts/dev_checkpoint_selection
DEV_SFT=qwen_vl_supervisor_v1/sft/sft_dev.jsonl
DEV_MANIFEST=qwen_vl_supervisor_v1/manifests/manifest_dev.jsonl
SEEDS=(2026080101 2026080102 2026080103)
STEPS=(25 50 75 100 125 150 175 200 225 250 275 300)

for SEED in "${SEEDS[@]}"; do
  for STEP in "${STEPS[@]}"; do
    CHECKPOINT="$TRAINING_ROOT/seed_$SEED/checkpoint-$STEP"
    EVIDENCE="$SELECTION_ROOT/seed_$SEED/checkpoint-$STEP"
    test -d "$CHECKPOINT"
    mkdir -p "$EVIDENCE"

    python -m qwen_vl_supervisor_v1.generate \
      --config "$CONFIG" \
      --data "$DEV_SFT" \
      --image-root /home/jiamo/VLM \
      --adapter "$CHECKPOINT" \
      --seed "$SEED" \
      --max-new-tokens 128 \
      --output "$EVIDENCE/predictions_dev.jsonl" \
      --report "$EVIDENCE/predictions_dev.jsonl.report.json"

    python -m qwen_vl_supervisor_v1.evaluate_offline \
      --manifest "$DEV_MANIFEST" \
      --predictions "$EVIDENCE/predictions_dev.jsonl" \
      --expected-seeds "$SEED" \
      --output "$EVIDENCE/offline_dev.json"
  done
done

python -m qwen_vl_supervisor_v1.select_dev_checkpoints select \
  --training-config "$CONFIG" \
  --training-root "$TRAINING_ROOT" \
  --evidence-root "$SELECTION_ROOT" \
  --dev-sft "$DEV_SFT" \
  --dev-manifest "$DEV_MANIFEST" \
  --artifact "$SELECTION_ROOT/dev_checkpoint_selection.json"

python -m qwen_vl_supervisor_v1.select_dev_checkpoints verify \
  --training-config "$CONFIG" \
  --training-root "$TRAINING_ROOT" \
  --evidence-root "$SELECTION_ROOT" \
  --dev-sft "$DEV_SFT" \
  --dev-manifest "$DEV_MANIFEST" \
  --artifact "$SELECTION_ROOT/dev_checkpoint_selection.json"
```

The selector refuses missing or extra checkpoints, incomplete development
coverage, report drift, unsafe checkpoint state, seed drift, and any protected
or frozen use. It selects one checkpoint independently per training seed.

## Future final freeze and scientific evaluation — forbidden now

The following is a future authorization sequence, not a runnable command for
the present task. First implement and review concrete frozen prediction and
closed-loop runners. Set `PREDICTION_COMMAND` and `CLOSED_LOOP_COMMAND` to their
complete commands; placeholders are rejected. The selected checkpoints must
resolve beneath `FINAL_CHECKPOINT`, and the final freeze re-hashes each selected
checkpoint plus the complete containing tree.

```bash
FINAL_CHECKPOINT=qwen_vl_supervisor_v1/artifacts/training/server_qwen25vl_3b
DEV_SELECTION_ARTIFACT=qwen_vl_supervisor_v1/artifacts/dev_checkpoint_selection/dev_checkpoint_selection.json
BASELINE_DEV_SELECTION_ARTIFACT=qwen_vl_supervisor_v1/artifacts/dev_baseline_selection/baseline_dev_selection.json
PREDICTION_RUNNER=/absolute/path/to/reviewed_formal_prediction_runner.py
CLOSED_LOOP_RUNNER=/absolute/path/to/reviewed_formal_closed_loop_runner.py
PREDICTION_COMMAND='the complete reviewed frozen prediction command'
CLOSED_LOOP_COMMAND='the complete reviewed paired closed-loop command'

FINAL_CHECKPOINT_SHA256="$(python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py hash --path "$FINAL_CHECKPOINT" --digest-only)"
TRAINING_CONFIG_SHA256="$(python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py hash --path qwen_vl_supervisor_v1/configs/training_server.yaml --digest-only)"
DEV_SELECTION_ARTIFACT_SHA256="$(python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py hash --path "$DEV_SELECTION_ARTIFACT" --digest-only)"
BASELINE_DEV_SELECTION_ARTIFACT_SHA256="$(python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py hash --path "$BASELINE_DEV_SELECTION_ARTIFACT" --digest-only)"
PREDICTION_RUNNER_SHA256="$(python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py hash --path "$PREDICTION_RUNNER" --digest-only)"
CLOSED_LOOP_RUNNER_SHA256="$(python qwen_vl_supervisor_v1/scripts/freeze_evaluation.py hash --path "$CLOSED_LOOP_RUNNER" --digest-only)"

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

Externally record the printed `EVALUATION_FREEZE_SHA256`. Immediately before
any future frozen prediction, require the formal preflight and run only the two
commands sealed above:

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

After those reviewed runners finish, reduce the complete two-split comparison
matrix. A/B/D each contain one `deterministic_reference` output per sample; C
contains every configured integer training seed:

```bash
set -euo pipefail
for SPLIT in frozen_iid frozen_ood; do
  for ARM in A B D; do
    python -m qwen_vl_supervisor_v1.evaluate_offline \
      --manifest "qwen_vl_supervisor_v1/manifests/manifest_${SPLIT}.jsonl" \
      --predictions "$FORMAL_OUTPUT/$SPLIT/arm_${ARM}_predictions.jsonl" \
      --expected-seeds deterministic_reference \
      --output "$FORMAL_OUTPUT/$SPLIT/offline_arm_${ARM}.json"
  done

  python -m qwen_vl_supervisor_v1.evaluate_offline \
    --manifest "qwen_vl_supervisor_v1/manifests/manifest_${SPLIT}.jsonl" \
    --predictions "$FORMAL_OUTPUT/$SPLIT/arm_C_predictions_all_training_seeds.jsonl" \
    --evaluation-config qwen_vl_supervisor_v1/configs/evaluation_frozen.yaml \
    --output "$FORMAL_OUTPUT/$SPLIT/offline_arm_C.json"
done
```

Do not run this final section during pipeline preparation. It is the future
scientific evaluation, distinct from both the successful local engineering
smoke and the future full server training.
