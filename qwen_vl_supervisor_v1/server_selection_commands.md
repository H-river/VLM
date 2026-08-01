# Frozen dev-only checkpoint-selection commands

These commands are prepared for the future server run; they were not executed
during the local engineering smoke. They evaluate **development data only**.
The selector refuses missing or extra checkpoints, incomplete generations,
missing predictions, seed drift, hash drift, non-dev inputs, and any report that
does not explicitly record protected/frozen use as false.

Run from the transferred repository root after all three training runs have
completed. Use a new `SELECTION_ROOT`; the final selector refuses to overwrite
an existing selection artifact.

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
  SEED_DIR="$TRAINING_ROOT/seed_$SEED"
  test -f "$SEED_DIR/run_manifest.latest.json"
  mapfile -t OBSERVED_CHECKPOINTS < <(
    find "$SEED_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
      | LC_ALL=C sort -V
  )
  test "${#OBSERVED_CHECKPOINTS[@]}" -eq "${#STEPS[@]}"

  for STEP in "${STEPS[@]}"; do
    CHECKPOINT="$SEED_DIR/checkpoint-$STEP"
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

The output contains one ranked list per declared training seed and selects rank
one independently for each. Ranking is, in order: higher joint exact accuracy,
higher diagnosis macro F1, higher valid-JSON rate, lower optimizer step, then
lexicographically lower deterministic checkpoint-tree SHA-256. The selector
recomputes every offline report from its raw development predictions before
using any metric.

The later formal offline matrix has different seed semantics: arm C supplies
all three integer training seeds and is reduced with `--evaluation-config`;
arms A, B, and D each supply one deterministic prediction per sample with the
literal seed label `deterministic_reference` and are reduced with
`--expected-seeds deterministic_reference`. A/B/D are not copied three times.
