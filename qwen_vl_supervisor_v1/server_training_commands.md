# Frozen server QLoRA launch commands

This is a future full-training launcher. It was prepared, not executed, during
the development smoke task. The server config evaluates `sft_dev.jsonl` only;
the trainer rejects frozen, held-out, OOD, and test rows before model loading.

Run the three frozen training seeds on one four-GPU server from the repository
root:

```bash
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

If the checkpoint is staged locally rather than fetched at the frozen revision,
append `--model-id /absolute/path/to/Qwen2.5-VL-3B-Instruct`. A local path does
not use Hugging Face revision resolution. Before any model load, the launcher
hashes the complete nonvolatile snapshot tree and refuses it unless the tree
SHA-256 is exactly
`2e1bd29589b91134a667572080bec76a5fb1446c49acddfa7f13049314bf3175`.
The run manifest records the aggregate identity and every included file hash.

Resume an interrupted seed without changing its configured final step:

```bash
cd /home/jiamo/VLM
SEED=2026080101
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  qwen_vl_supervisor_v1/train_qlora.py \
  --config qwen_vl_supervisor_v1/configs/training_server.yaml \
  --seed "$SEED" \
  --data-seed "$SEED" \
  --output-dir "qwen_vl_supervisor_v1/artifacts/training/server_qwen25vl_3b/seed_$SEED" \
  --resume-from-checkpoint latest
```

Resume accepts only the repository's hash-checked JSON+safetensors optimizer,
scheduler, and per-rank RNG sidecars. It never deserializes legacy
`optimizer.pt`, `scheduler.pt`, or `rng_state.pth` files.
The frozen server config also enables full deterministic algorithms so an
independent resume can be audited numerically, with the expected throughput
cost of deterministic CUDA execution.
