# Targeted v1.1 ablation protocol

The primary selection metric is the frozen v2 equal-task macro on all 120 pilot validation records. The legacy v1 score is reported only for continuity. The sealed test prompts and private test labels are not used.

Candidates:

1. Original 40-step trial checkpoint 40 (the strongest pre-ablation model under v2).
2. Continue the original checkpoint 20 for 20 optimizer steps on the targeted curriculum at learning rate 3e-5.
3. Train a fresh adapter for 40 optimizer steps on the same curriculum at learning rate 1e-4.

All generative evaluations use greedy decoding and identical prompts. Point estimates are accompanied by a 10,000-replicate paired bootstrap that resamples the 30 physical scenario groups, keeping the four related records in each sampled scenario together.

The model with the largest v2 point macro is the protocol winner. Per-task scores, feasibility-class recall, simulator control success, schema validity, and modality slices remain mandatory diagnostics. A macro winner with a severe task collapse is recorded as such and is not automatically suitable for deployment or sealed-test evaluation.
