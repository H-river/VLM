# Qwen-VL Supervisor v1 Progress Log

## 2026-08-01T20:10:00+08:00 — audit and scope freeze

- Status entering task: `PROVISIONALLY READY FOR QWEN-VL`.
- Repository instructions, attachment, branch, commit, and dirty worktree audited.
- Legal cohorts and exact source hashes recorded in `data_audit_report.md` before exporter implementation.
- No new data generation is required.
- Static same-state metrics are not represented as temporal history; controller traces without paired anomaly images are excluded from SFT.
- Fixed-pixel reflection is excluded; width-relative reflection remains a provisional engineering family with its known Q1/boundary limitations visible.
- Existing Qwen/TRL/PEFT/bitsandbytes stack will be reused. A contained trainer is required for approximately 20 steps, checkpoint save, true resume, structured logs, and resource reporting.
- Frozen test inference remains prohibited in this task.

## 2026-08-01T20:27:00+08:00 — manifest and export freeze

- Built strict schema v1 plus train/dev/frozen-IID/frozen-OOD manifests from hash-checked source files.
- Counts: 96 train, 36 dev, 72 frozen IID, and 60 saturation-only frozen OOD records.
- Full validator passed 264 unique images and 132 complete pairs with zero setup, pair, augmentation-base, episode, or image-hash overlap across splits.
- No old fixed-pixel reflection or protected source record entered train/dev.
- Exported 96 train and 36 dev Qwen chat records plus balanced 36/12 smoke views and three redacted family examples.
- Export contract has one image token, allow-listed structured state, compact strict target JSON, and no model-visible paths or provenance.
- Same-seed export regression and manifest/export/parser/controller unit tests pass (38 tests at this checkpoint).

## 2026-08-01T20:45:00+08:00 — actual processor and training smoke

- The actual local Qwen2.5-VL processor and TRL collator iterated all 48
  balanced smoke records with image tokens present, prompt loss fully masked,
  assistant targets supervised, and no truncation.
- The local 3B checkpoint loaded in NF4 4-bit mode on the RTX 4080 Laptop GPU;
  audited Qwen2.5-VL language and vision LoRA targets attached successfully.
- A fresh approximately 20-step optimization completed with finite loss and
  verified LoRA tensor changes. Checkpoint state is strict JSON plus
  safetensors; legacy pickle optimizer/RNG files are never loaded.
- Diagnostic resume variants exposed PEFT BF16-to-FP32 adapter upcasting,
  non-native bitsandbytes buffer restoration, and nondeterministic CUDA replay.
  Each issue was fixed and those diagnostic artifacts were retained but marked
  superseded rather than used as acceptance evidence.

## 2026-08-01T21:45:00+08:00 — authoritative deterministic resume audit

- Completed fresh run `20260801T134101.484544Z` with a byte-pinned local model
  tree, deterministic algorithms, and an unchanged 21-step scheduler horizon.
- Independently resumed checkpoint 10 and stopped after step 11.
- Step-11 loss, entropy, gradient norm, learning rate, accuracy, cumulative
  token count, and epoch were exactly equal.
- All 824 BF16 adapter tensors, all 4,656 optimizer/scheduler safetensors, and
  the corresponding serialized bytes were exact. Candidate restore invariants
  also verified parameter-name mapping, scheduler/RNG/data-skip state, and the
  TRL cumulative token counter.
- The authoritative replay artifact is
  `artifacts/training/resume_replay_comparison.step11.deterministic_pinned.json`;
  it is engineering resume evidence only.

## 2026-08-01T21:52:00+08:00 — authoritative development inference smoke

- Generated one raw continuation for each of 12 balanced development records
  from the authoritative fresh adapter, using greedy decoding and no JSON
  extraction or repair.
- Actual bytes for all 12 current images and the complete local base-model tree
  matched their expected SHA-256 values before inference.
- Coverage and strict JSON validity were 12/12. The development-only reducer
  produced the complete metric/confusion/subgroup structure; balanced accuracy
  and joint exact accuracy were both 0.6667, with no accuracy gate applied.
- The guarded controller-boundary smoke routed all three valid family decisions,
  emitted zero continuous actuator fields, rejected numeric injection, and
  converted invalid model output to a conservative non-dispatched stop.
- No frozen prediction was generated or opened; no scientific-performance
  conclusion was drawn.

## 2026-08-01T22:22:37+08:00 — pre-server protocol seal and final verification

- Repeated authoritative development generation while preserving the
  byte-pinned smoke-config SHA-256 and supplying the expected local-model tree
  as a recorded invocation override. Coverage and strict JSON validity again
  passed 12/12; balanced accuracy and joint exact accuracy remained 0.6667.
- Sealed `artifacts/evaluation_protocol_freeze.json` exactly once at SHA-256
  `53366f9d0c34908bfa97848c30631ecb29564c8beb4bf5086d23eed2c40de27e`.
  Two independent dry preparations, the real artifact, and reviewer output
  were byte-identical.
- The seal rehashed 264 manifest records and their actual image bytes plus all
  304 registered identities. It opened no prediction file and executed no
  model, controller, server training, or formal evaluation.
- Independent post-seal registry verification passed without pinned-file
  drift. The full test suite in the QLoRA environment passed 84/84, both smoke
  and server configurations passed validate-only checks, and the full manifest
  validator passed with every cross-split overlap count at zero.
- Engineering pipeline preparation is complete. Three-seed full server
  training, development-only checkpoint/baseline selection, concrete formal
  runners, the later final checkpoint-bound freeze, and frozen scientific
  evaluation remain future gated work.
