# Qwen-VL supervisor v1 development smoke report

Smoke date: 2026-08-01 (Asia/Singapore)  
Scope: engineering pipeline integrity on legal train/development records only  
Scientific status: no frozen prediction or formal frozen evaluation was run

## Outcome

The actual local Qwen2.5-VL 3B NF4/LoRA smoke passed. It exercised the
manifest, deterministic SFT export, actual Qwen processor/TRL collator,
forward/backward optimization, safe checkpoint save, exact nonterminal resume,
development generation, strict offline reducer, and guarded controller
boundary on an NVIDIA GeForce RTX 4080 Laptop GPU.

The balanced views contain 36 training records (12 nominal, 12 saturation,
12 width-relative reflection) and 12 development records (4 per class). They
are views of the validated source splits, not new split assignments.

## Integrity gates

- Full validation passed for 264 records, 132 complete counterfactual pairs,
  132 setups, and 264 unique image hashes.
- Setup, pair, episode, augmentation-base, and exact-image overlap across
  train/dev/frozen-IID/frozen-OOD were all zero.
- Protected rows in train/dev: zero. Fixed-pixel reflection rows: zero.
- Every source dataset and pair-metadata input was checked against a pinned
  SHA-256 before manifest construction.
- The actual Qwen processor and TRL collator iterated all 48 smoke rows. Each
  had 25 image-pad tokens; all system/user labels were `-100`; every target
  token was supervised; critical content was not truncated; maximum sequence
  length was 627 tokens.
- Trainer and generator independently enforce the SFT allowlist, data hashes,
  actual image-byte hashes, local model-tree hash, revision, and pixel budget.
- Protected/frozen prediction files were neither generated nor opened.

## Authoritative training run

The authoritative fresh run is `20260801T134101.484544Z` under
`artifacts/training/smoke_qwen25vl_3b_midresume_reference_deterministic_pinned`.
It used a 21-step audit horizon so that step 10 -> 11 was a nonterminal resume
check under an unchanged scheduler; this is the requested approximately
20-step smoke, not full training.

- Base: local `Qwen2.5-VL-3B-Instruct`; recorded upstream model/processor
  revision `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Local snapshot: 14 nonvolatile files, 7,520,919,614 bytes, tree SHA-256
  `2e1bd29589b91134a667572080bec76a5fb1446c49acddfa7f13049314bf3175`;
  the expected hash was verified before loading.
- Quantization/compute: NF4 4-bit, double quantization, bfloat16 compute, SDPA,
  gradient checkpointing, and deterministic algorithms.
- Pixel budget: 3,136 to 50,176 pixels.
- Batch: one example/device with gradient accumulation two.
- Optimizer/schedule: paged AdamW 8-bit, learning rate `2e-4`, cosine schedule,
  10% warmup.
- LoRA: rank 8, alpha 16, dropout 0.05; language attention `q/k/v/o`, language
  and vision `gate/up/down`, and audited Qwen2.5-VL vision `attn.qkv` plus
  `attn.proj` targets.
- Parameters: 2,054,566,912 total; 20,542,464 trainable (0.999844%).
- All 21 logged losses were finite. First/last loss was 0.189170/0.034835;
  observed range was 0.024535 to 0.273245.
- Development loss was 0.111401 at step 10, 0.045883 at step 20, and 0.045553
  at step 21. These are development diagnostics, not frozen results.
- All 32 sampled intended LoRA tensors changed; maximum absolute delta was
  0.091797.

The effective audit and reproduction commands explicitly supply full
determinism and the expected local-model tree SHA-256. The byte-pinned
`training_smoke.yaml` remains unchanged; the invocation-level overrides are
recorded in each run report.

## Runtime and GPU memory

- Fresh audit run: 108.99 seconds wall time around training; Trainer runtime
  108.14 seconds; 0.388 examples/second.
- Peak training GPU allocation/reservation:
  5,150,460,928 / 6,075,449,344 bytes.
- Resumed step-10 -> 11 run: 6.77 seconds wall time around training; Trainer
  runtime 5.14 seconds.
- Development generation: 26.69 seconds total model-generation latency,
  2.225 seconds/example (model loading excluded); about 38.70 seconds from
  recorded invocation start to completion.
- Peak generation GPU allocation/reservation:
  2,650,671,104 / 2,797,600,768 bytes.

## Checkpoint and exact resume result

The fresh run saved steps 10, 11, 20, and 21 with strict JSON plus safetensors
optimizer/scheduler and RNG sidecars. The independent candidate loaded step 10
and executed exactly step 11 under the same 21-step scheduler horizon; its
learning rate was nonzero (`1.3420201433256689e-4`). No legacy pickle optimizer,
scheduler, or RNG file was deserialized.

Before the replayed update, restore checks proved:

- all 824 optimizer parameter names/order matched;
- all 4,656 optimizer tensors (41,629,312 elements) restored exactly into
  native bitsandbytes storage;
- scheduler state, Python/NumPy/CPU/CUDA RNG, Trainer data skip, and the TRL
  cumulative token counter restored exactly; and
- all 824 adapter tensors remained bfloat16 (20,542,464 elements).

At step 11, loss, gradient norm, entropy, learning rate, token accuracy, token
count, and epoch were exactly equal. The saved uninterrupted and resumed
checkpoint-11 adapter, optimizer/scheduler, and RNG files were byte-identical;
all decoded tensor keys, dtypes, shapes, and values were exact. The common
adapter SHA-256 was
`1bf1ae7985dd7591128a16cc2967f4003bac00e2cbd5738b4c7a86a063ab3cb6`.
The primary machine-readable comparison is
`artifacts/training/resume_replay_comparison.step11.deterministic_pinned.json`
(SHA-256
`632eed2293c42c4459922eb7aed31c5a7da07ff63a024450ae51ececfc4981ad`);
the companion full-state comparison including post-step RNG tensors is
`artifacts/training/midresume_deterministic_comparison.step11.json`.

Earlier diagnostic resume variants are retained but superseded. They exposed
and led to fixes for PEFT adapter dtype upcasting, replacement of native
bitsandbytes state buffers, and nondeterministic CUDA replay. No result from
those variants is used as authoritative acceptance evidence.

## Development generation and offline reducer

Generation used the authoritative fresh run's final adapter
(`adapter_model.safetensors` SHA-256
`743789b04ad289bdbf554be14e80dbc6385339e45d3c8d4248930735133f2e54`),
greedy decoding, and the balanced 12-record development view. The generator
hashed all 12 images before model load and immediately before inference, and
captured the complete raw continuation without extraction or repair.

The superseding prediction artifact is
`artifacts/smoke/predictions_authoritative_d139_pinned_dev12_seed_2026080101.jsonl`
(SHA-256
`1c519846bb26aa49e300e58d5c6122774e060055b39ef5ea2416608b1a7a4519`).
Its report records the unchanged smoke-config SHA-256
`d139985ad70ecae470c102212b8f2a4052f67c81d9a41e28bd3501b12acef9cd`
and the successfully verified model-tree override. The matching offline report
is `artifacts/smoke/offline_evaluation_authoritative_d139_pinned_dev12.json`
(SHA-256
`f0cb2516a1a7c6e2a1d9b98d0c5c5da85bee65c8fa6992bb5e63a3c48ee17a22`).

- Coverage: 12/12; strict whole-string valid JSON: 12/12.
- Predicted diagnoses: 2 nominal, 8 saturation, 2 reflection.
- Development-only diagnosis balanced accuracy: 0.6667.
- Diagnosis macro F1: 0.6667.
- Measurement-policy accuracy: 0.6667.
- Supervisor-action macro F1: 0.6667.
- Joint diagnosis + policy + action exact accuracy: 0.6667.
- Recall: nominal 0.50, saturation 1.00, reflection 0.50.

Missing outputs are synthesized as invalid and retained in all applicable
denominators. Tests cover malformed/duplicate/trailing JSON, invalid enums,
missing rows, extra seeds, and masked targets. The observed model outputs all
happened to be valid; no accuracy target was required or used for acceptance.

## Controller boundary smoke

Actuator-free mock wiring dispatched one valid decision for each family to the
documented frozen policy names. It emitted zero continuous actuator fields.
A numeric actuator injection was rejected, and the closed-loop-facing guarded
API converted it to an audited conservative stop with
`execute_frozen_cem=false` and no callback dispatch.

The frozen H1/CEM controller itself was not run in this wiring smoke. The
available static anomaly images are not time-aligned sequential real-observation
episodes; inventing such episodes would fabricate temporal data. Formal paired
closed-loop evaluation remains future work after server training and the final
evaluation gate.

## Acceptance decision

All engineering smoke acceptance criteria passed. This validates pipeline
integrity only. It does not establish Qwen-VL performance, closed-loop benefit,
reflection generalization, or an overall benchmark pass. Full three-seed server
training and the later frozen scientific comparison remain unrun.
