# Supervisor v1.1 candidate progress log

Status: **NOT SEALED — FROZEN EVALUATION DISABLED**

## 2026-08-02 02:30 SGT — Full Qwen complete; baselines and safety dry-run complete

- Completed all preregistered Full QLoRA runs at seeds `2026080101`, `2026080102`, and `2026080103`, 200 optimizer steps each, without OOM or any post-result configuration change.
- Best checkpoints were steps 125, 175, and 125. Training wall times were 1897.09 s, 1920.69 s, and 1944.87 s; peak CUDA allocation was 5,239,299,072 bytes for every run.
- Completed greedy generation on all 60 candidate-dev records for all three Full seeds. Strict reducer result: valid JSON 1.0000; diagnosis balanced accuracy 0.9691 +/- 0.0167; diagnosis macro-F1 0.9659 +/- 0.0180; joint exact accuracy 0.9611 +/- 0.0208. These are development/engineering results only.
- Ran the preregistered fixed unified three-class baselines without a dev sweep. Balanced accuracy: logistic metrics 0.3889; metrics MLP 0.5611/0.5611/0.5500; tiny image CNN 0.6593/0.8037/0.7815; tiny image+metrics fusion 0.7370/0.8296/0.8926.
- Synthetic plus candidate train/dev state-machine dry-run passed strict JSON fail-safe stop, continuous-action injection rejection, H1-only high-level routing, reversible policy mapping, continuation/horizon guards, repeated-anomaly budget termination, and all visible 128/1024 frame/unit/range checks. No temporal recovery claim is made.
- Image-only and metrics-only Qwen retraining is in progress under the already-frozen 3-seed/200-step protocol. No frozen IID/OOD/protected model inference or image inspection has been performed.

- 2026-08-02: Started development-only pre-meeting candidate work. Existing v1 and frozen artifacts remain read-only. GPU preflight found one RTX 4080 Laptop GPU (12,282 MiB) with about 10,987 MiB free; no training process was running.
- 2026-08-02: Froze new-data seed, quotas, split assignment, width-relative reflection ranges, saturation range, counterfactual tolerances, and stop conditions before candidate generation in `protocol/data_preregistration.json`.
- 2026-08-02: Froze the three-seed Full/ablation training and dev-only selection protocol before candidate training in `protocol/experiment_protocol.json`.

This log is append-only for the candidate run. No frozen evaluation is authorized.

## 2026-08-02 07:29 SGT — Candidate run complete; human review required

- Completed all nine preregistered QLoRA trainings (Full, image-only, metrics-only; three seeds each; 200 optimizer steps) and all 45 complete-dev model generation reports (3 Full, 6 retrained-ablation, 36 non-no-op inference interventions). Every prediction file covers 60/60 candidate-dev records; all 2,700 raw continuations parse as JSON without repair.
- Full Qwen candidate-dev result: strict valid JSON 1.0000; diagnosis balanced accuracy 0.9691 +/- 0.0167; macro-F1 0.9659 +/- 0.0180; policy/joint accuracy 0.9611 +/- 0.0208. Setup-cluster/pair-preserving bootstrap 95% CI is [0.9368, 0.9931] for diagnosis BA. Two records are wrong in at least two seeds; 55/60 diagnoses agree across all three seeds.
- Inference-time image interventions caused large paired BA drops: blank image 0.6353 [0.5941, 0.6709], cross-class shuffle 0.9430 [0.9004, 0.9784], and three fixed random shuffles 0.6543-0.7121 point drops. Metrics blank/shuffle caused zero drop. This supports candidate-dev image dependence only, not general visual understanding.
- Retrained image-only Qwen reached BA 0.9037 +/- 0.0129 (Full-minus-ablation paired drop 0.0652, CI [0.0198, 0.1246]); retrained metrics-only Qwen remained at BA 0.3333. This is development evidence and is not interchangeable with inference-time intervention evidence.
- Unified baseline BA mean: logistic metrics 0.3889; metrics MLP 0.5574; tiny image CNN 0.7481; tiny image+metrics fusion 0.8198. All used the same 192-record train and 60-record dev with no dev sweep.
- Full best checkpoints were 125/175/125. Final dev loss exceeded best by 88.5%, 0.4%, and 30.4%, respectively, while training loss moved toward zero; this is seed-dependent overfitting evidence. Nine training wall times sum to 17,229.98 s (4.79 single-GPU hours). Maximum CUDA allocation/reservation was 5,239,299,072 / 8,059,355,136 bytes.
- The first aggregate-report attempt found an evaluation API field mismatch after deterministically writing only `reports/evaluations/full.json`; the file was preserved. The script was corrected to compute the summed matrix from per-seed matrices and to accept only byte-identical deterministic recomputation. The completed rerun produced all 22 condition reports and 4,000 paired bootstrap replicates.
- Final artifacts: `candidate_results.json` SHA-256 `28424e5b6995b7d74773e5672e28b3d365dcf0a383466ee45b5391e43a05bb73`; `machine_summary.json` `f1ad9b05d3a15e9269c64fa11992ad8e45c230f499b3a4c898da5ef43f6ed1ce`; `pilot_report.md` `143a51c1a86da7ab37f9aa411006d9aaaf00cbb1133ae82d65604ed7c1b9be69`; artifact registry `83917350dac0a5f66186d8288f608fa5148ffc8f4415a65f8876606af86c56d0`.
- Final disposition: **YELLOW — READY FOR HUMAN REVIEW BEFORE FINAL FREEZE**. No frozen IID/OOD/protected inference, formal frozen evaluation, final checkpoint-bound freeze, or real end-to-end closed-loop evaluation was run. Temporal recovery performance remains unverified.
- Post-report verification corrected the targeted pytest path in `commands.md` and reran the intended files: 39 passed in 0.12 s. Superseding hashes are `commands.md` `9c615494a1eba1f668532cdaefc05754175be4e19c7021869c53b035c0874bb2`, `machine_summary.json` `c114f3016716277716ffd50fcac4582b2169bcdb623832562a9918f6522d3838`, and artifact registry `c4da0c06f5f7e1998195c1a2d9336203a1a2e85038913d6717bf2f97e4147079`.
- Final runtime clarification records the 7.02-hour candidate artifact-to-report window separately from summed training and generation timings. Final superseding hashes are `machine_summary.json` `5e5e4a16d8a2e83382c68a951ec9d733415d4d840ca66f274f813b1d109ced31`, `pilot_report.md` `3311115b80cb778d6551ac854df808fe19974418e51eac43009effe62c63991f`, and artifact registry `d1028c5bc90c4320542794060c246f20de0ffabd55ebbb033be5e49613f2d221`.
