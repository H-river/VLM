# V10 specialist-improvement cycle results

Date: 2026-07-29

## Decision

The frozen selections are:

| Component | Selection |
|---|---|
| forward | `frozen_v9` |
| numerical inverse | `frozen_v9` |
| visual inverse | `frozen_v9_direct_visual_scorer` |
| direction | `frozen_separate_direction` |

No experimental artifact silently replaced a frozen component. All choices
were made on the new development split and hashed before the locked test was
opened.

## Evaluation boundary

The completed pilot contains 384 training,
160 development, and 160 locked-test
independent groups, each with all 81 actions. Setup and setup-plus-state
overlap across splits are both zero. The full 2,400-group training generation
was attempted first; 678 training shards completed
before the sustained estimate exceeded the current window. The deterministic
384-group prefix was finalized as the preregistered pilot, and extra shards
were excluded from every manifest and model.

## Controlled forward result

| Experiment | Natural-action strict all-five (group-bootstrap 95% CI) | Full-surface strict |
|---|---:|---:|
| A | 18.12% [12.50%, 24.38%] | 12.11% |
| B | 18.12% [12.48%, 24.38%] | 11.90% |
| C | 18.12% [12.50%, 24.38%] | 11.91% |
| D | 18.12% [12.50%, 24.39%] | 13.19% |
| frozen_v9 | 43.75% [36.25%, 51.25%] | 38.39% |

The isolated effects on the primary metric were:

| Question | Paired difference (95% CI) |
|---|---:|
| distribution alignment, B-A | 0.00% [0.00%, 0.00%] |
| metric alignment, C-B | 0.00% [0.00%, 0.00%] |
| structured action model, D-C | 0.00% [-2.50%, 2.50%] |

Selected forward: `frozen_v9`. The acceptance gates required a
three-point gain, positive paired lower bound versus A and frozen v9, ordinary
non-inferiority, and per-field protection.

## Direct inverse specialists

Numerical inverse development top-1 was
27.19% [22.81%, 31.56%] for the
direct multi-positive ranker versus
55.94% [50.47%, 61.56%] for
frozen v9. Candidate-minus-v9 was
-28.75% [-34.22%, -23.12%]; status:
`rejected`.

Visual inverse development top-1 was
10.83% [8.54%, 13.12%]
for the image-primary scorer versus
33.12% [28.54%, 37.92%]
for the existing direct visual scorer and
21.88% [18.12%, 25.62%]
for the measured-state route. Status:
`rejected`.

## Direction

Development strict all-five:

| Variant | Result |
|---|---:|
| frozen separate specialist | 36.01% [31.96%, 40.07%] |
| direct selected-forward thresholding | 35.47% [31.30%, 39.77%] |
| forward plus calibration head | 36.52% [33.02%, 40.16%] |

Selected direction: `frozen_separate_direction`.

## One-time locked test

| Task | Frozen selection result |
|---|---:|
| forward strict all-five natural action | 30.63% [23.75%, 38.12%] |
| numerical inverse top-1 physical success | 52.66% [47.50%, 58.28%] |
| visual inverse top-1 physical success | 34.38% [29.58%, 39.38%] |
| direction strict all-five full surface | 37.43% [33.38%, 41.54%] |

The locked test was evaluated exactly once after the freeze manifest was
written. No selection was changed afterward.

Locked per-output results:

| Output | Forward natural-action tolerance accuracy | Direction accuracy |
|---|---:|---:|
| centroid_x_px | 63.75% | 72.64% |
| centroid_y_px | 73.75% | 75.53% |
| sigma_x_px | 80.62% | 83.87% |
| sigma_y_px | 86.88% | 83.08% |
| peak_intensity | 60.62% | 72.63% |

Locked forward action-cardinality breakdown:

| Moving actuators | Full-surface strict all-five |
|---:|---:|
| 0 | 100.00% [100.00%, 100.00%] |
| 1 | 59.69% [54.92%, 64.06%] |
| 2 | 42.29% [37.47%, 47.01%] |
| 3 | 31.41% [26.58%, 36.74%] |
| 4 | 25.20% [20.51%, 30.35%] |

Locked forward natural-request regime breakdown:

| Regime | Strict all-five |
|---|---:|
| camera_boundary | 0.00% [0.00%, 0.00%] |
| clipping | 8.33% [0.00%, 20.83%] |
| focusing | 45.83% [25.00%, 66.67%] |
| high_offset_interaction | 25.00% [6.25%, 50.00%] |
| ordinary | 51.79% [39.29%, 64.29%] |
| tolerance_boundary | 18.75% [0.00%, 37.50%] |

## Post-freeze system comparability

The existing 1,600-request/150-case-per-route system replay was run only after
the selection and locked result were sealed. It reproduced the accepted v9
system metrics exactly:

| Task | Frozen v9 correct-route | Frozen v9 end-to-end | Final v10 correct-route | Final v10 end-to-end |
|---|---:|---:|---:|---:|
| direction strict all-five | 54.00% | 54.00% | 54.00% | 54.00% |
| forward strict all-five | 61.67% | 60.00% | 61.67% | 60.00% |
| combined inverse physical success | 61.00% | 59.33% | 61.00% | 59.33% |
| measurement strict all-five | 76.67% | 76.67% | 76.67% | 76.67% |

Final v10 equals frozen v9 because every experimental replacement was
rejected before the freeze. The comparability replay was not used for
training, tuning, or selection. Qwen routing was not tuned and remains the
authoritative frozen 98.76%.

## Compute and safety

| Stage | Wall time | Peak GPU memory | Peak GPU temperature | Minimum available RAM | Guard |
|---|---:|---:|---:|---:|---|
| pilot finalization | 322.7 s | 400 MiB | 57 C | 9518 MiB | passed |
| forward A-D training | 20.1 s | 740 MiB | 58 C | 9104 MiB | passed |
| forward development evaluation | 10.1 s | 702 MiB | 54 C | 8703 MiB | passed |
| inverse/visual/direction training | 60.3 s | 868 MiB | 66 C | 7413 MiB | passed |
| freeze | 5.0 s | 400 MiB | 52 C | 10426 MiB | passed |
| locked evaluation | 45.2 s | 790 MiB | 54 C | 5652 MiB | passed |
| system comparability | 266.4 s | 702 MiB | 60 C | 6736 MiB | passed |

Completed guarded stages totaled 729.8 seconds. Every completed
production stage used CPU cores 0 and 8 and passed the GPU-temperature,
GPU-memory, and RAM guards. This total excludes the interrupted full-generation
attempt because it was stopped interactively before its wrapper could finalize
an audit; the manifest records its 678 completed training shards.

## Answer to the v10 question

The controlled B-A, C-B, and D-C intervals above are the evidence for
distribution, metric, and structure effects. None changed the primary
natural-request metric, and none had a paired interval excluding zero. D did
raise the secondary full 81-action surface score from 11.91% to 13.19%, but
that did not transfer to requested-action success and remained far below
frozen v9. Therefore this pilot does not support distribution mismatch,
objective mismatch, or insufficient action structure as the dominant cause
of the current specialist ceiling; it only suggests that explicit structure
may help the broader surface modestly.

The pilot size limits power for small effects, especially regime and
1/81-positive breakdowns. A full generation can resume from the preserved
deterministic shards without changing the locked pilot result. The new neural
inverse and visual rankers also underfit their frozen counterparts, so their
negative results should not be generalized into a claim that direct ranking
is intrinsically inferior.

## Artifacts

- dataset manifest: `/home/jiamo/VLM_data/physics_structured_rebuild_v10/manifest.json`
- experiment registry: `/home/jiamo/VLM_runs/physics_structured_rebuild_v10_full/experiment_registry_v10.json`
- freeze manifest: `/home/jiamo/VLM_runs/physics_structured_rebuild_v10_full/freeze_manifest_v10.json`
- locked evaluation: `/home/jiamo/VLM_runs/physics_structured_rebuild_v10_full/locked_test_once_v10.json`
- post-freeze system comparability: `/home/jiamo/VLM_runs/physics_structured_rebuild_v10_full/postfreeze_system_comparability_v10.json`
- forward plot: `/home/jiamo/VLM_runs/physics_structured_rebuild_v10_full/forward_primary_v10.png`
- breakdown table: `/home/jiamo/VLM_runs/physics_structured_rebuild_v10_full/forward_breakdowns_v10.csv`
- smoke-test reports: `/home/jiamo/VLM_runs/physics_structured_rebuild_v10_smoke`
- reproducible commands: `/home/jiamo/VLM/physics_structured_rebuild_v10/REPRODUCIBLE_COMMANDS.md`
