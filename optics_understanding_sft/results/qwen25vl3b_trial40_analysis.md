# Forty-step QLoRA trial

The trial trained Qwen2.5-VL-3B with NF4 QLoRA rank 8 for 40 optimizer steps using seed 42 and gradient accumulation 4. Checkpoints were saved at steps 20 and 40. No paid API was used.

## Selection result

Checkpoint 20 is selected by the frozen equal-task validation macro score.

| Model | Macro | JSON | Schema | Trainer validation loss |
|---|---:|---:|---:|---:|
| Base | 0.416 | 94.2% | 58.3% | n/a |
| Checkpoint 20 | **0.518** | **99.2%** | 90.8% | 0.3123 |
| Checkpoint 40 | 0.514 | 98.3% | **93.3%** | **0.2703** |

Checkpoint 20 improves the macro score by 0.102 absolute, or about 24.5% relative to the base score. Checkpoint 40 has the lower teacher-forced validation loss, demonstrating why loss was not used as the checkpoint-selection metric.

## Per-task comparison

| Task | Base | Step 20 | Step 40 |
|---|---:|---:|---:|
| Setup interpretation | 0.000 | 0.167 | **0.229** |
| Information sufficiency | 0.450 | **0.500** | 0.350 |
| Causal effects | 0.644 | **0.856** | 0.789 |
| Forward prediction | 0.278 | 0.431 | **0.444** |
| Diagnosis | 0.402 | 0.432 | **0.466** |
| Constrained intervention | 0.487 | **0.600** | **0.600** |
| Counterfactual reasoning | **0.648** | 0.639 | 0.722 |

The adapter substantially improves format compliance, causal classification, forward prediction, and constrained control. Setup interpretation remains weak, and step 20 slightly regresses counterfactual reasoning relative to the base score. These are important targets for the next data or training revision.

## Uncertainty and modality

A paired 10,000-replicate bootstrap over the 30 validation scenario groups gives a step-20-minus-step-40 macro difference of 0.0036 with a 95% interval from -0.0303 to 0.0407. Step 20 wins 57.1% of replicates. The two checkpoints should therefore be treated as effectively close; step 20 is selected because the protocol was fixed in advance, not because the observed margin is conclusive.

Step 20 scores 0.540 on text and 0.438 on visual records. Step 40 scores 0.522 on text and 0.576 on visual records. With only 12 visual validation records and different task composition, this is not enough evidence to select a visual-specialized checkpoint instead of the frozen overall metric.

## Frozen selection

Selected adapter:

`/home/jiamo/VLM_runs/qwen25vl_3b_qlora_optics_understanding_trial40_v1/checkpoint-20`

All 120 validation records were used for checkpoint scoring. No test record was evaluated and no test prompt was used for training, inference, or checkpoint selection.
