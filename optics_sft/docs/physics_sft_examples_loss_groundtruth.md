# Physics-Aware Optics SFT Examples, Loss, and Ground Truth

This document explains how the physics-aware optics SFT rows are used for supervised fine-tuning, what the model sees, what is treated as ground truth, and what is reserved for evaluation.

The examples below are small illustrative rows. They are not generated datasets and do not require model training.

## 1. High-Level SFT Learning Rule

For every physics-aware SFT row, the training script converts:

```text
prompt_inputs + images
    -> user prompt

target
    -> assistant completion / ground-truth output
```

The training objective is:

```text
P(target JSON | prompt_inputs, images)
```

In plain terms, the model sees only the prompt-visible inputs and the referenced images, then it is trained to emit the JSON object stored in `row["target"]`.

Important details:

- The SFT loss is token-level cross-entropy on the assistant target JSON tokens.
- With `completion_only_loss: true`, prompt/user tokens are masked out by the trainer/collator path, so the model is supervised on the assistant completion rather than on the prompt text.
- The simulator is not directly backpropagated through during SFT.
- The simulator provides ground-truth labels and evaluation checks.
- The model is not directly minimizing beam residual during SFT unless a separate custom loss is added later.
- During SFT, numeric fields such as `lens_x_delta_mm` are learned as text tokens inside serialized JSON, not through a direct numeric MSE loss.

## 2. Example: `inverse_control`

Concrete example:

```json
{
  "sample_id": "toy_inverse_000",
  "sample_type": "inverse_control",
  "prompt_inputs": {
    "images": {
      "current_image_path": "toy/inverse_current.png",
      "target_image_path": "toy/inverse_target.png"
    },
    "safe_setup_metadata": {
      "wavelength_nm": 632.8,
      "beam_waist_mm": 1.0,
      "power_w": 1.0,
      "lens_focal_length_mm": 100.0,
      "lens_aperture_mm": 25.4,
      "source_to_lens_mm": 200.0,
      "lens_to_camera_mm": 150.0,
      "sensor_resolution": [1024, 1024],
      "pixel_size_um": 5.5,
      "coordinate_convention": "sensor pixel x increases right; y increases down",
      "propagation_backend": "fresnel_numpy",
      "grid_size": 1024,
      "grid_extent_mm": 30.0
    }
  },
  "target": {
    "task": "physics_aware_beam_alignment",
    "perception": {
      "current_relative_to_target_x": "right",
      "current_relative_to_target_y": "above"
    },
    "physics_reasoning_summary": {
      "relevant_parameters_used": ["lens_focal_length_mm", "lens_to_camera_mm"],
      "control_coupling_summary": "Lens offsets move the centroid on the sensor."
    },
    "control_plan": {
      "lens_x_delta_mm": -0.012,
      "lens_y_delta_mm": 0.018,
      "camera_x_delta_mm": 0.0,
      "camera_y_delta_mm": 0.0
    },
    "prediction": {
      "expected_residual_error_px": 1.2
    },
    "confidence": 0.74
  },
  "private_eval": {
    "current_state": {
      "centroid_x_px": 514.0,
      "centroid_y_px": 507.0
    },
    "target_state": {
      "centroid_x_px": 512.0,
      "centroid_y_px": 511.0
    },
    "true_control_plan": {
      "lens_x_delta_mm": -0.012,
      "lens_y_delta_mm": 0.018,
      "camera_x_delta_mm": 0.0,
      "camera_y_delta_mm": 0.0
    },
    "initial_error_px": 4.47,
    "post_action_error_px": 1.2
  },
  "split_tags": ["toy", "inverse_control"]
}
```

What the model sees:

- `current_image_path`
- `target_image_path`
- `safe_setup_metadata`
- the inverse-control prompt template

Ground truth for SFT:

- `row["target"]`
- Most importantly, `target.control_plan`
- Also the structured target fields such as `task`, `perception`, `physics_reasoning_summary`, `prediction`, and `confidence` when they are present

Ground truth for evaluation:

- `private_eval.true_control_plan`
- `private_eval.current_state`
- `private_eval.target_state`
- residual fields such as `initial_error_px` and `post_action_error_px`

Loss behavior:

- The trainer serializes `row["target"]` to JSON text.
- Cross-entropy is computed on the assistant JSON tokens.
- There is no direct MSE loss on `lens_x_delta_mm` or `lens_y_delta_mm`.
- Simulator-in-loop evaluation later parses the predicted `control_plan`, applies it to the reconstructed setup, simulates the new beam state, and checks post-action residual.

## 3. Example: `forward_transition`

Concrete example:

```json
{
  "sample_id": "toy_forward_000",
  "sample_type": "forward_transition",
  "prompt_inputs": {
    "images": {
      "before_image_path": "toy/forward_before.png"
    },
    "safe_setup_metadata": {
      "wavelength_nm": 632.8,
      "beam_waist_mm": 1.0,
      "power_w": 1.0,
      "lens_focal_length_mm": 100.0,
      "lens_aperture_mm": 25.4,
      "source_to_lens_mm": 200.0,
      "lens_to_camera_mm": 150.0,
      "sensor_resolution": [1024, 1024],
      "pixel_size_um": 5.5,
      "propagation_backend": "fresnel_numpy",
      "grid_size": 1024,
      "grid_extent_mm": 30.0
    },
    "action": {
      "lens_x_delta_mm": -0.01,
      "lens_y_delta_mm": 0.02,
      "camera_x_delta_mm": 0.0,
      "camera_y_delta_mm": 0.0
    }
  },
  "target": {
    "task": "forward_transition_prediction",
    "predicted_after_state": {
      "centroid_x_px": 511.8,
      "centroid_y_px": 510.6,
      "sigma_x_px": 95.4,
      "sigma_y_px": 98.2,
      "peak_intensity": 0.86
    },
    "predicted_change": {
      "centroid_shift_px": {
        "x": -2.2,
        "y": 3.6
      },
      "sigma_change_px": {
        "x": -0.6,
        "y": 0.2
      },
      "peak_intensity_change": 0.04
    }
  },
  "private_eval": {
    "before_state": {
      "centroid_x_px": 514.0,
      "centroid_y_px": 507.0,
      "sigma_x_px": 96.0,
      "sigma_y_px": 98.0,
      "peak_intensity": 0.82
    },
    "after_state": {
      "centroid_x_px": 511.8,
      "centroid_y_px": 510.6,
      "sigma_x_px": 95.4,
      "sigma_y_px": 98.2,
      "peak_intensity": 0.86
    },
    "simulator_config": {
      "source": "toy"
    }
  },
  "split_tags": ["toy", "forward_transition"]
}
```

What the model sees:

- before image
- safe setup metadata
- candidate action
- the forward-transition prompt template

Ground truth for SFT:

- `target.predicted_after_state`
- `target.predicted_change`
- any other fields inside `row["target"]`

Ground truth for evaluation:

- `private_eval.after_state`
- `private_eval.before_state` for true delta calculations

Loss behavior:

- Loss is cross-entropy on the JSON text containing the after-state and change values.
- The forward evaluator parses `predicted_after_state` and compares it against `private_eval.after_state`.
- Metrics include centroid, sigma, and peak-intensity errors.

## 4. Example: `counterfactual_pair`

Concrete example:

```json
{
  "sample_id": "toy_counterfactual_000",
  "sample_type": "counterfactual_pair",
  "prompt_inputs": {
    "images": {
      "scenario_a_current_image_path": "toy/cf_a_current.png",
      "scenario_a_target_image_path": "toy/cf_a_target.png",
      "scenario_b_current_image_path": "toy/cf_b_current.png",
      "scenario_b_target_image_path": "toy/cf_b_target.png"
    },
    "safe_setup_metadata": {
      "scenario_a": {
        "wavelength_nm": 632.8,
        "beam_waist_mm": 1.0,
        "power_w": 1.0,
        "lens_focal_length_mm": 100.0,
        "lens_aperture_mm": 25.4,
        "source_to_lens_mm": 200.0,
        "lens_to_camera_mm": 150.0,
        "sensor_resolution": [1024, 1024],
        "pixel_size_um": 5.5
      },
      "scenario_b": {
        "wavelength_nm": 632.8,
        "beam_waist_mm": 1.0,
        "power_w": 1.0,
        "lens_focal_length_mm": 100.0,
        "lens_aperture_mm": 25.4,
        "source_to_lens_mm": 200.0,
        "lens_to_camera_mm": 210.0,
        "sensor_resolution": [1024, 1024],
        "pixel_size_um": 5.5
      }
    }
  },
  "target": {
    "task": "counterfactual_action_comparison",
    "changed_parameter": "lens_to_camera_mm",
    "should_action_change": true,
    "scenario_a_control_plan": {
      "lens_x_delta_mm": -0.01,
      "lens_y_delta_mm": 0.014,
      "camera_x_delta_mm": 0.0,
      "camera_y_delta_mm": 0.0
    },
    "scenario_b_control_plan": {
      "lens_x_delta_mm": -0.007,
      "lens_y_delta_mm": 0.01,
      "camera_x_delta_mm": 0.0,
      "camera_y_delta_mm": 0.0
    },
    "action_difference_summary": {
      "lens_x_difference_mm": 0.003,
      "lens_y_difference_mm": -0.004
    }
  },
  "private_eval": {
    "scenario_a": {
      "current_state": {
        "centroid_x_px": 515.0,
        "centroid_y_px": 508.0
      },
      "target_state": {
        "centroid_x_px": 512.0,
        "centroid_y_px": 511.0
      },
      "searched_control_action": {
        "lens_x_delta_mm": -0.01,
        "lens_y_delta_mm": 0.014,
        "camera_x_delta_mm": 0.0,
        "camera_y_delta_mm": 0.0
      }
    },
    "scenario_b": {
      "current_state": {
        "centroid_x_px": 515.2,
        "centroid_y_px": 508.1
      },
      "target_state": {
        "centroid_x_px": 512.1,
        "centroid_y_px": 510.9
      },
      "searched_control_action": {
        "lens_x_delta_mm": -0.007,
        "lens_y_delta_mm": 0.01,
        "camera_x_delta_mm": 0.0,
        "camera_y_delta_mm": 0.0
      }
    }
  },
  "split_tags": ["toy", "counterfactual_pair", "ood:lens_to_camera_mm"]
}
```

What the model sees:

- scenario A current image
- scenario A target image
- scenario B current image
- scenario B target image
- safe setup metadata for both scenarios
- the counterfactual prompt template

Ground truth for SFT:

- `row["target"]`
- `target.changed_parameter`
- `target.should_action_change`
- `target.scenario_a_control_plan`
- `target.scenario_b_control_plan`
- `target.action_difference_summary`

Ground truth for evaluation:

- `target` comparison labels
- `private_eval.scenario_a` true states/actions
- `private_eval.scenario_b` true states/actions

Loss behavior:

- Loss is still token cross-entropy on serialized `row["target"]`.
- Evaluation checks whether the model identifies the changed parameter and changes action when the physical change requires it.

## 5. Example: `trajectory`

Concrete example:

```json
{
  "sample_id": "toy_trajectory_000",
  "sample_type": "trajectory",
  "prompt_inputs": {
    "images": {
      "initial_image_path": "toy/traj_initial.png",
      "target_image_path": "toy/traj_target.png"
    },
    "safe_setup_metadata": {
      "wavelength_nm": 632.8,
      "beam_waist_mm": 1.0,
      "power_w": 1.0,
      "lens_focal_length_mm": 100.0,
      "lens_aperture_mm": 25.4,
      "source_to_lens_mm": 200.0,
      "lens_to_camera_mm": 150.0,
      "sensor_resolution": [1024, 1024],
      "pixel_size_um": 5.5
    }
  },
  "target": {
    "task": "closed_loop_alignment_plan",
    "recommended_steps": [
      {
        "step_index": 0,
        "action": {
          "lens_x_delta_mm": -0.01,
          "lens_y_delta_mm": 0.018,
          "camera_x_delta_mm": 0.0,
          "camera_y_delta_mm": 0.0
        },
        "expected_residual_after_px": 2.4
      },
      {
        "step_index": 1,
        "action": {
          "lens_x_delta_mm": -0.003,
          "lens_y_delta_mm": 0.006,
          "camera_x_delta_mm": 0.0,
          "camera_y_delta_mm": 0.0
        },
        "expected_residual_after_px": 0.8
      }
    ]
  },
  "private_eval": {
    "all_actions": [
      {
        "lens_x_delta_mm": -0.01,
        "lens_y_delta_mm": 0.018,
        "camera_x_delta_mm": 0.0,
        "camera_y_delta_mm": 0.0
      },
      {
        "lens_x_delta_mm": -0.003,
        "lens_y_delta_mm": 0.006,
        "camera_x_delta_mm": 0.0,
        "camera_y_delta_mm": 0.0
      }
    ],
    "all_true_states": [
      {
        "step_index": 0,
        "residual_before_px": 5.1,
        "residual_after_px": 2.4
      },
      {
        "step_index": 1,
        "residual_before_px": 2.4,
        "residual_after_px": 0.8
      }
    ],
    "residuals": {
      "final_px": 0.8,
      "success": true
    }
  },
  "split_tags": ["toy", "trajectory"]
}
```

What the model sees:

- initial image
- target image
- safe setup metadata
- the trajectory prompt template
- optionally step images if the row includes them and the caller requests them; the default SFT path uses initial + target

Ground truth for SFT:

- `target.recommended_steps`
- each step action
- expected residuals after each step
- any other fields in `row["target"]`

Ground truth for evaluation:

- `private_eval.all_actions`
- `private_eval.all_true_states`
- `private_eval.residuals`
- simulator configs when available

Loss behavior:

- Loss is cross-entropy over the serialized multi-step JSON answer.
- Closed-loop evaluation is more important than exact token/action matching because a different first action can still reduce residual and succeed.

## 6. Where Ground Truth Lives

| Data type | SFT ground truth field | Evaluation ground truth field | What the model sees | What the model must not see |
|---|---|---|---|---|
| `inverse_control` | Usually `row["target"]`, especially `target.control_plan` | `private_eval.true_control_plan`, `private_eval.current_state`, `private_eval.target_state`, residuals, simulator config | `prompt_inputs.images.current_image_path`, `prompt_inputs.images.target_image_path`, `prompt_inputs.safe_setup_metadata`, inverse prompt | `private_eval`, centroids, target state, true control plan, residuals, answer labels |
| `forward_transition` | Usually `row["target"]`, especially `target.predicted_after_state` and `target.predicted_change` | `private_eval.before_state`, `private_eval.after_state`, simulator config | `prompt_inputs.images.before_image_path`, `prompt_inputs.safe_setup_metadata`, `prompt_inputs.action`, forward prompt | `private_eval.after_state`, hidden measured/simulated after-state values, answer labels |
| `counterfactual_pair` | Usually `row["target"]`, including changed parameter, action-change decision, and scenario control plans | `target` plus `private_eval.scenario_a` / `private_eval.scenario_b` states and true actions | scenario A/B image paths, scenario A/B safe setup metadata, counterfactual prompt | private states, searched actions if not in target, residuals, simulator configs |
| `trajectory` | Usually `row["target"]`, especially `target.recommended_steps` | `private_eval.all_actions`, `private_eval.all_true_states`, `private_eval.residuals`, simulator configs | initial image, target image, safe setup metadata, trajectory prompt | full true trajectory states, residuals, private simulator details |

Rules:

- SFT ground truth is usually `row["target"]`.
- `private_eval` is not shown to the model.
- `private_eval` is used for metrics, simulator reconstruction, residuals, and sanity checks.
- `prompt_inputs` is what the model sees.
- Answer-leaking fields like centroids, `true_control_plan`, residuals, `target_state`, and `after_state` should not appear in `prompt_inputs` unless they are intentionally part of a future task input.

## 7. How Loss Is Computed in Code

Actual code path in `optics_sft/scripts/train_qwen25vl_qlora.py`:

- `train(...)` reads `data.dataset_format`; if absent, it defaults to `legacy_pair`.
- For `physics_mixed`, it calls `physics_row_to_sft_example(row, image_root)`.
- `physics_row_to_sft_example(...)` calls `build_physics_prompt(row)`.
- It calls `expected_image_slots(row)` to determine the image order.
- It loads each expected image with `load_rgb_image(resolve_image_path(image_root, slot["path"]))`.
- It serializes the assistant answer with `json.dumps(row["target"], sort_keys=True)`.
- It returns an example with:
  - `images`
  - `prompt`: user message containing image placeholders plus prompt text
  - `completion`: assistant message containing the serialized target JSON
- `train(...)` converts examples into a Hugging Face `Dataset`.
- `build_sft_config(...)` passes `completion_only_loss` and `assistant_only_loss` from YAML into TRL `SFTConfig`.
- `SFTTrainer` receives the train/eval datasets and the QLoRA model.

Simplified pseudo-code:

```python
row = json.loads(line)

prompt = build_physics_prompt(row)
image_slots = expected_image_slots(row)
images = [
    load_rgb_image(resolve_image_path(image_root, slot["path"]))
    for slot in image_slots
]

completion = json.dumps(row["target"], sort_keys=True)

example = {
    "prompt": user_message_with_images_and_text(prompt, len(images)),
    "completion": assistant_message_with_target_json(completion),
    "images": images,
}

loss = CrossEntropy(
    logits_for_completion_tokens,
    target_completion_token_ids,
)
```

Expected masking behavior:

- Prompt tokens should have `label = -100` or equivalent masking when `completion_only_loss=True`.
- Completion tokens are supervised.
- LoRA adapter weights are updated by backprop.
- The base model is mostly frozen/quantized under QLoRA.

Important mismatch to know:

- The prompt templates describe rich JSON shapes.
- The trainer supervises exactly `row["target"]`.
- Some current generators emit compact targets that do not include every field named in the prompt template. This is not a loader bug, but it means prompt-template wording and generator target shapes should be aligned before a large training run.

## 8. Quick Sanity Check Commands

Run syntax validation:

```bash
python -m compileall optics_sft optical_sim
```

Run prompt and leakage smoke checks:

```bash
python optics_sft/scripts/smoke_check_physics_prompts.py
python optics_sft/scripts/smoke_check_metadata_policy.py
```

Generate a tiny forward-transition dataset only:

```bash
python optics_sft/scripts/generate_physics_transition_dataset.py \
  --output-dir ../VLM_data/physics_sft_transition_doc_smoke \
  --num-samples 4 \
  --seed 1
```

Inspect toy SFT examples without model loading:

```bash
python optics_sft/scripts/inspect_physics_sft_examples.py --use-toy-examples
```

Inspect rows from a JSONL file:

```bash
python optics_sft/scripts/inspect_physics_sft_examples.py \
  --input-jsonl ../VLM_data/physics_sft_transition_doc_smoke/train.jsonl \
  --max-rows 2
```

These commands do not train, do not load the full model, and do not require a GPU.
