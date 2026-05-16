# Stage 1E Qualitative Case Studies

## Scope

This report shows representative Stage-1 understanding cases for senior review. Cases are selected from rows with available fine-tuned LLM predictions and local model predictions.

## Inputs

- Stage-1 benchmark: `profile2setup/data/stage1_understanding/stage1_understanding_80.jsonl`
- Stage-1 labels: `profile2setup/data/stage1_understanding/stage1_understanding_labels.jsonl`
- LLM predictions requested: `profile2setup/results/stage1_understanding/finetuned_llm_predictions.jsonl`
- LLM predictions used: `profile2setup/results/stage1_understanding/finetuned_llm_predictions_25.jsonl`
- Local understanding proxy: `profile2setup/results/stage1_understanding/local_understanding_proxy.json`
- Local model eval: `profile2setup/results/stage1_understanding/local_model_eval.json`

## Coverage

- LLM prediction rows available: `25`
- Candidate cases: `25`
- Selected cases: `12`

## Warnings

- Primary LLM predictions missing: profile2setup/results/stage1_understanding/finetuned_llm_predictions.jsonl; using fallback profile2setup/results/stage1_understanding/finetuned_llm_predictions_25.jsonl.

## Case Studies

### 1. LLM clearly better on understanding

- record_id: `stage1_constraint_005__edit_rand_03223__rand_02918__4422`
- category: `constraint`
- task_type: `edit`
- prompt: increase source_to_lens and keep lens_to_camera fixed

Images:

- current_profile: [images/finetuned_llm_25/000014_stage1_constraint_005__edit_rand_03223__rand_02918__4422/current_profile.png](images/finetuned_llm_25/000014_stage1_constraint_005__edit_rand_03223__rand_02918__4422/current_profile.png)
- target_profile: [images/finetuned_llm_25/000014_stage1_constraint_005__edit_rand_03223__rand_02918__4422/target_profile.png](images/finetuned_llm_25/000014_stage1_constraint_005__edit_rand_03223__rand_02918__4422/target_profile.png)
- difference_profile: [images/finetuned_llm_25/000014_stage1_constraint_005__edit_rand_03223__rand_02918__4422/difference_profile.png](images/finetuned_llm_25/000014_stage1_constraint_005__edit_rand_03223__rand_02918__4422/difference_profile.png)
- composite_profile: [images/finetuned_llm_25/000014_stage1_constraint_005__edit_rand_03223__rand_02918__4422/composite_profile.png](images/finetuned_llm_25/000014_stage1_constraint_005__edit_rand_03223__rand_02918__4422/composite_profile.png)

- Ground truth changed variables: `['source_to_lens']`
- Ground truth directions: `source_to_lens: increase, lens_to_camera: unchanged, focal_length: unchanged, lens_x: unchanged, lens_y: unchanged, camera_x: unchanged, camera_y: unchanged`
- LLM predicted changed variables: `['source_to_lens', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- LLM predicted directions: `source_to_lens: increase, lens_to_camera: unchanged, focal_length: decrease, lens_x: increase, lens_y: increase, camera_x: increase, camera_y: increase`
- LLM reasoning_summary: Predicted setup fields are supervised labels from the existing profile2setup record.
- Local inferred changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- Local inferred directions: `source_to_lens: decrease, lens_to_camera: decrease, focal_length: decrease, lens_x: decrease, lens_y: decrease, camera_x: increase, camera_y: increase`
- LLM setup error, mean absolute over 7 variables: `0.089447`
- Local setup error, mean absolute over 7 variables: `0.0263183`
- LLM direction accuracy: `0.285714`
- Local direction accuracy: `0`
- Interpretation: The LLM matches the labeled direction pattern better, while the local model's inferred delta changes extra or wrong variables.

### 2. LLM clearly better on understanding

- record_id: `stage1_constraint_004__edit_rand_02683__rand_03128__9891`
- category: `constraint`
- task_type: `edit`
- prompt: decrease lens_y but keep the camera fixed

Images:

- current_profile: [images/finetuned_llm_25/000013_stage1_constraint_004__edit_rand_02683__rand_03128__9891/current_profile.png](images/finetuned_llm_25/000013_stage1_constraint_004__edit_rand_02683__rand_03128__9891/current_profile.png)
- target_profile: [images/finetuned_llm_25/000013_stage1_constraint_004__edit_rand_02683__rand_03128__9891/target_profile.png](images/finetuned_llm_25/000013_stage1_constraint_004__edit_rand_02683__rand_03128__9891/target_profile.png)
- difference_profile: [images/finetuned_llm_25/000013_stage1_constraint_004__edit_rand_02683__rand_03128__9891/difference_profile.png](images/finetuned_llm_25/000013_stage1_constraint_004__edit_rand_02683__rand_03128__9891/difference_profile.png)
- composite_profile: [images/finetuned_llm_25/000013_stage1_constraint_004__edit_rand_02683__rand_03128__9891/composite_profile.png](images/finetuned_llm_25/000013_stage1_constraint_004__edit_rand_02683__rand_03128__9891/composite_profile.png)

- Ground truth changed variables: `['lens_y']`
- Ground truth directions: `source_to_lens: unchanged, lens_to_camera: unchanged, focal_length: unchanged, lens_x: unchanged, lens_y: decrease, camera_x: unchanged, camera_y: unchanged`
- LLM predicted changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- LLM predicted directions: `source_to_lens: increase, lens_to_camera: decrease, focal_length: increase, lens_x: increase, lens_y: decrease, camera_x: increase, camera_y: increase`
- LLM reasoning_summary: Predicted setup fields are supervised labels from the existing profile2setup record.
- Local inferred changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- Local inferred directions: `source_to_lens: increase, lens_to_camera: decrease, focal_length: increase, lens_x: increase, lens_y: increase, camera_x: increase, camera_y: increase`
- LLM setup error, mean absolute over 7 variables: `0.0215901`
- Local setup error, mean absolute over 7 variables: `0.0295253`
- LLM direction accuracy: `0.142857`
- Local direction accuracy: `0`
- Interpretation: The LLM matches the labeled direction pattern better, while the local model's inferred delta changes extra or wrong variables.

### 3. Local clearly better numerically

- record_id: `stage1_normal_edit_002__edit_rand_01218__rand_00301__6906`
- category: `normal_edit`
- task_type: `edit`
- prompt: move the beam right and move the beam up

Images:

- current_profile: [images/finetuned_llm_25/000001_stage1_normal_edit_002__edit_rand_01218__rand_00301__6906/current_profile.png](images/finetuned_llm_25/000001_stage1_normal_edit_002__edit_rand_01218__rand_00301__6906/current_profile.png)
- target_profile: [images/finetuned_llm_25/000001_stage1_normal_edit_002__edit_rand_01218__rand_00301__6906/target_profile.png](images/finetuned_llm_25/000001_stage1_normal_edit_002__edit_rand_01218__rand_00301__6906/target_profile.png)
- difference_profile: [images/finetuned_llm_25/000001_stage1_normal_edit_002__edit_rand_01218__rand_00301__6906/difference_profile.png](images/finetuned_llm_25/000001_stage1_normal_edit_002__edit_rand_01218__rand_00301__6906/difference_profile.png)
- composite_profile: [images/finetuned_llm_25/000001_stage1_normal_edit_002__edit_rand_01218__rand_00301__6906/composite_profile.png](images/finetuned_llm_25/000001_stage1_normal_edit_002__edit_rand_01218__rand_00301__6906/composite_profile.png)

- Ground truth changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- Ground truth directions: `source_to_lens: increase, lens_to_camera: decrease, focal_length: decrease, lens_x: decrease, lens_y: increase, camera_x: increase, camera_y: increase`
- LLM predicted changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- LLM predicted directions: `source_to_lens: increase, lens_to_camera: increase, focal_length: decrease, lens_x: decrease, lens_y: increase, camera_x: increase, camera_y: increase`
- LLM reasoning_summary: Predicted setup fields are supervised labels from the existing profile2setup record.
- Local inferred changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- Local inferred directions: `source_to_lens: increase, lens_to_camera: decrease, focal_length: decrease, lens_x: decrease, lens_y: increase, camera_x: increase, camera_y: increase`
- LLM setup error, mean absolute over 7 variables: `0.0819253`
- Local setup error, mean absolute over 7 variables: `0.0338712`
- LLM direction accuracy: `0.857143`
- Local direction accuracy: `1`
- Interpretation: The local delta-derived direction pattern is closer to the label than the LLM understanding JSON on this example.

### 4. Local clearly better numerically

- record_id: `stage1_normal_edit_006__edit_rand_01092__rand_04208__2068`
- category: `normal_edit`
- task_type: `edit`
- prompt: move the beam left and move the beam up

Images:

- current_profile: [images/finetuned_llm_25/000005_stage1_normal_edit_006__edit_rand_01092__rand_04208__2068/current_profile.png](images/finetuned_llm_25/000005_stage1_normal_edit_006__edit_rand_01092__rand_04208__2068/current_profile.png)
- target_profile: [images/finetuned_llm_25/000005_stage1_normal_edit_006__edit_rand_01092__rand_04208__2068/target_profile.png](images/finetuned_llm_25/000005_stage1_normal_edit_006__edit_rand_01092__rand_04208__2068/target_profile.png)
- difference_profile: [images/finetuned_llm_25/000005_stage1_normal_edit_006__edit_rand_01092__rand_04208__2068/difference_profile.png](images/finetuned_llm_25/000005_stage1_normal_edit_006__edit_rand_01092__rand_04208__2068/difference_profile.png)
- composite_profile: [images/finetuned_llm_25/000005_stage1_normal_edit_006__edit_rand_01092__rand_04208__2068/composite_profile.png](images/finetuned_llm_25/000005_stage1_normal_edit_006__edit_rand_01092__rand_04208__2068/composite_profile.png)

- Ground truth changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- Ground truth directions: `source_to_lens: increase, lens_to_camera: increase, focal_length: increase, lens_x: increase, lens_y: increase, camera_x: decrease, camera_y: increase`
- LLM predicted changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- LLM predicted directions: `source_to_lens: decrease, lens_to_camera: decrease, focal_length: decrease, lens_x: increase, lens_y: increase, camera_x: decrease, camera_y: increase`
- LLM reasoning_summary: Predicted setup fields are supervised labels from the existing profile2setup record.
- Local inferred changed variables: `['source_to_lens', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- Local inferred directions: `source_to_lens: decrease, lens_to_camera: unchanged, focal_length: decrease, lens_x: increase, lens_y: increase, camera_x: decrease, camera_y: increase`
- LLM setup error, mean absolute over 7 variables: `0.061168`
- Local setup error, mean absolute over 7 variables: `0.0227501`
- LLM direction accuracy: `0.571429`
- Local direction accuracy: `0.571429`
- Interpretation: The local numerical setup is closer to ground truth even if the qualitative change labels are mixed.

### 5. Both correct

- record_id: `stage1_normal_edit_004__edit_rand_02627__rand_02822__1139`
- category: `normal_edit`
- task_type: `edit`
- prompt: move the beam left and move the beam up

Images:

- current_profile: [images/finetuned_llm_25/000003_stage1_normal_edit_004__edit_rand_02627__rand_02822__1139/current_profile.png](images/finetuned_llm_25/000003_stage1_normal_edit_004__edit_rand_02627__rand_02822__1139/current_profile.png)
- target_profile: [images/finetuned_llm_25/000003_stage1_normal_edit_004__edit_rand_02627__rand_02822__1139/target_profile.png](images/finetuned_llm_25/000003_stage1_normal_edit_004__edit_rand_02627__rand_02822__1139/target_profile.png)
- difference_profile: [images/finetuned_llm_25/000003_stage1_normal_edit_004__edit_rand_02627__rand_02822__1139/difference_profile.png](images/finetuned_llm_25/000003_stage1_normal_edit_004__edit_rand_02627__rand_02822__1139/difference_profile.png)
- composite_profile: [images/finetuned_llm_25/000003_stage1_normal_edit_004__edit_rand_02627__rand_02822__1139/composite_profile.png](images/finetuned_llm_25/000003_stage1_normal_edit_004__edit_rand_02627__rand_02822__1139/composite_profile.png)

- Ground truth changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- Ground truth directions: `source_to_lens: increase, lens_to_camera: decrease, focal_length: increase, lens_x: decrease, lens_y: decrease, camera_x: decrease, camera_y: increase`
- LLM predicted changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- LLM predicted directions: `source_to_lens: decrease, lens_to_camera: decrease, focal_length: increase, lens_x: decrease, lens_y: decrease, camera_x: decrease, camera_y: increase`
- LLM reasoning_summary: Predicted setup fields are supervised labels from the existing profile2setup record.
- Local inferred changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- Local inferred directions: `source_to_lens: decrease, lens_to_camera: decrease, focal_length: increase, lens_x: decrease, lens_y: decrease, camera_x: decrease, camera_y: increase`
- LLM setup error, mean absolute over 7 variables: `0.0452623`
- Local setup error, mean absolute over 7 variables: `0.0348814`
- LLM direction accuracy: `0.857143`
- Local direction accuracy: `0.857143`
- Interpretation: The local numerical setup is closer to ground truth even if the qualitative change labels are mixed.

### 6. Both correct

- record_id: `stage1_normal_edit_007__edit_rand_02847__rand_04615__8565`
- category: `normal_edit`
- task_type: `edit`
- prompt: move the beam left and make it wider

Images:

- current_profile: [images/finetuned_llm_25/000006_stage1_normal_edit_007__edit_rand_02847__rand_04615__8565/current_profile.png](images/finetuned_llm_25/000006_stage1_normal_edit_007__edit_rand_02847__rand_04615__8565/current_profile.png)
- target_profile: [images/finetuned_llm_25/000006_stage1_normal_edit_007__edit_rand_02847__rand_04615__8565/target_profile.png](images/finetuned_llm_25/000006_stage1_normal_edit_007__edit_rand_02847__rand_04615__8565/target_profile.png)
- difference_profile: [images/finetuned_llm_25/000006_stage1_normal_edit_007__edit_rand_02847__rand_04615__8565/difference_profile.png](images/finetuned_llm_25/000006_stage1_normal_edit_007__edit_rand_02847__rand_04615__8565/difference_profile.png)
- composite_profile: [images/finetuned_llm_25/000006_stage1_normal_edit_007__edit_rand_02847__rand_04615__8565/composite_profile.png](images/finetuned_llm_25/000006_stage1_normal_edit_007__edit_rand_02847__rand_04615__8565/composite_profile.png)

- Ground truth changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- Ground truth directions: `source_to_lens: decrease, lens_to_camera: increase, focal_length: decrease, lens_x: increase, lens_y: increase, camera_x: decrease, camera_y: increase`
- LLM predicted changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- LLM predicted directions: `source_to_lens: decrease, lens_to_camera: increase, focal_length: decrease, lens_x: increase, lens_y: increase, camera_x: decrease, camera_y: increase`
- LLM reasoning_summary: Predicted setup fields are supervised labels from the existing profile2setup record.
- Local inferred changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- Local inferred directions: `source_to_lens: decrease, lens_to_camera: increase, focal_length: decrease, lens_x: increase, lens_y: decrease, camera_x: decrease, camera_y: increase`
- LLM setup error, mean absolute over 7 variables: `0.0432506`
- Local setup error, mean absolute over 7 variables: `0.0338606`
- LLM direction accuracy: `1`
- Local direction accuracy: `0.857143`
- Interpretation: The LLM matches the labeled direction pattern better, while the local model's inferred delta changes extra or wrong variables.

### 7. Both fail

- record_id: `stage1_normal_edit_001__edit_rand_03833__rand_00917__4871`
- category: `normal_edit`
- task_type: `edit`
- prompt: move the beam left and move the beam down

Images:

- current_profile: [images/finetuned_llm_25/000000_stage1_normal_edit_001__edit_rand_03833__rand_00917__4871/current_profile.png](images/finetuned_llm_25/000000_stage1_normal_edit_001__edit_rand_03833__rand_00917__4871/current_profile.png)
- target_profile: [images/finetuned_llm_25/000000_stage1_normal_edit_001__edit_rand_03833__rand_00917__4871/target_profile.png](images/finetuned_llm_25/000000_stage1_normal_edit_001__edit_rand_03833__rand_00917__4871/target_profile.png)
- difference_profile: [images/finetuned_llm_25/000000_stage1_normal_edit_001__edit_rand_03833__rand_00917__4871/difference_profile.png](images/finetuned_llm_25/000000_stage1_normal_edit_001__edit_rand_03833__rand_00917__4871/difference_profile.png)
- composite_profile: [images/finetuned_llm_25/000000_stage1_normal_edit_001__edit_rand_03833__rand_00917__4871/composite_profile.png](images/finetuned_llm_25/000000_stage1_normal_edit_001__edit_rand_03833__rand_00917__4871/composite_profile.png)

- Ground truth changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_y']`
- Ground truth directions: `source_to_lens: decrease, lens_to_camera: decrease, focal_length: decrease, lens_x: decrease, lens_y: increase, camera_x: unchanged, camera_y: decrease`
- LLM predicted changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- LLM predicted directions: `source_to_lens: increase, lens_to_camera: increase, focal_length: decrease, lens_x: increase, lens_y: decrease, camera_x: decrease, camera_y: decrease`
- LLM reasoning_summary: Predicted setup fields are supervised labels from the existing profile2setup record.
- Local inferred changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'camera_x', 'camera_y']`
- Local inferred directions: `source_to_lens: decrease, lens_to_camera: increase, focal_length: decrease, lens_x: decrease, lens_y: unchanged, camera_x: decrease, camera_y: decrease`
- LLM setup error, mean absolute over 7 variables: `0.0482143`
- Local setup error, mean absolute over 7 variables: `0.0349492`
- LLM direction accuracy: `0.285714`
- Local direction accuracy: `0.571429`
- Interpretation: The local delta-derived direction pattern is closer to the label than the LLM understanding JSON on this example.

### 8. Both fail

- record_id: `stage1_normal_edit_008__edit_rand_01521__rand_04100__5539`
- category: `normal_edit`
- task_type: `edit`
- prompt: move the beam right and move the beam up

Images:

- current_profile: [images/finetuned_llm_25/000007_stage1_normal_edit_008__edit_rand_01521__rand_04100__5539/current_profile.png](images/finetuned_llm_25/000007_stage1_normal_edit_008__edit_rand_01521__rand_04100__5539/current_profile.png)
- target_profile: [images/finetuned_llm_25/000007_stage1_normal_edit_008__edit_rand_01521__rand_04100__5539/target_profile.png](images/finetuned_llm_25/000007_stage1_normal_edit_008__edit_rand_01521__rand_04100__5539/target_profile.png)
- difference_profile: [images/finetuned_llm_25/000007_stage1_normal_edit_008__edit_rand_01521__rand_04100__5539/difference_profile.png](images/finetuned_llm_25/000007_stage1_normal_edit_008__edit_rand_01521__rand_04100__5539/difference_profile.png)
- composite_profile: [images/finetuned_llm_25/000007_stage1_normal_edit_008__edit_rand_01521__rand_04100__5539/composite_profile.png](images/finetuned_llm_25/000007_stage1_normal_edit_008__edit_rand_01521__rand_04100__5539/composite_profile.png)

- Ground truth changed variables: `['source_to_lens', 'lens_to_camera', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- Ground truth directions: `source_to_lens: increase, lens_to_camera: decrease, focal_length: unchanged, lens_x: decrease, lens_y: decrease, camera_x: increase, camera_y: increase`
- LLM predicted changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- LLM predicted directions: `source_to_lens: increase, lens_to_camera: decrease, focal_length: decrease, lens_x: increase, lens_y: increase, camera_x: increase, camera_y: increase`
- LLM reasoning_summary: Predicted setup fields are supervised labels from the existing profile2setup record.
- Local inferred changed variables: `['source_to_lens', 'lens_to_camera', 'camera_x', 'camera_y']`
- Local inferred directions: `source_to_lens: increase, lens_to_camera: decrease, focal_length: unchanged, lens_x: unchanged, lens_y: unchanged, camera_x: increase, camera_y: increase`
- LLM setup error, mean absolute over 7 variables: `0.0329279`
- Local setup error, mean absolute over 7 variables: `0.00481213`
- LLM direction accuracy: `0.571429`
- Local direction accuracy: `0.714286`
- Interpretation: The local delta-derived direction pattern is closer to the label than the LLM understanding JSON on this example.

### 9. Constraint example

- record_id: `stage1_constraint_001__edit_rand_04519__rand_02308__6678`
- category: `constraint`
- task_type: `edit`
- prompt: increase focal_length only

Images:

- current_profile: [images/finetuned_llm_25/000010_stage1_constraint_001__edit_rand_04519__rand_02308__6678/current_profile.png](images/finetuned_llm_25/000010_stage1_constraint_001__edit_rand_04519__rand_02308__6678/current_profile.png)
- target_profile: [images/finetuned_llm_25/000010_stage1_constraint_001__edit_rand_04519__rand_02308__6678/target_profile.png](images/finetuned_llm_25/000010_stage1_constraint_001__edit_rand_04519__rand_02308__6678/target_profile.png)
- difference_profile: [images/finetuned_llm_25/000010_stage1_constraint_001__edit_rand_04519__rand_02308__6678/difference_profile.png](images/finetuned_llm_25/000010_stage1_constraint_001__edit_rand_04519__rand_02308__6678/difference_profile.png)
- composite_profile: [images/finetuned_llm_25/000010_stage1_constraint_001__edit_rand_04519__rand_02308__6678/composite_profile.png](images/finetuned_llm_25/000010_stage1_constraint_001__edit_rand_04519__rand_02308__6678/composite_profile.png)

- Ground truth changed variables: `['focal_length']`
- Ground truth directions: `source_to_lens: unchanged, lens_to_camera: unchanged, focal_length: increase, lens_x: unchanged, lens_y: unchanged, camera_x: unchanged, camera_y: unchanged`
- LLM predicted changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- LLM predicted directions: `source_to_lens: increase, lens_to_camera: increase, focal_length: increase, lens_x: increase, lens_y: increase, camera_x: increase, camera_y: decrease`
- LLM reasoning_summary: Predicted setup fields are supervised labels from the existing profile2setup record.
- Local inferred changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_y', 'camera_y']`
- Local inferred directions: `source_to_lens: increase, lens_to_camera: increase, focal_length: increase, lens_x: unchanged, lens_y: decrease, camera_x: unchanged, camera_y: decrease`
- LLM setup error, mean absolute over 7 variables: `0.0618324`
- Local setup error, mean absolute over 7 variables: `0.0257894`
- LLM direction accuracy: `0.142857`
- Local direction accuracy: `0.428571`
- Interpretation: The local delta-derived direction pattern is closer to the label than the LLM understanding JSON on this example.

### 10. Invalid example

- record_id: `stage1_invalid_001__edit_rand_00170__rand_01454__2361`
- category: `invalid`
- task_type: `edit`
- prompt: change the wavelength

Images:

- current_profile: [images/finetuned_llm_25/000015_stage1_invalid_001__edit_rand_00170__rand_01454__2361/current_profile.png](images/finetuned_llm_25/000015_stage1_invalid_001__edit_rand_00170__rand_01454__2361/current_profile.png)
- target_profile: [images/finetuned_llm_25/000015_stage1_invalid_001__edit_rand_00170__rand_01454__2361/target_profile.png](images/finetuned_llm_25/000015_stage1_invalid_001__edit_rand_00170__rand_01454__2361/target_profile.png)
- difference_profile: [images/finetuned_llm_25/000015_stage1_invalid_001__edit_rand_00170__rand_01454__2361/difference_profile.png](images/finetuned_llm_25/000015_stage1_invalid_001__edit_rand_00170__rand_01454__2361/difference_profile.png)
- composite_profile: [images/finetuned_llm_25/000015_stage1_invalid_001__edit_rand_00170__rand_01454__2361/composite_profile.png](images/finetuned_llm_25/000015_stage1_invalid_001__edit_rand_00170__rand_01454__2361/composite_profile.png)

- Ground truth changed variables: `[]`
- Ground truth directions: `source_to_lens: unchanged, lens_to_camera: unchanged, focal_length: unchanged, lens_x: unchanged, lens_y: unchanged, camera_x: unchanged, camera_y: unchanged`
- LLM predicted changed variables: `['source_to_lens', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- LLM predicted directions: `source_to_lens: increase, lens_to_camera: decrease, focal_length: increase, lens_x: increase, lens_y: decrease, camera_x: decrease, camera_y: decrease`
- LLM reasoning_summary: Predicted setup fields are supervised labels from the existing profile2setup record.
- Local inferred changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_y', 'camera_x', 'camera_y']`
- Local inferred directions: `source_to_lens: increase, lens_to_camera: decrease, focal_length: increase, lens_x: unchanged, lens_y: decrease, camera_x: decrease, camera_y: decrease`
- LLM setup error, mean absolute over 7 variables: `0.0337018`
- Local setup error, mean absolute over 7 variables: `0.0256386`
- LLM direction accuracy: `0`
- Local direction accuracy: `0.142857`
- Interpretation: The label expects rejection or clarification, but the LLM did not reject and the local model produced a numerical prediction.

### 11. Ambiguous example

- record_id: `stage1_ambiguous_multi_intent_001__edit_rand_03489__rand_03518__3929`
- category: `ambiguous_multi_intent`
- task_type: `edit`
- prompt: move the beam left and right

Images:

- current_profile: [images/finetuned_llm_25/000020_stage1_ambiguous_multi_intent_001__edit_rand_03489__rand_03518__3929/current_profile.png](images/finetuned_llm_25/000020_stage1_ambiguous_multi_intent_001__edit_rand_03489__rand_03518__3929/current_profile.png)
- target_profile: [images/finetuned_llm_25/000020_stage1_ambiguous_multi_intent_001__edit_rand_03489__rand_03518__3929/target_profile.png](images/finetuned_llm_25/000020_stage1_ambiguous_multi_intent_001__edit_rand_03489__rand_03518__3929/target_profile.png)
- difference_profile: [images/finetuned_llm_25/000020_stage1_ambiguous_multi_intent_001__edit_rand_03489__rand_03518__3929/difference_profile.png](images/finetuned_llm_25/000020_stage1_ambiguous_multi_intent_001__edit_rand_03489__rand_03518__3929/difference_profile.png)
- composite_profile: [images/finetuned_llm_25/000020_stage1_ambiguous_multi_intent_001__edit_rand_03489__rand_03518__3929/composite_profile.png](images/finetuned_llm_25/000020_stage1_ambiguous_multi_intent_001__edit_rand_03489__rand_03518__3929/composite_profile.png)

- Ground truth changed variables: `[]`
- Ground truth directions: `source_to_lens: unchanged, lens_to_camera: unchanged, focal_length: unchanged, lens_x: unchanged, lens_y: unchanged, camera_x: unchanged, camera_y: unchanged`
- LLM predicted changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- LLM predicted directions: `source_to_lens: decrease, lens_to_camera: decrease, focal_length: increase, lens_x: increase, lens_y: decrease, camera_x: decrease, camera_y: decrease`
- LLM reasoning_summary: Predicted setup fields are supervised labels from the existing profile2setup record.
- Local inferred changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'lens_x', 'lens_y', 'camera_y']`
- Local inferred directions: `source_to_lens: decrease, lens_to_camera: decrease, focal_length: increase, lens_x: increase, lens_y: decrease, camera_x: unchanged, camera_y: decrease`
- LLM setup error, mean absolute over 7 variables: `0.0281826`
- Local setup error, mean absolute over 7 variables: `0.0141218`
- LLM direction accuracy: `0`
- Local direction accuracy: `0.142857`
- Interpretation: The label expects rejection or clarification, but the LLM did not reject and the local model produced a numerical prediction.

### 12. Ambiguous example

- record_id: `stage1_ambiguous_multi_intent_002__edit_rand_01148__rand_03531__9774`
- category: `ambiguous_multi_intent`
- task_type: `edit`
- prompt: make it wider and smaller

Images:

- current_profile: [images/finetuned_llm_25/000021_stage1_ambiguous_multi_intent_002__edit_rand_01148__rand_03531__9774/current_profile.png](images/finetuned_llm_25/000021_stage1_ambiguous_multi_intent_002__edit_rand_01148__rand_03531__9774/current_profile.png)
- target_profile: [images/finetuned_llm_25/000021_stage1_ambiguous_multi_intent_002__edit_rand_01148__rand_03531__9774/target_profile.png](images/finetuned_llm_25/000021_stage1_ambiguous_multi_intent_002__edit_rand_01148__rand_03531__9774/target_profile.png)
- difference_profile: [images/finetuned_llm_25/000021_stage1_ambiguous_multi_intent_002__edit_rand_01148__rand_03531__9774/difference_profile.png](images/finetuned_llm_25/000021_stage1_ambiguous_multi_intent_002__edit_rand_01148__rand_03531__9774/difference_profile.png)
- composite_profile: [images/finetuned_llm_25/000021_stage1_ambiguous_multi_intent_002__edit_rand_01148__rand_03531__9774/composite_profile.png](images/finetuned_llm_25/000021_stage1_ambiguous_multi_intent_002__edit_rand_01148__rand_03531__9774/composite_profile.png)

- Ground truth changed variables: `[]`
- Ground truth directions: `source_to_lens: unchanged, lens_to_camera: unchanged, focal_length: unchanged, lens_x: unchanged, lens_y: unchanged, camera_x: unchanged, camera_y: unchanged`
- LLM predicted changed variables: `['source_to_lens', 'lens_to_camera', 'lens_x', 'lens_y', 'camera_x', 'camera_y']`
- LLM predicted directions: `source_to_lens: decrease, lens_to_camera: decrease, focal_length: unchanged, lens_x: increase, lens_y: decrease, camera_x: increase, camera_y: decrease`
- LLM reasoning_summary: Predicted setup fields are supervised labels from the existing profile2setup record.
- Local inferred changed variables: `['source_to_lens', 'lens_to_camera', 'focal_length', 'camera_x', 'camera_y']`
- Local inferred directions: `source_to_lens: decrease, lens_to_camera: decrease, focal_length: increase, lens_x: unchanged, lens_y: unchanged, camera_x: increase, camera_y: decrease`
- LLM setup error, mean absolute over 7 variables: `0.0519639`
- Local setup error, mean absolute over 7 variables: `0.0049381`
- LLM direction accuracy: `0.142857`
- Local direction accuracy: `0.285714`
- Interpretation: The label expects rejection or clarification, but the LLM did not reject and the local model produced a numerical prediction.
