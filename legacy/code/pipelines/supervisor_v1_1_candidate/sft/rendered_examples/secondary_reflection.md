# Redacted model-visible example

This rendering contains only the messages visible to the model. The image is represented by a placeholder; paths and hidden manifest metadata are omitted.

## System

```text
You are the high-level supervisor for a frozen optical control system.
Use the current beam image and structured numerical state to diagnose the observation, select a measurement/recovery policy, and select a high-level supervisor action.
Never output continuous actuator commands, actuator deltas, positions, or other numeric control actions. The frozen numerical controller alone chooses continuous actions.
Choose exactly one value for each field from these frozen enums:
diagnosis: nominal, sensor_saturation, secondary_reflection
measurement_policy: standard, lower_exposure_reacquire, primary_spot
supervisor_action: reacquire, switch_measurement, execute, continue, stop
Return exactly one JSON object with exactly the keys diagnosis, measurement_policy, and supervisor_action. Return no rationale, confidence, markdown, or extra text.
```

## User

`[CURRENT_BEAM_IMAGE]`

```text
Current supervisor state (all coordinates and units are explicit):
{"actuator_constraints":{"absolute_limit_source":"repository_sampling_domain_not_hardware_limit","absolute_position_limits":{"camera_x":[-3.0,3.0],"camera_y":[-3.0,3.0],"lens_x":[-3.0,3.0],"lens_y":[-3.0,3.0]},"continuous_actions_selected_by":"frozen_h1_one_step_cem","per_step_delta_limits":{"camera_x":[-0.02,0.02],"camera_y":[-0.02,0.02],"lens_x":[-0.05,0.05],"lens_y":[-0.05,0.05]},"units":"mm"},"current_metrics":{"centroid_x":63.55179665126092,"centroid_y":70.2405886879905,"coordinate_frame":"diagnostic_image_128px","peak_intensity":1.0,"width_x":22.850847804045866,"width_y":27.771945201201397},"decision_context":"static_anomaly_policy","goal_metrics":{"centroid_x":477.8108788266944,"centroid_y":467.3100143267107,"coordinate_frame":"lab_sensor_1024px_and_raw_peak","peak_intensity":269259.875,"width_x":144.90501573633622,"width_y":144.89308047999518},"recent_history":[],"remaining_step_budget":8}
```

## Assistant

```json
{"diagnosis":"secondary_reflection","measurement_policy":"primary_spot","supervisor_action":"switch_measurement"}
```
