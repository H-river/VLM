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
{"actuator_constraints":{"absolute_limit_source":"repository_sampling_domain_not_hardware_limit","absolute_position_limits":{"camera_x":[-3.0,3.0],"camera_y":[-3.0,3.0],"lens_x":[-3.0,3.0],"lens_y":[-3.0,3.0]},"continuous_actions_selected_by":"frozen_h1_one_step_cem","per_step_delta_limits":{"camera_x":[-0.02,0.02],"camera_y":[-0.02,0.02],"lens_x":[-0.05,0.05],"lens_y":[-0.05,0.05]},"units":"mm"},"current_metrics":{"centroid_x":63.53186456323575,"centroid_y":54.925979630548476,"coordinate_frame":"diagnostic_image_128px","peak_intensity":1.0,"width_x":15.472556614777412,"width_y":21.671019087761128},"decision_context":"static_anomaly_policy","goal_metrics":{"centroid_x":490.34516351543317,"centroid_y":713.9802844196587,"coordinate_frame":"lab_sensor_1024px_and_raw_peak","peak_intensity":697073.0625,"width_x":97.46737399794071,"width_y":96.21499688909658},"recent_history":[],"remaining_step_budget":8}
```

## Assistant

```json
{"diagnosis":"secondary_reflection","measurement_policy":"primary_spot","supervisor_action":"switch_measurement"}
```
