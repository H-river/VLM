# profile2setup (v2)

`profile2setup` is the v2 pipeline package.

`lang2setup` is preserved as v1.

v2 predicts 7-variable setup deltas from profile + prompt (with optional target profile and/or current setup context).

Canonical v2 variables are:
- `source_to_lens`
- `lens_to_camera`
- `focal_length`
- `lens_x`
- `lens_y`
- `camera_x`
- `camera_y`

`camera_x` and `camera_y` mean camera offsets.

Old alignment field names are not v2 output names. Any legacy alignment fields are only backward-compatible input fallback.
