# Measurement rebuild v3

The five regression labels are defined in the camera sensor-array frame:
centroid x/y are coordinates in the original 1024 by 1024 sensor image,
sigma x/y are one-standard-deviation widths in source pixels, and peak
intensity is the maximum linear simulator sample. This explicitly replaces the
legacy pseudo-pixel centroid convention, which did not align with PNG pixels
when the camera had a nonzero offset.

At inference, the neural head predicts a correction to the analytic
measurement. Its numerical scale is computed only from that analytic
measurement; no ground-truth target value is used as a model input.

This folder corrects the image dataset and replaces the position-destroying
measurement CNN without overwriting the frozen v2 dataset or checkpoints.

## Corrected image contract

- Base images are 512 x 512, linear-intensity, lossless 16-bit PNG files.
- Each physical state has one base image and seven deterministic view records:
  clean, noise, blur, dim+noise, saturation, boundary crop, and gamma shift.
- All transforms include explicit exposure, gamma, noise, blur, saturation, and
  crop metadata.
- Boundary cropping masks sensor pixels without resizing the image, so centroid
  and width labels remain in the original 1024 x 1024 sensor coordinate system.
- Train, validation, IID test, physics-OOD test, and visual-stress groups remain
  disjoint. They inherit the frozen v2 setups and actions, while simulator
  centroids are recomputed in the corrected camera sensor-array frame.

The builder reuses the verified v2 grids. It reruns only the current and selected
target images instead of recreating all 81 action transitions.

## Replacement measurement model

The model combines:

1. calibrated intensity moments for centroid, width, and peak baselines;
2. a coordinate-aware spatial CNN that keeps a 4 x 4 feature grid;
3. a neural residual measured directly in physical tolerance units.

The loss penalizes the mean normalized error, the worst field in each example,
and errors outside the declared tolerance. Model selection combines clean and
all-condition strict five-field success.

## Commands

```bash
python measurement_rebuild_v3/build_dataset.py \
  --source-dir /home/jiamo/VLM_data/specialist_rebuild_v2 \
  --output-dir /home/jiamo/VLM_data/measurement_rebuild_v3 \
  --workers 3 --worker-affinities '0;8;1,9'

python measurement_rebuild_v3/verify_dataset.py \
  /home/jiamo/VLM_data/measurement_rebuild_v3

python measurement_rebuild_v3/train.py \
  /home/jiamo/VLM_data/measurement_rebuild_v3 \
  /home/jiamo/VLM_runs/measurement_rebuild_v3_one_seed \
  --num-workers 1

python measurement_rebuild_v3/predict.py \
  /home/jiamo/VLM_runs/measurement_rebuild_v3_one_seed/measurement_v3.pt \
  /path/to/beam.png /path/to/image_calibration.json \
  --transform-json /path/to/image_transform.json
```
