# Visual quantitative failure audit

The paired ablation evaluated 30 visual records (54 images). The full-image
macro score was 0.618; withholding every image scored
0.632. Therefore the current checkpoint gains
-0.014 macro score from pixels.

The dominant data defect is a coordinate-frame mismatch. The legacy numerical centroid is in an
absolute lab frame because camera offsets are not subtracted, while the PNG array is indexed in the
camera sensor frame. A simple deterministic image centroid has median error
16.32 px against the legacy label and
1.85 px against the corrected sensor-frame label.

All 24/24 paired records use different render options for their
two images. Per-image normalization destroys absolute peak calibration, and independent gamma,
background, noise, blur, and saturation make paired width and brightness comparisons unreliable.

These results mean that additional training on the current visual records would reinforce
contradictory or non-identifiable targets. The next dataset version must use sensor-frame centroids,
shared pair calibration, and pixel-dependent prompts before visual fine-tuning.
