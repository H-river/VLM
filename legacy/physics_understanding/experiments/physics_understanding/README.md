# Physics Understanding Diagnostic

This experiment is a diagnostic test of whether light supervised fine-tuning
improves prompt-conditioned optical and physics understanding in the
`profile2setup` task.

It is not trying to optimize a production controller. The goal is to measure
whether the model uses prompt meaning and optical relationships when predicting
setup changes, not merely whether it emits valid JSON or low-friction structured
output.

## Main Comparison

The primary comparison is:

- Raw Base LLM
- Raw Fine-tuned LLM
- Local PyTorch baseline

The LLM runs should be evaluated as raw model outputs. No deterministic
post-processing, constraint repair, routing fixes, clipping, or JSON-field
rewrites should be applied in the main metrics. Validity and schema compliance
can be reported, but they are supporting diagnostics rather than the central
claim.

Optional deterministic post-processing may be added later only as a separate
engineering baseline. If that baseline exists, it should be reported separately
from the main raw-model metrics so the effect of SFT is not confused with
hand-coded repair logic.

## Diagnostic Question

The main question is:

Does light SFT improve the model's ability to use prompt meaning and physics
relationships when interpreting current and target beam profiles?

Important failure modes include:

- Producing valid JSON while ignoring prompt constraints.
- Changing variables that the prompt asks to keep fixed.
- Predicting directions that contradict the requested beam/profile change.
- Treating image evidence and prompt text independently instead of jointly.
- Failing shuffled-prompt or conflict probes where superficial format validity
  is insufficient.

## Test Conditions

The initial diagnostic configuration enables these probe families:

- `prompt_only`: prompt-conditioned understanding without profile images.
- `images_only`: image/profile-conditioned behavior without prompt semantics.
- `prompt_plus_images`: normal multimodal setup.
- `shuffled_prompt`: prompt/profile mismatches to detect prompt sensitivity.
- `conflict`: explicitly conflicting prompt and image/setup evidence.

The same selected records should be used across Base LLM, Fine-tuned LLM, and
Local PyTorch baseline where the model interface supports the condition. When a
condition is not native to the local baseline, report it as not applicable
rather than forcing an artificial interface.

## Configuration

`config.yaml` records model IDs, the existing `all_modes` dataset paths, the
local checkpoint path, output location, seed, and enabled diagnostic probes. It
does not change any training or inference behavior by itself.
