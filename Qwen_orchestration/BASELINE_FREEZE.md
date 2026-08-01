# Frozen baseline policy

Freeze identifier: `qwen_orchestration_baseline_20260724`

## Meaning of frozen

The freeze is an integrity snapshot. Each registered artifact is identified by
its path, byte size, and SHA-256 digest. A model or dataset is part of the
baseline only when it appears in `freeze/baseline_manifest.json`.

The freeze does not:

- copy large weights into Git;
- change permissions on existing user files;
- claim that an unfinished checkpoint is promoted;
- make external artifact paths portable to another machine.

## Qwen starting point

New orchestration training starts from the promoted visual tool-orchestration
adapter:

```text
../VLM_runs/qwen25vl_3b_qlora_visual_tool_orchestration_v10_2_seed49
```

The incomplete direction-all-field checkpoint at step 150 is recorded for
reproducibility but is not the orchestration starting point.

## Frozen specialists

The initial specialist set is:

1. calibrated deterministic beam-image meter;
2. three-member small direction ensemble;
3. general-action forward hybrid;
4. complete-grid forward hybrid;
5. direct inverse classifier;
6. validation-selected inverse ensemble policy.

The inverse ensemble policy is represented by its component bundles and frozen
selection summary because the current evaluator does not save a standalone
ensemble pickle.

## Change rule

Any future specialist update must receive a new semantic route version, for
example `predict_forward_from_state_v2`. Existing route versions must continue
to resolve to the frozen artifacts in this manifest.

Changes to paths or digests require a new freeze identifier. They must not
silently overwrite this baseline.
