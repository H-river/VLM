# Profile2Setup Input-Mode Expansion Plan

## Goal

Extend `profile2setup` so training and inference support the existing modes **plus** two additional modes:

1. **Current-only mode**: `current_profile + prompt` (no `target_profile`, no `current_setup`)
2. **Paired-no-setup mode**: `current_profile + prompt + target_profile` (no `current_setup`)

while preserving backward compatibility with the existing:

- `absolute` mode
- `edit` mode

---

## Why this change is needed

The current dataset validator and sample builder enforce strict field combinations for only two task types. This excludes both requested modes.

Current constraints:

- `_VALID_TASKS = {"absolute", "edit"}`
- `absolute` requires `current_profile_path=None`, `current_setup=None`, `target_delta=None`
- `edit` requires `current_profile_path`, `target_profile_path`, `current_setup`, `target_delta`

Additionally, model forward always requires `current_setup` input and has no explicit "setup missing" signal.

---

## Design principles

1. **Optional modalities, explicit masks**
   - Treat profile/setup presence as first-class signals.
2. **Backwards compatibility first**
   - Existing JSONL format and old modes must continue to work.
3. **Loss masking over branching**
   - Use per-head supervision masks instead of hard-coding many task-specific pathways.
4. **Single model contract across modes**
   - Keep one forward signature with explicit presence indicators.

---

## Proposed unified sample schema

Keep existing fields where possible, and add presence/supervision metadata:

### Inputs (per record)

- `prompt: str`
- `current_profile_path: str | null`
- `target_profile_path: str | null`
- `current_setup: dict | null`

### Targets (per record)

- `target_setup: dict | null`
- `target_delta: dict | null`
- `change_mask: optional derivable`

### New metadata (derived in dataset)

- `has_current_profile: bool`
- `has_target_profile: bool`
- `has_current_setup: bool`

### New supervision masks (derived in dataset)

- `delta_loss_mask: float` (0 or 1)
- `absolute_loss_mask: float` (0 or 1)
- `change_loss_mask: float` (0 or 1)

---

## Mode matrix

| Mode | current profile | target profile | current setup | target setup | target delta | Supervise absolute | Supervise delta | Supervise change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| absolute (existing) | optional/no | yes | no | yes | no | 1 | 0 | 0 |
| edit (existing) | yes | yes | yes | yes | yes | 1 (optional) | 1 | 1 |
| current-only (new) | yes | no | no | yes (recommended) | no | 1 | 0 | 0 |
| paired-no-setup (new) | yes | yes | no | yes (recommended) | no (or optional synthetic) | 1 | 0 (recommended) | 0/1 (optional) |

> Recommendation: for new modes without `current_setup`, treat `delta` as unsupervised by default unless a principled baseline is introduced.

---

## Detailed file-by-file implementation plan

## 1) `profile2setup/training/dataset.py`

### A. Validation refactor

Replace strict task-specific validation with capability-based validation:

- Validate top-level record type and prompt type.
- Validate any provided profile paths (`.npy` exists).
- Validate setup-like dicts only when present.
- Keep forbidden legacy-key checks.

Maintain old behavior by mapping old records to equivalent presence masks.

### B. Presence flags in `__getitem__`

Derive and return tensors:

- `setup_present: torch.float32` shape `[1]`
- optionally `target_present`, `current_profile_present`, `target_profile_present`

### C. Setup fallback behavior

If `current_setup` is missing:

- use zero vector for `current_setup` tensor
- set `setup_present=0.0`

If present:

- normalize as before
- set `setup_present=1.0`

### D. Target and loss masks

Emit:

- `target_setup` tensor (if missing, zero vector + `absolute_loss_mask=0`)
- `target_delta` tensor (if missing, zero vector + `delta_loss_mask=0`)
- `change_mask` tensor (if unavailable, zeros + `change_loss_mask=0`)

### E. Collate updates

Add new tensor keys to `tensor_keys`:

- `setup_present`
- `absolute_loss_mask`
- `delta_loss_mask`
- `change_loss_mask`

---

## 2) `profile2setup/models/fusion_model.py`

### A. Forward signature extension

Change forward to:

```python
forward(profile, prompt_tokens, current_setup, setup_present)
```

where `setup_present` is `[B,1]` or `[B]`.

### B. Setup branch gating

Apply explicit gating:

```python
setup_emb = self.setup_encoder(current_setup)
setup_emb = setup_emb * setup_present
```

This distinguishes:

- "real setup near zero" from
- "missing setup represented as zero"

### C. Optional fusion signal

Optionally concatenate `setup_present` into fusion input to let fusion layers reason about missingness directly.

---

## 3) `profile2setup/models/setup_encoder.py`

No required architecture change if gating is done in fusion model. Optional enhancement:

- add a learned missing-setup embedding and blend by `setup_present`.

---

## 4) Training loop (where losses are computed)

Implement masked multi-head loss:

```text
L_abs    = mse(abs_pred, target_setup)   * absolute_loss_mask
L_delta  = mse(delta_pred, target_delta) * delta_loss_mask
L_change = bce(change_logits, change_mask) * change_loss_mask
```

Aggregate by normalized masked means (guard against zero denominator):

```text
loss = w_abs * mean_masked(L_abs)
     + w_delta * mean_masked(L_delta)
     + w_change * mean_masked(L_change)
```

---

## 5) Inference routing policy

Implement explicit routing based on input availability:

- **has current_setup = 1**:
  - prioritize `delta` (+ combine with current setup)
  - optionally compare with `absolute` for consistency
- **has current_setup = 0**:
  - prioritize `absolute`
  - treat `delta` as secondary/diagnostic unless baseline policy exists

---

## 6) Preprocessing (`profile2setup/training/preprocessing.py`)

Current 4-channel scheme is compatible with missing target profile:

- when target missing, target channel = zeros, delta = zeros, mask = zeros

No mandatory changes required.

Optional enhancement:

- add `current_mask` channel for symmetry and clearer modality signaling.

---

## 7) Config changes

Add mode and loss controls to training config:

```yaml
data:
  allow_missing_current_setup: true
  allow_missing_target_profile: true
  allow_missing_current_profile: true

loss:
  absolute_weight: 1.0
  delta_weight: 1.0
  change_weight: 0.5
  normalize_by_mask: true

inference:
  prefer_absolute_when_setup_missing: true
```

---

## 8) CLI + smoke tests

Update smoke tests to include all four modes.

### Add tests for:

1. current-only (no target/setup)
2. paired-no-setup (current + target, no setup)
3. existing absolute
4. existing edit

Assertions:

- batch creation succeeds
- forward pass succeeds
- mask tensors are correct
- masked loss computes without NaNs/inf

---

## 9) Migration plan

### Phase 1 (safe)

- Add presence + supervision masks to dataset outputs.
- Keep old task validation operational.

### Phase 2

- Update model forward to accept `setup_present` and apply setup gating.

### Phase 3

- Add masked losses in training loop.

### Phase 4

- Enable new modes in production data pipeline and inference router.

### Phase 5

- Optional architecture improvements (missing-setup token, current-mask channel).

---

## 10) Backward compatibility checklist

- [ ] Existing JSONL files load unchanged
- [ ] Old `absolute`/`edit` records produce identical tensors (except additional mask keys)
- [ ] Existing model checkpoints can be loaded (or migration script provided)
- [ ] Inference API supports old call signature (compat shim)

---

## 11) Risks and mitigations

### Risk: mode imbalance

If most data has full setup, missing-setup modes may underperform.

**Mitigation**: mode-balanced sampling or weighted batches.

### Risk: ambiguous zero setup

Without setup-present signal, zero could mean missing or true zero.

**Mitigation**: explicit `setup_present` gating (required).

### Risk: unstable multi-head training

Different supervision availability across modes can destabilize gradients.

**Mitigation**: masked loss normalization + gradual curriculum.

---

## 12) Recommended first implementation slice (minimal viable)

1. Dataset emits:
   - `setup_present`
   - per-head loss masks
2. Model forward accepts `setup_present` and gates setup embedding.
3. Training uses masked losses.
4. Inference selects `absolute` when `setup_present=0`.

This slice alone unlocks both requested new modes while keeping current architecture intact.
