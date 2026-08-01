# Evidence-grounded v5A seed-42 decision

The single predeclared run stopped at 50 optimizer steps. Both saved checkpoints were evaluated on all 200 scenario-disjoint development records. Neither passed any promotion gate, so the 200-record confirmation split remains sealed.

| Run | Macro | Schema | Control F1 | Feasible recall | Control pair-joint | Action exact | Sufficiency F1 | Insufficient recall | Sufficiency pair-joint | Witness valid | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.417 | 0.760 | 0.650 | 0.660 | 0.380 | 0.040 | 0.333 | 0.000 | 0.000 | 0.000 | 0/9 |
| checkpoint 25 | 0.433 | 0.825 | 0.624 | 0.500 | 0.340 | 0.020 | 0.333 | 0.000 | 0.000 | 0.000 | 0/9 |
| checkpoint 50 | 0.460 | 0.885 | 0.605 | 0.380 | 0.340 | 0.040 | 0.333 | 0.000 | 0.000 | 0.000 | 0/9 |

## Decision

- Do not promote either checkpoint and do not open confirmation or the sealed pilot test.
- Do not extend the run or try another seed under this protocol. The point estimates move monotonically toward the conservative infeasible control branch while insufficiency remains completely unlearned.
- Lower completion loss mainly improves JSON/schema imitation. It does not improve pair-sensitive evidence aggregation or executable action selection.

## Failure interpretation

The v4 failure was partly missing evidence. V5A removes that problem: the prompt explicitly provides residuals and thresholded directions, and the deterministic visible-evidence solver is perfect. The remaining failure is therefore at the instruction/aggregation interface. Ordinary token-level SFT rewards many predictable JSON and copied numeric tokens, while the small set-valued decision and action-selection fields contribute little loss. One balanced pass over 200 records changes formatting more readily than it changes those decisions.

The next design should not simply add epochs or seeds. It should make prompt-visible evidence use an explicit supervised object—for example `observed_direction_set`, `successful_action_indices`, and `selected_index`—and score those fields before the final status. A deterministic tool should remain the reference and may be the production path for thresholding and exhaustive selection; the LLM can be evaluated on choosing the tool, supplying its inputs, and interpreting the result.

Passing such a decomposed tier would still establish evidence aggregation only, not internal Fresnel simulation or laboratory validity.
