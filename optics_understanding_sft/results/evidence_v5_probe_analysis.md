# Evidence-grounded v5 learnability probe

All runs use the same 28 target records (seven complete physical scenarios). They are a design probe built from already-used v4 training groups, not a future evaluation split. No additional training or paid API call was used.

| Representation | Model | Macro | Schema | Control F1 | Control pair-joint | Sufficiency F1 | Sufficiency pair-joint |
|---|---:|---:|---:|---:|---:|---:|---:|
| verbose raw centroids, base model | base | 0.325 | 0.500 | 0.333 | 0.000 | 0.333 | 0.000 |
| compact raw centroids, base model | base | 0.307 | 0.607 | 0.475 | 0.143 | 0.333 | 0.000 |
| compact raw centroids, v4 mixed-100 adapter | v4 adapter | 0.350 | 1.000 | 0.333 | 0.000 | 0.333 | 0.000 |
| numeric residual/delta scaffold, base model | base | 0.477 | 0.857 | 0.708 | 0.429 | 0.333 | 0.000 |
| residual plus direction scaffold, base model | base | 0.505 | 0.857 | 0.708 | 0.429 | 0.333 | 0.000 |
| symbolic scaffold with repeated decision rule, base model | base | 0.493 | 1.000 | 0.333 | 0.000 | 0.333 | 0.000 |

## Interpretation

- The numeric residual scaffold is the first representation that makes control decisions pair-sensitive: the base model reaches 0.708 status macro-F1 and 0.429 pair-joint accuracy on that small slice.
- The old v4 adapter erases that improvement and returns the conservative control class for every row, consistent with the v4 collapse analysis.
- Sufficiency remains a constant `answerable` classifier even when each row contains an explicit thresholded direction and the decision rule is repeated next to the output contract. Its pair-joint accuracy remains zero.
- Repeating the decision rule improves schema validity to 1.0 but collapses control status to the constant infeasible class. Prompt wording alone is therefore unstable and is not a promotion signal.
- The next justified experiment is a fresh, scenario-disjoint v5A dataset followed by one predeclared 50-step seed-42 format-and-aggregation trial. It should be stopped unless both task F1 and pair-joint gates improve; raw-centroid v5B/v5C transfer is evaluated only after that.

The 28-record numbers are diagnostic estimates, not benchmark claims.
