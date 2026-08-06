# Shared forward-direction rebuild v6

This isolated candidate uses one neural encoder for two output contracts:

1. five tolerance-normalized numerical beam changes;
2. five independent three-class direction predictions.

The input is the existing 46-value engineered feature vector plus the five
frozen forward-v5 predictions. Five regression heads learn corrections to the
v5 numerical predictions. Five direction heads receive the shared
representation, corrected numerical prediction, v5 prior, and distance from
the direction boundary.

All v1-v5 artifacts and datasets remain unchanged. The held-out test set is
not opened for training or checkpoint selection.

## Completed one-seed evaluation

The full run used seed `20260729`, 10,500 independent training setup groups,
and 850,500 transitions. The selected checkpoint was epoch 2 and contains
825,236 trainable parameters.

| Frozen specialist validation | Previous | Shared v6 |
| --- | ---: | ---: |
| Forward old IID, all five within tolerance | 57.95% | 57.94% |
| Forward difficult, all five within tolerance | 48.92% | 49.45% |
| Direction old IID, all five exact | 51.89% | 52.14% |
| Direction difficult, all five exact | 42.81% | 47.30% |

The unchanged 1,600-request system evaluation produced:

| Physical system metric | Previous | Shared v6 |
| --- | ---: | ---: |
| End-to-end forward, all five within tolerance | 49.67% | 50.00% |
| Correctly routed forward | 51.00% | 51.67% |
| End-to-end direction, all five exact | 52.00% | 47.67% |

The shared representation improves difficult specialist direction validation
and slightly improves forward prediction. It regresses direction performance
on the separate system distribution, so this candidate is evaluated but not
promoted. The currently selected v5 forward and v4 direction artifacts remain
unchanged.
