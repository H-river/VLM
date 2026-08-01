# Cross-fitted shared forward-direction rebuild v7

This candidate improves the v6 training protocol before adding new simulator
data. It generates five-fold out-of-fold forward predictions by complete
optical setup group, then trains a conservative shared forward-direction
residual model against those realistic unseen-setup errors.

All earlier datasets, checkpoints, and deployment manifests remain unchanged.

