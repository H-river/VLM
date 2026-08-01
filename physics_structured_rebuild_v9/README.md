# Physics-structured specialist rebuild v9

This experiment keeps every previously registered model frozen and trains one
seed (`20260801`) on completed data only.

The experiment order is:

1. audit and materialize the preserved targeted shards;
2. train a grouped physics-structured forward/direction model;
3. train calibrated boundary-only direction corrections from group-out-of-fold
   predictions;
4. rebuild numerical-inverse candidate predictions and train a full-candidate
   Set Transformer;
5. compare against the frozen v7 forward/direction and v5 inverse specialists.

No held-out test file is used for model selection, and no deployment manifest
is modified by these scripts.
