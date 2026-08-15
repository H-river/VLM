# Specialist configuration map

The retained measurement, direction, forward, and inverse experiments are in
`specialist_rebuild_v2/`. They are supporting experiments, not the canonical
continuous controller. Their historical data-generation and training settings
remain next to the implementation to preserve provenance.

The canonical numerical transition model is configured by
`configs/controller/branch_a.json` and implemented by
`continuous_control_v12.world_model.ForwardEnsemble`.

