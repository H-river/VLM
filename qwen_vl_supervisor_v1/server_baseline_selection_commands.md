# Future development-only A/B baseline selection

No A/B winner or threshold is selected in the current preparation task. This
document defines the evidence boundary for the future server run; it is not an
instruction to invent values. The final evaluation freeze must remain blocked
until a concrete artifact exists and passes the strict validator below.

Use only the configured train and development manifests. Arm A candidates may
fit on `manifest_train.jsonl` and must emit exactly one
`deterministic_reference` prediction for every `manifest_dev.jsonl` sample.
Arm B must run both pinned specialists on exactly those development samples and
record both calibrated probabilities per sample. Do not open or derive scores
from either frozen manifest.

The future evidence producer must write an artifact conforming to
`schema/baseline_dev_selection_artifact.schema.json`. Its hash-addressed raw
evidence must include:

- all three configured A candidates, their model and training evidence, raw
  dev predictions, recomputed offline reports, coverage, metrics, ranks, and
  the concrete rank-one candidate plus canonical diagnosis-to-policy/action
  mapping;
- the two pinned B specialist identities, raw dev probability scores, two
  explicit strictly increasing threshold lists, the complete Cartesian grid
  with recomputed metrics and ranks, and raw predictions/report for the
  concrete rank-one threshold pair; and
- the exact B arbitration contract: anomaly comparison uses `>=`; if neither
  score is strictly below its threshold, compare
  `(score-threshold)/max(abs(threshold),1e-12)`; the deterministic tie order is
  nominal, sensor saturation, then secondary reflection.

Every artifact and nested candidate records
`protected_or_frozen_data_used: false` and
`frozen_predictions_opened: false`. The artifact contains no timestamp, so
identical evidence produces identical bytes. The validator re-hashes every
referenced file/tree, recomputes every A report and every B grid row from raw
dev evidence, and refuses a selected entry that is not deterministic rank one.

After the reviewed future evidence producer has created a new artifact, run:

```bash
set -euo pipefail
cd /home/jiamo/VLM

EVALUATION_CONFIG=qwen_vl_supervisor_v1/configs/evaluation_frozen.yaml
BASELINE_DEV_SELECTION=qwen_vl_supervisor_v1/artifacts/dev_baseline_selection/baseline_dev_selection.json

test -f "$BASELINE_DEV_SELECTION"
python -m qwen_vl_supervisor_v1.validate_baseline_dev_selection \
  --repository-root /home/jiamo/VLM \
  --evaluation-config "$EVALUATION_CONFIG" \
  --artifact "$BASELINE_DEV_SELECTION"

BASELINE_DEV_SELECTION_SHA256="$(python \
  qwen_vl_supervisor_v1/scripts/freeze_evaluation.py hash \
  --path "$BASELINE_DEV_SELECTION" --digest-only)"
printf '%s\n' "$BASELINE_DEV_SELECTION_SHA256"
```

The printed SHA-256 is a required input to the future final-freeze command.
Absence of this artifact, a placeholder, any concrete value without its raw
evidence, or any hash/recomputation mismatch must stop the freeze. Validation
does not authorize frozen prediction or formal evaluation.
