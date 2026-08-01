# Supervisor v1 Leakage and Split Audit

Result: PASS for the generated source manifests.

The strict validator examined 264 records, 132 irreversible setup hashes, 132 complete counterfactual pairs, 264 unique sample IDs, and 264 unique image content hashes.

| Check | Result |
|---|---:|
| Setup overlap across train/dev/frozen IID/frozen OOD | 0 |
| Counterfactual-pair overlap across splits | 0 |
| Episode overlap across splits | 0 |
| Augmented/base-observation overlap across splits | 0 |
| Exact image-hash overlap across splits | 0 |
| Duplicate sample IDs | 0 |
| Broken counterfactual pairs | 0 |
| Protected/evaluation source rows in train or dev | 0 |
| Fixed-pixel reflection rows | 0 |
| Missing or unreadable images | 0 |
| Non-finite current/goal/provenance metrics | 0 |
| Causally invalid history entries | 0 |
| Hidden labels/post-decision fields in model input | 0 |

Manifest hashes:

- train: `17718c58516d1ae68b22b3d4e46e1d19758a851a9befbc83a0834c71792c3084`;
- dev: `8a9020195e37894915ff3f600c96bf5193b368a25d32a6186e3900086fb4b669`;
- frozen IID: `c90eadb09949c7db1b51a36df740ef13c018713c31f3cdd8cbf917e9d38e2910`;
- frozen OOD: `4f256c55c5db2df7100348d50bd053dec24858ecc3ecccab3d0bbe671f70c705`.

The machine-readable report is `reports/leakage_split_audit.json`. The builder checks every frozen source file against its expected SHA-256 before construction; a drift fails rather than silently rebuilding from changed inputs.

The frozen manifests were validated structurally and hashed. No model predictions were generated for them, and no frozen prediction inspection was performed.
