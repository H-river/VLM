# Checkpoints and artifacts

Checkpoints, adapters, optimizer state, prediction JSONL, and raw run bundles
are intentionally excluded from normal Git tracking. The current checkout has
large local artifacts, including Learned-H1, Qwen supervisor/selector, and
candidate transition-pilot checkpoints. They were preserved in place during
the repository cleanup.

The canonical Learned-H1 checkpoint is referenced by repository-relative path
and SHA-256 in `configs/controller/branch_a.json`. Its presence is optional for
installation and required only for Learned-H1 inference/controller replay.

For sharing, prefer a versioned external artifact store or a private release
with checksum verification. Git LFS can be suitable for a small, intentional
set of redistributable models, but is not permission to publish models or bulk
run state. Complete `manifests/checkpoint_manifest.template.json` first.

