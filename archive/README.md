# Archive policy

The repository already uses `legacy/` as its traceable source archive. It was
not renamed during this cleanup because historical imports, report links, and
Git rename detection would be needlessly disrupted.

- `legacy/code/` contains superseded rebuilds and pipelines.
- `legacy/experiments/` contains compact historical reports.
- `legacy/cleanup_reports/` records earlier cleanup actions.

Nothing in `legacy/` was deleted. Files should be removed only after a human
review confirms that their source, result, hash, and reproduction value are
represented elsewhere. See `docs/REPOSITORY_AUDIT.md` for deletion candidates
that were intentionally preserved.

