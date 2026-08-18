# DD resolution consumer cutover review

Verdict: **BLOCK**

Finding counts: **critical 0, high 5, medium 1, low 0**.

Commit reviewed: `2f2c801d15545af242fb2b85c51ccc09b3d1f4f2` (parent `798badbe`).

## Attack-surface verdicts

- **Raw/effective provenance: BLOCK.** Semantic authority and source-refresh snapshots replace raw DD fields with effective fields without retaining `raw_dd_context`, the applied/converged resolution identities, the manifest digest, or the typed marker.
- **Strict-load refusal: BLOCK.** At least two new production call paths catch `DDResolutionError` through broad `Exception` handlers and continue with untyped/empty legacy context.
- **Non-resolved behavior identity: BLOCK.** `get_extraction_candidates_dd` changes every pass-through row and drops three pre-existing output fields, including the description and the two grouping keys, even when no resolution applies.
- **DD-gap upstream convergence: BLOCK.** The release-fact boundary accepts resolved pipeline rows and treats their effective `unit` as raw publication evidence; the later resolver cannot recover provenance that the loader discarded.
- **Claimed consumer coverage: BLOCK.** The table lists bulk and targeted extraction as covered, but its named seam test calls only the private helper and never invokes either public extractor. The listed legacy tests contain no typed-resolution assertion.

## Findings

### HIGH 1 — effective source snapshots erase the raw publication identity

Locations: `imas_codex/standard_names/source_authority.py:397`; `imas_codex/standard_names/source_refresh.py:100`; `imas_codex/standard_names/source_refresh.py:169`.

`authority_snapshot` overwrites the locally assembled raw `dd_*` fields with the effective projection and returns no raw context or resolution marker. `stamp_source_snapshots` similarly persists only effective unit/documentation/path, while `detect_source_drift` substitutes effective values before comparison and emits no provenance. Concrete failure: for `camera_ir/channel/camera/direction/x`, a DD 4.1.1 raw unit of `m` is returned/stamped as `1`; a downstream consumer cannot distinguish “published DD converged to 1” from “local reviewed correction supplied 1”, and the snapshot hash/drift report can misreport upstream state.

Required correction: any effective snapshot must carry the exact raw fields, applied/converged resolution ids, manifest digest, and an unambiguous typed marker, or the API must expose distinct raw and effective snapshots with types that prevent substitution.

### HIGH 2 — review source enrichment silently falls back after authority failure

Locations: `imas_codex/standard_names/review/pipeline.py:697`; `imas_codex/standard_names/review/pipeline.py:718`; `imas_codex/standard_names/review/pipeline.py:853`.

Both new typed-resolution paths are inside broad `except Exception` blocks. Concrete failure: remove the packaged manifest or make it malformed; `resolve_dd_rows` raises `DDResolutionManifestInvalid`, the exception is swallowed, and review proceeds with empty/untyped `path_docs` or compose-parity stubs. That is the prohibited legacy fallback rather than a fail-closed refusal.

Required correction: let all `DDResolutionError` subclasses propagate (or explicitly re-raise them) and catch only the graph/query failures that this best-effort enrichment was originally intended to tolerate. Add absent and malformed manifest tests through the public review entry points.

### HIGH 3 — refine-doc enrichment silently falls back after authority failure

Locations: `imas_codex/standard_names/workers.py:10219`; `imas_codex/standard_names/workers.py:10225`.

`process_refine_docs_batch` resolves DD rows inside the existing best-effort graph enrichment block, whose broad handler catches the resolution failure. Concrete failure: with an absent manifest, `prompt_context["dd_paths"]` remains empty and the refine model runs without the typed DD context instead of refusing the batch. This independently violates strict-load fail-closed behavior at a new call site.

Required correction: keep graph lookup best-effort only if intended, but resolve outside that catch or explicitly re-raise `DDResolutionError`; pin malformed and absent authority at the actual refine-doc batch boundary.

### HIGH 4 — pass-through extraction drops existing row fields

Location: `imas_codex/standard_names/graph_ops.py:688`.

Before this commit, `get_extraction_candidates_dd` returned the query rows unchanged, including `description`, `ids_name`, and `cluster_label`. It now returns only `context.as_pipeline_item()`. That projection has `documentation` rather than `description` and has neither grouping field. Concrete failure: a non-resolved `equilibrium/...` candidate that previously returned six graph values now loses its source description and both IDS/cluster grouping keys, regardless of whether the manifest has a matching record. A consumer grouping by `ids_name` or `cluster_label` either raises or silently loses batching semantics.

Required correction: merge the effective projection into each original row (`{**row, **context.as_pipeline_item()}`) and add a no-active-resolution equality regression that asserts every pre-existing key and value remains unchanged except the explicitly additive typed metadata.

### HIGH 5 — resolved rows can falsely prove upstream DD-gap convergence

Locations: `imas_codex/standard_names/dd_gaps.py:1909`; `imas_codex/standard_names/dd_gaps.py:2096`; `imas_codex/cli/sn.py:6758`.

`build_unit_release_facts` reads only top-level `unit`/`units`; the JSON object branch also passes arbitrary mappings unchanged. Neither boundary rejects `_dd_resolution_marker` nor extracts and verifies `raw_dd_context`. `reconcile_dd_gaps` later creates a new receipt from that already-conflated top-level value and compares `resolved.raw.value`, but at that point “raw” is merely the caller-provided effective value. Concrete failure: give the loader a resolved row for `camera_ir/channel/camera/direction/x` containing `unit: "1"`, `raw_dd_context.unit: "m"`, and the typed marker. The loader retains `1` as the release fact; the exact 4.1.1 resolution classifies `1` as converged, and a gap expecting `1` becomes eligible for `resolved_upstream` even though the published raw declaration remains `m`.

Required correction: use a dedicated raw-release-fact type/loader that rejects typed effective pipeline items or extracts the raw value only after validating marker, exact version, path, and manifest identity. Add an active-resolution regression using a real manifest path and assert the effective value cannot populate `would_resolve`.

### MEDIUM 1 — the coverage matrix tests helpers, not the claimed consumer boundaries

Locations: `/tmp/reckon-s8-evidence/ddres-cutover/consumer-coverage.md:5`; `/tmp/reckon-s8-evidence/ddres-cutover/consumer-coverage.md:6`; `tests/standard_names/test_dd_resolution_consumers.py:82`.

The matrix claims bulk and targeted extraction are covered by `test_consumer_boundaries_call_typed_authority`, but that test directly invokes `_apply_typed_dd_resolutions`. It never invokes `extract_dd_candidates` or `extract_specific_paths`, so it cannot detect ordering mistakes, later legacy overwrites, or call-site exception handling. The other named files (`test_dd_sources.py` and `test_focus_reseed.py`) contain no typed marker/raw-context/manifest-digest assertion. The same seam test also exercises source refresh and attachment via private helpers, so its name overstates end-to-end coverage.

Required correction: drive each listed public consumer with malformed authority, an active resolution, and a no-resolution row; assert raw/effective provenance and exception propagation at the returned or persisted boundary.

## Validation evidence

- `git diff --check 2f2c801d^..2f2c801d`: pass.
- Focused pytest selection: 114 collected, 110 passed, 4 failed in the available editable environment. The four failures are not accepted as commit evidence because that environment imported `imas_codex` from `/home/ITER/mcintos/Code/imas-codex` rather than this checkout. Forcing this checkout cannot bootstrap its gitignored generated graph models because the assigned worktree is read-only. Full logs preserve both the result and the environment diagnosis.
- Static before/after proof: parent `get_extraction_candidates_dd` returns `list(results)`; reviewed code returns only `context.as_pipeline_item()`.
- Coverage audit: the named boundary test calls the private extraction helper at line 83; it does not call either claimed public extractor.
- Final checkout cleanliness: `git status --porcelain` emitted no output.

The commit cannot be approved because five high-severity findings remain.
