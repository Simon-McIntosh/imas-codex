# Reuse map for the four remaining repair classes

## Outcome

At dispatched revision `c81913747f78ead2c3bf57ab0ea0208543105d94`, the
repository has one reusable transaction envelope, two mature source-mirror
mutation kernels, a deterministic structural-source repair, and an exact-name
ordinary-review route. It does **not** yet have one generic interpreter that can
execute the relationship and lifecycle operations already declared by the
typed repair schema.

The smallest safe extension is therefore:

1. add one closed **source-target reconciliation** program beneath
   `apply_signed_manifest`, shared by the 23 dual-bound sources, the 1 scalar
   mismatch, and the 3 owner/geometry migrations;
2. add one closed **structural-source revival** program beneath the same
   envelope for `electron_diffusivity` and `ion_diffusivity`; and
3. use the generic executor's existing `set_properties` operation only for the
   exact deterministic staging/restamp of the 9-name review cohort, then use the
   existing exact-name ordinary-review path for the single joint draw.

This is two new closed branch programs, not a second manifest framework and not
four bespoke operators. The two owner/geometry rows requiring semantic review
join the same exact-name review route before any source-target reconciliation.

The inspection names **22 reuse candidates** below. Every candidate has a
repo-relative file-and-line anchor at this revision; the registry/interpreter
candidate deliberately has separate declaration and execution anchors. The
recorded resolver run checked **23/23 anchors, 0 unresolved**; its complete output is
`anchor-resolution.log` in the node delivery directory.

## Class decision table

| Remaining class | Live scope from the plan | Does generic `apply_signed_manifest` already serve it? | Closed-program verdict |
|---|---:|---|---|
| Dual-bound retarget | 23 multiple-live-target sources plus 1 scalar mismatch | **No.** The executor cannot add/delete relationships, recompute projections, or rebuild source-path mirrors; the current generic registry interprets only `set_properties`, `delete`, `supersede`, and `detach`. | **One new source-target reconciliation branch is required.** Reuse the fan-out-aware semantic-source repair kernel, but make the signed target explicit and protect every losing target's complete producer closure before deleting any edge. |
| Owner/geometry migration | 3 exact migrations plus 2 semantic reviews | **No for the migrations.** `set_properties` cannot move the synchronized source relationship/scalar/DD-projection/source-path closure. | **Reuse the same source-target reconciliation branch; do not create an owner/geometry-specific program.** The 3 sole-predecessor rows use the exact retarget kernel; the 2 unresolved owner identities go through the existing joint-review route first. |
| Bare-structural cleanup | 2 names: `electron_diffusivity`, `ion_diffusivity` | **No.** Both have stale `derived:<name>` sources; the current structural reconcile deliberately refuses stale revival, and the generic executor cannot transition that source and recreate `PRODUCED_NAME` atomically. | **One new structural-source revival branch is required.** It should reuse the canonical derived-source metadata, admission classifier, and exact batched cardinality query while signing stale-state, live-child, lifecycle, and receipt closure. |
| Joint review draw | 9 exact target identities, once as one cohort | **Yes for deterministic graph staging; not applicable to the paid draw itself.** Generic `set_properties` can atomically restage/restamp the exact signed identities. | **No new mutation program.** Use the existing exact-name scope and ordinary `review_name` claim/process/persist chain for one bounded invocation. Do not call the single-name `sn rescore` path nine times and do not use DD-source `--drain-batch` scope. |

## Reuse candidates

### Shared signed authority and transaction envelope

| Candidate | Repository anchor | One-line fitness verdict |
|---|---|---|
| `apply_signed_manifest` | `imas_codex/standard_names/signed_manifest.py:2469` | **Reuse as the sole public envelope:** it already binds file and payload digests, re-reads inside the applying transaction, locks, verifies collateral/counters, writes receipts, rolls back, and recognizes replay. |
| Typed `RepairMutationKind` vocabulary | `imas_codex/schemas/standard_name.yaml:787` | **Reuse and implement, do not redesign:** the schema already declares `add_relationship`, `delete_relationship`, `recompute_projection`, `clear_source_paths`, `change_source_lifecycle`, and `materialize_derived_parent`. |
| Generic mutation registry and interpreter | `imas_codex/standard_names/signed_manifest.py:49`; `imas_codex/standard_names/signed_manifest.py:2091` | **Extend in closed branches:** the loader/executor currently admits only four kinds, which is the precise reason the first three remaining classes are not expressible today. |
| Canonical DD source-authority capture | `imas_codex/standard_names/source_authority.py:488` | **Reuse for every DD-backed source row:** extend its signed closure with every current live target, the prospective target, backing projection, and each losing target's full incoming producer set. |
| Canonical participant locks | `imas_codex/standard_names/source_authority.py:562` | **Reuse unchanged where possible:** lock the exact participant set and require exact cardinality before re-reading canonical bytes; add absent/stale structural-source state as a typed branch rather than weakening the lock. |

### Dual-bound and owner/geometry source reconciliation

| Candidate | Repository anchor | One-line fitness verdict |
|---|---|---|
| `repair_semantic_source_invariants` | `imas_codex/standard_names/provenance_lifecycle.py:1376` | **Best mutation kernel for the 23+1 fan-out class, not sufficient authority:** it already repairs edge, scalar, backing projection, caches, receipts, and exact-batch rollback, but its inferred/override target policy lacks signed losing-name producer closure. |
| `retarget_standard_name_sources` | `imas_codex/standard_names/provenance_lifecycle.py:310` | **Best mutation kernel for the 3 sole-predecessor owner rows:** it is exact, compare-and-set, guarded, mirror-complete, and idempotent, but it cannot consume an already dual-bound source and does not itself protect an orphaned predecessor. |
| `guard_source_pairings` | `imas_codex/standard_names/attachment_audit.py:595` | **Reuse inside the locked transaction with exact-set semantics:** every requested source must be admitted for the signed target; an admitted subset or receipt-less `already_bound` result is a whole-program refusal. |
| `bind_sources_exclusively` | `imas_codex/standard_names/provenance_lifecycle.py:887` | **Reject as the public repair path:** it silently narrows to the guard-admitted subset and lacks signed old-state, losing-target closure, immutable receipt identity, and complete old-target cache repair. |
| `reconcile_standard_name_dd_edges` | `imas_codex/standard_names/graph_ops.py:12770` | **Regression net only:** it is global, projection-only, and non-manifested; after either source-target program it must report zero work for the repaired exact cohort. |

The source-target branch must select the kernel by signed pre-state, not by
caller preference: a source with several live targets uses the semantic-source
repair kernel with one signed authoritative target; a source with exactly one
signed predecessor uses the retarget kernel. Both branches share one authority
shape, one lock/re-hash envelope, the exact pairing guard, losing-target
last-producer protection, mirror-complete postflight, and one receipt policy.

### Bare structural-source cleanup

| Candidate | Repository anchor | One-line fitness verdict |
|---|---|---|
| `classify_orphan_parent_source_candidates` | `imas_codex/standard_names/graph_ops.py:25264` | **Reuse as the semantic admission preflight:** it distinguishes derived scaffolds owned by parent admission from pipeline/catalog parents whose provenance must remain origin-owned. |
| `_derived_parent_source_metadata` | `imas_codex/standard_names/graph_ops.py:3630` | **Reuse as canonical source identity:** it prevents the recovery branch from inventing a second representation of `derived:<parent>`. |
| `reconcile_orphan_parent_sources` | `imas_codex/standard_names/graph_ops.py:25304` | **Do not call as the applying operator for these two rows:** its global scalar loop is intentionally idempotent for absent/non-stale sources and correctly refuses a stale tombstone instead of reviving it implicitly. |
| `reconcile_orphan_parent_sources_batched` | `imas_codex/standard_names/graph_ops.py:25406` | **Extract/reuse its exact-cohort write and cardinality check:** it already creates the canonical source, binding, and optional event atomically, but it also refuses stale sources and has no signed preview/replay envelope. |

The structural branch must sign both stale sources exactly, prove the two
parents remain accepted and childful, prove no competing producer exists,
transition each source from the explicitly signed stale state, recreate exactly
one `PRODUCED_NAME` relationship and scalar, emit exactly two immutable change
rows, and replay with zero writes. It must not make the general reconcile start
reviving arbitrary stale derived sources; that would erase the distinction
between a lifecycle tombstone and explicit recovery authority.

### Nine-name joint ordinary-review draw

| Candidate | Repository anchor | One-line fitness verdict |
|---|---|---|
| `stage_name_for_rescore` | `imas_codex/standard_names/graph_ops.py:23219` | **Reuse as the property-policy template, not nine sequential calls:** it clears stale score/diagnosis/claims while preserving spent refine attempts, but its public contract is one reviewed/exhausted identity at a time. |
| `rescore_name` | `imas_codex/standard_names/edit.py:2884` | **Reject as the cohort driver:** nine independent invocations can partially stage or spend before a later identity refuses, splitting the single permitted cohort draw. |
| `scope_exact_standard_names` | `imas_codex/standard_names/graph_ops.py:1786` | **Reuse for exact cohort fencing:** one preflight covers the complete requested set and lineage, and one compare-and-set write stamps exactly those identities or none. |
| CLI `sn run --name` scope | `imas_codex/cli/sn.py:1250` | **Reuse as the public bounded invocation:** repeatable exact identities are preflighted atomically and routed through normal pools without seeding DD sources. |
| DD-source `--drain-batch` scope | `imas_codex/cli/sn.py:1294` | **Do not use for this draw:** its authority is a DD-source manifest and can scope the sources' current targets rather than the nine reviewed target identities. |
| `claim_review_name_batch` | `imas_codex/standard_names/graph_ops.py:14574` | **Reuse after one signed restage:** it supplies claim fencing, exact run scope, real-description and validity gates, and review/refine exclusivity. |
| `process_review_name_batch` | `imas_codex/standard_names/workers.py:7921` | **Reuse unchanged for the draw:** this is the ordinary RD-quorum path, including current grammar context, provider accounting, failure release, and no hand acceptance. |
| `persist_reviewed_name` | `imas_codex/standard_names/graph_ops.py:14740` | **Reuse unchanged for outcomes:** token/sequence compare-and-set persistence records the score and lifecycle decision rather than treating invocation completion as acceptance. |

The deterministic precursor should be one signed 9-row `set_properties`
authority derived from the staged machine-readable block and the current
rescore policy: exact target identities, descriptions, validation state,
review-stage transition, cleared stale review diagnosis/claims, preserved
refine-attempt spend, exact pre-state fingerprints, and exact receipt count.
After replay proves zero additional writes, invoke the exact nine through the
ordinary name-review pool once. The required result is nine accounted outcomes,
not nine acceptances; below-threshold or provider-refused rows remain visible
qualified results.

## Program boundaries and execution order

1. Persist/replay the exact 9-row deterministic review staging authority with
   generic `set_properties`; then run one exact-name `review_name` invocation.
2. Fold the two owner semantic outcomes into the owner/geometry adjudication.
3. Sign one source-target authority covering all ready dual-bound, scalar, and
   owner/geometry rows; partition internally by fan-out versus sole predecessor,
   but apply one serialized, all-or-nothing branch program per independently
   adjudicated cohort.
4. After stale derived-source prevention has settled, sign and apply the exact
   two-row structural-source revival branch.
5. Require the graph-wide source/projection/structural ratchets and exact replay
   checks to report no new work; do not use a global reconcile to finish a
   partially written transaction.

## Explicit non-reuse decisions

- Do not add a second manifest format or accept authority-supplied Cypher.
- Do not let `repair_semantic_source_invariants` infer the winning identity for
  these adjudicated rows; the signed authority names it.
- Do not use `bind_sources_exclusively` on either migration class.
- Do not weaken stale structural-source refusal globally; revive only the exact
  signed two-row cohort.
- Do not run `sn rescore` once per target and do not spend on the four currently
  eligible names before the exact nine are ready together.
- Do not treat a review invocation as acceptance. Ordinary quorum scores and
  fail-closed provider or claim outcomes remain the authority.

## Anchor verification

The resolver checked the anchor line plus a symbol/token pattern for all 22
candidates against commit `c81913747f78ead2c3bf57ab0ea0208543105d94`.
Headline result: **23 anchors resolved, 0 unresolved**. The log is intentionally kept as
node evidence rather than committed product documentation:

`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T194833268178-repairreuse/anchor-resolution.log`
