# Restore receipt reuse map

## Authority and scope

Derived from the live plan `paid-identity-deletion-protection` at version 64
and the source tree at `4c9d1dabb4ab0afb679ccf15b85eb552e585169c` (HEAD, read
from disk in this worktree). This map performs no graph write, runs no restore,
and changes no source file. Every `file:line` below was read at that revision.

It answers one question for the open followup
`f-pidp-the-restore-receipt-outlives-its-transaction`: what already exists that
the followup can reuse, so that nothing is rebuilt? The followup's premise is
that the parity guard proves a complete restore and then discards what it
proved. Part of that premise has since been closed — the receipt now carries
the per-identity role tally — so this map records what remains.

## The adapter and its receipt

`_apply_archive_reconstruction` at
`imas_codex/standard_names/signed_manifest.py:7235` is the whole restore
transaction. It is reached only through `apply_signed_manifest`
(`imas_codex/standard_names/signed_manifest.py:7400`), which dispatches on
`authority_adapter == _ARCHIVE_RECONSTRUCTION_ADAPTER`
(`imas_codex/standard_names/signed_manifest.py:7427`; the constant is at
`imas_codex/standard_names/signed_manifest.py:178`) and calls the adapter at
`imas_codex/standard_names/signed_manifest.py:7436`.

The receipt is the dict each return path builds. Its schema label is
`SIGNED_MANIFEST_RECEIPT_SCHEMA = "imas-codex.signed-repair-receipt.v1"` at
`imas_codex/standard_names/signed_manifest.py:54`.

### Receipt fields, preview path (outcome `would_apply` or `refused`)

Built at `imas_codex/standard_names/signed_manifest.py:7292`–`:7302`, returned
after the transaction rolls back. In every table below, a bare
`signed_manifest.py:NNN` abbreviates
`imas_codex/standard_names/signed_manifest.py:NNN`.

| Field | Line | Content |
|---|---|---|
| `schema` | `signed_manifest.py:7293` | `imas-codex.signed-repair-receipt.v1` |
| `outcome` | `signed_manifest.py:7294` | `"refused"` when any refusal exists, else `"would_apply"` |
| `changed` | `signed_manifest.py:7295` | always `0` on this path |
| `would_change` | `signed_manifest.py:7296` | `0` when refused, else the authority row count |
| `counts` | `signed_manifest.py:7297` | the aggregate dict below |
| `identity_roles` | `signed_manifest.py:7298` | per-identity roles with an empty `reinstated` tally |
| `refusals` | `signed_manifest.py:7299` | one entry per refused authority row, carrying `row_id` |
| `manifest` | `signed_manifest.py:7300` | the freshly derived closure the digest is taken over |
| `manifest_sha256` | `signed_manifest.py:7301` | the authorization a later apply must name |

The `counts` aggregate is built once at
`imas_codex/standard_names/signed_manifest.py:7283`–`:7289`:

| Sub-field | Line | Content |
|---|---|---|
| `authority_rows` | `signed_manifest.py:7284` | archived identities in the authority artifact |
| `counterpart_rows` | `signed_manifest.py:7285` | counterpart records the artifact carries |
| `admitted` | `signed_manifest.py:7286` | authority rows minus refused rows |
| `refused` | `signed_manifest.py:7288` | number of refused rows |

### Receipt fields, applied path (outcome `applied`)

Built at `imas_codex/standard_names/signed_manifest.py:7376`–`:7389`, returned
only after the parity guard passes and the transaction commits.

| Field | Line | Content |
|---|---|---|
| `schema` | `signed_manifest.py:7377` | `imas-codex.signed-repair-receipt.v1` |
| `outcome` | `signed_manifest.py:7378` | `"applied"` |
| `changed` | `signed_manifest.py:7379` | authority row count |
| `mutations` | `signed_manifest.py:7380`–`:7384` | authority + counterpart + edge writes committed |
| `counts` | `signed_manifest.py:7385` | the same aggregate dict as the preview |
| `identity_roles` | `signed_manifest.py:7386` | per-identity role outcomes, with a non-empty `reinstated` tally |
| `refusals` | `signed_manifest.py:7387` | always `[]` — a refusal never reaches this path |
| `manifest_sha256` | `signed_manifest.py:7388` | the digest the apply was authorized against |

The two paths are **not** field-symmetric: the preview carries `manifest` and
`would_change`; the applied path carries `mutations` and no `manifest`. A caller
reading a receipt must branch on `outcome` rather than assume one shape.

### Where `identity_roles` comes from

`_archive_role_outcomes` at
`imas_codex/standard_names/signed_manifest.py:7200` builds
`{identity: {"reinstated": {role: count}, "unreinstatable": {role: count}}}`.
`reinstated` is filtered from the live per-type tally the parity guard read;
`unreinstatable` is every role the archive record named that the reconstruction
registry has no route for (`imas_codex/standard_names/signed_manifest.py:7223`–
`:7227`).

The parity guard at `imas_codex/standard_names/signed_manifest.py:7363`–`:7374`
builds `expected` from the registry keys with an explicit zero for an absent
role (`imas_codex/standard_names/signed_manifest.py:7365`), reads `live`
through `_archive_edge_counts`
(`imas_codex/standard_names/signed_manifest.py:7185`), and raises
`SignedManifestConflict` on any difference
(`imas_codex/standard_names/signed_manifest.py:7370`). The `reinstated` record
and the guard are the same measurement — the guard's local variable is now
returned instead of discarded.

The archived role counts enter through the authority artifact's `archive_roles`
field, loaded by `_load_archive_role_counts` at
`imas_codex/standard_names/signed_manifest.py:6896`. That loader is why an
unroutable role can still be reported: a role outside the registry is
dimensioned by the archive record rather than by an edge, and a registry role
the reconstruction edges under-cover is refused
(`imas_codex/standard_names/signed_manifest.py:6934`–`:6938`).

## Registry coverage: `EVIDENCED_BY`

| Question | Answer at HEAD |
|---|---|
| Is `EVIDENCED_BY` absent from the reconstruction registry? | **Yes — still absent.** |
| The registry symbol | `_ARCHIVE_EDGE_COUNTERPARTS`, `dict[str, dict[str, str]]` at `imas_codex/standard_names/signed_manifest.py:205`, entries `:206`–`:239`, **34 roles** |
| Is the registry readable, i.e. is this an absence over something present? | Yes. Entries adjacent to the would-be position read: `HAS_UNIT` `signed_manifest.py:211`, `HAS_PHYSICS_DOMAIN` `signed_manifest.py:213`, `HAS_PARENT` `signed_manifest.py:232`, `HAS_DOCS_REVIEW_ADMISSION` `signed_manifest.py:239`. The 34-entry count is from the source text at this revision (34 keys at lines 206–239). |
| Does the role exist in the graph schema registry? | Yes — the *other* registry. `RELATIONSHIPS` at `imas_codex/graph/schema_context_data.py:2048` carries `('DDResolution', 'EVIDENCED_BY', 'DDGap', 'one')` at `imas_codex/graph/schema_context_data.py:2075` and `('PromotionCandidate', 'EVIDENCED_BY', 'StandardName', 'many')` at `imas_codex/graph/schema_context_data.py:2146`. So the absence is a missing **reconstruction route**, not a missing schema relationship. |
| What writes the identity-facing edge? | `imas_codex/standard_names/vocab_promotion.py:224` writes `MERGE (pc)-[:EVIDENCED_BY]->(sn)` — the promotion-candidate edge a restored `StandardName` could have carried and cannot get back. |

Two facts that must not be conflated, because the first is settled and the
second is not:

1. **The reconstruction registry has no route for `EVIDENCED_BY`** —
   established by reading `_ARCHIVE_EDGE_COUNTERPARTS` at this revision.
   Consequences, each read rather than inferred: the loader refuses any edge of
   that type (`imas_codex/standard_names/signed_manifest.py:7067`); the parity
   guard counts only registry roles, so the loss is invisible to it
   (`imas_codex/standard_names/signed_manifest.py:7365`); and the receipt prints
   the role under `unreinstatable` when the archive record names it (test
   `test_archive_reconstruction_receipt_names_unreinstatable_archived_roles`,
   `tests/standard_names/test_archive_reconstruction.py:538`).
2. **Whether any archived identity in the restore set actually held one** — not
   established. Live reads cannot settle it: the deletion record covers none of
   the cohort and the live identity-facing slice is 0 edges. The only read that
   can is the 2026-09-06 archive dump through `start_temp_neo4j`
   (`imas_codex/graph/temp_neo4j.py:207`), named in
   `docs/evidence/paid-identity-deletion-protection/unreinstatable-role-gap.md`.
   **This map does not read the archive and does not assert a count.**

Adding a route means two edits in the same dict family: a role entry in
`_ARCHIVE_EDGE_COUNTERPARTS` (`imas_codex/standard_names/signed_manifest.py:205`)
and a counterpart label in `_ARCHIVE_RECONSTRUCTABLE_COUNTERPART_LABELS`
(`imas_codex/standard_names/signed_manifest.py:186`, currently a `frozenset` of
`StandardNameReview`, `DocsRevision`, `StandardNameSource`), because an edge
cannot exist without both endpoints. That widening is exactly what the
adapter's refusals exist to bound; it is a decision the plan does not settle and
must not be taken unilaterally.

## CLI entry point and composer

| Question | Answer at HEAD |
|---|---|
| Does a CLI entry point for the adapter exist? | **No.** The string `archive-reconstruction` occurs in exactly two files in the tree: `imas_codex/standard_names/signed_manifest.py` (the constant at `:178`, plus Cypher comment tags) and `tests/standard_names/test_archive_reconstruction.py` (`_ADAPTER = "archive-reconstruction"`, `tests/standard_names/test_archive_reconstruction.py:21`). |
| Who calls `apply_signed_manifest` with this adapter? | **Only the test module.** Every in-package call site names another adapter — `imas_codex/standard_names/graph_ops.py:3433`, `:4552`, `:5311`, `:15056`, `:21547`, `:21871`, `:21900`, `:22377`, `:22410` and `imas_codex/standard_names/provenance_lifecycle.py:994` — and none is `archive-reconstruction`. Nothing under `imas_codex/cli/` references the adapter or `signed_manifest` at all. |
| Does a committed composer for `imas-codex.archive-reconstruction.v1` exist? | **No.** The schema constant is at `imas_codex/standard_names/signed_manifest.py:185` and is matched as an accepted input by the loader; nothing in the tree writes that artifact. The followup's statement holds: the composer lived out of tree, so the archive-based route depends on an uncommitted input. |

The adapter is therefore reachable programmatically and by test only. An
operator wanting a restore has no command to run, and an archive must be turned
into a signed authority by hand.

## Verdict table

| Need of the followup | Existing entry point | Verdict |
|---|---|---|
| Emit a receipt that outlives the transaction | `_apply_archive_reconstruction` `imas_codex/standard_names/signed_manifest.py:7235`; returns at `:7292` (preview) and `:7376` (applied) | **REUSE** — the receipt exists and carries `identity_roles` on both paths |
| Per-identity role tally, with unroutable roles named | `_archive_role_outcomes` `imas_codex/standard_names/signed_manifest.py:7200`; archive-role channel `_load_archive_role_counts` `imas_codex/standard_names/signed_manifest.py:6896` | **REUSE** — the tally and the unroutable-role channel both landed; the followup's central premise is closed | 
| Per-type parity as the anti-partial-restore gate | `imas_codex/standard_names/signed_manifest.py:7363`–`:7374` | **REUSE** — keep it; it is the guard that made the first live restore refuse 13 of 90 edges rather than commit them |
| Restore an `EVIDENCED_BY` edge | none — absent from `_ARCHIVE_EDGE_COUNTERPARTS` `imas_codex/standard_names/signed_manifest.py:205` | **CANNOT REUSE** — no route; adding one is a registry widening the plan does not settle, and the archive read that would size the loss is a separate node |
| CLI entry point for the adapter | none | **CANNOT REUSE** — nothing to reuse; a new command is the work |
| Compose the signed archive-reconstruction artifact | none | **CANNOT REUSE** — the composer is out of tree; this is the uncommitted input the followup names |
| Record a deletion so a future restore can be reconstructed | `deletion_change_cypher` in `imas_codex/standard_names/provenance_lifecycle.py` (two of ten delete routes bypass it — followup `f-pidp-two-delete-routes-bypass-the-snapshot-writer`) | **reuse the writer; not this node's concern** — named so the followup does not re-derive it |

## Invariants the followup must not weaken

- A refusal short-circuits the apply: the preview path rolls back and returns
  the refusal set rather than a partial success
  (`imas_codex/standard_names/signed_manifest.py:7290`).
- The parity guard is per-identity and per-type over the registry's 34 roles,
  with an explicit zero for an absent role
  (`imas_codex/standard_names/signed_manifest.py:7365`). A role cannot be
  dropped from the count to make a restore pass.
- A role outside the registry is reported, never dropped
  (`imas_codex/standard_names/signed_manifest.py:7223`), and an archived role
  the edges under-cover is refused rather than reinstated partially
  (`imas_codex/standard_names/signed_manifest.py:6934`).
- The receipt is a return value on this path, not a persisted row: no node
  records it. If the followup needs the account to survive the process, that is
  a new durable write and needs its own fence and its own test.

## Proposed exclusive write-path set

For the implementing node that takes
`f-pidp-the-restore-receipt-outlives-its-transaction`, split by beat. Each beat
is one worker, one worktree, one exclusive set; the landing files go to the node
that lands last. `signed_manifest.py` is 7,878 lines / ~333 KB, within a
worker's context window, but it is one module holding the whole adapter
transaction — the three beats below must not run concurrently, because all
three would write it.

**Beat A — receipt durability / the account outliving its transaction**

- `imas_codex/standard_names/signed_manifest.py`
- `tests/standard_names/test_archive_reconstruction.py`
- `docs/evidence/paid-identity-deletion-protection/restore-receipt-roles.md`

**Beat B — committed composer for `imas-codex.archive-reconstruction.v1`**

- `imas_codex/standard_names/signed_manifest.py`
- `tests/standard_names/test_archive_reconstruction.py`
- `docs/evidence/paid-identity-deletion-protection/restore-receipt-reuse-map.md` (this map)

**Beat C — CLI entry point**

- `imas_codex/cli/sn.py` (the `sn` surface this repo already routes
  restore-shaped operations through)
- the covering test module under `tests/cli/`
- `docs/evidence/paid-identity-deletion-protection/restore-receipt-reuse-map.md`

**Beat D — `EVIDENCED_BY` sizing (independent; read-only, no graph write)**

- the archive dump read through `start_temp_neo4j`
  (`imas_codex/graph/temp_neo4j.py:207`) — no source change expected
- `docs/evidence/paid-identity-deletion-protection/unreinstatable-role-gap.md`

**Every beat lands through**

- `docs/evidence/archive/paid-identity-deletion-protection-landed.html`
- `docs/plans/paid-identity-deletion-protection.html`

Beat D is independent of A–C and may run beside one of them. Not in any set,
and named so it is not forgotten: the stale sibling
`docs/evidence/paid-identity-deletion-protection/restore-reuse-map.md`, which
still reads CANNOT REUSE on the archive reconstruction that has since landed and
whose line references no longer match. Correcting it is a one-file write; give
it its own node rather than folding it into Beat B, so a stale-record repair is
one file's owner.

## Headline result

The followup's central premise — the parity guard proves a complete restore and
then discards what it proved — **no longer holds and must not be re-derived**:
`identity_roles` carries the per-identity per-role record on both receipt paths
(`imas_codex/standard_names/signed_manifest.py:7298`, `:7386`), fed by the
guard's own tally (`:7363`–`:7374`) through `_archive_role_outcomes` (`:7200`).
`EVIDENCED_BY` is still absent from the reconstruction registry
`_ARCHIVE_EDGE_COUNTERPARTS` (`imas_codex/standard_names/signed_manifest.py:205`,
34 roles), so the guard cannot see it and the receipt reports it unreinstatable;
whether any archived identity held one remains unmeasured and needs the archive
read. No CLI entry point and no committed composer exist for the adapter at this
revision.