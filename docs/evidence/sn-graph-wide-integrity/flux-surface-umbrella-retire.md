# Flux-surface umbrella retirement — sequential signed invocations

Node `umbrellaseq`, plan `sn-graph-wide-integrity` §3c.

Goal: retire the derived-parent umbrella `area_of_flux_surface` through three
sequential single-kind signed invocations — (1) relocate its two
flux-coordinate derivative children onto
`poloidal_plane_cross_sectional_area_of_flux_surface`, (2) release
`surface_area_of_flux_surface` to no parent, (3) supersede the now-childless
umbrella into `poloidal_plane_cross_sectional_area_of_flux_surface`.

## Starting state (read before any mutation)

`area_of_flux_surface` (`name_stage=accepted`) had exactly three live
`HAS_PARENT` children:

| Child | Producers | Parent before |
|---|---|---|
| `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface` | 1 (`dd:equilibrium/time_slice/profiles_1d/darea_dpsi`) | `area_of_flux_surface` |
| `derivative_with_respect_to_toroidal_flux_coordinate_of_area_of_flux_surface` | 1 (`dd:equilibrium/time_slice/profiles_1d/darea_drho_tor`) | `area_of_flux_surface` |
| `surface_area_of_flux_surface` | 20 | `area_of_flux_surface` |

## Invocation 1 — relocate the two derivative children (landed)

Mechanism: the closed `recompute_projection` structural-reparent program in
`imas_codex/standard_names/signed_manifest.py` (`_validate_structural_reparent_program`,
`_apply_structural_reparent`), which already exists in this checkout and is
exercised by `tests/standard_names/test_structural_reparent.py`. It is
`all_or_nothing` by construction for a pure-reparent authority
(`_all_or_nothing`).

Authority: 2 rows, one per derivative child, each a single
`recompute_projection` mutation moving its `HAS_PARENT` edge from
`area_of_flux_surface` to `poloidal_plane_cross_sectional_area_of_flux_surface`,
with the exact incumbent relationship properties preserved
(`operator_kind`, `operator`).

- **Preview** (`authority_file_sha256=0c484544fac80c13c6e1d87c557d7c642b8aca0d6fedfec7022c0ef3bcd66bb7`,
  `authority_payload_sha256=ce1f5a2f691fd537a01bb14b98c9f0160602bae5eda637ba6c01af6a1407baaf`):
  `outcome=would_apply`, `counts={authority_rows: 2, admitted: 2, refused: 0}`,
  `would_change=2`, `refusals=[]`,
  `manifest_sha256=d5898c53b468a6c782a1ecaf0d8df452799ade70a5ea72d0448244afcbe34873`.
- **Apply**: `outcome=applied`, `changed=2`, `mutations=2`, `receipt_rows=2`,
  `persistent_writes=4` (2 relationship creates + 2 relationship deletes, one
  pair per relocated edge).
- **StandardNameChange delta**: 2 new `StandardNameChange` rows with
  `operation=reparent_structural_standard_name_children` and
  `manifest_sha256=d5898c53b468a6c782a1ecaf0d8df452799ade70a5ea72d0448244afcbe34873`,
  one per row_id — equal to `receipt_rows`.
- **Replay** (same manifest_sha256): `outcome=already_applied`, `changed=0`,
  `persistent_writes=0`, `receipt_rows=2` — confirms idempotency.
- **Post-state**: both derivative children now read
  `parent_id=poloidal_plane_cross_sectional_area_of_flux_surface`;
  `area_of_flux_surface` live children dropped from 3 to 1
  (`surface_area_of_flux_surface` only). Neither derivative child's
  `name_stage`/`status`/producers changed — only the `HAS_PARENT` edge moved.

**Admitted 2 / refused 0, against 2 in-invocation derived rows.**

## Invocation 2 — release `surface_area_of_flux_surface` to no parent (BLOCKED — no closed program exists)

The plan's own comment `c-20260822-umbrella-survives` named this precisely:
*"the signed-manifest relationship registry is closed to StandardNameSource
`PRODUCED_NAME` reconciliation, revival and migration; it cannot create or
delete StandardName `HAS_PARENT` edges... Any future reparenting therefore
needs a fourth closed program."* Invocation 1 shows that fourth program
(`recompute_projection`) now exists and covers **relocation between two live
parents**. It does not cover **release to no parent** — that is a different
shape (one node, one relationship, one delete, no replacement edge), and no
registry entry admits it.

Verified directly rather than assumed. Two authority shapes were built
against the live `surface_area_of_flux_surface` → `area_of_flux_surface`
edge and both were rejected by the registry before any mutation ran
(read-only `apply_signed_manifest` preview call, no `apply=True`):

1. A single `delete_relationship` mutation on the `HAS_PARENT` edge with
   only the `out-of-allowlist-immutability` guard →
   `SignedManifestAuthorityError: repair row 'surface_area_of_flux_surface'
   is missing guards: last-producing-source` (any row carrying
   `delete_relationship` unconditionally requires that guard, per
   `_load_authority`'s `required_guards` derivation).
2. The same mutation with the required `last-producing-source` guard added →
   `SignedManifestAuthorityError: repair row 'surface_area_of_flux_surface'
   is not a closed source-target reconciliation program`. This is the
   **only** registry program that currently admits a bare
   `delete_relationship` (`_validate_source_target_reconciliation_program`),
   and it is hard-scoped to `StandardNameSource --PRODUCED_NAME--> StandardName`
   reconciliation: it requires `identity.kind == "source"`, a
   `StandardNameSource` participant, and at least two `PRODUCED_NAME`
   bindings with a surviving target. A `StandardName --HAS_PARENT-->
   StandardName` release has none of that shape — `identity.kind` is
   `"standard_name"`, there is no `StandardNameSource` participant, and the
   validator raises rather than passing through.

No other registered program (`add_relationship`/structural-source-revival,
the paired delete+add ordinary-source-migration, or
`recompute_projection`/structural-reparent, which requires `old_end_id !=
new_end_id` with both being live `StandardName` nodes) admits a
release-to-no-parent shape either. **This is a genuine machinery gap, not an
authority or scope ambiguity**, and closing it means adding a fifth closed
program (validate + apply + postcondition-verify functions) to
`imas_codex/standard_names/signed_manifest.py` plus its disposable-graph
contract test — a code change outside this node's write scope (fenced to
this evidence file only) and outside its remaining time budget.

**Invocation 2 did not run. 0 rows admitted, 0 refused — the authority was
never accepted past registry validation, so no preview or apply outcome
exists to report.**

## Invocation 3 — supersede the umbrella (BLOCKED — precondition not met)

Per the fence, invocation 3 is attempted only if `area_of_flux_surface`'s
live child count reads 0. Re-read immediately before considering it:

**`area_of_flux_surface` live child count = 1** (`surface_area_of_flux_surface`,
unchanged from before invocation 1 — it was never a derivative child and was
not touched). The precondition is not met, so invocation 3 was correctly not
attempted.

## Summary

| Invocation | Outcome | Admitted / refused | Receipt rows | StandardNameChange delta | Replay |
|---|---|---|---|---|---|
| 1. Relocate 2 derivative children | **Applied** | 2 / 0 | 2 | 2 | `already_applied`, `changed=0`, `persistent_writes=0` |
| 2. Release `surface_area_of_flux_surface` | **Blocked — no closed program** | n/a (rejected pre-preview) | — | 0 | not run |
| 3. Supersede `area_of_flux_surface` | **Blocked — precondition (live child count = 1, not 0)** | — | — | 0 | not run |

`surface_area_of_flux_surface` retains all 20 producers and its lifecycle
(`name_stage=accepted`, `status=draft`, `origin=catalog_edit`) is unchanged
throughout — it was never touched by invocation 1 and invocation 2 never
reached the graph. Both derivative children's lifecycle is likewise
unchanged; only their `HAS_PARENT` edge target moved.

## Follow-on (out of this node's scope)

Add a fifth closed program to the signed-manifest registry —
`release_structural_standard_name_child` or similar — admitting exactly one
`delete_relationship` mutation on a `HAS_PARENT` edge with no replacement
edge, gated by `last-producing-source` +
`out-of-allowlist-immutability` (+ a guard confirming the released child
remains independently valid with no parent, mirroring the
`no-live-structural-child` guard's inverse). Once that lands, invocation 2
can run, which will drop `area_of_flux_surface`'s live child count to 0 and
unblock invocation 3 (supersede into
`poloidal_plane_cross_sectional_area_of_flux_surface`).
