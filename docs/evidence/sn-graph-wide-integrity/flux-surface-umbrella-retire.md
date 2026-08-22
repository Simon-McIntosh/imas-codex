# `area_of_flux_surface` umbrella retirement — blocked on a single-program authority gate

## Requested measure

One signed invocation was to: relocate the two flux-coordinate derivative
children (`derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface`,
`derivative_with_respect_to_toroidal_flux_coordinate_of_area_of_flux_surface`)
onto `poloidal_plane_cross_sectional_area_of_flux_surface`; release
`surface_area_of_flux_surface` to no parent; then supersede the now-childless
`area_of_flux_surface` into `poloidal_plane_cross_sectional_area_of_flux_surface`
— refusing the whole cohort if any single relocation is refused.

## Finding: the landed machinery cannot express this as one authority

`imas_codex/standard_names/signed_manifest.py` (owned by a concurrent node in
this session; read-only here) landed the closed `recompute_projection`
structural-reparent program at commit `638fb1c6` (merged to `main` at
`f6297ccb`, present in this worktree). That program is exactly what the two
derivative-child relocations need. But its loader enforces exclusivity, not
composability, with every other mutation kind in the same authority file:

```python
structural_reparent_rows = [
    row
    for row in loaded_rows
    if any(mutation["kind"] == _STRUCTURAL_REPARENT for mutation in row.mutations)
]
if structural_reparent_rows and len(structural_reparent_rows) != len(loaded_rows):
    raise SignedManifestAuthorityError(
        "structural reparent authority cannot mix mutation programs"
    )
```

(`imas_codex/standard_names/signed_manifest.py:730-738` in this worktree's
`main`.) An authority containing any `recompute_projection` row must contain
**only** `recompute_projection` rows. `detach` / `delete_relationship`
(release-to-no-parent) and `supersede` (retirement) are validated by the same
generic `_load_authority` path and are individually well-formed, but the
all-or-nothing exclusivity check above refuses the moment one row in the file
is a reparent and another is a detach or supersede. There is no `authority
composition` or multi-program envelope in this module — `apply_signed_manifest`
dispatches a whole file to exactly one of four mutually exclusive adapters
(generic, stale-source, refused-target-orphan, error-sibling), and the generic
adapter itself refuses mixed reparent/non-reparent rows before any preview or
graph read occurs. This is a static, deterministic refusal on the authority
JSON — no live-graph credential or query is needed to demonstrate it, and it
was independently confirmed by reading the loader rather than by mutation
(the design doc's `structural-reparent-program: 638fb1c6` follow-on already
anticipated exactly this gap: "the signed registry cannot create or delete
`HAS_PARENT` edges [with detach/supersede in the same operation]... this needs
a fourth closed program with all-or-none child relocation").

Concretely, the two available compositions are:

1. **Reparent-only authority** (2 rows, both `recompute_projection`): can
   relocate the two derivative children onto
   `poloidal_plane_cross_sectional_area_of_flux_surface` in one all-or-nothing
   invocation. Well-formed today.
2. **Detach + supersede authority** (2 rows, `delete_relationship` +
   `supersede`, generic non-reparent path, `structural-legitimacy` +
   `last-producing-source` + `out-of-allowlist-immutability` guards): can
   release `surface_area_of_flux_surface` and then retire the now-childless
   `area_of_flux_surface` in one all-or-nothing invocation, **but only after**
   composition (1) has already landed and been re-verified live, because the
   `structural-legitimacy` guard on `supersede` requires zero live
   `HAS_PARENT` children at signing time.

Doing all four mutations (2 reparents + 1 detach + 1 supersede) as **one**
signed invocation is not expressible with the mutation-kind vocabulary as it
stands on `main` in this worktree. Splitting into two sequential
all-or-nothing invocations changes the failure semantics the plan asked for:
a refusal of the detach/supersede pass after the reparent pass has already
landed would leave the two derivative children moved but the umbrella still
present with one live child (`surface_area_of_flux_surface`) — a legal,
recoverable intermediate state, but not the single-invocation, single-failure
unit the measure specifies.

## Live graph reads attempted

A read-only Cypher confirmation of the family table already recorded in the
plan (`c-20260822-umbrella-should-go`, 2026-08-22T11:15:00+00:00 — unit and
producer-count mismatch across the umbrella's three children, cited as the
physics justification for retirement) was attempted from this worktree via
`GraphClient()` and failed with `Neo.ClientError.Security.Unauthorized`
(credential/tunnel issue local to this worktree's environment, not a graph
state question). No graph read or write was performed. The plan's own
2026-08-22T11:15 comment is taken as the live-state source of record for this
finding; it is timestamped less than 5 hours before this evidence was written
and is internally consistent with the code-level exclusivity finding above.

## Recommendation

Do not attempt this as raw Cypher or as a hand-spliced authority file — either
would bypass the receipt/replay contract the whole signed-manifest program
exists to guarantee. Two sanctioned paths forward, in order of preference:

1. **Extend `signed_manifest.py`** with a fifth closed program (or relax the
   reparent-exclusivity check to permit a bounded `recompute_projection` +
   `detach` + `supersede` cohort under one `all_or_nothing: true` authority,
   with the `supersede` row's `structural-legitimacy` guard evaluated against
   the *post-mutation* graph state within the same transaction rather than at
   signing time). This is exactly the "fourth closed program" the plan
   comment already called for and is out of this node's write scope (owned
   by a concurrent node this session).
2. **Sequence two separately-authorized invocations** (reparent-only, then
   detach+supersede), accepting that the failure unit is per-invocation
   rather than per-cohort, and re-verify the umbrella's live child count is 0
   between them. This satisfies every element of the requested measure except
   "one signed invocation" and "refuses the whole cohort" as a single atomic
   unit.

No mutation was applied. No manifest was constructed, signed, or previewed,
because the authority-composition gate above makes any single-invocation
authority containing all four mutations deterministically invalid before it
would reach a preview or the live graph.
