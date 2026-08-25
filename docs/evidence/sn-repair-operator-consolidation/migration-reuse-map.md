# Migration reuse map for the seven remaining repair operators

## Result

At repository commit `86cf6bb2`, **7 of the 10** characterised repair entry
points still have bespoke implementations. The three landed migrations establish
one reusable adapter pattern: preserve the old callable and authority bytes at
the boundary, translate the legacy authority into the generic closed repair
program inside `signed_manifest.py`, and let `apply_signed_manifest` own preview,
transactional re-read, participant locking, closure re-hash, mutation, receipt,
collateral proof, and replay. The bespoke body is deleted in the same migration,
and its pre-existing disposable-graph suite is not edited.

The seven entries are covered by **6 existing disposable-Neo4j suite files**
(the source-disposition suite covers two entry points). All seven therefore have
the behavioural baseline required for class-by-class migration.

## Reusable pattern proven by the three landed migrations

The generic operator is `apply_signed_manifest` at
`imas_codex/standard_names/signed_manifest.py:4374`; canonical new authorities
can be emitted through `build_repair_authority` at
`imas_codex/standard_names/repair_authority.py:135`.

The landed adapters show three variants of the same boundary pattern:

| Landed public spelling | Boundary adapter | Fixed repair program | Evidence of the compatibility-export pattern |
|---|---|---|---|
| `retire_signed_provenance_orphans` | Read the existing `refused-target-orphan` authority unchanged and project its legacy receipt. | `supersede`; `signed-lifecycle-and-claim`, `no-live-producing-source`, `no-live-structural-child`, `out-of-allowlist-immutability`. | `imas_codex/standard_names/graph_ops.py:21460` installs a `functools.partial`; `:21470` installs the dynamically constructed legacy public name. |
| `detach_signed_stale_source_bindings` | Read the existing `stale-source-lifecycle` file and its rows-only canonical signature unchanged, then project the old receipt shape. | `detach`; `signed-lifecycle-and-claim`, `last-producing-source`, `out-of-allowlist-immutability`. | `imas_codex/standard_names/provenance_lifecycle.py:874` installs the partial and `:884` installs the legacy public name. |
| `reconcile_error_siblings` | Deterministic self-heal: pass no external artifact, derive the live row set inside the invocation, and project the generic receipt back to `{"stale_marked": n}`. | `set_properties`; `recognized-error-sibling-parent-absence`, `out-of-allowlist-immutability`; fixed reason and `apply=True`. | `imas_codex/standard_names/graph_ops.py:13173` installs the partial and `:13186` installs the legacy public name. |

Every remaining migration should use the same seven-step implementation recipe:

1. Add one closed authority adapter/selection implementation to
   `signed_manifest.py`. It may translate existing external bytes or derive a
   live preview cohort, but it must not accept caller Cypher or silently re-sign
   committed evidence.
2. Express each legacy row as typed participants, ordered closed mutations,
   named semantic guards, explicit orphan policy, receipt cardinality, and replay
   projection. Extend the closed registries when the row genuinely needs a node
   label, relationship type, mutation program, or guard not yet supported.
3. Keep external authority digests and their original canonicalization as
   provenance. The locked plan decision is read-existing, not convert-and-sign.
4. Move legacy receipt shaping and exception compatibility into a projection at
   the adapter boundary; preserve historical operation strings used to detect
   replay.
5. Replace the public function body in its owning module with a fixed
   `functools.partial(apply_signed_manifest, ...)`, using the dynamically
   constructed old spelling as the compatibility export. No second generic
   public surface is added.
6. Run the existing disposable-graph suite unchanged. Require identical admitted
   and refused identities, verbatim refusal reasons, receipt cardinality, and
   write-free replay/graph snapshots. For the two formerly receipt-less
   self-heals, preserve the public return projection while adding an internal
   generic receipt, as the error-sibling migration did.
7. Delete the bespoke body in that same migration. A migration is incomplete if
   the old implementation remains alongside the adapter.

## Seven-entry migration map

The order below is newest-first, as required by the plan. “Envelope guards” in
every row additionally means participant locks, in-transaction closure re-hash,
exact mutation/receipt cardinality, replay postcondition, and
out-of-allowlist immutability.

| Order and bespoke entry point | Authority artifact shape and selection | Mutation kind/program | Semantic guard set to preserve, plus envelope guards | Existing disposable-graph equivalence suite | Adapter pattern and required closed-registry work |
|---|---|---|---|---|---|
| 1. `retire_signed_dual_authority_targets` (`imas_codex/standard_names/graph_ops.py:21105`) | **Two external authorities plus fresh preview.** Join `imas-codex.catalog-edit-dual-binding-adjudication.v1` to `imas-codex.refused-target-orphan-adjudication.v2`; the signed source/target binding intersection and every signed retirement target form the complete cohort. Fresh preview schema: `imas-codex.signed-dual-authority-retirement-manifest`. | Ordered compound program: `delete_relationship` for each `PRODUCED_NAME` binding and backing `HAS_STANDARD_NAME` projection; `set_properties` on each source survivor; `supersede` each target; clear/recompute target `source_paths`; one receipt per retired target. | Both original digests and source row signatures; exact join coverage/uniqueness; exact source scalar, binding, backing and projection closure; source and target unclaimed; exact target live stage; target producers equal signed release set; no live structural child. | `tests/standard_names/test_dual_authority_retirement.py:192` (exact joined apply), `:234` (out-of-authority exclusion), `:280` (new-child refusal), `:323` (atomic final binding), `:371` (write-free replay). | Use the **external-authority partial** pattern. Add a named dual-authority adapter and `authority_join` guard, retain both original digest contracts, and project the singular legacy operation `retire_signed_dual_authority_target`. Extend participants to `HAS_STANDARD_NAME` and register the exact compound release/supersede program; a flat global `delete_relationship` is insufficient. |
| 2. `repair_scalar_projection_mismatches` (`imas_codex/standard_names/graph_ops.py:19830`) | **Fresh preview authority only.** Exact unique caller source ids become `imas-codex.semantic-source-mirror-repair-manifest`; rows carry source, sole live binding, typed backing, current scalar/projection, actions, already-clean rows, and refusals. | `set_properties` repairs `produced_sn_id`; a closed `add_relationship` program restores a missing backing `HAS_STANDARD_NAME` projection; one cohort receipt includes already-clean replay state. | Non-empty unique set/reason; source unclaimed and in composed/attached lifecycle; supported source type; exact typed backing; exactly one live target; at most one matching projection; scalar/binding/projection CAS. | `tests/standard_names/test_signed_source_dispositions.py:762` (exact scalar/projection classes), `:903` (non-unique target refusal), `:930` (preview drift), with replay asserted from `:874`. | Use a **preview-only compatibility partial**: the adapter takes legacy source ids, derives typed rows inside the invocation, and projects the existing receipt. Register `HAS_STANDARD_NAME` and a backing-projection add program; the current generic `add_relationship` branch is deliberately a `PRODUCED_NAME` revival program and must not be broadened ambiguously. |
| 3. `apply_adjudicated_source_dispositions` (`imas_codex/standard_names/graph_ops.py:19147`) | **Required external authority plus fresh preview**, with optional second structural authority. Read `imas-codex.catalog-edit-dual-binding-adjudication.v1` unchanged; optionally read `imas-codex.refused-target-orphan-adjudication.v2`. Select either the complete signed cohort or the admitted subset while retaining parent manifest hash, excluded ids, counts, and refusal lineage. Fresh preview schema: `imas-codex.signed-source-disposition-manifest.v2`. | Ordered compound program: `set_properties` selects the signed survivor; `delete_relationship` removes each losing `PRODUCED_NAME` and matching `HAS_STANDARD_NAME`; recompute target `source_paths`; one cohort ledger receipt linked to kept targets. | Whole-payload and per-row hashes; closed disposition; exact sorted candidate/removal sets; exact DD backing and catalog-edit participants; source unclaimed and not stale; scalar/binding/projection equality; optional exact structural-authority intersection; last live producer unless signed live-child exemption; final live authority. | `tests/standard_names/test_signed_source_dispositions.py:249` (all modes and collateral), `:366` (scalar/claim drift), `:412` (projection refusal), `:442` and `:584` (last-producer/structural authority), `:628` (global binding drift), `:669` (admitted subset replay), `:970` (tamper-before-graph). | Use the **external-authority partial** pattern with a named catalog-disposition adapter and optional joined structural-authority input. Preserve both selection modes explicitly. Extend `HAS_STANDARD_NAME`, target `source_paths` recomputation, the live-child exemption form of last-producing-source, and cohort receipt projection; do not encode those branches as artifact-supplied Cypher. |
| 4. `retire_ineligible_standard_name_sources` (`imas_codex/standard_names/graph_ops.py:20288`) | **Fresh preview authority only.** Exact unique caller DD-source ids become `imas-codex.ineligible-standard-name-source-retirement-manifest.v1`, carrying full source/binding/backing/projection closure, actions, and refusals. | `detach` all source bindings/projections; `change_source_lifecycle`/`set_properties` parks the source as `not_physical_quantity`, clears scalar and claims, and stamps skip data; recompute target `source_paths`; one cohort receipt. | Source unclaimed; exact DD backing and `node_category`; runtime category is outside `SN_SOURCE_CATEGORIES`; exact binding/projection element ids and post-detach state. **Permitted orphan hand-off is intentional:** do not add last-producing-source; return newly orphaned names to the separate workflow. | `tests/standard_names/test_dual_binding_dedup.py:387` (orphan reporting), `:444` (eligible-category refusal), `:474` (write-free replay). | Use a **preview-only compatibility partial**. Register `HAS_STANDARD_NAME`, closed source-lifecycle transition, source-path recomputation, runtime category authority, and explicit `permit_orphan_handoff`. This cannot reuse stale-source detach's guard tuple unchanged because that adapter refuses the very orphan state this operator deliberately reports. |
| 5. `reconcile_lifecycleless_standard_name_stubs` (`imas_codex/standard_names/graph_ops.py:5018`) | **Complete auto-discovered fresh preview.** Every node with null `name_stage`, `status`, and `origin` enters `imas-codex.lifecycleless-stub-manifest` and is partitioned into materialize-derived-parent, delete-dead-link-stub, rebind-source, or refused. Any refusal rejects the complete cohort. | Branching ordered programs: `materialize_derived_parent` plus canonical structural edges; `emit_retry` for DD-source reset; `set_properties` for accepted-sibling scalar repair; `delete` stub and explicitly owned derived-source/review/revision scaffolding. Receipt mechanisms remain branch-specific under one envelope operation. | Complete cohort; admissible-parent authority; complete child/operator/unit authority; exact accepted-valid sibling and producer sets; DD identity/status/scalar/bindings/unit; source claims; incident-edge closure; producer-set and materialization/deletion CAS; postflight proves no signed stub remains. | `tests/standard_names/test_reconcile_lifecycleless_stubs.py:255` (hash/apply/replay), `:385` and `:418` (parent authority/refusal), `:458` (incident-edge drift), `:498` (childful rebind), `:548` (rollback), `:609` (sibling deletion/scalar), `:794` and `:871` (late CAS races). | Use a **preview-only complete-cohort partial**, but first implement the largest genuine extension: closed branching mutation programs and branch-specific receipt projection. Register the explicitly schema-declared `materialize_derived_parent` and `emit_retry` programs plus owned-scaffolding deletion; keep arbitrary deletion or arbitrary Cypher impossible. This should migrate only after the narrower preview adapters above have proven their reusable loader/projection seam. |
| 6. `reconcile_structural_edges_for_standard_names` (`imas_codex/standard_names/graph_ops.py:3200`) | **Deterministic self-heal; no external artifact.** Exact de-duplicated caller ids, with whole-request existence preflight and no closure expansion. The live canonical structural derivation is authority. | Closed `recompute_projection` program delegates the exact requested ids to the canonical writer for `HAS_PARENT`, `HAS_ERROR`, and `HAS_LOCUS`, including canonical relationship properties and permitted structural-target materialization. | Every requested id is non-empty and exists before any write; exact request scoping; canonical grammar/derivation authority; `expand_closure=False`. | `tests/standard_names/test_structural_edge_reconcile_graph.py:105` (requested partition only), `:169` (deduplication), `:193` (empty no-op), `:202` and `:219` (verbatim input refusals), `:239` (write-free replay). | Use the **deterministic self-heal partial** proven by error siblings: empty external authority, fixed adapter, fixed mutation/guard set, apply inside the invocation, and legacy integer return projection. Register a canonical structural-reconcile program plus `HAS_ERROR`/`HAS_LOCUS` participants; do not replace the canonical writer with a second derivation. Preserve empty-request `0` and the two exact preflight errors before graph mutation. |
| 7. `repair_normalization_peel_parent_units` (`imas_codex/standard_names/graph_ops.py:4234`) | **Deterministic self-heal; no external artifact.** Auto-discover derived parents with scalar unit `1`, no normalization marker, a recorded unit-consistency finding, and every **unit-bearing** child a dimensionless normalization variant. The corrected null-unit-child interpretation is now the baseline. | Ordered `delete_relationship` for `HAS_UNIT` when present plus `set_properties(unit=null)`; legacy result projects the sorted repaired name ids. | Exact corrected normalization-peel predicate; derived origin; recorded inconsistency; all non-null-unit children normalized/normalised and unit `1`; idempotent non-selection after repair. | `tests/standard_names/test_normalization_peel_unit_repair_graph.py:160` (exact mixed cohort), `:212`, `:231`, and `:251` (non-admission), `:273` (null-unit child admitted under corrected rule), `:297` (scalar-only candidate), `:321` (write-free replay). | Use the **deterministic self-heal partial** with fixed semantic guard, automatic apply, and repaired-id list receipt projection. Register `Unit`/`HAS_UNIT` participants and the exact delete-edge-plus-null-scalar program. Do not regress to the pre-adjudication “every child” predicate; the disposable suite intentionally makes that one-row distinction visible. |

## Recommended migration waves

The newest-first order above is the semantic ordering. For implementation risk,
land each row independently and stop after each unchanged-suite equivalence gate:

1. Dual-authority retirement, semantic mirror repair, source dispositions, and
   ineligible-source retirement exercise the already-proven external/preview
   adapter seam while adding narrowly named relationship and projection programs.
2. Lifecycleless reconciliation follows after those narrower preview adapters,
   because it is the only remaining branching, multi-receipt cohort.
3. Structural-edge reconciliation and normalization-unit repair use the
   deterministic self-heal pattern and preserve their public return projections.
   Their suites were deliberately written against the bespoke live code before
   migration and are now valid equivalence baselines.

No row should share a deletion switch with another. Each lands only when its
bespoke `def` is absent, its legacy callable is a partial targeting
`apply_signed_manifest`, its old suite is byte-unchanged, and the suite proves
the same admitted/refused identities, reasons, receipt count, and replay state.

## Quantitative completeness

- Remaining bespoke entry points named: **7/7**.
- Each row includes an authority shape: **7/7**.
- Each row includes a mutation kind/program: **7/7**.
- Each row includes a semantic guard set and the shared envelope guards: **7/7**.
- Each row identifies an existing disposable-graph suite: **7/7**, across **6** files.
- Each row maps to one of the three landed adapter variants and names required
  closed-registry work: **7/7**.
- Repository source was inspected read-only at `86cf6bb2`; no graph endpoint and
  no test suite were run for this investigation.
