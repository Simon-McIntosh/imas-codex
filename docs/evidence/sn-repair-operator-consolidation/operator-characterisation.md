# Repair operator characterization

This is the behavioral specification for a generic repair operator. It covers
all ten operators named by the plan. “Authority artifact” distinguishes three
contracts that the implementation currently conflates:

- **external authority** — reviewed evidence exists before preview and its
  original digest is provenance;
- **preview authority** — the operator derives a graph closure and the caller
  authorizes exactly its manifest SHA-256 for apply;
- **deterministic self-heal** — no signed artifact exists; the current graph and
  code predicate are the only authority.

The generic core therefore needs, at minimum, a stable operation identity,
authority adapter and original-digest provenance, row key and selection mode,
canonical participant closure, mutation program, guard program, receipt policy,
and replay postcondition. The table below specifies those fields from current
behavior; it does not imply that every irregularity should become schema.

| Operator | Authority artifact shape | Selection predicate | Mutation set | Guard set | Receipt operation name | Test location |
|---|---|---|---|---|---|---|
| `reconcile_structural_edges_for_standard_names` | **Deterministic self-heal; no artifact, manifest, digest, or receipt.** Caller supplies a de-duplicated ordered list of Standard Name ids. | Exact caller ids; reject empty ids; require every requested `StandardName` to exist before any structural write. | Delegate the exact ids to `_write_standard_name_edges(..., expand_closure=False)`: reconcile derived `HAS_PARENT`, `HAS_ERROR`, and `HAS_LOCUS` structure and canonical relationship properties; the canonical writer may materialize referenced structural targets. | Whole-request existence preflight; canonical grammar/derivation logic; no closure expansion beyond the named children. No preview/apply CAS, participant locks, replay proof, or collateral digest. | **None.** Returns only the count of requested names. | `tests/standard_names/test_rename_cascade.py::TestExactStructuralReconciliation` (mocked delegation and missing-id refusal). |
| `repair_normalization_peel_parent_units` | **Deterministic self-heal; no artifact, manifest, digest, or receipt.** Authority is a single live Cypher predicate. | Derived parent with scalar unit `1`; id has no `normalized`/`normalised` token; `validation_issues` contains `name_unit_consistency_check`; every child has unit `1` and is a normalization variant. | Delete the parent's `HAS_UNIT` edge to `Unit {id: '1'}` and set `StandardName.unit = null`; caller must route returned ids through validation re-stamping. | All semantic safeguards are embedded in the selection predicate. Idempotency follows because repaired rows cease to match. No transaction-wide closure, locks, hash, replay receipt, or collateral proof. | **None.** Returns repaired ids and logs a count. | `tests/standard_names/test_component_system.py::test_repair_normalization_peel_parent_units_scoping` (source inspection plus mocked result only). |
| `reconcile_lifecycleless_standard_name_stubs` | **Preview authority.** Fresh complete graph cohort becomes `imas-codex.lifecycleless-stub-manifest`, with `rows` partitioned into `materialize-as-derived-parent`, `delete-as-dead-link-stub`, `rebind-source`, and `refused`. Each row carries full stub properties, child/operator data, DD-source and producer closure, every incident relationship, reviews, and docs revisions. Apply presents the manifest SHA-256. | All `StandardName` nodes with `name_stage`, `status`, and `origin` all null; no caller subset. Partition by presence and completeness of children and DD producers, accepted-valid sibling authority, unique parent-owned unit, and `is_admissible_parent_name`. Any refused row refuses the complete cohort. | Branching transaction: materialize a derived parent and bootstrap structural edges; reset DD sources through `StandardNameSourceRetry`; repair a null scalar to an accepted valid sibling; delete dead/spurious stubs plus derived-source, review, and revision scaffolding. | Canonical order-insensitive full closure; exact accepted-sibling producer set; DD identity/status/scalar/bindings/unit; child lifecycle/unit/structural admission; source claims; producer-set CAS; materialization/deletion cardinality; complete-cohort manifest hash; postflight proves no signed stub remains; transaction rollback on any late failure. | `materialize_derived_parent` for materialization; `remove_derived_parent` for deletion; source resets write `StandardNameSourceRetry` rather than a `StandardNameChange.operation`. Receipt envelope operation is `reconcile_lifecycleless_standard_name_stubs`. | `tests/standard_names/test_reconcile_lifecycleless_stubs.py` (partition, hash/apply/replay, authority refusals, incident-edge drift, CAS races, and rollback). |
| `reconcile_error_siblings` | **Deterministic self-heal; no artifact, manifest, digest, or receipt.** Operator prefixes come from `ERROR_SUFFIX_TO_OPERATOR`. | Non-quarantined `StandardName` with `model='deterministic:dd_error_modifier'`; strip a recognized uncertainty prefix; select when no parent `StandardName` exists. Unrecognized forms are skipped. | Set `validation_status='quarantined'` and fixed `quarantine_reason='orphaned error sibling (parent name deleted)'`. | Dynamic closed prefix vocabulary and parent-existence check only. Discovery and mutation are separate queries; no claim guard, lifecycle guard, transaction, CAS, locks, replay proof, or collateral digest. | **None.** Returns `{"stale_marked": n}`. | `tests/standard_names/test_error_siblings.py::TestReconcileErrorSiblings` (mocked orphan, live-parent, and empty-graph cases). |
| `apply_adjudicated_source_dispositions` | **External authority plus preview authority.** Required `imas-codex.catalog-edit-dual-binding-adjudication.v1` signs the complete payload and each row; rows name exact DD source, candidates, prior scalar, survivor, removed targets, and one of three dispositions. Optional `imas-codex.refused-target-orphan-adjudication.v2` signs structural-legitimacy evidence. Fresh `imas-codex.signed-source-disposition-manifest.v2` captures participants, backing projections, removed-target producer/child closure, structural exemptions, actions, and refusals. | All signed adjudication rows, or `admitted_subset=True`: recompute the complete parent manifest, select only its admitted source ids, and retain the parent manifest hash, counts, exclusions, and refusal evidence. | Set each source scalar to its signed survivor; delete every signed non-survivor `PRODUCED_NAME` binding and matching backing `HAS_STANDARD_NAME` projection; recompute `source_paths` on all affected targets; write one cohort ledger event linked to kept targets. | Artifact payload and row hashes; closed disposition logic and exact sorted candidate/removal sets; exact DD source/backing; no active claim or stale source; scalar, live-binding, catalog-edit participant, and projection equality; optional signed structural target intersection; last-live-producer guard with a signed live-child exemption; participant node/relationship locks; manifest re-hash; deletion cardinality; final live authority. | `apply_adjudicated_source_dispositions` (one `StandardNameChange` for the cohort). | `tests/standard_names/test_signed_source_dispositions.py` (all dispositions, signature pin/tamper, scalar/claim/projection/global-binding drift, last-producer and structural-exemption refusals, admitted-subset replay). |
| `repair_scalar_projection_mismatches` | **Preview authority.** `imas-codex.semantic-source-mirror-repair-manifest` contains exact caller source ids, complete source/binding/backing/projection participants, actions, already-clean rows, and refusals. Apply presents the manifest SHA-256; there is no earlier adjudication artifact. | Unique exact caller source ids; source is `composed` or `attached`; exactly one live `PRODUCED_NAME` target; DD/signal source has exactly one matching typed upstream backing, while derived source has no projected backing policy; select when scalar differs or the sole target projection is missing. | CAS `produced_sn_id` to the sole live binding; create a missing backing `HAS_STANDARD_NAME` projection; include already-clean requested rows in the replay postcondition; write one cohort event linked to all targets. | Non-empty set/reason; active-claim and lifecycle checks; supported source type; exact source/backing identity; exactly one live target; at most one matching projection; participant node/relationship locks; closure re-hash; per-class mutation cardinality; replay verifies binding, scalar, and projection. | `repair_scalar_projection_mismatches` (one `StandardNameChange` for the cohort). | `tests/standard_names/test_signed_source_dispositions.py` mirror-repair cases (scalar and projection mutations, non-unique target refusal, preview drift, write-free replay). |
| `retire_ineligible_standard_name_sources` | **Preview authority.** `imas-codex.ineligible-standard-name-source-retirement-manifest.v1` contains exact source ids, full source/binding/backing/projection participants, actions, and refusals. Apply presents the manifest SHA-256; there is no earlier adjudication artifact. | Unique exact caller source ids; source is DD-backed; exactly one `IMASNode` backing with `node_category`; category is not in runtime `SN_SOURCE_CATEGORIES`; at least one `PRODUCED_NAME` binding. | Delete all source `PRODUCED_NAME` bindings and matching backing projections; set source status to `not_physical_quantity`, clear scalar and claims, and stamp skip reason/detail; recompute target `source_paths`; report newly orphaned names without retiring them; write one cohort event. | Exact source/backing/relationship closure; active-claim guard; runtime category authority; full binding and projection element-id cardinality; source status/scalar/backing-category CAS after detach; participant locks and manifest re-hash; replay verifies retired source state and zero bindings. It intentionally does **not** enforce the last-producing-source guard because orphan retirement is a separate workflow. | `retire_ineligible_standard_name_sources` (one `StandardNameChange` for the cohort). | `tests/standard_names/test_dual_binding_dedup.py` ineligible-retirement cases (orphan reporting, eligible-category refusal, write-free replay). |
| `retire_signed_dual_authority_targets` | **Two external authorities plus preview authority.** Join the signed source-disposition adjudication and `imas-codex.refused-target-orphan-adjudication.v2`; validate that retirement `current_removed_bindings` are the exact source-row/target intersection. Fresh `imas-codex.signed-dual-authority-retirement-manifest` records both original hashes, source participants, target producer/child closures, actions, and refusals. | Only source rows participating in the exact signed binding intersection and every signed `retire_under_orphan_policy` target; no caller subset. | Set each source scalar to its survivor; delete each jointly signed binding and backing projection; atomically set every target's `name_stage` and `status` to `superseded`, preserve `superseded_from_stage`, clear `source_paths` and claims; create one ledger row per retired identity. | Both artifact digests and all source row signatures; exact intersection/coverage/uniqueness; source scalar/live-binding/projection/backing closure; source and target claims; exact target stage; target producers equal the signed release set; no live structural child; participant locks and re-hash; release and lifecycle CAS; replay proves all bindings absent and all targets remain superseded, source-less, and childless. | `retire_signed_dual_authority_target` (singular; one `StandardNameChange` per target). | `tests/standard_names/test_dual_authority_retirement.py` (exact joined apply, out-of-authority exclusion, acquired-child refusal, final-binding atomicity, write-free replay). |
| `retire_signed_provenance_orphans` | **External authority plus preview authority.** `imas-codex.refused-target-orphan-adjudication.v2` is whole-file hashed; `retire_under_orphan_policy` rows must carry `classification_only`, a live signed stage, empty live structural closure, name-specific removed-binding evidence, and derived row hash. Fresh `imas-codex.signed-provenance-orphan-retirement-manifest.v1` records original authority hash, participants/producers/children, actions, and refusals. | The complete signed retirement cohort. Optional `name_ids` must equal the signed set exactly: outside ids and omitted signed ids are both refusals. | Set each name's lifecycle/status to `superseded`, preserve prior stage, clear claims, and create one linked ledger row per identity. Stale producer bindings are not detached by this operator. | Artifact hash/schema/read-only flag/summary; allowed disposition and live stage; signed empty structural closure and binding evidence; exact requested set; current stage and claim CAS; no non-stale producer; no live `HAS_PARENT` child; target locks and re-hash; mutation cardinality; replay verifies lifecycle, zero live producers, zero live children, and every receipt. | `retire_signed_provenance_orphan` (one `StandardNameChange` per target). | `tests/standard_names/test_structural_orphan_retirement.py` (16-row apply/replay, outside-target refusal, acquired live source/child refusal). |
| `detach_signed_stale_source_bindings` | **External file authority plus preview authority.** `imas-codex.stale-source-lifecycle-disposition.v1` signs canonical `rows` only (`jq -cS '.rows'`) and the operator also records the raw file hash. Selected rows identify DD or derived stale source, source/DD lifecycle shape, scalar and live targets, `detach`, absent configured path, and reason. Fresh manifest embeds signed rows, full source/backing/target/current-DD closure, actions, and an out-of-allowlist source count/hash. | Unique exact caller subset of signed rows; every selected row must have `disposition='detach'` and complete source-type-specific authority. Unlike provenance-orphan retirement, omission of other signed rows is allowed. | Delete every selected source binding and DD backing projection; set source scalar null; remove that source from target `source_paths`; create one linked ledger event per source. Source lifecycle remains `stale`. | Raw-file and canonical-row hashes; exact DD/derived signed shape; source remains stale, unclaimed, same type/DD version/scalar; signed targets cover every live binding; DD source has one removed backing and equal projection set, derived source has none; exactly one current `DDVersion` hard-coded to `4.1.1`; last-live-producer guard with live-child alternative; participant locks and closure re-hash; postcondition; out-of-allowlist immutability; exact `StandardNameChange` delta and zero `LLMCost` delta. | `detach_stale_source_binding` (one `StandardNameChange` per source; deliberately differs from function name). | `tests/standard_names/test_stale_source_detach.py` (58-row authority/signature, exact subset, DD/derived/multi-binding shapes, last-producer and unsigned-drift refusals, apply/replay). |

## Expressivity gaps and irregularities

The following list is exhaustive for differences exposed by these ten
operators. **Genuine extension point** means the generic schema or its runtime
registry must model the distinction. **Accident** means migration should
normalize it rather than enshrine it as another per-class feature.

1. **Genuine extension point — authority provenance adapters.** Existing
   external artifacts use whole-payload signatures, per-row signatures,
   rows-only signatures, raw-file hashes, or combinations of them. The generic
   operator must retain each original digest and canonicalization contract and
   adapt it losslessly; it must not re-sign old evidence. New artifacts can use
   one canonical signature profile.
2. **Genuine extension point — authority source mode.** The schema must
   distinguish external reviewed authority, graph-derived preview authority,
   and deterministic self-heal. The latter shares mutations and guards but not
   the signed-artifact envelope and is explicitly a later migration surface.
3. **Genuine extension point — row identity and participant domains.** Rows are
   keyed by source id, Standard Name id, or a source/target binding pair, and
   closures include nodes and relationships of different labels. A single
   string `row_id` without typed participants cannot express exact CAS.
4. **Genuine extension point — selection semantics.** Current modes are exact
   caller set, complete auto-discovered cohort, exact complete signed cohort,
   signed subset, and “admitted subset of a complete signed parent” with parent
   hash and excluded-refusal lineage. Selection mode and completeness policy
   must be explicit data.
5. **Genuine extension point — joined authority.** Dual-authority retirement
   requires an exact relational intersection between two independently signed
   artifacts, not merely two hashes attached to one row. The schema needs a
   join/coverage constraint or a named authority adapter.
6. **Genuine extension point — compound and branching mutations.** A row may
   set properties, add or delete relationships, recompute projections, clear
   source paths, change source lifecycle, change name lifecycle, materialize a
   derived parent, delete a stub and its owned scaffolding, or emit a retry
   event. Lifecycleless reconciliation selects among multiple programs in one
   cohort. A single `mutation_kind` enum with one flat payload is insufficient
   unless kinds can expand to ordered atomic steps.
7. **Genuine extension point — semantic guard plug-ins.** Some guards are
   declarative equality/cardinality checks; others invoke runtime authorities:
   `SN_SOURCE_CATEGORIES`, `is_admissible_parent_name`, parent-unit derivation,
   canonical structural-edge writing, live-child semantics, and exact
   source-type backing policy. These require named, closed, testable guard
   implementations rather than embedded arbitrary Cypher.
8. **Genuine extension point — receipt cardinality and replay projection.**
   Current logical receipts are cohort-wide, per source, or per target, and
   replay checks different postcondition projections. The schema must declare
   receipt keying, links, expected count, and the postcondition fields that make
   `already_applied` trustworthy.
9. **Genuine extension point — permitted orphan hand-off.** Ineligible-source
   retirement intentionally permits a name to become producer-less and returns
   it to a separate orphan workflow, whereas detach and disposition operations
   refuse that state unless structural authority remains. This is semantic
   policy, not a guard that can be globally enabled.
10. **Accident — “signed operator” is not one current contract.** Three named
    operators have no manifest at all, and three others sign only their fresh
    preview rather than consume adjudicated evidence. The generic API and docs
    should use the three authority modes above instead of treating all ten as
    equivalent signed artifacts.
11. **Accident — incomplete safety-envelope coverage.** Only stale-source
    detach proves out-of-allowlist source immutability and global ledger/provider
    counter deltas. Several operators lock and re-hash participants; three
    deterministic self-heals do neither. The generic core should apply one
    collateral-proof and locking policy wherever a signed apply mutates graph
    state, with explicit opt-outs justified by mutation scope.
12. **Accident — non-atomic self-heals and missing claim guards.** Error-sibling
    reconciliation discovers and mutates in separate queries; normalization
    repair and structural reconciliation have no preview/apply CAS or durable
    receipt. Their initial exclusion from signed migration is genuine, but the
    absence of auditability and race protection is not behavior to preserve.
13. **Accident — hard-coded current DD identity.** Stale-source detach requires
    exactly current DD `4.1.1`. “Current configured DD authority matches the
    signed row/manifest” is a real guard; embedding one historical DD version in
    the generic schema is not.
14. **Accident — operation identity drift.** Function, manifest, and receipt
    identities vary in number or wording (`retire_signed_dual_authority_targets`
    versus `retire_signed_dual_authority_target`, and
    `detach_signed_stale_source_bindings` versus
    `detach_stale_source_binding`); lifecycleless reconciliation emits three
    receipt mechanisms. Preserve old operation strings for replay compatibility,
    but map them to one stable generic operation id plus explicit receipt kind.
15. **Accident — unversioned schema identifiers.** Several manifest/receipt
    identifiers omit an explicit version while others end in `.v1` or `.v2`.
    Existing ids need adapters; every new generic authority, manifest, and
    receipt schema should carry an explicit version.
16. **Accident — incomparable `changed` semantics.** Cohort operators return
    `changed=1` for one ledger event while per-row operators return the number of
    identities/sources changed, and lifecycleless returns materialized plus
    deleted nodes while reporting source effects separately. A generic receipt
    should use named counts (`rows_changed`, `relationships_added`,
    `relationships_removed`, `receipt_rows`) and define `changed` once or omit
    it.
17. **Accident — uneven behavioral evidence.** Seven operators have
    disposable-graph hash/apply/refusal/replay coverage. Structural-edge,
    normalization-unit, and error-sibling self-heals have only mocked or
    source-inspection tests. They must gain disposable-graph equivalence tests
    before later migration; their current tests cannot establish unchanged
    refusal or collateral behavior.

Coverage: **10/10 named operators**, with **4 external-authority operators**,
**3 preview-authority-only operators**, and **3 deterministic self-heals**.
The expressivity audit records **9 genuine extension points** and **8 accidents**.
