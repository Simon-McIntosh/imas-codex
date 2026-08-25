# Regression repair reuse map

Date: 2026-08-25  
Repository revision inspected: `86cf6bb2ebbdf3ed8caca8f9ed752f2454f1f7a9`

## Outcome

The two cohorts have different repair shapes and should not share a mutation
program merely because the same census reports them.

- The **3 scalar-mirror rows** are an exact reuse case for
  `repair_scalar_projection_mismatches`: it derives the sole live edge as
  authority, signs the complete source/backing/projection closure, locks and
  re-hashes it, compare-and-sets the scalar, emits one cohort receipt, and
  proves write-free replay. No new mutation program is needed.
- The **39 no-live-target rows** split into **36 derived + 3 DD**. Existing code
  already contains the exact lifecycle-release mechanics in
  `reconcile_source_status_liveness`, including terminal-edge, DD projection,
  and `source_paths` cleanup, but that function is an unsigned bespoke write
  and only returns the source to `extracted`; it does not establish a new live
  target. The generic signed envelope and schema-valid authority builder are
  reusable as-is, but a **new closed signed no-live-target release program** is
  required before production use. The derived and DD rows must remain separate
  authority partitions; any later reattachment or structural revival requires
  independent current target authority.

The measured cohort authority is the read-only verification at
`docs/evidence/archive/integrity-and-operator-closure-verification.md:42-56`:
39 no-live-target rows (36 derived, 3 DD) and 3 scalar mismatches (2 DD, 1
derived), stable over two reads. The three DD no-target identities are
`dd:ntms/time_slice/mode`, `dd:summary/pedestal_fits`, and
`dd:waves/coherent_wave`. No live mutation was performed by this investigation.

## Candidate modules and invocable symbols

Every verdict below is against both measured cohorts. “Reuse as-is” means the
candidate can be invoked without broadening its mutation contract; “extend”
means its mechanics are reusable but the current callable is not production
authority for that cohort; “unfit because” is a deliberate fail-closed result.

| Candidate module and anchor | Invocable symbol | Fitness against 39 no-live-target rows | Fitness against 3 scalar-mirror rows |
|---|---|---|---|
| `imas_codex/standard_names/provenance_lifecycle.py:1002` | `find_semantic_source_invariant_violations(gc)` | **Reuse as-is (census):** selects composed/attached rows with zero live targets and reports source type, all bindings, scalar, and projection state; use it for the exact pre/post cohort, not as a mutator. | **Reuse as-is (census):** selects sole-live rows whose scalar disagrees and provides the exact 3-source allowlist for preview and postflight. |
| `imas_codex/standard_names/graph_ops.py:11847` | `reconcile_source_status_liveness(gc=..., source_ids=[...])` | **Extend into a closed signed program:** its exact-scoped branch already deletes terminal `PRODUCED_NAME` edges, unshared DD projections and cache URIs, clears scalar/claim/composition state, and returns composed/attached no-live rows to `extracted`; it is unsigned, writes directly, and does not restore a target, so it is not sufficient production authority as-is. | **Unfit because:** a row with one live edge never enters the orphan branch and the live realignment branch excludes already composed/attached sources, so the stale scalar remains unchanged. |
| `imas_codex/standard_names/graph_ops.py:19830` | `repair_scalar_projection_mismatches(source_ids, reason=..., apply=..., manifest_sha256=..., run_id=...)` | **Unfit because:** preview refuses every source that does not have exactly one live target; it must not infer a target from a stale scalar. | **Reuse as-is:** exact operator for all 3 rows, provided the fresh preview confirms one live target per source and no refusal; it supports signed digest apply, participant locks, re-hash, CAS, receipt, and write-free replay. |
| `imas_codex/standard_names/provenance_lifecycle.py:1379` | `repair_semantic_source_invariants(gc, source_ids, authority_overrides=...)` | **Unfit because:** even an explicit override must already be exactly one current live target (`:1306-1310`), so zero-live rows remain ambiguous and untouched. | **Extend only as diagnostic comparison:** it can repair a scalar from a sole live edge, but lacks the signed-preview/digest/replay contract already supplied by `repair_scalar_projection_mismatches`; do not choose the weaker duplicate. |
| `imas_codex/standard_names/signed_manifest.py:4374` | `apply_signed_manifest(authority_path, ..., apply=..., authorized_manifest_sha256=..., run_id=...)` | **Reuse envelope as-is; add one closed program:** canonical file/payload digests, exact selection, collateral proof, lock/re-hash, ordered mutation, receipt, postcondition, and replay are already generic; only the no-live lifecycle-release mutation shape is missing. | **Unfit as the immediate operator:** the scalar repair is still a bespoke signed operator and has not migrated into this registry; forcing the 3 rows through a generic `set_properties` row would discard its sole-live/backing closure. |
| `imas_codex/standard_names/repair_authority.py:135` | `build_repair_authority(specification)` | **Reuse as-is:** emit schema-valid per-row authority for a new closed release program, with the 36 derived and 3 DD rows in separately adjudicated partitions and complete terminal/backing/cache participants. | **Unfit as current scalar authority:** this builder emits generic-envelope artifacts, while the existing scalar operator deliberately derives its own canonical preview manifest inside the applying invocation. |
| `imas_codex/standard_names/signed_manifest.py:340` | `_validate_unbound_source_attachment_program(...)` through `apply_signed_manifest` | **Unfit because:** it admits only unbound `dd:` sources in `status='extracted'`, with `produced_sn_id=null`, no binding, one exact DD backing, and no projection (`:2331-2444`); the 39 measured rows are composed/attached, and 36 are derived. It is a possible later DD reattachment step only after separately governed release and target adjudication. | **Unfit because:** it creates an edge and advances lifecycle rather than correcting the scalar beside an existing sole live edge. |
| `imas_codex/standard_names/signed_manifest.py:254` | `_validate_structural_source_revival_program(...)` through `apply_signed_manifest` | **Unfit as-is:** it requires a `stale` derived source with no binding, an accepted target equal to the source identity, and at least one live structural child; the 36 measured derived rows are composed/attached and may retain terminal bindings. It may be reused later only for rows independently proven to meet that different state. | **Unfit because:** it adds a derived-source binding and rewrites lifecycle rather than reconciling a scalar on a sole live binding. |

## Candidate authority artifacts

There is deliberately **no current signed authority artifact** for either live
regression cohort. The applying node must derive fresh participants from the
live graph; old bytes are evidence and templates, not reusable mutation
authority.

| Candidate artifact and anchor | Fitness against 39 no-live-target rows | Fitness against 3 scalar-mirror rows |
|---|---|---|
| `docs/evidence/archive/integrity-and-operator-closure-verification.md:42-56` | **Reuse as read-only selection evidence:** it fixes the quantitative partition (36 derived + 3 DD) and names the DD rows, but contains neither per-row participants nor a signature, so it cannot authorize writes. | **Reuse as read-only selection evidence:** it establishes 3 rows (2 DD + 1 derived), but the source IDs and live closure must be recaptured in the scalar preview. |
| `docs/evidence/sn-graph-wide-integrity/stale-source-lifecycle.json:1` | **Unfit because:** this is a 2026-08-20 authority over 58 `status='stale'` sources and explicitly authorizes detach only; none of its semantics authorizes the current composed/attached 39-row release or a replacement target. | **Unfit because:** it authorizes stale detach, not scalar equality to a sole live edge. |
| `docs/evidence/sn-graph-wide-integrity/scalar-mirror-reconcile.md:36-58` | **Unfit because:** the retained example has one accepted live edge plus one exhausted historical edge, which is the opposite of zero live targets. | **Reuse as an execution template, never as authority:** it records the proven source `dd:plasma_sources/source/ggd/neutral/state/momentum/phi`, accepted target `toroidal_neutral_internal_state_torque_density`, stale scalar `neutral_internal_state_torque_density`, exact run/digest receipt recovery, and zero-write replay; its old digest must not be reused for the new 3 rows. |
| Fresh preview manifest from `repair_scalar_projection_mismatches` (`imas-codex.semantic-source-mirror-repair-manifest`) | **Unfit because:** the preview returns verbatim refusal `source does not have exactly one live target` for the 39 rows. | **Reuse as-is and treat as the authority artifact:** preview the exact current 3-source allowlist immediately before serialized apply, retain its canonical `manifest_sha256`, and recover receipts only by exact `run_id` plus digest. |
| Fresh `imas-codex.repair-authority.v1` file from `build_repair_authority` | **Extend with the new closed release specification:** this is the correct durable artifact form, but it must sign current source nodes, every terminal binding/target, exact DD backing/projection/cache closure where applicable, row-specific cleanup mutations, and explicit receipt cardinality. | **Unfit because:** the deployed scalar operator has its own narrower authority schema; replacing that schema is consolidation work, not regression repair. |

## Disposable-graph and adjacent suites

| Candidate suite and anchor | Fitness against 39 no-live-target rows | Fitness against 3 scalar-mirror rows |
|---|---|---|
| `tests/standard_names/test_signed_source_dispositions.py:762` — `test_signed_mirror_repair_applies_scalar_and_projection_classes_exactly` plus refusal/drift cases at `:903` and `:930` | **Unfit because:** the refusal case proves zero-live/non-unique authority cannot pass this operator; retain it as a negative guard. | **Reuse as-is:** disposable Neo4j covers scalar-only, projection-only, combined repair, outside-allowlist immutability, exact receipt, ambiguity refusal, drift conflict, and write-free replay. |
| `tests/standard_names/test_signed_manifest_operator.py:276-639` | **Reuse as envelope baseline, extend with a no-live program fixture:** disposable Neo4j already proves canonical manifest admission, all generic mutations, collateral refusal, participant drift rollback, receipt counters and replay, but has no composed/attached zero-live lifecycle-release row. | **Unfit as semantic proof:** it proves the envelope mechanics, not sole-live scalar authority or source/backing projection closure. |
| `tests/standard_names/test_unbound_source_attachment.py:236-405` | **Unfit as direct regression coverage:** disposable Neo4j proves extracted DD attachment and its refusals, not composed/attached release and not derived rows; reuse only for a separately authorized later reattachment of eligible members among the 3 DD rows. | **Unfit because:** it adds a missing edge instead of repairing the mirror of an existing edge. |
| `tests/standard_names/test_structural_source_revival.py:286-404` | **Unfit as direct regression coverage:** disposable Neo4j proves stale, childful, unbound derived revival; extend it or add a sibling suite for the distinct composed/attached terminal-binding release program before applying any of the 36 derived rows. | **Unfit because:** it proves derived binding creation, not scalar reconciliation. |
| `tests/standard_names/test_source_revival.py:367-475` | **Reuse as behavioral seed, not as the required disposable suite:** the graph-marked tests pin live realignment, exact source scoping, terminal projection/cache cleanup, shared-backing preservation, and idempotency; port the relevant cases to disposable Neo4j under the signed closed program. | **Unfit because:** it does not exercise scalar mismatch repair. |
| `tests/standard_names/test_semantic_source_repair.py:473` | **Unfit because:** its fail-closed live test preserves ambiguity when no live target selects authority. | **Adjacent comparison only:** it covers atomic repair and cache rebuild for the legacy operator, but the disposable signed-source-dispositions suite is the authoritative regression gate for the 3 rows. |

## Recommended execution boundary

1. Re-run `find_semantic_source_invariant_violations` and freeze two exact
   allowlists: 39 no-live rows (partitioned 36 derived / 3 DD) and 3 scalar rows.
2. Apply the scalar cohort first and alone with
   `repair_scalar_projection_mismatches`: fresh preview, require
   `requested=3`, `admitted=3`, `refused=0`, apply the same digest, recover one
   cohort receipt by exact `run_id` + digest, replay, and require the scalar
   census to be 0 without edge changes.
3. Do not invoke `reconcile_source_status_liveness` directly on production.
   Lift its exact-scoped orphan-release mechanics into one closed signed
   manifest program and add disposable-graph cases for derived and DD rows,
   shared DD projection ownership, terminal-edge cleanup, claim refusal,
   collateral immutability, preview drift, receipt cardinality, and replay.
4. Preview the 39-row release authority derived from current state. Require
   `admitted + refused = 39` and retain every refusal verbatim. Apply only the
   admitted maximal-safe subset in one serialized invocation; prove mutation
   from exact receipts, then require no composed/attached zero-live rows in the
   postflight census. Returning a row to `extracted` closes the broken terminal
   lifecycle claim; it does **not** by itself authorize a new Standard Name.
5. Any later DD attachment or derived structural revival is a separate action
   with independently reviewed target authority. Never infer that target from
   a terminal edge or stale `produced_sn_id`.

## Quantitative coverage

- Live regression classes covered: **2 of 2**.
- Live rows covered by the map: **42 of 42** = 39 no-live-target + 3
  scalar-mirror.
- Candidate module/symbol rows: **8**.
- Candidate authority-artifact rows: **5**.
- Candidate suite rows: **6**, including **4 disposable-Neo4j suites** and 2
  adjacent graph suites.
- Unresolved repository anchors after validation: **0**.

