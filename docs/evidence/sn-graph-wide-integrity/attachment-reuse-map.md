# Reuse map for the exact 25-row source-attachment workflow

## Outcome

The workflow should extend the typed signed-manifest transaction envelope and
the canonical source-authority closure; it should not introduce another
stand-alone manifest format or call a permissive binding helper as its public
write surface. The existing repository already supplies the difficult pieces:
independently pinned authority bytes, typed/canonical hashes, full participant
locks, lock-then-rehash, the shared source-to-name compatibility guard,
compare-and-set retargeting of every provenance mirror, semantic-source
postflight checks, immutable receipts, rollback, and replay detection.

One material gap remains: the generic signed-manifest executor currently
interprets only `delete`, `supersede`, and `detach`. It does not yet execute a
closed attachment program spanning `StandardNameSource`, `PRODUCED_NAME`, the
scalar `produced_sn_id`, the DD-side `HAS_STANDARD_NAME` projection, and both
affected names' `source_paths`. The 25-row workflow should add that program to
the existing closed registry (or a narrowly typed adapter using the same
envelope), with no authority-supplied Cypher.

This inspection used the read-only worktree at commit
`d00744813a781e7ddf1ee0d93359d26eb8a00b9d` and the live plan at version 219.
The plan's current semantic boundary is decisive: reverse semantic-search
candidates are evidence, never attachment authority; all 25 rows still owe
independent adjudication, exact DD/version identity, mandatory unit evidence,
the shared semantic guard, a dry-run hash, compare-and-set apply, and ordinary
review for any name-level correction.

## Reuse candidates

| Existing machinery | Repository anchor | What it already proves | Fitness verdict for the 25-row workflow |
|---|---|---|---|
| `apply_signed_manifest` generic transaction envelope | `imas_codex/standard_names/signed_manifest.py:1315` | Requires independent file and canonical-payload digests; regenerates the graph closure in the applying transaction; locks, re-hashes, mutates, checks receipts/counters/collateral, commits atomically, and recognizes replay. | **Extend; highest-value spine.** Keep this preview/hash/apply/replay envelope. Add a closed attachment mutation/guard implementation. Do not fork a second transaction framework. The present executor's mutation registry is deliberately narrower than the schema and cannot attach yet. |
| Typed repair authority schema | `imas_codex/schemas/standard_name.yaml:787` | Already models ordered typed mutations, typed guards, exact selection policies, repair rows, participants, digests, and receipt policy; it includes `set_properties`, `add_relationship`, and `delete_relationship` vocabulary in addition to the currently interpreted lifecycle operations. | **Extend, do not invent a new JSON shape.** Represent every source/name/DD/projection participant and every synchronized mirror mutation as typed rows. Add only the closed participant/guard semantics the attachment program genuinely needs. |
| Canonical DD source-authority closure and participant locks | `imas_codex/standard_names/source_authority.py:488` | Canonicalizes an exact DD source, current DD node/version, immutable snapshot, target-protection closure, participant IDs, precondition hash, and preserved-state hash; the adjacent helpers require complete ordered paths and prove lock cardinality. | **Reuse directly for existing sources; factor for absent sources.** This is the correct source/DD/version authority substrate. Extend its closure to include the prospective target and both old/new target producer closures. A missing `StandardNameSource` needs a typed absent-state row rather than indexing `row["sources"][0]`. |
| Bounded-manifest exact source creation | `imas_codex/standard_names/graph_ops.py:1203` | Validates exact DD paths, reads and locks DD/source/name authority, creates an absent source only while the requested DD version is uniquely current, pins the immutable DD snapshot, and compare-and-set stamps scope. | **Extract the source-materialization seam; do not invoke the drain workflow wholesale.** It proves how an absent `dd:<path>` is safely born. Drain-scope claims, five-way generation disposition, and pool routing are unrelated to a direct reviewed attachment and must not leak into this operator. |
| Write-boundary batch guard `guard_source_pairings` | `imas_codex/standard_names/attachment_audit.py:595` | Loads source path, existing target siblings, DD/SN units, and target lifecycle on the caller's query handle; evaluates deterministic compose-order semantics and returns explicit accepted/rejected source IDs. | **Reuse inside the locked applying transaction, requiring exact-set admission.** Refuse the whole 25-row apply if any requested source is absent from the accepted set. Tighten the adapter contract: `already_bound` is replay only when the matching immutable receipt exists, and a missing DD path/unit must not be treated as attachment approval. |
| Shared semantic/unit predicate `_is_attachment_consistent` | `imas_codex/standard_names/workers.py:2701` | Central compatibility logic for rate/tense, state resolution, surface kind, distinct vectors, locus/device and geometry representation, plus DD-resolution-aware unit dimensionality. Compose, audit, retarget, and edit paths converge on this predicate. | **Reuse unchanged as the shared semantic verdict, but surround it with completeness gates.** The predicate intentionally accepts absent or unparseable units as DD-completeness gaps; the new authority must require resolvable DD and target units before calling it because these 25 reviewed attachments explicitly owe unit agreement. |
| Exact compare-and-set retarget `retarget_standard_name_sources` | `imas_codex/standard_names/provenance_lifecycle.py:305` | Requires a non-empty explicit homogeneous predecessor cohort and exact expected current bindings; rejects stale/claimed/fan-out/scalar drift; synchronizes `PRODUCED_NAME`, `produced_sn_id`, backing `HAS_STANDARD_NAME`, and old/new `source_paths`; provides idempotent manifest semantics. | **Reuse as the mutation primitive for rows already bound to one old name.** Call it on the outer transaction handle with guard already proven and generic receipt ownership retained. Partition by declared predecessor because one invocation forbids heterogeneous old targets. It is not the unbound/absent-source branch. |
| Exclusive binding helper `bind_sources_exclusively` | `imas_codex/standard_names/provenance_lifecycle.py:1604` | Can replace prior edges and rebuild scalar, projection, and target `source_paths` for an explicit list. | **Do not expose or call as the top-level 25-row operator.** It silently narrows to the guard-admitted subset, has no signed old-state compare-and-set, no immutable receipt/replay identity, and no old-name `source_paths` repair. At most extract its synchronized write query into a private, post-lock exact mutation that requires the caller to prove all rows. |
| Semantic-source invariant census and repair | `imas_codex/standard_names/provenance_lifecycle.py:2093` | On an exact allowlist, re-reads source identity/backing ownership/current edges/scalar/projection in one transaction, repairs every mirror, rebuilds affected `source_paths`, and verifies convergence with rollback on drift. | **Reuse the inspection/postflight invariants, not its authority policy.** It chooses a target from an existing sole live edge or scalar; therefore it cannot authorize a new orphan-name attachment and would merely normalize whatever topology was written. After apply, assert every one of the 25 rows is `already_clean` against the signed target; never run policy inference to choose the target. |
| DD projection reconcile `reconcile_standard_name_dd_edges` | `imas_codex/standard_names/graph_ops.py:12770` | Rebuilds missing DD-side `HAS_STANDARD_NAME` projections from existing source provenance and subjects fresh projections to the shared attachment predicate. | **Regression net only.** It is global, projection-only, non-manifested, and deliberately treats missing units as completeness gaps. The exact attachment transaction must write and verify its own projection; a later reconcile should report zero work for these 25 rows. |
| Lock/rehash/guard/retarget orchestration in `supersede_into_ancestor` | `imas_codex/standard_names/graph_ops.py:21967` | Provides a production example of preview hashing, replay validation, full participant locks, closure re-read, exact-set attachment guard, transactional retarget, postconditions, and one durable event. | **Use as the closest end-to-end template, not as a callable operation.** Its ancestor/fold/stale-detach policy is unrelated, but its sequencing is exactly what the attachment adapter needs: derive → hash → lock → rederive → exact guard → synchronized mutation → postflight → receipt/replay. |
| `sn edit` side-car engine `apply_edit` | `imas_codex/standard_names/edit.py:521` | Stages a reasoned hint/rename/docs candidate, records scope and edit provenance, blocks unsafe cascades, and routes the result through ordinary grammar validation and RD-quorum review rather than editing graph text. | **Mandatory refusal route for name defects; never an attachment shortcut.** If adjudication or a uniform guard failure says the target identity itself is wrong, keep the attachment row non-executable and hand the identity to `sn edit`. Only a reviewed accepted target may return to a later signed attachment authority. |
| Existing exact-cohort and lock-drift tests | `tests/standard_names/test_provenance_lifecycle.py:43` | Pin the important distinction: retarget refuses a partially admitted explicit cohort, while the exclusive-bind helper is intentionally capable of mutating an admitted subset. Signed-manifest graph tests separately exercise digest mismatch, collateral drift, lock-time drift, rollback, and replay. | **Reuse as the testing vocabulary.** New graph tests should compose these contracts around `attach`: one rejected row rolls back all 25; a 24/26-row authority refuses; stale/claimed/unit-incomplete/current-binding drift refuses; postflight is exact; replay writes zero. |

The table names 13 candidates. All 13 anchors resolve in the assigned tree;
the quantitative proof is recorded in `anchor-resolution.log` beside the node
manifest.

## Recommended extension boundary

Implement one public, dry-run-first operation over a canonical signed authority,
conceptually `apply_signed_source_attachments(...)`. It should be a thin typed
adapter over `apply_signed_manifest`, or a closed `attach` program interpreted
by the generic registry. It must not accept caller-provided row filters at apply
time: the executable cohort is exactly the signed 25 rows.

Each authority row should carry, at minimum:

- stable row ID, exact `dd:<path>` source ID and prospective target name ID;
- the independently adjudicated candidate/evidence digest and row-set digest;
- exact current DD version, node identity/category, DD unit and immutable DD
  snapshot fields;
- exact target identity/lifecycle/validation/unit and its current producer
  closure;
- source existence/lifecycle/claim state, current scalar, all current live
  bindings, backing node, backing projections, and old-target producer closure;
- declared disposition: create-and-attach, attach-unbound, exact retarget from a
  named predecessor, or already-applied replay;
- named guards for exact row count, signed lifecycle/claims, mandatory unit
  resolution/agreement, attachment consistency, one-live-name-per-source,
  last-producing-source protection on any losing target, exact receipt
  cardinality, and out-of-allowlist immutability.

The source-to-name relation is not one edge. A successful row must leave all of
these synchronized in the same transaction:

1. exactly one live `PRODUCED_NAME` edge to the signed target;
2. `StandardNameSource.produced_sn_id` equal to that target;
3. exactly one backing DD `HAS_STANDARD_NAME` projection to that target;
4. the target's `source_paths` containing the exact source;
5. any losing target's `source_paths` rebuilt from its surviving producers;
6. one immutable `StandardNameChange` receipt tied to authority file digest,
   signed payload digest, fresh manifest digest, row ID and run ID.

For a source already bound to another live target, the authority must name that
predecessor and sign its complete incoming-producer closure. The apply either
uses the exact retarget primitive or refuses. It must never add a second live
binding. If moving the source would remove the predecessor's last producer, the
row refuses unless a separate signed lifecycle disposition explicitly covers
that name; attachment authority alone is not retirement/deletion authority.

For an absent source node, reuse the pinned-snapshot creation logic in the same
outer transaction, then run the guard over the newly materialized source before
the first binding write. If source creation cannot be cleanly factored into the
generic envelope, make it a separately signed prerequisite operation and require
all 25 sources to exist before attachment preview; do not create them through an
unbounded seed or ordinary paid compose run.

## Required execution sequence

1. Load the authority and verify both exact file bytes and canonical signed
   payload digests; require exactly 25 unique row IDs, paths, and declared
   target pairs.
2. Begin one transaction; read the full source/DD/target/old-target/projection
   closure and compute the canonical preview manifest plus collateral baseline.
3. Classify every row without writing. Any ambiguity, stale/active claim,
   absent target, unapproved evidence, missing/unresolvable unit, unexpected
   binding, last-producer loss, or guard rejection makes the exact apply refuse.
4. Return a write-free preview and manifest SHA-256. The apply invocation then
   re-reads the closure; the preview hash authorizes no stale snapshot.
5. Lock every node and relationship participant, re-read and re-hash, and
   require byte-equivalent authority after locking.
6. Require `guard_source_pairings` to admit the exact expected set, with no
   subset semantics and no `already_bound` exemption except receipt-backed
   replay.
7. Execute create/attach or exact retarget mutations, synchronize every mirror,
   write exactly 25 row receipts, and run selected semantic-source postflight.
8. Verify exact counters, zero `LLMCost` delta, byte-identical collateral, no
   new unsourced losing target, and no remaining invariant violation among the
   25 sources; then commit.
9. Replay the same manifest and require `already_applied`, `changed=0`, no new
   receipts, and byte-identical graph closure.

## Test and review matrix

The minimum focused suite for the extension should cover:

- authority file digest and signed-payload digest are independently required;
- exact cohort cardinality: 24, 26, duplicate row/path/target pair, and apply
  narrowing all refuse before mutation;
- one guard rejection or one unit-completeness failure rolls back all 25;
- current source missing, existing-unbound, exact-old-target, stale, claimed,
  fan-out, wrong scalar, and unexpected projection cases;
- absent-source creation pins the exact current DD version and refuses version
  drift or incomplete DD identity/category/unit authority;
- source already on the signed target without the matching receipt is a
  conflict, while matching receipt plus exact postcondition is a zero-write
  replay;
- retarget repairs edge, scalar, backing projection, new `source_paths`, and old
  `source_paths`, and refuses a last-producing-source loss;
- mutation/receipt cardinality is exactly 25 and provider cost remains flat;
- participant or collateral drift between preview, lock, and apply rolls back;
- selected semantic-source postflight is clean for all 25 and the global DD
  projection reconcile has zero work for them;
- a uniform name-level semantic rejection yields a recorded refusal pointing
  to `sn edit`; it never forces an edge or marks the name accepted.

## Explicit non-reuse decisions

- Do not treat semantic-search scores, matching units, or a candidate manifest
  alone as mutation authority.
- Do not call `bind_sources_exclusively` over the 25 rows: its admitted-subset
  behavior violates exact-cohort semantics.
- Do not use `repair_semantic_source_invariants` to select the new target: its
  current-edge/scalar policy is mirror repair, not semantic adjudication.
- Do not rely on `reconcile_standard_name_dd_edges` to finish a partially
  written attachment: the transaction itself must leave every mirror correct.
- Do not use `sn edit`, `sn source-hint`, raw Cypher, hand acceptance, or a paid
  compose run as a substitute for the independently reviewed exact attachment
  authority. `sn edit` is the separate reviewed branch when the name is wrong.
