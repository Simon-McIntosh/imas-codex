# Catalog-edit source-disposition instrument and signed preview

## Outcome

The exact source-disposition instrument is implemented in
`imas_codex/standard_names/graph_ops.py`. Its write-free production preview
covered all **216** adjudicated catalog-edit dual bindings and returned
`would_apply`: **216 admitted + 0 fail-closed refusals = 216 requested**.
No live mutation or provider call occurred.

The signed preview would reconcile the complete cohort to one surviving live
target per source by changing **105** scalars, then removing **240** exact
`PRODUCED_NAME` bindings and their **240** exact `HAS_STANDARD_NAME`
projections. The scalar changes are the adjudicated 102 retargets plus three
missing-scalar selections; the 111 retained scalars remain unchanged.

## Signed authority

The committed adjudication artifact is
`docs/evidence/sn-graph-wide-integrity/catalog-edit-dual-binding-adjudication.json`:

- file SHA-256:
  `5ca7761a7b022ac7889387d7bf63a027114a168cc3785ed4fdc8d31c08417b6e`;
- declared canonical payload SHA-256:
  `c227e70ec5cd940577ca778ce5ec63e4df3a63bf68c3e845eba92d0a4b9a0efb`;
- canonical signed row-set SHA-256:
  `0cb907d04aeb33b46e1f8ede9b5927d2c574f3500200750c8bc4734e6f8633bf`;
- exact live preview manifest SHA-256:
  `26bd75977fb0a505e9d829ce8eaf4aba26947f2c1a69d2a11f44e7a239e29190`.

The artifact's outer payload and every per-row signature use sorted compact
JSON with the standard ASCII escaping retained. A dedicated verifier implements
that already-signed contract; it does not reuse or alter the graph participant
canonicalizer. A regression pins the helper to the known-good outer payload
digest above and verifies all 216 row signatures, so changing either canonical
form cannot silently reinterpret the committed authority.

## Instrument contract

`apply_adjudicated_source_dispositions` validates the artifact schema, outer
payload digest, all row digests, the closed disposition set, row uniqueness,
candidate/survivor/removal identities, and summary counts before graph access.
It then signs the complete live source, target, binding, DD backing, origin and
projection closure. Admission refuses missing or stale sources, active claims,
scalar drift, candidate-binding drift, loss of the catalog-edit participant,
non-exact DD backing, and projection drift.

An apply requires the exact preview manifest SHA-256. Inside one transaction,
the instrument locks every signed participant, rebuilds the authority manifest,
requires a byte-identical digest, compare-and-sets both the adjudicated scalar
and null claim state, and deletes only the signed relationship element IDs. A
single internal `StandardNameChange` receipt and source-path mirror refresh are
part of the same transaction. Exact replay verifies the one-target
postcondition and rolls back without a persistent write.

## Out-of-allowlist immutability and counter proof

The preview allowlist contains exactly the 216 signed source IDs. The complete
participant closure for every other live `StandardNameSource` was read before
and after the preview:

| Measure | Before | After | Verdict |
|---|---:|---:|---|
| Out-of-allowlist source rows | 9,282 | 9,282 | identical population |
| Out-of-allowlist closure SHA-256 | `b9864539a6d64c523a1259522f6c7ab1a0adf63c1a9f46a238b66f9c7e819f24` | `b9864539a6d64c523a1259522f6c7ab1a0adf63c1a9f46a238b66f9c7e819f24` | immutable |
| `StandardNameChange` nodes | 7,451 | 7,451 | unchanged |
| `LLMCost` nodes | 27,467 | 27,467 | unchanged |

The matching complete-closure digest proves every source outside the signed
allowlist, together with its properties, target bindings, backing origins and
projections, remained immutable. The unchanged ledger and cost counts prove
that preview rollback created neither an operation receipt nor provider spend.

## Regression evidence

The final focused file executed **6 passed, 0 failed, 0 skipped**. Five cases
ran against a disposable Neo4j instance and cover:

1. retain-scalar, retarget-scalar and select-missing-scalar dispositions in one
   exact apply, with one-target postconditions and byte-identical
   out-of-allowlist state;
2. write-free exact replay;
3. scalar drift between preview and apply;
4. active-claim drift between preview and apply; and
5. incomplete backing-projection authority.

The sixth case is the committed-artifact signature regression: it pins the
known-good outer digest and all 216 per-row signatures. A separate tamper path
proves altered disposition content is rejected before graph access.

Durable logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T060926254275-catalog-edit-disposition-instrument/disposable-tests-authorized-final.log`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T060926254275-catalog-edit-disposition-instrument/live-preview-authorized.log`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T060926254275-catalog-edit-disposition-instrument/signature-tamper-test-final.log`.

The live preview was read at plan version 176. The adjudication bytes and the
authority digests referenced by the prior version were not regenerated or
re-signed.
