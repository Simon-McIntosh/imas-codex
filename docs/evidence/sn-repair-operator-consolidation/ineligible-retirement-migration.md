# Ineligible-source retirement operator migration

Date: 2026-08-25

## Result

`retire_ineligible_standard_name_sources` is now a fixed
`functools.partial` of `apply_signed_manifest`. The partial selects the closed
`ineligible-source-retirement` adapter, the
`release-ineligible-source-authority` compound mutation, and exactly these
guards: signed lifecycle and claim state, permitted orphan hand-off, and
out-of-allowlist immutability. It deliberately does **not** select a
last-producing-source guard: reporting a newly orphaned name to the downstream
orphan workflow is the signed behavior of this operator.

The former public function body and the literal
`def retire_ineligible_standard_name_sources` are absent from `graph_ops.py`.
The runtime export remains source compatible and has the signature
`(source_ids, *, reason, apply=False, manifest_sha256=None, run_id=None,
gc=None)`.

## Byte-unchanged equivalence gate

The existing disposable-Neo4j suite
`tests/standard_names/test_dual_binding_dedup.py` remained byte-unchanged and
passed all **9 of 9** cases with **0 skipped and 0 failed**. The three Cypher
property checks also passed, making the focused invocation **12 passed, 0
skipped, 0 failed** in 16.85 seconds.

The operator-specific identities, counts, and reasons remained exact:

| Case | Admitted | Refused | Orphan report | Receipt and replay |
|---|---:|---:|---|---|
| Ineligible `dd:structuralcontainer/value`, backing category `structural` | 1 source; 2 bindings and 2 projections detached | 0 | 2: `structuralcontainer_redundant`, `structuralcontainer_selected` | Exactly 1 `StandardNameChange`; both names remain accepted for intentional orphan hand-off |
| Eligible `dd:quantitysource/value`, backing category `quantity` | 0 | 1 source, verbatim reason `backing DD node category 'quantity' is SN-eligible` | 0 | 0 writes; source remains attached with both bindings |
| Replay of `dd:retirementreplay/value`, backing category `representation` | 1 on the original apply | 0 | 2 on apply and the same 2 returned on replay | Replay is `already_applied`, `changed=0`, with a byte-identical full graph snapshot |

The applied ineligible source retains the exact postcondition: status
`not_physical_quantity`, null scalar mirror, zero `PRODUCED_NAME` bindings, and
zero backing `HAS_STANDARD_NAME` projections. No name is superseded by this
operator.

## Auditable relocation

The transaction core was selected from the first statement
`requested = sorted(set(source_ids))` through the client-close branch. Before
and after relocation it is exactly **17,200 bytes** with SHA-256
`9b0679953b79fc2abd160a2c2ffca446ad768427c4acd898d6eb137c22aa7e10`.
The comparison is byte-for-byte equal; only the adapter seam was added ahead of
the core to fix the mutation and guard registry choices.

Verification used disposable Neo4j 2026.01.4 on loopback Bolt port 56687 with
authentication disabled; no production endpoint was contacted. Complete logs
are retained under the worker run directory as `logs/pytest-focused.log`,
`logs/transaction-core-byte-equivalence.log`, and `logs/neo4j-console.log`.
