# Lifecycle-less stub reconciliation migration

Date: 2026-08-25

## Result

`reconcile_lifecycleless_standard_name_stubs` now resolves to a fixed
`functools.partial` of `apply_signed_manifest`. The partial selects only the
closed `lifecycleless-stub-reconcile` adapter, the
`reconcile-lifecycleless-stub` mutation program, and the complete ordered
registry of four named branch programs:
`materialize-derived-parent`, `delete-dead-link-stub`, `rebind-source`, and
`refused`. The adapter rejects a caller-supplied authority, a different
mutation, or any incomplete, extended, or reordered program tuple. Authority
rows and Cypher therefore remain derived inside the closed adapter; neither an
arbitrary deletion target nor arbitrary Cypher is reachable through the public
compatibility export.

The former literal
`def reconcile_lifecycleless_standard_name_stubs` is absent from
`graph_ops.py`. Its compatibility signature remains
`(*, apply=False, manifest_sha256=None, gc=None)`.

## Byte-unchanged behavioral gate

`tests/standard_names/test_reconcile_lifecycleless_stubs.py` was not edited;
its retained SHA-256 is
`4636bc8b6903d028d2c9635aaa415348c51ebe8b0428d7ad7357400b6e5fc343`.
Run with `tests/graph/test_cypher_property_check.py` against disposable Neo4j
2026.01.4, the final gate executed **26 tests: 26 passed, 0 failed, 0
skipped** in 17.41 seconds.

The four registered programs retained these exact counts and receipt shapes:

| Named program | Admitted | Refused | Receipt and postcondition |
|---|---:|---:|---|
| `materialize-derived-parent` | 1 (`electron_temperature`) | 0 | Exactly 1 `StandardNameChange` with operation `materialize_derived_parent`; the parent becomes accepted and derived while its child edge remains |
| `delete-dead-link-stub` | 1 (`dead_link_endpoint`) | 0 | Exactly 1 deletion `StandardNameChange`; the stub is absent after apply |
| `rebind-source` | 1 (`dd_backed_endpoint`) | 0 | Exactly 1 deletion `StandardNameChange` plus 1 source-reset retry record; the DD source returns to `extracted` with a null scalar |
| `refused` | 0 | 1 per refusal probe | Exactly 0 mutation receipts and 0 graph writes; any one refused row rejects the complete cohort |

The mixed three-row cohort therefore admitted **3**, refused **0**, changed
**3**, reset **1** source, and wrote exactly **3**
`StandardNameChange` rows, one for each admitted identity. Replay returned
`already_applied` with `changed=0` and made no persistent write. The accepted
sibling case independently admitted **1** dead-link stub, refused **0**,
reconciled **1** source scalar, deleted **1** stub, and wrote exactly **1**
deletion receipt without changing the accepted sibling.

Refusal text and transactional behavior also remained exact:

- missing or incomplete DD authority: `incomplete DD source or unit authority`;
- missing parent-owned unit: `no parent-owned unit authority`;
- structural oracle refusal: `structural parent admission refused: suppressed single-child shadow`;
- preview drift: `manifest SHA-256 does not match the fresh lifecycle-less cohort`;
- accepted-sibling binding race: `accepted sibling authority changed before stub deletion`;
- new-producer race: `stub deletion cardinality changed before mutation`.

The rollback probe injected a failure after parent materialization and proved
the parent lifecycle, derived source, and dead stub all returned byte-for-byte
to their pre-transaction state. Both late compare-and-set race probes likewise
left the source scalar, bindings, accepted sibling, and stub unchanged. These
results preserve the complete-cohort rule: a single refusal never permits the
other admitted partitions to commit.

## Relocation proof

The transaction core after lock acquisition, beginning with the fresh cohort
read and ending with the client-close branch, is **224 lines / 9,969 bytes**
before and after relocation. Both byte streams have SHA-256
`2e9411577e4c0ffc735eba746f4c603ceb4aec4b649c522a55db065f6083ecd5`;
`cmp` returned zero. The transient lock query remains the already-adjudicated
closed constant in `graph_ops.py` and is invoked by the relocated core, so the
Cypher-property checker remains green without duplicating an undeclared lock
property at a second source location.

Verification used loopback Bolt port 57687 with authentication disabled and a
non-resolving production-endpoint sentinel. Full logs are retained under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T090731714825-n-lifecyclelessmigrate/logs/`:
`focused-tests-final.log`, `failed-tests-rerun.log`,
`transaction-core-byte-equivalence.log`, `source-contract.log`, and
`neo4j-console-final.log`.
