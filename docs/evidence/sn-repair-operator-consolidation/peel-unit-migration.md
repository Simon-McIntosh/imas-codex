# Normalization-peel parent-unit migration

Date: 2026-08-25

## Outcome

`repair_normalization_peel_parent_units` now resolves to a fixed
`functools.partial` of `apply_signed_manifest`. The partial selects the closed
`normalization-peel-unit-repair` adapter, the
`clear-normalization-peel-parent-unit` mutation program, and exactly two
guards: `corrected-normalization-peel-unit-authority` and
`out-of-allowlist-immutability`. It always applies inside its invocation and
retains the public `(graph_client) -> list[str]` contract.

The literal public function definition is absent from `graph_ops.py`. The
adapter auto-discovers the complete live cohort and retains the corrected
predicate: it projects non-null-unit children into `unit_kids` before applying
the every-child quantifier. A null-unit normalization child therefore cannot
veto admission.

## Byte-unchanged behavioral gate

`tests/standard_names/test_normalization_peel_unit_repair_graph.py` was not
edited; its retained SHA-256 is
`1e7a02a718c1295f74eb5f3a1fcfb1fa9c2bb6c38a622c61dd3a4f6b69df034b`.
Against disposable Neo4j 2026.01.4, the final gate executed **7 tests: 7
passed, 0 failed, 0 skipped** in 16.68 seconds.

The unchanged suite retained these exact results:

| Case | Admitted | Refused | Exact result |
|---|---:|---:|---|
| Mixed six-parent cohort | 2 | 4 | Returned `electric_current`, then `particle_mass`; left `normalized_collisionality`, `magnetic_field`, `ion_density`, and `electron_temperature` unchanged |
| Own normalization marker | 0 | 1 | Returned the predecessor's exact empty list and left parent state byte-identical |
| Missing unit-consistency finding | 0 | 1 | Returned the predecessor's exact empty list and left parent state byte-identical |
| Non-normalization unit-bearing child | 0 | 1 | Returned the predecessor's exact empty list and left parent state byte-identical |
| Null-unit normalization child | 1 | 0 | Returned `particle_mass`, cleared scalar unit `1`, and removed its unit edge; this is the corrected one-row distinction |
| Scalar-only candidate | 1 | 0 | Returned `particle_mass` and cleared the scalar even when no unit edge existed |
| Exact replay | 2, then 0 | 0, then 2 already repaired | First returned the same sorted two-name projection; replay returned the exact empty list and left the full graph snapshot byte-identical |

The predecessor exposes non-admission as the exact empty-list projection rather
than textual refusal objects. Those verbatim results remain unchanged, and the
four mixed-cohort identities continue to refuse for the same predicate clauses:
own normalization marker, absent unit-consistency finding, non-derived origin,
and a non-normalization unit-bearing child.

## Relocation proof

The executable core from `rows = gc.query(...)` through `return repaired` is
**30 lines / 1,127 bytes** before and after relocation. `cmp` returned zero,
and both byte streams have SHA-256
`74d180c04183c1ca71fe0ad20b56226642c53c704f0c4fae0c63ebeaed1f3208`.
The moved core contains the same corrected non-null `unit_kids` projection,
normalization-token predicate, unit-edge deletion, scalar clear, sorted return
projection, logging, and replay selection; none of those behaviors was
re-derived.

`tests/graph/test_cypher_property_check.py` also remained green at **3 passed,
0 failed**. Verification used loopback Bolt port 59687 with authentication
disabled and a non-resolving production-endpoint sentinel. The disposable
server stopped cleanly, and both Bolt 59687 and HTTP 59474 were closed after
the run.
