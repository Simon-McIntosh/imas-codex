# Structural-edge reconciliation migration

Date: 2026-08-25

## Outcome

`reconcile_structural_edges_for_standard_names` now resolves to a fixed
`functools.partial` of `apply_signed_manifest`. The partial selects the closed
`structural-edge-reconcile` adapter, the
`recompute-canonical-structural-edges` mutation program, and exactly three
guards: `exact-request-scope`, `canonical-structural-derivation`, and
`out-of-allowlist-immutability`. It always applies inside its invocation and
retains the public `(graph_client, name_ids) -> int` contract.

The literal public function definition is absent from `graph_ops.py`. The
adapter performs no grammar parse or structural derivation of its own: after
the whole-request preflight, it delegates the requested identities to the
existing `_write_standard_name_edges` canonical writer with
`expand_closure=False`. The successor-rewire authorization added at commit
`b926b8dd` therefore remains in the same canonical writer path and is not
bypassed or duplicated.

## Byte-unchanged behavioral gate

`tests/standard_names/test_structural_edge_reconcile_graph.py` was not edited;
its retained SHA-256 is
`b90e69f129b578d1e7fe017429c488c97f8442ab335d7b74a2be6867c5bd69df`.
Run with `tests/standard_names/test_rewire_authorization_guard.py` and
`tests/graph/test_cypher_property_check.py` against disposable Neo4j
2026.01.4, the final gate executed **11 tests: 11 passed, 0 failed, 0 skipped**
in 16.62 seconds.

The unchanged structural-edge suite retained these exact results:

| Case | Requested | Changed | Exact result |
|---|---:|---:|---|
| Mixed structural cohort | 3 | 3 | `maximum_of_electron_temperature`, `upper_uncertainty_of_electron_temperature`, and `major_radius_of_magnetic_axis` received their canonical `HAS_PARENT`, `HAS_ERROR`, and `HAS_LOCUS` edges; the identical stale edge on non-requested `minimum_of_electron_temperature` remained byte-identical |
| Duplicate identities | 3 copies of 1 identity | 1 | `maximum_of_electron_temperature` reconciled exactly once |
| Empty request | 0 | 0 | Returned the integer `0` and performed no graph read or write |
| Empty identity preflight | 2 | 0 | Raised verbatim `name_ids must contain only non-empty StandardName ids` before mutation |
| Missing-identity preflight | 3 | 0 | Raised verbatim `cannot reconcile structural edges for missing StandardName ids: 'missing_alpha', 'missing_zeta'` before mutation |
| Exact replay | 2 | 2 | Returned the same integer result and left the full graph snapshot byte-identical |

The successor-rewire suite independently retained both authorization outcomes:
the six-hop qualifier relocation to an unauthorized successor tip admitted
**0** rewires and preserved the incumbent edge, while the current unary-prefix
successor admitted **1** rewire. This proves the migration did not weaken the
current-derivation authorization guard.

## Relocation proof

The executable core from `requested = list(dict.fromkeys(name_ids))` through
`return len(requested)` is **31 lines / 1,005 bytes** before and after
relocation. `cmp` returned zero, and both byte streams have SHA-256
`ebec6302581d5eb418ea057e0b69e74bc119ddcda4f075e14a0a6f45f4e4c136`.
The moved core contains the same deduplication, empty-request return, exact
existence query, sorted missing-id refusal, canonical writer call, fixed
`expand_closure=False`, and integer projection; no receipt, refusal, or replay
behavior was re-derived.

Verification used loopback Bolt port 58687 with authentication disabled and a
non-resolving production-endpoint sentinel. The complete test output and both
relocation byte streams are retained under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T092825019429-n-structuraledgemigrate/logs/`.
