# Structural successor-rewire authorization guard

## Outcome

`graph_ops._rewire_has_parent_off_superseded` now treats current structural
derivation as the authorization boundary for successor migration. A
`REFINED_FROM` chain and equality of the existing four compatibility fields are
still necessary, but they are no longer sufficient: the current
`derive_edges(child)` result must name the successor tip as its `HAS_PARENT`
target with the same complete relationship-property map.

The authorization predicate is evaluated twice in the mutation statement:

1. beside the existing `physical_base`, `geometric_base`, `subject`, and
   `component` compatibility checks, before a successor is selected; and
2. immediately before the successor `MERGE`, using the captured edge-property
   map after the predecessor edge is deleted within the same transaction.

This makes refinement lineage incapable of relocating a grammar edge by
itself. A missing or semantically different current derivation leaves the edge
on its superseded parent for explicit investigation.

## Reproduction and non-blanket proof

The disposable-Neo4j regression creates the measured seven-node lineage (six
`REFINED_FROM` hops) from
`signal_to_noise_ratio_of_spectrometer_channel` to
`logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel`. It then
runs the former unguarded Cypher shape and observes one relocation of
`spectral_signal_to_noise_ratio_of_spectrometer_channel` onto the logarithm
tip. After restoring the original qualifier edge, the guarded function reports
zero migrations, retains the correct edge to the plain-ratio predecessor, and
creates no edge to the logarithm tip.

A separate disposable-graph case proves the rule is not a blanket successor
block. `maximum_of_electron_temperature` currently derives
`electron_temperature` with
`operator='maximum', operator_kind='unary_prefix'`; its edge is therefore
migrated successfully from a superseded predecessor to that authorized tip.

| Exercised case | Unguarded relocation | Guarded relocation | Resulting tip authorized by `derive_edges` |
|---|---:|---:|---|
| spectral qualifier through six-hop lineage | 1 | 0 | no |
| legitimate unary-prefix parent | not needed | 1 | yes |

## Measured impact

The input census was the read-only, admission-aware simulation recorded in
`qualifier-parse-verdict.md`. It enumerated 78 distinct non-self successor
`MERGE` pairs that pass the former four-field predicate after production parent
admission. Independent comparison with current lossless `derive_edges` output
authorized none of their resulting tips.

| Measured cohort | Guard refuses | Guard admits |
|---|---:|---:|
| 78 formerly eligible non-self successor pairs | **78** | **0** |

Those figures are evidence inputs from the prior read-only census; this node
did not contact or remeasure the production graph. The executable disposable
tests independently establish both the named refusal mechanism and the valid
unary-prefix admission.

## Verification

The focused gate ran 58 tests against a loopback-only disposable Neo4j 2026.01.4
instance: 58 passed, 0 failed, and 0 skipped. It included both new transaction
cases, every pre-existing test file that references
`_rewire_has_parent_off_superseded`, and
`tests/graph/test_cypher_property_check.py`. The two pre-existing test files
were byte-unchanged. The project endpoint was replaced with an invalid sentinel
for the run, and the test fixture independently refused an endpoint equal to
the configured project URI.

Full log:
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T083217828180-n-rewireauthguard/logs/focused-first.log`.
