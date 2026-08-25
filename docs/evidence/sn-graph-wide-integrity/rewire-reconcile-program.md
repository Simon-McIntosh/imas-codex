# Governed successor-rewire reconciliation program

## Outcome

`retire_unauthorized_has_parent_relocations` is a closed signed-manifest
program for moving a live Standard Name's exact `HAS_PARENT` relationship off
an unauthorized refinement successor and back onto the exact parent emitted by
current `derive_edges(child)` output. The authority is emitted by
`build_repair_authority()` and executed only through `apply_signed_manifest()`;
it carries no caller-supplied Cypher or open selection predicate.

Each row must sign the child, incumbent tip, derivable replacement parent,
complete incumbent relationship identity, and complete non-null relationship
property map. At preview and again after participant locking, the executor
requires all of the following:

- current derivation does not authorize the incumbent tip with that property
  map;
- current derivation does authorize the signed replacement parent with that
  exact property map;
- the live child has exactly the signed incumbent parent edge;
- the child remains live, and its lifecycle and producing-source closure do not
  change through the transaction; and
- all graph state outside the signed participant closure remains immutable.

The mutation is an exact compare-and-set: delete the signed incumbent edge and
create the signed derivable edge with the same complete property map. It cannot
leave a derivable live child parentless.

## Disposable-graph proof

The graph test built an authority containing three rows in one signed cohort.
Its preview was deliberately partial rather than all-or-none, so an unrelated
protected edge does not hold an independently safe reconciliation.

| Fixture row | Admitted | Refused | Verbatim reason |
|---|---:|---:|---|
| Six-hop spectral relocation | 1 | 0 | — |
| Legitimate `maximum_of_electron_temperature` unary-prefix edge | 0 | 1 | `current derivation still authorizes incumbent HAS_PARENT tip` |
| Live child whose signed replacement is not its derivable parent | 0 | 1 | `removal would leave a derivable HAS_PARENT path absent` |
| **Total** | **1** | **2** | — |

The admitted fixture reproduces the six-hop `REFINED_FROM` lineage from
`signal_to_noise_ratio_of_spectrometer_channel` to
`logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel`. The
program removes the relocated spectral-to-logarithm edge and restores
`spectral_signal_to_noise_ratio_of_spectrometer_channel` to its current
derivation parent, `signal_to_noise_ratio_of_spectrometer_channel`, preserving
`operator=spectral` and `operator_kind=qualifier`.

Apply reported **changed 1, receipt rows 1**. The receipt is keyed to the one
removed relocated edge's authority row. Exact replay reported
**already_applied, changed 0, persistent writes 0, receipt rows 1**, and a full
node-and-relationship snapshot was byte-equivalent before and after replay.

## Measured 78-pair impact

The read-only predecessor census enumerated **78** distinct non-self successor
rewire pairs after production parent admission. It also recorded that current
`derive_edges` output authorizes **0 of the 78 successor tips** and retains an
exact original derived parent and property map for each simulated move. Against
this closed program's predicates, that recorded cohort partitions as:

| Prior measured cohort | Program would admit | Program would refuse |
|---|---:|---:|
| 78 successor relocations | **78** | **0** |

This is a predicate projection over the prior read-only census, not a fresh
production preview and not mutation authority. A later production authority
must still be built from then-current element identities and exact relationship
closures; any missing participant, changed edge, newly authorized tip, absent
derivable replacement, or additional parent edge becomes a visible refusal.

## Verification boundary

The focused disposable-Neo4j run passed **1/1 tests, 0 failures, 0 skips** under
SLURM job `1254629`. The fixture required
`IMAS_CODEX_TEST_NEO4J_EPHEMERAL=1`, refused an endpoint equal to the resolved
project URI, started an empty temporary Neo4j 2026.01.4 instance, and destroyed
that instance after the test. **No production graph was queried or mutated.**

The signed-manifest regression gate passed **17 tests, 0 failures** across the
authority builder, generic operator, structural reparent, and structural
release suites. Ruff check and formatting were clean on both changed Python
paths.

Logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T095155506755-n-rewirereconcileprogram/logs/test-rewire-retire-graph.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T095155506755-n-rewirereconcileprogram/logs/disposable-neo4j-job.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T095155506755-n-rewirereconcileprogram/logs/test-signed-manifest-regression.log`
