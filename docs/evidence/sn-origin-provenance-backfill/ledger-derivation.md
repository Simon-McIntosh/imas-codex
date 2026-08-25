# Origin derivation from the ordered change ledger

## Verdict

**REFUSAL — the latest `StandardNameChange.origin` is not an authority for a
Standard Name's editorial `origin`.** At the live measurement time,
`2026-08-25T16:03:44.991Z`, the literal rule reproduced only **105 of 1,798**
accepted controls and failed **1,693 of 1,798**. The required failure count is
**0**. The strict held-answer error rate is therefore **94.16%**.

The failure has two independent causes. First, the ledger is incomplete: 214
controls have no change row. Second, `StandardNameChange.origin` describes what
initiated an internal operation, not the identity's editorial axis. Of the
1,584 controls with at least one change row, the most recent row is an exact
origin match for 105, an explicit non-matching value for 1,371, and null for
108. No graph value was written.

The complete machine-readable read-only measurement is retained at
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T155837298440-n-ledgerderivation/logs/ledger-derivation-measurement.json`.

## Rule tested

For every identity, the query followed the schema-declared outgoing
`HAS_INTERNAL_CHANGE` relationship, ordered the linked changes by
`changed_at DESC` with `id DESC` only as a deterministic tie-break, and selected
the first row:

```text
changes = StandardName-HAS_INTERNAL_CHANGE->StandardNameChange
latest = first(changes ordered by changed_at descending)
derived_origin = latest.origin
```

For the held-answer check, a missing row or a null `latest.origin` cannot equal
the declared answer and therefore counts as an error. The report also separates
those unavailable results from explicit non-null disagreements so that coverage
failure is not disguised as classification error.

## Held-answer result

The control predicate was exactly `name_stage = 'accepted'` and declared
`origin IN ['pipeline', 'catalog_edit']`. It reproduced the required cohort of
426 pipeline controls plus 1,372 catalog-edit controls.

| Declared origin | Controls | Exact latest-row match | Explicit mismatch | Latest row has null origin | No change row | Strict errors |
|---|---:|---:|---:|---:|---:|---:|
| `pipeline` | 426 | 104 | 211 | 99 | 12 | **322** |
| `catalog_edit` | 1,372 | 1 | 1,160 | 9 | 202 | **1,371** |
| **Total** | **1,798** | **105** | **1,371** | **108** | **214** | **1,693** |

The operational values found in latest rows include `campaign`,
`semantic_source_reconciliation`, `deterministic_documentation_audit`,
`signed_manifest`, `human`, and `attachment_consistency_reconcile`. Those
values are useful audit provenance for the change event, but they do not share
the `pipeline`/`catalog_edit` editorial vocabulary and cannot be copied into
`StandardName.origin`.

## Ledger coverage

Across **all 2,302 accepted identities**, irrespective of current origin,
**446 have no `StandardNameChange` row of any kind**. The accepted ledger
coverage is 1,856/2,302. The gap comprises 202 accepted `catalog_edit`, 12
accepted `pipeline`, 95 accepted null-origin, and 137 accepted `derived`
identities. Restricting the same measure to the 1,798 held controls gives the
214-row gap shown above.

Across the requested complete null-origin cohort, exactly **362 of 1,091**
identities hold at least one change row and are row-covered at all; **729 of
1,091** have no change row. Even row coverage does not make the scalar an
editorial origin: among the 362 covered identities, 39 have null on the latest
row, 319 have a non-null operational origin outside the two held editorial
values, and only four have a literal editorial value (`catalog_edit` for three,
`pipeline` for one).

## The five shared-import counterexamples

All five names were found with declared `origin='pipeline'`,
`imported_at='2026-07-04T21:21:17.079Z'`, and catalog commit
`a2f8831cf9d14af2f7120969c728f990bdd923cf`. Their ledgers do **not** explain
the declared pipeline origin. Four names carry 15 linked change rows in total,
but zero of those 15 rows has `origin='pipeline'`; `safety_factor` has no change
row at all.

| Accepted control | Change rows | Most recent change | Most recent change origin | Explains declared `pipeline`? |
|---|---:|---|---|---|
| `electron_density` | 4 | `move_source_snapshot`, 2026-08-21 | `source_snapshot_authority` | No |
| `normalized_toroidal_flux_coordinate` | 2 | `repair_invariant_sign_convention_documents`, 2026-08-25 | `deterministic_documentation_audit` | No |
| `safety_factor` | 0 | none | none | No |
| `toroidal_magnetic_field` | 2 | `apply_adjudicated_source_dispositions`, 2026-08-19 | `semantic_source_reconciliation` | No |
| `vertical_coordinate_of_camera` | 7 | `repair_invariant_sign_convention_documents`, 2026-08-25 | `deterministic_documentation_audit` | No |

Thus the ledger neither supplies a pipeline event for the five cases nor
distinguishes why their shared catalog import did not change their declared
editorial origin.

## Schema sanity and positive controls

The graph held 4,658 `StandardName` candidates; all 4,658 carried `id` and
`name_stage`. It held 8,596 `StandardNameChange` candidates; all 8,596 carried
`id` and `operation`, 8,586 carried `changed_at`, and 7,800 carried `origin`.
There were 5,438 schema-declared `HAS_INTERNAL_CHANGE` relationships linking
2,978 distinct Standard Names to 4,855 distinct change rows. These counts prove
that the queries were aimed at live labels, properties, and the declared edge
direction rather than at a plausible but absent key.

Two named positive controls proved that both literal result branches could
fire through the same relationship and ordering instrument:

- `absorbed_radiated_power_at_divertor_target` is accepted with declared
  `pipeline` and has one timestamped `backfill_refine` row whose origin is
  `pipeline`.
- `gyrocenter_frequency` is accepted with declared `catalog_edit` and has one
  timestamped `reclassify_domain` row whose origin is `catalog_edit`.

They control for relationship direction, `changed_at`, `origin`, and both
expected editorial values. They establish observability, not discrimination;
the full 1,798-control result supplies the latter test and fails it.

Every material observed zero is backed by a firing instrument: zero required
errors is the acceptance threshold, not an observed result; zero pipeline
change rows across the five cases was measured over 15 linked rows with 7,800
live change-origin properties; and zero graph mutations is backed by identical
ordered `id/origin/status/name_stage` digests across 4,658 Standard Names before
and after (`9edf3c90fb45c1da138645648f34a086ba5d24efffed20fbaefae0834351ea07`).
The measurement issued only `MATCH ... RETURN` queries.

## Consequence

Do not derive or recover `StandardName.origin` from the most recent
`StandardNameChange.origin`, and do not mutate the 1,091 null-origin identities
from this ledger. A usable recovery authority needs an ordered event whose
field explicitly records the identity's editorial origin, not the actor or
subsystem that initiated an arbitrary internal change, and it must again clear
all 1,798 accepted controls with zero errors.
