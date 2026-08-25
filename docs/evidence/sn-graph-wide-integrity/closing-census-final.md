# Final live closing census

Date: 2026-08-25

Live graph measurement: 2026-08-25T09:08:28.534Z to 2026-08-25T09:09:33.912Z

Source checkout: `2c76e911b0592227f38a3f1f3d48665e26c5f3b7`

Plan authority: `imas-codex:sn-graph-wide-integrity` section 8, version 275

## Verdict

**Seven of the nine declared deliverable rows are CLOSED. Two remain
OUTSTANDING.** The two unrecorded regressions that reopened this plan are now
demonstrably closed: composed/attached sources with no live target are **0**, and
sole-live-target scalar mismatches are **0**. The two outstanding rows are the
previously recorded residual classes, re-examined against current live state:

- **one dual-bound source remains last-producer-blocked**; and
- **two owner/geometry sources still lack an admissible reviewed replacement**.

`OUTSTANDING` is not mutation authority. Each of these three exact source rows
still satisfies its recorded fail-closed condition. No current live fact makes
the previously refused operation safe.

The entire measurement was read-only. It queried no documentation field,
`docs_stage`, `DocsRevision`, or docs-axis review state. A concurrent session is
writing that axis, so including it would measure peer activity rather than this
plan's closing state.

## Nine-row closure table

| Declared deliverable | Live measurement | Verdict |
|---|---|---|
| Every composed/attached semantic source has a live target | **0** of **5,054** semantic-source candidates have zero live targets. | **CLOSED** |
| `produced_sn_id` mirrors the sole live target | **0** sole-live-target scalar mismatches. | **CLOSED** |
| DD/signal upstream projection mirrors the sole live target | **0** projection mismatches. DD identity coverage is 61,366/61,366; signal identity coverage is 46,872/46,872. | **CLOSED** |
| Every accepted identity strictly parses and losslessly round-trips under the installed grammar | **0 failures of 2,295 accepted identities** under `imas-standard-names` **0.8.0rc67**. | **CLOSED** |
| Dual-bound sources are eliminated or carry a named last-producer refusal | **1** dual-bound source remains. Its losing target still has exactly **1** producer, the same source, so the recorded last-producer refusal still applies. | **OUTSTANDING — recorded refusal still stands** |
| Owner/geometry rows leave the old target or carry a named policy/identity refusal | **2** exact rows still select the old accepted target. The field-map replacement still fails the installed grammar; the reflector split identity is still absent. | **OUTSTANDING — both recorded refusals still stand** |
| Live names without producing sources have an exclusive disposition | **4** live unsourced names; all **4/4** occur in the exact 2026-08-22 disposition record. | **CLOSED — qualified dispositions** |
| Structurally bare live names have no live child and a named reason | The same cohort partitions **0 childful / 4 childless**; all **4/4** have named disposition reasons. | **CLOSED — qualified dispositions** |
| Repairs use closed programs under the shared signed envelope, with schema-valid authority construction | The live ledger contains **36/36** selected closed-program change rows with both manifest digest and run id; the current authority builder suite passes **14/14** and both builder and shared envelope entry points are callable. | **CLOSED** |

The current plan fraction, `impl=0.778`, is consistent with this census:
**7/9 rows closed**. It is not being inferred from the fraction; the fraction is
corroborated by the nine independent verdicts above.

## Schema sanity and zero validity

Cypher returns a plausible zero for a missing property, so every zero above was
accepted only after proving the queried keys and lifecycle properties exist on
the candidate population:

| Surface | Candidates | Required property coverage |
|---|---:|---:|
| `StandardName` | 4,656 | `id` 4,656/4,656; `name_stage` 4,656/4,656 |
| `StandardNameSource` | 9,668 | `id` 9,668/9,668; `status` 9,668/9,668; `source_type` 9,668/9,668 |
| Authored `PRODUCED_NAME` relationships | 5,315 | source `id` 5,315/5,315; target `id` 5,315/5,315; target `name_stage` 5,315/5,315 |
| Reversed `StandardName` to `StandardNameSource` `PRODUCED_NAME` relationships | 0 | The schema-authored direction is independently populated at 5,315, proving that the reverse-direction zero is real rather than a guessed empty join. |
| `IMASNode` | 61,366 | `id` 61,366/61,366 |
| `FacilitySignal` | 46,872 | `id` 46,872/46,872 |

The source-integrity predicate is the production predicate: a live target is a
`PRODUCED_NAME` target whose `name_stage` is neither `superseded` nor
`exhausted`; a semantic source is a source whose status is `composed` or
`attached`; and projection parity applies only to `dd` and `signals` sources.
The live partition is:

```text
semantic-source candidates   5,054
no live target                   0
multiple live targets            1
sole-live scalar mismatch        0
upstream projection mismatch     0
```

The lossless grammar result used the repository's strict public `parse(...,
strict=True)` and public IR composer through `parse_canonical_name`. An initial
legacy flat-model parse was discarded because it rejects valid nested operator
trees and therefore cannot measure this deliverable. The replacement census is
the authoritative result: 4,656/4,656 names have both `id` and `name_stage`,
2,295 are accepted, and **2,295/2,295** strictly parse and canonically recompose.

## Residual re-examination

### Dual-bound source: refusal still required

Exact source:
`dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial`

Current source state is `attached`; its scalar is
`radial_neutral_internal_state_momentum_flux`. It has exactly two live,
accepted, valid targets:

| Current live target | Current producing-source closure | Consequence |
|---|---|---|
| `radial_neutral_internal_state_momentum_flux` | **1 producer**: the exact dual-bound source itself | Removing this edge would leave the identity with no producer. The last-producer guard must refuse. |
| `radial_neutral_state_momentum_flux` | **7 producers**, including the exact dual-bound source | This remains the survivor selected by the prior semantic adjudication, but its healthy closure does not authorize orphaning the other accepted identity. |

The prior adjudication proposed retaining
`radial_neutral_state_momentum_flux` and removing
`radial_neutral_internal_state_momentum_flux`. The current graph does not make
that operation newly resolvable: the latter still has no replacement producer.
The row therefore remains outstanding behind the same exact last-producer
condition, rather than behind stale evidence.

### Owner/geometry source 1: field-map grid

Exact source:
`dd:b_field_non_axisymmetric/time_slice/field_map/grid/phi`

Current state is `composed`. Its scalar and sole live target are both the
accepted, valid `toroidal_angle_of_measurement_position`. The intended
owner-specific candidate `toroidal_coordinate_of_field_map_grid` is absent from
the current Standard Name graph and still fails strict parsing under installed
ISN 0.8.0rc67. The old target remains semantically inexact, but there is still
no grammar-admissible reviewed replacement to receive an exact migration. The
recorded vocabulary/policy refusal still stands.

### Owner/geometry source 2: reflector surface center

Exact source:
`dd:spectrometer_x_ray_crystal/channel/reflector/centre/phi`

Current state is `attached`. Its scalar and sole live target are both the
accepted, valid `toroidal_angle_of_measurement_position`. The only current
toroidal reflector identity is accepted, valid
`toroidal_coordinate_of_reflector`, and its only producer is the distinct
`dd:spectrometer_x_ray_crystal/channel/reflector/sphere_centre/phi` source. The
current graph contains no separate toroidal reflector surface/local-frame center
identity; the tested `..._reflector_center` and
`..._reflector_surface_center` spellings both fail the installed grammar.
Migrating `reflector/centre/phi` onto the sphere-center identity would therefore
still collapse two distinct points. The ordinary-reviewed identity split has
not arrived, so the recorded identity refusal still stands.

These conclusions use the locked DD semantics already recorded in the two
adjudication artifacts. No live description or docs-axis field was read.

## Qualified unsourced and structural closure

The live unsourced cohort is exactly four accepted identities, each with zero
live structural children:

| Identity | Current partition | Recorded disposition |
|---|---|---|
| `fast_ion_charge_state_power_at_inside_flux_surface` | childless, unsourced | Held on the named DD population/prose contradiction. |
| `neutron_flux_due_to_fusion` | childless, unsourced | Named attachment/last-producer condition; not a no-measuring-path deletion. |
| `tendency_of_total_thermal_plasma_internal_energy` | childless, unsourced | Named attachment/lifecycle condition; not deletion authority. |
| `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | childless, unsourced | Retired on the recorded source-versus-recipient physics conflict. |

All four appear in `sprint-closing-census.md`; the live set has no fifth row and
no childful row. The unsourced and structural deliverables are therefore closed
as qualified dispositions, not worded into an unqualified zero.

## Shared-envelope closure

Current live `StandardNameChange` evidence for the closed programs selected by
this plan is:

| Operation | Rows | With run id | With manifest SHA-256 |
|---|---:|---:|---:|
| `reconcile_standard_name_source_targets` | 19 | 19 | 19 |
| `migrate_ordinary_standard_name_source` | 3 | 3 | 3 |
| `revive_structural_standard_name_sources` | 2 | 2 | 2 |
| `signed_source_attachment` | 9 | 9 | 9 |
| `release_legacy_dd_source_lifecycle` | 3 | 3 | 3 |
| **Total** | **36** | **36** | **36** |

The current `RepairMutationKind` enum exposes 11 typed mutation kinds, the
schema-valid authority builder and `apply_signed_manifest` entry point are both
callable, and `tests/standard_names/test_repair_authority_builder.py` passes
**14/14**. This is current verification of the shared-envelope deliverable; it
does not depend on the historical count of mutation kinds at the 2026-08-22
close.

## Evidence and exclusions

- Raw live graph census:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T090325031091-n-closingcensus/live-census.json`
  — SHA-256 `37fdc49e3cd527a3a894af60dd172b2885274374c7071350c1930a4e7e432eda`.
  Its legacy-flat-parser grammar subsection is explicitly superseded by the
  lossless grammar census below; all graph counts and exact source rows remain
  authoritative.
- Lossless accepted-name grammar census:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T090325031091-n-closingcensus/grammar-census.json`
  — SHA-256 `1ec7e79f607fb06926d128d534cb1e78f822a111d6ec9b6af4509634a1a0cc08`.
- Authority-builder test log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T090325031091-n-closingcensus/repair-authority-tests.log`
  — SHA-256 `df31c8acb2858155ef4939f52914b290378a9d760972187dfacdc3ed1aa6aa05`.
- The live graph ratchet suite also passed **4/4** after the census:
  `tests/graph/test_sn_integrity_ratchets.py` with `-m graph`.
- Historical semantic authority used without re-reading live docs fields:
  `dual-bound-residue-adjudication.md`,
  `owner-geometry-residue-adjudication.md`, and
  `sprint-closing-census.md`.

No graph mutation, provider call, pipeline action, docs-axis digest, or
`docs_stage` measurement was made by this node.
