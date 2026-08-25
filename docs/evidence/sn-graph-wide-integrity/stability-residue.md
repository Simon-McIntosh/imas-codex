# Standard Name stability residue diagnosis

## Result

The production graph was measured read-only twice, from
**2026-08-25T15:23:19.851253Z to 15:24:19Z**. Both reads returned the same
cohort: **5 semantic sources with no live target, 1 semantic source with two
live targets, and 4 accepted identities with no producing source**. The graph
was not mutated and no model was called.

The important classification is not the count. All five no-live-target rows
have `source_type=dd` and ids beginning `dd:`. Their retained targets have
`origin=null` (four rows) or `origin=pipeline` (one row). Therefore the
partition is **DD 5 / derived 0 / other 0**. The derived zero is valid against
`StandardNameSource.source_type` coverage of **9,668/9,668**. These five are
**genuine lifecycle residue**, not the `source_type=derived` class whose 36
rows the settled liveness reconciliation demonstrated clear earlier.

The five DD rows were created by one later ordinary pipeline run. Their target
names all became terminal between 09:43:19Z and 09:44:37Z in run
`c51460c7-39a0-409d-ac9f-4bc0d7926752`. The exact writer is
`stop_refine_name_attempt` (`imas_codex/standard_names/graph_ops.py:21919`): a
deterministic collision or grammar failure sets `name_stage='exhausted'` while
the producing source remains `composed` or `attached`. The existing
`reconcile_source_status_liveness` program
(`imas_codex/standard_names/graph_ops.py:11610`) is expressly designed to
return exactly this DD shape to `extracted` and clear its terminal edge,
scalar, projection, source-path cache, and composition timestamp.

Disposition totals are therefore:

- **5 `repairable-by-existing-program`** — the five DD liveness rows;
- **5 `needs-new-authority`** — the dual-bound row and the four qualified
  unsourced identities; and
- **0 `expected-transient`** — there is no derived source in the live cohort.

The last category's zero is not inferred from the earlier count of 36. It is
proved from the current source ids and complete `source_type` coverage.

## Five semantic sources with no live target

For this census, a live target is an authored
`StandardNameSource -[:PRODUCED_NAME]-> StandardName` target whose
`name_stage` is neither `superseded` nor `exhausted`. Every row below retains
one terminal edge, its scalar mirror, and its DD projection. Each reported
per-row live-target count of **0** is aimed at the schema-authored direction;
the same read found **5,315** authored edges with source id, target id, and
target stage populated on **5,315/5,315**, while the reverse direction was
**0** with those 5,315 authored edges as its directional positive control.

| Exact source id | Status; source partition | Retained terminal identity | Exact entry time and write path | Disposition |
|---|---|---|---|---|
| `dd:gyrokinetics_local/linear/wavevector/eigenmode/poloidal_turns` | `composed`; `dd` | `poloidal_turn_count` (`origin=null`) | **2026-08-25T09:44:37.167Z**. `stop_refine_name_attempt` exhausted the identity after a `successor_collision` with `poloidal_field_line_turn_count` in run `c51460c7-39a0-409d-ac9f-4bc0d7926752`. | **repairable-by-existing-program**: exact-id `reconcile_source_status_liveness`. |
| `dd:iron_core/segment/geometry/arcs_of_circle/r` | `composed`; `dd` | `radial_coordinate_of_arc_of_circle_center` (`origin=pipeline`) | **2026-08-25T09:44:00.250Z**. `stop_refine_name_attempt` exhausted the identity after `grammar_invalid`, in the same run. | **repairable-by-existing-program**: exact-id liveness settlement. |
| `dd:pellets/time_slice/pellet/path_profiles/position/r` | `composed`; `dd` | `radial_coordinate_of_pellet_path` (`origin=null`) | **2026-08-25T09:43:26.448Z**. `stop_refine_name_attempt` exhausted the identity after a `successor_collision` with `radial_coordinate_of_pellet_path_point`, in the same run. | **repairable-by-existing-program**: exact-id liveness settlement. |
| `dd:summary/line_average/dn_e_dt/value` | `attached`; `dd` | `time_derivative_of_electron_density` (`origin=null`) | **2026-08-25T09:43:32.087Z**. `stop_refine_name_attempt` exhausted the identity after `grammar_invalid`, in the same run. | **repairable-by-existing-program**: exact-id liveness settlement. |
| `dd:summary/local/pedestal/q/value` | `composed`; `dd` | `safety_factor_at_pedestal` (`origin=null`) | **2026-08-25T09:43:19.687Z**. `stop_refine_name_attempt` exhausted the identity after a `successor_collision` with `safety_factor_at_pedestal_top`, in the same run. | **repairable-by-existing-program**: exact-id liveness settlement. |

This is genuine residue even though an existing settlement can repair it. The
distinction is provenance: these are ordinary DD sources parked on identities
made terminal by a paid review/refine run, not newly materialized derived
sources awaiting the normal structural settlement.

## One source with multiple live targets

| Exact source id | Exact live targets | Exact entry time and write path | Disposition |
|---|---|---|---|
| `dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial` | `radial_neutral_internal_state_momentum_flux` (scalar-selected; `origin=pipeline`) and `radial_neutral_state_momentum_flux` (`origin=catalog_edit`) | The incumbent internal-state edge traces to `backfill_refine` at **2026-07-17T05:11:00.962Z**. The second edge entered through the `regenerate` write at **2026-07-28T08:02:37.037656Z**, when `write_standard_names` called `retarget_standard_name_sources` for `z_neutral_state_momentum_flux -> radial_neutral_state_momentum_flux` without removing the independently live backfilled binding. | **needs-new-authority**. The existing signed source-target reconciliation already previewed and refused this exact row because removing `radial_neutral_internal_state_momentum_flux` would leave that identity with **0** producers; that zero is valid against 5,315 authored edges with complete endpoints. A replacement producer or separate identity disposition is required before the closed program can admit it. |

Both targets remain accepted and valid. The semantic adjudication prefers the
shorter `radial_neutral_state_momentum_flux`, but a semantic preference is not
authority to defeat the last-producer closure guard.

## Four accepted identities with no producing source

All four identities have `origin=pipeline`. Each has **0** incoming authored
`PRODUCED_NAME` edges; this is a valid per-row zero because the same query used
the authored direction and its **5,315/5,315** endpoint-complete positive
control. `StandardName.id` and `name_stage` are populated on **4,658/4,658**
nodes, so the four ids and their accepted lifecycle are not missing-property
artifacts.

For three old pre-fix refinements, the graph does not carry a separate source
detachment receipt. Their exact transition evidence is therefore the
successor's `created_at`, its `REFINED_FROM` predecessor, and the later
`backfill_refine` audit row. That means the exact identity-write time and writer
are recoverable, while no claim is made that a later unreceipted detachment
occurred. This is itself evidence that they are durable historical residue,
not a transient structural materialization.

| Exact accepted identity | Exact entry time and write path | Current source evidence and governing disposition | Disposition |
|---|---|---|---|
| `fast_ion_charge_state_power_at_inside_flux_surface` | Created **2026-07-15T08:44:34.608Z** as the refined successor of `fast_ion_state_power_at_inside_flux_surface` through the pre-fix `persist_refined_name` path; `backfill_refine` recorded that lineage at **2026-07-17T05:11:00.962Z**. | Its named DD candidate `dd:waves/coherent_wave/profiles_1d/ion/state/power_inside_fast` is currently attached to `ion_charge_state_power_at_inside_flux_surface`. The DD leaf says `fast` while its prose says thermal-ion deposition, and the distinct thermal sibling repeats that prose. | **needs-new-authority**: upstream DD resolution and a refreshed owner read must precede the already-closed ordinary-source migration. |
| `neutron_flux_due_to_fusion` | Created **2026-08-11T10:33:51.374Z** by `persist_refined_name`; the exact `refine` receipt at **2026-08-11T10:33:51.736269Z** records predecessor `total_neutron_flux_due_to_fusion_reactions` and run `66237f23-96ec-4abe-b1a5-deac9538b22e`. It was born without a producing-source migration. | The reviewed candidate source remains constrained by the exhausted power predecessor's last-producer closure. Ordinary migration exists, but correctly refuses until that predecessor gains an authoritative replacement producer or separately signed disposition. | **needs-new-authority**: the lifting producer/disposition does not exist. |
| `tendency_of_total_thermal_plasma_internal_energy` | Created **2026-07-11T06:33:07.771Z** as a pre-fix refined successor; `backfill_refine` recorded predecessor `tendency_of_total_thermal_internal_energy` at **2026-07-17T05:11:00.962Z**. | Candidate `dd:summary/global_quantities/denergy_thermal_dt/value` currently produces `plasma_internal_energy`. Reusing it would erase the explicit tendency, total, and thermal semantics; the residue needs a reviewed distinct unowned scalar path and sanctioned revalidation. | **needs-new-authority**: no existing program may invent that path or semantic authority. |
| `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | Created **2026-07-15T09:36:32.025Z** as a pre-fix refined successor; `backfill_refine` recorded predecessor `toroidal_trapped_thermal_ion_state_torque_density_due_to_collisions` at **2026-07-17T05:11:00.962Z**. | Its historical candidate source is now `extracted` with **0** bindings; the zero uses the same 5,315-edge directional control. Physics adjudication found that the DD hierarchy describes a trapped non-Maxwellian source transferring torque to a thermal recipient, while the identity conflates both qualifiers onto the recipient. The earlier delete attempt proved the generic deletion branch cannot yet remove the signed connected closure. | **needs-new-authority**: retain the physics retirement decision, but extend and re-authorize the closed deletion machinery before applying it. |

None of the four is `expected-transient`: all are accepted pipeline identities,
all were already explicitly dispositioned in the earlier closing census, and
none is a childful derived parent awaiting provenance settlement.

## Controls, reproducibility, and immutability

The schema sanity read at **2026-08-25T15:23:29.237Z** reported:

| Surface | Covered / candidates |
|---|---:|
| `StandardName.id` | 4,658 / 4,658 |
| `StandardName.name_stage` | 4,658 / 4,658 |
| non-null `StandardName.origin` | 3,567 / 4,658; null is an explicit partition |
| `StandardNameSource.id` | 9,668 / 9,668 |
| `StandardNameSource.status` | 9,668 / 9,668 |
| `StandardNameSource.source_type` | 9,668 / 9,668 |
| Authored `PRODUCED_NAME` source id / target id / target stage | 5,315 / 5,315 on each field |
| Reversed `PRODUCED_NAME` | **0**, controlled by the 5,315 authored-direction edges above |
| `StandardNameChange.id` | 8,596 / 8,596 |
| `StandardNameChange.operation` | 8,596 / 8,596 |
| `StandardNameChange.changed_at` | 8,586 / 8,596; the 10 explicit nulls are not used for transition attribution |

The named positive control is the **5,315 authored
`StandardNameSource -> StandardName` relationships**, all with both endpoint
ids and target lifecycle populated. It controls for relationship direction and
for the keys used to report every no-target/no-producer zero. The aimed
source-partition control is `source_type` coverage of **9,668/9,668**, which
controls the **derived 0** result for the five no-live-target rows.

Artifacts:

- structured result:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T151846068553-n-stabilityresidue/stability-detail.json`
  — SHA-256 `dbde2f3b507331dc9624d1386b793ab0369b8d1358a6cefd5390e67c64d63582`;
- full refreshed query log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T151846068553-n-stabilityresidue/logs/stability-detail-refresh.log`
  — SHA-256 `97b257cf199e175ebd03ff0eb31968fa539977c288e95504cf372d205570216e`;
- read-only driver:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T151846068553-n-stabilityresidue/inspect_stability.py`.

The driver contains only `MATCH`, `OPTIONAL MATCH`, `WHERE`, `WITH`, `UNWIND`,
and `RETURN` queries. It makes no graph write. The two reads retained the same
8,596 `StandardNameChange` rows and 5,315 authored `PRODUCED_NAME` edges; the
evidence run therefore observed no mutation in its measurement window.
