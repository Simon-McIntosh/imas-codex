# Orphan delete-cohort collision adjudication

Measured read-only against the live graph on 2026-08-21. This record adjudicates the two identities flagged after the 36-row release-provenance partition and checks the other 34 identities against every locked decision in the live plan and against `_KNOWN_AXIS_RESIDUE` in `tests/graph/test_sn_integrity_ratchets.py`. It does not mutate the graph and is not a signed delete manifest.

## Result

- Starting delete cohort: **36** identities, ordered identity SHA-256 `295cda8bb7d9e8393723640f49803bf5d57f5dbe165660582bae4f7c4cd336dc`.
- Collision dispositions: **1 delete**, **1 hold**, **0 re-adjudicate**.
- Remaining identities checked: **34**, all with **no collision** and therefore still assigned to delete.
- Final adjudicated delete count: **35**.
- Held out of the delete cohort: **1** — `parallel_mach_number`.
- Accounting: **35 delete + 1 hold + 0 re-adjudicate = 36**.

The original partition already proves delete conditions 1–3 for all 36 identities: zero producing sources, zero live structural children, and zero catalog-release provenance. This record changes no such measurement. It applies the additional semantic-authority check required before condition 4 can be signed. Any downstream operator must bind its exact manifest count to **35**, not 36, and must refuse on any membership or closure drift.

## Named collision: `parallel_mach_number`

**Disposition: hold. Do not include this identity in the delete manifest.**

The live graph reproduces the exact single row permitted by `_KNOWN_AXIS_RESIDUE`:

| Field | Live value |
|---|---|
| DD source | `dd:langmuir_probes/reciprocating/plunge/mach_number_parallel` |
| Source lifecycle | `composed` |
| Scalar and sole live target | `mach_number` — accepted, valid |
| Projection child | `parallel_mach_number` — accepted, valid, unit `1` |
| Structural relation | `parallel_mach_number` `HAS_PARENT` `mach_number` |
| Relation semantics | `operator_kind=projection`, `axis=parallel` |

The DD path and its documentation say **parallel Mach number**, while the live source is bound to the generic parent `mach_number`. The accepted child is therefore not an unrelated unsourced artifact: it is the exact axis-qualified identity that explains why the source-to-parent row is a known residue and records the intended repair target.

Deleting `parallel_mach_number` would not turn the row into an “unexpected identity” reported by the current test. The ratchet query begins by matching a live projection child; deletion would remove that node and its `HAS_PARENT` edge, so the query would return zero rows while the source remained bound to `mach_number`. In other words, deletion would erase the witness and make the still-wrong generic binding **undetected and unexplained**, rather than repair it. That conflicts with the source-axis fidelity evidence locked in the plan and with the ratchet's explicit named exception.

The locked evidence already determines the direction, so no new semantic decision is required: hold the child against deletion, migrate the exact DD source from `mach_number` to `parallel_mach_number` through the governed source-binding operator, then rerun the axis ratchet and orphan census. A successful retarget makes the child sourced and removes it from the delete predicate.

## Named collision: `cross_section_of_flux_surface`

**Disposition: delete. Disposal is consistent with the locked flux-surface-area decision.**

The locked decision rejects folding a poloidal cross-sectional area into generic `area_of_flux_surface` and requires a distinct identity from the swept toroidal surface area. Current graph closure shows that this requirement has already been realized through reviewed successor identities:

| Identity | Live state | Producers | Meaning in the locked distinction |
|---|---|---:|---|
| `cross_section_of_flux_surface` | pending, quarantined | 0 | Rejected ambiguous predecessor; omits area and the poloidal-plane distinction |
| `poloidal_cross_sectional_area_of_flux_surface` | superseded, valid; score 0.9625 | 0 | Earlier precise spelling, refined onward |
| `poloidal_plane_cross_sectional_area_of_flux_surface` | accepted, valid; score 1.0 | 21 DD sources | Current explicit poloidal cross-sectional-area identity |
| `surface_area_of_flux_surface` | accepted, valid; score 0.9 | 20 DD sources | Distinct swept/full surface-area identity |
| `area_of_flux_surface` | accepted, valid, derived umbrella | 1 derived source | Structural umbrella only; it does not own the 21 DD cross-sectional-area sources |

The rejected spelling is also an earlier predecessor of `area_of_flux_surface`, but it has no current source or live structural child. Deleting this unreleased pending/quarantined residue does not fold a source into the generic umbrella and does not remove the explicitly poloidal identity. It removes the ambiguous spelling while leaving both physical measures independently live and sourced. The current accepted alternative is `poloidal_plane_cross_sectional_area_of_flux_surface`; the previously cited `poloidal_cross_sectional_area_of_flux_surface` has itself been superseded through ordinary reviewed refinement.

Accordingly, the old `cross_section_of_flux_surface` row remains in the delete cohort. The downstream ledgered deletion must preserve the existing successor/change history in its receipt, but no further semantic re-adjudication is needed before signing this row.

## Remaining 34 identities: locked-decision and ratchet scan

The live plan's locked decisions were checked as a complete set: one-live-name-per-source; conservative orphan disposition; parent-owned dimensional units; distinct poloidal versus swept flux-surface area; unreleased WEST repair scope; geometry cardinality; mean-based budget reservation and presentation; topology repair followed by recomposition; no ordinal positions in identity; guarded funded-drain cadence; local compose readiness; snapshot-as-forensic-comparator; delete-unpublished authority; and the four-condition delete bound. Operational decisions unrelated to identity semantics do not create a collision. Validation, grammar, unit, or transient failures are not used as deletion triggers for any row.

The exact live ratchet query returned only the `parallel_mach_number` tuple above. Thus an axis word in another identity is not treated as a collision by spelling alone: a collision requires the complete live source → generic parent plus live projection-child topology represented by `_KNOWN_AXIS_RESIDUE`.

| Identity | Collision check | Disposition |
|---|---|---|
| `capacitance_of_ion_cyclotron_heating_antenna` | No collision — `catalog_edit` is editorial origin, not release provenance; the four delete bounds remain the authority. | delete |
| `fast_ion_charge_state_power_at_inside_flux_surface` | No collision — neither an ordinal-position identity nor a flux-surface-area identity. | delete |
| `flux_surface_averaged_toroidal_flux_coordinate_gradient_magnitude` | No collision — flux-surface averaging is not the locked cross-sectional-versus-swept area distinction. | delete |
| `forward_wave_phase_of_ion_cyclotron_heating_antenna` | No collision — the name does not encode the source description's first-module reference, so it honors the no-ordinal rule. | delete |
| `impurity_ion_photon_radiance_of_spectral_line_due_to_charge_exchange` | No collision — no locked owner, geometry, area, ordinal, or axis-residue commitment applies. | delete |
| `line_integrated_electron_density` | No collision — no locked decision requires retaining this exact unsourced identity. | delete |
| `magnetic_field_at_pedestal_top_low_field_side_magnitude` | No collision — its locus and side are semantic locations, not ordered sample positions. | delete |
| `minimum_magnetic_field_magnitude` | No collision — `minimum` is an aggregation, not forbidden sample ordinality. | delete |
| `minimum_of_safety_factor` | No collision — `minimum` is an aggregation, not forbidden sample ordinality. | delete |
| `neutral_state_power_density` | No collision — `catalog_edit` is not publication evidence and no other locked identity commitment applies. | delete |
| `neutron_flux_due_to_fusion` | No collision — no locked identity commitment applies beyond the satisfied delete bounds. | delete |
| `parallel_current_density_due_to_ohmic_current_drive` | No collision — axis-qualified spelling alone is not `_KNOWN_AXIS_RESIDUE`; the live ratchet returned no tuple for this identity. | delete |
| `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | No collision — axis-qualified spelling alone is not the exact live residue topology. | delete |
| `parallel_neutral_momentum_diffusion_coefficient` | No collision — axis-qualified spelling alone is not the exact live residue topology. | delete |
| `poloidal_neutral_internal_state_convection_velocity` | No collision — poloidal qualification is not the locked flux-area identity and has no generic-parent residue row. | delete |
| `poloidal_neutral_internal_state_momentum_convected_velocity` | No collision — poloidal qualification has no matching generic-parent residue row. | delete |
| `poloidal_neutral_state_particle_convection_velocity` | No collision — poloidal qualification has no matching generic-parent residue row. | delete |
| `poloidal_straight_field_line_angle` | No collision — the poloidal component is not ordered sample position and has no matching generic-parent residue row. | delete |
| `radial_effective_electron_diffusivity` | No collision — radial qualification has no matching generic-parent residue row. | delete |
| `radial_effective_ion_diffusivity` | No collision — radial qualification has no matching generic-parent residue row. | delete |
| `radial_effective_neutral_diffusivity` | No collision — radial qualification has no matching generic-parent residue row. | delete |
| `radial_thermal_ion_charge_state_energy_diffusion_coefficient` | No collision — radial qualification has no matching generic-parent residue row. | delete |
| `tendency_of_total_thermal_plasma_internal_energy` | No collision — the transformation is not an ordinal position and no locked identity commitment applies. | delete |
| `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | No collision — toroidal projection and orbit class are semantic qualifiers; no generic-parent residue row exists. | delete |
| `toroidal_ion_charge_state_torque_density` | No collision — toroidal qualification has no matching generic-parent residue row. | delete |
| `toroidal_line_integrated_impurity_ion_velocity` | No collision — toroidal qualification has no matching generic-parent residue row. | delete |
| `toroidal_neutral_state_momentum_diffusivity` | No collision — toroidal qualification has no matching generic-parent residue row. | delete |
| `toroidal_thermal_ion_charge_state_torque_due_to_collisions` | No collision — toroidal qualification has no matching generic-parent residue row. | delete |
| `toroidal_thermal_ion_torque_density_due_to_thermalization` | No collision — toroidal qualification has no matching generic-parent residue row. | delete |
| `toroidal_trapped_fast_ion_charge_state_torque_density_due_to_collisions` | No collision — toroidal projection and orbit class have no matching generic-parent residue row. | delete |
| `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | No collision — toroidal projection and orbit class have no matching generic-parent residue row. | delete |
| `variation_of_length_of_interferometer_beam` | No collision — variation is a quantity transformation, not endpoint or sample ordinality. | delete |
| `x_direction_unit_vector_of_sensor` | No collision — the live explicit-axis ratchet returned no source/generic-parent tuple for this identity. | delete |
| `z_direction_unit_vector_of_sensor` | No collision — the live explicit-axis ratchet returned no source/generic-parent tuple for this identity. | delete |

The table contains exactly **34** identities. Together with the collision dispositions, it yields the final exact adjudication of **35 deletes and 1 hold**.

## Evidence and mutation boundary

- Input partition: `docs/evidence/sn-graph-wide-integrity/orphan-release-partition.md` — 36 delete candidates, 0 supersede candidates, identity SHA-256 `295cda8bb7d9e8393723640f49803bf5d57f5dbe165660582bae4f7c4cd336dc`.
- Locked semantic authority: `docs/sn-graph-wide-integrity.html`, plan version 210.
- Ratchet authority: `tests/graph/test_sn_integrity_ratchets.py`, including the exact `_KNOWN_AXIS_RESIDUE` tuple.
- Prior per-identity semantic evidence: `docs/evidence/sn-graph-wide-integrity/unsourced-name-adjudication.md`.
- Live collision query log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T104147015089-orphanadj/live-graph-collision-query.log`, SHA-256 `ea38dc81ae89bda5654078a3ea549d6b086459246a76795c2ca3d537d294c978`.
- Live flux-area closure log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T104147015089-orphanadj/live-graph-flux-area-closure.log`, SHA-256 `63b51a8bfd0ddbfe5b5d6ec1aa3b02f3479bfa76a0bbdd6da8f7372cef44be5a`.
- Live full-axis ratchet log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T104147015089-orphanadj/live-graph-axis-ratchet.log`, SHA-256 `35cfc6c9a39ca1ed93527afce3321be13371ba1145661792cbf1c7b800ac3d9f`.
- Read-only counters at the final graph query: `StandardNameChange=7704`, `LLMCost=27591`; the query executed no write and no provider call.

No manifest was signed. This evidence authorizes the coordinator to construct a separate exact 35-row candidate manifest; the applying operator must independently re-read all four delete conditions, bind its row count to 35, ledger every deletion, and refuse the whole cohort on drift.
