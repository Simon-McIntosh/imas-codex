# Bare structural census

## Outcome

**PASS — 29 of 29 live bare names are partitioned, with 0 unpartitioned.**
The read-only production invocation derived the cohort after the signed
dual-bound reconciliation and signed structural-source revival had settled. It
did not treat the cohort as one undifferentiated bare total:

| Partition measure | Count |
|---|---:|
| Derived live bare cohort | **29** |
| With at least one live structural child | **9** |
| With no live structural child | **20** |
| `transient-reconcile-output` | **9** |
| `revivable-through-the-signed-program` | **0** |
| `genuine-gap-with-a-named-reason` | **20** |
| Partitioned | **29** |
| Unpartitioned | **0** |

Here, *bare* means a live, nonterminal `StandardName` with no incoming
`PRODUCED_NAME` relationship from any `StandardNameSource`. A live name or
child excludes lifecycle stages `superseded` and `exhausted` and status values
`deprecated` and `superseded`. The `genuine-gap-with-a-named-reason` label is a
provenance disposition, not delete authority: several rows have a known
recovery route whose current lifecycle, semantic, or last-producer condition
is named below.

## Settled transaction authority

Completion was checked by receipt identity, not by guessing an operation name
or observing only a global counter.

| Transaction | Exact receipt selector | Durable rows |
|---|---|---:|
| Structural-source revival | `run_id=r-20260821T223719510924-structapply`, `manifest_sha256=ce5302b1baaafaed897c77287610f60c91ab86e00b9f7be784137e4a713d33be` | **2** |
| Dual-bound source reconciliation | `run_id=dual-bound-source-reconciliation`, `manifest_sha256=b202f2637e6fba595508e70f809febdbae90e7caed12c919c8e2f46b4d34519b` | **19** |

The two revival receipts are exactly `derived:electron_diffusivity` and
`derived:ion_diffusivity`. Both names are now sourced and therefore correctly
absent from the current bare cohort. That is why the live
`revivable-through-the-signed-program` count is zero rather than two: the
signed program has already revived its stale-source cohort.

The 19 dual-bound receipts identify the transaction that made the nine
childful parents visible. Its exact authority removed over-broad direct DD
bindings while retaining more specific survivors. The current childful set is
therefore reconcile output, not 9 newly discovered childless orphans.

## Childful partition: 9 transient reconcile outputs

All nine have live structural children, no `derived:<name>` source node, and
no producing relationship. The same invocation passed the exact set through
`classify_orphan_parent_source_candidates`: **9 repairable, 0 rejected**. They
are eligible for deterministic ordinary parent-source seeding; none needs the
stale-source-only signed revival branch.

| Bare parent | Live structural children | Exact reconciled DD source row(s) that removed the broad binding | Disposition |
|---|---|---|---|
| `area_of_langmuir_probe` | `effective_area_of_langmuir_probe`; `wetted_area_of_langmuir_probe` | `dd:langmuir_probes/embedded/surface_area`; `dd:langmuir_probes/reciprocating/surface_area` | `transient-reconcile-output` |
| `electrostatic_potential_imaginary_part` | `perturbed_electrostatic_potential_imaginary_part` | `dd:mhd_linear/time_slice/toroidal_mode/plasma/phi_potential_perturbed/imaginary` | `transient-reconcile-output` |
| `momentum_source` | `total_momentum_source`; `poloidal_momentum_source`; `radial_momentum_source`; `ion_charge_state_momentum_source`; `neutral_momentum_source`; `ion_momentum_source` | `dd:core_sources/source/profiles_1d/ion/momentum/radial` | `transient-reconcile-output` |
| `neutral_species_energy_convection_velocity` | `parallel_neutral_species_energy_convection_velocity` | `dd:plasma_transport/model/ggd/neutral/energy/v_parallel/values` | `transient-reconcile-output` |
| `neutral_state_particle_diffusivity` | `poloidal_neutral_state_particle_diffusivity` | `dd:edge_transport/model/ggd/neutral/state/particles/d_pol/values` | `transient-reconcile-output` |
| `normalized_perturbed_current_density` | `parallel_normalized_perturbed_current_density` | `dd:gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_gyrocenter_bessel_1/j_parallel` | `transient-reconcile-output` |
| `outer_squareness_of_flux_surface` | `upper_outer_squareness_of_flux_surface`; `lower_outer_squareness_of_flux_surface` | `dd:equilibrium/time_slice/profiles_1d/squareness_upper_outer` | `transient-reconcile-output` |
| `parallel_normalized_perturbed_current_density` | `parallel_normalized_perturbed_current_density_bessel_1` | `dd:gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_gyrocenter_bessel_1/j_parallel` | `transient-reconcile-output` |
| `volume_averaged_runaway_electron_current_density` | `parallel_volume_averaged_runaway_electron_current_density` | `dd:runaway_electrons/global_quantities/volume_average/current_density` | `transient-reconcile-output` |

The shared disposition reason is exact: the signed reconcile removed a broad
DD interpretation, the listed child preserves the parent's structural role,
and the ordinary structural classifier admits reconstruction of the derived
provenance source. Reattaching the removed DD row to the parent would undo the
semantic repair; the deterministic `derived:<parent>` source is the correct
provenance shape.

## Childless partition: 20 genuine provenance gaps

Each row has zero live structural children and zero producing sources. Every
row carries a named reason, so none is being collapsed into an unexplained
orphan count.

| Bare name | Current lifecycle | Named reason / next authority condition |
|---|---|---|
| `capacitance_of_ion_cyclotron_heating_antenna` | accepted, valid | Candidate-source retarget refused because the incumbent target would lose its final producer. |
| `cross_section_of_flux_surface` | pending, quarantined | Legacy ambiguous identity; the source must target the distinct poloidal-plane cross-sectional identity without an umbrella fold. |
| `fast_ion_charge_state_power_at_inside_flux_surface` | accepted, valid | DD `power_inside_fast` declares W but describes thermal power while a `power_inside_thermal` sibling exists. |
| `line_integrated_electron_density` | drafted, quarantined | Legacy identity; the source must target `line_integrated_electron_number_density` directly without a lineage fold. |
| `magnetic_field_at_pedestal_top_low_field_side_magnitude` | accepted, valid | Accepted through its existing quorum after the first attachment refusal; the governed attachment has not yet been re-applied. |
| `minimum_of_safety_factor` | reviewed, valid | The reviewed target has not earned accepted lifecycle. |
| `neutral_state_power_density` | accepted, valid | Accepted through its existing quorum after the first attachment refusal; the governed attachment has not yet been re-applied. |
| `neutron_flux_due_to_fusion` | accepted, valid | Candidate-source retarget refused because the incumbent target would lose its final producer. |
| `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | accepted, valid | Candidate-source retarget refused because the incumbent target would lose its final producer. |
| `parallel_neutral_momentum_diffusion_coefficient` | accepted, valid | Candidate DD path is state-resolved while this identity is species-level. |
| `poloidal_neutral_internal_state_momentum_convected_velocity` | reviewed, valid | The reviewed target remains below the accepted lifecycle gate. |
| `poloidal_straight_field_line_angle` | drafted, quarantined | Legacy identity; the source must target `straight_field_line_angle` directly without a lineage fold. |
| `tendency_of_total_thermal_plasma_internal_energy` | accepted, quarantined | The identity remains validation-quarantined. |
| `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | accepted, valid | Accepted through its existing quorum after the first attachment refusal; the governed attachment has not yet been re-applied. |
| `toroidal_ion_charge_state_torque_density` | accepted, valid | No exact DD source simultaneously carries total, charge-state-resolved, and toroidal semantics. |
| `toroidal_line_integrated_impurity_ion_velocity` | drafted, valid | The drafted target has not earned accepted lifecycle. |
| `toroidal_neutral_state_momentum_diffusivity` | accepted, valid | Its candidate DD source still has multiple live targets and lacks an exact survivor adjudication. |
| `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | accepted, valid | The candidate describes a trapped non-Maxwellian source distribution transferring torque to a thermal recipient, not torque delivered to trapped thermal ions. |
| `x_direction_unit_vector_of_sensor` | accepted, valid | DD's unit-vector parent conflicts with metre-valued children while the direction-cosine identity correctly requires unit 1. |
| `z_direction_unit_vector_of_sensor` | accepted, valid | DD's unit-vector parent conflicts with metre-valued children while the direction-cosine identity correctly requires unit 1. |

This table deliberately preserves qualified outcomes. An accepted, valid name
can still be a provenance gap when its candidate source is held by a
last-producer or semantic guard. Conversely, a quarantined legacy spelling is
not delete authority; its named canonical target remains the governed route.

## Nonmutation proof

The production invocation sampled both required counters before any census
query and again after receipt lookup, cohort derivation, structural admission,
authority-file attribution, and partition assertions:

| Graph measure | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameChange` nodes | **7,780** | **7,780** | **0** |
| `PRODUCED_NAME` relationships | **5,770** | **5,770** | **0** |

Both counters were identical, and the invocation exited zero only after
asserting all of the following: receipt counts `2` and `19`; childful
classifier equality `9/9`; named-reason coverage `20/20`; partition arithmetic
`29 = 9 + 0 + 20`; and `unpartitioned = 0`.

## Evidence record

- Machine-readable production query and assertions:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T225117730640-barecensus/logs/bare-structural-census.json`
- Structural revival authority payload SHA-256:
  `0d8d6d1330fed4808e28d188fea07eb2dfe427a3d22923e0a9f5323f34366be0`
- Dual-bound authority file:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T204309912344-dualapply/dual-bound-source-target-authority.json`

No production graph mutation, provider call, review draw, or plan-state edit
was performed by this census node.
