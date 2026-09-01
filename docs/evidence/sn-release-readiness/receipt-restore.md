NEEDS-HELP: the live 42-row receipt-proven cohort cannot be restored through the governed signed-manifest operators because 41 targets are not accepted live names and one receipted source is already bound to the signed survivor selected by an earlier dual-authority retirement.

# Receipt-proven source restoration

## Outcome

**BLOCKED — 0 of 42 identities restored; 42 of 42 remain source-lost; 0 graph
rows changed; 0 identity strings changed; attributable spend USD 0.000000.**

The population was re-measured against the live `codex` graph on 2026-09-01;
the plan's stored count was not used as the result. The aimed query selected
chain-cap `StandardName` nodes with no incoming
`StandardNameSource-[:PRODUCED_NAME]` edge and no scalar
`StandardNameSource.produced_sn_id`, then joined their durable source-migration,
detach, terminal-recovery, dual-authority-retirement, and semantic-binding-repair
receipts. It returned the same **42 receipt-proven source-lost identities**, each
with a before source count of 0. Because the governed operator refused the
first authority-conflicted row and the cohort cannot meet its all-row objective,
no apply was attempted; the after source count is also 0 for every row.

The live lifecycle distribution independently proves that the ordinary governed
attachment/migration programs cannot admit this cohort:

| Live target state | Rows |
|---|---:|
| exhausted, valid | 27 |
| superseded, valid, status superseded | 11 |
| exhausted, quarantined | 2 |
| superseded, valid, status draft | 1 |
| accepted, valid | 1 |
| **Total** | **42** |

Both signed ordinary-source programs require the target to be accepted, valid,
and not deprecated or superseded. Therefore 41 rows are closed by lifecycle
authority before source-specific guards are considered.

## Deciding signed preview

The exact receipted binding for `radial_plasma_momentum_source` is:

| Identity | Receipted exact source | Receipt | Before | Current source owner | DD exists | Signed preview | After |
|---|---|---|---:|---|---|---|---:|
| `radial_plasma_momentum_source` | `dd:core_sources/source/profiles_1d/ion/momentum/radial` | `sn-change:dual-authority-retirement:28736a30fa250c58a07a30f5ad986703818b1cd8b59174a57ee4501e1a866b17:0e003f441b06773ae333` | 0 | `radial_ion_momentum_source` | yes | **REFUSED**: `signed ordinary source lifecycle does not match migration authority` | 0 |

The receipt is not an accidental detach: it is a signed
`retire_signed_dual_authority_target` event with manifest digest
`28736a30fa250c58a07a30f5ad986703818b1cd8b59174a57ee4501e1a866b17`,
source-authority digest
`c227e70ec5cd940577ca778ce5ec63e4df3a63bf68c3e845eba92d0a4b9a0efb`,
and retirement-authority digest
`4bac6110486390e95c1cab9620c4723df96fe6f2190b85e6496464c77fbba873`.
The current DD node exists, and its source is bound by both scalar and
`PRODUCED_NAME` edge to `radial_ion_momentum_source`. Restoring it to the
superseded historical target would reverse that signed adjudication and remove
the current survivor's authority. The canonical `ordinary-source-migration`
preview was built with the repair-authority builder and passed to
`apply_signed_manifest` without `apply=True`; it returned:

```text
outcome=refused
authority_rows=1
admitted=0
refused=1
would_change=0
changed=0
refusal=signed ordinary source lifecycle does not match migration authority
manifest_sha256=2efde6a02d57d1f0b55a68ad2f17d1a0d7fccecf053751f97ada6647dcbdf77a
```

This is a hard authority contradiction, not conservative scheduling: the node
goal says to restore the receipt binding, while the later signed receipt and
the current governed operator require that exact binding to remain retired.

## Live 42-identity census

Every identity below had source count **0 before / 0 after**. No round-trip was
run because no row was restored; the required diagnostic is explicitly scoped
to restored rows. Consequently round-trip divergence could neither veto nor
approve any restore, and local-model attributable cost remained exactly
**USD 0.000000**.

```text
poloidal_cross_sectional_area_of_plasma_boundary
radial_plasma_momentum_source
parallel_neutral_state_convection_velocity
inverse_of_spectral_surface_curvature_of_optical_element
ratio_of_neutral_species_gas_count_to_total_gas_count
poloidal_parity_of_gyrokinetic_eigenmode
net_coefficient_due_to_neoclassical_tearing_mode
root_mean_square_of_fluctuating_floating_electrostatic_potential
particle_probability
toroidal_net_plasma_torque_of_neoclassical_tearing_mode
total_launched_wave_power_of_electron_cyclotron_launcher
poloidal_accumulated_magnetic_flux_due_to_resistive_dissipation
flux_surface_normal_neutral_energy_diffusion_coefficient
surface_thickness_of_cryostat
first_local_tangential_back_surface_radius_of_optical_element
ion_temperature_at_outboard_midplane_separatrix
normal_distance_of_antenna_strap
root_mean_square_of_spectral_width_of_spectrometer_channel
vertical_coordinate_of_plasma_filament
net_plasma_power_density
normal_width_of_plasma_filament
radial_offset_of_lower_hybrid_antenna
inverse_of_tangential_curvature_of_optical_element
plasma_electrostatic_potential_at_wall
plasma_electrostatic_potential_at_outboard_midplane
net_forward_power_of_wave_beam
total_launched_power_due_to_ion_cyclotron_heating
flux_surface_normal_momentum_convection_velocity
neutral_species_kinetic_energy_flux_at_wall_due_to_surface_emission
deposited_power_at_divertor_target
absorbed_coolant_power_of_plant_component_port
total_particle_flux_at_divertor_target_due_to_recycling
molecular_gas_count_due_to_pellet_injection
tendency_of_runaway_electron_density
non_axisymmetric_current_of_conductor
front_surface_area_of_langmuir_probe
energy_flux_at_control_surface
wave_critical_ordinary_mode_frequency
wave_magnetic_field_amplitude
total_incident_thermal_power
power_over_scrape_off_layer_due_to_radiation
lithium_volume_of_breeder_blanket
```

## Request for adjudication

tried: re-measured the receipt-proven population live, verified all 42 remain
fully unbound, measured their lifecycle states, resolved the exact radial DD
source and receipt, and ran the canonical signed ordinary-source-migration
preview; it refused with zero admitted and zero changed.

options: (1) amend the restoration cohort to exclude bindings superseded by
later signed authority and restore only rows whose targets are currently live;
(2) explicitly revoke the later retirement authority and authorize lifecycle
reopening plus source migration back from the current survivors; or (3) define
a new governed historical-provenance record that preserves the receipt binding
without making terminal identities current source owners.

leaning: option 3. The receipts prove historical provenance, but recreating
live bindings to exhausted/superseded identities conflates provenance with
current semantic authority and, for the radial row, directly undoes a later
signed adjudication.

cost-if-wrong: option 1 leaves some historical identities ungrounded; option 2
requires re-adjudicating current survivors and could recreate dual bindings or
move DD sources away from accepted names; option 3 requires a schema/operator
change and a new receipt-aware reporting path before this node can be rerun.

