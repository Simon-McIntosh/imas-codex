# Protection exposure census

## Verdict

**Both sides of the current protection predicate are wrong on the live graph.**
At the read-only measurement time, **2026-08-25T17:47:51.745037Z**,
`filter_protected` shielded **2,096** identities through
`origin = 'catalog_edit' OR name_stage = 'approved'`. Of those, **2,069**
carried no accepted evidence of a current human or catalog editorial action
and therefore should not be protected on the durable evidence available.

In the other direction, **315** identities carried a complete, terminal human
edit receipt but were not shielded. Those are the identities a pipeline write
can currently overwrite despite direct evidence that a human shaped the
current result. The exact 315-name manifest appears below.

The 1,091 null-origin identities do create real exposure, but not across the
whole cohort: **12/1,091** carry current human editorial evidence and are
unprotected; **1,079/1,091** carry none of the accepted evidence. No graph
property or relationship was written.

Machine-readable measurements are retained at:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T173907453065-n-protectionexposure/logs/protection-exposure-final.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T173907453065-n-protectionexposure/logs/current-editorial-authority.json`

## What counts as editorial action

The census accepted evidence only when it identified a human or catalog actor
and showed a terminal action on the current identity:

1. A **complete applied human edit receipt** on the `StandardName`:
   `edit_origin='human'`, `edit_status='applied'`, and non-null `edit_mode`,
   `edit_reason`, and `edit_requested_at`. This is direct current evidence:
   the schema says `edit_origin` identifies the proposer and `applied` means
   review accepted the steered candidate.
2. A **complete current catalog approval receipt**:
   `name_stage='approved'` together with `catalog_pr_number`, `catalog_pr_url`,
   `catalog_approved_at`, and `catalog_merge_commit_sha`.
3. A current approved identity with a linked, complete catalog promotion or
   override event: `catalog_promotion` with `content_edit` or
   `unchanged_ratification`, or `catalog_override` with `content_edit`.
   These operation/origin pairs are the explicit editorial outcomes written by
   the promotion boundary.

Only the first branch exists in the live graph: **342** identities carry a
complete applied human edit receipt. Of those, 27 are already shielded and
315 are not. The current graph carries **zero** approved names, zero complete
PR receipts, zero linked `catalog_promotion` outcomes, and zero linked
`catalog_override` outcomes.

A historical `StandardNameChange` whose initiator happens to be `human` was
not accepted by itself. The prior ledger validation proved that this field
records an operational initiator rather than the most recent editorial
authority; later agent edits can supersede it, and open, rejected, or exhausted
human proposals did not become the current accepted content. Such history is
corroboration only. The current terminal receipt above is the authority signal.

The following were deliberately rejected as editorial evidence:

- `origin='catalog_edit'` on 2,096 identities, because it is the predicate
  being audited and the repository's scoped guidance identifies this cohort as
  one bulk import of earlier pipeline output.
- `imported_at` on 2,104 identities and `catalog_commit_sha` on 2,053, because
  they prove a round-trip occurred, not that a human edited the identity.
- `source_types` containing `catalog` on 1,939 identities, because it records
  an input kind, not an editorial act.
- LLM generation, review, and cost evidence, because those prove automated
  pipeline participation rather than human or catalog authority.

## Schema sanity and zero controls

The same live census saw **4,658** `StandardName` candidates; all 4,658 carried
the schema identity `id` and all 4,658 carried `name_stage`. It saw **8,596**
`StandardNameChange` candidates; 8,596 carried `id`, 8,596 carried
`operation`, 7,800 carried `origin`, 8,488 carried `reason`, and 8,586 carried
`changed_at`. On the name side, 785 identities carried each of `edit_mode`,
`edit_reason`, `edit_origin`, `edit_status`, and `edit_requested_at`.

Every zero reported here is anchored to a schema-positive population and a
positive control:

| Reported zero | Population and schema sanity | Positive control |
|---|---|---|
| `name_stage='approved'`: **0** | 4,658/4,658 names carry `name_stage`; `approved` is declared in the LinkML `NameStage` enum | The same predicate finds 2,302 `accepted` identities |
| Any PR receipt field: **0**; complete PR receipts: **0** | The four exact properties are declared on `StandardName`; the same 4,658-name scan finds 2,104 `imported_at` and 2,053 `catalog_commit_sha` values | Import evidence fires while approval evidence does not |
| Linked catalog promotion/override outcomes: **0** | 8,596 change candidates; 8,596 operations and 7,800 origins populated | The same join finds 137 identities with complete linked human-edit change rows |
| Graph mutations: **0** | 4,658 identities before and after | The ordered state digest is identical before and after |

The ordered digest of `id`, `origin`, `name_stage`, and `status` was
`9edf3c90fb45c1da138645648f34a086ba5d24efffed20fbaefae0834351ea07`
before and after. The measurement used only `MATCH ... RETURN` queries.
This is a direct **no graph mutation** result.

## Over-protection

The current shield is entirely the legacy origin branch:

| Current protection result | Count |
|---|---:|
| `origin='catalog_edit'` | 2,096 |
| `name_stage='approved'` | 0 |
| union shielded by `filter_protected` | **2,096** |
| shielded with accepted current editorial evidence | 27 |
| shielded without accepted current editorial evidence | **2,069** |

The 2,069 unsupported shields span 1,354 accepted, two drafted, 115 exhausted,
69 reviewed, and 529 superseded identities. A representative positive control
for the protection predicate is `absorbed_power_of_beam_tracing_beam`: it is
`origin='catalog_edit'`, `name_stage='accepted'`, and `status='draft'`, so the
current shield fires; it has no applied human edit, catalog approval receipt,
or catalog editorial outcome. This controls that the census measures the real
code predicate, not an inferred approximation.

Under a fail-closed authority rule, absence of durable editorial evidence is
not permission to claim catalog ownership. These 2,069 identities are therefore
over-protected on the evidence the graph can actually produce.

## Under-protection

The **315** unprotected identities with accepted current editorial evidence
split as follows:

| Axis | Partition | Count |
|---|---|---:|
| `origin` | `pipeline` | 302 |
| `origin` | null | 12 |
| `origin` | `derived` | 1 |
| `name_stage` | `accepted` | 240 |
| `name_stage` | `exhausted` | 2 |
| `name_stage` | `superseded` | 73 |

`accumulated_deposited_energy_of_plasma_facing_component` is the editorial
positive control. It is unshielded (`origin='pipeline'`,
`name_stage='accepted'`) but carries a complete applied human rename receipt:
`edit_origin='human'`, `edit_status='applied'`, a timestamp, and a reason
explaining that `thermal` would assert a population distinction absent from
the DD description. It controls that the census can detect human shaping even
when the current protection predicate does not.

### Exact unprotected editorial manifest (315 identities)

The list below is lexicographically ordered and is the complete second set.

- `accumulated_deposited_energy_of_plasma_facing_component`
- `atomic_neutral_internal_state_power_density_due_to_collisions`
- `breakdown_initial_time`
- `co_passing_fast_ion_charge_state_power_density_due_to_collisions`
- `co_passing_fast_ion_charge_state_torque_density_due_to_collisions`
- `co_passing_thermal_ion_charge_state_power_density_due_to_collisions`
- `co_passing_thermal_ion_charge_state_torque_density_due_to_collisions`
- `counter_passing_fast_ion_charge_state_power_density_due_to_collisions`
- `counter_passing_fast_ion_charge_state_torque_density_due_to_collisions`
- `counter_passing_thermal_ion_charge_state_power_density_due_to_collisions`
- `counter_passing_thermal_ion_charge_state_torque_density_due_to_collisions`
- `cross_section_of_plasma_boundary`
- `derivative_with_respect_to_toroidal_flux_coordinate_of_area_of_flux_surface`
- `difference_of_total_plasma_heating_power_and_time_derivative_of_plasma_stored_energy`
- `difference_of_total_plasma_heating_power_and_time_derivative_of_stored_plasma_energy`
- `effective_neutral_internal_state_energy_diffusivity`
- `effective_neutral_internal_state_momentum_velocity_due_to_convection`
- `effective_particle_energy`
- `electron_volumetric_source_rate`
- `energy_flux_at_divertor_target`
- `energy_flux_at_first_wall`
- `fast_electron_absorbed_wave_power`
- `fast_ion_absorbed_wave_power`
- `fast_ion_charge_state_absorbed_wave_power`
- `fast_ion_charge_state_density`
- `fast_ion_charge_state_power`
- `fast_ion_charge_state_power_at_inside_flux_surface`
- `fast_ion_charge_state_power_density`
- `fast_ion_charge_state_power_due_to_collisions`
- `fast_ion_charge_state_pressure`
- `fast_neutral_internal_state_density`
- `fast_neutral_internal_state_number_density`
- `fast_neutral_internal_state_pressure`
- `first_local_tangential_coordinate_of_aperture`
- `first_local_tangential_coordinate_of_bragg_crystal`
- `first_local_tangential_coordinate_of_electron_cyclotron_launcher_mirror`
- `first_local_tangential_coordinate_of_filter`
- `first_local_tangential_coordinate_of_filter_window`
- `first_local_tangential_coordinate_of_neutral_beam_injector`
- `first_local_tangential_coordinate_of_neutron_detector`
- `first_local_tangential_coordinate_of_reflectometer_antenna`
- `first_local_tangential_coordinate_of_soft_xray_detector`
- `fluctuating_saturated_ion_current_density`
- `flux_surface_averaged_inverse_major_radius`
- `flux_surface_averaged_inverse_of_major_radius`
- `flux_surface_averaged_inverse_of_square_of_magnetic_field_magnitude`
- `flux_surface_averaged_inverse_square_magnetic_field_magnitude`
- `flux_surface_averaged_magnetic_field`
- `flux_surface_averaged_magnetic_field_magnitude`
- `flux_surface_averaged_metric`
- `flux_surface_averaged_metric_tensor`
- `flux_surface_averaged_ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_major_radius`
- `flux_surface_averaged_ratio_of_square_toroidal_flux_coordinate_gradient_magnitude_to_square_major_radius`
- `flux_surface_averaged_square_of_magnetic_field_magnitude`
- `flux_surface_averaged_square_of_toroidal_flux_coordinate_gradient_magnitude`
- `flux_surface_averaged_square_toroidal_flux_coordinate_gradient_magnitude`
- `flux_surface_averaged_toroidal_flux_coordinate_gradient_magnitude`
- `flux_surface_normal_flux_surface_averaged_contravariant_metric_tensor`
- `flux_surface_normal_flux_surface_averaged_metric`
- `flux_surface_normal_surface_integrated_net_plasma_energy_flux_at_plasma_boundary`
- `incident_neutral_particle_flux_at_wall_due_to_recombination`
- `incident_neutral_state_particle_flux_at_wall_due_to_recombination`
- `ion_charge_state_absorbed_power_of_beam_tracing_beam`
- `ion_charge_state_convection_velocity`
- `ion_charge_state_diffusion_coefficient`
- `ion_charge_state_energy_convection_velocity`
- `ion_charge_state_energy_diffusion_coefficient`
- `ion_charge_state_energy_flux`
- `ion_charge_state_energy_flux_at_wall_due_to_recombination`
- `ion_charge_state_kinetic_energy_flux_at_wall_due_to_surface_emission`
- `ion_charge_state_momentum_convection_velocity`
- `ion_charge_state_momentum_damping_rate`
- `ion_charge_state_momentum_diffusion_coefficient`
- `ion_charge_state_momentum_flux_limiter_coefficient`
- `ion_charge_state_momentum_source`
- `ion_charge_state_particle_flux_at_wall`
- `ion_charge_state_power_density_due_to_collisions`
- `ion_charge_state_power_due_to_collisions`
- `ion_charge_state_pressure`
- `ion_charge_state_rotation_frequency`
- `ion_charge_state_torque_density`
- `ion_charge_state_torque_due_to_collisions`
- `ion_charge_state_upper_bound_charge`
- `ion_charge_state_velocity`
- `ion_charge_state_velocity_due_to_diamagnetic_drift`
- `ion_charge_state_velocity_due_to_e_cross_b_drift`
- `ion_momentum_source`
- `ion_state_maximum_charge_number`
- `ion_state_momentum_source`
- `length_of_poloidal_magnetic_field_probe`
- `line_integrated_electron_number_density`
- `maximum_of_energy_flux_along_limiter`
- `maximum_of_energy_flux_at_divertor_target`
- `maximum_of_energy_flux_at_first_wall`
- `maximum_of_energy_flux_at_limiter`
- `momentum_source_due_to_diamagnetic_drift`
- `neutral_internal_state_density`
- `neutral_internal_state_diffusion_coefficient`
- `neutral_internal_state_energy_convection_velocity`
- `neutral_internal_state_energy_diffusion_coefficient`
- `neutral_internal_state_energy_diffusivity`
- `neutral_internal_state_energy_flux`
- `neutral_internal_state_momentum_convection_velocity`
- `neutral_internal_state_momentum_diffusion_coefficient`
- `neutral_internal_state_momentum_diffusivity`
- `neutral_internal_state_momentum_flux_limiter_coefficient`
- `neutral_internal_state_momentum_flux_limiter_coefficient_over_edge_region`
- `neutral_internal_state_momentum_source`
- `neutral_internal_state_particle_flux`
- `neutral_internal_state_power_density`
- `neutral_internal_state_pressure`
- `neutral_internal_state_temperature`
- `neutral_internal_state_velocity`
- `neutral_internal_state_velocity_due_to_diamagnetic_drift`
- `neutral_internal_state_velocity_due_to_e_cross_b_drift`
- `neutral_momentum_source`
- `neutral_state_momentum_source`
- `normal_extent_of_magnetic_field_probe`
- `normalized_internal_inductance`
- `normalized_momentum_flux_due_to_perturbed_parallel_vector_potential`
- `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift`
- `parallel_fast_ion_charge_state_pressure`
- `parallel_fast_neutral_internal_state_pressure`
- `parallel_ion_charge_state_convection_velocity`
- `parallel_ion_charge_state_energy_diffusivity`
- `parallel_ion_charge_state_energy_flux`
- `parallel_ion_charge_state_momentum_convection_velocity`
- `parallel_ion_charge_state_momentum_damping_rate`
- `parallel_ion_charge_state_momentum_source`
- `parallel_ion_charge_state_velocity`
- `parallel_ion_charge_state_velocity_due_to_diamagnetic_drift`
- `parallel_ion_momentum_source`
- `parallel_ion_state_momentum_source`
- `parallel_net_ion_momentum_source`
- `parallel_neutral_internal_state_diffusivity`
- `parallel_neutral_internal_state_energy_convection_velocity`
- `parallel_neutral_internal_state_energy_diffusivity`
- `parallel_neutral_internal_state_energy_flux`
- `parallel_neutral_internal_state_momentum_convection_velocity`
- `parallel_neutral_internal_state_momentum_diffusivity`
- `parallel_neutral_internal_state_momentum_flux`
- `parallel_neutral_internal_state_momentum_flux_limiter_coefficient`
- `parallel_neutral_internal_state_particle_flux`
- `parallel_neutral_internal_state_velocity_due_to_diamagnetic_drift`
- `parallel_neutral_internal_state_velocity_due_to_e_cross_b_drift`
- `parallel_neutral_momentum_source`
- `parallel_normalized_gyrocenter_momentum_flux_of_gyrokinetic_eigenmode_due_to_perturbed_parallel_magnetic_field`
- `parallel_normalized_gyrocenter_momentum_flux_of_gyrokinetic_eigenmode_due_to_perturbed_parallel_vector_potential`
- `parallel_normalized_momentum_flux_due_to_perturbed_parallel_vector_potential`
- `particle_source_rate_due_to_distribution_function_driven`
- `per_toroidal_mode_fast_electron_absorbed_wave_power`
- `per_toroidal_mode_fast_ion_absorbed_wave_power`
- `per_toroidal_mode_fast_ion_charge_state_power`
- `per_toroidal_mode_fast_ion_charge_state_power_density`
- `per_toroidal_mode_thermal_electron_absorbed_wave_power`
- `per_toroidal_mode_thermal_ion_absorbed_wave_power`
- `per_toroidal_mode_thermal_ion_charge_state_power`
- `per_toroidal_mode_thermal_ion_charge_state_power_density`
- `perpendicular_fast_ion_charge_state_pressure`
- `perpendicular_fast_neutral_internal_state_pressure`
- `perpendicular_ion_charge_state_velocity_due_to_diamagnetic_drift`
- `perpendicular_normalized_momentum_flux_due_to_perturbed_parallel_magnetic_field`
- `perpendicular_suprathermal_neutral_internal_state_pressure`
- `photon_radiance_at_spectral_line`
- `poloidal_ion_charge_state_convection_velocity`
- `poloidal_ion_charge_state_diffusion_coefficient`
- `poloidal_ion_charge_state_diffusivity`
- `poloidal_ion_charge_state_energy_convection_velocity`
- `poloidal_ion_charge_state_energy_diffusivity`
- `poloidal_ion_charge_state_momentum_convection_velocity`
- `poloidal_ion_charge_state_momentum_damping_rate`
- `poloidal_ion_charge_state_momentum_flux_limiter_coefficient`
- `poloidal_ion_charge_state_momentum_source`
- `poloidal_ion_charge_state_velocity`
- `poloidal_ion_charge_state_velocity_due_to_diamagnetic_drift`
- `poloidal_ion_charge_state_velocity_due_to_e_cross_b_drift`
- `poloidal_ion_state_momentum_source`
- `poloidal_neutral_internal_state_diffusion_coefficient`
- `poloidal_neutral_internal_state_energy_convection_velocity`
- `poloidal_neutral_internal_state_energy_diffusion_coefficient`
- `poloidal_neutral_internal_state_energy_diffusivity`
- `poloidal_neutral_internal_state_energy_flux`
- `poloidal_neutral_internal_state_linear_momentum_flux`
- `poloidal_neutral_internal_state_momentum_convection_velocity`
- `poloidal_neutral_internal_state_momentum_diffusion_coefficient`
- `poloidal_neutral_internal_state_momentum_flux`
- `poloidal_neutral_internal_state_momentum_flux_limiter_coefficient`
- `poloidal_neutral_internal_state_particle_flux`
- `poloidal_neutral_internal_state_velocity`
- `poloidal_neutral_internal_state_velocity_due_to_e_cross_b_drift`
- `poloidal_neutral_momentum_source`
- `poloidal_plane_cross_sectional_area_of_plasma_boundary`
- `radial_contravariant_flux_surface_averaged_metric`
- `radial_effective_neutral_internal_state_velocity_due_to_convection`
- `radial_flux_surface_averaged_metric`
- `radial_ion_charge_state_diffusivity`
- `radial_ion_charge_state_energy_diffusivity`
- `radial_ion_charge_state_momentum_convection_velocity`
- `radial_ion_charge_state_momentum_damping_rate`
- `radial_ion_charge_state_momentum_diffusion_coefficient`
- `radial_ion_charge_state_velocity`
- `radial_ion_momentum_flux`
- `radial_momentum_flux`
- `radial_neutral_internal_state_convection_velocity`
- `radial_neutral_internal_state_diffusion_coefficient`
- `radial_neutral_internal_state_energy_convection_velocity`
- `radial_neutral_internal_state_energy_diffusion_coefficient`
- `radial_neutral_internal_state_momentum_flux`
- `radial_neutral_internal_state_particle_flux`
- `radial_neutral_internal_state_velocity_due_to_e_cross_b_drift`
- `ratio_of_electron_temperature_at_magnetic_axis_to_volume_averaged_electron_temperature`
- `ratio_of_line_averaged_electron_density_to_greenwald_density`
- `ratio_of_parallel_ion_velocity_to_magnetic_field_magnitude`
- `ratio_of_particle_count_to_particle_simulated_count`
- `ratio_of_particle_temperature_to_particle_reference_temperature`
- `ratio_of_plasma_upper_bound_current_to_plasma_initial_current_due_to_disruption`
- `runaway_electron_current_decay_time_due_to_disruption`
- `runaway_electron_decay_time_due_to_disruption`
- `safety_factor`
- `saturated_ion_current`
- `saturated_ion_current_density`
- `second_local_tangential_coordinate_of_aperture`
- `second_local_tangential_coordinate_of_bragg_crystal`
- `second_local_tangential_coordinate_of_electron_cyclotron_launcher_mirror`
- `second_local_tangential_coordinate_of_filter`
- `second_local_tangential_coordinate_of_filter_window`
- `second_local_tangential_coordinate_of_hard_xray_detector`
- `second_local_tangential_coordinate_of_neutral_beam_injector`
- `second_local_tangential_coordinate_of_neutron_detector`
- `second_local_tangential_coordinate_of_optical_element`
- `second_local_tangential_coordinate_of_reflectometer_antenna`
- `second_local_tangential_coordinate_of_soft_xray_detector`
- `stray_breakdown_magnetic_field_magnitude`
- `thermal_electron_absorbed_wave_power`
- `thermal_electron_energy`
- `thermal_ion_absorbed_wave_power`
- `thermal_ion_charge_state_density`
- `thermal_ion_charge_state_number_density`
- `thermal_ion_charge_state_power`
- `thermal_ion_charge_state_power_at_inside_flux_surface`
- `thermal_ion_charge_state_power_density`
- `thermal_ion_charge_state_power_due_to_collisions`
- `thermal_ion_charge_state_torque_due_to_collisions`
- `thermal_neutral_internal_state_density`
- `thermal_plasma_field_aligned_power_over_halo_region_due_to_conductive_losses`
- `toroidal_average_center_of_mass_velocity_at_along_line_of_sight`
- `toroidal_co_passing_fast_ion_charge_state_torque_density_due_to_collisions`
- `toroidal_co_passing_thermal_ion_charge_state_torque_density_due_to_collisions`
- `toroidal_coordinate_at_beam_tracing_point`
- `toroidal_coordinate_at_pellet_path_point`
- `toroidal_coordinate_at_shattering_position`
- `toroidal_counter_passing_fast_ion_charge_state_torque_density_due_to_collisions`
- `toroidal_counter_passing_thermal_ion_charge_state_torque_density_due_to_collisions`
- `toroidal_fast_ion_charge_state_torque_density_due_to_collisions`
- `toroidal_fast_ion_charge_state_torque_due_to_collisions`
- `toroidal_flux_coordinate_at_internal_transport_barrier`
- `toroidal_flux_coordinate_at_pedestal_top`
- `toroidal_flux_coordinate_at_plasma_boundary`
- `toroidal_flux_coordinate_of_magnetic_axis`
- `toroidal_flux_surface_averaged_metric`
- `toroidal_ion_charge_state_momentum_damping_rate`
- `toroidal_ion_charge_state_momentum_flux_limiter_coefficient`
- `toroidal_ion_charge_state_momentum_source`
- `toroidal_ion_charge_state_rotation_frequency`
- `toroidal_ion_charge_state_torque_density`
- `toroidal_ion_charge_state_velocity_due_to_diamagnetic_drift`
- `toroidal_ion_charge_state_velocity_due_to_e_cross_b_drift`
- `toroidal_ion_momentum_source`
- `toroidal_ion_state_momentum_source`
- `toroidal_line_averaged_plasma_velocity`
- `toroidal_line_integrated_impurity_ion_velocity`
- `toroidal_neutral_internal_state_momentum_convection_velocity`
- `toroidal_neutral_internal_state_momentum_diffusion_coefficient`
- `toroidal_neutral_internal_state_momentum_flux`
- `toroidal_neutral_internal_state_momentum_flux_limiter_coefficient`
- `toroidal_neutral_internal_state_torque_density`
- `toroidal_neutral_internal_state_velocity_due_to_e_cross_b_drift`
- `toroidal_neutral_momentum_source`
- `toroidal_offset_at_measurement_position`
- `toroidal_thermal_ion_charge_state_torque_density_due_to_collisions`
- `toroidal_thermal_ion_charge_state_torque_due_to_collisions`
- `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions`
- `total_ion_charge_state_density`
- `total_ion_charge_state_power_due_to_collisions`
- `total_ion_charge_state_torque_due_to_collisions`
- `total_ion_particle_volumetric_source_rate`
- `total_lost_plasma_power_at_plasma_boundary`
- `total_neutral_internal_state_density`
- `total_neutral_internal_state_pressure`
- `total_volumetric_ion_particle_source_rate`
- `trapped_fast_ion_charge_state_power_density_due_to_collisions`
- `trapped_thermal_ion_charge_state_power_density_due_to_collisions`
- `trapped_thermal_ion_charge_state_torque_density_due_to_collisions`
- `upper_bound_ion_charge_number`
- `vertical_bulk_neutral_internal_state_velocity`
- `vertical_image_up_unit_vector_of_camera`
- `vertical_ion_charge_state_momentum_convection_velocity`
- `vertical_ion_charge_state_momentum_diffusivity`
- `vertical_ion_charge_state_velocity`
- `vertical_ion_charge_state_velocity_due_to_e_cross_b_drift`
- `vertical_neutral_internal_state_momentum_convection_velocity`
- `vertical_neutral_internal_state_momentum_diffusivity`
- `vertical_neutral_internal_state_momentum_flux`
- `vertical_neutral_internal_state_velocity`
- `vertical_neutral_internal_state_velocity_due_to_e_cross_b_drift`
- `vertical_neutral_species_internal_state_velocity_due_to_e_cross_b_drift`
- `volume_averaged_ion_charge_state_average_charge_number`
- `volume_averaged_time_derivative_of_electron_density`
- `x_image_up_unit_vector_of_camera`
- `z_coordinate_of_divertor_target`
- `z_coordinate_of_ferritic_element_centroid`
- `z_coordinate_of_sensor_attachment_point`
- `z_direction_unit_vector_of_neutron_detector`
- `z_image_up_unit_vector`
- `z_image_up_unit_vector_of_camera`

## Null-origin exposure

The null-origin cohort is not uniformly safe. Exactly **12** of its 1,091
members carry complete applied human edit receipts while remaining outside
`filter_protected`. Four are accepted and eight are superseded:

- `flux_surface_averaged_magnetic_field`
- `flux_surface_averaged_metric`
- `flux_surface_averaged_metric_tensor`
- `flux_surface_averaged_square_of_magnetic_field_magnitude`
- `flux_surface_normal_flux_surface_averaged_metric`
- `length_of_poloidal_magnetic_field_probe`
- `radial_flux_surface_averaged_metric`
- `radial_ion_momentum_flux`
- `radial_momentum_flux`
- `toroidal_line_integrated_impurity_ion_velocity`
- `z_direction_unit_vector_of_neutron_detector`
- `z_image_up_unit_vector`

These are real exposure, not a property-name zero: all 12 have the complete
five-field receipt, drawn from the 785 names on which every edit receipt field
is populated. For example, `radial_momentum_flux` has an applied human hint
stating that a magnetic vector potential and a momentum flux are unrelated
quantities; a later unrestricted pipeline write can currently overwrite the
result despite that accepted human intervention.

The remaining **1,079** null-origin identities carry no accepted current
editorial receipt. They do not create protection exposure under the evidence
rule used here. The correct conclusion is therefore neither “all 1,091 are
unsafe” nor “null origin creates no exposure”: the exact exposed subset is 12.

## Consequence

The current guard conflates a legacy import label with authority and ignores
the direct human-edit receipt already stored on the identity. A safe repair
must derive protection from current terminal editorial receipts, preserve the
315-name under-protection manifest until the guard changes, and avoid granting
authority from import markers or historical change initiators alone. This node
is read-only and does not implement that repair.
