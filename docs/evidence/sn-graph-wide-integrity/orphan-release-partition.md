# Unsourced standard-name release-provenance partition

Measured against the live graph on 2026-08-21. The census reused the exact predicate from `_UNSOURCED_WITHOUT_LIVE_CHILD_QUERY` in `tests/graph/test_sn_integrity_ratchets.py`: a live name is selected only when no `StandardNameSource` has a `PRODUCED_NAME` edge to it and no nonterminal `StandardName` has a `HAS_PARENT` edge to it.

## Result

- Measured identities: **36**.
- Delete cohort: **36**.
- Supersede cohort: **0**.
- Accounting: **36 + 0 = 36**, with no identity unassigned or multiply assigned.
- Every row has `producing_source_count=0` and `live_child_count=0`.
- Every row lacks catalog release provenance: none has `name_stage=approved`, and `catalog_pr_number`, `catalog_pr_url`, `catalog_approved_at`, and `catalog_merge_commit_sha` are null on every row.
- The ordered identity-list SHA-256 (identities joined with a newline, without a trailing newline) is `295cda8bb7d9e8393723640f49803bf5d57f5dbe165660582bae4f7c4cd336dc`.

The release test is deliberately conservative: either `name_stage=approved` or any non-null merged-PR metadata field would count as catalog release provenance and place the identity in the supersede cohort. A canonical catalog approval has the approved stage and complete PR number, URL, approval time, and merge-commit metadata. `origin=catalog_edit` is editorial origin, not publication evidence; the three such rows below have no merged-PR metadata and therefore remain in the delete cohort.

This record proves the first three delete conditions per identity and is the exact enumeration input for the fourth. It is **not** a signed apply manifest and authorizes no mutation. A downstream delete operator must still sign this exact 36-row membership, compare the applying row count with 36, and refuse the whole cohort on any drift or excess.

Lifecycle distribution is 27 accepted, 4 drafted, 4 reviewed, and 1 pending. Origin distribution is 14 pipeline, 3 catalog edit, and 19 null.

## Per-identity evidence and assignment

In the catalog-evidence column, “PR metadata all null” means the graph returned null for `catalog_pr_number`, `catalog_pr_url`, `catalog_approved_at`, and `catalog_merge_commit_sha`.

| Identity | Lifecycle; origin | Producing sources | Live children | Catalog release provenance and graph evidence | Cohort |
|---|---|---:|---:|---|---|
| `capacitance_of_ion_cyclotron_heating_antenna` | accepted; catalog_edit | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `cross_section_of_flux_surface` | pending; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `fast_ion_charge_state_power_at_inside_flux_surface` | accepted; pipeline | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `flux_surface_averaged_toroidal_flux_coordinate_gradient_magnitude` | accepted; pipeline | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `forward_wave_phase_of_ion_cyclotron_heating_antenna` | accepted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `impurity_ion_photon_radiance_of_spectral_line_due_to_charge_exchange` | accepted; pipeline | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `line_integrated_electron_density` | drafted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `magnetic_field_at_pedestal_top_low_field_side_magnitude` | drafted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `minimum_magnetic_field_magnitude` | accepted; catalog_edit | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `minimum_of_safety_factor` | reviewed; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `neutral_state_power_density` | reviewed; catalog_edit | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `neutron_flux_due_to_fusion` | accepted; pipeline | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `parallel_current_density_due_to_ohmic_current_drive` | accepted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | accepted; pipeline | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `parallel_mach_number` | accepted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `parallel_neutral_momentum_diffusion_coefficient` | accepted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `poloidal_neutral_internal_state_convection_velocity` | accepted; pipeline | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `poloidal_neutral_internal_state_momentum_convected_velocity` | reviewed; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `poloidal_neutral_state_particle_convection_velocity` | accepted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `poloidal_straight_field_line_angle` | drafted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `radial_effective_electron_diffusivity` | accepted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `radial_effective_ion_diffusivity` | accepted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `radial_effective_neutral_diffusivity` | accepted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `radial_thermal_ion_charge_state_energy_diffusion_coefficient` | accepted; pipeline | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `tendency_of_total_thermal_plasma_internal_energy` | accepted; pipeline | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | reviewed; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `toroidal_ion_charge_state_torque_density` | accepted; pipeline | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `toroidal_line_integrated_impurity_ion_velocity` | drafted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `toroidal_neutral_state_momentum_diffusivity` | accepted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `toroidal_thermal_ion_charge_state_torque_due_to_collisions` | accepted; pipeline | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `toroidal_thermal_ion_torque_density_due_to_thermalization` | accepted; pipeline | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `toroidal_trapped_fast_ion_charge_state_torque_density_due_to_collisions` | accepted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | accepted; pipeline | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `variation_of_length_of_interferometer_beam` | accepted; pipeline | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `x_direction_unit_vector_of_sensor` | accepted; null | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |
| `z_direction_unit_vector_of_sensor` | accepted; pipeline | 0 | 0 | No — `name_stage != approved`; PR metadata all null | delete |

## Measurement method

One read-only aggregate Cypher statement applied the ratchet predicate and returned, for each selected identity, total incoming producing-source count, nonterminal child count, stage, origin, and all four catalog-approval metadata fields. The complete retained query output is `/tmp/orphan-release-partition-query.log`, SHA-256 `1fd541fe0407354753327ba1b7d1b1991c3ced5732ad6328fb7f16899cc4f1b4`.

The query made no graph write and invoked no provider or pipeline operation. The resulting checks were all true: row count equals 36; every producing-source count equals zero; every live-child count equals zero; every catalog-approval field is null; delete plus supersede equals 36; and every identity has exactly one assignment.
