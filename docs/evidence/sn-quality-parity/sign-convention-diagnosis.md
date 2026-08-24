# Conditional sign-convention diagnosis

Snapshot: `2026-08-24T20:56:56Z`, worktree HEAD
`79c31525a8caace334a84badf06e7a44f70136da`, live `codex` graph. All graph
operations in this diagnosis were read-only and no model was called.

## Result

The 281 failures are real, but they are not one homogeneous documentation
cohort. The exact partition is:

| Root cause | Count | Correct transformation authority | Cheapest faithful repair |
|---|---:|---|---|
| DD-backed quantity is genuinely invariant; documentation generation nevertheless required a sign paragraph | 199 | `one_like` is justified by 540 authoritative DD paths whose transformation property is null, which the holdout authority contract treats as an explicit non-sensitive declaration | Deterministic, zero LLM: remove the one final sign-convention paragraph |
| Structural quantity is genuinely invariant; documentation generation nevertheless required a sign paragraph | 28 | the current derived-parent rule sees exactly one non-null eligible child class, `one_like` | Deterministic, zero LLM: remove the one final sign-convention paragraph |
| Stale `one_like` survives even though structural authority now recomputes no unique class | 35 | null | Deterministic, zero LLM: clear `cocos_transformation_type`, clear its scalar COCOS value and `HAS_COCOS` edge, then remove the unsupported final sign paragraph |
| `one_like` has no DD binding and no eligible structural-child authority | 18 | null until authority is restored | Deterministic, zero LLM: fail closed by clearing the ungrounded metadata and COCOS edge, then remove the unsupported final sign paragraph; source-edge recovery is a separate provenance repair |
| `magnetic_field` retains `one_like` although its unique eligible non-null child class is `b0_like` | 1 | `b0_like` | Metadata is a deterministic edit, but the document requires one-item regeneration and ordinary docs quorum because its existing paragraph was generated from the generic `one_like` fallback rather than the `b0_like` physics guidance |

The totals are exact: **227 genuinely invariant documentation defects + 54
metadata defects = 281 identities**. Of the 54 metadata defects, 53 must become
null and one must become `b0_like`.

Every failing document contains exactly one paragraph containing “sign
convention”, and in all 281 cases it is the final paragraph. Every failing node
also currently has exactly one `HAS_COCOS` edge, to COCOS 17. That uniform shape
is why 280 repairs can be exact text/property edits rather than regeneration.

## Root cause

The primary cause is an internal contradiction in the generation prompt. The
system prompt correctly says to omit a sign convention for an invariant
quantity (`imas_codex/llm/prompts/sn/generate_docs_system.md:72-86`). The dynamic
user prompt then treats **any** truthy `item.cocos_label` as sensitive and says
the model **MUST** emit a sign paragraph
(`imas_codex/llm/prompts/sn/generate_docs_user.md:140-151`). `one_like` is truthy,
has no label-specific template, and therefore receives the generic fallback
“this quantity transforms” text
(`imas_codex/llm/config/cocos_sign_guidance.yaml:100-102`). The more specific
per-item instruction wins, producing exactly the final paragraphs the new gate
rejects. The prevention fix is to omit COCOS guidance and the mandatory paragraph
when the label is `one_like`.

The secondary cause is nullable metadata written with preservation semantics.
The structural authority computes a class only when eligible children expose
exactly one non-null class (`graph_ops.py:3780-3782`), but persistence uses
`coalesce(row.cocos_transformation_type, parent.cocos_transformation_type)`
(`graph_ops.py:4080-4082`). A newly correct null therefore cannot clear a stale
`one_like`. This explains the 35 structural-null rows and leaves the same stale
value on `magnetic_field`, where the recomputed value is `b0_like`. The 18
unbacked rows have neither a DD authority edge nor an eligible structural child,
so retaining a non-null class is absence-as-permission; null is the only
faithful fail-closed value.

## Live measurement

Before accepting any zero, the candidate and property coverage query was:

```cypher
MATCH (sn:StandardName)
WHERE sn.docs_stage = 'accepted'
RETURN count(sn) AS candidates,
       count(sn.id) AS with_id,
       count(sn.documentation) AS with_documentation,
       count(sn.cocos_transformation_type) AS with_transformation_type
```

It returned 2,727 candidates, 2,727 ids, 2,727 documents and 381 transformation
properties. The documents were selected with:

```cypher
MATCH (sn:StandardName)
WHERE sn.docs_stage = 'accepted'
RETURN sn.id AS id,
       sn.documentation AS documentation,
       sn.cocos_transformation_type AS sn_type
ORDER BY sn.id
```

Each row was evaluated through the shipped gate call, not through a text-only
Cypher surrogate:

```python
score_documentation(
    documentation,
    physics_context=DocumentationPhysicsContext(
        cocos_transformation_type=sn_type,
    ),
).gate_vector["sign_convention"]
```

This reproduced **281 fail**, all with the exact reason
`COCOS-invariant quantity states a sign convention`; the other accepted rows
were 100 pass and 2,346 `not_evaluable`, matching the prior closure census. The
gate implementation is `docs_gates.py:429-480`.

For the failing identity list, live graph authority was projected with both the
canonical `StandardNameSource` route and the DD-side materialized route:

```cypher
UNWIND $ids AS target_id
MATCH (sn:StandardName {id: target_id})
CALL (sn) {
  OPTIONAL MATCH (dd:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
  RETURN collect(DISTINCT {
    id: dd.id,
    type: dd.cocos_transformation_type,
    source: dd.cocos_label_source,
    unit: dd.unit
  }) AS direct_dd
}
CALL (sn) {
  OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
  OPTIONAL MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)
  RETURN collect(DISTINCT {
    source_id: src.id,
    source_type: src.source_type,
    produced_sn_id: src.produced_sn_id,
    dd_id: dd.id,
    dd_type: dd.cocos_transformation_type,
    dd_type_source: dd.cocos_label_source,
    dd_unit: dd.unit
  }) AS provenance
}
RETURN sn.id AS id, direct_dd, provenance
ORDER BY sn.id
```

That query establishes the 199 DD-backed identities and 540 unique DD paths.
All 540 path transformation values are null; this is an authoritative
non-sensitive declaration under the existing holdout contract
(`benchmark_reference.py:344-352`), not a missing-value failure.

Rows without DD authority were resolved against structural authority with:

```cypher
UNWIND $ids AS target_id
MATCH (parent:StandardName {id: target_id})
OPTIONAL MATCH (child:StandardName)-[rel:HAS_PARENT]->(parent)
RETURN parent.id AS id,
       collect(DISTINCT {
         child_id: child.id,
         child_type: child.cocos_transformation_type,
         child_unit: child.unit,
         operator_kind: rel.operator_kind,
         operator: rel.operator,
         role: rel.role
       }) AS children
ORDER BY parent.id
```

The exact production rule was then applied: exclude binary children; when the
parent is not normalized, exclude normalized children; collect non-null child
classes; return the sole class only when the set cardinality is one, otherwise
return null (`graph_ops.py:3681-3696`, `graph_ops.py:3780-3782`). This yields 28
`one_like`, 35 null with structural children, 18 null with no structural child,
and one `b0_like`. The count sum is 82, complementing the 199 DD-backed rows.

For `magnetic_field`, 16 children are present; after excluding the binary child,
15 are eligible and the sole non-null eligible class is `b0_like`. Its current
paragraph is:

> Sign convention: Positive when a resolved component of $\mathbf{B}$ points
> along the corresponding positive direction of the right-handed cylindrical
> $(R,\phi,Z)$ frame: increasing $R$, increasing $\phi$, or increasing $Z$.

That is not the configured `b0_like` convention, which anchors the vacuum
toroidal field to increasing toroidal angle. Retaining or mechanically relabeling
the paragraph would therefore preserve a physics mismatch; this is the sole row
that needs regeneration.

## Complete identity partition

### DD-backed invariant quantities — 199, deterministic strip

`accumulated_deuterated_methane_prefill_count`, `accumulated_neutral_count_at_wall`, `accumulated_total_prefill_gas_count`, `ammonia_source_rate`, `argon_density_at_internal_transport_barrier`, `argon_density_at_pedestal_top`, `argon_density_at_separatrix`, `atomic_number`, `beryllium_density_at_pedestal_top`, `boron_density_at_divertor_target`, `boron_density_at_internal_transport_barrier`, `boron_density_at_plasma_boundary`, `carbon_density_at_pedestal_top`, `coolant_transit_time_of_plant_component_port`, `core_atomic_count_of_pellet`, `current_density_due_to_diamagnetic_drift`, `current_density_due_to_ohmic_current_drive`, `derivative_with_respect_to_toroidal_flux_coordinate_of_volume_of_flux_surface`, `deuterated_methane_source_rate`, `deuterium_density_at_pedestal_top`, `deuterium_tritium_density_at_pedestal_top`, `effective_turn_count_of_coil_conductor_element`, `effective_turn_count_of_passive_loop`, `electron_decay_time_at_magnetic_axis_due_to_disruption`, `equilibrium_weight_of_flux_loop`, `first_local_tangential_coordinate_of_aperture`, `first_local_tangential_coordinate_of_optical_element`, `first_local_tangential_coordinate_of_reflector`, `flux_surface_averaged_argon_density_at_plasma_boundary`, `flux_surface_averaged_beryllium_density_at_plasma_boundary`, `flux_surface_averaged_boron_density_at_plasma_boundary`, `flux_surface_averaged_carbon_density_at_plasma_boundary`, `flux_surface_averaged_helium_3_density_at_plasma_boundary`, `flux_surface_averaged_helium_4_density_at_plasma_boundary`, `flux_surface_averaged_ion_density_at_plasma_boundary`, `flux_surface_averaged_krypton_density_at_plasma_boundary`, `flux_surface_averaged_lithium_density_at_plasma_boundary`, `flux_surface_averaged_neon_density_at_plasma_boundary`, `flux_surface_averaged_nitrogen_density_at_plasma_boundary`, `flux_surface_averaged_tungsten_density_at_plasma_boundary`, `flux_surface_averaged_xenon_density_at_plasma_boundary`, `frequency_of_neoclassical_tearing_mode`, `gap_at_outboard_midplane`, `heat_decay_length_over_scrape_off_layer`, `helium_3_density_at_pedestal_top`, `helium_3_density_at_plasma_boundary`, `helium_4_density_at_pedestal_top`, `hydrogen_density_at_pedestal_top`, `ion_average_temperature_at_post_sawtooth_crash`, `ion_charge_number`, `iron_density_at_pedestal_top`, `length_of_poloidal_field_coil`, `lithium_density`, `lithium_density_at_pedestal_top`, `magnetic_shear_at_flux_surface`, `neon_density_at_pedestal_top`, `neutral_state_convection_velocity`, `neutral_state_particle_convection_velocity`, `neutral_temperature`, `nitrogen_density_at_pedestal_top`, `normalized_energy_flux_due_to_e_cross_b_drift`, `normalized_energy_flux_due_to_perturbed_parallel_magnetic_field`, `normalized_energy_flux_due_to_perturbed_parallel_vector_potential`, `normalized_particle_flux_due_to_perturbed_parallel_magnetic_field`, `normalized_poloidal_flux_coordinate_at_measurement_position`, `normalized_poloidal_flux_coordinate_of_pedestal`, `normalized_toroidal_flux_coordinate`, `normalized_toroidal_flux_coordinate_at_beam_tracing_point`, `normalized_toroidal_flux_coordinate_at_constraint_position`, `normalized_toroidal_flux_coordinate_at_internal_transport_barrier`, `normalized_toroidal_flux_coordinate_at_pedestal_top`, `normalized_toroidal_flux_coordinate_at_pellet_path`, `normalized_toroidal_flux_coordinate_of_neoclassical_tearing_mode_center`, `normalized_toroidal_wave_vector_of_beam_tracing_beam`, `outer_squareness_of_flux_surface`, `oxygen_density_at_plasma_boundary`, `parallel_current_density_due_to_anomalous_transport`, `parallel_current_density_due_to_perpendicular_viscosity`, `parallel_electron_energy_convection_velocity`, `parallel_electron_particle_convection_velocity`, `parallel_ion_charge_state_velocity_due_to_e_cross_b_drift`, `parallel_ion_momentum_flux_limiter_coefficient`, `parallel_mach_number`, `parallel_momentum_convection_velocity`, `parallel_neutral_state_velocity_due_to_diamagnetic_drift`, `parallel_normalized_momentum_flux_due_to_perturbed_parallel_magnetic_field`, `parallel_peak_launched_refractive_index_of_lower_hybrid_antenna`, `parallel_plasma_velocity`, `perpendicular_normalized_momentum_flux_due_to_e_cross_b_drift`, `plasma_beta`, `polarization_ellipticity_of_polarimeter_beam`, `poloidal_current_density_due_to_anomalous_transport`, `poloidal_ion_momentum_flux_limiter_coefficient`, `poloidal_ion_state_particle_convection_velocity`, `poloidal_ion_velocity`, `poloidal_neutral_state_particle_convection_velocity`, `poloidal_neutral_state_velocity_due_to_diamagnetic_drift`, `poloidal_total_current_density`, `pulse_duration`, `radial_coordinate`, `radial_coordinate_at_outboard_midplane`, `radial_coordinate_of_antenna_strap`, `radial_coordinate_of_camera`, `radial_coordinate_of_coil_conductor_element`, `radial_coordinate_of_geometric_axis`, `radial_coordinate_of_line_of_sight`, `radial_coordinate_of_measurement_position`, `radial_coordinate_of_poloidal_field_coil`, `radial_coordinate_of_poloidal_magnetic_field_probe`, `radial_coordinate_of_strike_point`, `radial_coordinate_of_thomson_scattering_laser`, `radial_coordinate_of_x_point`, `radial_current_density_due_to_anomalous_transport`, `radial_current_density_due_to_parallel_viscosity`, `radial_flux_limiter_coefficient`, `radial_ion_momentum_flux_limiter_coefficient`, `radial_neutral_state_velocity_due_to_diamagnetic_drift`, `radial_neutral_velocity`, `radial_outline_of_fiber_optic_current_sensor`, `radial_outline_of_plasma_boundary`, `radial_outline_of_wall`, `radial_vorticity`, `ratio_of_ion_density_to_electron_density`, `safety_factor_at_plasma_boundary`, `second_local_tangential_coordinate_of_aperture`, `second_local_tangential_width_of_aperture`, `second_local_tangential_width_of_neutral_beam_injector`, `spectral_efficiency_of_hard_xray_detector`, `spectral_efficiency_of_soft_xray_detector`, `spectral_efficiency_of_thomson_scattering_detector`, `square_of_ion_charge_number`, `toroidal_beryllium_velocity_at_plasma_boundary`, `toroidal_current_density_due_to_anomalous_transport`, `toroidal_current_density_due_to_diamagnetic_drift`, `toroidal_deuterium_tritium_velocity_at_plasma_boundary`, `toroidal_deuterium_velocity_at_pedestal_top`, `toroidal_electron_velocity`, `toroidal_flux_surface_averaged_beryllium_velocity_at_plasma_boundary`, `toroidal_flux_surface_averaged_deuterium_tritium_velocity_at_plasma_boundary`, `toroidal_flux_surface_averaged_helium_3_velocity_at_plasma_boundary`, `toroidal_flux_surface_averaged_helium_4_velocity_at_plasma_boundary`, `toroidal_flux_surface_averaged_hydrogen_velocity_at_plasma_boundary`, `toroidal_flux_surface_averaged_krypton_velocity_at_plasma_boundary`, `toroidal_flux_surface_averaged_lithium_velocity_at_plasma_boundary`, `toroidal_flux_surface_averaged_nitrogen_velocity_at_plasma_boundary`, `toroidal_flux_surface_averaged_tritium_velocity_at_plasma_boundary`, `toroidal_helium_4_velocity_at_magnetic_axis`, `toroidal_helium_4_velocity_at_plasma_boundary`, `toroidal_hydrogen_velocity_at_plasma_boundary`, `toroidal_krypton_velocity_at_plasma_boundary`, `toroidal_lithium_velocity_at_plasma_boundary`, `toroidal_neutral_internal_state_velocity_due_to_diamagnetic_drift`, `toroidal_neutral_state_velocity_due_to_diamagnetic_drift`, `toroidal_neutral_velocity`, `toroidal_nitrogen_velocity_at_plasma_boundary`, `toroidal_pfirsch_schlueter_current_density`, `toroidal_total_current_density`, `toroidal_tritium_velocity_at_plasma_boundary`, `toroidal_velocity_of_pellet`, `total_ion_particle_source_rate`, `total_neutral_source_rate_due_to_gas_injection`, `tritium_density_at_pedestal_top`, `tritium_velocity`, `tungsten_density`, `tungsten_density_at_internal_transport_barrier`, `tungsten_density_at_pedestal_top`, `turn_count_of_poloidal_magnetic_field_probe`, `turn_count_of_toroidal_magnetic_field_probe`, `upper_inner_squareness_of_plasma_boundary`, `upper_triangularity_of_flux_surface`, `vertical_back_surface_curvature_of_optical_element`, `vertical_coordinate_of_camera`, `vertical_coordinate_of_conductor_cross_section`, `vertical_coordinate_of_electron_cyclotron_launcher_mirror`, `vertical_coordinate_of_geometric_axis`, `vertical_coordinate_of_ion_cyclotron_heating_antenna`, `vertical_coordinate_of_iron_core_segment`, `vertical_coordinate_of_launching_position`, `vertical_coordinate_of_line_of_sight`, `vertical_coordinate_of_measurement_position`, `vertical_coordinate_of_plasma_boundary_gap_reference_point`, `vertical_coordinate_of_shunt`, `vertical_coordinate_of_strike_point`, `vertical_coordinate_of_x_point`, `vertical_current_density_due_to_anomalous_transport`, `vertical_current_density_due_to_heat_viscosity`, `vertical_ion_charge_state_velocity_due_to_diamagnetic_drift`, `vertical_ion_velocity`, `vertical_neutral_state_velocity_due_to_diamagnetic_drift`, `vertical_outline_of_antenna_strap`, `volume_of_plasma_boundary`, `wavelength_of_bragg_crystal`, `x_direction_unit_vector_of_camera`, `x_direction_unit_vector_of_neutron_detector`, `xenon_density_at_pedestal_top`, `y_direction_unit_vector_of_camera`, `y_direction_unit_vector_of_neutron_detector`, `y_unit_vector_of_pellet_injector`, `z_direction_unit_vector_of_neutron_detector`.

### Structurally backed invariant quantities — 28, deterministic strip

`atomic_count_of_pellet`, `current_density`, `current_density_due_to_anomalous_transport`, `current_density_due_to_heat_viscosity`, `current_density_due_to_parallel_viscosity`, `current_density_due_to_perpendicular_viscosity`, `decay_length_over_scrape_off_layer`, `density_at_internal_transport_barrier`, `density_at_pedestal_top`, `diamagnetic_momentum_flux_limiter_coefficient`, `electron_particle_convection_velocity`, `electron_pressure`, `electron_velocity`, `fast_ion_state_pressure`, `fast_neutral_state_pressure`, `inner_squareness_of_plasma_boundary`, `ion_momentum_flux_limiter_coefficient`, `ion_state_momentum_convection_velocity`, `ion_state_momentum_flux_limiter_coefficient`, `mach_number`, `neon_density`, `neutral_momentum_flux_limiter_coefficient`, `neutral_state_momentum_convection_velocity`, `neutral_state_momentum_flux_limiter_coefficient`, `neutral_velocity`, `pfirsch_schlueter_current_density`, `plasma_velocity`, `triangularity_of_flux_surface`.

### Stale `one_like` that structural authority resolves to null — 35, deterministic metadata and text edit

`area_of_flux_surface`, `argon_density`, `beryllium_density`, `boron_density`, `center_of_mass_velocity`, `current_density_due_to_ion_inertia`, `current_density_due_to_ion_neutral_friction`, `current_density_due_to_viscosity`, `density_at_limiter`, `density_at_magnetic_axis`, `efficiency_of_plant_system`, `energy_convection_velocity`, `fast_electron_pressure`, `fast_neutral_pressure`, `helium_3_density`, `hydrogen_density`, `inner_squareness_of_flux_surface`, `ion_momentum_convection_velocity`, `ion_momentum_damping_rate`, `iron_density`, `neutral_momentum_convection_velocity`, `neutral_species_momentum_convection_velocity`, `normalized_gyrocenter_perturbed_pressure`, `normalized_particle_perturbed_pressure`, `normalized_perturbed_vector_potential`, `outer_squareness_of_plasma_boundary`, `oxygen_density`, `perturbed_pressure`, `refractive_index`, `source_rate_due_to_gas_injection`, `source_rate_due_to_injection`, `total_fast_ion_pressure`, `total_ion_temperature`, `tritium_density`, `xenon_density`.

### Ungrounded `one_like` with no DD or structural authority — 18, deterministic fail-closed metadata and text edit

`cross_sectional_area_of_flux_surface`, `internal_inductance`, `ion_state_density`, `ion_state_momentum_damping_rate`, `ion_state_velocity`, `ion_state_velocity_due_to_e_cross_b_drift`, `neutral_state_velocity`, `parallel_momentum_flux_due_to_perturbed_parallel_vector_potential`, `perpendicular_momentum_flux_due_to_perturbed_parallel_magnetic_field`, `perturbed_particle_pressure`, `poloidal_cross_sectional_area_of_flux_surface`, `radial_outline_of_limiter`, `toroidal_magnetic_field_at_constraint_position`, `vertical_coordinate_of_divertor_target`, `vertical_coordinate_of_sensor_attachment_point`, `x_direction_unit_vector_of_sensor`, `x_image_up_unit_vector_of_camera`, `z_direction_unit_vector_of_sensor`.

### Stale `one_like` that structural authority resolves to `b0_like` — 1, regeneration required

`magnetic_field`.

## Repair boundary

The cheapest faithful campaign is therefore **280 deterministic zero-LLM row
repairs plus one regenerated document**, not regeneration of all 281. The
deterministic mutation still needs the repository's governed repair operator and
an exact signed identity manifest; this diagnosis is evidence, not mutation
authority. The `magnetic_field` row must pass the ordinary documentation review
and refine quorum after regeneration. The prompt condition and nullable-property
writer must also be fixed before applying the campaign, otherwise new docs or a
later structural reconcile can recreate the same defects.
