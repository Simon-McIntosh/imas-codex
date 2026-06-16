## ISN Exemplars

Study the **reasoning**, not just the names. The canonical form often surprises.

### ✅ Positive Exemplars — produce names like these

**E1. Zero-operator base** — `electron_temperature`
Subject `electron` + base `temperature`. No operators, no locus. The simplest valid form.

**E2. Unary prefix operator** — `gradient_of_electron_pressure`
Operator `gradient` wraps the inner name `electron_pressure` with explicit `_of_` scope.

**E3. Unary postfix operator** — `magnetic_field_magnitude`
Operator `magnitude` appends directly to the base. Postfix operators never use `_of_`.

**E4. Binary operator** — `ratio_of_electron_to_ion_temperature`
Binary `ratio` uses `_of_` scope + `_to_` separator between the two operands.

**E5. Axis projection** — `radial_ion_velocity`
Projection prefix `radial_` precedes the qualified base (short form). The `_component_of_` connector is REJECTED by grammar.

**E6. Postfix locus (_at_)** — `electron_density_at_magnetic_axis`
The locus `_at_magnetic_axis` follows the base. `_at_` marks a field value sampled at a point.

**E7. Mechanism qualifier** — `plasma_current_due_to_bootstrap`
The process `bootstrap` is attached via `_due_to_` after the base quantity.

**E8. Bare-prefix transformation** — `line_averaged_electron_density`
`line_averaged` is an averaging transformation: it attaches BARE (no `_of_`), parsed as the outermost qualifier. Averaging/integrating transformations (`volume_averaged`, `flux_surface_averaged`, `normalized`, `surface_integrated`, `per_toroidal_mode`, ...) never carry `_of_`. Differential/scalar operators (`time_derivative`, `gradient`, ...) DO: `gradient_of_electron_pressure`.

**E9. Projection + locus combo** — `radial_magnetic_field_at_separatrix`
Projection prefix + locus postfix coexist cleanly: `[projection] base [locus]`.

**E10. Nested operators** — `time_derivative_of_volume_averaged_electron_density`
Outer operator `time_derivative` (a differential prefix, `_of_` scope) wraps the
inner bare-prefix transformation `volume_averaged` on base `electron_density`.
Differential operators carry `_of_`; averaging/integrating transformations attach
bare — so the two compose as `time_derivative_of_` + `volume_averaged_electron_density`.

**E11. Spectral decomposition** — `per_toroidal_mode_wave_absorbed_power`
Bare-prefix transformation `per_toroidal_mode` on base `wave_absorbed_power` (bare, no `_of_`). Indicates the quantity is resolved per Fourier toroidal-mode component. `per_toroidal_mode` and `per_poloidal_mode` are registered bare-prefix transformations — do not flag them as unknown.

### ❌ Negative Exemplars — rc20 forms now rejected

**N1.** ❌ `real_part_of_perturbed_electrostatic_potential`
*Rejected:* `real_part` is a postfix operator. Prefix `real_part_of_X` creates nested `_of_` ambiguity.
✅ Use: `perturbed_electrostatic_potential_real_part`

**N2.** ❌ `amplitude_of_parallel_component_of_wave_electric_field`
*Rejected:* `amplitude` is postfix. Prefix form with nested `_of_` chains is unparseable. Also `_component_of_` is REJECTED.
✅ Use: `parallel_wave_electric_field_amplitude`

**N3.** ❌ `volume_averaged_of_electron_density`
*Rejected:* Averaging/integrating transformations are BARE prefixes (parsed as qualifiers); the `_of_` scope marker is ungrammatical for them.
✅ Use: `volume_averaged_electron_density`

**N4.** ❌ `diamagnetic_component_of_ion_velocity`
*Rejected:* `diamagnetic` is a drift mechanism, not a projection axis. The diamagnetic drift IS a velocity, attached via `_due_to_`.
✅ Use: `ion_velocity_due_to_diamagnetic_drift`

**N5.** ❌ `reconstructed_safety_factor`
*Rejected:* Provenance verbs (`reconstructed_`, `measured_`, `fitted_`) describe the data pipeline, not the physics quantity.
✅ Use: `safety_factor`

**N6.** ❌ `ion_rotation_frequency_toroidal`
*Rejected:* Trailing component suffix violates the canonical projection-prefix pattern.
✅ Use: `toroidal_ion_rotation_frequency`

**N7.** ❌ `electron_thermal_pressure`
*Rejected:* Population qualifier `thermal` must precede species `electron`. Species is a qualifier; population class is also a qualifier — they compose as `[population]_[species]`.
✅ Use: `thermal_electron_pressure`

**N8.** ❌ `electron_temperature_profile`
*Rejected:* The `_profile` suffix encodes data rank (1D array), not physics. Every standard name is a point value; profiles are implicit.
✅ Use: `electron_temperature`

**N9.** ❌ `norm_poloidal_flux`
*Rejected:* Abbreviations fragment the vocabulary. All tokens must be spelled in full.
✅ Use: `normalized_poloidal_magnetic_flux`

**N10.** ❌ `poloidal_magnetic_flux_of_plasma_boundary_at_plasma_boundary`
*Rejected:* Duplicated preposition — `_of_` and `_at_` for the same entity. Only one locus per name.
✅ Use: `poloidal_magnetic_flux_of_plasma_boundary`
