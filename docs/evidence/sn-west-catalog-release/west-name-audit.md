# WEST batch accepted names — physical-correctness audit

provisional: true

Are the accepted standard names bound to WEST batch sources physically correct
and self-descriptive? Each row below is judged against the data-dictionary
documentation of the source path it is bound to, its unit, and the name's own
description. Rows are appended as they are judged.

## How the cohort was drawn

The WEST batch membership predicate is the 342 data-dictionary source paths in
`imas_codex/standard_names/manifests/west_production_dd_paths.yaml`. Drawn
through `GraphClient()` against the live graph:

```cypher
MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
WHERE src.source_id IN $paths AND sn.name_stage = 'accepted'
OPTIONAL MATCH (dd:IMASNode {id: src.source_id})
RETURN sn.id AS name, src.source_id AS path, sn.unit, sn.description,
       dd.unit, dd.documentation
```

**341 accepted names** are bound to those 342 paths. The audited sample is
**every fourth row of the path-ordered cohort — 86 names**, which spreads the
sample over 20 IDSs (equilibrium 19, summary 15, magnetics 6, ic_antennas 5,
spectrometer_visible 5, camera_x_rays 4, core_profiles 4, hard_x_rays 4,
interferometer 4, ece 3, polarimeter 3, soft_x_rays 3, bremsstrahlung_visible 2,
calorimetry 2, wall 2, barometry 1, camera_ir 1, pf_active 1, pf_passive 1,
spectrometer_mass 1). Sampling is deterministic and independent of the verdict,
so it cannot select for agreement.

### The instrument was controlled before any absence was reported

The first two query forms returned **0 rows** and **null names**, and both were
instrument faults rather than findings: `StandardNameSource` carries
`source_id`, not `path`; the accepted flag is `name_stage`, not `status`
(`sn.status` is the catalog lifecycle field and holds only `draft` (2934) and
`superseded` (2196), so a `status = 'accepted'` predicate matches nothing); and
the name string is `sn.id`. `IMASNode` likewise keys on `id` and `unit`, not
`path`/`units` — the first join returned `dd_doc` on 0 of 341 rows. The
corrected join returns documentation on **341 of 341**, and a positive control
counts 5130 `StandardName` nodes. For a `/value` leaf whose own documentation is
the literal string `Value`, the parent container's documentation is used
(present for 332 of 341).

## Verdicts

### 1. `neutral_gas_pressure_of_gauge` — **correct**

- source path: `barometry/gauge/pressure`
- unit: `Pa` (data dictionary: `Pa`)
- data-dictionary text: Pressure
- name description: Neutral gas pressure of a gage is the pressure indicated by a barometry gage for residual neutral gas in the vacuum vessel or surrounding vacuum system. It represents local vacuum conditions rather than a kinetic neutral-particle pressure moment.
- note: The name is correct; its description spells the instrument "gage" where the DD path and the name spell it `gauge`.

### 2. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `bremsstrahlung_visible/channel/line_of_sight/first_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 3. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `bremsstrahlung_visible/channel/line_of_sight/second_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.

### 4. `coolant_temperature_at_inlet` — **correct**

- source path: `calorimetry/cooling_loop/temperature_in`
- unit: `K` (data dictionary: `K`)
- data-dictionary text: Temperature of the coolant when entering the loop
- name description: Absolute thermodynamic temperature of coolant entering a cooling loop or plant component, establishing the upstream state for enthalpy-based heat-transfer balances.

### 5. `coolant_absorbed_power_of_calorimetry_component` — **correct**

- source path: `calorimetry/group/component/power`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: Power extracted from the component
- name description: Thermal power removed from a calorimetry component by its coolant, determined from the coolant enthalpy increase between inlet and outlet.

### 6. `surface_temperature` — **INCORRECT**

- source path: `camera_ir/channel/camera/frame/apparent_temperature`
- unit: `K` (data dictionary: `K`)
- data-dictionary text: Processed image in apparent temperature. First dimension : row index (horizontal orientation). Second dimension: column index (vertical orientation). The size of this matrix is assumed to be constant over time
- name description: Surface temperature is the thermodynamic temperature of the exposed material boundary of a plasma-facing component at a specified surface point. It refers to the surface material state, not to an area aggregate or regional extremum.
- **rejected spelling** `surface_temperature` → **proposed spelling** `apparent_surface_temperature`
- why: The DD text is explicit: the frame is a **processed image in apparent temperature**. An infrared camera measures radiance and converts it to a temperature under an assumed emissivity, so the quantity is a brightness/apparent temperature, not the thermodynamic temperature of the surface — the two differ by the emissivity correction and can diverge by hundreds of kelvin on a metallic plasma-facing component. The name drops the word the DD source puts first, and the description then asserts the stronger claim ("thermodynamic temperature") that the data does not support.

### 7. `toroidal_coordinate_of_camera` — **correct**

- source path: `camera_x_rays/camera/centre/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal coordinate of a camera's geometric center in the right-handed cylindrical (R, φ, Z) frame, where φ is the toroidal angle about the machine symmetry axis.

### 8. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `camera_x_rays/camera/line_of_sight/first_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 9. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `camera_x_rays/camera/line_of_sight/second_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.

### 10. `relative_humidity_of_detector` — **correct**

- source path: `camera_x_rays/detector_humidity`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Fraction of humidity (0-1) measured at the detector level
- name description: Ratio of ambient water-vapor partial pressure to its saturation vapor pressure at the detector, describing local moisture conditions that can affect detector performance.

### 11. `total_electron_density` — **correct**

- source path: `core_profiles/profiles_1d/electrons/density`
- unit: `m^-3` (data dictionary: `m^-3`)
- data-dictionary text: Density (thermal+non-thermal)
- name description: Number density of free electrons per physical volume for the complete local population, combining thermal and fast electrons rather than selecting one population component.

### 12. `electron_temperature` — **correct**

- source path: `core_profiles/profiles_1d/electrons/temperature`
- unit: `eV` (data dictionary: `eV`)
- data-dictionary text: Temperature
- name description: Energy-equivalent measure of the mean random kinetic energy per electron in the bulk electron population, defined by its second velocity moment.

### 13. `poloidal_magnetic_flux_at_magnetic_axis` — **correct**

- source path: `core_profiles/profiles_1d/grid/psi_magnetic_axis`
- unit: `Wb` (data dictionary: `Wb`)
- data-dictionary text: Value of the poloidal magnetic flux at the magnetic axis (useful to normalize the psi array values when the radial grid doesn't go from the magnetic axis to the plasma boundary)
- name description: Signed poloidal magnetic flux evaluated at the magnetic axis, providing the inner reference value for normalized poloidal-flux coordinates.

### 14. `volume_of_flux_surface` — **correct**

- source path: `core_profiles/profiles_1d/grid/volume`
- unit: `m^3` (data dictionary: `m^3`)
- data-dictionary text: Volume enclosed inside the magnetic surface
- name description: Geometric volume contained within a nested magnetic flux surface, cumulative from the magnetic axis toward the outermost closed surface.
- note: `volume_of_flux_surface` names the volume enclosed by the surface rather than a volume of the surface itself; this is the settled convention of the `*_of_flux_surface` family (adjudicated with `area_of_flux_surface`) and is not relitigated here.

### 15. `frequency_of_wave_diagnostic_channel` — **correct**

- source path: `ece/channel/frequency`
- unit: `Hz` (data dictionary: `Hz`)
- data-dictionary text: Frequency of the channel
- name description: Temporal frequency associated with a wave-diagnostic channel, defining the oscillation or probing frequency used to establish its plasma interaction.

### 16. `radial_coordinate_of_measurement_position` — **correct**

- source path: `ece/channel/position/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate locating a measurement position by perpendicular distance from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.
- note: Correct at this locus: an ECE channel `position` genuinely is the measurement position. The same name is bound elsewhere to loci that are not — see the out-of-sample findings.

### 17. `radiative_temperature_at_ece_channel` — **correct**

- source path: `ece/channel/t_radiation`
- unit: `eV` (data dictionary: `eV`)
- data-dictionary text: Radiation temperature
- name description: Equivalent blackbody temperature of electron cyclotron emission intensity detected by a frequency-selective ECE channel, representing the radiation brightness temperature.

### 18. `vertical_coordinate_of_geometric_axis` — **correct**

- source path: `equilibrium/time_slice/boundary/geometric_axis/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed Z height of the plasma boundary’s geometric axis, computed as the midpoint of its maximum and minimum vertical extents.

### 19. `poloidal_magnetic_flux_at_plasma_boundary` — **correct**

- source path: `equilibrium/time_slice/boundary/psi`
- unit: `Wb` (data dictionary: `Wb`)
- data-dictionary text: Value of the poloidal flux at which the boundary is taken
- name description: Signed poloidal magnetic flux evaluated on the last closed flux surface, providing the outer reference for normalized poloidal-flux coordinates.

### 20. `upper_triangularity_of_plasma_boundary` — **correct**

- source path: `equilibrium/time_slice/boundary/triangularity_upper`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Upper triangularity of the plasma boundary
- name description: Dimensionless shape parameter for the plasma-boundary poloidal cross-section, expressing the inward radial displacement of its upper extremum relative to the geometric center.

### 21. `faraday_angle` — **INCORRECT**

- source path: `equilibrium/time_slice/constraints/faraday_angle/reconstructed`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Value calculated from the reconstructed equilibrium
- name description: Faraday rotation of a probing wave’s polarization plane caused by electron density and the magnetic-field component along its plasma path.
- **rejected spelling** `faraday_angle` → **proposed spelling** `faraday_rotation_angle`
- why: `faraday_angle` names no physical angle on its own — the Faraday effect is a *rotation of the polarization plane*, and "Faraday angle" is not the quantity's name in the literature or in the DD's own polarimetry documentation. The name's own description has to supply the missing word ("rotation of a probing wave's polarization plane"), which is the test for self-descriptiveness failing.

### 22. `line_integrated_electron_number_density` — **INCORRECT**

- source path: `equilibrium/time_slice/constraints/n_e_line/reconstructed`
- unit: `m^-2` (data dictionary: `m^-2`)
- data-dictionary text: Value calculated from the reconstructed equilibrium
- name description: Free-electron column density accumulated along a complete electromagnetic propagation path, including both forward and return segments when present.
- **rejected spelling** `line_integrated_electron_number_density` → **proposed spelling** `line_integrated_electron_density`
- why: The cohort spells this physical base `electron_density` in eight names (`electron_density_at_magnetic_axis`, `line_averaged_electron_density`, `thermal_electron_density`, `total_electron_density`, `volume_averaged_electron_density`, and three more) and `electron_number_density` in exactly one — this one. Two spellings of one base inside one published batch is the same defect class the plan already recorded for the etendue row. The majority spelling is the survivor; the semantic content is identical, so the minority spelling is an inconsistency rather than a distinction.

### 23. `toroidal_beta` — **correct**

- source path: `equilibrium/time_slice/global_quantities/beta_tor`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2
- name description: Toroidal beta is a dimensionless equilibrium ratio of volume-averaged total perpendicular plasma pressure to the magnetic pressure of a reference toroidal field.

### 24. `plasma_current` — **correct**

- source path: `equilibrium/time_slice/global_quantities/ip`
- unit: `A` (data dictionary: `A`)
- data-dictionary text: Plasma current. Positive sign means anti-clockwise when viewed from above.
- name description: Net toroidal electric current carried by the entire plasma column, obtained by integrating toroidal current density over its enclosed poloidal cross-section.

### 25. `radial_coordinate_of_magnetic_axis` — **correct**

- source path: `equilibrium/time_slice/global_quantities/magnetic_axis/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius of the magnetic axis
- name description: Major-radius coordinate locating the magnetic-axis O-point in the right-handed cylindrical (R, φ, Z) frame around which nested closed flux surfaces are organized.

### 26. `safety_factor_at_normalized_poloidal_magnetic_flux_equal_to_0_95` — **correct**

- source path: `equilibrium/time_slice/global_quantities/q_95`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: q at the 95% poloidal flux surface
- name description: Signed field-line winding number on the closed magnetic flux surface labeled by normalized poloidal magnetic flux 0.95, near but inside the plasma boundary.

### 27. `surface_area_of_flux_surface` — **correct**

- source path: `equilibrium/time_slice/global_quantities/surface`
- unit: `m^2` (data dictionary: `m^2`)
- data-dictionary text: Surface area of the toroidal flux surface
- name description: Geometric area of a closed toroidal magnetic flux surface formed by revolving its poloidal contour through one complete toroidal revolution in the right-handed cylindrical (R, φ, Z) frame.

### 28. `maximum_magnetic_field_magnitude` — **correct**

- source path: `equilibrium/time_slice/profiles_1d/b_field_max`
- unit: `T` (data dictionary: `T`)
- data-dictionary text: Maximum(modulus(B)) on the flux surface (always positive, irrespective of the sign convention for the B-field direction)
- name description: Largest magnitude of the magnetic-field vector over a specified evaluation domain, distinguishing total field strength from any signed field component.

### 29. `radial_derivative_of_poloidal_magnetic_flux` — **INCORRECT**

- source path: `equilibrium/time_slice/profiles_1d/dpsi_drho_tor`
- unit: `Wb.m^-1` (data dictionary: `Wb.m^-1`)
- data-dictionary text: Derivative of Psi with respect to Rho_Tor
- name description: Rate of change of signed poloidal magnetic flux with the dimensionful toroidal-flux radius labeling nested equilibrium flux surfaces.
- **rejected spelling** `radial_derivative_of_poloidal_magnetic_flux` → **proposed spelling** `derivative_of_poloidal_magnetic_flux_with_respect_to_toroidal_flux_coordinate`
- why: The DD source is `dpsi_drho_tor` — the derivative with respect to **rho_tor**, the dimensionful toroidal-flux radius. "Radial derivative" in a cylindrical-coordinate catalog reads as d/dR, the major-radius derivative, which is a different quantity with the same units. The cohort already carries the explicit convention: its sibling spells out `..._with_respect_to_poloidal_magnetic_flux_coordinate` in full, so the precise form was available and this row departed from it.

### 30. `product_of_poloidal_current_function_and_derivative_of_poloidal_current_function_with_respect_to_poloidal_magnetic_flux_coordinate` — **correct**

- source path: `equilibrium/time_slice/profiles_1d/f_df_dpsi`
- unit: `T^2.Wb^-1.m^2` (data dictionary: `T^2.Wb^-1.m^2`)
- data-dictionary text: Derivative of F w.r.t. Psi, multiplied with F
- name description: Grad–Shafranov equilibrium source term formed by multiplying the poloidal current function by its derivative with respect to signed poloidal magnetic flux.
- note: Long, but every segment is load-bearing and the name is exactly reconstructible from the DD source; length is not a defect where the quantity is a product of a function and its own derivative.

