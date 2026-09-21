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

### 31. `flux_surface_averaged_inverse_of_square_of_magnetic_field_magnitude` — **correct**

- source path: `equilibrium/time_slice/profiles_1d/gm4`
- unit: `T^-2` (data dictionary: `T^-2`)
- data-dictionary text: Flux surface averaged 1/B^2
- name description: Flux-surface average of the reciprocal square of total magnetic-field strength, a geometric coefficient in neoclassical and flux-surface-averaged transport relations.

### 32. `flux_surface_averaged_major_radius` — **correct**

- source path: `equilibrium/time_slice/profiles_1d/gm8`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Flux surface averaged R
- name description: Flux-surface-averaged nonnegative perpendicular distance from the toroidal symmetry axis to points on a nested magnetic flux surface, giving the mean major-radius coordinate in the right-handed cylindrical (R, φ, Z) frame.

### 33. `magnetic_shear_at_flux_surface` — **correct**

- source path: `equilibrium/time_slice/profiles_1d/magnetic_shear`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Magnetic shear, defined as rho_tor/q . dq/drho_tor
- name description: Dimensionless logarithmic radial gradient of the safety factor on a magnetic flux surface, measuring how magnetic-field-line pitch changes across neighboring surfaces.

### 34. `safety_factor` — **correct**

- source path: `equilibrium/time_slice/profiles_1d/q`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Safety factor
- name description: Signed ratio of toroidal to poloidal field-line winding on a magnetic flux surface, equal to the toroidal turns made during one poloidal circuit.

### 35. `normalized_toroidal_flux_coordinate` — **correct**

- source path: `equilibrium/time_slice/profiles_1d/rho_tor_norm`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm, is the toroidal flux coordinate at the equilibrium boundary (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation)
- name description: Dimensionless radial label equal to the square root of toroidal magnetic flux normalized between the magnetic axis and equilibrium boundary.

### 36. `volume_of_flux_surface` — **correct**

- source path: `equilibrium/time_slice/profiles_1d/volume`
- unit: `m^3` (data dictionary: `m^3`)
- data-dictionary text: Volume enclosed in the flux surface
- name description: Geometric volume contained within a nested magnetic flux surface, cumulative from the magnetic axis toward the outermost closed surface.

### 37. `area_of_diagnostic_aperture` — **INCORRECT**

- source path: `hard_x_rays/channel/detector/surface`
- unit: `m^2` (data dictionary: `m^2`)
- data-dictionary text: Surface of the detector/aperture, derived from the above geometric data
- name description: Geometric area of the designated diagnostic aperture surface that admits or collects radiation or particles.
- **rejected spelling** `area_of_diagnostic_aperture` → **proposed spelling** `area_of_diagnostic_detector`
- why: The bound path is `hard_x_rays/channel/**detector**/surface`, the detector's own collecting surface. Detector area and aperture area are distinct quantities — their product with the separation is what sets the channel etendue, so conflating them corrupts the one relation they both enter. The DD doc string is shared boilerplate ("Surface of the detector/aperture") and does not disambiguate; the path does. The cohort uses `..._of_aperture` for genuine aperture nodes elsewhere (`camera_x_rays/aperture/centre/phi`), so the name claims a locus the batch already spends on something else.

### 38. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `hard_x_rays/channel/line_of_sight/first_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 39. `hard_xray_brightness` — **INCORRECT**

- source path: `hard_x_rays/channel/radiance`
- unit: `m^-2.s^-1.sr^-1` (data dictionary: `m^-2.s^-1.sr^-1`)
- data-dictionary text: Photons received by the detector per unit time, per unit solid angle and per unit area (i.e. photon flux divided by the etendue), in multiple energy bands if available from the detector
- name description: Band-integrated hard X-ray photon radiance received by a channel, normalized by projected collecting area and accepted solid angle.
- **rejected spelling** `hard_xray_brightness` → **proposed spelling** `hard_xray_photon_radiance`
- why: The DD path is `radiance`, the DD text defines photon flux per unit time, solid angle and area, the unit is `m^-2.s^-1.sr^-1`, and the name's own description says "photon radiance". Three sibling quantities in this same cohort are spelled with `photon_radiance` — `photon_radiance_at_spectral_line` carries the identical unit. "Brightness" is also the established informal word for *brightness temperature*, a quantity in kelvin, so the spelling collides with a different physical base at the one place it is least affordable.

### 40. `lower_bound_photon_energy` — **correct**

- source path: `hard_x_rays/emissivity_profile_1d/lower_bound`
- unit: `eV` (data dictionary: `eV`)
- data-dictionary text: Lower bound of the energy band
- name description: Lower boundary of an X-ray photon-energy band, specifying the minimum photon energy included in the defined detection or emission band.

### 41. `frequency_of_ion_cyclotron_heating_antenna` — **correct**

- source path: `ic_antennas/antenna/frequency`
- unit: `Hz` (data dictionary: `Hz`)
- data-dictionary text: Frequency (average over modules)
- name description: Radio-frequency drive frequency of an ion-cyclotron heating antenna, averaged over its modules. This non-negative scalar sets the ion-cyclotron resonance layer locations and the ion species and harmonics accessible for wave absorption.

### 42. `reflected_phase_of_ion_cyclotron_heating_antenna` — **correct**

- source path: `ic_antennas/antenna/module/phase_reflected`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Phase of the reflected power with respect to the forward power of this module
- name description: Relative phase angle of a reflected radio-frequency power-wave phasor at an antenna element in an ion-cyclotron heating launcher, measured relative to the forward power-wave phasor of that same element.

### 43. `back_surface_distance_of_antenna_strap` — **correct**

- source path: `ic_antennas/antenna/module/strap/distance_to_conductor`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Distance to conducting wall or other conductor behind the antenna strap
- name description: Geometric rear-surface clearance between an ion-cyclotron heating antenna strap and the conducting wall or conductor behind it.
- note: Already adjudicated through the sanctioned rename route and accepted; not reopened here.

### 44. `wave_phase_of_antenna_strap` — **correct**

- source path: `ic_antennas/antenna/module/strap/phase`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Phase of the strap current
- name description: Phase angle of the radio-frequency current phasor carried by one ion-cyclotron-heating antenna strap, defined relative to a specified RF phase reference.

### 45. `net_power_due_to_ion_cyclotron_heating` — **correct**

- source path: `ic_antennas/antenna/power_launched`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: Power launched from this antenna into the vacuum vessel
- name description: Net ion-cyclotron radio-frequency power launched into the vacuum vessel by a specified heating launcher before absorption by plasma particles.
- note: Matches the live successor spelling the plan records for the superseded ICH identity.

### 46. `toroidal_coordinate_of_line_of_sight` — **correct**

- source path: `interferometer/channel/line_of_sight/second_point/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of the first reference point on a diagnostic line of sight, locating that point around the machine symmetry axis.
- note: Name correct; its **description is wrong** for this binding — it says "the first reference point" while the bound path is `second_point/phi`. The identity is shared across first/second/third points, so the description must not name one of them.

### 47. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `interferometer/channel/line_of_sight/third_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 48. `length_variation_of_interferometer_beam` — **correct**

- source path: `interferometer/channel/path_length_variation`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Optical path length variation due to the plasma
- name description: Signed plasma-induced change in the effective optical path accumulated along an interferometer beam relative to its corresponding no-plasma reference path.

### 49. `total_electron_count` — **correct**

- source path: `interferometer/electrons_n`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Total number of electrons in the plasma, estimated from the line densities measured by the various channels
- name description: Total inventory of free electrons within the plasma, including thermal and fast populations and excluding electrons bound in atoms or molecules.

### 50. `length_of_toroidal_magnetic_field_probe` — **correct**

- source path: `magnetics/b_field_phi_probe/length`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Length of the sensor along it's normal vector (n)
- name description: Physical length of a toroidal magnetic-field probe is the non-negative axial extent of the probe coil along its local normal sensing axis. It characterizes sensor geometry and effective sensing volume; magnetic-flux sensitivity is associated with enclosed coil area and winding turn count rather than this length alone.

### 51. `vertical_coordinate_of_toroidal_magnetic_field_probe` — **correct**

- source path: `magnetics/b_field_phi_probe/position/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Vertical position of the geometric center of a toroidal magnetic-field probe coil, expressed as the signed Z coordinate in the right-handed cylindrical (R, φ, Z) frame.

### 52. `poloidal_magnetic_field` — **correct**

- source path: `magnetics/b_field_pol_probe/field`
- unit: `T` (data dictionary: `T`)
- data-dictionary text: Measured magnetic field
- name description: Poloidal magnetic-field strength of the local total induction, formed from radial and vertical components in the right-handed cylindrical (R, φ, Z) frame.

### 53. `radial_coordinate_of_poloidal_magnetic_field_probe` — **correct**

- source path: `magnetics/b_field_pol_probe/position/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate locating the center of a poloidal magnetic-field probe in the right-handed cylindrical (R, φ, Z) frame.

### 54. `voltage_of_poloidal_magnetic_field_probe` — **correct**

- source path: `magnetics/b_field_pol_probe/voltage`
- unit: `V` (data dictionary: `V`)
- data-dictionary text: Voltage on the coil terminals
- name description: Terminal voltage at a poloidal magnetic-field probe is the inductive voltage at the coil terminals caused by time variation of the local poloidal magnetic-field component in the R-Z plane of the right-handed cylindrical (R, φ, Z) frame threading the probe windings.

### 55. `vertical_coordinate_of_flux_loop` — **correct**

- source path: `magnetics/flux_loop/position/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Vertical Z coordinate of a position point associated with one flux loop in the right-handed cylindrical (R, φ, Z) frame.

### 56. `radial_coordinate_of_conductor_cross_section` — **correct**

- source path: `pf_active/coil/element/geometry/rectangle/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Geometric centre R
- name description: Major-radius coordinate of the designated reference point associated with a conductor cross-section, measured from the toroidal symmetry axis.

### 57. `current_of_passive_loop` — **correct**

- source path: `pf_passive/loop/current`
- unit: `A` (data dictionary: `A`)
- data-dictionary text: Passive loop current
- name description: Signed conventional electric current circulating in one axisymmetric passive conducting loop, induced by changing linked magnetic flux and electromagnetic coupling.

### 58. `toroidal_coordinate_of_line_of_sight` — **correct**

- source path: `polarimeter/channel/line_of_sight/first_point/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of the first reference point on a diagnostic line of sight, locating that point around the machine symmetry axis.

### 59. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `polarimeter/channel/line_of_sight/second_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 60. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `polarimeter/channel/line_of_sight/third_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.

### 61. `lower_bound_photon_energy` — **correct**

- source path: `soft_x_rays/channel/energy_band/lower_bound`
- unit: `eV` (data dictionary: `eV`)
- data-dictionary text: Lower bound of the energy band
- name description: Lower boundary of an X-ray photon-energy band, specifying the minimum photon energy included in the defined detection or emission band.

### 62. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `soft_x_rays/channel/line_of_sight/first_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 63. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `soft_x_rays/channel/line_of_sight/second_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.

### 64. `voltage_of_mass_spectrometer_channel` — **INCORRECT**

- source path: `spectrometer_mass/channel/photomultiplier_voltage`
- unit: `V` (data dictionary: `V`)
- data-dictionary text: Voltage applied to the photomultiplier
- name description: Signed bias potential applied between the photomultiplier detector electrode and its electronics reference for one mass-resolved channel, setting ion-signal gain.
- **rejected spelling** `voltage_of_mass_spectrometer_channel` → **proposed spelling** `photomultiplier_voltage_of_mass_spectrometer_channel`
- why: The DD source is `photomultiplier_voltage` — the bias applied to the photomultiplier. A mass-spectrometer channel carries several distinct voltages (ion-source, quadrupole/analyser, detector bias), so `voltage_of_mass_spectrometer_channel` does not identify which one and cannot be resolved by a reader without opening the DD path. The description already knows the answer ("bias potential applied between the photomultiplier detector electrode and its electronics reference").

### 65. `wavelength_of_spectral_line` — **correct**

- source path: `spectrometer_visible/channel/grating_spectrometer/processed_line/wavelength_central`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Central wavelength of the processed line
- name description: Characteristic vacuum wavelength assigned to an atomic, ionic, molecular, or nuclear spectral transition, identifying its central line position.

### 66. `cold_neutral_temperature` — **correct**

- source path: `spectrometer_visible/channel/isotope_ratios/isotope/cold_neutrals_temperature`
- unit: `eV` (data dictionary: `eV`)
- data-dictionary text: Temperature of cold neutrals for this isotope
- name description: Translational kinetic temperature, expressed as energy per particle, of the cold neutral population, based on random translational motion after removal of its bulk flow.

### 67. `atomic_number` — **correct**

- source path: `spectrometer_visible/channel/isotope_ratios/isotope/element/z_n`
- unit: `1` (data dictionary: `e`)
- data-dictionary text: Nuclear charge
- name description: Nuclear proton count identifying the selected element in a plasma or neutral-particle species, independent of isotope and ionization state.
- note: Name correct. The unit disagrees with the DD: the standard name carries `1` and the DD carries `e`. Atomic number is a proton count and dimensionless, so the standard name is the defensible side of the disagreement.

### 68. `toroidal_coordinate_of_line_of_sight` — **correct**

- source path: `spectrometer_visible/channel/line_of_sight/first_point/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of the first reference point on a diagnostic line of sight, locating that point around the machine symmetry axis.

### 69. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `spectrometer_visible/channel/line_of_sight/second_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 70. `gap_at_closest_wall_point` — **correct**

- source path: `summary/boundary/gap_limiter_wall/value`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Distance between the separatrix and the nearest limiter or wall element
- name description: Minimum geometric clearance between the plasma separatrix and the nearest limiter or wall element, evaluated at the closest-wall point.

### 71. `vertical_coordinate_of_magnetic_axis` — **correct**

- source path: `summary/boundary/magnetic_axis_z/value`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Z position of the magnetic axis
- name description: Signed vertical Z coordinate of the magnetic axis in the right-handed cylindrical (R, φ, Z) frame, marking the interior extremum organizing nested magnetic flux surfaces.

### 72. `radial_coordinate_of_strike_point` — **INCORRECT**

- source path: `summary/boundary/strike_point_outer_r/value`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: R position of the outer strike point
- name description: Major-radius location of an individual magnetic strike point where a separatrix leg intersects a divertor target, expressed in the right-handed cylindrical (R, φ, Z) frame.
- **rejected spelling** `radial_coordinate_of_strike_point` → **proposed spelling** `radial_coordinate_of_outer_strike_point`
- why: This one name is bound to **two different DD paths**: `summary/boundary/strike_point_inner_r/value` and `summary/boundary/strike_point_outer_r/value` (and `vertical_coordinate_of_strike_point` likewise covers both `_inner_z` and `_outer_z`). The inner and outer strike points are physically distinct locations on opposite divertor legs, with different heat flux, different geometry and different control significance. A published catalog entry that resolves to either is not a standard name — the pair must be `radial_coordinate_of_inner_strike_point` and `radial_coordinate_of_outer_strike_point`.

### 73. `radial_coordinate_of_x_point` — **INCORRECT**

- source path: `summary/boundary/x_point_main/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate locating the main X-point, the poloidal magnetic-field null that defines the separatrix in an equilibrium.
- **rejected spelling** `radial_coordinate_of_x_point` → **proposed spelling** `radial_coordinate_of_primary_x_point`
- why: The two coordinates of one DD container disagree with each other: `summary/boundary/x_point_main/r` is `radial_coordinate_of_x_point` while `summary/boundary/x_point_main/z` is `vertical_coordinate_of_**primary**_x_point`. One container, one locus, two spellings — so at most one is right, and the DD's own qualifier (`_main`) says the qualified spelling is. An unqualified `x_point` is also wrong on its own terms in a double-null-capable machine description, where secondary X-points exist.

### 74. `total_neutral_source_rate_due_to_gas_injection` — **correct**

- source path: `summary/gas_injection_rates/total/value`
- unit: `s^-1` (data dictionary: `s^-1`)
- data-dictionary text: Total gas injection rate (sum over species)
- name description: Instantaneous equivalent-electron source rate of neutral gas introduced into the vessel by gas injection, summed over all injected species.

### 75. `normalized_toroidal_beta` — **correct**

- source path: `summary/global_quantities/beta_tor_norm_mhd/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Normalised toroidal beta, using the pressure determined by an equilibrium reconstruction code
- name description: Normalized toroidal beta is a dimensionless whole-plasma equilibrium measure formed from volume-averaged total perpendicular pressure and toroidal magnetic and plasma-current scales.

### 76. `plasma_current` — **correct**

- source path: `summary/global_quantities/ip/value`
- unit: `A` (data dictionary: `A`)
- data-dictionary text: Total plasma current
- name description: Net toroidal electric current carried by the entire plasma column, obtained by integrating toroidal current density over its enclosed poloidal cross-section.

### 77. `total_plasma_radiated_power` — **correct**

- source path: `summary/global_quantities/power_radiated/value`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: Radiated power
- name description: Total plasma radiated power is the electromagnetic power emitted by the full plasma, summed over all photon-emission mechanisms and integrated over the plasma volume.

### 78. `reference_major_radius` — **correct**

- source path: `summary/global_quantities/r0/value`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Reference major radius where the vacuum toroidal magnetic field is given (usually a fixed position such as the middle of the vessel at the equatorial midplane)
- name description: Nonnegative perpendicular distance from the toroidal symmetry axis to a designated reference location, serving as the major-radius coordinate where the vacuum toroidal magnetic field is specified in the right-handed cylindrical (R, φ, Z) frame.

### 79. `volume_of_plasma_boundary` — **correct**

- source path: `summary/global_quantities/volume/value`
- unit: `m^3` (data dictionary: `m^3`)
- data-dictionary text: Volume of the confined plasma
- name description: Volume enclosed by the plasma boundary, representing the total confined-plasma region inside the last closed magnetic flux surface.
- note: Same enclosed-volume convention as `volume_of_flux_surface`; consistent with the family.

### 80. `total_power_due_to_ion_cyclotron_heating` — **INCORRECT**

- source path: `summary/heating_current_drive/power_ic/value`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: Total IC power coupled to the plasma
- name description: Total ion-cyclotron radio-frequency power launched into the vacuum vessel by the complete heating-antenna system, summed over all antennas.
- **rejected spelling** `total_power_due_to_ion_cyclotron_heating` → **proposed spelling** `total_coupled_power_due_to_ion_cyclotron_heating`
- why: The DD text says "Total IC power **coupled to the plasma**", while the name's description says "launched into the vacuum vessel" — those are separated by the coupling efficiency and differ by the power reflected or dissipated in the antenna and vessel structure. The cohort's sibling `net_power_due_to_ion_cyclotron_heating` is bound to `ic_antennas/antenna/power_launched` and genuinely is launched power, so the two names are one word apart from being indistinguishable while denoting quantities that are routinely 10-30 % apart.

### 81. `line_averaged_effective_charge` — **correct**

- source path: `summary/line_average/zeff/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Effective charge
- name description: Dimensionless line-averaged effective ionic charge of a plasma mixture, obtained by averaging local effective charge along a prescribed plasma line of sight.

### 82. `toroidal_magnetic_field_at_magnetic_axis` — **correct**

- source path: `summary/local/magnetic_axis/b_field_tor/value`
- unit: `T` (data dictionary: `T`)
- data-dictionary text: Toroidal magnetic field
- name description: Toroidal magnetic field at the magnetic axis is the signed toroidal component of the total equilibrium magnetic field at the magnetic axis in the right-handed cylindrical (R, φ, Z) frame. It includes externally applied vacuum-field and plasma-current-generated contributions.

### 83. `safety_factor_at_magnetic_axis` — **correct**

- source path: `summary/local/magnetic_axis/q/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Safety factor
- name description: Limiting signed field-line winding number on the innermost closed flux surface, giving toroidal turns per poloidal circuit at the magnetic axis.

### 84. `breakdown_initial_time` — **correct**

- source path: `summary/time_breakdown/value`
- unit: `s` (data dictionary: `s`)
- data-dictionary text: Time of the plasma breakdown
- name description: Timestamp at which plasma breakdown begins and discharge current starts to flow.

### 85. `radial_outline_of_limiter_tile` — **correct**

- source path: `wall/description_2d/limiter/unit/outline/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate R locating each point on the boundary outline of a plasma-facing limiter tile in the right-handed cylindrical (R, φ, Z) frame.

### 86. `vertical_outline_of_plasma_facing_component` — **correct**

- source path: `wall/description_2d/mobile/unit/outline/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: The vertical coordinate of each point on a plasma-facing component boundary outline is the signed height in the right-handed cylindrical (R, φ, Z) frame.

