# WEST batch accepted names — physical-correctness audit, cohort indices 1–51

provisional: false — every one of the 51 rows carries a verdict and the result section is closed.

The second half of the same audit as
[west-name-audit.md](/imas-codex/evidence/sn-west-catalog-release/west-name-audit), in the same
shape, so the two can be summed rather than compared. That file judged 86 rows drawn every-fourth
from the 341 accepted WEST bindings; this file judges **51 rows — indices 1 to 51 of the 255-row
remainder** in `west-name-cohort-remainder.json`, running from
`bremsstrahlung_visible/channel/filter/wavelength_lower` to
`equilibrium/time_slice/boundary/geometric_axis/r`.

Each row carries `name`, `path`, `sn_unit`, `sn_description`, `dd_unit`, `dd_doc` and
`dd_doc_parent`, all already drawn from the live graph. **No graph query was issued and no database
was opened for this file** — the rows are the whole evidence base, which is what makes the
judgement reproducible from the JSON alone.

Three questions per row: does the data-dictionary text of the bound path describe the quantity the
name claims; do `sn_unit` and `dd_unit` agree and if not which side is defensible; and is the name
self-descriptive to a reader without the source path in hand. A defect that lies in the
**description** rather than in the name is recorded as a note on a **correct** row and counted
separately — it does not carry an incorrect verdict.

**Units agree on all 51 rows** — `sn_unit == dd_unit` for every binding in this range, so no row
turns on a unit disagreement and that axis contributes nothing to the verdicts below.

A uniform column is a claim about the instrument before it is a result, so the comparison was
controlled against something known present rather than reported as an absence. Run over all 255
remainder rows the same comparison finds **three** disagreements — indices 152, 159 and 198
(`turn_count_of_toroidal_magnetic_field_probe`, `turn_count_of_poloidal_magnetic_field_probe`,
`atomic_count`, each carrying `1` against an empty `dd_unit`) — all of them outside indices 1–51.
So the comparison can see a disagreement and this range genuinely has none. Fourteen distinct units
appear across the 51 rows (`1`, `J`, `K`, `Pa`, `T`, `Wb`, `eV`, `kg.s^-1`, `m`, `m^-3`,
`m^-3.s^-1.sr^-1`, `m^2`, `rad`, `s`), so the column is not uniform either.

**Spellings the prior half settled are reused, not re-minted**: the `*_of_flux_surface` family
including `volume_of_flux_surface` and `area_of_flux_surface`,
`back_surface_distance_of_antenna_strap`, and the etendue `_detector` spelling. None of those
adjudications is relitigated here.

## Verdicts

### 1. `lower_wavelength_of_filter` — **correct**

- source path: `bremsstrahlung_visible/channel/filter/wavelength_lower`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Lower bound of the filter wavelength range
- name description: Lower wavelength of a filter is the spectral coordinate, expressed as vacuum wavelength, that identifies the short-wavelength boundary of a diagnostic filter's transmitted spectral band.

### 2. `upper_wavelength_of_filter` — **correct**

- source path: `bremsstrahlung_visible/channel/filter/wavelength_upper`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Upper bound of the filter wavelength range
- name description: Upper wavelength of a filter is the vacuum-wavelength coordinate marking the long-wavelength, low-frequency edge of its transmitted spectral band.

### 3. `toroidal_coordinate_of_line_of_sight` — **correct**

- source path: `bremsstrahlung_visible/channel/line_of_sight/first_point/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of the first reference point on a diagnostic line of sight, locating that point around the machine symmetry axis.
- collision, deferred: this identity is bound to 14 source paths, 10 of them outside indices 1–51 (`camera_x_rays`, `interferometer`, `polarimeter`, `soft_x_rays`, `spectrometer_visible`). The collision itself belongs to the whole-cohort collision sweep, not to this file.

### 4. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `bremsstrahlung_visible/channel/line_of_sight/first_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.
- collision, deferred: 16 source paths, 14 outside this index range. The description here is correctly locus-neutral ("a designated point"), which is what row 5's is not.

### 5. `toroidal_coordinate_of_line_of_sight` — **correct**

- source path: `bremsstrahlung_visible/channel/line_of_sight/second_point/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of the first reference point on a diagnostic line of sight, locating that point around the machine symmetry axis.
- note: **the defect is in the description, not the name.** The name is correct for any point on a line of sight, and this binding is the **second** point, which the description calls "the first reference point". The prior half recorded the same defect on its row 46; this row makes it demonstrably false rather than merely narrow. The remedy is to make the description locus-neutral as row 4's already is, not to rename.

### 6. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `bremsstrahlung_visible/channel/line_of_sight/second_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 7. `spectral_bremsstrahlung_photon_radiance` — **correct**

- source path: `bremsstrahlung_visible/channel/radiance_spectral`
- unit: `m^-3.s^-1.sr^-1` (data dictionary: `m^-3.s^-1.sr^-1`)
- data-dictionary text: Calibrated spectral radiance (radiance per unit wavelength)
- name description: Spectrally resolved photon-number radiance of free-free bremsstrahlung, giving the directional photon emission rate per projected area and wavelength interval.
- The name is more precise than the data dictionary and correctly so: the unit `m^-3.s^-1.sr^-1` is a photon-number rate per area per solid angle per wavelength, not an energy radiance, so `photon_radiance` is what the dimensions say and `radiance` alone would not be.

### 8. `line_averaged_effective_charge` — **correct**

- source path: `bremsstrahlung_visible/channel/zeff_line_average`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Average effective charge along the line of sight
- name description: Dimensionless line-averaged effective ionic charge of a plasma mixture, obtained by averaging local effective charge along a prescribed plasma line of sight.

### 9. `coolant_mass_flow_rate` — **correct**

- source path: `calorimetry/cooling_loop/mass_flow`
- unit: `kg.s^-1` (data dictionary: `kg.s^-1`)
- data-dictionary text: Mass flow of the coolant going through the loop
- name description: Coolant mass throughput across a fluid-flow boundary, representing the rate at which coolant mass passes through a port or cooling loop.
- This identity is also bound to row 12 (`calorimetry/group/component/mass_flow`), both inside this range. It is the same physical quantity at two hardware loci of the same kind, which the prior half already settled as acceptable by accepting `coolant_temperature_at_inlet` across the loop and the component. Not a collision of the strike-point class.

### 10. `coolant_temperature_at_outlet` — **correct**

- source path: `calorimetry/cooling_loop/temperature_out`
- unit: `K` (data dictionary: `K`)
- data-dictionary text: Temperature of the coolant when exiting the loop
- name description: Absolute thermodynamic temperature of coolant leaving a cooling component or loop, specifying the fluid's outlet thermal state for enthalpy-based heat balance.

### 11. `accumulated_coolant_absorbed_energy_of_calorimetry_component` — **correct**

- source path: `calorimetry/group/component/energy_cumulated`
- unit: `J` (data dictionary: `J`)
- data-dictionary text: Energy extracted from the component since the start of the pulse
- name description: Cumulative thermal energy transferred from a calorimetry component to its coolant since pulse start, determined from the coolant enthalpy gain.
- `accumulated` carries the "since the start of the pulse" clause that distinguishes this from an instantaneous power, and `calorimetry_component` is the same locus spelling the prior half accepted on `coolant_absorbed_power_of_calorimetry_component`.

### 12. `coolant_mass_flow_rate` — **correct**

- source path: `calorimetry/group/component/mass_flow`
- unit: `kg.s^-1` (data dictionary: `kg.s^-1`)
- data-dictionary text: Mass flow of the coolant going through the component
- name description: Coolant mass throughput across a fluid-flow boundary, representing the rate at which coolant mass passes through a port or cooling loop.

### 13. `coolant_temperature_at_inlet` — **correct**

- source path: `calorimetry/group/component/temperature_in`
- unit: `K` (data dictionary: `K`)
- data-dictionary text: Temperature of the coolant when entering the component
- name description: Absolute thermodynamic temperature of coolant entering a cooling loop or plant component, establishing the upstream state for enthalpy-based heat-transfer balances.

### 14. `coolant_temperature_at_outlet` — **correct**

- source path: `calorimetry/group/component/temperature_out`
- unit: `K` (data dictionary: `K`)
- data-dictionary text: Temperature of the coolant when exiting the component
- name description: Absolute thermodynamic temperature of coolant leaving a cooling component or loop, specifying the fluid's outlet thermal state for enthalpy-based heat balance.

### 15. `coolant_transit_time_of_plant_component_port` — **INCORRECT**

- source path: `calorimetry/group/component/transit_time`
- unit: `s` (data dictionary: `s`)
- data-dictionary text: Transit time for the coolant to go from the input to the output of the component
- name description: Elapsed time for coolant to traverse the associated plant component from inlet to outlet, defining the thermal delay between the two coolant states.

| rejected | proposed |
| --- | --- |
| `coolant_transit_time_of_plant_component_port` | `coolant_transit_time_of_calorimetry_component` |

The name attributes the quantity to a **port**; the data dictionary and the name's own description
both attribute it to the **component**. A port is an opening through which coolant enters or
leaves, and a coolant residence time is not a property of an opening — it is the time to traverse
the flow path between two openings, which is a property of what lies between them. The name's own
description says "traverse the associated plant component from inlet to outlet", so the name
contradicts the text that ships with it. `calorimetry_component` is also the locus spelling this
cohort already uses twice, at row 11 and at the prior half's
`coolant_absorbed_power_of_calorimetry_component`, so the fix reuses a settled spelling rather than
minting one.

### 16. `toroidal_coordinate_of_aperture` — **correct**

- source path: `camera_x_rays/aperture/centre/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Azimuthal position of an aperture's geometric center around the machine symmetry axis, measured as the toroidal angle in the right-handed (R, φ, Z) frame.
- This is the correct member of the split the prior half flagged as its finding 2. Rows 17 and 18 are the other two coordinates of this same point and they name a different object; this row is the one to keep.

### 17. `radial_coordinate_of_measurement_position` — **INCORRECT**

- source path: `camera_x_rays/aperture/centre/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate locating a measurement position by perpendicular distance from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

| rejected | proposed |
| --- | --- |
| `radial_coordinate_of_measurement_position` | `radial_coordinate_of_aperture` |

An aperture centre is a point on the **instrument**: a machined opening whose position is fixed by
the hardware. A measurement position is where in the **plasma** a value was measured. The two are
different loci, separated by the whole optical path, and a reader mapping the catalog onto a
geometry would place a plasma sample point at the pinhole. This is not a new finding — the prior
half named it as its finding 1 and could not judge it because the row fell outside the
every-fourth draw. It falls inside this range, so it is judged here.

The name is also self-contradicting inside its own data-dictionary container: row 16 spells the φ
of this identical point `toroidal_coordinate_of_aperture`. A catalog cannot publish one point whose
φ belongs to an aperture and whose R belongs to a measurement position.

### 18. `vertical_coordinate_of_measurement_position` — **INCORRECT**

- source path: `camera_x_rays/aperture/centre/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a generic measurement location, measured from the machine midplane along the Z direction of the right-handed cylindrical (R, φ, Z) frame.

| rejected | proposed |
| --- | --- |
| `vertical_coordinate_of_measurement_position` | `vertical_coordinate_of_aperture` |

The Z half of row 17, with the same distinction: the DD object is the vertical position of an
aperture's geometric centre, and the name asserts a generic plasma measurement location. Together
rows 16–18 are one point carrying two loci across three coordinates.

### 19. `radial_coordinate_of_camera` — **correct**

- source path: `camera_x_rays/camera/centre/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius location of a camera's geometric center, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 20. `vertical_coordinate_of_camera` — **correct**

- source path: `camera_x_rays/camera/centre/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of the camera's geometric center, measured along Z in the right-handed cylindrical (R, φ, Z) frame.

### 21. `toroidal_coordinate_of_line_of_sight` — **correct**

- source path: `camera_x_rays/camera/line_of_sight/first_point/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of the first reference point on a diagnostic line of sight, locating that point around the machine symmetry axis.

### 22. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `camera_x_rays/camera/line_of_sight/first_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.

### 23. `toroidal_coordinate_of_line_of_sight` — **correct**

- source path: `camera_x_rays/camera/line_of_sight/second_point/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of the first reference point on a diagnostic line of sight, locating that point around the machine symmetry axis.
- note: **the defect is in the description, not the name** — the second occurrence of row 5's note, on a different IDS. The identity is bound to first, second and third points across the cohort; its description names only the first. One description fix serves all 14 bindings.

### 24. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `camera_x_rays/camera/line_of_sight/second_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 25. `toroidal_coordinate_at_detector_pixel` — **INCORRECT**

- source path: `camera_x_rays/camera/pixel_position/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Azimuthal coordinate locating the geometric center of each detector pixel in the right-handed cylindrical (R, φ, Z) frame.

| rejected | proposed |
| --- | --- |
| `toroidal_coordinate_at_detector_pixel` | `toroidal_coordinate_of_detector_pixel` |

The cohort uses the two prepositions for two different things, and this row uses the wrong one.
`at_<locus>` marks a **field evaluated at** a named locus — `poloidal_magnetic_flux_at_plasma_boundary`
(row 36), `poloidal_magnetic_flux_at_measurement_position` (row 45),
`radiative_temperature_at_magnetic_axis` (row 49). `of_<object>` marks a **coordinate that locates**
the object itself — `radial_coordinate_of_camera` (row 19), `radial_coordinate_of_detector_pixel`
(row 26), `vertical_coordinate_of_detector_pixel` (row 27). A pixel's own φ locates the pixel, so it
takes `of_`, which is exactly what its two siblings do. As spelled, the three coordinates of one
pixel centre read as two different kinds of quantity.

### 26. `radial_coordinate_of_detector_pixel` — **correct**

- source path: `camera_x_rays/camera/pixel_position/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a detector pixel's geometric center, locating the pixel relative to the machine symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 27. `vertical_coordinate_of_detector_pixel` — **correct**

- source path: `camera_x_rays/camera/pixel_position/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Vertical coordinate locating the geometric center of an individual detector pixel in the right-handed cylindrical (R, φ, Z) frame.

### 28. `temperature_of_soft_xray_detector` — **INCORRECT**

- source path: `camera_x_rays/detector_temperature`
- unit: `K` (data dictionary: `K`)
- data-dictionary text: Temperature measured at the detector level
- name description: Thermodynamic temperature of the soft X-ray detector assembly, describing its physical thermal state for calibration and monitoring rather than a plasma temperature inferred from radiation.

| rejected | proposed |
| --- | --- |
| `temperature_of_soft_xray_detector` | `temperature_of_x_ray_detector` |

Two defects, one dominant. The **band assertion is unsupported**: the path is in `camera_x_rays`,
an X-ray imaging camera IDS that names no spectral band, while `soft_x_rays` and `hard_x_rays` are
separate IDSs that do. Calling this detector a soft X-ray detector asserts a band membership
nothing in the path or its documentation carries, and it would collide with a genuine
`soft_x_rays` detector temperature if one were ever named. Second, `xray` is the minority spelling
the prior half already rejected on `hard_xray_brightness`; the fix adopts `x_ray` in the same move.

### 29. `thickness_of_filter` — **correct**

- source path: `camera_x_rays/filter_window/thickness`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Thickness of the filter window
- name description: Geometric thickness of one diagnostic filter window, determining the amount of material available for photon attenuation and spectral selection.
- The DD container is a filter *window* and the name says *filter*; that is the same functional object under its two names, and `_of_filter` is the spelling rows 1 and 2 already use.

### 30. `effective_charge` — **INCORRECT**

- source path: `core_profiles/global_quantities/z_eff_resistive`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Volume average plasma effective charge, estimated from the flux consumption in the ohmic phase
- name description: Dimensionless effective ionic charge of a plasma mixture, given by the second ionic charge moment divided by free-electron density and indicating impurity content.

| rejected | proposed |
| --- | --- |
| `effective_charge` | `volume_averaged_effective_charge` |

**One name, two physically different quantities.** The bare identity `effective_charge` is bound
both here and at row 40 (`core_profiles/profiles_1d/zeff`). Row 40 is the local profile value of
Zeff at a flux-surface label — a function of radius. This row is a **single global scalar**: the
volume average over the whole plasma, and moreover one inferred from ohmic flux consumption rather
than measured per surface. They have the same units and the same physical dimension and they are
not the same quantity; a consumer resolving `effective_charge` cannot tell which it is holding.

Row 40 keeps the bare name, which is right for the local quantity. This row takes the averaging
operator into the name, parallel to `line_averaged_effective_charge` at row 8 — the cohort already
distinguishes an averaged Zeff from a local one by exactly this construction.

The resistive-flux-consumption provenance deliberately does **not** enter the name. Provenance is
a controlled vocabulary carried on the source binding, not a name segment, so
`volume_averaged_effective_charge` is the full extent of the naming fix and the "estimated from
flux consumption" clause belongs to the binding's provenance field.

### 31. `thermal_electron_density` — **correct**

- source path: `core_profiles/profiles_1d/electrons/density_thermal`
- unit: `m^-3` (data dictionary: `m^-3`)
- data-dictionary text: Density of thermal particles
- name description: Number density of free electrons per physical volume for the local thermal electron population, excluding fast and suprathermal electron contributions.
- The DD text says only "thermal particles"; the electron species comes from the container, and the name carries it — which is what makes it self-descriptive without the path.

### 32. `total_electron_pressure` — **correct**

- source path: `core_profiles/profiles_1d/electrons/pressure`
- unit: `Pa` (data dictionary: `Pa`)
- data-dictionary text: Pressure (thermal+non-thermal)
- name description: Kinetic pressure associated with the entire electron population, combining thermal and suprathermal contributions in the isotropic electron stress.
- `total` is doing real work here: it is what separates this row from row 33's thermal-only pressure in the same container, and the DD's "(thermal+non-thermal)" is exactly that distinction.

### 33. `thermal_electron_pressure_at_post_sawtooth_crash` — **INCORRECT**

- source path: `core_profiles/profiles_1d/electrons/pressure_thermal`
- unit: `Pa` (data dictionary: `Pa`)
- data-dictionary text: Pressure (thermal) associated with random motion ~average((v-average(v))^2)
- name description: Isotropic kinetic pressure produced by random motion of bulk thermal electrons, evaluated in the plasma state immediately after a sawtooth crash.

| rejected | proposed |
| --- | --- |
| `thermal_electron_pressure_at_post_sawtooth_crash` | `thermal_electron_pressure` |

**The name asserts more than the data supports, and the excess is a discharge phase.** Nothing in
the path, the data-dictionary text or the container mentions a sawtooth. `pressure_thermal` in
`core_profiles/profiles_1d/electrons` is the thermal electron pressure profile at whatever time
slice the record holds — most of which are not after a sawtooth crash, and many discharges have no
sawteeth at all.

The consequence is not cosmetic. As spelled, every ordinary profile sample is either unnameable or
silently mislabelled as post-crash, and an analysis selecting on the name would build a
sawtooth-conditioned dataset out of unconditioned data. The bare `thermal_electron_pressure` is
true of every binding, and it is the exact counterpart of row 32's `total_electron_pressure` in the
same container — the pair then reads as the thermal/total split the DD actually makes.

### 34. `poloidal_plane_cross_sectional_area_of_flux_surface` — **correct**

- source path: `core_profiles/profiles_1d/grid/area`
- unit: `m^2` (data dictionary: `m^2`)
- data-dictionary text: Cross-sectional area of the flux surface
- name description: Geometric area enclosed by a closed magnetic-flux-surface contour in a fixed-toroidal-angle poloidal plane of the right-handed cylindrical (R, φ, Z) frame.
- The long spelling is load-bearing rather than verbose, and this is worth stating because it looks at first like a parallel minting of the settled `area_of_flux_surface`. The cohort carries **two different areas of the same surface**: this poloidal cross-section, and `surface_area_of_flux_surface` (remainder index 103, `equilibrium/time_slice/profiles_1d/surface`), which is the area of the toroidal surface itself. They differ by roughly a factor of 2πR and are not interchangeable. A bare `area_of_flux_surface` on this row would lose the distinction the cohort has correctly made.

### 35. `poloidal_magnetic_flux_at_flux_surface` — **correct**

- source path: `core_profiles/profiles_1d/grid/psi`
- unit: `Wb` (data dictionary: `Wb`)
- data-dictionary text: Poloidal magnetic flux
- name description: Signed poloidal magnetic flux assigned to a nested magnetic surface, serving as the equilibrium label for its position in the plasma.

### 36. `poloidal_magnetic_flux_at_plasma_boundary` — **correct**

- source path: `core_profiles/profiles_1d/grid/psi_boundary`
- unit: `Wb` (data dictionary: `Wb`)
- data-dictionary text: Value of the poloidal magnetic flux at the plasma boundary (useful to normalize the psi array values when the radial grid doesn't go from the magnetic axis to the plasma boundary)
- name description: Signed poloidal magnetic flux evaluated on the last closed flux surface, providing the outer reference for normalized poloidal-flux coordinates.

### 37. `normalized_poloidal_flux_coordinate` — **correct**

- source path: `core_profiles/profiles_1d/grid/rho_pol_norm`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Normalised poloidal flux coordinate = sqrt((psi(rho)-psi(magnetic_axis) / (psi(LCFS)-psi(magnetic_axis)))
- name description: Normalized poloidal flux coordinate is a radial label for nested magnetic flux surfaces based on poloidal magnetic flux. It is zero at the magnetic axis and one at the last closed flux surface; for a line-of-sight locus, it denotes the minimum value reached along the line.
- note: **a description observation, not a name defect.** The description's closing clause defines a convention for a line-of-sight locus, and no binding of this identity inside indices 1–51 is a line of sight. The clause is harmless where it sits — it is a conditional, not a claim about this binding — but it is the same shape as rows 5 and 23: a description written for one locus travelling with an identity bound to others. Whether a line-of-sight binding exists elsewhere in the cohort is a whole-cohort question and is not answered here.

### 38. `toroidal_flux_coordinate` — **correct**

- source path: `core_profiles/profiles_1d/grid/rho_tor`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Toroidal flux coordinate. rho_tor = sqrt(b_flux_tor/(pi*b0)) ~ sqrt(pi*r^2*b0/(pi*b0)) ~ r [m]. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0
- name description: Non-negative, radius-like label of a nested magnetic flux surface, derived from enclosed toroidal magnetic flux using a positive reference vacuum toroidal field.
- The unit `m` is correct and is what separates this from row 39: `rho_tor` carries a length dimension, its normalized counterpart does not.

### 39. `normalized_toroidal_flux_coordinate` — **correct**

- source path: `core_profiles/profiles_1d/grid/rho_tor_norm`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm, is the toroidal flux coordinate at the equilibrium boundary (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation, see time_slice/boundary/b_flux_pol_norm in the equilibrium IDS)
- name description: Dimensionless radial label equal to the square root of toroidal magnetic flux normalized between the magnetic axis and equilibrium boundary.
- collision, deferred: 3 source paths, 2 outside this index range.

### 40. `effective_charge` — **correct**

- source path: `core_profiles/profiles_1d/zeff`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Effective charge
- name description: Dimensionless effective ionic charge of a plasma mixture, given by the second ionic charge moment divided by free-electron density and indicating impurity content.
- This is the binding the bare name is right for: the local profile quantity, which is what the description describes. Row 30 is the one that must move.

### 41. `toroidal_vacuum_magnetic_field` — **correct**

- source path: `core_profiles/vacuum_toroidal_field/b0`
- unit: `T` (data dictionary: `T`)
- data-dictionary text: Vacuum toroidal field at R0 [T]; Positive sign means anti-clockwise when viewing from above. The product R0B0 must be consistent with the b_tor_vacuum_r field of the tf IDS.
- name description: Signed toroidal component of the current-free vacuum magnetic field at a reference major radius, defining the nominal externally generated field strength.
- collision, deferred: 3 source paths, 2 outside this index range.

### 42. `reference_major_radius` — **correct**

- source path: `core_profiles/vacuum_toroidal_field/r0`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Reference major radius where the vacuum toroidal magnetic field is given (usually a fixed position such as the middle of the vessel at the equatorial midplane)
- name description: Nonnegative perpendicular distance from the toroidal symmetry axis to a designated reference location, serving as the major-radius coordinate where the vacuum toroidal magnetic field is specified in the right-handed cylindrical (R, φ, Z) frame.
- collision, deferred: 3 source paths, 2 outside this index range.

### 43. `opacity_at_ece_channel_emission_position` — **INCORRECT**

- source path: `ece/channel/optical_depth`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Optical depth of the plasma at the position of the measurement. This parameter is a proxy for the local / non-local character of the ECE emission. It must be greater than 1 to guarantee that the measurement is dominated by local ECE emission (non-local otherwise)
- name description: Dimensionless optical depth along the electron-cyclotron-emission viewing path at a channel emission position, indicating whether the detected emission is local and thermal.

| rejected | proposed |
| --- | --- |
| `opacity_at_ece_channel_emission_position` | `optical_depth_at_ece_channel_emission_position` |

**The name denotes a different physical quantity from the one it is bound to, and the unit settles
it.** Opacity is a material property — the mass absorption coefficient κ, carrying units of
m²·kg⁻¹. Optical depth is the dimensionless line integral of the absorption coefficient along a
ray, τ = ∫κρ ds. The DD text says optical depth, the name's own description says optical depth, and
the unit is `1`, which opacity cannot be. Only the name says opacity.

The locus half of the name is correct and is kept. Note that the mismatch here is in the
**quantity**, not in the locus, which is what distinguishes this from rows 17, 18 and 48 even
though the result table groups them together — the grouping is by remedy, and both remedies are a
rename to the object the data dictionary names.

### 44. `toroidal_angle_of_measurement_position` — **INCORRECT**

- source path: `ece/channel/position/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate locating a measurement position around the machine symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

| rejected | proposed |
| --- | --- |
| `toroidal_angle_of_measurement_position` | `toroidal_coordinate_of_measurement_position` |

**Minority spelling of a base the cohort already fixes.** The locus is right — the prior half
established that `ece/channel/position/*` is a genuine plasma measurement position, unlike the
aperture centre of rows 17 and 18. What is wrong is the base: within indices 1–51 alone, six
bindings spell the toroidal angle `toroidal_coordinate_*` (rows 3, 5, 16, 21, 23, 25) and this one
row spells it `toroidal_angle_*`, for the same DD text and the same unit.

`toroidal_angle` is arguably the more precise English, and that is the argument that must not win
here: the cohort has already converged, and a catalog carrying both makes the φ of an ECE
measurement position and the φ of an aperture centre read as different quantities. Row 47 keeps
`poloidal_angle_` for the same reason in reverse — there is no competing `poloidal_coordinate_`
spelling anywhere in the cohort, so `angle` is the settled form for θ and the minority form for φ.

### 45. `poloidal_magnetic_flux_at_measurement_position` — **correct**

- source path: `ece/channel/position/psi`
- unit: `Wb` (data dictionary: `Wb`)
- data-dictionary text: Poloidal flux
- name description: Signed equilibrium poloidal magnetic-flux function evaluated at a diagnostic measurement location, identifying the nested magnetic surface intersecting that location.
- Correct use of `at_`: this is a field evaluated at a locus, not a coordinate locating an object — the distinction row 25 gets wrong.

### 46. `normalized_toroidal_flux_coordinate_at_measurement_position` — **correct**

- source path: `ece/channel/position/rho_tor_norm`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Normalised toroidal flux coordinate
- name description: Dimensionless normalized toroidal-flux label that maps a physical measurement position onto a nested magnetic surface between the magnetic axis and equilibrium boundary.
- collision, deferred: 2 source paths, 1 outside this index range.

### 47. `poloidal_angle_of_measurement_position` — **correct**

- source path: `ece/channel/position/theta`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Poloidal angle (oriented clockwise when viewing the poloidal cross section on the right hand side of the tokamak axis of symmetry, with the origin placed on the plasma magnetic axis)
- name description: Geometric poloidal angle locating a measurement position around the magnetic axis in the right-handed cylindrical (R, φ, Z) frame.
- The description's "increasing clockwise" matches the DD's orientation clause, which is the half of a poloidal-angle definition most often dropped.

### 48. `vertical_coordinate_of_ece_channel` — **INCORRECT**

- source path: `ece/channel/position/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of an electron-cyclotron-emission channel's measurement position in the right-handed cylindrical (R, φ, Z) frame.

| rejected | proposed |
| --- | --- |
| `vertical_coordinate_of_ece_channel` | `vertical_coordinate_of_measurement_position` |

**Bound to the wrong object, and it breaks a coordinate set its four siblings share.** An ECE
channel is a receiver: a piece of hardware outside the vessel at a fixed position. The DD container
`ece/channel/position` is not where the channel is — it is where in the plasma the emission that
channel detects originates, which moves with the magnetic field and the density and is different
on every time slice. Naming it the channel's own vertical coordinate asserts that a diagnostic
receiver is somewhere inside the plasma.

The name's own description gives the game away: it says "channel's measurement position", which is
the right object under the wrong name. Its four siblings in the same container all name that
object — `toroidal_angle_of_measurement_position` (row 44, itself a minority spelling),
`poloidal_magnetic_flux_at_measurement_position` (45),
`normalized_toroidal_flux_coordinate_at_measurement_position` (46),
`poloidal_angle_of_measurement_position` (47) — and the prior half judged
`radial_coordinate_of_measurement_position` on `ece/channel/position/r` correct. So five of the six
coordinates of this position name the position and one names the instrument.

### 49. `radiative_temperature_at_magnetic_axis` — **INCORRECT**

- source path: `ece/t_radiation_central`
- unit: `eV` (data dictionary: `eV`)
- data-dictionary text: Radiation temperature from the closest channel to the magnetic axis, together with its radial location
- name description: Temperature-equivalent radiation energy scale of electron-cyclotron emission, evaluated at the magnetic axis, the degenerate innermost magnetic surface where the poloidal magnetic-flux gradient vanishes. It is a brightness-temperature quantity, not necessarily the local kinetic electron temperature.

| rejected | proposed |
| --- | --- |
| `radiative_temperature_at_magnetic_axis` | `radiative_temperature_at_innermost_ece_channel` |

**The name asserts more than the data supports.** The data dictionary is explicit: this is the
radiation temperature *from the closest channel to* the magnetic axis. The closest channel is not
the axis. The DD path is `t_radiation_central` — "central", which is a description of where the
channel sits, not a claim that it sits on a defined equilibrium locus.

The data dictionary itself treats the offset as material: it ships the channel's **radial
location** alongside the value, in the same sentence, which it would not need to do if the value
were at the axis. The magnetic axis moves during a discharge while the channel set is fixed, so the
offset is not a constant and cannot be calibrated away by a reader.

The cohort does have genuine `*_at_magnetic_axis` names — the prior half accepted
`toroidal_magnetic_field_at_magnetic_axis` and `safety_factor_at_magnetic_axis`, where the DD does
say magnetic axis. Keeping this row in that family would make the family unreliable: a consumer
selecting on `_at_magnetic_axis` would get two quantities evaluated on the axis and one measured
near it.

### 50. `elongation_of_plasma_boundary` — **correct**

- source path: `equilibrium/time_slice/boundary/elongation`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Elongation of the plasma boundary
- name description: Dimensionless ratio of the plasma boundary's vertical half-height to its minor radius, quantifying the elongation of its cross-sectional shape.
- The surface is explicit in the name, which is what keeps it distinct from `elongation_of_flux_surface` (remainder index 87, the per-surface profile). A bare `elongation` would merge a boundary scalar with a radial profile.
- collision, deferred: 2 source paths, 1 outside this index range.

### 51. `radial_coordinate_of_geometric_axis` — **correct**

- source path: `equilibrium/time_slice/boundary/geometric_axis/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate locating the midpoint of the plasma boundary's radial extrema, thereby specifying the horizontal position of its geometric axis.
- collision, deferred: 2 source paths, 1 outside this index range.

## Result

| | count |
| --- | --- |
| rows judged (remainder indices 1–51, contiguous) | **51** |
| judged **correct** | 40 |
| judged **INCORRECT**, each with a proposed spelling | **11** |
| correct + incorrect | **51** |
| description-only defects, recorded as notes on correct rows | 3 |
| rows whose identity is also bound outside this index range, deferred | 13 |
| rows where `sn_unit` and `dd_unit` disagree | 0 |

**11 of 51 — 21.6 % of this range — are not publishable as spelled.** Grouped into the five classes
the first half established:

- **The name asserts more than the data supports** (3): `thermal_electron_pressure_at_post_sawtooth_crash`
  for an unconditioned profile (33); `radiative_temperature_at_magnetic_axis` for the nearest
  channel to the axis (49); `temperature_of_soft_xray_detector` for an imaging camera that names no
  band (28).
- **The name is bound to the wrong object** (5): `radial_coordinate_of_measurement_position` (17)
  and `vertical_coordinate_of_measurement_position` (18) on an aperture centre;
  `vertical_coordinate_of_ece_channel` (48) on a plasma emission position;
  `coolant_transit_time_of_plant_component_port` (15) on a component rather than a port; and
  `opacity_at_ece_channel_emission_position` (43), where the mismatch is in the quantity rather
  than the locus but the remedy is the same rename to what the data dictionary names.
- **One name, two physically different quantities** (1): `effective_charge` (30) across a local
  profile value and a volume-averaged scalar. As in the first half this is the most serious class,
  because no reader can repair it — only a split can.
- **Not self-descriptive** (0): none in this range.
- **Minority spelling of a base the cohort already fixes** (2):
  `toroidal_coordinate_at_detector_pixel` (25), `toroidal_angle_of_measurement_position` (44).

### The three description-only defects

None carries an incorrect verdict, and all three are the same shape — a description written for one
locus travelling with an identity bound to several:

1. Rows 5 and 23 — the shared line-of-sight identity's description names "the first reference
   point" while these bindings are second points. One description fix serves all 14 bindings.
2. Row 37 — `normalized_poloidal_flux_coordinate` carries a line-of-sight convention that no
   binding inside this range exercises.

### Why this range's rate is higher than the first half's 11.6 %

The two numbers measure different draws and should be summed with that stated, not averaged
blindly. The first half was **every fourth row**, which spreads across IDSs and almost never lands
two coordinates of the same point in the same sample. This range is **contiguous**, so it contains
whole coordinate triples and whole containers — and three of its eleven defects (17, 18, 48) are
precisely the kind that only becomes visible when you hold a full coordinate set at once, with two
of them being the rows the first half flagged as pointing outside its own sample.

So the higher rate is partly the block structure revealing clustered defects rather than a
different population. Combined, **137 rows of the 341 accepted bindings are now judged and 21 carry
a name defect — 15.3 %.**
