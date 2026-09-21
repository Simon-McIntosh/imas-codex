# WEST batch accepted names — physical-correctness audit, cohort indices 1–51

provisional: true — verdicts are appended as they are judged; the closing pass rewrites this line.

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

**Units agree on all 51 rows.** `sn_unit == dd_unit` for every binding in this range, so no row
turns on a unit disagreement and the axis contributes nothing to the verdicts below. That is a
result, not an omission: the prior half found exactly one unit disagreement in 86 rows
(`atomic_number`), so a second range with none is consistent rather than suspicious.

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
