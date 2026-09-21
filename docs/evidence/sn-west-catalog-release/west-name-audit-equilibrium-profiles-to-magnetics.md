# WEST batch accepted names — physical-correctness audit, equilibrium profiles to magnetics

provisional: false — all 51 rows carry a verdict and the result section is closed.

This is the **second half** of the audit begun in
[`west-name-audit.md`](west-name-audit.md). That file judged 86 of the 341
accepted bindings; this one judges the 51 whose `index` field lies in the
inclusive range **103 to 153** of the `bindings` list in
[`west-name-cohort-remainder.json`](west-name-cohort-remainder.json) — the
255 rows the every-fourth sample did not reach. The two halves use the same
section shape and the same verdict vocabulary so that their counts sum rather
than merely compare.

## Evidence base and what was not done

Every row was drawn from the live graph before this node began and is carried
in the JSON: `name`, `path`, `sn_unit`, `sn_description`, `dd_unit`, `dd_doc`
and `dd_doc_parent`. **No graph query was issued and no database was opened
here** — the JSON is the whole evidence base, by instruction. All 51 rows
carry a non-empty `dd_doc`; none needed the `dd_doc_parent` fallback, so the
parent field is null throughout this slice and that absence is a property of
the slice rather than a failed read.

Each name is judged on three questions:

1. Does the data-dictionary text of the path it is bound to describe the
   quantity the name claims?
2. Do `sn_unit` and `dd_unit` agree, and where they differ, which side is
   defensible?
3. Is the name self-descriptive to a reader who does not have the source path
   in hand?

Sections are numbered by **cohort index**, not by position in this file, so a
section number here is directly the row's `index` in the JSON and cannot be
confused with the first half's 1–86.

### Adjudications reused rather than relitigated

The first half settled several families, and this half reuses them instead of
minting parallel spellings: the `*_of_flux_surface` family including
`volume_of_flux_surface` and `area_of_flux_surface` (the name denotes the
quantity *of* the surface in the family's established sense),
`back_surface_distance_of_antenna_strap`, the etendue `_detector` spelling,
`lower_bound_photon_energy` (first half, row 40, accepted),
`vertical_coordinate_of_toroidal_magnetic_field_probe` (row 51, accepted),
`length_variation_of_interferometer_beam` (row 48, accepted),
`line_integrated_electron_density` as the proposed survivor of the
`electron_number_density` minority spelling (row 22), and
`hard_xray_photon_radiance` as the proposed survivor of `hard_xray_brightness`
(row 39). Where a row here repeats a defect the first half already adjudicated,
its proposed spelling is that adjudication's, not a new one.

## Verdicts

### 103. `surface_area_of_flux_surface` — **correct**

- source path: `equilibrium/time_slice/profiles_1d/surface`
- unit: `m^2` (data dictionary: `m^2`)
- data-dictionary text: Surface area of the toroidal flux surface
- name description: Geometric area of a closed toroidal magnetic flux surface formed by revolving its poloidal contour through one complete toroidal revolution in the right-handed cylindrical (R, φ, Z) frame.
- note: The same identity is accepted in the first half at `equilibrium/time_slice/global_quantities/surface` (row 27). The two paths are the profile-resolved and boundary-resolved instances of one quantity, so this is a legitimate shared identity rather than a collision; the whole-cohort collision sweep owns the decision on whether the batch publishes them as one name.

### 104. `lower_triangularity_of_flux_surface` — **correct**

- source path: `equilibrium/time_slice/profiles_1d/triangularity_lower`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Lower triangularity w.r.t. magnetic axis
- name description: Dimensionless lower triangularity shape parameter for a nested magnetic flux surface, given by the normalized inward radial displacement of its lower vertical extremum.
- The surface-explicit locus is the point of the name: the first half accepts `upper_triangularity_of_plasma_boundary` for the boundary path `equilibrium/time_slice/boundary/triangularity_upper` (row 20), and this row is the profile-resolved sibling. The two loci are physically different — one is a single scalar for the last closed surface, the other a profile over every nested surface — and the names distinguish them.

### 105. `upper_triangularity_of_flux_surface` — **correct**

- source path: `equilibrium/time_slice/profiles_1d/triangularity_upper`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Upper triangularity w.r.t. magnetic axis
- name description: Dimensionless upper triangularity shape parameter for a nested magnetic flux surface, given by the normalized inward radial displacement of its upper vertical extremum.

### 106. `toroidal_vacuum_magnetic_field` — **INCORRECT**

- source path: `equilibrium/vacuum_toroidal_field/b0`
- unit: `T` (data dictionary: `T`)
- data-dictionary text: Vacuum toroidal field at R0 [T]; Positive sign means anti-clockwise when viewing from above. The product R0B0 must be consistent with the b_tor_vacuum_r field of the tf IDS.
- name description: Signed toroidal component of the current-free vacuum magnetic field at a reference major radius, defining the nominal externally generated field strength.
- **rejected spelling** `toroidal_vacuum_magnetic_field` → **proposed spelling** `toroidal_vacuum_magnetic_field_at_reference_major_radius`
- why: The vacuum toroidal field is not one number — it falls as 1/R, so a value of it means nothing until the radius is given. The data-dictionary text says "at R0" and the name's own description has to supply "at a reference major radius", which is the same test the first half applied to `faraday_angle`: the description repairing an omission the name made. The omission is not cosmetic here, because the companion `reference_major_radius` (row 107) is published as a separate name and a consumer resolving one without the other gets a field strength with no location. The cohort already spells position-qualified fields explicitly — `toroidal_magnetic_field_at_magnetic_axis` is accepted in the first half (row 82) — so the qualified form is the batch's own convention, not an invention.
- The identity is also bound to `core_profiles/vacuum_toroidal_field/b0` (index 41) and `summary/global_quantities/b0/value` (index 220), both outside this range and both the same quantity; the collision itself belongs to the whole-cohort collision sweep, and the spelling fix applies to all three.

### 107. `reference_major_radius` — **correct**

- source path: `equilibrium/vacuum_toroidal_field/r0`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Reference major radius where the vacuum toroidal magnetic field is given (usually a fixed position such as the middle of the vessel at the equatorial midplane)
- name description: Nonnegative perpendicular distance from the toroidal symmetry axis to a designated reference location, serving as the major-radius coordinate where the vacuum toroidal magnetic field is specified in the right-handed cylindrical (R, φ, Z) frame.
- Accepted in the first half at `summary/global_quantities/r0/value` (row 78) and not relitigated; also bound to `core_profiles/vacuum_toroidal_field/r0` (index 42), which the collision sweep owns.

### 108. `gas_flow_of_valve` — **correct**

- source path: `gas_injection/valve/flow_rate`
- unit: `Pa.m^3.s^-1` (data dictionary: `Pa.m^3.s^-1`)
- data-dictionary text: Flow rate at the exit of the valve
- name description: Pressure-volume throughput of gas crossing an injection-valve exit, representing delivered gas flow for plasma fueling, impurity seeding, and density control.
- note: "Gas flow" names three physically different quantities in fuelling practice — particle rate (`s^-1`), mass rate (`kg.s^-1`) and pressure-volume throughput (`Pa.m^3.s^-1`) — and only the published unit tells a reader which one this is. The unit is part of the catalog record and `Pa.m^3.s^-1` is unambiguous in gas-injection practice, so the name stands; `gas_throughput_of_valve` would carry the distinction in the name itself and is worth considering in a consistency pass. Recorded as a note rather than a rejection because the name misstates nothing.

### 109. `lower_bound_photon_energy` — **correct**

- source path: `hard_x_rays/channel/energy_band/lower_bound`
- unit: `eV` (data dictionary: `eV`)
- data-dictionary text: Lower bound of the energy band
- name description: Lower boundary of an X-ray photon-energy band, specifying the minimum photon energy included in the defined detection or emission band.
- This spelling is settled: the first half accepts it at `hard_x_rays/emissivity_profile_1d/lower_bound` (row 40). It is the anchor against which row 110 is judged.

### 110. `upper_photon_energy` — **INCORRECT**

- source path: `hard_x_rays/channel/energy_band/upper_bound`
- unit: `eV` (data dictionary: `eV`)
- data-dictionary text: Upper bound of the energy band
- name description: High-energy boundary of an X-ray photon-acceptance band, defining the maximum photon energy included in a selected spectral window.
- **rejected spelling** `upper_photon_energy` → **proposed spelling** `upper_bound_photon_energy`
- why: This is the other half of a two-member pair and it is spelled differently from its partner. The lower member is `lower_bound_photon_energy` — accepted, settled, and bound to the sibling leaf of this very container. Dropping `bound` from one member of a bounded interval makes the pair read as two unrelated quantities: `upper_photon_energy` reads as *the highest photon energy present*, a measured maximum, where the datum is the *declared upper edge of an acceptance band*. The semantic distinction is between a property of the radiation and a property of the instrument's spectral window, and the missing word is the only thing separating them.

### 111. `etendue_of_hard_xray_detector` — **correct**

- source path: `hard_x_rays/channel/etendue`
- unit: `m^2.sr` (data dictionary: `m^2.sr`)
- data-dictionary text: Etendue (geometric extent) of the channel's optical system
- name description: Geometric optical throughput of a hard X-ray detector's optical system, determined by its effective collecting area and accepted solid angle.
- The `_detector` locus spelling is settled and is not relitigated here.

### 112. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `hard_x_rays/channel/line_of_sight/first_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.

### 113. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `hard_x_rays/channel/line_of_sight/second_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 114. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `hard_x_rays/channel/line_of_sight/second_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.

### 115. `hard_xray_emissivity` — **INCORRECT**

- source path: `hard_x_rays/emissivity_profile_1d/emissivity`
- unit: `m^-3.s^-1.sr^-1` (data dictionary: `m^-3.s^-1.sr^-1`)
- data-dictionary text: Radial profile of the plasma emissivity in this energy band
- name description: Local production rate of hard-X-ray bremsstrahlung photons per plasma volume and emission solid angle, integrated over a specified photon-energy band.
- **rejected spelling** `hard_xray_emissivity` → **proposed spelling** `hard_xray_photon_emissivity`
- why: The unit is a photon rate per volume per solid angle, not a power density — a power emissivity would be `W.m^-3.sr^-1`. The two differ by the mean photon energy of the band, which for a hard X-ray channel is tens of keV, so they are not interchangeable by any factor a reader could guess. The name's own description supplies the missing word ("production rate of ... photons"). The first half rejected `hard_xray_brightness` on exactly this distinction, proposing `hard_xray_photon_radiance` because the unit was photon-based; the same reasoning applied to the same instrument's emissivity gives `hard_xray_photon_emissivity`, and using it keeps the two hard X-ray radiometric names in one family.

### 116. `outer_hard_xray_half_width_at_emissivity_peak` — **correct**

- source path: `hard_x_rays/emissivity_profile_1d/half_width_external`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: External (towards separatrix) half width of the emissivity peak (in normalised toroidal flux)
- name description: Outward extent of the hard X-ray emissivity peak from its maximum to the outer half-maximum point toward the separatrix in normalized toroidal flux.
- The name's `outer` maps to the data dictionary's "external (towards separatrix)" correctly, and the asymmetric inner/outer pair is a real physical distinction for a peaked emissivity profile.
- note: A "half width" in a name conventionally implies a length, and this one is dimensionless — it is a width in normalized toroidal flux, which only the unit `1` and the description disclose. The name is accepted because the dimensionless unit is published with it and the `_at_emissivity_peak` qualifier ties it to a flux-labelled profile, but a reader scanning names alone would expect metres.

### 117. `inner_hard_xray_half_width_at_emissivity_peak` — **correct**

- source path: `hard_x_rays/emissivity_profile_1d/half_width_internal`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Internal (towards magnetic axis) half width of the emissivity peak (in normalised toroidal flux)
- name description: Dimensionless inward extent of the hard X-ray emissivity peak from its maximum to the inner half-maximum point in normalized toroidal flux.

### 118. `normalized_toroidal_flux_coordinate_at_measurement_position` — **INCORRECT**

- source path: `hard_x_rays/emissivity_profile_1d/peak_position`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Normalised toroidal flux coordinate position at which the emissivity peaks
- name description: Dimensionless normalized toroidal-flux label that maps a physical measurement position onto a nested magnetic surface between the magnetic axis and equilibrium boundary.
- **rejected spelling** `normalized_toroidal_flux_coordinate_at_measurement_position` → **proposed spelling** `normalized_toroidal_flux_coordinate_at_emissivity_peak`
- why: The datum is where the *emissivity peaks*, which is a property of the inverted profile — an outcome of the measurement. A measurement position is where the instrument looks, which is an input to it. For a hard X-ray channel the two are routinely far apart: the sightline crosses the whole plasma while the emissivity peak sits at one flux label. Naming an inferred peak location as an instrument position invites a consumer to use it as a viewing geometry, which it is not.
- The same identity is bound to `ece/channel/position/rho_tor_norm` (index 46), where it genuinely *is* a measurement position, so one published name currently covers both an instrument locus and a profile feature. That collision spans outside this index range; it is deferred to the node that owns the whole-cohort collision sweep, and only this path's binding is judged here.

### 119. `normalized_toroidal_flux_coordinate` — **correct**

- source path: `hard_x_rays/emissivity_profile_1d/rho_tor_norm`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Normalised toroidal flux coordinate grid
- name description: Dimensionless radial label equal to the square root of toroidal magnetic flux normalized between the magnetic axis and equilibrium boundary.
- The bare, unqualified form is right for a grid axis; it is also bound to `core_profiles/profiles_1d/grid/rho_tor_norm` (index 39), the same kind of object.

### 120. `upper_photon_energy` — **INCORRECT**

- source path: `hard_x_rays/emissivity_profile_1d/upper_bound`
- unit: `eV` (data dictionary: `eV`)
- data-dictionary text: Upper bound of the energy band
- name description: High-energy boundary of an X-ray photon-acceptance band, defining the maximum photon energy included in a selected spectral window.
- **rejected spelling** `upper_photon_energy` → **proposed spelling** `upper_bound_photon_energy`
- why: The same identity and the same defect as row 110 — this is the sibling of the settled `lower_bound_photon_energy`, which the first half accepts at this very container (row 40), and the pair must be spelled symmetrically. It is listed separately because it is a separate binding; the fix is one rename covering both, and `soft_x_rays/channel/energy_band/upper_bound` (index 181) outside this range carries the same identity.

### 121. `wave_current_amplitude_of_antenna_strap` — **INCORRECT**

- source path: `ic_antennas/antenna/module/current/amplitude`
- unit: `A` (data dictionary: `A`)
- data-dictionary text: Amplitude of the measurement
- name description: Peak magnitude of the radio-frequency current at a specified position along an ion-cyclotron-heating antenna strap.
- **rejected spelling** `wave_current_amplitude_of_antenna_strap` → **proposed spelling** `wave_current_amplitude_of_ion_cyclotron_heating_antenna`
- why: The bound path is `module/current`, a quantity of the module. The data dictionary gives the strap its own subtree at `module/strap/`, and this batch uses it — `back_surface_distance_of_antenna_strap`, `wave_phase_of_antenna_strap` (both accepted in the first half) and rows 127 to 130 here are all genuine `module/strap/` leaves. Binding a `_of_antenna_strap` name to the module container therefore claims a locus the batch spends on something else. The decisive evidence is internal: the immediate sibling `module/voltage/amplitude` is spelled `voltage_amplitude_of_ion_cyclotron_heating_antenna` (row 131), so the current and the voltage of one identical container are published under two different loci. A module may carry more than one strap, so the two are not even the same object in general.

### 122. `capacitance_of_ion_cyclotron_heating_antenna` — **INCORRECT**

- source path: `ic_antennas/antenna/module/matching_element/capacitance`
- unit: `F` (data dictionary: `F`)
- data-dictionary text: Capacitance of the macthing element
- name description: Effective capacitance of an impedance-matching element in an ion-cyclotron heating antenna, relating stored charge to the RF voltage used for circuit tuning.
- **rejected spelling** `capacitance_of_ion_cyclotron_heating_antenna` → **proposed spelling** `capacitance_of_impedance_matching_element`
- why: An antenna capacitance and a matching-element capacitance are different components of the same circuit and different numbers. The antenna's own capacitance is a fixed property of its geometry; the matching element's is the *tuned* value an operator varies shot to shot to cancel the reactive part of the load. Publishing the tuning capacitor under the antenna's name means a consumer reading `capacitance_of_ion_cyclotron_heating_antenna` across a shot sees a quantity that moves, and attributes the movement to the antenna. The name's description already says "of an impedance-matching element", so the correct locus is known and simply absent from the name. (The data dictionary's own text misspells "matching"; that is upstream and not a naming defect.)

### 123. `forward_wave_phase_of_ion_cyclotron_heating_antenna` — **correct**

- source path: `ic_antennas/antenna/module/phase_forward`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Phase of the forward power with respect to the first module
- name description: Relative phase of the forward radio-frequency power-wave phasor at an ion-cyclotron-heating antenna module, referenced to the first module for inter-module toroidal phasing.
- The `forward` qualifier is present and the antenna locus matches the first half's accepted `reflected_wave_phase_of_ion_cyclotron_heating_antenna` at the sibling leaf `module/phase_reflected`, so the forward/reflected pair is spelled symmetrically.

### 124. `forward_power_of_ion_cyclotron_heating_antenna` — **correct**

- source path: `ic_antennas/antenna/module/power_forward`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: Forward power arriving to the back of the module
- name description: Forward RF power of an ion-cyclotron heating antenna is the incident power traveling from the generator toward the antenna before reflection and coupling to the plasma.
- Forward power is correctly kept distinct here from both launched and coupled power, which is the distinction the first half found violated in `total_power_due_to_ion_cyclotron_heating` (row 80).

### 125. `reflected_power_of_ion_cyclotron_heating_antenna` — **correct**

- source path: `ic_antennas/antenna/module/power_reflected`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: Reflected power
- name description: Reflected power of an ion-cyclotron heating antenna is the backward-traveling radio-frequency power returned from one antenna toward the transmission-line source rather than coupled into the plasma.

### 126. `pressure_of_ion_cyclotron_heating_antenna` — **INCORRECT**

- source path: `ic_antennas/antenna/module/pressure/amplitude`
- unit: `Pa` (data dictionary: `Pa`)
- data-dictionary text: Amplitude of the measurement
- name description: Pressure associated with an ion-cyclotron-heating antenna module, representing mechanical normal loading or contained-gas pressure at hardware components.
- **rejected spelling** `pressure_of_ion_cyclotron_heating_antenna` → **proposed spelling** `pressure_amplitude_of_ion_cyclotron_heating_antenna`
- why: The leaf is `/amplitude` — the peak magnitude of an oscillating signal, carried in the same amplitude-and-phase wrapper the data dictionary uses for the module's current and voltage. Its two siblings in that identical container both keep the qualifier: `wave_current_amplitude_of_antenna_strap` (row 121) and `voltage_amplitude_of_ion_cyclotron_heating_antenna` (row 131). This one drops it, so the published name reads as a static pressure while the datum is the amplitude of a varying one — for a sinusoid those differ by the factor between peak and mean, and the mean of a pure oscillation is zero.
- The description compounds the problem by hedging between two unrelated physical quantities, "mechanical normal loading or contained-gas pressure": a structural stress on the strap and a gas pressure in the feedthrough are not the same measurement and cannot both be right. That hedge is a description defect and would be a note on its own; it is recorded here because it shows the amplitude omission is not the only thing unresolved about this row.

### 127. `toroidal_angle_of_antenna_strap` — **INCORRECT**

- source path: `ic_antennas/antenna/module/strap/outline/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of each antenna-strap outline point around the machine symmetry axis in the right-handed cylindrical (R, φ, Z) frame.
- **rejected spelling** `toroidal_angle_of_antenna_strap` → **proposed spelling** `toroidal_angle_of_antenna_strap_outline`
- why: Two different things are called the toroidal angle of a strap: the single installation angle at which the strap is mounted, and the per-point φ of each vertex of its outline. This row is the second, and the name reads as the first — a reader would take one scalar where the datum is an array around a contour. The row's own two siblings at the identical container say so: `radial_outline_of_antenna_strap` (row 128) and `vertical_outline_of_antenna_strap` (row 129) both carry `outline`, so one geometric point set is published with two naming schemes across its three coordinates. This is the same shape as the aperture-centre finding the first half reported, where φ belonged to one object and R and Z to another.
- The `*_outline_of_X` family is settled by the first half (`radial_outline_of_limiter_tile`, `vertical_outline_of_plasma_facing_component`) but has no φ member in it, so there is no settled spelling to copy. The proposal keeps the physically correct word `toroidal_angle` for an angle in radians and names the outline locus explicitly rather than inventing a `toroidal_outline` that would read as a length.

### 128. `radial_outline_of_antenna_strap` — **correct**

- source path: `ic_antennas/antenna/module/strap/outline/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of each point on an antenna-strap outline in the right-handed cylindrical (R, φ, Z) frame.
- Member of the `*_outline_of_X` family the first half settled; not relitigated.

### 129. `vertical_outline_of_antenna_strap` — **correct**

- source path: `ic_antennas/antenna/module/strap/outline/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of each contour point defining an ion-cyclotron antenna strap boundary, distinguishing the strap outline from its geometric-center location.

### 130. `toroidal_width_of_antenna_strap` — **correct**

- source path: `ic_antennas/antenna/module/strap/width_phi`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Width of strap in the toroidal direction
- name description: Full toroidal width of the rectangular cross-section of an ICRH antenna strap conductor element, measured along the toroidal direction in the right-handed cylindrical (R, φ, Z) frame.
- A toroidal width in metres rather than radians is the physically useful form and matches the data dictionary exactly.

### 131. `voltage_amplitude_of_ion_cyclotron_heating_antenna` — **correct**

- source path: `ic_antennas/antenna/module/voltage/amplitude`
- unit: `V` (data dictionary: `V`)
- data-dictionary text: Amplitude of the measurement
- name description: Peak magnitude of the radio-frequency voltage waveform on a single transmission-line feed of an ion-cyclotron-heating antenna module.
- This row is the reference against which rows 121, 126 and 132 are judged: it keeps the `amplitude` qualifier, names the measured electrical quantity, and puts the locus at the antenna, which is what the module container supports.

### 132. `wave_phase_of_ion_cyclotron_heating_antenna` — **INCORRECT**

- source path: `ic_antennas/antenna/module/voltage/phase`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Phase of the measurement
- name description: Relative phase angle of a radio-frequency phasor at an antenna element in an ion-cyclotron heating launcher, measured relative to the first element for inter-element phasing.
- **rejected spelling** `wave_phase_of_ion_cyclotron_heating_antenna` → **proposed spelling** `voltage_phase_of_ion_cyclotron_heating_antenna`
- why: Four distinct phases live in this one data-dictionary container — of the voltage, of the current, of the forward power and of the reflected power — and the name identifies none of them. This row is the voltage's, the direct sibling of `voltage/amplitude`; the forward and reflected ones are already published as `forward_wave_phase_of_ion_cyclotron_heating_antenna` (row 123) and `reflected_wave_phase_of_ion_cyclotron_heating_antenna` (first half). The differences are not academic: the voltage-to-current phase is what sets the strap's reactive loading, so a consumer that reads `wave_phase_of_ion_cyclotron_heating_antenna` as the current phase computes the wrong sign of reactive power. The name's own description says only "a radio-frequency phasor", which is the self-descriptiveness test failing in the description as well.

### 133. `toroidal_coordinate_of_line_of_sight` — **correct**

- source path: `interferometer/channel/line_of_sight/first_point/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of the first reference point on a diagnostic line of sight, locating that point around the machine symmetry axis.
- The settled line-of-sight coordinate family; the description's "first reference point" is accurate at this binding.

### 134. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `interferometer/channel/line_of_sight/first_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 135. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `interferometer/channel/line_of_sight/first_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.

### 136. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `interferometer/channel/line_of_sight/second_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 137. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `interferometer/channel/line_of_sight/second_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.

### 138. `toroidal_coordinate_of_line_of_sight` — **correct**

- source path: `interferometer/channel/line_of_sight/third_point/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of the first reference point on a diagnostic line of sight, locating that point around the machine symmetry axis.
- note: The name is correct and deliberately generic across the points of a sightline, but its shared description says "the **first** reference point" while this binding is the **third** point. The first half recorded this same description defect at its row 46; it is one text to repair on one shared identity, not a per-binding fault, and it carries no incorrect verdict.

### 139. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `interferometer/channel/line_of_sight/third_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.
- The vertical member of the family carries the point-agnostic wording "a designated point", so it does not have the defect row 138 records.

### 140. `line_integrated_electron_number_density` — **INCORRECT**

- source path: `interferometer/channel/n_e_line`
- unit: `m^-2` (data dictionary: `m^-2`)
- data-dictionary text: Line integrated density, possibly obtained by a combination of multiple interferometry wavelengths. Corresponds to the density integrated along the full line-of-sight (i.e. forward AND return for a reflected channel: NO dividing by 2 correction)
- name description: Free-electron column density accumulated along a complete electromagnetic propagation path, including both forward and return segments when present.
- **rejected spelling** `line_integrated_electron_number_density` → **proposed spelling** `line_integrated_electron_density`
- why: Already adjudicated in the first half (row 22) and reused here rather than re-argued. The cohort spells this physical base `electron_density` in eight names — including `line_averaged_electron_density` at the very next leaf of this container (row 141) and `volume_averaged_electron_density` (row 145) — and `electron_number_density` in this one identity only. The semantic content is identical, so the minority spelling is an inconsistency rather than a distinction, and two spellings of one base inside one published batch is not publishable.
- The identity is also bound to `equilibrium/time_slice/constraints/n_e_line/measured` (index 63) and, per the first half, to the `reconstructed` sibling; the collision belongs to the whole-cohort sweep and the rename covers every binding.

### 141. `line_averaged_electron_density` — **correct**

- source path: `interferometer/channel/n_e_line_average`
- unit: `m^-3` (data dictionary: `m^-3`)
- data-dictionary text: Line average density, possibly obtained by a combination of multiple interferometry wavelengths. Corresponds to the density integrated along the full line-of-sight and then divided by the length of the line-of-sight
- name description: Number density of free electrons per physical volume averaged along a complete plasma propagation chord, equal to the path integral divided by chord length.
- The `m^-3` unit and the "divided by the length" wording both confirm this is the averaged rather than the integrated form, and the name says `averaged`. This row and row 140 are the pair whose distinction the `electron_number_density` spelling blurs.

### 142. `wave_phase_of_wave_beam` — **INCORRECT**

- source path: `interferometer/channel/wavelength/phase_corrected`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Phase measured for this wavelength, corrected from fringe jumps
- name description: Fringe-jump-corrected phase angle of a probing electromagnetic signal at a selected wavelength, referenced to the channel's defined launch phase.
- **rejected spelling** `wave_phase_of_wave_beam` → **proposed spelling** `fringe_jump_corrected_phase_of_interferometer_beam`
- why: Two things are missing and both matter. The datum is the phase *after* fringe-jump correction, and a fringe jump is a discrete 2π-multiple error an interferometer accumulates when the density moves faster than the acquisition can follow; the raw and corrected phases of the same channel can differ by many radians, so publishing them under a name that does not say which is which invites a consumer to take a corrupted trace for a clean one. The name's own description supplies "fringe-jump-corrected", which the name omits. Second, `wave_phase_of_wave_beam` names neither the instrument nor the measured quantity — every probing beam in the batch is a wave beam — and the cohort already spells this object `interferometer_beam` in the accepted `length_variation_of_interferometer_beam` (first half, row 48), which the proposal reuses.

### 143. `spectral_calibration_factor_at_line_of_sight` — **INCORRECT**

- source path: `interferometer/channel/wavelength/phase_to_n_e_line`
- unit: `m^-2.rad^-1` (data dictionary: `m^-2.rad^-1`)
- data-dictionary text: Conversion factor to be used to convert phase into line density for this wavelength
- name description: Wavelength-specific interferometric conversion coefficient that converts a signed phase change along a line of sight into electron column density.
- **rejected spelling** `spectral_calibration_factor_at_line_of_sight` → **proposed spelling** `phase_to_line_integrated_electron_density_conversion_factor`
- why: The name is not merely vague, it is misdescriptive. A "spectral calibration factor" in diagnostics is the intensity response of an instrument as a function of wavelength, in radiometric units; this quantity is a phase-to-column-density conversion coefficient, `m^-2.rad^-1`, which is a different physical object entirely and is fixed by the probing wavelength and fundamental constants rather than by any calibration measurement. The `_at_line_of_sight` locus is wrong too: the coefficient varies with wavelength, not with which chord it is applied to, and the data dictionary says so ("for this wavelength"). The proposed spelling states both ends of the conversion and uses `line_integrated_electron_density`, the survivor spelling proposed at row 140, so the factor and the quantity it produces name the same base.

### 144. `wavelength_of_wave_beam` — **correct**

- source path: `interferometer/channel/wavelength/value`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Wavelength value
- name description: Vacuum spatial period of a probing electromagnetic wave propagating as a diagnostic beam, defining its spectral wavelength.
- note: The generic `wave_beam` locus is justified here and not at row 142, because this identity is genuinely cross-diagnostic: it is also bound to `polarimeter/channel/wavelength` (index 179), where an interferometer-specific spelling would be wrong. The batch nevertheless carries two spellings for a probing beam — `wave_beam` here and `interferometer_beam` in the accepted `length_variation_of_interferometer_beam` — and which survives where is a cohort-wide consistency question rather than a defect in this row.

### 145. `volume_averaged_electron_density` — **correct**

- source path: `interferometer/n_e_volume_average`
- unit: `m^-3` (data dictionary: `m^-3`)
- data-dictionary text: Volume average plasma density estimated from the line densities measured by the various channels
- name description: Number density of free electrons per physical volume averaged over the plasma volume enclosed by the last closed flux surface, giving the global mean free-electron density.
- The data dictionary says the quantity is *estimated from* the line densities, which is a statement about how it was obtained rather than about what it is; the name correctly denotes the volume average itself. Also bound to `summary/volume_average/n_e/value` (index 250), the same quantity.

### 146. `area_of_toroidal_magnetic_field_probe` — **INCORRECT**

- source path: `magnetics/b_field_phi_probe/area`
- unit: `m^2` (data dictionary: `m^2`)
- data-dictionary text: Area of each turn of the sensor; becomes effective area when multiplied by the turns
- name description: Geometric area enclosed by one complete winding turn of a toroidal magnetic-field probe coil, setting its single-turn magnetic-flux sensitivity.
- **rejected spelling** `area_of_toroidal_magnetic_field_probe` → **proposed spelling** `turn_area_of_toroidal_magnetic_field_probe`
- why: The data dictionary is explicit that two areas exist and that this is the smaller one: the per-turn area, which "becomes effective area when multiplied by the turns". The effective area is what converts a measured coil voltage into a field, so using the per-turn area in that conversion under-reads the field by the full turn count — and this probe's turn count is itself published in this same cohort as `turn_count_of_toroidal_magnetic_field_probe` (row 152), so both factors are in the catalog and only the name distinguishes them. An unqualified `area_of_<probe>` reads as the probe's area, which a consumer would naturally take as the effective one. The name's description already says "one complete winding turn"; the name does not.

### 147. `toroidal_magnetic_field` — **correct**

- source path: `magnetics/b_field_phi_probe/field`
- unit: `T` (data dictionary: `T`)
- data-dictionary text: Magnetic field component in direction of sensor normal axis (n) averaged over sensor volume defined by area and length, where n = cos(poloidal_angle)*cos(toroidal_angle)*grad(R) - sin(poloidal_angle)*grad(Z) + cos(poloidal_angle)*sin(toroidal_angle)*grad(Phi)/norm(grad(Phi))
- name description: Signed toroidal component of the local total magnetic induction, resolved along increasing toroidal angle in the right-handed cylindrical (R, φ, Z) frame.
- The first half accepts the exact analogue `poloidal_magnetic_field` at `magnetics/b_field_pol_probe/field` (row 52), and a toroidal-field probe's reading is nominally the toroidal component, so the name is consistent with a settled sibling and is accepted.
- note: The data dictionary defines the value as the component along the *sensor normal*, volume-averaged over the sensor, and the probe publishes both a `poloidal_angle` (row 148) and a `toroidal_angle` (row 151) that let that normal depart from φ. So the name is exact only for an ideally aligned probe, and for a tilted one the datum is a projection. This is recorded because a consumer combining probes should use the published angles rather than assume the name; it does not rise to a rejected spelling, since rejecting it would also unsettle the accepted poloidal sibling whose data-dictionary text says only "Measured magnetic field".

### 148. `poloidal_angle_of_toroidal_magnetic_field_probe` — **correct**

- source path: `magnetics/b_field_phi_probe/poloidal_angle`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Angle of the sensor normal vector (vector parallel to the the axis of the coil, n on the diagram) with respect to horizontal plane (clockwise theta-like angle). Zero if sensor normal vector fully in the horizontal plane and oriented towards increasing major radius. Values in [0 , 2Pi]
- name description: Signed poloidal tilt angle of a toroidal magnetic-field probe's sensitive-axis normal in the right-handed cylindrical (R, φ, Z) frame, measured clockwise from +R.
- The name correctly denotes an orientation of the probe rather than a position of it, which is the distinction rows 149 and 150 get wrong.
- note: The description calls the angle "signed" while the data dictionary states the range as `[0, 2Pi]`, which is unsigned. The two conventions agree modulo 2π on the same physical orientation, so no value is wrong, but a consumer writing a range check against the description would reject valid data. This is a description defect and carries no incorrect verdict.

### 149. `toroidal_angle_of_measurement_position` — **INCORRECT**

- source path: `magnetics/b_field_phi_probe/position/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate locating a measurement position around the machine symmetry axis in the right-handed cylindrical (R, φ, Z) frame.
- **rejected spelling** `toroidal_angle_of_measurement_position` → **proposed spelling** `toroidal_coordinate_of_toroidal_magnetic_field_probe`
- why: A magnetic probe's `position` is where the sensor is installed, a fixed property of the machine; a measurement position is where a diagnostic samples the plasma, which for a line-integrating or imaging instrument is somewhere else entirely. The probe case is decided inside its own container: the `/z` sibling of this very node is already published as `vertical_coordinate_of_toroidal_magnetic_field_probe` and accepted in the first half (row 51), so one probe position currently carries two different loci across its coordinates — the same defect this half records at row 127 for the antenna strap outline.
- The proposal deliberately says `toroidal_coordinate`, not `toroidal_angle`, because `toroidal_angle_of_toroidal_magnetic_field_probe` is already taken by row 151 and means something else: the probe's sensing *orientation*. The cohort's settled position spelling is `toroidal_coordinate_of_line_of_sight`, so `toroidal_coordinate` for a location and `toroidal_angle` for an orientation is the distinction the batch already draws.
- The identity is also bound to `ece/channel/position/phi` (index 44), where it is a genuine measurement position, and to `magnetics/b_field_pol_probe/position/phi` (index 156), which repeats this defect on the poloidal probe. That collision spans outside this index range and is deferred to the whole-cohort collision sweep.

### 150. `radial_coordinate_of_measurement_position` — **INCORRECT**

- source path: `magnetics/b_field_phi_probe/position/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate locating a measurement position by perpendicular distance from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.
- **rejected spelling** `radial_coordinate_of_measurement_position` → **proposed spelling** `radial_coordinate_of_toroidal_magnetic_field_probe`
- why: The same defect as row 149 on the radial axis of the same point, and the spelling proposed here is the one the first half already proposed when it reported this exact binding as a whole-cohort finding. A sensor location is not a measurement position, and the `/z` member of this identical point is accepted as `vertical_coordinate_of_toroidal_magnetic_field_probe`, so R, φ and Z of one installed probe are presently published under three different loci.
- Also bound to `camera_x_rays/aperture/centre/r` (index 17), an aperture centre, which is a third distinct object under the one name; that collision is the first half's recorded finding and is deferred to the collision sweep.

### 151. `toroidal_angle_of_toroidal_magnetic_field_probe` — **correct**

- source path: `magnetics/b_field_phi_probe/toroidal_angle`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Angle of the projection of the sensor normal vector (n) in the horizontal plane with the increasing R direction (i.e. grad(R)) (angle is counter-clockwise from above as in cocos=11 phi-like angle). Values should be taken modulo pi with values within (-pi/2,pi/2]. Zero if projected sensor normal is parallel to grad(R), pi/2 if it is parallel to grad(phi).
- name description: The toroidal angle of a toroidal magnetic-field probe is the signed azimuthal orientation of the probe sensitive-axis normal projected onto the horizontal plane, measured from +R toward increasing φ in the right-handed cylindrical (R, φ, Z) frame. Opposite projected normals are folded into the same principal orientation.
- The description captures both the reference direction and the modulo-π fold the data dictionary specifies, which is the non-obvious part of this quantity; the name denotes an orientation and the locus is the probe, both correct. It is the name whose existence forces row 149's proposal to use `toroidal_coordinate` for the position.

### 152. `turn_count_of_toroidal_magnetic_field_probe` — **correct**

- source path: `magnetics/b_field_phi_probe/turns`
- unit: `1` (data dictionary: *empty*)
- data-dictionary text: Turns in the coil, including sign
- name description: Signed number of complete winding turns in a toroidal magnetic-field probe coil, with orientation referenced to the positive toroidal direction.
- note: **The one unit disagreement in this half.** The standard name carries `1` and the data dictionary carries no unit at all. The standard name is the defensible side: a turn count is a signed dimensionless integer, and `1` is the catalog's spelling for dimensionless, whereas an empty unit is indistinguishable from an unfilled field. The name correctly keeps `signed`, which matters because the sign encodes winding orientation and flipping it inverts the measured field.

### 153. `area_of_poloidal_magnetic_field_probe` — **INCORRECT**

- source path: `magnetics/b_field_pol_probe/area`
- unit: `m^2` (data dictionary: `m^2`)
- data-dictionary text: Area of each turn of the coil
- name description: Geometric cross-sectional area enclosed by one winding turn of a poloidal magnetic-field probe coil, defining its per-turn magnetic-flux coupling.
- **rejected spelling** `area_of_poloidal_magnetic_field_probe` → **proposed spelling** `turn_area_of_poloidal_magnetic_field_probe`
- why: The same defect as row 146 on the poloidal probe. The data dictionary says "area of each turn of the coil" and the name says the area of the probe; the two differ by the turn count, and the quantity that converts a coil voltage to a field is the product, not this factor. The proposal keeps the two probes' spellings parallel, which is why both rows are renamed together rather than one of them.

## Where the defects fall

![Verdict for each of the 51 bindings at cohort indices 103 to 153, plotted against cohort index and banded by IDS, with each rejected spelling labelled and coloured by defect class](/imas-codex/figures/sn-west-catalog-release/west-name-verdicts-equilibrium-profiles-to-magnetics.png)

The figure is the reason the 33.3 % below should not be read as a cohort-wide
rate. This half is a **contiguous slice** of the path-ordered cohort, not a
stratified sample, and the slice happens to land on the batch's
instrument-hardware containers: 12 rows of `hard_x_rays`, 12 of `ic_antennas`
and 8 of `magnetics` account for 13 of the 17 rejections, while the
5 `equilibrium` rows yield 1 and the 13 `interferometer` rows yield 3. Every
rejection in the three dense bands is a locus or qualifier error at a
diagnostic's own container — which object the name is attached to, or which of
several sibling signals it denotes — and none is an error about plasma physics.
The first half's 11.6 % came from a draw spread over 20 IDSs; the two numbers
measure different populations and only their sum over the whole cohort would
be an estimate of it.

## Result

| | count |
| --- | --- |
| rows judged (cohort indices 103–153, inclusive) | **51** |
| judged **correct** | 34 |
| judged **incorrect**, each with a proposed spelling | **17** |
| correct + incorrect | **51** |
| incorrect as a percentage of 51 | **33.3 %** |
| notes recorded on **correct** rows (counted separately, not rejections) | 8 |
| unit disagreements between `sn_unit` and `dd_unit` | 1 of 51 |
| rows whose identity is also bound to source paths outside this index range | 22 |

**17 of 51 — 33.3 % — are not publishable as spelled.** Grouped into the five
classes the first half established, because the classes have different
remedies:

- **The name asserts more than the data supports** (3): `toroidal_vacuum_magnetic_field` for a 1/R field quoted without its radius (106); `area_of_toroidal_magnetic_field_probe` (146) and `area_of_poloidal_magnetic_field_probe` (153) for per-turn coil areas published as probe areas.
- **The name is bound to the wrong object** (5): `normalized_toroidal_flux_coordinate_at_measurement_position` on an emissivity-peak location (118); `wave_current_amplitude_of_antenna_strap` on a module current whose voltage sibling uses the antenna locus (121); `capacitance_of_ion_cyclotron_heating_antenna` on a matching element (122); `toroidal_angle_of_measurement_position` (149) and `radial_coordinate_of_measurement_position` (150) on an installed probe's position, whose `z` member already carries the probe locus.
- **One name covers two physically different quantities** (0 in this half). The two candidates are recorded as wrong-object at the path judged here, with their cross-range halves deferred: `..._at_measurement_position` spans an ECE measurement position and a hard X-ray emissivity peak, and `*_of_measurement_position` spans an ECE channel, an X-ray camera aperture centre and two magnetic probe positions. Both collisions reach outside indices 103–153 and belong to the whole-cohort collision sweep, which is why this class reads zero here rather than absent.
- **Not self-descriptive** (6): `hard_xray_emissivity` for a photon-rate emissivity (115); `pressure_of_ion_cyclotron_heating_antenna` dropping the `amplitude` both its siblings keep (126); `toroidal_angle_of_antenna_strap` reading as an installation angle where its two sibling coordinates say `outline` (127); `wave_phase_of_ion_cyclotron_heating_antenna` naming none of the four phases in its container (132); `wave_phase_of_wave_beam` omitting the fringe-jump correction (142); `spectral_calibration_factor_at_line_of_sight` for a phase-to-column-density coefficient (143).
- **Minority spelling of a base the cohort already fixes** (3): `upper_photon_energy` against the settled `lower_bound_photon_energy` (110 and 120, one identity at two paths); `line_integrated_electron_number_density` against the settled `electron_density` base (140).

Eight **notes** sit on rows whose verdict is correct and are counted apart from
the rejections, because each is a defect in a description, a convention or a
shared text rather than in a name: the shared-identity question at row 103; the
three senses of "gas flow" that only the unit resolves (108); a dimensionless
`half_width` in flux coordinate (116); the shared line-of-sight description
saying "first reference point" while bound to the third point (138, the first
half's row-46 defect recurring); the `wave_beam` and `interferometer_beam`
spellings both in use for a probing beam (144); a field defined along the
sensor normal published as the toroidal field (147); a "signed" angle whose
data dictionary gives `[0, 2Pi]` (148); and the one unit disagreement (152).

**The single unit disagreement is row 152**, `turn_count_of_toroidal_magnetic_field_probe`:
the standard name carries `1` and the data dictionary carries nothing. The
standard name is the defensible side — a turn count is dimensionless, `1` is
the catalog's spelling for dimensionless, and an empty data-dictionary unit is
indistinguishable from an unfilled field. The other 50 rows agree exactly.

### What sums with the first half

| | first half | this half | cohort so far |
| --- | --- | --- | --- |
| rows judged | 86 | 51 | 137 of 341 |
| correct | 76 | 34 | 110 |
| incorrect | 10 | 17 | 27 |

The two halves are drawn differently — every-fourth across the cohort against a
contiguous slice — so the 27 is a count of rejections found, not a rate to
extrapolate from. What does carry across is that no rejection in either half
turned on a disputed physics claim: all 27 are about which object a name is
attached to, which of several sibling quantities it denotes, or which of two
spellings of one base survives.

### Deferred to the whole-cohort collision sweep

22 of the 51 rows carry an identity that is also bound to source paths outside
indices 103–153. Most are benign reuse of one settled name across equivalent
leaves — the line-of-sight coordinate family accounts for 10 of them. Four are named in the
verdicts above as genuine collisions and are deferred rather than judged here:
rows 106 (`core_profiles` and `summary` vacuum field), 118 (ECE measurement
position against a hard X-ray emissivity peak), 140 (equilibrium constraint
against the interferometer channel) and 149/150 (ECE channel, X-ray camera
aperture centre and two magnetic probe positions under one measurement-position
name). This node judges only the binding at the path inside its range; the
choice of which binding keeps a shared identity belongs to the node that owns
the sweep.
