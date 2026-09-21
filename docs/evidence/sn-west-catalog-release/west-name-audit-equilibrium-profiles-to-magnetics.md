# WEST batch accepted names — physical-correctness audit, equilibrium profiles to magnetics

provisional: true — verdicts are being appended as they are judged and the result section is not yet closed.

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
