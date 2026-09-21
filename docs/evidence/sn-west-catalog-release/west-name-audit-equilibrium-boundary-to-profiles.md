# WEST cohort remainder, indices 52–102 — physical-correctness audit

provisional: false — every one of the 51 rows carries a verdict and the
result section is closed.

Are the accepted standard names bound to WEST batch sources physically correct
and self-descriptive? This file judges the 51 bindings whose `index` field lies
in the inclusive range 52 to 102 of the `bindings` list in
`west-name-cohort-remainder.json` — the block that runs from
`equilibrium/time_slice/boundary/minor_radius` to
`equilibrium/time_slice/profiles_1d/rho_tor`. Sections are numbered by cohort
index, so this half and the other sum rather than overlap.

`west-name-audit.md` judged 86 rows of the same cohort in the same shape; this
file matches it, reuses the spellings it accepted, and does not reopen the
adjudications it records as settled — the `*_of_flux_surface` family (including
that `volume_of_flux_surface` and `area_of_flux_surface` name the volume and
area *enclosed by* the surface), `back_surface_distance_of_antenna_strap`, and
the etendue `_detector` spelling.

## What each verdict is judged against

Every row was drawn from the live graph before this node began and is carried in
`west-name-cohort-remainder.json`; no graph query was issued here and no database
was opened. Each row supplies the name, the DD source path, `sn_unit`,
`sn_description`, `dd_unit`, `dd_doc` and `dd_doc_parent`. Three questions are
asked of each:

1. does the data-dictionary text of the path it is bound to describe the
   quantity the name claims;
2. do `sn_unit` and `dd_unit` agree, and where they differ which side is
   defensible;
3. is the name self-descriptive to a reader who does not have the source path in
   hand.

**Units agree on 51 of 51 rows in this block.** A uniform column is a claim
about the instrument before it is a claim about the data, so it was controlled
rather than reported: `sn_unit` and `dd_unit` are populated and non-null on all
51 rows, and the same comparison run over the whole 255-row remainder finds
**three genuine disagreements** — indices 152 and 159
(`turn_count_of_*_magnetic_field_probe`, standard name `1` against an empty DD
unit) and index 198 (`atomic_count`, the same shape). The comparison can
therefore see a difference where one exists, and its silence over this block is
a result. The dimensional arithmetic was additionally checked by hand on the
rows where it discriminates — `darea_dpsi` as `Wb^-1.m^2`, `darea_drho_tor` as
`m`, `dvolume_dpsi` as `Wb^-1.m^3`, `dvolume_drho_tor` as `m^2`, and the `gm*`
metric coefficients at `m^-2`, `1`, `T^2`, `T^-2` and `m^-1`, each consistent
with its stated definition. So question 2 discriminates nothing in this block,
and every verdict below turns on questions 1 and 3.

### Where the data-dictionary evidence is thin, it is said so

`dd_doc_parent` is null on all 51 rows of this block. That is also controlled:
the remainder file populates the parent only where the leaf's own documentation
is empty or the literal string `Value`, and it does so on **41 of the 255**
cohort rows (for example index 207, `summary/boundary/elongation/value`, leaf
`Value`, parent `Elongation of the plasma boundary`). No leaf in this block
matches that predicate, which is why the column is empty here rather than
unpopulated everywhere.

Six rows (58–63) nevertheless carry the boilerplate leaf strings `Measured
value` and `Value calculated from the reconstructed equilibrium`. Those strings
are nearly contentless and their container text was not captured, so those six
are judged against the container **name** in the path plus the unit, and each
section says so rather than presenting a reading the text does not support.

## Verdicts

### 52. `minor_radius_of_plasma_boundary` — **correct**

- cohort index: 52
- source path: `equilibrium/time_slice/boundary/minor_radius`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Minor radius of the plasma boundary (defined as (Rmax-Rmin) / 2 of the boundary)
- name description: Half the difference between the maximum and minimum major-radius coordinates of the last closed plasma-boundary contour, defining its horizontal cross-sectional size.
- The description reproduces the DD's parenthetical definition exactly, including that this is a half-width of the boundary's radial extent rather than a distance from the magnetic axis — the two differ whenever the plasma is Shafranov-shifted.
- collision outside this range: the same identity is also bound to `summary/boundary/minor_radius/value`, which is the same physical quantity reported by a different IDS. The collision itself belongs to the whole-cohort sweep.

### 53. `radial_outline_of_plasma_boundary` — **correct**

- cohort index: 53
- source path: `equilibrium/time_slice/boundary/outline/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of each point on the plasma-boundary contour in a poloidal cross-section.
- Follows the `radial_outline_of_<object>` family the other half accepted at `wall/description_2d/limiter/unit/outline/r`; the object named is the one the path binds.

### 54. `vertical_outline_of_plasma_boundary` — **correct**

- cohort index: 54
- source path: `equilibrium/time_slice/boundary/outline/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical height of each point on the plasma-boundary outline in the right-handed cylindrical (R, φ, Z) frame, defining its poloidal cross-sectional shape.
- Paired with row 53 on the same outline container; the two coordinates carry one locus, which is the property the other half found broken on `camera_x_rays/aperture/centre`.

### 55. `normalized_poloidal_flux_coordinate_of_plasma_boundary` — **correct**

- cohort index: 55
- source path: `equilibrium/time_slice/boundary/psi_norm`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Value of the normalised poloidal flux at which the boundary is taken (typically 99.x %), the flux being normalised to its value at the separatrix
- name description: Radial flux-surface location of the selected plasma boundary, expressed by the normalized poloidal flux coordinate. It is the linear normalized poloidal flux value, increasing from zero at the magnetic axis to one at the separatrix, of the closed flux surface taken as the plasma boundary; fixed-boundary equilibrium calculations commonly set it slightly below one to select a surface just inside the separatrix.
- The name reads as a *coordinate of* the boundary rather than as the boundary itself, which is what the DD means: the scalar says which flux surface was taken as the boundary. The description states the linear (not square-root) normalisation explicitly, which is the one place this quantity is routinely misread.

### 56. `triangularity_of_plasma_boundary` — **correct**

- cohort index: 56
- source path: `equilibrium/time_slice/boundary/triangularity`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Triangularity of the plasma boundary
- name description: Dimensionless shape parameter for the plasma-boundary poloidal cross-section, expressing the aggregate inward radial displacement of its upper and lower extrema relative to the geometric center.
- The surface the shape parameter belongs to is explicit in the name rather than left to the reader, which is the property the catalog requires of `triangularity`, `elongation` and `squareness`.

### 57. `lower_triangularity_of_plasma_boundary` — **correct**

- cohort index: 57
- source path: `equilibrium/time_slice/boundary/triangularity_lower`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Lower triangularity of the plasma boundary
- name description: Dimensionless shaping parameter equal to the normalized inward radial displacement of the lower plasma-boundary extremum from the geometric center.
- Exactly mirrors `upper_triangularity_of_plasma_boundary`, accepted in the other half at the `triangularity_upper` sibling; the upper/lower pair is spelled symmetrically.
- collision outside this range: also bound to `summary/boundary/triangularity_lower/value` — the same quantity in a different IDS; deferred to the whole-cohort sweep.

### 58. `poloidal_magnetic_field` — **correct**

- cohort index: 58
- source path: `equilibrium/time_slice/constraints/b_field_pol_probe/measured`
- unit: `T` (data dictionary: `T`)
- data-dictionary text: Measured value
- name description: Poloidal magnetic-field strength of the local total induction, formed from radial and vertical components in the right-handed cylindrical (R, φ, Z) frame.
- The leaf text is boilerplate, so the quantity is read from the container `b_field_pol_probe` and the unit `T`: the poloidal field local to a magnetic probe. The name claims exactly that and no more.
- note on the shared identity: this name is also bound to the `reconstructed` sibling (row 59) and to `magnetics/b_field_pol_probe/field`. That is not a two-quantity collision — measured against reconstructed is *how the number was obtained*, not *what it is*. The repository's own audit encodes this: `provenance_verb_check` in `imas_codex/standard_names/audits.py:1215` documents that "standard names should describe the physical quantity, not how it was obtained" and treats `measured`, `reconstructed`, `fitted`, `computed`, `calculated` as provenance verbs. Exercised directly, the guard refuses `reconstructed_poloidal_magnetic_field` against a source path lacking the word — `audit:provenance_verb_check: name contains 'reconstructed' but source path does not` — while returning clean for the unqualified `poloidal_magnetic_field`. The guard does carry a carve-out (a name may keep the verb when the path itself contains it), so the shared identity is permitted rather than compelled; the choice between them is the collision sweep's, not this row's.

### 59. `poloidal_magnetic_field` — **correct**

- cohort index: 59
- source path: `equilibrium/time_slice/constraints/b_field_pol_probe/reconstructed`
- unit: `T` (data dictionary: `T`)
- data-dictionary text: Value calculated from the reconstructed equilibrium
- name description: Poloidal magnetic-field strength of the local total induction, formed from radial and vertical components in the right-handed cylindrical (R, φ, Z) frame.
- The reconstructed value is the equilibrium solver's prediction at the same probe; it is the same physical quantity as row 58 and differs from it by the fit residual. Naming them alike is the provenance rule of row 58 applied, not a defect.

### 60. `faraday_angle` — **INCORRECT**

- cohort index: 60
- source path: `equilibrium/time_slice/constraints/faraday_angle/measured`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Measured value
- name description: Faraday rotation of a probing wave’s polarization plane caused by electron density and the magnetic-field component along its plasma path.
- **rejected spelling** `faraday_angle` → **proposed spelling** `faraday_rotation_angle`
- why: `faraday_angle` names no physical angle on its own. The Faraday effect is a *rotation of the plane of polarization*, and the name has to be rescued by its own description, which supplies the missing word. A reader holding only the name cannot tell whether the angle is the rotation, the orientation of the analyser, or the inclination of the probing beam.
- this is the same defect the other half recorded at the `reconstructed` sibling of this identical container, with the same proposed spelling. Recording it again is not duplication: the defect is a property of the identity, and the identity is bound at both loci, so a repair applied at one path only would leave this row unfixed.

### 61. `poloidal_magnetic_flux_of_flux_loop` — **correct**

- cohort index: 61
- source path: `equilibrium/time_slice/constraints/flux_loop/measured`
- unit: `Wb` (data dictionary: `Wb`)
- data-dictionary text: Measured value
- name description: Signed poloidal magnetic flux linked by an individual flux loop, defined by the magnetic field threading an oriented surface bounded by that loop.
- Leaf text is boilerplate; the container `flux_loop` and the unit `Wb` fix the quantity, and the name states both the quantity and the object that links it. The description supplies the sign convention (an oriented surface), which is the part a reader cannot recover from the name.

### 62. `poloidal_magnetic_flux_of_flux_loop` — **correct**

- cohort index: 62
- source path: `equilibrium/time_slice/constraints/flux_loop/reconstructed`
- unit: `Wb` (data dictionary: `Wb`)
- data-dictionary text: Value calculated from the reconstructed equilibrium
- name description: Signed poloidal magnetic flux linked by an individual flux loop, defined by the magnetic field threading an oriented surface bounded by that loop.
- Same identity as row 61 at the reconstructed locus; the provenance rule of row 58 applies unchanged.

### 63. `line_integrated_electron_number_density` — **INCORRECT**

- cohort index: 63
- source path: `equilibrium/time_slice/constraints/n_e_line/measured`
- unit: `m^-2` (data dictionary: `m^-2`)
- data-dictionary text: Measured value
- name description: Free-electron column density accumulated along a complete electromagnetic propagation path, including both forward and return segments when present.
- **rejected spelling** `line_integrated_electron_number_density` → **proposed spelling** `line_integrated_electron_density`
- why: the cohort spells this physical base `electron_density` in eight names and `electron_number_density` in this one identity alone. The semantic content is identical — a number density per volume, integrated along a path — so the extra segment marks no distinction and leaves one published batch carrying two spellings of one base. The majority spelling is the survivor.
- as with row 60, the other half recorded this defect at the `reconstructed` sibling; it reappears here because the identity, not the path, carries it.

### 64. `radial_coordinate_of_magnetic_axis` — **INCORRECT**

- cohort index: 64
- source path: `equilibrium/time_slice/contour_tree/node/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate locating the magnetic-axis O-point in the right-handed cylindrical (R, φ, Z) frame around which nested closed flux surfaces are organized.
- **rejected spelling** `radial_coordinate_of_magnetic_axis` → **proposed spelling** `radial_coordinate_of_flux_map_critical_point`
- why: the bound path is an indexed **node of the contour tree** of the poloidal flux map, not the magnetic axis. A contour tree has many nodes — the critical points of ψ, comprising O-points and X-points and the saddles that join them — whereas an equilibrium has exactly one magnetic axis. A name asserting "magnetic axis" over an array whose members are, by construction, not all the magnetic axis is bound to the wrong object, and the error is not recoverable by a reader: every node in the array arrives carrying a claim to be the O-point. The DD leaf text here is the generic `Major radius` and says nothing that supports the stronger claim; the name's specificity comes from nowhere in the row.
- this identity is correctly bound elsewhere. `equilibrium/time_slice/global_quantities/magnetic_axis/r` genuinely is the magnetic axis and the other half judged it correct; the repair is to detach this locus, not to rename the identity. The identity is additionally bound to `summary/boundary/magnetic_axis_r/value` and `summary/local/magnetic_axis/position/r`, both genuine axis loci; the collision itself belongs to the whole-cohort sweep.

### 65. `poloidal_plane_cross_sectional_area_of_plasma_boundary` — **correct**

- cohort index: 65
- source path: `equilibrium/time_slice/global_quantities/area`
- unit: `m^2` (data dictionary: `m^2`)
- data-dictionary text: Area of the LCFS poloidal cross section
- name description: Area enclosed by the last closed magnetic flux surface in a poloidal plane, defining the equilibrium plasma cross-sectional size at the plasma boundary.
- The name names the surface the DD names — the LCFS, spelled `plasma_boundary` — and distinguishes the poloidal-plane area from the toroidal surface area, which is a different quantity with the same unit. This row and row 80 together establish that the cohort *does* distinguish a global boundary scalar from a per-surface profile; that convention is what row 79 departs from.

### 66. `poloidal_beta` — **correct**

- cohort index: 66
- source path: `equilibrium/time_slice/global_quantities/beta_pol`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2]
- name description: Poloidal beta is a dimensionless measure of total plasma pressure relative to the magnetic-pressure scale of the plasma-current-generated poloidal field.
- collision outside this range: also bound to `summary/global_quantities/beta_pol_mhd/value`; deferred to the whole-cohort sweep.

### 67. `normalized_toroidal_beta` — **correct**

- cohort index: 67
- source path: `equilibrium/time_slice/global_quantities/beta_tor_norm`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Normalized toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA]
- name description: Normalized toroidal beta is a dimensionless whole-plasma equilibrium measure formed from volume-averaged total perpendicular pressure and toroidal magnetic and plasma-current scales.
- The Troyon-normalised form is what the DD defines and what the name claims; `normalized` is load-bearing, since the unnormalised `toroidal_beta` is a separate identity the other half judged correct at the `beta_tor` sibling.

### 68. `normalized_toroidal_beta` — **correct**, with a duplicate producer binding

- cohort index: 68
- source path: `equilibrium/time_slice/global_quantities/beta_tor_norm`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Normalized toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA]
- name description: Normalized toroidal beta is a dimensionless whole-plasma equilibrium measure formed from volume-averaged total perpendicular pressure and toroidal magnetic and plasma-current scales.
- note: **this row and row 67 are the same binding recorded twice.** Every field of the two rows is byte-identical except `index`, and the cohort's own collision list for this identity repeats `equilibrium/time_slice/global_quantities/beta_tor_norm` twice. Counted across the whole 255-row remainder, this is the **only** duplicated `(name, path)` pair, so the cohort is not generally double-counting and this is one duplicate edge rather than a systematic inflation. The name is correct at both; what is wrong is that one identity holds two producer bindings to one source path, which means the accepted-binding count of 341 is one higher than the number of distinct bindings. Reported as a graph-hygiene follow-on, not as a name verdict — no reader of the catalog can see it.

### 69. `mhd_energy` — **INCORRECT**

- cohort index: 69
- source path: `equilibrium/time_slice/global_quantities/energy_mhd`
- unit: `J` (data dictionary: `J`)
- data-dictionary text: Plasma energy content = 3/2 * int(p,dV) with p being the total pressure (thermal + fast particles) [J]. Time-dependent; Scalar
- name description: Global plasma stored energy obtained from the volume integral of total kinetic pressure, including thermal and fast-particle pressure contributions.
- **rejected spelling** `mhd_energy` → **proposed spelling** `total_plasma_stored_energy`
- why: the DD text is unambiguous that this is the *kinetic* energy content, three halves of the volume-integrated pressure, thermal plus fast particles. `mhd_energy` does not say that. In a catalog whose neighbouring identities include poloidal and toroidal magnetic flux, "MHD energy" reads at least as naturally as the magnetic energy of the configuration, which is a different quantity with the same unit and a comparable magnitude. The name also carries an unexpanded acronym, which this batch otherwise refuses — it spells `ion_cyclotron_heating` rather than `ich` and `electron_cyclotron` rather than `ece` wherever the quantity, not the instrument, is being named.
- the proposed spelling reuses a convention the cohort already fixed rather than minting one: `total_` marks the sum over populations in `total_electron_density` (thermal plus non-thermal) and `total_plasma_pressure` (row 98), which is exactly the thermal-plus-fast inclusion the DD states here. It also leaves room for the thermal-only stored energy to be named without collision, which `mhd_energy` does not.
- collision outside this range: also bound to `summary/global_quantities/energy_mhd`, the same quantity; the rename must carry both.

### 70. `poloidal_length_of_flux_surface` — **correct**

- cohort index: 70
- source path: `equilibrium/time_slice/global_quantities/length_pol`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Poloidal length of the magnetic surface
- name description: Poloidal arc length of a closed magnetic-flux-surface contour in an equilibrium poloidal cross-section, measuring the perimeter of that selected surface.
- The name says what the DD text says. This row sits at `global_quantities`, so in practice the surface is the boundary — but the DD text itself says "the magnetic surface" and not "the plasma boundary", and the name agrees with the text it is bound to. That is the line this audit draws, and it is the same line the other half drew in accepting `surface_area_of_flux_surface` at `global_quantities/surface`, whose text likewise reads "the toroidal flux surface". Row 79 is judged differently because its DD text does not agree with its name.

### 71. `normalized_plasma_internal_inductance` — **correct**, with a defect that is not in the name

- cohort index: 71
- source path: `equilibrium/time_slice/global_quantities/li_3`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Internal inductance
- name description: Dimensionless li_3 parameter measuring poloidal magnetic-field peaking through a plasma-volume integral normalized by total plasma current and a reference major radius.
- The name is a correct, self-descriptive reading of the DD text: a dimensionless measure of how peaked the poloidal field is inside the plasma.
- note: the **description** identifies the definition only by echoing the DD field spelling, "the li_3 parameter", which tells a reader nothing unless they hold the path — the thing the name is supposed to make unnecessary. Three conventional definitions of internal inductance are in use and they differ numerically by several percent to tens of percent on the same discharge; the description should state which volume and which normalising radius it integrates over rather than naming the DD field. A description defect, so not an incorrect verdict.
- follow-on: within this batch both bindings of the identity are `li_3` (the other is `summary/global_quantities/li_3/value`), so no two-quantity collision exists today. It would arise the moment a source supplying `li_1` or `li_2` is bound to the same identity, which the name as spelled would not refuse.

### 72. `toroidal_magnetic_field_at_magnetic_axis` — **correct**

- cohort index: 72
- source path: `equilibrium/time_slice/global_quantities/magnetic_axis/b_field_phi`
- unit: `T` (data dictionary: `T`)
- data-dictionary text: Total toroidal magnetic field at the magnetic axis
- name description: Toroidal magnetic field at the magnetic axis is the signed toroidal component of the total equilibrium magnetic field at the magnetic axis in the right-handed cylindrical (R, φ, Z) frame. It includes externally applied vacuum-field and plasma-current-generated contributions.
- The description's "total" matches the DD's "Total", meaning vacuum plus plasma contributions — the distinction that separates this from the vacuum reference field `b0`, which the catalog names separately.

### 73. `vertical_coordinate_of_magnetic_axis` — **correct**

- cohort index: 73
- source path: `equilibrium/time_slice/global_quantities/magnetic_axis/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height of the magnetic axis
- name description: Signed vertical Z coordinate of the magnetic axis in the right-handed cylindrical (R, φ, Z) frame, marking the interior extremum organizing nested magnetic flux surfaces.
- Bound to a genuine `magnetic_axis` container, unlike row 64, and its R sibling at the same container was accepted in the other half — the point is one locus.

### 74. `poloidal_magnetic_flux_at_magnetic_axis` — **correct**

- cohort index: 74
- source path: `equilibrium/time_slice/global_quantities/psi_axis`
- unit: `Wb` (data dictionary: `Wb`)
- data-dictionary text: Poloidal flux at the magnetic axis
- name description: Signed poloidal magnetic flux evaluated at the magnetic axis, providing the inner reference value for normalized poloidal-flux coordinates.
- Same identity the other half accepted at `core_profiles/profiles_1d/grid/psi_magnetic_axis`, which is the collision partner; the two paths carry one quantity.

### 75. `poloidal_magnetic_flux_at_plasma_boundary` — **correct**

- cohort index: 75
- source path: `equilibrium/time_slice/global_quantities/psi_boundary`
- unit: `Wb` (data dictionary: `Wb`)
- data-dictionary text: Poloidal flux at the selected plasma boundary
- name description: Signed poloidal magnetic flux evaluated on the last closed flux surface, providing the outer reference for normalized poloidal-flux coordinates.
- The `_at_magnetic_axis` / `_at_plasma_boundary` pair is spelled symmetrically with row 74 and the two together are what row 55's normalisation is referred to.

### 76. `safety_factor_at_magnetic_axis` — **correct**

- cohort index: 76
- source path: `equilibrium/time_slice/global_quantities/q_axis`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: q at the magnetic axis
- name description: Limiting signed field-line winding number on the innermost closed flux surface, giving toroidal turns per poloidal circuit at the magnetic axis.
- "Limiting" is the right word: the safety factor on the axis is a limit of the profile, since the poloidal circuit degenerates there. The name expands the DD's bare `q`, which is the self-descriptiveness the catalog requires.

### 77. `normalized_toroidal_flux_coordinate_at_minimum_absolute_safety_factor` — **correct**

- cohort index: 77
- source path: `equilibrium/time_slice/global_quantities/q_min/rho_tor_norm`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Minimum q position in normalised toroidal flux coordinate
- name description: Dimensionless magnetic-surface label giving the normalized toroidal flux at the closed surface where the magnitude of the safety factor reaches its minimum.
- Long, but every segment is load-bearing: the quantity is a surface label, the label is the normalized toroidal flux coordinate, and the surface is selected by an extremum of the safety factor. The name and its description agree that the extremum is of the **magnitude**, which matters in a catalog where the safety factor is signed.
- note: its value sibling, row 78, drops the word `absolute` and so does not read as the same locus. See that row.

### 78. `minimum_safety_factor` — **INCORRECT**

- cohort index: 78
- source path: `equilibrium/time_slice/global_quantities/q_min/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Minimum q value
- name description: Signed safety-factor value on the closed magnetic flux surface where the magnitude of field-line winding is smallest, distinct from that surface’s location.
- **rejected spelling** `minimum_safety_factor` → **proposed spelling** `minimum_absolute_safety_factor`
- why: rows 77 and 78 are the two children of one DD container, `q_min` — one is where the minimum occurs and the other is what it is. As spelled, they do not read as one object: the location says the minimum is of the *magnitude* of the safety factor and the value says it is of the safety factor itself. Those are different extrema whenever the safety factor is negative or changes sign, which is exactly the case a signed-safety-factor catalog exists to handle, and a reader holding both names cannot tell that they refer to one surface.
- the qualifier the name drops is supplied by both of its neighbours: by its sibling's spelling, and by its own description, which says the surface is the one "where the magnitude of field-line winding is smallest". Two independent statements of the magnitude reading against one name that omits it is what decides the direction of the repair.
- if instead the signed reading is the intended one, then the pair is still not publishable and the repair falls on row 77 rather than here — the two names cannot both stand as spelled. This audit takes the magnitude reading for the reasons above and records the alternative so the decision is visible rather than assumed.

### 79. `volume_of_flux_surface` — **INCORRECT**

- cohort index: 79
- source path: `equilibrium/time_slice/global_quantities/volume`
- unit: `m^3` (data dictionary: `m^3`)
- data-dictionary text: Total plasma volume
- name description: Geometric volume contained within a nested magnetic flux surface, cumulative from the magnetic axis toward the outermost closed surface.
- **rejected spelling** `volume_of_flux_surface` → **proposed spelling** `volume_of_plasma_boundary`
- why: this is a global scalar whose DD text says **"Total plasma volume"**, while the name and description say a per-surface profile value, "cumulative from the magnetic axis toward the outermost closed surface". One equilibrium has one total plasma volume and a whole profile of enclosed volumes; binding the profile identity to the scalar makes the two indistinguishable in the catalog.
- the cohort already fixes both spellings and already applies the distinction: `volume_of_plasma_boundary` is the accepted name for the same physical scalar at `summary/global_quantities/volume/value`, and in this very block `global_quantities/area` is spelled `..._of_plasma_boundary` (row 65) while `profiles_1d/area` is spelled `..._of_flux_surface` (row 80). The area pair is the positive control: the convention exists, is applied one container away, and this row is where it lapsed.
- this does **not** reopen the settled `*_of_flux_surface` family adjudication. That adjudication fixed what the suffix *means* — the volume and area enclosed by the surface rather than of the surface — and it is taken as given here. The finding is about which locus the identity is attached to, not about its semantics.
- the identity is correctly bound at `equilibrium/time_slice/profiles_1d/volume` and `core_profiles/profiles_1d/grid/volume`, both genuine per-surface profiles accepted in the other half; as with row 64 the repair detaches this locus rather than renaming the identity, and the collision itself belongs to the whole-cohort sweep.

### 80. `poloidal_plane_cross_sectional_area_of_flux_surface` — **correct**

- cohort index: 80
- source path: `equilibrium/time_slice/profiles_1d/area`
- unit: `m^2` (data dictionary: `m^2`)
- data-dictionary text: Cross-sectional area of the flux surface
- name description: Geometric area enclosed by a closed magnetic-flux-surface contour in a fixed-toroidal-angle poloidal plane of the right-handed cylindrical (R, φ, Z) frame.
- The plane is named, which is what separates this from the toroidal surface area at the same unit. With row 65 this is the pair that shows the cohort distinguishes the per-surface profile from the global boundary scalar.
- collision outside this range: also bound to `core_profiles/profiles_1d/grid/area`, the same per-surface quantity used as a grid label; deferred to the whole-cohort sweep.

### 81. `flux_surface_averaged_magnetic_field_magnitude` — **correct**

- cohort index: 81
- source path: `equilibrium/time_slice/profiles_1d/b_field_average`
- unit: `T` (data dictionary: `T`)
- data-dictionary text: Flux surface averaged modulus of B (always positive, irrespective of the sign convention for the B-field direction).
- name description: Flux-surface average of the local magnetic-field strength, retaining the pointwise modulus before averaging over each magnetic surface.
- `magnitude` carries the DD's "modulus", and the description states the order of operations — modulus first, then average — which is the distinction that makes the result sign-convention independent as the DD says.

### 82. `minimum_over_flux_surface_magnetic_field_magnitude` — **correct**

- cohort index: 82
- source path: `equilibrium/time_slice/profiles_1d/b_field_min`
- unit: `T` (data dictionary: `T`)
- data-dictionary text: Minimum(modulus(B)) on the flux surface (always positive, irrespective of the sign convention for the B-field direction)
- name description: Smallest total magnetic-field strength attained along a specified nested magnetic flux surface, using the field magnitude rather than a signed component.
- Every element of the DD text is in the name: the extremum, the domain it is taken over, and that it is the modulus being extremised.
- note: this is the low-field-side counterpart of `b_field_max` in the same container, which the other half accepted as `maximum_magnetic_field_magnitude` — without the `over_flux_surface` segment, leaving the domain to its description ("a specified evaluation domain"). The minimum and the maximum of one container are therefore spelled asymmetrically, and only one of the two states the domain. This row is the better-specified of the pair and is correct on its own terms; the asymmetry is recorded as a cohort-consistency follow-on rather than as a defect here, since repairing it means changing the row the other half already accepted.

### 83. `derivative_of_area_of_flux_surface_with_respect_to_poloidal_magnetic_flux_coordinate` — **correct**

- cohort index: 83
- source path: `equilibrium/time_slice/profiles_1d/darea_dpsi`
- unit: `Wb^-1.m^2` (data dictionary: `Wb^-1.m^2`)
- data-dictionary text: Radial derivative of the cross-sectional area of the flux surface with respect to psi
- name description: Derivative of the poloidal cross-sectional area enclosed by a closed magnetic flux surface with respect to the signed poloidal magnetic flux coordinate. It is a flux-coordinate metric for how enclosed area changes between neighboring flux surfaces.
- The differentiation variable is spelled out rather than compressed into "radial", which is the convention the other half held up when rejecting `radial_derivative_of_poloidal_magnetic_flux` at the neighbouring `dpsi_drho_tor`. The unit confirms the variable: `m^2` divided by `Wb`.
- observation, because it could have come out otherwise: this is the exact spelling `the-flux-surface-area-derivative.md` records as the intended target for this source, back when the source had **no** standard name and the escalation to a vendor-diverse composer seat had failed with grammar-invalid output. The row shows an accepted binding carrying that target spelling, so the gap that file reports is closed, and closed on the intended name rather than on a near miss.

### 84. `derivative_of_area_of_flux_surface_with_respect_to_toroidal_flux_coordinate` — **correct**

- cohort index: 84
- source path: `equilibrium/time_slice/profiles_1d/darea_drho_tor`
- unit: `m` (data dictionary: `m`)
- name description: Radial rate of change of the poloidal cross-sectional area enclosed by a nested magnetic flux surface as the toroidal flux coordinate increases.
- data-dictionary text: Radial derivative of the cross-sectional area of the flux surface with respect to rho_tor
- The unit discriminates the differentiation variable and confirms the name: `m^2` per metre of `rho_tor` is `m`, which is only consistent with the dimensionful toroidal flux coordinate, not its normalized form. `toroidal_flux_coordinate` is the spelling row 102 fixes for `rho_tor`, so the pair is consistent.

### 85. `derivative_of_volume_of_flux_surface_with_respect_to_poloidal_magnetic_flux_coordinate` — **correct**

- cohort index: 85
- source path: `equilibrium/time_slice/profiles_1d/dvolume_dpsi`
- unit: `Wb^-1.m^3` (data dictionary: `Wb^-1.m^3`)
- data-dictionary text: Radial derivative of the volume enclosed in the flux surface with respect to Psi
- name description: Geometric rate of change of the cumulative volume enclosed by a nested magnetic flux surface as its signed poloidal flux coordinate varies.
- `m^3` per `Wb` matches the stated derivative, and the enclosed-volume reading is the settled family convention, not reopened here.

### 86. `derivative_of_volume_of_flux_surface_with_respect_to_toroidal_flux_coordinate` — **correct**

- cohort index: 86
- source path: `equilibrium/time_slice/profiles_1d/dvolume_drho_tor`
- unit: `m^2` (data dictionary: `m^2`)
- data-dictionary text: Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor
- name description: Geometric rate at which the cumulative volume enclosed by a nested magnetic flux surface changes with its dimensionful toroidal-flux coordinate.
- `m^3` per metre is `m^2`, again consistent only with the dimensionful coordinate, which the description states in as many words. Rows 83 to 86 form a complete and internally consistent derivative family over two quantities and two coordinates.

### 87. `elongation_of_flux_surface` — **correct**

- cohort index: 87
- source path: `equilibrium/time_slice/profiles_1d/elongation`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Elongation
- name description: Ratio of the vertical to horizontal semi-axes of a nested magnetic flux surface’s poloidal cross-section, independent of its overall size.
- The DD text is the bare word; the name supplies the surface the shape parameter belongs to, which is the whole reason a shape parameter needs a surface segment — the boundary elongation and a mid-radius elongation are different numbers and the catalog carries both.

### 88. `flux_surface_averaged_inverse_of_square_of_major_radius` — **correct**

- cohort index: 88
- source path: `equilibrium/time_slice/profiles_1d/gm1`
- unit: `m^-2` (data dictionary: `m^-2`)
- data-dictionary text: Flux surface averaged 1/R^2
- name description: Flux-surface average of reciprocal squared major radius, providing a geometric metric coefficient for parallel-gradient and neoclassical transport calculations.
- The name reconstructs the DD formula exactly and replaces an opaque field spelling — `gm1` carries no meaning at all to a reader — with one that does. The unit confirms the reading.

### 89. `flux_surface_averaged_ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_major_radius` — **correct**

- cohort index: 89
- source path: `equilibrium/time_slice/profiles_1d/gm2`
- unit: `m^-2` (data dictionary: `m^-2`)
- data-dictionary text: Flux surface averaged grad_rho^2/R^2
- name description: Flux-surface-averaged magnetic-geometry coefficient measuring the squared toroidal-flux-coordinate gradient relative to the squared major radius.
- Long, and every segment is load-bearing: numerator, denominator, that both are squared, and that the whole ratio is averaged rather than the average being ratioed. The unit is the check — a dimensionless squared gradient of the dimensionful `rho_tor` over `R^2` gives `m^-2`, which is what the row carries.

### 90. `flux_surface_averaged_square_of_toroidal_flux_coordinate_gradient_magnitude` — **correct**, with a defect that is not in the name

- cohort index: 90
- source path: `equilibrium/time_slice/profiles_1d/gm3`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Flux surface averaged grad_rho^2
- name description: Flux-surface average of the squared magnitude of the gradient of the normalized toroidal-flux radius. It is the averaged contravariant radial metric coefficient for the flux coordinate rho_tor.
- The name is correct and consistent with rows 89, 92 and 93, which take the gradient of the same coordinate.
- note: the **description contradicts itself and the unit.** Its first sentence says the gradient is of the *normalized* toroidal-flux radius; its second says the coordinate is `rho_tor`, which is the dimensionful one. The unit settles it: the gradient of the dimensionful coordinate is dimensionless, so its square is `1` as recorded, whereas the gradient of the normalized coordinate would carry `m^-1` and its square `m^-2`. The first sentence is the wrong one. A description defect, so not an incorrect verdict — but it is the kind that propagates, because a reader who trusts it will mis-dimension every transport coefficient built on this metric.

### 91. `flux_surface_averaged_square_of_magnetic_field_magnitude` — **correct**

- cohort index: 91
- source path: `equilibrium/time_slice/profiles_1d/gm5`
- unit: `T^2` (data dictionary: `T^2`)
- data-dictionary text: Flux surface averaged B^2
- name description: Flux-surface average of the squared magnitude of the total equilibrium magnetic field, providing a geometric coefficient for equilibrium and neoclassical transport relations.
- Square-then-average, in that order, which is what the DD writes and what makes the quantity distinct from the square of row 81.

### 92. `flux_surface_averaged_ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_magnetic_field_magnitude` — **correct**

- cohort index: 92
- source path: `equilibrium/time_slice/profiles_1d/gm6`
- unit: `T^-2` (data dictionary: `T^-2`)
- data-dictionary text: Flux surface averaged grad_rho^2/B^2
- name description: Flux-surface average of the squared toroidal-flux-coordinate gradient magnitude divided by the squared local magnetic-field magnitude, a metric coefficient for equilibrium geometry.
- Same construction as row 89 with the denominator changed, and spelled the same way; the unit follows from a dimensionless numerator over `T^2`.

### 93. `flux_surface_averaged_toroidal_flux_coordinate_gradient_magnitude` — **correct**

- cohort index: 93
- source path: `equilibrium/time_slice/profiles_1d/gm7`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Flux surface averaged grad_rho
- name description: Geometric metric coefficient formed by averaging the local magnitude of the toroidal-flux-coordinate gradient over each nested magnetic flux surface.
- The unsquared counterpart of row 90, spelled by dropping exactly the `square_of` segment, and dimensionless for the same reason.

### 94. `flux_surface_averaged_inverse_of_major_radius` — **correct**

- cohort index: 94
- source path: `equilibrium/time_slice/profiles_1d/gm9`
- unit: `m^-1` (data dictionary: `m^-1`)
- data-dictionary text: Flux surface averaged 1/R
- name description: Flux-surface average of reciprocal major radius, providing a geometric equilibrium coefficient for toroidal geometry in flux-coordinate equations.
- The unsquared counterpart of row 88, spelled by dropping `square_of`. Rows 88 to 94 with the two the other half judged (`gm4`, `gm8`) form one family in which the name is always reconstructible from the DD formula and never from the field spelling.

### 95. `flux_surface_averaged_parallel_current_density` — **correct**

- cohort index: 95
- source path: `equilibrium/time_slice/profiles_1d/j_parallel`
- unit: `A.m^-2` (data dictionary: `A.m^-2`)
- data-dictionary text: Flux surface averaged parallel current density = average(j.B) / B0, where B0 = Equilibrium/Global/Toroidal_Field/B0
- name description: Flux-surface-averaged parallel current density is the signed scalar defined on each magnetic flux surface as the flux-surface average of the plasma current-density vector dotted with the magnetic-field vector, divided by a single global reference field B0.
- The name is what the DD text calls the quantity, word for word.
- note: the DD's own definition, `<j·B>/B0`, is not literally the flux-surface average of the parallel current density — that would be `<j·B/|B|>`, and the two differ by the variation of `|B|` over the surface, which on a tokamak flux surface is of order the inverse aspect ratio. The name inherits an imprecision that originates in the data dictionary rather than in the composition, and the description is the part that gets it right, spelling out the dot product and the single global normalising field. Correct against its source; the imprecision is recorded so that a later decision to depart from the DD wording is a decision rather than an accident.

### 96. `flux_surface_averaged_toroidal_current_density` — **correct**

- cohort index: 96
- source path: `equilibrium/time_slice/profiles_1d/j_phi`
- unit: `A.m^-2` (data dictionary: `A.m^-2`)
- data-dictionary text: Flux surface averaged toroidal current density = average(j_tor/R) / average(1/R)
- name description: Geometry-weighted flux-surface average of the net conventional toroidal current density driving the equilibrium poloidal magnetic-flux distribution.
- The description carries the one thing the name cannot: that the average is `1/R`-weighted rather than plain. That is the right division of labour — the name states the quantity, the description states the convention.

### 97. `toroidal_magnetic_flux` — **correct**

- cohort index: 97
- source path: `equilibrium/time_slice/profiles_1d/phi`
- unit: `Wb` (data dictionary: `Wb`)
- data-dictionary text: Toroidal flux
- name description: Toroidal magnetic flux is the signed surface integral of the magnetic field over the poloidal cross-section enclosed by a nested magnetic flux surface.
- The name expands the DD's bare "Toroidal flux" to say which field is being fluxed, which is the ambiguity worth removing in a catalog that also carries particle and heat fluxes.
- note: this is a `profiles_1d` quantity — one value per flux surface — and the name carries no surface segment, while its poloidal counterpart in the same container is spelled `poloidal_magnetic_flux_at_flux_surface` (row 99). The two are the same kind of object named with and without their locus. Neither name is wrong, and the toroidal flux is arguably intrinsic to the surface that encloses it, but a reader meeting `toroidal_magnetic_flux` beside `poloidal_magnetic_flux_at_flux_surface` will reasonably infer a distinction that is not there. Recorded as a cohort-consistency follow-on.

### 98. `total_plasma_pressure` — **correct**

- cohort index: 98
- source path: `equilibrium/time_slice/profiles_1d/pressure`
- unit: `Pa` (data dictionary: `Pa`)
- data-dictionary text: Pressure
- name description: Total plasma pressure is the isotropic kinetic pressure summed over all represented plasma particle species and populations for equilibrium force balance.
- The DD leaf text is the bare word `Pressure` and the container text was not captured, so the reading rests on the locus: the pressure profile of an equilibrium time slice is the one that enters force balance, and force balance is satisfied by the total kinetic pressure, not by any single species. The name's `total_` is therefore supported by where it is bound, and the description says exactly which sum is meant.
- `total_` is used here in the same sense the cohort fixed in `total_electron_density` — summed over populations rather than restricted to the thermal one — which is the sense row 69's proposed spelling also reuses.

### 99. `poloidal_magnetic_flux_at_flux_surface` — **correct**

- cohort index: 99
- source path: `equilibrium/time_slice/profiles_1d/psi`
- unit: `Wb` (data dictionary: `Wb`)
- data-dictionary text: Poloidal flux
- name description: Signed poloidal magnetic flux assigned to a nested magnetic surface, serving as the equilibrium label for its position in the plasma.
- The surface is named, which distinguishes this profile from the two scalars at rows 74 and 75 that take the same quantity at the axis and the boundary. The three together are spelled consistently on one pattern.
- collision outside this range: also bound to `core_profiles/profiles_1d/grid/psi`, the same quantity used as a grid label; deferred to the whole-cohort sweep.

### 100. `radial_coordinate_at_inboard_midplane` — **INCORRECT**

- cohort index: 100
- source path: `equilibrium/time_slice/profiles_1d/r_inboard`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Radial coordinate (major radius) on the inboard side of the magnetic axis
- name description: Major-radius coordinate of a magnetic flux surface at its inboard midplane intersection, locating the high-field-side point relative to the toroidal symmetry axis.
- **rejected spelling** `radial_coordinate_at_inboard_midplane` → **proposed spelling** `radial_coordinate_of_flux_surface_at_inboard_midplane`
- why: the name never says what it is the inboard midplane radius **of**. Read without the path it is a machine geometry constant — the inner wall, the inner limiter, the vessel — and there are several such radii a tokamak catalog will eventually need to carry. What it actually is is a per-surface profile: the high-field-side intersection of each nested flux surface, one value per surface. The object is in the description and nowhere in the name, which is the self-descriptiveness test failing in the same way the other half's `faraday_angle` and `voltage_of_mass_spectrometer_channel` failed it.
- the block itself shows the omission is an outlier rather than a convention: every other `profiles_1d` row here names its object — `elongation_of_flux_surface`, `poloidal_plane_cross_sectional_area_of_flux_surface`, `poloidal_magnetic_flux_at_flux_surface`, and the seven `flux_surface_averaged_*` metric coefficients. The proposed spelling is built from segments the cohort already uses, `..._of_flux_surface` and `..._at_<locus>`, and mints nothing new.
- note, separately from the verdict: the DD text says "on the inboard side of the **magnetic axis**", so the midplane meant is the horizontal plane through the magnetic axis, not the machine equatorial plane. Those differ by the vertical position of the axis, which is a controlled and time-varying quantity carried at row 73. The description says only "midplane" and should say which one.

### 101. `radial_coordinate_at_outboard_midplane` — **INCORRECT**

- cohort index: 101
- source path: `equilibrium/time_slice/profiles_1d/r_outboard`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Radial coordinate (major radius) on the outboard side of the magnetic axis
- name description: Major-radius coordinate of the low-field-side intersection between a magnetic flux surface and the equatorial midplane.
- **rejected spelling** `radial_coordinate_at_outboard_midplane` → **proposed spelling** `radial_coordinate_of_flux_surface_at_outboard_midplane`
- why: the low-field-side counterpart of row 100 and defective in exactly the same way — the object whose midplane radius this is appears only in the description. The pair must be repaired together; repairing one would leave the inboard and outboard radii of one surface spelled on two different patterns, which is worse than the present state.
- note: this row's description says "the equatorial midplane" where the DD text says the inboard/outboard side of the **magnetic axis**. Its inboard partner says only "midplane". So the two descriptions disagree with each other about which plane is meant, and the one that is specific is specific in the direction the DD does not support. A description defect in both, resolved by stating the axis-height plane in each.

### 102. `toroidal_flux_coordinate` — **correct**

- cohort index: 102
- source path: `equilibrium/time_slice/profiles_1d/rho_tor`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0
- name description: Non-negative, radius-like label of a nested magnetic flux surface, derived from enclosed toroidal magnetic flux using a positive reference vacuum toroidal field.
- The name is the DD's own phrase, and the unit `m` distinguishes it from the normalized form the other half accepted as `normalized_toroidal_flux_coordinate` at `rho_tor_norm`. Both spellings are the ones rows 84, 86, 89, 90, 92 and 93 refer to, so the coordinate is named once and used consistently.
- collision outside this range: also bound to `core_profiles/profiles_1d/grid/rho_tor`, the same coordinate used as a grid label; deferred to the whole-cohort sweep.

## Result

| | count |
| --- | --- |
| rows judged (cohort indices 52–102, inclusive) | **51** |
| judged **correct** | 43 |
| judged **incorrect**, with a proposed spelling | **8** |
| correct + incorrect | **51** |
| incorrect as a percentage of rows judged | **15.7 %** |
| notes on **correct** rows, counted separately | 6 |

**8 of 51 — 15.7 % of this block — are not publishable as spelled.** Grouped
into the five classes the first half of the cohort established, because the
classes have different remedies:

- **The name asserts more than the data supports** (0): none in this block.
- **The name is bound to the wrong object** (2): `radial_coordinate_of_magnetic_axis`
  on an indexed node of the flux map's contour tree (row 64);
  `volume_of_flux_surface` on the global scalar the data dictionary calls the
  total plasma volume (row 79). Both identities are *correct elsewhere*, so the
  remedy is to detach the locus, not to rename.
- **One name, two physically different quantities** (0): none in this block —
  see the note below, because this is the class that nearly had members.
- **Not self-descriptive** (5): `faraday_angle` (row 60), `mhd_energy` (row 69),
  `minimum_safety_factor` (row 78), `radial_coordinate_at_inboard_midplane` and
  `radial_coordinate_at_outboard_midplane` (rows 100–101).
- **Minority spelling of a base the cohort already fixes** (1):
  `line_integrated_electron_number_density` (row 63).

Two of the five classes are empty here, and that is a property of the block
rather than of the judging. Indices 52–102 are one contiguous region of one IDS
— equilibrium boundary, constraints, global quantities and 1-D profiles — where
almost every quantity arrives with an explicit formula in its data-dictionary
text. The over-claim failure the other half found (`surface_temperature` for an
apparent temperature, coupled power named as launched) needs a source whose text
qualifies the measurement, and this block has few.

**The empty two-quantity class was the one at risk.** Rows 58–63 bind single
identities across `measured` and `reconstructed` loci, which is the exact shape
of the split-required defect. They are judged correct because the difference is
provenance rather than physics, and that is not an opinion of this audit: the
repository encodes it in `provenance_verb_check`
(`imas_codex/standard_names/audits.py:1215`), whose docstring states that
standard names describe the physical quantity and not how it was obtained. The
guard was made to refuse rather than assumed to work — `reconstructed_poloidal_magnetic_field`
against a source path lacking the word returns
`audit:provenance_verb_check: name contains 'reconstructed' but source path
does not`, while the unqualified `poloidal_magnetic_field` returns clean at the
same path. The guard also permits the verb when the path carries it, so the
shared identity is permitted rather than compelled; that choice belongs to the
collision sweep.

### The six notes on correct rows

Counted separately because none of them makes a name unpublishable:

- **Description defects** (2): row 71, whose description identifies which
  internal-inductance definition is meant only by echoing the data-dictionary
  field spelling; row 90, whose description says the gradient is of the
  *normalized* toroidal-flux radius in one sentence and of `rho_tor` in the
  next, with the recorded unit `1` showing the first sentence is the wrong one.
- **Cohort-consistency follow-ons** (2): row 82, the flux-surface minimum of
  the field modulus, states the domain it extremises over while the maximum at
  the same container — accepted in the other half — does not; row 97, the
  toroidal flux, carries no surface segment while its poloidal counterpart in
  the same container is spelled `..._at_flux_surface`.
- **An imprecision inherited from the source** (1): row 95's name repeats the
  data dictionary's own wording, under which `<j·B>/B0` is called the
  flux-surface-averaged parallel current density though it is not literally
  that average. Recorded so that departing from the DD wording later is a
  decision rather than an accident.
- **A duplicate producer binding** (1): row 68 is byte-identical to row 67 in
  every field but `index`. It is the only duplicated `(name, path)` pair in the
  255-row remainder, so the 341 accepted bindings comprise 340 distinct ones.
  Invisible to a catalog reader; a graph-hygiene repair.

Two further description defects sit on rows already judged incorrect and so are
not counted again: rows 100 and 101 disagree with each other about which
midplane is meant, and the specific one of the two — "the equatorial midplane"
— is specific in the direction the data-dictionary text does not support, since
that text says the inboard and outboard sides of the **magnetic axis**.

### How this block relates to the rest of the cohort

| | judged | incorrect | rate |
| --- | --- | --- | --- |
| every-fourth sample, `west-name-audit.md` | 86 | 10 | 11.6 % |
| this block, cohort indices 52–102 | 51 | 8 | 15.7 % |
| **both, summed** | **137** | **18** | **13.1 %** |

The two halves are disjoint by construction — the remainder file contains
exactly the 255 bindings the every-fourth sample did not judge — so the rows sum
rather than needing to be reconciled. **The rates do not sum in the same sense.**
The 11.6 % comes from a deterministic sample spread over 20 IDSs and is an
estimate of the whole cohort; the 15.7 % here comes from one contiguous region
of one IDS, so it estimates equilibrium geometry naming and nothing wider. The
two figures are close enough that neither contradicts the other, and the
combined 13.1 % over 137 of 341 bindings is the honest summary, but a
cohort-wide rate should be taken from the sample rather than from this block.

### What is deferred, and to whom

**Twenty-three of the 51 rows** carry an identity that is also bound to at least
one source path outside indices 52–102: rows 52, 57, 58, 59, 60, 61, 62, 63, 64,
66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 79, 80, 99 and 102, counted from the
cohort file's own collision map. The sections above name the partner paths in
place for the rows where the partner bears on the verdict. Every one is recorded
and none is adjudicated here: the collisions belong to the node that owns the
whole-cohort sweep, which is the only vantage from which a shared identity can
be judged across all of its loci at once. Rows 64 and 79 are the two where the
collision and the verdict interact — in both, the identity is sound and one of
its bindings is not — so the sweep needs those verdicts before it can decide the
detachment.

All 51 rows carry a verdict, the counts above close, and the `provisional` line
at the head of this file has been rewritten to `false`.

### One thing this file would be better for and does not have

A poloidal-cross-section figure would carry rows 64, 100 and 101 better than the
prose does: nested flux surfaces with the magnetic axis and the contour tree's
other critical points marked, and the inboard and outboard intersections drawn
on the magnetic-axis plane beside the machine equatorial plane they are not. All
three findings are about *where* something sits, which is the case a figure is
for. It is not here because this node's write scope is the three files it was
fenced to, and a figures directory for this plan is not one of them; writing one
would also race the peer node judging the first half of the same cohort.
Recorded as a follow-on rather than taken.
