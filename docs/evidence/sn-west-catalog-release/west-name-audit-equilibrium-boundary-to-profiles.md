# WEST cohort remainder, indices 52–102 — physical-correctness audit

provisional: true — verdicts are being appended as they are judged; the result
section is not yet closed.

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

### 68. `normalized_toroidal_beta` — **correct**, with a defect that is not in the name

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
