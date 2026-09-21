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
