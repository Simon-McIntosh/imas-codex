# WEST accepted names, cohort indices 154–204 — physical-correctness audit

provisional: true — verdicts are appended row by row as each name is judged;
the closing pass rewrites this line.

This is the second half of the same cohort judged in
[the WEST name audit](west-name-audit.md), in the same shape so the two halves
can be summed rather than compared. That file judged 86 rows drawn as every
fourth row of the path-ordered cohort; this one judges the **51 accepted
bindings whose cohort `index` lies in the inclusive range 154–204** of the
`bindings` list in `west-name-cohort-remainder.json`. The block runs from
`magnetics/b_field_pol_probe/length` to
`spectrometer_visible/channel/line_of_sight/second_point/phi` and covers seven
IDSs — magnetics 10, polarimeter 10, spectrometer_visible 15, soft_x_rays 8,
pf_active 5, spectrometer_mass 2, pf_passive 1.

## The evidence base, and the control on it

No query was issued and no database was opened: every row was already drawn
from the live graph into `west-name-cohort-remainder.json` by the node that
judged the first half, which recorded the instrument faults it had to correct
before any absence could be reported. Each row carries `name`, `path`,
`sn_unit`, `sn_description`, `dd_unit`, `dd_doc` and `dd_doc_parent`, and that
is the whole evidence base for the judgement below.

The control on the slice before judging it: the range holds **51 rows**, which
is the count the measure names; **51 of 51** carry non-empty `dd_doc`, so no
verdict rests on absent documentation, and **0 of 51** needed the
`dd_doc_parent` fallback — none of these paths is a `/value` leaf whose own
documentation is the literal string `Value`. A slice that had come back short,
or with a uniform empty `dd_doc` column, would have been a fault in the slice
predicate rather than a finding about the names.

Each name is judged on three questions: does the data-dictionary text of the
path it is bound to describe the quantity the name claims; do `sn_unit` and
`dd_unit` agree, and where they differ which side is defensible; and is the
name self-descriptive to a reader who does not have the source path in hand.
Section numbers are the cohort indices, not a local count.

Adjudications the first half records as settled are reused, not relitigated:
the `*_of_flux_surface` family, `back_surface_distance_of_antenna_strap`, and
the etendue `_detector` spelling. Where the first half rejected a spelling, the
rejection is carried forward to the same identity here rather than re-argued.

## Verdicts

### 154. `length_of_poloidal_magnetic_field_probe` — **correct**

- source path: `magnetics/b_field_pol_probe/length`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Length of the coil
- name description: Physical extent of a poloidal magnetic-field probe coil measured along its assigned local normal sensing axis, contributing to the coil's magnetic-flux response.
- The DD leaf is the coil's own length and the name binds it to the probe that
  carries the coil. A pickup coil's length is measured along its winding axis,
  which is its sensing direction, so the description's "local normal sensing
  axis" adds a geometric statement the DD does not contradict.

### 155. `poloidal_angle_of_flux_surface` — **INCORRECT**

- source path: `magnetics/b_field_pol_probe/poloidal_angle`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Poloidal angle of the coil orientation
- name description: Pointwise geometric angle locating a position around a magnetic flux-surface cross-section, measured clockwise from its low-field-side midplane.
- **rejected spelling** `poloidal_angle_of_flux_surface` → **proposed spelling** `poloidal_orientation_angle_of_poloidal_magnetic_field_probe`
- why: The DD quantity is the **orientation of a sensor coil** — which way the
  probe's sensitive axis points within the poloidal plane. The name instead
  claims a **position on a magnetic flux surface**, and its description spells
  that claim out in full: an angle "locating a position around a magnetic
  flux-surface cross-section". Those are different kinds of object. A flux
  surface is a plasma equilibrium construct that moves shot to shot and does not
  exist before breakdown; the probe's orientation is a fixed property of the
  hardware, known from the machine drawing. Published as spelled, an
  equilibrium code reading `poloidal_angle_of_flux_surface` would take a
  hardware installation angle for a plasma geometry coordinate.

### 156. `toroidal_angle_of_measurement_position` — **INCORRECT**

- source path: `magnetics/b_field_pol_probe/position/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate locating a measurement position around the machine symmetry axis in the right-handed cylindrical (R, φ, Z) frame.
- **rejected spelling** `toroidal_angle_of_measurement_position` → **proposed spelling** `toroidal_coordinate_of_poloidal_magnetic_field_probe`
- why: Two separate reasons, and either alone is decisive. First, the locus:
  the first half already rejected this identity's radial twin on
  `magnetics/b_field_phi_probe/position/r`, on the ground that a sensor's
  installed location is not a "measurement position" in the sense the ECE
  channel position carries, and proposed
  `radial_coordinate_of_toroidal_magnetic_field_probe`. The same reasoning
  binds here. Second, and visible entirely inside this range: the **`z`
  sibling of this very container** (row 157) is already spelled
  `vertical_coordinate_of_poloidal_magnetic_field_probe`, so φ and Z of one
  probe position currently carry two different loci — a catalog cannot publish
  a point whose φ belongs to one object and whose Z belongs to another. The
  proposed spelling uses `toroidal_coordinate_` rather than `toroidal_angle_`
  deliberately: `toroidal_angle_of_poloidal_magnetic_field_probe` is already
  taken by row 158 for the coil's **orientation**, a different quantity in the
  same DD container.
- The identity is also bound to `ece/channel/position/phi` and
  `magnetics/b_field_phi_probe/position/phi`, both outside indices 154–204. The
  ECE binding is the one genuine measurement position of the three. The
  collision itself is deferred to the node that owns the whole-cohort sweep.

### 157. `vertical_coordinate_of_poloidal_magnetic_field_probe` — **correct**

- source path: `magnetics/b_field_pol_probe/position/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Vertical position of the geometric center of a poloidal magnetic-field probe, expressed as its signed Z coordinate in the right-handed cylindrical (R, φ, Z) frame.
- This is the spelling row 156 should match, and the one the first half's
  proposal for the toroidal-field probe already follows.

### 158. `toroidal_angle_of_poloidal_magnetic_field_probe` — **INCORRECT**

- source path: `magnetics/b_field_pol_probe/toroidal_angle`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle of coil orientation (0 if fully in the poloidal plane)
- name description: Signed azimuthal orientation angle of a poloidal magnetic-field probe's sensitive-axis normal in the right-handed cylindrical (R, φ, Z) frame.
- **rejected spelling** `toroidal_angle_of_poloidal_magnetic_field_probe` → **proposed spelling** `toroidal_orientation_angle_of_poloidal_magnetic_field_probe`
- why: The DD text puts the distinguishing word first — this is the toroidal
  angle **of coil orientation**, not of coil position — and the name drops it.
  The same DD container carries `position/phi` (row 156), which is the probe's
  toroidal *location*. As spelled, one name would have to serve both: where the
  probe sits around the torus, and which way its sensitive axis tilts out of
  the poloidal plane. The two are independent, and for a probe mounted fully in
  the poloidal plane the second is identically zero while the first is not. The
  name's own description has to supply the missing word ("orientation angle of
  … sensitive-axis normal"), which is the self-descriptiveness test failing in
  the same way the first half recorded for `faraday_angle`. With row 155 this
  makes a matched pair: the coil's poloidal and toroidal **orientation** angles.

### 159. `turn_count_of_poloidal_magnetic_field_probe` — **correct**

- source path: `magnetics/b_field_pol_probe/turns`
- unit: `1` (data dictionary: *empty*)
- data-dictionary text: Turns in the coil, including sign
- name description: Signed count of complete winding turns in a poloidal magnetic-field probe coil, fixing the orientation and magnitude of its magnetic-flux linkage.
- note: The units do not agree — the DD declares no unit string at all while the
  standard name carries `1`. The standard name is the defensible side: a turn
  count is dimensionless, and `1` is the explicit spelling of that, where an
  empty string is indistinguishable from a unit the DD simply never filled in.
  The name correctly keeps `signed` in the description rather than the name,
  since the sign is a convention on the count, not a separate quantity.

### 160. `toroidal_magnetic_flux_due_to_diamagnetic_drift` — **correct**

- source path: `magnetics/diamagnetic_flux`
- unit: `Wb` (data dictionary: `Wb`)
- data-dictionary text: Diamagnetic flux. The array of structure corresponds to a set of calculation methods (starting with the generally recommended method).
- name description: Measured toroidal projection of the magnetic-flux contribution produced by plasma diamagnetism relative to a reference field without the pressure-gradient response.
- note: The name attributes the flux to the **diamagnetic drift**, which is a
  fluid drift velocity carrying no net particle flux; what actually perturbs
  the toroidal flux is the plasma magnetization (diamagnetic) **current** that
  the pressure gradient supports. The description says this correctly ("produced
  by plasma diamagnetism"), so the physics recorded in the catalog is right and
  the verdict stays **correct**; the mechanism word in the name is imprecise
  rather than wrong, and `..._due_to_plasma_diamagnetism` would be exact.

### 161. `poloidal_magnetic_flux_of_flux_loop` — **correct**

- source path: `magnetics/flux_loop/flux`
- unit: `Wb` (data dictionary: `Wb`)
- data-dictionary text: Measured flux
- name description: Signed poloidal magnetic flux linked by an individual flux loop, defined by the magnetic field threading an oriented surface bounded by that loop.
- A flux loop encircles the machine toroidally and links the poloidal flux
  through the surface it bounds, so the axis word matches the instrument.
- The identity is also bound to `equilibrium/time_slice/constraints/flux_loop/measured`
  and `/reconstructed`, both outside this range. That sharing is the designed
  behaviour rather than a collision: measured / reconstructed is a provenance
  axis carried on the source binding, never a name segment.

### 162. `radial_coordinate_of_flux_loop` — **correct**

- source path: `magnetics/flux_loop/position/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of each geometric point defining a flux-loop position, measured as perpendicular distance from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.
- The loop position is an array of points, and the description says "each
  geometric point" rather than implying a single centre — correct for a DD leaf
  whose parent is a multi-point outline.

### 163. `plasma_current` — **correct**

- source path: `magnetics/ip`
- unit: `A` (data dictionary: `A`)
- data-dictionary text: Plasma current. Positive sign means anti-clockwise when viewed from above. The array of structure corresponds to a set of calculation methods (starting with the generally recommended method).
- name description: Net toroidal electric current carried by the entire plasma column, obtained by integrating toroidal current density over its enclosed poloidal cross-section.
- The first half judged the same identity correct on
  `equilibrium/time_slice/global_quantities/ip`; the binding here is the
  magnetics measurement of the same physical quantity, and the third binding is
  `summary/global_quantities/ip/value`. One identity across a measurement, an
  equilibrium reconstruction and a summary aggregate is legitimate reuse, not a
  collision, and the sweep node needs no action on it.
