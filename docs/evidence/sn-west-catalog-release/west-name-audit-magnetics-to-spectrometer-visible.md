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

### 164. `current_of_poloidal_field_coil` — **correct**

- source path: `pf_active/coil/current`
- unit: `A` (data dictionary: `A`)
- data-dictionary text: Current in the coil
- name description: Conventional electrical current through a poloidal-field coil winding, with sign set by winding orientation and distinct from the winding's ampere-turn product.
- The one place this quantity is routinely misread is the conductor current
  versus the ampere-turns, and the description names that distinction
  explicitly. It pairs correctly with `effective_turn_count_of_coil_conductor_element`
  (row 168), whose product with this gives the field-producing ampere-turns.

### 165. `height_of_poloidal_field_coil` — **INCORRECT**

- source path: `pf_active/coil/element/geometry/rectangle/height`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Vertical full height
- name description: Full vertical extent of the rectangular cross-section of one active poloidal-field coil conductor, measured parallel to Z in the right-handed cylindrical (R, φ, Z) frame.
- **rejected spelling** `height_of_poloidal_field_coil` → **proposed spelling** `height_of_conductor_cross_section`
- why: The DD leaf is the height of **one rectangular conductor element's
  cross-section**, not of the coil. A poloidal-field coil is an array of such
  elements, and its overall vertical extent is a different number — larger, and
  not recoverable from any single element. The name's own description says
  "cross-section of one … conductor", so it already contradicts the name. The
  proposed spelling is not minted here: the first half **accepted**
  `radial_coordinate_of_conductor_cross_section` for
  `pf_active/coil/element/geometry/rectangle/r`, and row 167 below carries
  `vertical_coordinate_of_conductor_cross_section` for the `z` sibling. Three
  of the five leaves of one DD rectangle already name the cross-section; these
  two name the coil.

### 166. `width_of_poloidal_field_coil` — **INCORRECT**

- source path: `pf_active/coil/element/geometry/rectangle/width`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Horizontal full width
- name description: Full radial extent of the rectangular cross-section of one active poloidal-field coil conductor, measured parallel to the major-radius coordinate R in the right-handed cylindrical (R, φ, Z) frame.
- **rejected spelling** `width_of_poloidal_field_coil` → **proposed spelling** `width_of_conductor_cross_section`
- why: Identical to row 165 on the radial axis. The coil's radial build and one
  element's conductor width differ by the number of elements across the pack,
  and a filament model fed the coil width where it expects the element width
  places the current at the wrong radius.

### 167. `vertical_coordinate_of_conductor_cross_section` — **correct**

- source path: `pf_active/coil/element/geometry/rectangle/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Geometric centre Z
- name description: Signed vertical position of the geometric center of a rectangular conductor cross-section in the right-handed cylindrical (R, φ, Z) frame.
- This is the spelling rows 165 and 166 should match, and it is consistent with
  the `radial_coordinate_of_conductor_cross_section` the first half accepted for
  the `r` sibling of the same rectangle.

### 168. `effective_turn_count_of_coil_conductor_element` — **correct**

- source path: `pf_active/coil/element/turns_with_sign`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Number of effective turns in the element for calculating magnetic fields of the coil/loop; includes the sign of the number of turns (positive means current is counter-clockwise when seen from above)
- name description: Signed effective winding count used to weight the magnetic-field contribution of an individual poloidal-field coil conductor element.
- The name carries the element locus the DD text carries, and `effective_`
  carries the DD's "effective turns … for calculating magnetic fields".
- note: The DD permits this quantity to be fractional (an element may represent
  a homogenised fraction of a winding pack), and a word spelled `count` is read
  by most consumers as an integer. The cohort spells the base `turn_count` in
  three places (rows 159, 168, 169), so the spelling is consistent and the
  remark is a caution for the base rather than a defect in this row.

### 169. `effective_turn_count_of_passive_loop` — **INCORRECT**

- source path: `pf_passive/loop/element/turns_with_sign`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Number of effective turns in the element for calculating magnetic fields of the coil/loop; includes the sign of the number of turns (positive means current is counter-clockwise when seen from above)
- name description: Effective turn count of a passive loop is the signed number of equivalent windings assigned to a passive conducting loop for electromagnetic coupling. It weights the passive-loop current contribution and may be fractional for equivalent or homogenized loop elements.
- **rejected spelling** `effective_turn_count_of_passive_loop` → **proposed spelling** `effective_turn_count_of_passive_loop_element`
- why: The bound path ends in `element/turns_with_sign` and the DD text is
  word-for-word the same string as row 168, which the cohort spells
  `..._of_coil_conductor_element`. Two identically documented per-element leaves
  are given two different granularities — one the element, one the whole loop —
  and a passive loop decomposed into several elements has a loop-level effective
  turn count that is the *sum* of theirs, so the two readings differ by a factor
  equal to the element count. The description already hedges toward the element
  ("may be fractional for equivalent or homogenized loop **elements**") while
  the name claims the loop.

### 170. `initial_polarization_ellipticity_of_polarimeter_beam` — **correct**

- source path: `polarimeter/channel/ellipticity_initial`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Initial ellipticity before entering the plasma
- name description: Signed ratio of the minor to major semiaxes of the electric-field polarization ellipse of a polarimeter probing beam in its incident state, before plasma entry. It gives only the ellipticity component of the initial polarization vector, not the ellipse orientation.
- This is the binding the name was built for: the DD leaf is the beam's initial
  ellipticity and the name says exactly that.
- note: The units agree and both are wrong. An ellipticity is a ratio of two
  semiaxes and is dimensionless, yet the DD declares `m` and the standard name
  has inherited it. Neither side is defensible as spelled; the defect
  originates in the data dictionary, so the standard name cannot repair it
  without diverging from the source it is bound to. It is recorded here for the
  DD rather than counted against the name.

### 171. `faraday_angle` — **INCORRECT**

- source path: `polarimeter/channel/faraday_angle`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Faraday angle (variation of the Faraday angle induced by crossing the plasma)
- name description: Faraday rotation of a probing wave's polarization plane caused by electron density and the magnetic-field component along its plasma path.
- **rejected spelling** `faraday_angle` → **proposed spelling** `faraday_rotation_angle`
- why: Carried forward, not re-argued. The first half rejected this identity on
  `equilibrium/time_slice/constraints/faraday_angle/reconstructed` with the
  same proposed spelling: the Faraday effect is a *rotation of the polarization
  plane*, `faraday_angle` names no physical angle on its own, and the name's own
  description has to supply the missing word. This row is the polarimeter
  binding of that same identity, so one repair fixes both. The DD text here is
  the stronger evidence for the rejection: it has to gloss its own leaf name
  ("variation of the Faraday angle induced by crossing the plasma") because the
  bare phrase does not carry the meaning.

### 172. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `polarimeter/channel/line_of_sight/first_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 173. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `polarimeter/channel/line_of_sight/first_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.

### 174. `toroidal_coordinate_of_line_of_sight` — **correct**

- source path: `polarimeter/channel/line_of_sight/second_point/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of the first reference point on a diagnostic line of sight, locating that point around the machine symmetry axis.
- note: This is the shared line-of-sight description defect the first half
  already recorded, observed again here: the description says "the **first**
  reference point" while this binding is the **second** point. The name is
  correct and deliberately generic over the points; only the description
  over-specifies. No new defect, and no incorrect verdict.

### 175. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `polarimeter/channel/line_of_sight/second_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.

### 176. `toroidal_coordinate_of_line_of_sight` — **correct**

- source path: `polarimeter/channel/line_of_sight/third_point/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of the first reference point on a diagnostic line of sight, locating that point around the machine symmetry axis.
- note: Same recorded description defect as row 174, here against the **third**
  point — the polarimeter is the one WEST diagnostic whose line of sight carries
  three points, so it exhibits the defect most sharply.

### 177. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `polarimeter/channel/line_of_sight/third_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 178. `initial_polarization_ellipticity_of_polarimeter_beam` — **INCORRECT**

- source path: `polarimeter/channel/polarization_initial`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Initial polarization vector before entering the plasma
- name description: Signed ratio of the minor to major semiaxes of the electric-field polarization ellipse of a polarimeter probing beam in its incident state, before plasma entry. It gives only the ellipticity component of the initial polarization vector, not the ellipse orientation.
- **rejected spelling** `initial_polarization_ellipticity_of_polarimeter_beam` → **proposed spelling** `initial_polarization_vector_of_polarimeter_beam`
- why: One name is bound to two physically different quantities, and both
  bindings sit inside this index range, so the collision is judged here rather
  than deferred. Row 170 is the DD leaf `ellipticity_initial` — the ellipse's
  axis ratio, a single number. This row is `polarization_initial`, the DD's
  **polarization vector**, which carries the ellipse orientation as well as its
  shape; that orientation is precisely what a polarimeter's Faraday and
  Cotton–Mouton analysis needs, and it is the component the name's own
  description explicitly disclaims ("not the ellipse orientation"). A consumer
  resolving `initial_polarization_ellipticity_of_polarimeter_beam` cannot tell
  which of the two DD leaves it will receive, and the two are not
  interconvertible: the ellipticity can be recovered from the vector, the vector
  cannot be recovered from the ellipticity.

### 179. `wavelength_of_wave_beam` — **correct**

- source path: `polarimeter/channel/wavelength`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Wavelength used for polarimetry
- name description: Vacuum spatial period of a probing electromagnetic wave propagating as a diagnostic beam, defining its spectral wavelength.
- The identity is also bound to `interferometer/channel/wavelength/value`,
  outside this range. That is legitimate reuse rather than a collision: both are
  the vacuum wavelength of a launched probing beam, and the name is
  deliberately instrument-independent — it describes the beam, which is the
  object both diagnostics share.

### 180. `incident_soft_xray_radiance` — **correct**

- source path: `soft_x_rays/channel/brightness`
- unit: `W.m^-2.sr^-1` (data dictionary: `W.m^-2.sr^-1`)
- data-dictionary text: Power flux received by the detector, per unit solid angle and per unit area (i.e. power divided by the etendue), in multiple energy bands if available from the detector
- name description: Incident soft X-ray power radiance gives received electromagnetic power per projected detector area and solid angle, integrated over the selected energy band.
- The DD leaf is spelled `brightness` and the name is not, which is the right
  way round: the first half rejected `hard_xray_brightness` for exactly this
  reason and proposed `hard_xray_photon_radiance`. The energy/photon
  distinction is carried consistently across the two — this row's unit is
  `W.m^-2.sr^-1`, an **energy** radiance, and it is spelled `radiance`, while
  the hard X-ray sibling at `m^-2.s^-1.sr^-1` is a **photon** radiance and is
  spelled `photon_radiance`. The unit is the discriminator and the names track
  it.
- note: This row carries an `incident_` qualifier that the hard X-ray sibling
  does not. The qualifier is an improvement rather than an inconsistency — it
  is what row 187 below is judged for lacking — but the cohort should apply it
  to one or to both.

### 181. `upper_photon_energy` — **INCORRECT**

- source path: `soft_x_rays/channel/energy_band/upper_bound`
- unit: `eV` (data dictionary: `eV`)
- data-dictionary text: Upper bound of the energy band
- name description: High-energy boundary of an X-ray photon-acceptance band, defining the maximum photon energy included in a selected spectral window.
- **rejected spelling** `upper_photon_energy` → **proposed spelling** `upper_bound_photon_energy`
- why: A minority spelling of a base the cohort already fixes, of the same class
  as `line_integrated_electron_number_density` and `hard_xray_brightness` in the
  first half. The first half **accepted** `lower_bound_photon_energy` for the
  lower edge of the very same kind of DD `energy_band` container, so the two
  edges of one band are published under two grammars — `lower_bound_photon_energy`
  and `upper_photon_energy`. The semantic content is identical and only the
  spelling differs, which makes the majority spelling the survivor. The
  difference is not cosmetic for a consumer that constructs the pair
  programmatically from the base name.
