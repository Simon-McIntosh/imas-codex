# WEST accepted names, cohort indices 154–204 — physical-correctness audit

provisional: false — all 51 rows in cohort indices 154–204 carry a verdict and
the result section is closed.

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

### 182. `etendue_of_soft_xray_detector` — **correct**

- source path: `soft_x_rays/channel/etendue`
- unit: `m^2.sr` (data dictionary: `m^2.sr`)
- data-dictionary text: Etendue (geometric extent) of the channel's optical system
- name description: Geometric optical throughput of a soft X-ray detector channel, set by the collecting area and accepted solid angle of its optical system.
- The `_detector` spelling of the etendue family is recorded as settled and is
  not relitigated here. The row is consistent with the settled adjudication:
  the DD text names the channel's optical system, the description names the
  collecting area and solid angle whose product it is, and the unit `m^2.sr`
  matches that product.

### 183. `toroidal_coordinate_of_line_of_sight` — **correct**

- source path: `soft_x_rays/channel/line_of_sight/first_point/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of the first reference point on a diagnostic line of sight, locating that point around the machine symmetry axis.
- This is the one binding of the shared identity where the description's "first
  reference point" is accurate, since the path is `first_point`.

### 184. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `soft_x_rays/channel/line_of_sight/first_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.

### 185. `toroidal_coordinate_of_line_of_sight` — **correct**

- source path: `soft_x_rays/channel/line_of_sight/second_point/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of the first reference point on a diagnostic line of sight, locating that point around the machine symmetry axis.
- note: The recorded description defect again — bound to the **second** point,
  described as the first.

### 186. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `soft_x_rays/channel/line_of_sight/second_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 187. `power_of_soft_xray_detector` — **INCORRECT**

- source path: `soft_x_rays/channel/power`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: Power received on the detector, in multiple energy bands if available from the detector
- name description: Band-integrated radiant power received by a soft X-ray detector channel over its selected photon-energy band or bands.
- **rejected spelling** `power_of_soft_xray_detector` → **proposed spelling** `incident_power_of_soft_xray_detector`
- why: Not self-descriptive, and ambiguous in a way that matters for exactly
  this instrument. "Power of a detector" reads first as the detector's own
  electrical power — bias supply, preamplifier dissipation, cooling load — all
  of which are real quantities for a soft X-ray diode array and all of which are
  also in watts, so the unit does not disambiguate it. The DD text puts the
  direction first ("Power **received on** the detector") and the name drops it,
  leaving the description to carry the whole meaning. The repair reuses two
  spellings the cohort already holds rather than minting one: `incident_` is
  accepted at row 180 for this same channel's radiance, and `_of_soft_xray_detector`
  is the settled object segment from row 182.

### 188. `atomic_mass` — **correct**

- source path: `spectrometer_mass/channel/a`
- unit: `u` (data dictionary: `u`)
- data-dictionary text: Atomic mass measured by this channel
- name description: Mass parameter assigned to a specified ion or neutral particle species, encoding its isotope or constituent composition for inertial and transport calculations.
- The identity is also bound to
  `spectrometer_visible/channel/isotope_ratios/isotope/element/a` (row 197),
  inside this range. Both are the atomic mass of a species, so the sharing is
  legitimate reuse and needs no action from the collision sweep — the two
  bindings differ in which instrument reports the mass, not in what the mass is.

### 189. `ion_current_of_mass_spectrometer_channel` — **correct**

- source path: `spectrometer_mass/channel/current`
- unit: `A` (data dictionary: `A`)
- data-dictionary text: Collected current
- name description: Collected conventional electrical current carried by the ion population assigned to one mass-resolved channel.
- The first half rejected `voltage_of_mass_spectrometer_channel` for naming a
  channel that carries several distinct voltages. This row does not repeat that
  defect: the `ion_` qualifier names which current is meant, and the DD channel
  has exactly one current leaf, so the object segment resolves uniquely.

### 190. `spectral_rate_of_spectrometer_channel` — **INCORRECT**

- source path: `spectrometer_visible/channel/grating_spectrometer/intensity_spectrum`
- unit: `s^-1` (data dictionary: `s^-1`)
- data-dictionary text: Intensity spectrum (not calibrated), i.e. number of photoelectrons detected by unit time by a wavelength pixel of the channel, taking into account electronic gain compensation and channels relative calibration
- name description: Detected photoelectron rate assigned to each wavelength pixel of a spectrometer channel after electronic gain compensation and relative calibration, but before absolute radiometric calibration.
- **rejected spelling** `spectral_rate_of_spectrometer_channel` → **proposed spelling** `photoelectron_rate_of_spectrometer_channel`
- why: Two defects, both of self-descriptiveness. The name states **no
  measurand**: a rate of what is never said, and every other accepted name in
  this cohort names its physical quantity — radiance, current, power,
  wavelength, flux. The DD says what it is in one word, photoelectrons, and the
  name's own description has to supply it. Second, `spectral_` is used here for
  a quantity that is **not** a spectral density. In this same cohort
  `spectral_photon_radiance` (row 193) is per unit wavelength and its unit
  carries the extra inverse metre to prove it — `m^-3.s^-1.sr^-1` against the
  per-line `m^-2.s^-1.sr^-1` of row 192. This row's unit is a bare `s^-1`: a
  count rate in each wavelength pixel, with the pixel structure carried by the
  array coordinate rather than by a division. Reading `spectral_` as the
  cohort's own convention would make a consumer divide by a wavelength interval
  that has already not been applied.

### 191. `intensity_at_spectral_line` — **correct**

- source path: `spectrometer_visible/channel/grating_spectrometer/processed_line/intensity`
- unit: `s^-1` (data dictionary: `s^-1`)
- data-dictionary text: Non-calibrated intensity (integrated over the spectrum for this line)
- name description: Uncalibrated photoelectron detection rate obtained by integrating the measured emission signal across an identified spectral line.
- The name names a measurand and a locus, and pairs with
  `photon_radiance_at_spectral_line` (row 192) over the same DD `processed_line`
  container, so the uncalibrated / calibrated distinction is carried by the
  measurand word rather than left to the reader. That pairing is what separates
  this row from row 190, which names no measurand at all.
- note: `intensity` is used in the detector-signal sense — an uncalibrated count
  rate at `s^-1` — not in the SI radiometric sense of radiant intensity, which
  is W·sr⁻¹. The unit makes the intended reading unambiguous and the DD uses the
  same word, so the verdict stands; the remark is recorded because `intensity`
  is the most overloaded word in radiometry and the catalog should not let a
  second, radiometric `intensity` base in beside it.

### 192. `photon_radiance_at_spectral_line` — **correct**

- source path: `spectrometer_visible/channel/grating_spectrometer/processed_line/radiance`
- unit: `m^-2.s^-1.sr^-1` (data dictionary: `m^-2.s^-1.sr^-1`)
- data-dictionary text: Calibrated, background subtracted radiance (integrated over the spectrum for this line)
- name description: Photon-count radiance from an identified emission transition, integrated over its spectral line interval and resolved by projected area and viewing solid angle.
- The unit is a photon rate per area per solid angle and the name says
  `photon_radiance`; the first half cited this row by name as the spelling
  `hard_xray_brightness` should have used.

### 193. `spectral_photon_radiance` — **correct**

- source path: `spectrometer_visible/channel/grating_spectrometer/radiance_spectral`
- unit: `m^-3.s^-1.sr^-1` (data dictionary: `m^-3.s^-1.sr^-1`)
- data-dictionary text: Calibrated spectral radiance (radiance per unit wavelength)
- name description: Calibrated spectral photon radiance of plasma emission, giving photon rate per projected area, solid angle, and wavelength interval along a viewing direction.
- `spectral_` is used correctly here and the unit proves it: `m^-3` against row
  192's `m^-2` is the per-unit-wavelength division. The name carries no
  instrument segment, which is right — a calibrated radiance is a property of
  the emitting plasma along the view, not of the channel that measured it.

### 194. `spectral_wavelength_of_optical_element` — **INCORRECT**

- source path: `spectrometer_visible/channel/grating_spectrometer/wavelengths`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Measured wavelengths
- name description: Photon vacuum wavelength values defining the spectral coordinate for a soft X-ray filter window's wavelength-dependent optical response.
- **rejected spelling** `spectral_wavelength_of_optical_element` → **proposed spelling** `spectral_wavelength_of_spectrometer_channel`
- why: Bound to the wrong object. The DD leaf is the **measured wavelength axis
  of a visible grating spectrometer's spectrum** — the coordinate against which
  `intensity_spectrum` and `radiance_spectral` (rows 190 and 193) are indexed.
  The name instead assigns it to an "optical element", a component whose
  transmission or reflectivity is tabulated against wavelength. Those are
  opposite roles: one is the abscissa of a measurement, the other a property of
  a piece of glass or foil. The description makes the misbinding explicit and
  worse, naming "a **soft X-ray filter window's** wavelength-dependent optical
  response" — an object that appears nowhere on this path, in this IDS, or in
  this diagnostic. The identity is bound to this path alone in the cohort, so
  there is no second binding for which the description would be correct: it is
  wrong wherever it is read.

### 195. `cold_neutral_fraction` — **correct**

- source path: `spectrometer_visible/channel/isotope_ratios/isotope/cold_neutrals_fraction`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Fraction of cold neutrals for this isotope (n_cold_neutrals/(n_cold_neutrals+n_hot_neutrals))
- name description: Dimensionless fraction of the cold recycled component of a hydrogen-isotope neutral population relative to the total neutral atom population.
- The isotope scoping is carried by the DD array index rather than the name,
  which is the cohort's convention for per-species quantities, and the name
  itself claims only what the DD gives: the cold share of a neutral population.
- note: The description's denominator is loose. The DD formula is explicit that
  the denominator is `n_cold + n_hot` **for this isotope**, whereas "the total
  neutral atom population" invites a reading summed over all isotopes. The two
  differ by the isotope's own abundance, which for a deuterium plasma with a
  hydrogen minority is a factor of several. The defect is in the description,
  not the name, so it carries no incorrect verdict.

### 196. `ratio_of_neutral_density_of_isotope_to_difference_of_total_neutral_density_and_neutral_density_of_isotope` — **INCORRECT**

- source path: `spectrometer_visible/channel/isotope_ratios/isotope/density_ratio`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Ratio of the density of neutrals of this isotope over the summed neutral densities of all other isotopes described in the ../isotope array
- name description: Odds ratio of the neutral density of a selected hydrogen isotope to the summed neutral density of all other hydrogen isotopes, distinguishing composition odds from total isotope fraction.
- **rejected spelling** `ratio_of_neutral_density_of_isotope_to_difference_of_total_neutral_density_and_neutral_density_of_isotope` → **proposed spelling** `ratio_of_neutral_density_of_isotope_to_neutral_density_of_other_isotopes`
- why: The name asserts more than the data supports. Its denominator is spelled
  as a difference from the **total** neutral density, which claims a total over
  all neutrals present. The DD is careful to claim less: the sum is over "all
  other isotopes **described in the `../isotope` array**" — the isotopes the
  diagnostic happens to resolve on this channel, which on WEST is typically
  hydrogen and deuterium and never the impurity or helium neutrals also present
  in the edge. Subtracting one isotope from a true total and subtracting it from
  a two-element array sum give different numbers whenever anything outside the
  array is neutral, and the name's arithmetic construction invites a consumer to
  reconstruct the total by inverting it. The proposed spelling says what the DD
  says, and is shorter by six segments — the original is 103 characters of
  circumlocution for the phrase "all other isotopes" that the DD uses directly.

### 197. `atomic_mass` — **correct**

- source path: `spectrometer_visible/channel/isotope_ratios/isotope/element/a`
- unit: `u` (data dictionary: `u`)
- data-dictionary text: Mass of atom
- name description: Mass parameter assigned to a specified ion or neutral particle species, encoding its isotope or constituent composition for inertial and transport calculations.
- The second binding of the identity judged at row 188, and legitimate for the
  same reason.

### 198. `atomic_count` — **correct**

- source path: `spectrometer_visible/channel/isotope_ratios/isotope/element/atoms_n`
- unit: `1` (data dictionary: *empty*)
- data-dictionary text: Number of atoms of this element in the molecule
- name description: Stoichiometric multiplicity of a selected element within the chemical formula of an atomic or molecular species.
- note: Two remarks, neither reaching an incorrect verdict. The units do not
  agree — the DD declares no unit string and the standard name carries `1`; as
  at row 159 the standard name is the defensible side, since a count is
  dimensionless and `1` says so where an empty string says nothing. And
  `atomic_count` sits one word from `atomic_number`, which this same cohort
  publishes for the nuclear charge; the two are different quantities on the same
  element, the descriptions carry the distinction, and the catalog should not
  let the pair drift closer.

### 199. `hot_neutral_fraction` — **correct**

- source path: `spectrometer_visible/channel/isotope_ratios/isotope/hot_neutrals_fraction`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Fraction of hot neutrals for this isotope (n_hot_neutrals/(n_cold_neutrals+n_hot_neutrals))
- name description: Dimensionless fraction of the hot charge-exchange-born component of a hydrogen-isotope neutral population relative to the total neutral atom population.
- The complement of row 195 and correct for the same reasons; the description
  carries the same loose denominator, which is one defect observed on two rows
  rather than two defects.

### 200. `hot_neutral_temperature_at_plasma_boundary` — **INCORRECT**

- source path: `spectrometer_visible/channel/isotope_ratios/isotope/hot_neutrals_temperature`
- unit: `eV` (data dictionary: `eV`)
- data-dictionary text: Temperature of hot neutrals for this isotope
- name description: Energy-equivalent translational temperature of hot neutral atoms at the plasma boundary, characterizing their random motion after removal of bulk flow.
- **rejected spelling** `hot_neutral_temperature_at_plasma_boundary` → **proposed spelling** `hot_neutral_temperature`
- why: The name asserts more than the data supports. The DD gives a temperature
  of hot neutrals for an isotope and says nothing whatever about where. The name
  pins it to the **plasma boundary**, a specific locus, and the plasma boundary
  is a defined surface in this catalog — the same one `volume_of_plasma_boundary`
  and the `*_of_flux_surface` family are built on. A visible spectrometer's
  isotope-ratio channel views along a chord and reports a line-of-sight quantity
  weighted by emissivity; where along that chord the hot neutrals it sees are
  born is a modelling result, not a datum, and on WEST the charge-exchange-born
  population extends well inside the separatrix. The repair is the cohort's own
  convention rather than a minted one: the two sibling leaves in the same DD
  container, rows 195 and 199, are spelled `cold_neutral_fraction` and
  `hot_neutral_fraction` — bare, with no locus and no isotope segment.

### 201. `spectral_signal_to_noise_ratio_of_spectrometer_channel` — **correct**

- source path: `spectrometer_visible/channel/isotope_ratios/signal_to_noise`
- unit: `dB` (data dictionary: `dB`)
- data-dictionary text: Log10 of the ratio of the powers in two bands, one with the spectral lines of interest (signal) the other without spectral lines (noise).
- name description: Spectrometer-channel spectral signal-to-noise ratio is the scalar comparison of spectral power integrated over a selected signal wavelength interval with spectral power integrated over a line-free reference interval for the same channel. It names the shared signal/reference comparison; scale-specific child names define the numerical mapping.
- The name names its measurand, its two intervals and its object, and
  `spectral_` is legitimate here in its band sense: the comparison is between
  two wavelength bands. A signal-to-noise ratio in decibels is a valid base.
- note: The data dictionary contradicts itself by a factor of ten, and the
  standard name cannot repair it. The text defines the quantity as
  `log10(P_signal / P_noise)`, which is a ratio in **bels**; the declared unit
  is `dB`, which is `10 log10` of the same ratio. The standard name inherits the
  declared unit, so `sn_unit` and `dd_unit` agree and neither is independently
  wrong — the disagreement is internal to the DD leaf and must be resolved
  there. A consumer that trusts the text and a consumer that trusts the unit
  will differ by 10× on every value.

### 202. `radial_coordinate_of_line_of_sight` — **correct**

- source path: `spectrometer_visible/channel/line_of_sight/first_point/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of a specified geometric point associated with a line of sight, measured from the toroidal symmetry axis in the right-handed cylindrical (R, φ, Z) frame.

### 203. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `spectrometer_visible/channel/line_of_sight/first_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.

### 204. `toroidal_coordinate_of_line_of_sight` — **correct**

- source path: `spectrometer_visible/channel/line_of_sight/second_point/phi`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Toroidal angle (oriented counter-clockwise when viewing from above)
- name description: Toroidal angular coordinate of the first reference point on a diagnostic line of sight, locating that point around the machine symmetry axis.
- note: The recorded description defect a fourth time — bound to the **second**
  point, described as the first.

## Identities bound outside this index range

Seven of the 51 rows carry an identity that is also bound to source paths
outside indices 154–204. Each is named in its own section; none is judged here,
because a collision cannot be adjudicated from one of its halves. Three need no
action and are recorded so the sweep does not re-open them —
`poloidal_magnetic_flux_of_flux_loop` (rows differing only on the measured /
reconstructed provenance axis, which is a source-binding property and never a
name segment), `plasma_current` (one quantity reported by a measurement, an
equilibrium reconstruction and a summary aggregate), and `wavelength_of_wave_beam`
(the vacuum wavelength of a probing beam, deliberately instrument-independent).
Two carry a rejection that a single repair closes on both halves —
`faraday_angle` (row 171) and `upper_photon_energy` (row 181). One,
`toroidal_angle_of_measurement_position` (row 156), is a genuine three-way locus
collision whose ECE binding is the only correct member, and it is deferred to
the whole-cohort sweep. The three shared `*_coordinate_of_line_of_sight`
identities span sixteen paths each and are correct as spellings; only their
shared description is defective, as recorded.

`initial_polarization_ellipticity_of_polarimeter_beam` is the exception that is
judged here rather than deferred: **both** of its bindings, rows 170 and 178,
lie inside this range, so the whole collision is visible and row 178 carries the
rejection.

## No figure was written

The landing contract asks for a figure wherever a spatial or structural
relationship reads better shown than described, and one relationship here
qualifies: the sibling leaves of a single data-dictionary container splitting
across two different named objects, which is what rows 155–158 and rows 165–167
each are. The node's write fence is three files and does not include
`docs/figures/`, so the graphic is left as follow-on work with its content
specified rather than written outside scope, following the precedent this plan's
drafted-successor node set. What it should show: two container trees, one
`magnetics/b_field_pol_probe` and one `pf_active/coil/element/geometry/rectangle`,
each leaf tinted by the object its accepted name claims, making the split
visible as two colours under one parent.

## Result

| | count |
| --- | --- |
| `rows_judged` — accepted bindings with cohort `index` in 154–204, one verdict section each | **51** |
| judged **correct** | 37 |
| judged **INCORRECT**, each with a proposed spelling | **14** |
| `correct + incorrect` | 37 + 14 = **51** |
| incorrect as a percentage of 51 | **27.5 %** |
| defects recorded as notes on a **correct** row, carrying no incorrect verdict | 8 |
| re-observations of the already-recorded shared line-of-sight description defect | 4 |
| identities also bound to paths outside 154–204, deferred to the collision sweep | 7 |

**14 of 51 — 27.5 % of this block — are not publishable as spelled.** Grouped
into the five classes the first half established:

- **The name is bound to the wrong object** (6): `poloidal_angle_of_flux_surface`
  on a probe coil's orientation; `toroidal_angle_of_measurement_position` on a
  probe's installed location whose own vertical sibling names the probe;
  `height_of_poloidal_field_coil` and `width_of_poloidal_field_coil` on one
  conductor element's cross-section, against an accepted sibling spelling;
  `effective_turn_count_of_passive_loop` on a per-element leaf;
  `spectral_wavelength_of_optical_element` on a spectrometer's measured
  wavelength axis.
- **Not self-descriptive** (4): `toroidal_angle_of_poloidal_magnetic_field_probe`,
  which cannot be told from the probe's positional φ in the same container;
  `faraday_angle`, carried forward from the first half;
  `power_of_soft_xray_detector`, which reads first as the detector's own
  electrical power; `spectral_rate_of_spectrometer_channel`, which names no
  measurand and misuses `spectral_` for a quantity that is not a spectral
  density.
- **The name asserts more than the data supports** (2): the isotope density
  ratio claiming a **total** neutral density where the DD sums only the isotopes
  present in its own array; `hot_neutral_temperature_at_plasma_boundary`, which
  pins a locus the source never states.
- **One name covers two physically different quantities** (1):
  `initial_polarization_ellipticity_of_polarimeter_beam` across the
  polarimeter's `ellipticity_initial` and `polarization_initial` leaves. As in
  the first half this is the class a reader cannot resolve — only a split can.
- **Minority spelling of a base the cohort already fixes** (1):
  `upper_photon_energy` against the accepted `lower_bound_photon_energy` for the
  opposite edge of the same kind of energy band.

The eight defects recorded as **notes on correct rows** are three unit
disagreements where the standard name is the defensible side (rows 159 and 198,
`1` against an empty DD unit string) or where neither side is (row 170, an
ellipticity declared in metres by the DD and inherited by the name); one DD leaf
that contradicts itself by a factor of ten (row 201, text in bels, unit in
decibels); one loose description denominator observed on two rows (195 and 199);
one imprecise mechanism word (row 160); and two cautions about spellings that
are correct but sit close to a different base (rows 191 and 198).

### Reading the rate against the first half

This block's 27.5 % is higher than the first half's 11.6 %, and the two numbers
are not measuring the same thing. The first half sampled every fourth row of the
path-ordered cohort, spreading 86 rows over 20 IDSs; this block is 51
**contiguous** rows over 7, and it is the part of the cohort densest in
instrument geometry — probe positions and orientations, coil conductor
cross-sections, three-point lines of sight. That is exactly where the
wrong-object class lives, and it is 6 of the 14 here against 2 of the 10 there.
Neither figure should be extrapolated to the cohort on its own. Summed, the two
halves now give **24 incorrect of 137 judged — 17.5 %** of the 137 rows judged
so far out of 341 accepted bindings, and that sum is the number to carry
forward, since the halves were judged in one shape against one evidence base.
