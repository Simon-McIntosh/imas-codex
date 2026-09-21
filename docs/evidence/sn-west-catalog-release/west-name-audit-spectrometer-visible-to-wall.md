# WEST accepted names, cohort rows 205–255 — physical-correctness audit

provisional: true — verdicts are appended as they are judged; the closing pass
rewrites this line and adds the result section.

This file judges the **51 accepted name bindings at indices 205 to 255** of the
unjudged remainder cohort recorded in
[`west-name-cohort-remainder.json`](west-name-cohort-remainder.json) — from
`spectrometer_visible/channel/line_of_sight/second_point/z` to
`wall/description_2d/mobile/unit/outline/r`. It is the fifth of the remainder
blocks and is written in the shape of
[`west-name-audit.md`](west-name-audit.md), which judged 86 rows of the same
341-binding cohort, so that the two can be **summed rather than compared**.

## What is judged, and from what

Every row was drawn from the live graph when the remainder cohort was cut, so
**no graph query was issued and no database was opened here**. Each row carries
`name`, `path`, `sn_unit`, `sn_description`, `dd_unit`, `dd_doc` and
`dd_doc_parent` — the parent container's text, used where the leaf's own
documentation is empty or the literal `Value` — and that is the whole evidence
base for the judgement.

Each name is judged on three questions:

1. Does the data-dictionary text of the path it is bound to describe the
   quantity the name claims?
2. Do `sn_unit` and `dd_unit` agree, and where they differ, which side is
   defensible?
3. Is the name self-descriptive to a reader who does not have the source path
   in hand?

A verdict is **correct** or **INCORRECT**. Every INCORRECT row shows the
rejected spelling beside a proposed one and states the semantic distinction in
plain language. A defect that lies in the **description** rather than in the
name is recorded as a **note** on a correct row and never as an INCORRECT
verdict; the result table counts notes separately.

### Adjudications carried over, not relitigated

The first half settled these and they are reused rather than reopened: the
`*_of_flux_surface` family including `volume_of_flux_surface` and
`area_of_flux_surface`; `back_surface_distance_of_antenna_strap`; and the
etendue `_detector` spelling. Where the first half accepted a spelling, this
half reuses it instead of minting a parallel one — the proposed
`total_coupled_power_due_to_lower_hybrid_heating` at row 238 is built from the
first half's proposed `total_coupled_power_due_to_ion_cyclotron_heating` at its
row 80, and the proposed `radial_outline_of_plasma_facing_component` at row 255
is built from the accepted `vertical_outline_of_plasma_facing_component` at its
row 86.

### Collisions are named here and adjudicated elsewhere

Twenty of these 51 rows carry an identity that is also bound to source paths
**outside** indices 205–255. Each such row says so and defers the collision
itself to the node that owns the whole-cohort collision sweep. Two collisions
lie **wholly inside** this range — `vertical_coordinate_of_strike_point` across
rows 213 and 214, and `launched_power_of_lower_hybrid_antenna` across rows 236
and 238 — and those are judged here, because both halves are in hand.

## Verdicts

### 205. `vertical_coordinate_of_line_of_sight` — **correct**

- source path: `spectrometer_visible/channel/line_of_sight/second_point/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, φ, Z) frame.
- note: The identity is bound to 15 further paths outside this range, across seven diagnostics and three line-of-sight points. That is the settled shared-geometry-point pattern the first half accepted at its rows 2 and 3, and the description here says "a designated point" rather than naming one, so the row is correct as spelled. The collision itself is deferred to the whole-cohort sweep.

### 206. `gap_at_outboard_midplane` — **INCORRECT**

- source path: `summary/boundary/distance_inner_outer_separatrices/value`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Distance between the inner and outer separatrices, in the major radius direction, at the plasma outboard and at the height corresponding to the maximum R for the inner separatrix.
- name description: Signed major-radius separation between a measurement position and the magnetic separatrix, mapped along its flux surface to the outboard midplane; positive outside the separatrix.
- **rejected spelling** `gap_at_outboard_midplane` → **proposed spelling** `radial_separation_of_inner_and_outer_separatrices_at_outboard_midplane`
- why: The name and its description denote the gap between **a measurement position and the separatrix** — the quantity a probe or a reflectometer reports. The data dictionary denotes the gap between **two separatrices**, the inner and the outer, which exists only in a near-double-null configuration and is the control quantity conventionally written drsep. They are different objects: one has a diagnostic locus as its second endpoint, the other has a second magnetic surface. A reader taking the published name at face value would map a drsep trace onto a wall-gap signal, and the sign conventions differ too — the description's "positive outside the separatrix" has no meaning for a separatrix-to-separatrix distance.

### 207. `elongation_of_plasma_boundary` — **correct**

- source path: `summary/boundary/elongation/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Elongation of the plasma boundary
- name description: Dimensionless ratio of the plasma boundary’s vertical half-height to its minor radius, quantifying the elongation of its cross-sectional shape.
- note: Also bound to `equilibrium/time_slice/boundary/elongation` outside this range — the same physical quantity from the equilibrium IDS, so a legitimate shared identity. Deferred to the collision sweep.

### 208. `radial_coordinate_of_geometric_axis` — **correct**

- source path: `summary/boundary/geometric_axis_r/value`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: R position of the geometric axis (defined as (Rmax+Rmin) / 2 of the boundary)
- name description: Major-radius coordinate locating the midpoint of the plasma boundary's radial extrema, thereby specifying the horizontal position of its geometric axis.
- note: Also bound to `equilibrium/time_slice/boundary/geometric_axis/r`; same quantity, deferred to the collision sweep.

### 209. `vertical_coordinate_of_geometric_axis` — **correct**

- source path: `summary/boundary/geometric_axis_z/value`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Z position of the geometric axis (defined as (Zmax+Zmin) / 2 of the boundary)
- name description: Signed Z height of the plasma boundary’s geometric axis, computed as the midpoint of its maximum and minimum vertical extents.
- note: Also bound to `equilibrium/time_slice/boundary/geometric_axis/z`; same quantity, deferred to the collision sweep.

### 210. `radial_coordinate_of_magnetic_axis` — **correct**

- source path: `summary/boundary/magnetic_axis_r/value`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: R position of the magnetic axis
- name description: Major-radius coordinate locating the magnetic-axis O-point in the right-handed cylindrical (R, φ, Z) frame around which nested closed flux surfaces are organized.
- note: This path is genuinely the magnetic axis and the row is correct. The identity is also bound to three paths outside this range, and one of them — `equilibrium/time_slice/contour_tree/node/r` — is **not** a magnetic axis: a contour-tree node is any critical point of the poloidal flux map, X-points included. That is the same shape as the first half's `radial_coordinate_of_measurement_position` finding and is deferred to the collision sweep, which owns the whole set.

### 211. `minor_radius_of_plasma_boundary` — **correct**

- source path: `summary/boundary/minor_radius/value`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Minor radius of the plasma boundary (defined as (Rmax-Rmin) / 2 of the boundary)
- name description: Half the difference between the maximum and minimum major-radius coordinates of the last closed plasma-boundary contour, defining its horizontal cross-sectional size.
- note: Also bound to `equilibrium/time_slice/boundary/minor_radius`; same quantity, deferred to the collision sweep.

### 212. `radial_coordinate_of_strike_point` — **INCORRECT**

- source path: `summary/boundary/strike_point_inner_r/value`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: R position of the inner strike point
- name description: Major-radius location of an individual magnetic strike point where a separatrix leg intersects a divertor target, expressed in the right-handed cylindrical (R, φ, Z) frame.
- **rejected spelling** `radial_coordinate_of_strike_point` → **proposed spelling** `radial_coordinate_of_inner_strike_point`
- why: This is the **inner half of the defect the first half rejected at its row 72**, where the same identity was judged on `strike_point_outer_r`. One name is bound to both divertor legs: opposite targets, different heat-flux loading, different control significance, and nothing in the published entry tells a reader which one a value came from. The remedy is the split the first half proposed — `radial_coordinate_of_inner_strike_point` here and `radial_coordinate_of_outer_strike_point` there. With this row both halves of the radial pair now carry a verdict.

### 213. `vertical_coordinate_of_strike_point` — **INCORRECT**

- source path: `summary/boundary/strike_point_inner_z/value`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Z position of the inner strike point
- name description: Signed vertical (Z) coordinate of the inner divertor strike point, where the inner separatrix leg intersects the divertor target in the right-handed cylindrical (R, φ, Z) frame.
- **rejected spelling** `vertical_coordinate_of_strike_point` → **proposed spelling** `vertical_coordinate_of_inner_strike_point`
- why: The vertical twin of row 212, and unlike that pair **both halves lie inside this range** (rows 213 and 214), so the collision is judged here rather than deferred. The unqualified name is bound to `strike_point_inner_z` and `strike_point_outer_z` — two physically distinct locations on opposite divertor legs. The first half predicted exactly this as the third of its outside-the-sample findings; the cohort rows confirm it.

### 214. `vertical_coordinate_of_strike_point` — **INCORRECT**

- source path: `summary/boundary/strike_point_outer_z/value`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Z position of the outer strike point
- name description: Signed vertical (Z) coordinate of the **inner** divertor strike point, where the inner separatrix leg intersects the divertor target in the right-handed cylindrical (R, φ, Z) frame.
- **rejected spelling** `vertical_coordinate_of_strike_point` → **proposed spelling** `vertical_coordinate_of_outer_strike_point`
- why: This is the sharpest instance of the one-name-two-quantities class in either half, because the shared identity's single description **hard-codes the inner leg** — "the inner divertor strike point", "the inner separatrix leg" — while this binding is to the **outer** strike point. A reader who resolves the published name and reads its description is not merely unable to tell which leg a value came from; they are told the wrong one. The defect is in the name, not the description: one identity cannot carry a description that is true of only one of the two paths it is bound to, so the split is the only remedy.

### 215. `lower_triangularity_of_plasma_boundary` — **correct**

- source path: `summary/boundary/triangularity_lower/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Lower triangularity of the plasma boundary
- name description: Dimensionless shaping parameter equal to the normalized inward radial displacement of the lower plasma-boundary extremum from the geometric center.
- note: Surface-explicit, as the cohort's shape-parameter convention requires — the surface a triangularity belongs to is in the name rather than left to the path. Also bound to `equilibrium/time_slice/boundary/triangularity_lower`; same quantity, deferred to the collision sweep.

### 216. `upper_triangularity_of_plasma_boundary` — **correct**

- source path: `summary/boundary/triangularity_upper/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Upper triangularity of the plasma boundary
- name description: Dimensionless shape parameter for the plasma-boundary poloidal cross-section, expressing the inward radial displacement of its upper extremum relative to the geometric center.
- note: Also bound to `equilibrium/time_slice/boundary/triangularity_upper`; same quantity, deferred to the collision sweep.

### 217. `vertical_coordinate_of_primary_x_point` — **correct**

- source path: `summary/boundary/x_point_main/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical position of the primary magnetic X-point, the poloidal-field null where separatrix branches meet, in the right-handed cylindrical (R, φ, Z) frame.
- note: This is the qualified spelling the first half endorsed when it rejected the sibling `summary/boundary/x_point_main/r` as `radial_coordinate_of_x_point` at its row 73. The row is correct and is the survivor of that pair.

### 218. `total_neutron_rate` — **correct**

- source path: `summary/fusion/neutron_rates/total/value`
- unit: `Hz` (data dictionary: `Hz`)
- data-dictionary text: Total neutron rate from all reactions
- name description: Total neutron emission rate produced by deuterium–tritium, deuterium–deuterium, and tritium–tritium fusion reactions in the plasma.
- note: "Total" carries the sum-over-reactions sense the data dictionary states, and the description enumerates the three branches. The name is an emission rate of the plasma, not a detector count rate; the description says so.

### 219. `accumulated_total_gas_count` — **INCORRECT**

- source path: `summary/gas_injection_accumulated/total/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Total accumulated injected gas (sum over species)
- name description: Cumulative equivalent-electron inventory of all gas species delivered by injection after plasma breakdown, summed across species.
- **rejected spelling** `accumulated_total_gas_count` → **proposed spelling** `accumulated_total_neutral_particle_count_due_to_gas_injection`
- why: Nothing in the name says the gas was **injected**. A reader without the path cannot distinguish an accumulated injection inventory from the residual gas inventory of the vessel, which this same batch measures through barometry, and "total" is left ambiguous between summed-over-species and summed-over-the-vessel. The same physical source already has a rate spelling in this cohort — the first half's row 74, `total_neutral_source_rate_due_to_gas_injection`, judged correct — and this is that quantity's time integral. A catalog that spells the rate `..._due_to_gas_injection` and its own integral `gas_count` publishes one process under two unrelated bases.

### 220. `toroidal_vacuum_magnetic_field` — **correct**

- source path: `summary/global_quantities/b0/value`
- unit: `T` (data dictionary: `T`)
- data-dictionary text: Vacuum toroidal field at R0. Positive sign means anti-clockwise when viewed from above. The product R0B0 must be consistent with the b_tor_vacuum_r field of the tf IDS.
- name description: Signed toroidal component of the current-free vacuum magnetic field at a reference major radius, defining the nominal externally generated field strength.
- note: The name omits the locus its own description supplies. The vacuum toroidal field falls as 1/R, so a scalar value is meaningless without the radius it is quoted at, and the cohort elsewhere puts the locus in the name (`toroidal_magnetic_field_at_magnetic_axis`, first half row 82). The verdict is correct rather than incorrect because B0-at-R0 is a universal convention, this batch publishes `reference_major_radius` beside it, and row 251's `vacuum_poloidal_current_function` carries the radius-independent form — so the reader has the pair. Recorded as a candidate for a locus-explicit spelling. Also bound to the `core_profiles` and `equilibrium` `vacuum_toroidal_field/b0` nodes outside this range; same quantity, deferred to the collision sweep.

### 221. `poloidal_beta` — **correct**

- source path: `summary/global_quantities/beta_pol_mhd/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Poloidal beta estimated from the pressure determined by an equilibrium reconstruction code. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2]
- name description: Poloidal beta is a dimensionless measure of total plasma pressure relative to the magnetic-pressure scale of the plasma-current-generated poloidal field.
- note: The path carries the `_mhd` qualifier and the name does not. The first half settled this at its row 75, where `normalized_toroidal_beta` was judged correct on `beta_tor_norm_mhd` against a data-dictionary text that likewise says "using the pressure determined by an equilibrium reconstruction code" — the reconstruction provenance is a property of how the value was obtained, not a different physical quantity. That adjudication is applied here rather than reopened. Also bound to `equilibrium/time_slice/global_quantities/beta_pol`; deferred to the collision sweep.

### 222. `toroidal_beta` — **correct**

- source path: `summary/global_quantities/beta_tor/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2
- name description: Toroidal beta is a dimensionless equilibrium ratio of volume-averaged total perpendicular plasma pressure to the magnetic pressure of a reference toroidal field.
- note: Also bound to `equilibrium/time_slice/global_quantities/beta_tor`; same quantity, deferred to the collision sweep.

### 223. `mhd_energy` — **INCORRECT**

- source path: `summary/global_quantities/energy_mhd`
- unit: `J` (data dictionary: `J`)
- data-dictionary text: Plasma energy content = 3/2 * integral over the plasma volume of the total kinetic pressure (pressure determined by an equilibrium reconstruction code)
- name description: Global plasma stored energy obtained from the volume integral of total kinetic pressure, including thermal and fast-particle pressure contributions.
- **rejected spelling** `mhd_energy` → **proposed spelling** `total_plasma_stored_energy`
- why: The name is a transliteration of the data-dictionary leaf `energy_mhd` and says nothing a reader can use. "MHD energy" most naturally reads as the energy of magnetohydrodynamic activity, or as magnetic energy — it does not read as the plasma's stored kinetic energy content, which is what the data dictionary and the description both say it is. The quantity is the volume integral of total kinetic pressure, thermal plus fast-particle, so the base is `stored_energy` and the qualifier is `total` against the thermal-only sibling. By the adjudication carried at row 221 the reconstruction provenance does not belong in the name, which is why the proposed spelling drops `mhd` rather than expanding it.

### 224. `ratio_of_line_averaged_electron_density_to_greenwald_density` — **correct**

- source path: `summary/global_quantities/greenwald_fraction/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Greenwald fraction =line_average/n_e/value divided by (global_quantities/ip/value *1e6 * pi * minor_radius^2)
- name description: Dimensionless Greenwald fraction comparing the electron number density averaged along a plasma chord with the empirical density limit set by plasma current and effective minor radius.
- note: The name states both sides of the ratio explicitly, so a reader needs neither the path nor the eponym to know what is divided by what — the strongest form of self-description in either half. Its numerator matches the cohort's `line_averaged_electron_density` at row 240 exactly.

### 225. `energy_confinement_enhancement_factor` — **INCORRECT**

- source path: `summary/global_quantities/h_98/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Energy confinement time enhancement factor over the IPB98(y,2) scaling
- name description: Dimensionless ratio comparing the plasma energy confinement time with the IPB98(y,2) reference scaling prediction.
- **rejected spelling** `energy_confinement_enhancement_factor` → **proposed spelling** `ipb98y2_confinement_enhancement_factor`
- why: A confinement enhancement factor is a ratio **against a named scaling law**, and there is no default one: H98, H89 and HIPB20 are all in routine use and differ by tens of percent on the same discharge. The published name names none of them, so it cannot distinguish the value it is bound to from the value a second scaling would produce — and a later H89 binding would have nowhere to go but onto this same identity, which is the one-name-two-quantities failure already present at rows 212 to 214, arriving prospectively. The cut already fixes the spelling of the scaling in this name's own denominator: row 233 is `ipb98y2_confinement_time`. The proposed spelling reuses it.
