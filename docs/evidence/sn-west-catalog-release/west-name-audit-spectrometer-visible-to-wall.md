# WEST accepted names, cohort rows 205–255 — physical-correctness audit

provisional: false — all 51 rows carry a verdict and the result section is
closed. (This line opened as `provisional: true` and was rewritten by the
closing pass; the marker is repeated at the foot of the file.)

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
- why: The name is a transliteration of the data-dictionary leaf `energy_mhd` and says nothing a reader can use. "MHD energy" most naturally reads as the energy of magnetohydrodynamic activity, or as magnetic energy — it does not read as the plasma's stored kinetic energy content, which is what the data dictionary and the description both say it is. The quantity is the volume integral of total kinetic pressure, thermal plus fast-particle, so the base is `stored_energy` and the qualifier is `total` against the thermal-only sibling. By the adjudication carried at row 221 the reconstruction provenance does not belong in the name, which is why the proposed spelling drops `mhd` rather than expanding it. The
  identity is also bound to `equilibrium/time_slice/global_quantities/energy_mhd`
  outside this range — the same quantity, so the rejection applies to both
  bindings; the collision itself is deferred to the whole-cohort sweep.

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

### 226. `normalized_plasma_internal_inductance` — **correct**

- source path: `summary/global_quantities/li_3/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Internal inductance. The li_3 definition is used, i.e. li_3 = 2/R0/mu0^2/Ip^2 * int(Bp^2 dV).
- name description: Dimensionless li_3 parameter measuring poloidal magnetic-field peaking through a plasma-volume integral normalized by total plasma current and a reference major radius.
- note: This is the weaker half of the shape rejected at row 225 — the description names the `li_3` definition and the name does not, and `li_1`, `li_2` and `li_3` differ by several percent on a shaped plasma. It stays **correct** because the variants are three normalizations of one quantity rather than three independent empirical laws, and nothing else in this cut is bound to this identity under a competing definition. If a `li_2` binding ever enters the catalog this row becomes the same defect as row 225; recorded so the sweep can see it coming. Also bound to `equilibrium/time_slice/global_quantities/li_3` — the same definition — deferred to the collision sweep.

### 227. `total_power_at_separatrix` — **correct**

- source path: `summary/global_quantities/power_loss/value`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: Power through separatrix
- name description: Net energy power leaving the confined plasma through the last closed magnetic flux surface and entering the scrape-off layer.
- note: The name's `at` denotes a flux **through** a surface rather than a value **on** it, which the data dictionary says plainly and the description repeats; `total_power_through_separatrix` would carry that in the name. The distinction matters because the surface also carries quantities that genuinely are evaluated on it — row 234's `loop_voltage_at_plasma_boundary` and row 248's `electron_density_at_plasma_boundary` both use `at` in the on-the-surface sense, so one preposition is doing two jobs inside one cut.

### 228. `total_power_due_to_ohmic_dissipation` — **correct**

- source path: `summary/global_quantities/power_ohm/value`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: Ohmic power
- name description: Aggregate rate of irreversible electrical-energy conversion into heat by resistive current flow across the applicable conducting plasma or structure.
- note: The defect is in the description, not the name. The description admits "the applicable conducting plasma **or structure**", which would cover resistive dissipation in the vessel and the coils; the data-dictionary node sits under `summary/global_quantities` and is the **plasma** ohmic power, the counterpart of the external heating at row 237. The name is correct and self-descriptive; the description should be narrowed to the plasma.

### 229. `radiated_power_over_core_region` — **INCORRECT**

- source path: `summary/global_quantities/power_radiated_inside_lcfs/value`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: Radiated power from the plasma inside the Last Closed Flux Surface
- name description: Total electromagnetic power radiated by plasma inside the last closed flux surface, representing the core-region contribution to global radiative energy loss.
- **rejected spelling** `radiated_power_over_core_region` → **proposed spelling** `radiated_power_inside_plasma_boundary`
- why: "Core region" is not a boundary any reader can resolve. In transport usage it names the inner plasma as distinct from the pedestal and edge — typically inside r/a of roughly 0.8 — while in edge-code usage it means everything inside the separatrix. The two readings differ by the whole pedestal, which in a radiating divertor scenario carries a large share of the radiated power, so the same published name can be off by tens of percent depending on which sense a consumer assumes. The data dictionary states the boundary exactly and the cut already has a settled spelling for it: rows 227, 234 and 248 all use `plasma_boundary` for the last closed flux surface, and the first half's `volume_of_plasma_boundary` does too. The proposed spelling says the boundary the data dictionary says, in the words the rest of the cohort already uses.

### 230. `difference_of_total_plasma_heating_power_and_time_derivative_of_plasma_stored_energy` — **correct**

- source path: `summary/global_quantities/power_steady/value`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: Total power coupled to the plasma minus dW/dt (correcting from transient energy content)
- name description: Instantaneous plasma power available for confinement losses after correcting total coupled heating for the rate of change of stored plasma energy.
- note: Long, and exactly right: the `difference_of_X_and_Y` construction states both operands and their order, so the sign convention is readable from the name alone. The same construction carries row 252. It is the counter-example to the data-dictionary leaf name at row 223 — `power_steady` would have been as opaque as `energy_mhd`, and the name does not use it.

### 231. `safety_factor_at_normalized_poloidal_magnetic_flux_equal_to_0_95` — **correct**

- source path: `summary/global_quantities/q_95/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: q at the 95% poloidal flux surface
- name description: Signed field-line winding number on the closed magnetic flux surface labeled by normalized poloidal magnetic flux 0.95, near but inside the plasma boundary.
- note: The surface label is in the name, including which flux is normalized — so the reader is not left to guess between normalized poloidal flux and normalized toroidal flux, which select different surfaces. Also bound to `equilibrium/time_slice/global_quantities/q_95`; same quantity, deferred to the collision sweep.

### 232. `energy_confinement_time` — **correct**

- source path: `summary/global_quantities/tau_energy/value`
- unit: `s` (data dictionary: `s`)
- data-dictionary text: Energy confinement time
- name description: Characteristic timescale for loss of confined plasma thermal energy, defined as stored thermal energy divided by the net power leaving that energy reservoir.

### 233. `ipb98y2_confinement_time` — **correct**

- source path: `summary/global_quantities/tau_energy_98/value`
- unit: `s` (data dictionary: `s`)
- data-dictionary text: Energy confinement time estimated from the IPB98(y,2) scaling
- name description: IPB98(y,2) empirical-scaling reference for global energy confinement time, providing the denominator for the H98 confinement enhancement factor.
- note: The scaling law is in the name, which is what row 225 was rejected for lacking — this row is the evidence that the cut can spell it. The name drops `energy` where its measured counterpart at row 232 keeps it (`energy_confinement_time`); `ipb98y2_energy_confinement_time` would make the pair symmetric. A note rather than a rejection, because the scaling law itself is defined only for energy confinement, so nothing is ambiguous.

### 234. `loop_voltage_at_plasma_boundary` — **correct**

- source path: `summary/global_quantities/v_loop/value`
- unit: `V` (data dictionary: `V`)
- data-dictionary text: LCFS loop voltage
- name description: Toroidal electromotive force evaluated on the last closed flux surface that drives the conventionally signed ohmic plasma current.
- note: The locus is in the name, and it matters — the loop voltage measured at a flux loop on the vessel differs from the boundary value during current ramps. The `plasma_boundary` spelling is the cohort's settled one for the LCFS.

### 235. `net_power_due_to_ion_cyclotron_heating` — **INCORRECT**

- source path: `summary/heating_current_drive/ic/power/value`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: IC heating power coupled to the plasma from this launcher
- name description: Net ion-cyclotron radio-frequency power launched into the vacuum vessel by a specified heating launcher before absorption by plasma particles.
- **rejected spelling** `net_power_due_to_ion_cyclotron_heating` → **proposed spelling** `coupled_power_due_to_ion_cyclotron_heating`
- why: One identity spans the launched–coupled boundary. The first half accepted this name at `ic_antennas/antenna/power_launched`, where it genuinely is launched power, and rejected `total_power_due_to_ion_cyclotron_heating` at `summary/.../power_ic/value` precisely because the data dictionary there says **coupled to the plasma**. This row is the same mismatch on the per-launcher summary node: the name and its description both say launched into the vessel, the data dictionary says coupled to the plasma, and the two are separated by the coupling efficiency — routinely 10 to 30 percent apart, and far more at poor loading. Because the identity is bound to **both** an antenna launched-power node and a summary coupled-power node, a consumer cannot recover which quantity a value is, and the remedy is a split rather than a rewording: `net_power_due_to_ion_cyclotron_heating` stays with the antenna node and the summary per-launcher node takes `coupled_power_due_to_ion_cyclotron_heating`, under the same `total_coupled_power_due_to_ion_cyclotron_heating` the first half proposed for the system total.

### 236. `launched_power_of_lower_hybrid_antenna` — **INCORRECT**

- source path: `summary/heating_current_drive/lh/power/value`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: LH heating power coupled to the plasma from this launcher
- name description: Launched power of a lower-hybrid antenna is the net RF power entering the vacuum vessel after reflection at the antenna input reference plane.
- **rejected spelling** `launched_power_of_lower_hybrid_antenna` → **proposed spelling** `coupled_power_of_lower_hybrid_antenna`
- why: The launched–coupled mismatch of row 235 on the lower-hybrid side. The name asserts power **entering the vessel**, measured after reflection at the antenna reference plane; the data dictionary states power **coupled to the plasma**, which is what remains after the fraction lost to the launcher structure and to parasitic absorption in the edge. On WEST's lower-hybrid launchers that difference is a first-order quantity, not a rounding term, and it is the quantity most often disputed in a power balance. The name asserts the larger of the two and the data supports only the smaller.

### 237. `total_external_heating_power` — **correct**

- source path: `summary/heating_current_drive/power_additional/value`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: Total additional external power (NBI+EC+IC+LH, without ohmic) coupled to the plasma
- name description: Total non-inductive power coupled from external neutral-beam and radio-frequency heating systems to the plasma, summed over auxiliary sources and excluding Ohmic.
- note: The defect is in the description, not the name. The description calls the sum "non-inductive power", which is the vocabulary of **current drive** — the fraction of the plasma current driven other than by the transformer. The data dictionary's "without ohmic" is a statement about which heating **sources** are summed, not about how current is driven, and a heating system can deliver large power while driving no current. The name says external heating power, which is right; the description should drop "non-inductive".

### 238. `launched_power_of_lower_hybrid_antenna` — **INCORRECT**

- source path: `summary/heating_current_drive/power_lh/value`
- unit: `W` (data dictionary: `W`)
- data-dictionary text: Total LH power coupled to the plasma
- name description: Launched power of a lower-hybrid antenna is the net RF power entering the vacuum vessel after reflection at the antenna input reference plane.
- **rejected spelling** `launched_power_of_lower_hybrid_antenna` → **proposed spelling** `total_coupled_power_due_to_lower_hybrid_heating`
- why: One name, two physically different quantities, and **both halves are inside this range**: row 236 is the **per-launcher** power (`heating_current_drive/lh/power/value`, "from this launcher") and this row is the **system total summed over launchers** ("Total LH power"). On WEST, which runs two lower-hybrid launchers, they differ by roughly a factor of two, and a published entry that resolves to either is unusable in a power balance. The name is also singular — "of a lower-hybrid antenna" — which is false of the total. The proposed spelling is built from the first half's proposed `total_coupled_power_due_to_ion_cyclotron_heating` at its row 80, so the two heating systems carry one grammar rather than two.

### 239. `ratio_of_line_averaged_hydrogen_density_to_line_averaged_total_hydrogenic_density` — **correct**

- source path: `summary/line_average/isotope_fraction_hydrogen/value`
- unit: `1` (data dictionary: `1`)
- data-dictionary text: Fraction of hydrogen density among the hydrogenic species (nH/(nH+nD+nT))
- name description: Dimensionless line-averaged protium fraction among hydrogenic ions, obtained by dividing the protium density by the total hydrogenic density along the same chord.
- note: The whole distinction rides on `hydrogen` meaning protium while `hydrogenic` means protium, deuterium and tritium together — the data dictionary's `nH/(nH+nD+nT)` settles it and the description says protium explicitly. The name also states that **both** sides are line-averaged over the same chord, which is what makes the ratio well defined; a ratio of two differently averaged densities would not be.

### 240. `line_averaged_electron_density` — **correct**

- source path: `summary/line_average/n_e/value`
- unit: `m^-3` (data dictionary: `m^-3`)
- data-dictionary text: Electron density
- name description: Number density of free electrons per physical volume averaged along a complete plasma propagation chord, equal to the path integral divided by chord length.
- note: The leaf's own text is the bare "Electron density" and the averaging is carried entirely by the `line_average` container — so the name supplies the qualifier the data-dictionary leaf omits, which is the correct direction. Also bound to `interferometer/channel/n_e_line_average` outside this range; deferred to the collision sweep, which owns the question of whether a per-channel chord average and a summary line average are one identity.

### 241. `electron_density_at_divertor_target` — **correct**

- source path: `summary/local/divertor_target/n_e/value`
- unit: `m^-3` (data dictionary: `m^-3`)
- data-dictionary text: Electron density
- name description: Number density of the electron species at the divertor target is the local particle count per volume for free electrons at the sheath entrance immediately upstream of the target-facing plasma boundary.
- note: The description pins the evaluation point to the sheath entrance, which is the convention divertor measurements and two-point models use; without it, "at the target" would be ambiguous between the sheath entrance and the material surface.

### 242. `energy_flux_maximum_at_divertor_target` — **correct**

- source path: `summary/local/divertor_target/power_flux_peak/value`
- unit: `W.m^-2` (data dictionary: `W.m^-2`)
- data-dictionary text: Peak power flux on the divertor target or limiter surface
- name description: Peak local energy-deposition rate per unit area on one divertor target surface, combining incident plasma-particle and radiative energy-carrying channels.
- note: The data-dictionary text admits "the divertor target **or limiter** surface" while the name commits to a divertor target. The name is faithful to its path — the container is `summary/local/divertor_target` — so the verdict is correct, but WEST operates in both limited and diverted configurations and a limiter-phase value stored under this node would be published under a name that denies it. Recorded for the sweep rather than rejected, because the remedy lies in the data dictionary's container naming and not in this cut.

### 243. `electron_temperature_at_divertor_target` — **correct**

- source path: `summary/local/divertor_target/t_e/value`
- unit: `eV` (data dictionary: `eV`)
- data-dictionary text: Electron temperature
- name description: Electron temperature at the divertor target is the thermal energy per particle of the electron population evaluated at the plasma side of a divertor-target sheath, distinct from the target material temperature.
- note: The description draws exactly the distinction a reader of the name could get wrong — plasma electron temperature at the target, not the temperature of the target tile, which this batch also publishes from infrared thermography. Units agree at `eV` on both sides.

### 244. `electron_density_at_magnetic_axis` — **correct**

- source path: `summary/local/magnetic_axis/n_e/value`
- unit: `m^-3` (data dictionary: `m^-3`)
- data-dictionary text: Electron density
- name description: Particle number density of the electron population evaluated at the magnetic axis is the local electron count per physical volume at that location.
- note: `electron_density` is the base the first half found spelled eight ways to one and settled as the survivor; this row uses it.

### 245. `radial_coordinate_of_magnetic_axis` — **correct**

- source path: `summary/local/magnetic_axis/position/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate locating the magnetic-axis O-point in the right-handed cylindrical (R, φ, Z) frame around which nested closed flux surfaces are organized.
- note: The same identity as row 210, reached through the `summary/local/magnetic_axis/position` container rather than the `summary/boundary` scalar mirror. Both are genuinely the magnetic axis, so both are correct; the contour-tree binding named at row 210 is the one that is not, and it is deferred to the collision sweep with the rest of that identity's out-of-range paths.

### 246. `vertical_coordinate_of_magnetic_axis` — **correct**

- source path: `summary/local/magnetic_axis/position/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical Z coordinate of the magnetic axis in the right-handed cylindrical (R, φ, Z) frame, marking the interior extremum organizing nested magnetic flux surfaces.
- note: Also bound to `equilibrium/time_slice/global_quantities/magnetic_axis/z` and `summary/boundary/magnetic_axis_z/value`, both genuinely the magnetic axis; deferred to the collision sweep. Unlike its radial twin this identity is **not** bound to a contour-tree node, so the R and the Z of one locus reach different numbers of objects.

### 247. `electron_temperature_at_magnetic_axis` — **correct**

- source path: `summary/local/magnetic_axis/t_e/value`
- unit: `eV` (data dictionary: `eV`)
- data-dictionary text: Electron temperature
- name description: Thermal energy per particle of the electron population, expressed as an energy-equivalent temperature and evaluated at the magnetic axis.

### 248. `electron_density_at_plasma_boundary` — **correct**

- source path: `summary/local/separatrix/n_e/value`
- unit: `m^-3` (data dictionary: `m^-3`)
- data-dictionary text: Electron density
- name description: Number density of electrons at the plasma boundary, evaluated as the local, intensive particle count per physical volume on the last closed magnetic flux surface.
- note: The path's container is `separatrix` and the name says `plasma_boundary`. In a diverted configuration the two coincide and the substitution is the cohort's settled spelling for the last closed flux surface, which is why the verdict is correct. In a limited configuration they do not: the last closed surface is then set by the limiter contact point and a separatrix may lie outside the vessel or not exist in the confined region at all. WEST runs both, so the row is recorded for the sweep as a case where the name's boundary definition and the path's are the same object only under one magnetic topology.

### 249. `pulse_duration` — **INCORRECT**

- source path: `summary/plasma_duration/value`
- unit: `s` (data dictionary: `s`)
- data-dictionary text: Duration of existence of a confined plasma during the pulse
- name description: Elapsed duration of the confined-plasma phase in a single discharge, from plasma breakdown until termination of the confined plasma.
- **rejected spelling** `pulse_duration` → **proposed spelling** `confined_plasma_duration`
- why: The data dictionary says this is the duration of the confined plasma **during** the pulse — a sub-interval — and the name gives it the whole pulse. They are not the same number on any discharge: the machine pulse starts with the toroidal-field and gas prefill phases and ends after the current has fully decayed, while the confined plasma exists only between breakdown and termination, and on a disruptive shot the two can differ by seconds. WEST's long-pulse programme reports both quantities routinely, so publishing the shorter one under the longer one's name is a direct source of error. The description is already correct and says the confined-plasma phase; only the name asserts the pulse. The cut also publishes `breakdown_initial_time` (first half, row 84), the instant this interval begins, which makes the intended interval unambiguous once the name says which one it is.

### 250. `volume_averaged_electron_density` — **correct**

- source path: `summary/volume_average/n_e/value`
- unit: `m^-3` (data dictionary: `m^-3`)
- data-dictionary text: Electron density
- name description: Number density of free electrons per physical volume averaged over the plasma volume enclosed by the last closed flux surface, giving the global mean free-electron density.
- note: The averaging qualifier is supplied by the name where the data-dictionary leaf is bare, as at row 240, and the description states the averaging volume. Also bound to `interferometer/n_e_volume_average` outside this range; deferred to the collision sweep.

### 251. `vacuum_poloidal_current_function` — **correct**

- source path: `tf/b_field_phi_vacuum_r`
- unit: `T.m` (data dictionary: `T.m`)
- data-dictionary text: Vacuum field times major radius in the toroidal field magnet. Positive sign means anti-clockwise when viewed from above
- name description: Signed major-radius-weighted toroidal magnetic field produced by external coils in an axisymmetric current-free vacuum region.
- note: The name states the physics rather than the storage: R·B_φ **is** the poloidal current function F of an axisymmetric equilibrium, and in the vacuum region it is the constant set by the toroidal-field coil current. The unit `T.m` agrees on both sides and is itself the check that the quantity is the R-weighted field and not the field. This is the radius-independent counterpart of row 220, which is why that row's missing locus is recoverable by a reader who has both.

### 252. `difference_of_vacuum_poloidal_current_function_and_initial_vacuum_poloidal_current_function` — **correct**

- source path: `tf/delta_b_field_phi_vacuum_r`
- unit: `T.m` (data dictionary: `T.m`)
- data-dictionary text: Variation of (vacuum field times major radius in the toroidal field magnet) from the start of the plasma.
- name description: Signed change in the combined external-coil vacuum poloidal current function relative to its initial reference value, representing variation of the toroidal field multiplied by major radius.
- note: The `difference_of_X_and_Y` construction of row 230, applied to row 251's quantity, so both operands and the sign are readable from the name. The reference instant is what the name calls "initial" and the data dictionary calls "the start of the plasma"; those agree here but "initial" would not resolve on its own if a second reference instant ever entered the catalog.

### 253. `vertical_outline_of_limiter_tile` — **correct**

- source path: `wall/description_2d/limiter/unit/outline/z`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Height
- name description: Signed vertical (Z) coordinate of each point on a limiter-tile boundary contour in the right-handed cylindrical (R, φ, Z) frame.
- note: The vertical twin of the first half's row 85, `radial_outline_of_limiter_tile`, accepted there. One container, one object, two coordinates, one locus — which is what row 255 fails to do.

### 254. `toroidal_angular_width_of_limiter_tile` — **INCORRECT**

- source path: `wall/description_2d/limiter/unit/phi_extensions`
- unit: `rad` (data dictionary: `rad`)
- data-dictionary text: Simplified description of toroidal angle extensions of the unit, by a list of zones defined by their center and full width (in toroidal angle). In each of these zones, the unit outline remains the same. Leave this node empty for an axisymmetric unit. The first dimension gives the center and full width toroidal angle values for the unit. The second dimension represents the toroidal occurrences of the unit countour (i.e. the number of toroidal zones).
- **rejected spelling** `toroidal_angular_width_of_limiter_tile` → **proposed spelling** `toroidal_angular_centre_and_full_width_of_limiter_tile`
- why: The node is not a width. Its first dimension carries **two different quantities** — a zone centre angle and a zone full width — and its second dimension runs over the toroidal repetitions of the unit. The name publishes one of the two and silently drops the other, so a consumer reading `toroidal_angular_width_of_limiter_tile` and indexing the array gets a centre angle half the time. The description compounds it with a third reading: "full toroidal angular span of one limiter tile between its minimum and maximum φ coordinates" is the span of the tile, not the width of a zone in which the tile's outline is constant, and those differ whenever a unit repeats. This is the one-name-two-quantities class in its least fixable form, because the two quantities share a single data-dictionary node and cannot be split into two bindings; the name must therefore say that it covers a packed pair, or the catalog must decline the node until the data dictionary separates them.

### 255. `radial_outline_of_wall` — **INCORRECT**

- source path: `wall/description_2d/mobile/unit/outline/r`
- unit: `m` (data dictionary: `m`)
- data-dictionary text: Major radius
- name description: Major-radius coordinate of every point on a wall boundary outline, measured from the machine symmetry axis in the right-handed cylindrical (R, φ, Z) frame.
- **rejected spelling** `radial_outline_of_wall` → **proposed spelling** `radial_outline_of_plasma_facing_component`
- why: Its own Z sibling already carries the right base. The first half accepted `vertical_outline_of_plasma_facing_component` at its row 86 for `wall/description_2d/mobile/unit/outline/**z**`, and this is the same container's `r`. One container, one object, and two different loci across its two coordinates — the shape the first half rejected at its row 73 for the X-point. The spelling is wrong on its own terms as well: `description_2d/mobile/unit` is a **single movable plasma-facing unit**, and WEST's mobile units are its movable limiters, so a name that says "the wall" claims the whole vessel contour for one component's outline and collides with the fixed-wall outlines in the neighbouring `limiter` container at row 253. The proposed spelling is the one already accepted for the sibling, so this is a minority spelling of a base the cohort has already fixed, with a wrong-object consequence.

## Result

| | count |
| --- | --- |
| rows judged (cohort indices 205–255, inclusive) | **51** |
| judged **correct** | 37 |
| judged **INCORRECT**, each with a proposed spelling | **14** |
| correct + incorrect | **51** |
| incorrect as a share of the 51 rows | **27.5 %** |
| notes on correct rows (defect outside the name, or a convention recorded) | 35 note lines over 35 rows |
| — of those, a defect that lies in the description or the container rather than the name | 9 (rows 220, 226, 227, 228, 233, 237, 242, 248, 252) |
| rows whose identity also binds outside 205–255, named here and deferred to the collision sweep | 20 |
| collisions lying wholly inside 205–255 and therefore judged here | 2 (rows 213/214, rows 236/238) |

**14 of 51 — 27.5 % of this block — are not publishable as spelled.** Grouped
into the five classes the first half established, because the groups have
different remedies:

- **The name asserts more than the data supports** (3): `radiated_power_over_core_region` for power inside the whole last closed flux surface (229); `launched_power_of_lower_hybrid_antenna` for power the data dictionary says is coupled to the plasma (236); `pulse_duration` for the confined-plasma sub-interval of the pulse (249).
- **The name is bound to the wrong object** (1): `gap_at_outboard_midplane`, a measurement-position-to-separatrix gap, on the inner-to-outer separatrix distance (206).
- **One name, two physically different quantities** (6): `radial_coordinate_of_strike_point` on the inner leg, closing the pair the first half opened (212); `vertical_coordinate_of_strike_point` on both legs, with one description that names only the inner one (213, 214); `net_power_due_to_ion_cyclotron_heating` spanning an antenna launched-power node and a summary coupled-power node (235); `launched_power_of_lower_hybrid_antenna` spanning a per-launcher power and the system total (238); `toroidal_angular_width_of_limiter_tile` on an array that packs a zone centre beside its width (254). This remains the class a reader cannot repair — only a split can, and row 254 cannot even be split, because both quantities share one data-dictionary node.
- **Not self-descriptive** (3): `accumulated_total_gas_count` (219), `mhd_energy` (223), `energy_confinement_enhancement_factor` (225).
- **Minority spelling of a base the cohort already fixes** (1): `radial_outline_of_wall`, whose own vertical sibling is already `vertical_outline_of_plasma_facing_component` (255).

### What this block adds to the first half

The two halves are written in one shape and are summed, not compared:

| | first half | this block | together |
| --- | --- | --- | --- |
| rows judged | 86 | 51 | **137** |
| correct | 76 | 37 | **113** |
| incorrect | 10 | 14 | **24** |
| incorrect share | 11.6 % | 27.5 % | **17.5 %** |

The higher rate here is not a change of standard: it is where the rows sit.
This block is almost entirely `summary`, the IDS in which one leaf name has to
carry a locus, an averaging rule and a provenance that the path supplies for
free — and four of the fourteen rejections (212, 213, 214, 235) are the second
halves of defects the first half had already opened and could not close,
because their partner binding lay outside its every-fourth draw. Counting them
here is correct and is not double counting: each is a distinct binding with its
own verdict, and the first half's 10 does not contain any of them.

Three of the first half's predictions are confirmed by rows it could not see.
Its third outside-the-sample finding said `vertical_coordinate_of_strike_point`
would repeat the row-72 defect on the vertical axis; rows 213 and 214 are that
pair, and the confirmation is stronger than predicted, because the shared
identity's single description hard-codes the inner leg and is therefore false
of one of its two bindings. Its row-80 rejection turned on the data dictionary
saying **coupled** where the name said **launched**; rows 235, 236 and 238 are
three more bindings of that same distinction, one of which also spans a
per-launcher quantity and a system total. And its row-86 acceptance of
`vertical_outline_of_plasma_facing_component` is what makes row 255's
`radial_outline_of_wall` a rejection rather than a judgement call.

### The cohort estimate the two halves now support

The first half's sample was every fourth path-ordered row and was drawn before
any verdict, so its 11.6 % was an unbiased estimate of the cohort. This block
is a contiguous path-ordered range, not a random sample, so its 27.5 % is a
measurement **of this range** and not an estimate of the cohort — the summary
IDS is over-represented in it by construction. The defensible combined
statement is the count, not a re-extrapolation: **137 of the 341 accepted
bindings have now been judged one by one, and 24 of them carry a name defect.**
The remaining 204 bindings are judged by the other blocks of this sweep.

provisional: false — all 51 rows carry a verdict and the result section is closed.
