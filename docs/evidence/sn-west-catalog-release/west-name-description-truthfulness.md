# Is each shared identity's description true of every binding it holds?

provisional: false

Source record: `docs/evidence/sn-west-catalog-release/west-name-shared-identities.json`
(53 groups, 164 bindings, drawn over the whole 341-binding / 230-identity WEST
production cohort). No graph query was issued and no database was opened for
this sweep: every group already carries the identity, and for each binding the
source path, both units and both descriptions, with `dd_doc_parent` supplying
the parent container's text wherever the leaf's own documentation is empty or
the literal `"Value"`.

## What is being judged, and what is not

The question is **truthfulness**, not naming: does the identity's own
description state something that holds for *every* source path bound to it.
A shared identity is the normal case, not a defect — ordinal and index
dimensions are not carried in standard names, so an identity binding both
members of a pair is correct by convention. The damage this sweep looks for
sits in descriptions written while those bindings still looked like
ambiguities. No name is judged here, no rename is proposed, and none of the
five read-only inputs (the shared-identity record and its JSON, the repair
worklist, the ordinal revision, the superseded-target resolution) is edited.

Three verdicts:

| Verdict | Meaning |
|---|---|
| **TRUE-OF-ALL** | the description covers the whole set without selecting a member — the correct state for a shared identity |
| **NARROWER** | the description names fewer loci than the identity is bound to, but asserts nothing false about the others |
| **FALSE-OF-SOME** | the description asserts something untrue of at least one binding — a reader who trusts it attributes the value to the wrong object |

FALSE-OF-SOME is the serious class, and it is the class a rename cannot fix.

Where a leaf's documentation was too thin to decide (`"Major radius"`,
`"Value"`), the data-dictionary path documentation was read through the DD tool
at the configured version. Four such reads decided four rows and are quoted in
the detail sections below: `equilibrium/time_slice/contour_tree/node`,
`polarimeter/channel/polarization_initial`,
`hard_x_rays/emissivity_profile_1d/peak_position`, and the
`summary/heating_current_drive` power containers.

## The two known instances, reproduced independently

Both were re-derived from the JSON record before the prior write-ups were
opened, and both reproduce. One disagrees with the brief on a factual detail.

**`vertical_coordinate_of_strike_point` — reproduced as FALSE-OF-SOME.**
Its description opens *"Signed vertical (Z) coordinate of the **inner**
divertor strike point, where the **inner** separatrix leg intersects the
divertor target…"* while the identity is bound to both
`summary/boundary/strike_point_inner_z/value` (parent: *"Z position of the
inner strike point"*) and `summary/boundary/strike_point_outer_z/value`
(parent: *"Z position of the outer strike point"*). A reader of the outer
binding is told it is the inner leg.

**Disagreement to report: the identity holds two bindings in this cohort, not
three, and none of them is a constraint path.** The brief describes it as bound
to "the inner leg, the outer leg and a constraint path". The record carries
exactly two bindings, both under `summary/boundary/`, and a scan of all 53
groups finds no strike-point binding anywhere else in the cohort. The reason is
visible in the production manifest: `equilibrium/time_slice/boundary/strike_point/{r,z}`
and `equilibrium/time_slice/boundary_separatrix/strike_point/{r,z}` are excluded
at `imas_codex/standard_names/manifests/west_production_dd_paths.yaml:488–493`
with the reason `no_equilibrium_resolved_home (covered via summary/boundary/strike_point_*)`.
The defect class is confirmed exactly as described; the binding count is 2.

**`radial_coordinate_of_strike_point` — reproduced as TRUE-OF-ALL, and it is the
model for a correct one.** *"Major-radius location of **an individual** magnetic
strike point where **a** separatrix leg intersects a divertor target, expressed
in the right-handed cylindrical (R, φ, Z) frame."* Same two loci, same pairing,
and it selects neither. The register of that sentence — an indefinite bearer
plus the frame — is the register the replacement sentences below are written in.

**`toroidal_coordinate_of_line_of_sight` — reproduced as NARROWER, with the
count confirmed.** The description names *"the **first** reference point"*;
of its 14 bindings, 6 are `first_point/phi`, 6 are `second_point/phi` and 2 are
`third_point/phi` — **8 of 14 are second or third points**, matching the brief.
Its two sibling identities over the same geometry,
`radial_coordinate_of_line_of_sight` (16 bindings) and
`vertical_coordinate_of_line_of_sight` (16), both say *"a specified geometric
point"* / *"a designated point"* and are TRUE-OF-ALL — so within one family the
correct wording and the defective wording sit side by side.

## Per-group verdicts

`locus term` records whether the description names a locus, an ordinal or an
index term **at all** — wording that points at *which* object, place, member or
ordinal position the value belongs to, as distinct from wording that only
defines the physical quantity. `selects` marks the subset where that wording
picks out fewer than all of the identity's own bindings.

| # | Identity | Bindings | Verdict | Locus / ordinal / index term | Selects |
|---|---|---|---|---|---|
| 1 | `radial_coordinate_of_line_of_sight` | 16 | TRUE-OF-ALL | "a specified geometric point associated with a line of sight" | no |
| 2 | `vertical_coordinate_of_line_of_sight` | 16 | TRUE-OF-ALL | "a designated point defining a diagnostic line of sight" | no |
| 3 | `toroidal_coordinate_of_line_of_sight` | 14 | **NARROWER** | "the first reference point" | **yes** |
| 4 | `radial_coordinate_of_magnetic_axis` | 4 | **FALSE-OF-SOME** | "the magnetic-axis O-point" | **yes** |
| 5 | `faraday_angle` | 3 | TRUE-OF-ALL | — | no |
| 6 | `line_integrated_electron_number_density` | 3 | TRUE-OF-ALL | — | no |
| 7 | `lower_bound_photon_energy` | 3 | TRUE-OF-ALL | "lower boundary of an X-ray photon-energy band" | no |
| 8 | `normalized_toroidal_beta` | 3 | TRUE-OF-ALL | — | no |
| 9 | `normalized_toroidal_flux_coordinate` | 3 | TRUE-OF-ALL | "between the magnetic axis and equilibrium boundary" | no |
| 10 | `plasma_current` | 3 | TRUE-OF-ALL | — | no |
| 11 | `poloidal_magnetic_field` | 3 | TRUE-OF-ALL | — | no |
| 12 | `poloidal_magnetic_flux_at_plasma_boundary` | 3 | TRUE-OF-ALL | "on the last closed flux surface" | no |
| 13 | `poloidal_magnetic_flux_of_flux_loop` | 3 | TRUE-OF-ALL | "an individual flux loop" | no |
| 14 | `radial_coordinate_of_measurement_position` | 3 | **FALSE-OF-SOME** | "a measurement position" | **yes** |
| 15 | `reference_major_radius` | 3 | TRUE-OF-ALL | "a designated reference location" | no |
| 16 | `toroidal_angle_of_measurement_position` | 3 | TRUE-OF-ALL | "a measurement position" | no |
| 17 | `toroidal_vacuum_magnetic_field` | 3 | TRUE-OF-ALL | "at a reference major radius" | no |
| 18 | `upper_photon_energy` | 3 | TRUE-OF-ALL | "high-energy boundary of an X-ray photon-acceptance band" | no |
| 19 | `vertical_coordinate_of_magnetic_axis` | 3 | TRUE-OF-ALL | "the magnetic axis" | no |
| 20 | `volume_of_flux_surface` | 3 | TRUE-OF-ALL | "a nested magnetic flux surface" | no |
| 21 | `atomic_mass` | 2 | TRUE-OF-ALL | "a specified ion or neutral particle species" | no |
| 22 | `coolant_mass_flow_rate` | 2 | **NARROWER** | "a port or cooling loop" | **yes** |
| 23 | `coolant_temperature_at_inlet` | 2 | TRUE-OF-ALL | "a cooling loop or plant component" | no |
| 24 | `coolant_temperature_at_outlet` | 2 | TRUE-OF-ALL | "a cooling component or loop" | no |
| 25 | `effective_charge` | 2 | TRUE-OF-ALL | — | no |
| 26 | `elongation_of_plasma_boundary` | 2 | TRUE-OF-ALL | "the plasma boundary" | no |
| 27 | `initial_polarization_ellipticity_of_polarimeter_beam` | 2 | **FALSE-OF-SOME** | "only the ellipticity component of the initial polarization vector" | **yes** |
| 28 | `launched_power_of_lower_hybrid_antenna` | 2 | **FALSE-OF-SOME** | "of a lower-hybrid antenna … at the antenna input reference plane" | **yes** |
| 29 | `line_averaged_effective_charge` | 2 | TRUE-OF-ALL | "along a prescribed plasma line of sight" | no |
| 30 | `line_averaged_electron_density` | 2 | TRUE-OF-ALL | "a complete plasma propagation chord" | no |
| 31 | `lower_triangularity_of_plasma_boundary` | 2 | TRUE-OF-ALL | "the lower plasma-boundary extremum" | no |
| 32 | `mhd_energy` | 2 | TRUE-OF-ALL | — | no |
| 33 | `minor_radius_of_plasma_boundary` | 2 | TRUE-OF-ALL | "the last closed plasma-boundary contour" | no |
| 34 | `net_power_due_to_ion_cyclotron_heating` | 2 | **FALSE-OF-SOME** | "launched into the vacuum vessel by a specified heating launcher" | **yes** |
| 35 | `normalized_plasma_internal_inductance` | 2 | TRUE-OF-ALL | — | no |
| 36 | `normalized_toroidal_flux_coordinate_at_measurement_position` | 2 | **FALSE-OF-SOME** | "a physical measurement position" | **yes** |
| 37 | `poloidal_beta` | 2 | TRUE-OF-ALL | — | no |
| 38 | `poloidal_magnetic_flux_at_flux_surface` | 2 | TRUE-OF-ALL | "a nested magnetic surface" | no |
| 39 | `poloidal_magnetic_flux_at_magnetic_axis` | 2 | TRUE-OF-ALL | "at the magnetic axis" | no |
| 40 | `poloidal_plane_cross_sectional_area_of_flux_surface` | 2 | TRUE-OF-ALL | "a closed magnetic-flux-surface contour … poloidal plane" | no |
| 41 | `radial_coordinate_of_geometric_axis` | 2 | TRUE-OF-ALL | "the midpoint of the plasma boundary's radial extrema" | no |
| 42 | `radial_coordinate_of_strike_point` | 2 | TRUE-OF-ALL | "an individual magnetic strike point … a separatrix leg" | no |
| 43 | `safety_factor_at_magnetic_axis` | 2 | TRUE-OF-ALL | "the innermost closed flux surface … the magnetic axis" | no |
| 44 | `safety_factor_at_normalized_poloidal_magnetic_flux_equal_to_0_95` | 2 | TRUE-OF-ALL | "the surface labeled … 0.95, near but inside the plasma boundary" | no |
| 45 | `surface_area_of_flux_surface` | 2 | TRUE-OF-ALL | "a closed toroidal magnetic flux surface" | no |
| 46 | `toroidal_beta` | 2 | TRUE-OF-ALL | — | no |
| 47 | `toroidal_flux_coordinate` | 2 | TRUE-OF-ALL | "a nested magnetic flux surface" | no |
| 48 | `toroidal_magnetic_field_at_magnetic_axis` | 2 | TRUE-OF-ALL | "at the magnetic axis" | no |
| 49 | `upper_triangularity_of_plasma_boundary` | 2 | TRUE-OF-ALL | "its upper extremum … the geometric center" | no |
| 50 | `vertical_coordinate_of_geometric_axis` | 2 | TRUE-OF-ALL | "the plasma boundary's geometric axis" | no |
| 51 | `vertical_coordinate_of_strike_point` | 2 | **FALSE-OF-SOME** | "the inner divertor strike point … the inner separatrix leg" | **yes** |
| 52 | `volume_averaged_electron_density` | 2 | TRUE-OF-ALL | "the plasma volume enclosed by the last closed flux surface" | no |
| 53 | `wavelength_of_wave_beam` | 2 | TRUE-OF-ALL | — | no |

## The nine rows that are not TRUE-OF-ALL

Every row below carries the offending sentence verbatim, the bindings it is
untrue or under-informative of, and the replacement sentence I would write, in
the register the correct descriptions in this cohort already use — the register
of `radial_coordinate_of_strike_point`: an indefinite bearer, the mechanism,
and the frame.

### 4. `radial_coordinate_of_magnetic_axis` — FALSE-OF-SOME (1 of 4 bindings)

> "Major-radius coordinate locating the magnetic-axis O-point in the
> right-handed cylindrical (R, φ, Z) frame around which nested closed flux
> surfaces are organized."

False of **`equilibrium/time_slice/contour_tree/node/r`**. The leaf's own
documentation is only `"Major radius"`, which is why this row needed the
container read: `equilibrium/time_slice/contour_tree/node` is documented as
*"Nodes representing critical points (O-points and X-points) within the
poloidal flux map. These connectivity nodes define the topological structure of
the magnetic equilibrium."* A contour-tree node may be an X-point, which is not
the magnetic axis and has no nested closed flux surfaces organized around it;
a reader who trusts the description reads a separatrix saddle as the axis. True
of the other three (`equilibrium/time_slice/global_quantities/magnetic_axis/r`,
`summary/boundary/magnetic_axis_r/value`, `summary/local/magnetic_axis/position/r`).

**Replacement:** *"Major-radius coordinate of a designated critical point of
the poloidal flux map — a magnetic-axis O-point, or another O-point or X-point
node of the equilibrium contour tree — in the right-handed cylindrical
(R, φ, Z) frame."*

The cost of that sentence is real and worth stating: it is true of all four
bindings and tells three of them less than they deserve. The alternative remedy
is on the binding axis rather than the description axis, and that is outside
this node.

### 14. `radial_coordinate_of_measurement_position` — FALSE-OF-SOME (1 of 3)

> "Major-radius coordinate locating a measurement position by perpendicular
> distance from the toroidal symmetry axis in the right-handed cylindrical
> (R, φ, Z) frame."

False of **`camera_x_rays/aperture/centre/r`** — the major radius of the
camera's *aperture centre*, a viewing-optics element of the diagnostic's
geometry, not a position at which anything is measured. True of
`ece/channel/position/r` and `magnetics/b_field_phi_probe/position/r`, which
are measurement positions.

**Replacement:** *"Major-radius coordinate of a designated diagnostic reference
position — a channel's measurement position or an aperture centre — given as
perpendicular distance from the toroidal symmetry axis in the right-handed
cylindrical (R, φ, Z) frame."*

Recorded without proposing it: the read-only repair worklist already carries a
name repair moving the camera aperture binding elsewhere. If that repair lands
first, the present description becomes true of what remains and needs no edit.
The two remedies are alternatives, not additions.

### 22. `coolant_mass_flow_rate` — NARROWER (2 bindings)

> "Coolant mass throughput across a fluid-flow boundary, representing the rate
> at which coolant mass passes through a port or cooling loop."

Names the loop, while the identity is also bound to
`calorimetry/group/component/mass_flow` — *"Mass flow of the coolant going
through the component"*, under a container documented as *"Definition of
cooling loop components targeted for calorimetry"*. Nothing in the sentence is
false of a component; the enumeration simply names fewer loci than the identity
holds. Its two siblings `coolant_temperature_at_inlet` and
`coolant_temperature_at_outlet` already enumerate both loci, so this row is out
of line with its own family.

**Replacement:** *"Coolant mass throughput across a fluid-flow boundary, giving
the rate at which coolant mass passes through an instrumented cooling loop or
one of its components."*

### 27. `initial_polarization_ellipticity_of_polarimeter_beam` — FALSE-OF-SOME (1 of 2)

> "It gives only the ellipticity component of the initial polarization vector,
> not the ellipse orientation."

False of **`polarimeter/channel/polarization_initial`**, documented as
*"Initial polarization vector state before an optical beam enters the plasma,
serving as the reference for measured Faraday rotation and ellipticity"* — that
path *is* the polarization state the sentence says it is not. True of
`polarimeter/channel/ellipticity_initial` (*"Phase ellipticity of the
polarimeter beam before it propagates through the plasma"*).

**Replacement:** *"Initial polarization state of a polarimeter probing beam
before it enters the plasma, serving as the reference against which Faraday
rotation and ellipticity are measured on that beam."*

Two observations that are not verdicts. The two bindings are a state and one
component of that state, so no single sentence naming the component can be true
of both; the replacement above is the truthful minimum and the rest is a
binding question. And both paths carry data-dictionary unit `m` for a
dimensionless ellipticity — the identity's `sn_unit` agrees with `dd_unit` on
both bindings, so that discrepancy is upstream in the data dictionary and is
neither a description nor a naming defect.

### 28. `launched_power_of_lower_hybrid_antenna` — FALSE-OF-SOME (1 of 2)

> "Launched power of a lower-hybrid antenna is the net RF power entering the
> vacuum vessel after reflection at the antenna input reference plane."

False of **`summary/heating_current_drive/power_lh/value`** on the bearer:
that path is documented *"Total Lower Hybrid (LH) power coupled to the plasma.
Aggregated scalar parameter…"* — a machine total over all launchers, not one
antenna, against the per-launcher `summary/heating_current_drive/lh/power`
(*"LH heating power (PLH) coupled to the plasma from specific Lower Hybrid
launchers"*). A reader attributes the whole LHCD system's power to a single
antenna.

The sentence additionally asserts power *entering the vacuum vessel at the
antenna input reference plane* while both data-dictionary texts say *coupled to
the plasma*, which coupling efficiency separates by 10–30 %. That half is an
open contradiction already recorded against this plan and is **not settled
here**.

**Replacement:** *"Lower-hybrid radio-frequency power coupled to the plasma,
taken either from a single launcher or as the machine total over all launchers
according to the bound source."* If the open adjudication rules that these
nodes hold launched rather than coupled power, substitute *"launched into the
vacuum vessel"* for *"coupled to the plasma"* in that same sentence — the
per-launcher-versus-total half of the defect is independent of that ruling and
has to be fixed either way.

### 34. `net_power_due_to_ion_cyclotron_heating` — FALSE-OF-SOME (1 of 2)

> "Net ion-cyclotron radio-frequency power launched into the vacuum vessel by a
> specified heating launcher before absorption by plasma particles."

True of `ic_antennas/antenna/power_launched` (*"Total Ion Cyclotron Radio
Frequency (ICRF) power launched from the antenna into the vacuum vessel"*).
False of **`summary/heating_current_drive/ic/power/value`**, documented as
*"Ion Cyclotron (IC) resonance heating power coupled to the plasma from a
specific launcher"*. Coupled power is what survives reflection at the
antenna–plasma interface, so a reader of the summary binding is handed a
launched-power reading of a coupled-power number. Unlike the lower-hybrid row
above, the prior instruments agree this node holds coupled power, so the
differential falsity here is not contingent on the open adjudication.

**Replacement:** *"Ion-cyclotron radio-frequency heating power attributed to a
single launcher, taken at the launcher's output into the vacuum vessel or as
coupled to the plasma according to the bound source."*

No sentence can reconcile a 10–30 % physical difference between two bindings.
Telling the reader which binding is which is the most a description can do, and
it is strictly better than asserting one of them of both.

### 36. `normalized_toroidal_flux_coordinate_at_measurement_position` — FALSE-OF-SOME (1 of 2)

> "Dimensionless normalized toroidal-flux label that maps a physical
> measurement position onto a nested magnetic surface between the magnetic axis
> and equilibrium boundary."

False of **`hard_x_rays/emissivity_profile_1d/peak_position`**, documented as
*"Radial position, in normalized toroidal flux, where hard X-ray emissivity
reaches its maximum."* That is an inferred feature of a reconstructed
emissivity profile — the hard X-ray channels measure along chords and the peak
is derived afterwards — so there is no physical measurement position there to
map. True of `ece/channel/position/rho_tor_norm`.

**Replacement:** *"Dimensionless normalized toroidal-flux label locating a
designated position on a nested magnetic surface between the magnetic axis and
the equilibrium boundary — a diagnostic channel's measurement position, or a
feature of an emission profile such as its peak."*

### 51. `vertical_coordinate_of_strike_point` — FALSE-OF-SOME (1 of 2)

> "Signed vertical (Z) coordinate of the inner divertor strike point, where the
> inner separatrix leg intersects the divertor target in the right-handed
> cylindrical (R, φ, Z) frame."

False of **`summary/boundary/strike_point_outer_z/value`** (parent: *"Z
position of the outer strike point"*). True of
`summary/boundary/strike_point_inner_z/value`.

**Replacement**, mirroring its radial twin exactly: *"Signed vertical (Z)
location of an individual magnetic strike point where a separatrix leg
intersects a divertor target, expressed in the right-handed cylindrical
(R, φ, Z) frame."*

### 3. `toroidal_coordinate_of_line_of_sight` — NARROWER (8 of 14 bindings under-covered)

> "Toroidal angular coordinate of the first reference point on a diagnostic
> line of sight, locating that point around the machine symmetry axis."

The identity holds 6 `first_point/phi`, 6 `second_point/phi` and 2
`third_point/phi` bindings; the sentence names one ordinal member of an ordered
point set and leaves the other 8 uncovered. It is classed NARROWER rather than
FALSE-OF-SOME because first/second/third are ordered members of a single point
set — a reader of the second point is under-informed about which member is
meant — whereas inner/outer in row 51 is a contrastive qualifier that positively
excludes the other member. That is the distinction the two classes turn on, and
this row sits closest to the line.

**Replacement**, in the register its two siblings already use: *"Toroidal
angular coordinate of a designated point defining a diagnostic line of sight,
locating that point around the machine symmetry axis."*

## How far the class reaches beyond these 53 groups

**42 of the 53 descriptions (79.2 %) mention a locus, an ordinal or an index
term at all**; 11 define only the physical quantity and name no bearer
(rows 5, 6, 8, 10, 11, 25, 32, 35, 37, 46, 53).

Of those 42, **9 use that wording selectively** — naming an object, member or
ordinal that fewer than all of the identity's own bindings satisfy — and those
9 are exactly the 9 rows that are not TRUE-OF-ALL. Mentioning a locus is not
itself a defect: 33 of the 42 name one and still cover every binding, which is
what the correct register looks like. The defect is a locus term that *selects*.

The cohort holds 230 identities over 341 bindings; 53 are shared and the
remaining **177 identities hold exactly one binding each** (341 − 164 = 177).
A single-binding identity cannot carry this defect today — with one binding
there is no "some" for the description to be false of. So the estimate this
count supports is one of **exposure, not of present defects**:

| Quantity | Value | Basis |
|---|---|---|
| Shared identities whose description mentions a locus/ordinal/index | 42 of 53 (79.2 %) | counted above |
| Of those, wording that selects among the bindings | 9 of 42 (21.4 %) | the 9 non-TRUE-OF-ALL rows |
| Single-binding identities in the cohort | 177 | 341 − 164 |
| Expected to carry locus/ordinal/index wording | ≈ 140 | 79.2 % of 177 |
| Expected to become a truthfulness defect on a second attachment | ≈ 30 | 21.4 % of ≈ 140 |

The second figure carries an explicit assumption and should be read as an
order of magnitude: it holds only if a future second binding is drawn like the
ones in this cohort — a sibling member of the same array or a `summary` mirror
of the same quantity. It is not a prediction that 30 defects exist; it is the
number of descriptions that would have to be re-read the next time those
identities gain a binding.

## Observations that are not verdicts

- **164 bindings stand for 163 distinct source paths.**
  `equilibrium/time_slice/global_quantities/beta_tor_norm` appears twice under
  `normalized_toroidal_beta` (row 8) — the one-path-two-source-nodes case
  already recorded against this plan. Binding counts here follow the record's
  own edge count of 164, so one TRUE-OF-ALL binding is that duplicate.
- **Row 11 `poloidal_magnetic_field`** describes the magnitude *"formed from
  radial and vertical components"* while all three bindings are poloidal-field
  *probe* quantities, which measure a projection along the probe axis. The
  imprecision is uniform across all three bindings rather than differential, so
  it is not a truthfulness verdict under this node's question; it is a docs-axis
  observation.
- **Row 20 `volume_of_flux_surface`** reads *"cumulative from the magnetic axis
  toward the outermost closed surface"*, which describes a profile, while
  `equilibrium/time_slice/global_quantities/volume` is the scalar total plasma
  volume. That scalar is the limiting member of the family the sentence
  describes, so nothing is asserted falsely of it — TRUE-OF-ALL. The identity
  is separately contested on the naming axis in three read-only records; this
  sweep takes no position there.
- **Row 25 `effective_charge`** holds a local profile and a volume-averaged
  resistive estimate. The description names no locus and no averaging, so it is
  true of both; whether one identity should hold both is a naming question, not
  a description one.
- **Provenance is correctly absent from every description.** Fourteen bindings
  across rows 5, 6, 11 and 13 are `equilibrium/time_slice/constraints/*/measured`
  and `*/reconstructed` pairs, and no description mentions measurement or
  reconstruction. That is the intended design — provenance is an edge property,
  not description text — and it is why those four rows are TRUE-OF-ALL rather
  than split by provenance.

## Result

| Metric | Value |
|---|---|
| **groups_judged** | **53** |
| TRUE-OF-ALL | **44** groups — **131** of 164 bindings |
| NARROWER | **2** groups — **16** of 164 bindings |
| FALSE-OF-SOME | **7** groups — **17** of 164 bindings |
| Verdicts sum | 44 + 2 + 7 = **53** |
| Bindings accounted for | 131 + 16 + 17 = **164** |
| Bindings the defective wording is actually untrue of | **7** (one per FALSE-OF-SOME group) |
| Bindings a NARROWER description leaves uncovered | **9** (8 in row 3, 1 in row 22) |
| Descriptions mentioning a locus / ordinal / index at all | **42** of 53 (79.2 %) |
| Of those, wording that selects among the bindings | **9** — identical to the non-TRUE-OF-ALL set |
| Replacement sentences supplied | **9** — one for every NARROWER and FALSE-OF-SOME row |

**The headline.** 83 % of shared identities (44 of 53, 131 of 164 bindings)
carry a description that is true of everything they hold, so shared identity is
working as the convention intends. The damage is concentrated: **7 groups, 17
bindings, 7 individual bindings whose reader is told the wrong object** — an
X-point read as the magnetic axis, an aperture centre read as a measurement
position, a polarization state read as one component of itself, a machine-total
LH power read as one antenna's, a coupled IC power read as launched, an
emissivity peak read as a measurement position, and the outer strike point read
as the inner. Five of the seven are new to this sweep; two reproduce the
instances the brief named. None of them is fixable by a rename.
