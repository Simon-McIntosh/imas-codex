# Is each shared identity's description true of every binding it holds?

provisional: true

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
