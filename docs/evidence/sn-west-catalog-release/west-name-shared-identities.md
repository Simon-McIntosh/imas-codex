# WEST batch shared identities — one quantity, or several?

provisional: false — all 53 groups carry a verdict, the container-level
resolutions are recorded and the result table is closed.

53 of the 230 accepted identities in the WEST batch cohort are bound to more
than one data-dictionary source path, covering 164 of the 341 bindings. Sharing
is the normal case: one geometric or global quantity measured by many
instruments is exactly what a standard name is for. The question each group is
judged on is narrower and has one answer:

> Do all of this identity's bindings denote **the same physical quantity**
> measured at different places or by different instruments — **ONE-QUANTITY**,
> one spelling is correct — or do two or more of them denote **physically
> different quantities** — **MUST-SPLIT**, because a reader given the name
> alone would attribute a value to the wrong object?

A shared geometric identity across many diagnostics is not a defect for being
shared. What makes a group MUST-SPLIT is that the name cannot be read back onto
the right object.

## Inputs, and why no query was issued

Every group and every binding is read from
[`west-name-shared-identities.json`](west-name-shared-identities.json), which
was drawn from the live graph by the node that produced it. Each binding
carries its source path, `sn_unit`, `dd_unit`, the name's description and the
data-dictionary text, with `dd_doc_parent` supplying the parent container's
documentation wherever the leaf's own text is empty or the literal `Value`. No
database was opened for this judgement and no graph query was issued.

Two further files are read, both already in the tree:
[`west-name-audit.md`](west-name-audit.md), the every-fourth-row physical
correctness audit whose closing section points at three of these groups, and
[`west-name-cohort-remainder.json`](west-name-cohort-remainder.json), which
supplies the **sibling spellings** — what the other coordinates of the same
data-dictionary container are called — that a container-level judgement needs.
`imas_codex/standard_names/manifests/west_production_dd_paths.yaml` supplies
the batch's own migration map, which turns out to settle one group outright.

### The unit comparator was controlled before its zero was reported

Comparing `sn_unit` against `dd_unit` over all 164 bindings returns **zero
disagreements**. A zero is a claim about the instrument until the instrument is
shown to see something known present, so the identical comparator was run over
the 255 bindings of the cohort remainder, where it returns **three**:

```
DISAGREE: turn_count_of_toroidal_magnetic_field_probe | magnetics/b_field_phi_probe/turns | sn= '1' dd= ''
DISAGREE: turn_count_of_poloidal_magnetic_field_probe | magnetics/b_field_pol_probe/turns | sn= '1' dd= ''
DISAGREE: atomic_count | spectrometer_visible/channel/isotope_ratios/isotope/element/atoms_n | sn= '1' dd= ''
```

The comparator fires. The zero over the shared-identity bindings is therefore a
real absence: **no group in this record carries a unit disagreement**, and the
unit half of the narrower-failure check is reported once here rather than
repeated as an empty line under all 53 verdicts.

### What is not relitigated

Three adjudications are recorded as settled and are taken as given wherever a
group touches them: the `*_of_flux_surface` family including
`volume_of_flux_surface` and `area_of_flux_surface`; the
`back_surface_distance_of_antenna_strap` spelling; and the etendue `_detector`
spelling. Where a group falls inside one of these, the verdict says so and
stops.

## Verdicts

### 1. `radial_coordinate_of_line_of_sight` — **ONE-QUANTITY** (16 bindings)

Sixteen bindings, eight diagnostics (`bremsstrahlung_visible`, `camera_x_rays`,
`hard_x_rays`, `interferometer`, `polarimeter`, `soft_x_rays`,
`spectrometer_visible`), across `first_point`, `second_point` and — for
`interferometer` and `polarimeter` — `third_point`. Every data-dictionary text
is `Major radius`; every unit is `m`. The quantity is the major-radius
coordinate of a point that defines a diagnostic sight line, and it is the same
quantity wherever the sight line is. Which endpoint of the chord a value belongs
to is carried by the source path, not by the quantity, so the sharing is the
normal case.

### 2. `vertical_coordinate_of_line_of_sight` — **ONE-QUANTITY** (16 bindings)

The Z axis of group 1, with the same sixteen containers and the same
data-dictionary text throughout (`Height`, `m`). The description —
"designated point defining a diagnostic line of sight" — is written for a point
in general and so covers first, second and third points alike.

### 3. `toroidal_coordinate_of_line_of_sight` — **ONE-QUANTITY**, with a description note (14 bindings)

The φ axis of the same family, fourteen bindings (`hard_x_rays` has no φ in the
batch). One physical quantity for the same reason as groups 1 and 2.

> **Narrower-description note — confirms the prior reading.** The description
> reads "Toroidal angular coordinate of **the first reference point** on a
> diagnostic line of sight", while the identity is bound to `second_point/phi`
> in all seven diagnostics and to `third_point/phi` in `interferometer` and
> `polarimeter` — 8 of its 14 bindings are to a point the description does not
> name. `west-name-audit.md` reports this at sample row 46 as a description
> defect rather than a name defect; judged here independently, that reading is
> **confirmed**. The remedy is the group 1 and 2 wording ("a specified /
> designated point"), not a split: the union of the two records counts this
> finding once.

> **Family note.** This group spells the axis `toroidal_coordinate_`, while
> group 16 spells the same angular axis `toroidal_angle_of_...` and the
> `camera_x_rays` aperture sibling spells it `toroidal_coordinate_of_aperture`.
> The catalog carries both stems for one axis. That is a grammar question for
> the whole catalog rather than a defect in this group, and is recorded under
> follow-ons.

### 4. `radial_coordinate_of_magnetic_axis` — **MUST-SPLIT** (4 bindings)

Three of the four bindings are the magnetic axis and are correctly named:
`equilibrium/time_slice/global_quantities/magnetic_axis/r` ("Major radius of the
magnetic axis"), `summary/boundary/magnetic_axis_r/value` (parent: "R position
of the magnetic axis") and `summary/local/magnetic_axis/position/r`.

The fourth, **`equilibrium/time_slice/contour_tree/node/r`**, is not the
magnetic axis. Its leaf text is the bare "Major radius", so the name was formed
without the container's meaning — and this batch's own manifest states that
meaning explicitly, because `contour_tree/node` is where the retired X-point
paths migrate to:

```yaml
- { from: equilibrium/time_slice/boundary/x_point/r,            to: equilibrium/time_slice/contour_tree/node/r }
- { from: equilibrium/time_slice/boundary_separatrix/x_point/r, to: equilibrium/time_slice/contour_tree/node/r }
```
(`imas_codex/standard_names/manifests/west_production_dd_paths.yaml`, the
migration section of the WEST batch predicate itself.)

A contour-tree node is a **critical point of the poloidal flux** — an O-point,
an X-point, or a saddle — and X-points are precisely what the manifest routes
into it. A reader given `radial_coordinate_of_magnetic_axis` would read an
X-point's major radius as the magnetic axis: the two are different objects,
metres apart, with opposite topological character.

| binding | spelling |
| --- | --- |
| `equilibrium/time_slice/global_quantities/magnetic_axis/r` | keeps `radial_coordinate_of_magnetic_axis` |
| `summary/boundary/magnetic_axis_r/value` | keeps `radial_coordinate_of_magnetic_axis` |
| `summary/local/magnetic_axis/position/r` | keeps `radial_coordinate_of_magnetic_axis` |
| `equilibrium/time_slice/contour_tree/node/r` | **needs** `radial_coordinate_of_flux_contour_critical_point` |

The distinction in plain language: the magnetic axis is the single innermost
point around which closed flux surfaces nest; a contour-tree node is any point
where the flux contours change topology, which includes every X-point of the
separatrix. One is unique per equilibrium; the other is an indexed set.

> **Container note.** `contour_tree/node/r` is the only member of its container
> in the WEST batch — the manifest lists no `contour_tree/node/z` or
> `/phi` — so there is no sibling disagreement to resolve here, only the single
> mis-attributed axis. `west-name-audit.md` row 73 finds the mirror-image defect
> on the surviving `summary/boundary/x_point_main/r`, where the X-point spelling
> is under- rather than over-specified.

### 5. `faraday_angle` — **ONE-QUANTITY** (3 bindings)

`equilibrium/time_slice/constraints/faraday_angle/measured`, the same
container's `/reconstructed`, and `polarimeter/channel/faraday_angle`. All three
are the rotation of a probing wave's polarization plane on crossing the plasma,
in `rad`. Measured against reconstructed is a **provenance** distinction — the
same quantity carried by an observation and by an equilibrium solution — and
provenance is a controlled vocabulary on the source relation, never a name
segment. One spelling is correct.

> `west-name-audit.md` separately judges the spelling `faraday_angle` as not
> self-descriptive. That is a different axis from this node's question and is
> neither confirmed nor contradicted here; the two records do not double-count,
> because that finding is about the stem and this verdict is about the sharing.

### 6. `line_integrated_electron_number_density` — **ONE-QUANTITY** (3 bindings)

`equilibrium/.../constraints/n_e_line/measured`, `/reconstructed`, and
`interferometer/channel/n_e_line`. One quantity — the free-electron column
density along the full chord, the interferometer text being explicit that no
divide-by-two correction is applied for a reflected channel — carried by an
observation and by a reconstruction. Provenance again, not a second quantity.

### 7. `lower_bound_photon_energy` — **ONE-QUANTITY** (3 bindings)

`hard_x_rays/channel/energy_band/lower_bound`,
`hard_x_rays/emissivity_profile_1d/lower_bound` and
`soft_x_rays/channel/energy_band/lower_bound`, all "Lower bound of the energy
band" in `eV`. The low edge of a photon-energy acceptance band is one quantity
whether the band belongs to a detector channel, to an inverted emissivity
profile, or to a different X-ray diagnostic.

### 8. `normalized_toroidal_beta` — **ONE-QUANTITY**, with a duplicate-binding note (3 bindings)

`equilibrium/time_slice/global_quantities/beta_tor_norm` and
`summary/global_quantities/beta_tor_norm_mhd/value` (parent: "Normalised
toroidal beta, using the pressure determined by an equilibrium reconstruction
code"). Both are 100·β_tor·a·B0/Ip; the summary qualifier records where the
pressure came from, which is provenance.

> **Duplicate-binding note.** This group's `source_count` is 3 but it carries
> only **two distinct paths**: `equilibrium/time_slice/global_quantities/beta_tor_norm`
> appears **twice**, with identical unit, description and documentation. It is
> the only group of the 53 in which a path repeats. That is a duplicate
> `StandardNameSource` row rather than a shared identity, and it is recorded
> under follow-ons — it does not change this verdict, and the binding
> accounting below counts the row as drawn so that the total still reconciles
> to 164.

### 9. `normalized_toroidal_flux_coordinate` — **ONE-QUANTITY** (3 bindings)

`core_profiles/profiles_1d/grid/rho_tor_norm`,
`equilibrium/time_slice/profiles_1d/rho_tor_norm` and
`hard_x_rays/emissivity_profile_1d/rho_tor_norm`. One dimensionless radial
label, normalized to the same equilibrium boundary in all three, used as the
abscissa of three different profiles. Sharing a coordinate across the profiles
it indexes is the case this identity exists for.

### 10. `plasma_current` — **ONE-QUANTITY** (3 bindings)

`equilibrium/time_slice/global_quantities/ip`, `magnetics/ip` and
`summary/global_quantities/ip/value`. One quantity — net toroidal current in the
plasma column, `A`, same sign convention stated in two of the three texts —
obtained three ways. The `magnetics/ip` text notes its array corresponds to a
set of calculation methods, which is again provenance within one quantity.

### 11. `poloidal_magnetic_field` — **ONE-QUANTITY** (3 bindings)

`equilibrium/.../constraints/b_field_pol_probe/measured`, `/reconstructed` and
`magnetics/b_field_pol_probe/field`. All three are the poloidal-plane magnetic
field at a poloidal field probe, in `T`, split only by provenance.

> **Description note (not a split).** The description says the quantity is
> "formed from radial and vertical components", i.e. a magnitude, whereas
> `magnetics/b_field_pol_probe/field` is the field **component along the
> probe's sensing direction** in the R–Z plane. All three bindings share that
> reading, so it is an assertion defect common to the group rather than a
> distinction between its members, and it is recorded under follow-ons.

### 12. `poloidal_magnetic_flux_at_plasma_boundary` — **ONE-QUANTITY** (3 bindings)

`core_profiles/profiles_1d/grid/psi_boundary`,
`equilibrium/time_slice/boundary/psi` and
`equilibrium/time_slice/global_quantities/psi_boundary`. One quantity: the value
of ψ on the surface taken as the plasma boundary, which is what all three texts
describe and what all three are used for — the outer normalization reference.

### 13. `poloidal_magnetic_flux_of_flux_loop` — **ONE-QUANTITY** (3 bindings)

`equilibrium/.../constraints/flux_loop/measured`, `/reconstructed` and
`magnetics/flux_loop/flux`. Flux linked by one loop, `Wb`, observation and
reconstruction of the same quantity.

### 14. `radial_coordinate_of_measurement_position` — **MUST-SPLIT** (3 bindings)

Three bindings, three different kinds of locus, all with the bare
data-dictionary text "Major radius":

- `ece/channel/position/r` — where the electron-cyclotron channel's emission
  originates. A genuine measurement position.
- `camera_x_rays/aperture/centre/r` — the centre of an **instrument aperture**.
  The aperture is a hole in front of the detector; no measurement is made there.
- `magnetics/b_field_phi_probe/position/r` — the **location of a sensor coil**.
  The probe measures the field at itself, so its position is a piece of
  instrument geometry, not the locus of a plasma quantity.

A reader given `radial_coordinate_of_measurement_position` alone would place an
aperture and a pickup coil where a plasma measurement was taken.

| binding | spelling |
| --- | --- |
| `ece/channel/position/r` | keeps `radial_coordinate_of_measurement_position` |
| `camera_x_rays/aperture/centre/r` | **needs** `radial_coordinate_of_aperture` |
| `magnetics/b_field_phi_probe/position/r` | **needs** `radial_coordinate_of_toroidal_magnetic_field_probe` |

Both proposed spellings already exist in the cohort on the sibling axes, so
neither invents a stem: `camera_x_rays/aperture/centre/phi` is already
`toroidal_coordinate_of_aperture`, and `magnetics/b_field_phi_probe/position/z`
is already `vertical_coordinate_of_toroidal_magnetic_field_probe`
(`west-name-audit.md` row 51, judged correct there).

**Confirms the prior reading.** `west-name-audit.md`'s closing finding 1 reports
this group with the same three loci and the same two proposed spellings. Judged
here independently from the group's own evidence, it is confirmed; the union of
the two records counts it once.

> **Sibling-disagreement resolution for `magnetics/b_field_phi_probe/position`.**
> The container's three axes currently read
> `radial_coordinate_of_measurement_position` (r),
> `toroidal_angle_of_measurement_position` (φ, group 16 below) and
> `vertical_coordinate_of_toroidal_magnetic_field_probe` (z). The **z sibling is
> the outlier by count and the correct one by physics**, so the container as a
> whole should carry the `_of_toroidal_magnetic_field_probe` locus and the r and
> φ axes are the two that move.

### 15. `reference_major_radius` — **ONE-QUANTITY** (3 bindings)

`core_profiles/vacuum_toroidal_field/r0`,
`equilibrium/vacuum_toroidal_field/r0` and `summary/global_quantities/r0/value`
carry word-for-word the same data-dictionary text. One quantity, three IDSs
declaring the same machine reference.

### 16. `toroidal_angle_of_measurement_position` — **MUST-SPLIT** (3 bindings)

The φ axis of the group 14 defect, and it is **not** the same three containers:
here the poloidal probe joins the toroidal one, so the collision reaches a
container group 14 does not touch.

- `ece/channel/position/phi` — a genuine measurement position.
- `magnetics/b_field_phi_probe/position/phi` — a sensor coil's toroidal
  location.
- `magnetics/b_field_pol_probe/position/phi` — a second sensor coil's toroidal
  location, on a different probe type.

| binding | spelling |
| --- | --- |
| `ece/channel/position/phi` | keeps `toroidal_angle_of_measurement_position` |
| `magnetics/b_field_phi_probe/position/phi` | **needs** `toroidal_angle_of_toroidal_magnetic_field_probe` |
| `magnetics/b_field_pol_probe/position/phi` | **needs** `toroidal_angle_of_poloidal_magnetic_field_probe` |

The distinction in plain language: an ECE channel's position is a point in the
plasma whose emission reaches the instrument; a magnetic probe's position is
where a piece of hardware is bolted to the vessel. Attributing one to the other
misplaces the measurement by the whole minor radius.

> **Sibling-disagreement resolution for `magnetics/b_field_pol_probe/position`.**
> This container reads `radial_coordinate_of_poloidal_magnetic_field_probe` (r,
> `west-name-audit.md` row 53, correct),
> `toroidal_angle_of_measurement_position` (φ) and
> `vertical_coordinate_of_poloidal_magnetic_field_probe` (z, correct). **φ is
> the outlier, two-to-one**, and the container should carry the
> `_of_poloidal_magnetic_field_probe` locus its other two axes already carry.

> **New relative to the prior record — extends rather than contradicts.**
> `west-name-audit.md` finding 1 reaches the toroidal probe's `r` only. The φ
> axis of both probes is found here, and the poloidal probe's φ is a container
> the prior record does not name at all. Nothing in the prior reading is
> contradicted; two bindings are added to the same defect class. Counted once
> in the union as two additional bindings.

### 17. `toroidal_vacuum_magnetic_field` — **ONE-QUANTITY** (3 bindings)

`core_profiles/vacuum_toroidal_field/b0`,
`equilibrium/vacuum_toroidal_field/b0` and `summary/global_quantities/b0/value`
— the partner of group 15, with the same text in all three and the same
consistency requirement against the `tf` IDS. One quantity.

### 18. `upper_photon_energy` — **ONE-QUANTITY** (3 bindings)

The high-energy mirror of group 7, on the same three containers. One quantity.

> **Family note.** The low edge is spelled `lower_bound_photon_energy` and the
> high edge `upper_photon_energy` — one carries `bound`, the other does not,
> for the two edges of a single band. That is a grammar asymmetry across two
> identities rather than a defect inside either, and it is recorded under
> follow-ons.

### 19. `vertical_coordinate_of_magnetic_axis` — **ONE-QUANTITY** (3 bindings)

`equilibrium/time_slice/global_quantities/magnetic_axis/z`,
`summary/boundary/magnetic_axis_z/value` and
`summary/local/magnetic_axis/position/z`. All three are the magnetic axis. Note
that this group does **not** carry a `contour_tree/node/z` binding — the WEST
manifest admits only the `r` axis of that container — so the group 4 defect has
no counterpart here and this identity is clean.

### 20. `volume_of_flux_surface` — **ONE-QUANTITY** (3 bindings)

`core_profiles/profiles_1d/grid/volume`,
`equilibrium/time_slice/profiles_1d/volume` and
`equilibrium/time_slice/global_quantities/volume`. The first two are explicitly
the volume enclosed by a flux surface; the third is "Total plasma volume", which
is the same quantity evaluated on the outermost closed surface rather than a
different one. **This name is on the settled list** — the `*_of_flux_surface`
family is adjudicated and is not relitigated here — and the sharing is
consistent with that adjudication.

### 21. `atomic_mass` — **ONE-QUANTITY** (2 bindings)

`spectrometer_mass/channel/a` ("Atomic mass measured by this channel") and
`spectrometer_visible/channel/isotope_ratios/isotope/element/a` ("Mass of
atom"), both in `u`. The mass assigned to an atomic species is one quantity;
whether it is what a mass-spectrometer channel is tuned to or what an isotope
entry declares is a locus, not a second quantity.

### 22. `coolant_mass_flow_rate` — **ONE-QUANTITY** (2 bindings)

`calorimetry/cooling_loop/mass_flow` and `calorimetry/group/component/mass_flow`
— the same quantity across a whole loop and across one component in it. One
spelling; the source path says which boundary the flow crosses.

### 23. `coolant_temperature_at_inlet` — **ONE-QUANTITY** (2 bindings)

`calorimetry/cooling_loop/temperature_in` and
`calorimetry/group/component/temperature_in`, loop and component inlet. Same
quantity, two loci.

### 24. `coolant_temperature_at_outlet` — **ONE-QUANTITY** (2 bindings)

The outlet partner of group 23, on the same two containers.

### 25. `effective_charge` — **MUST-SPLIT** (2 bindings)

The two bindings differ by an **averaging operator**, which this catalog already
treats as producing distinct identities:

- `core_profiles/profiles_1d/zeff` — "Effective charge". A **local profile**
  value, Z_eff at one flux surface.
- `core_profiles/global_quantities/z_eff_resistive` — "**Volume average**
  plasma effective charge, estimated from the flux consumption in the ohmic
  phase". A single **volume-averaged scalar** for the discharge.

These are different numbers with different physical content: a local Z_eff can
be several times the volume average in an impurity-peaked or impurity-hollow
profile, and the two are not interchangeable in any calculation. The catalog
itself settles the principle, because it already carries
`line_averaged_effective_charge` (group 29) as a separate identity from
`effective_charge`, and `volume_averaged_electron_density` (group 52) as a
separate identity from the local electron density. A volume average is a
different quantity from the local field it averages.

| binding | spelling |
| --- | --- |
| `core_profiles/profiles_1d/zeff` | keeps `effective_charge` |
| `core_profiles/global_quantities/z_eff_resistive` | **needs** `volume_averaged_effective_charge` |

The distinction in plain language: one is the impurity content *here*, on one
flux surface; the other is the impurity content *of the whole plasma*, a single
number per time. The proposed spelling completes an averaging family the catalog
already spells two-thirds of, and deliberately omits "resistive" — how the
average was obtained is provenance and does not belong in the name.

### 26. `elongation_of_plasma_boundary` — **ONE-QUANTITY** (2 bindings)

`equilibrium/time_slice/boundary/elongation` and
`summary/boundary/elongation/value`, the summary parent repeating the
equilibrium text verbatim. One shape parameter of one boundary.

### 27. `initial_polarization_ellipticity_of_polarimeter_beam` — **MUST-SPLIT** (2 bindings)

The two bindings are two different optical descriptors of the same beam, not one
descriptor at two loci — they are siblings inside one `polarimeter/channel`, so
a single discharge carries both simultaneously with different values:

- `polarimeter/channel/ellipticity_initial` — "Initial ellipticity before
  entering the plasma". The ratio of the polarization ellipse's semiaxes.
- `polarimeter/channel/polarization_initial` — "Initial **polarization vector**
  before entering the plasma". The polarization state itself, which is the
  ellipse's orientation, not its shape.

The name's own description contains the admission: it says it gives "**only the
ellipticity component of the initial polarization vector, not the ellipse
orientation**" — a sentence written to explain away a collision rather than to
describe a quantity. A reader given the name alone would take
`polarization_initial` for an ellipticity, which is exactly the component the
description says it is not.

| binding | spelling |
| --- | --- |
| `polarimeter/channel/ellipticity_initial` | keeps `initial_polarization_ellipticity_of_polarimeter_beam` |
| `polarimeter/channel/polarization_initial` | **needs** `initial_polarization_of_polarimeter_beam` |

The distinction in plain language: ellipticity is how round the polarization
ellipse is; polarization is which way it is tilted. Faraday rotation changes the
second and Cotton–Mouton the first, so a polarimeter analysis that confuses them
attributes the wrong plasma effect to the signal.

> **Unit note (both bindings, not a split driver).** Both carry `m` in both
> `sn_unit` and `dd_unit`, so the unit comparator is silent — yet an ellipticity
> and a polarization state are dimensionless. The two agree on a unit that is
> wrong for either quantity, which is a defect the agreement check cannot see.
> Recorded under follow-ons.

### 28. `launched_power_of_lower_hybrid_antenna` — **MUST-SPLIT** (2 bindings)

Both bindings are in `summary/heating_current_drive`, and they differ by
**aggregation**:

- `.../lh/power/value` — parent: "LH heating power coupled to the plasma **from
  this launcher**". Per-launcher, indexed.
- `.../power_lh/value` — parent: "**Total** LH power coupled to the plasma".
  The machine total, summed over launchers.

On WEST, with more than one lower-hybrid launcher, these are different numbers
in the same time trace, and the total is the larger by construction. A reader
given `launched_power_of_lower_hybrid_antenna` would attribute the whole
system's power to a single antenna.

| binding | spelling |
| --- | --- |
| `summary/heating_current_drive/lh/power/value` | keeps `launched_power_of_lower_hybrid_antenna` |
| `summary/heating_current_drive/power_lh/value` | **needs** `total_launched_power_of_lower_hybrid_antennas` |

The distinction in plain language: one antenna's contribution, against every
antenna's contribution added up.

> **Assertion note (shared by both bindings, so not the split driver).** The
> name says **launched** while both data-dictionary parents say **coupled to the
> plasma**. Launched power is what leaves the antenna; coupled power is what the
> plasma absorbs, the difference being reflection at the launcher mouth — on a
> lower-hybrid system a non-negligible fraction. `west-name-audit.md` makes the
> same finding on the ion-cyclotron analogue. Because it applies equally to both
> members here, it does not separate them; the corrected pair should read
> `coupled_power_...` on both sides if the data-dictionary text is taken at its
> word. Recorded under follow-ons.

### 29. `line_averaged_effective_charge` — **ONE-QUANTITY** (2 bindings)

`bremsstrahlung_visible/channel/zeff_line_average` ("Average effective charge
along the line of sight") and `summary/line_average/zeff/value`. Same averaging
operator, same quantity, one measured on a named chord and one published as the
discharge's line average. This group is the precedent group 25 is judged
against.

### 30. `line_averaged_electron_density` — **ONE-QUANTITY** (2 bindings)

`interferometer/channel/n_e_line_average` — explicitly the full-chord integral
divided by chord length — and `summary/line_average/n_e/value`. Same operator,
same quantity.

### 31. `lower_triangularity_of_plasma_boundary` — **ONE-QUANTITY** (2 bindings)

`equilibrium/time_slice/boundary/triangularity_lower` and
`summary/boundary/triangularity_lower/value`, identical text. One quantity.

### 32. `mhd_energy` — **ONE-QUANTITY** (2 bindings)

`equilibrium/time_slice/global_quantities/energy_mhd` and
`summary/global_quantities/energy_mhd`, both 3/2 ∫p dV with p the total kinetic
pressure. The summary text adds that the pressure comes from an equilibrium
reconstruction code, which is provenance. One quantity.

### 33. `minor_radius_of_plasma_boundary` — **ONE-QUANTITY** (2 bindings)

`equilibrium/time_slice/boundary/minor_radius` and
`summary/boundary/minor_radius/value`, both carrying the same
`(Rmax − Rmin)/2` definition. One quantity.

### 34. `net_power_due_to_ion_cyclotron_heating` — **MUST-SPLIT** (2 bindings)

Unlike group 28, the two bindings here are **both per-launcher** and differ in
**which power** they are:

- `ic_antennas/antenna/power_launched` — "Power **launched** from this antenna
  into the vacuum vessel".
- `summary/heating_current_drive/ic/power/value` — parent: "IC heating power
  **coupled to the plasma** from this launcher".

Launched and coupled power are separated by the reflected power at the antenna
mouth. On an ion-cyclotron system the coupling resistance swings with the
edge-density profile and with ELMs, so the two traces differ transiently by
tens of percent and their ratio is itself a measured quantity. They cannot share
a name: a reader computing a power balance from the coupled trace would
double-count reflection if handed the launched one.

| binding | spelling |
| --- | --- |
| `ic_antennas/antenna/power_launched` | keeps a launched spelling — `launched_power_of_ion_cyclotron_antenna` |
| `summary/heating_current_drive/ic/power/value` | **needs** `coupled_power_of_ion_cyclotron_antenna` |

Neither keeps `net_power_due_to_ion_cyclotron_heating` unchanged: "net" names
neither side of the distinction, and the existing description ("launched into
the vacuum vessel before absorption") matches only the first binding. The plain
language: one is what the transmitter puts into the antenna's output, the other
is what the plasma takes.

> `west-name-audit.md` makes the launched-against-coupled finding on
> `total_power_due_to_ion_cyclotron_heating` at a different source path. That
> finding and this one are the same defect class on different identities; the
> union counts two instances, not one.

### 35. `normalized_plasma_internal_inductance` — **ONE-QUANTITY** (2 bindings)

`equilibrium/time_slice/global_quantities/li_3` and
`summary/global_quantities/li_3/value`, the summary parent giving the li_3
definition the equilibrium leaf leaves implicit. Same definition, one quantity.

### 36. `normalized_toroidal_flux_coordinate_at_measurement_position` — **MUST-SPLIT** (2 bindings)

The two bindings locate **different kinds of thing** on the same coordinate:

- `ece/channel/position/rho_tor_norm` — "Normalised toroidal flux coordinate".
  Where a channel's measurement comes from: an instrument property, known from
  the channel's frequency and the field.
- `hard_x_rays/emissivity_profile_1d/peak_position` — "Normalised toroidal flux
  coordinate position **at which the emissivity peaks**". Where a **feature of
  an inverted profile** sits: a plasma property, an output of the inversion that
  moves with the discharge.

One is where the instrument looks; the other is where the plasma is brightest.
A reader given the shared name would take an emissivity peak for a diagnostic
sight position — and on a hard-X-ray system the peak migrates during current
drive while the channel geometry does not move at all.

| binding | spelling |
| --- | --- |
| `ece/channel/position/rho_tor_norm` | keeps `normalized_toroidal_flux_coordinate_at_measurement_position` |
| `hard_x_rays/emissivity_profile_1d/peak_position` | **needs** `normalized_toroidal_flux_coordinate_of_emissivity_peak` |

### 37. `poloidal_beta` — **ONE-QUANTITY** (2 bindings)

`equilibrium/time_slice/global_quantities/beta_pol` and
`summary/global_quantities/beta_pol_mhd/value`, both carrying
`4∫p dV / (R₀ μ₀ Ip²)`. One quantity; the summary's `_mhd` qualifier names the
pressure's provenance.

### 38. `poloidal_magnetic_flux_at_flux_surface` — **ONE-QUANTITY** (2 bindings)

`core_profiles/profiles_1d/grid/psi` and
`equilibrium/time_slice/profiles_1d/psi` — the same flux-surface label used as
the abscissa of two profile sets. One quantity.

### 39. `poloidal_magnetic_flux_at_magnetic_axis` — **ONE-QUANTITY** (2 bindings)

`core_profiles/profiles_1d/grid/psi_magnetic_axis` and
`equilibrium/time_slice/global_quantities/psi_axis`. Same value, two IDSs; the
inner normalization reference to group 12's outer one.

### 40. `poloidal_plane_cross_sectional_area_of_flux_surface` — **ONE-QUANTITY** (2 bindings)

`core_profiles/profiles_1d/grid/area` and
`equilibrium/time_slice/profiles_1d/area`, both "Cross-sectional area of the
flux surface". Inside the settled `*_of_flux_surface` family; not relitigated.

### 41. `radial_coordinate_of_geometric_axis` — **ONE-QUANTITY** (2 bindings)

`equilibrium/time_slice/boundary/geometric_axis/r` and
`summary/boundary/geometric_axis_r/value`, the summary parent supplying the
`(Rmax + Rmin)/2` definition the equilibrium leaf leaves as the bare "Major
radius". One quantity.

### 42. `radial_coordinate_of_strike_point` — **MUST-SPLIT** (2 bindings)

`summary/boundary/strike_point_inner_r/value` (parent: "R position of the
**inner** strike point") and `summary/boundary/strike_point_outer_r/value`
(parent: "R position of the **outer** strike point"). The two divertor legs
land at different major radii — on WEST roughly the inner and outer halves of
the lower divertor — and carry different heat flux, different geometry and
different control significance. Nothing in the name chooses between them.

| binding | spelling |
| --- | --- |
| `summary/boundary/strike_point_inner_r/value` | **needs** `radial_coordinate_of_inner_strike_point` |
| `summary/boundary/strike_point_outer_r/value` | **needs** `radial_coordinate_of_outer_strike_point` |

**Neither binding keeps the existing spelling.** An unqualified strike point is
not a locus a reader can resolve, so the existing name is retired rather than
narrowed to one leg.

**Confirms the prior reading.** `west-name-audit.md` row 72 rejects this
spelling and proposes the same pair. Judged here independently from the group's
own evidence, it is confirmed; counted once in the union.

### 43. `safety_factor_at_magnetic_axis` — **ONE-QUANTITY** (2 bindings)

`equilibrium/time_slice/global_quantities/q_axis` and
`summary/local/magnetic_axis/q/value`, whose parent context is the magnetic
axis. One quantity.

### 44. `safety_factor_at_normalized_poloidal_magnetic_flux_equal_to_0_95` — **ONE-QUANTITY** (2 bindings)

`equilibrium/time_slice/global_quantities/q_95` and
`summary/global_quantities/q_95/value`, identical text ("q at the 95% poloidal
flux surface"). One quantity.

### 45. `surface_area_of_flux_surface` — **ONE-QUANTITY** (2 bindings)

`equilibrium/time_slice/global_quantities/surface` and
`equilibrium/time_slice/profiles_1d/surface`, both "Surface area of the toroidal
flux surface" — the global one being the profile evaluated at the boundary.
Inside the settled `*_of_flux_surface` family; not relitigated.

### 46. `toroidal_beta` — **ONE-QUANTITY** (2 bindings)

`equilibrium/time_slice/global_quantities/beta_tor` and
`summary/global_quantities/beta_tor/value`, word-for-word the same definition.
One quantity.

### 47. `toroidal_flux_coordinate` — **ONE-QUANTITY** (2 bindings)

`core_profiles/profiles_1d/grid/rho_tor` and
`equilibrium/time_slice/profiles_1d/rho_tor`. Both are
`sqrt(Φ/(π B₀))` referred to the same `vacuum_toroidal_field/b0`, which group 17
confirms is one shared value across the two IDSs. One quantity.

### 48. `toroidal_magnetic_field_at_magnetic_axis` — **ONE-QUANTITY** (2 bindings)

`equilibrium/time_slice/global_quantities/magnetic_axis/b_field_phi` ("Total
toroidal magnetic field at the magnetic axis") and
`summary/local/magnetic_axis/b_field_tor/value`, whose parent container is the
magnetic axis. Both are the **total** toroidal field at that point, distinct
from the vacuum field of group 17 — and the catalog keeps them as separate
identities, which is the distinction being drawn correctly.

### 49. `upper_triangularity_of_plasma_boundary` — **ONE-QUANTITY** (2 bindings)

The upper partner of group 31, identical text on both bindings. One quantity.

### 50. `vertical_coordinate_of_geometric_axis` — **ONE-QUANTITY** (2 bindings)

`equilibrium/time_slice/boundary/geometric_axis/z` and
`summary/boundary/geometric_axis_z/value`, the summary parent supplying the
`(Zmax + Zmin)/2` definition. One quantity, and the Z partner of group 41.

### 51. `vertical_coordinate_of_strike_point` — **MUST-SPLIT**, with a description note (2 bindings)

`summary/boundary/strike_point_inner_z/value` (parent: "Z position of the
**inner** strike point") and `summary/boundary/strike_point_outer_z/value`
(parent: "Z position of the **outer** strike point") — the Z axis of group 42,
and the same defect.

| binding | spelling |
| --- | --- |
| `summary/boundary/strike_point_inner_z/value` | **needs** `vertical_coordinate_of_inner_strike_point` |
| `summary/boundary/strike_point_outer_z/value` | **needs** `vertical_coordinate_of_outer_strike_point` |

**Neither binding keeps the existing spelling**, for the reason given in group
42.

> **Narrower-description note, and it is worse than narrow.** The description
> reads "Signed vertical (Z) coordinate of the **inner** divertor strike point,
> where the **inner** separatrix leg intersects the divertor target". It names
> one locus while the identity is bound to two, so half its bindings carry a
> description that is not merely incomplete but **affirmatively false about
> them**: a reader consulting the catalog entry for
> `summary/boundary/strike_point_outer_z/value` is told it is the inner leg.
> This is the only group of the 53 where the description asserts the wrong
> member rather than an incomplete set.

**Confirms the prior reading.** `west-name-audit.md`'s closing finding 3 reports
this group and proposes the same pair. Judged independently here, confirmed;
counted once in the union. The description defect is additional to that finding.

### 52. `volume_averaged_electron_density` — **ONE-QUANTITY** (2 bindings)

`interferometer/n_e_volume_average` ("Volume average plasma density estimated
from the line densities measured by the various channels") and
`summary/volume_average/n_e/value`. Same averaging operator over the same
volume; the interferometer text describes how the average is obtained, which is
provenance. One quantity — and the second precedent group 25 is judged against.

### 53. `wavelength_of_wave_beam` — **ONE-QUANTITY** (2 bindings)

`interferometer/channel/wavelength/value` and `polarimeter/channel/wavelength`
("Wavelength used for polarimetry"). The vacuum wavelength of a probing beam is
one quantity; which diagnostic launches the beam is the locus. Two instruments
on WEST use different wavelengths, and that is exactly the case a shared
identity is meant to cover — one quantity, many instruments.

## Container-level resolutions

Three data-dictionary containers have their coordinate axes named by more than
one locus. A container is one point in space: all of its axes must name the same
object, so these are resolved at container level rather than axis by axis.

![Three coordinate containers whose sibling axes carry different loci](/imas-codex/figures/sn-west-catalog-release/coordinate-container-sibling-loci.svg)

**`camera_x_rays/aperture/centre`** — φ reads `toroidal_coordinate_of_aperture`
while R and Z read `radial_/vertical_coordinate_of_measurement_position`. **φ is
the outlier and φ is the correct one**: the container is an aperture, so the
container as a whole should carry the aperture locus and the two axes that move
are R and Z. R is group 14's second binding; Z
(`vertical_coordinate_of_measurement_position`) is bound to this one path only
and so forms no group of its own, which is why the container needs naming here
rather than being visible in the 53.

> **Confirms the prior reading, and resolves it.** `west-name-audit.md`'s
> closing finding 2 reports this container and leaves the question open —
> "whichever spelling wins". Judged here from the container's own evidence, the
> aperture spelling wins, because the data dictionary calls the container an
> aperture and nothing in it is a measurement position. The prior finding is
> confirmed and narrowed, not contradicted; the union counts it once, with the
> resolution supplied by this record.

**`magnetics/b_field_phi_probe/position`** — R and φ read `*_of_measurement_position`,
Z reads `vertical_coordinate_of_toroidal_magnetic_field_probe`. **Z is the
outlier and Z is the correct one**; R and φ move. (Groups 14 and 16.)

**`magnetics/b_field_pol_probe/position`** — R and Z read
`*_of_poloidal_magnetic_field_probe`, φ reads
`toroidal_angle_of_measurement_position`. **φ is the outlier and φ is the wrong
one**; only φ moves. (Group 16.)

In all three containers the correct spelling is the minority one. The count is
therefore not the instrument: what decides each container is what the data
dictionary says the container **is**, and in every case it says a piece of
instrument hardware rather than a place in the plasma.

## Result

| | count |
| --- | --- |
| shared identities in the WEST batch cohort | 53 |
| **groups judged** | **53** |
| ONE-QUANTITY | 43 |
| MUST-SPLIT | 10 |
| ONE-QUANTITY + MUST-SPLIT | **53** |
| | |
| bindings involved | 164 |
| bindings under a ONE-QUANTITY verdict | 140 |
| bindings under a MUST-SPLIT verdict | 24 |
| bindings accounted for | **164** |

**10 of 53 shared identities — 18.9 % — spell more than one physical quantity**,
covering 24 of the 164 shared bindings. Grouped by what the sharing conflates:

- **Instrument geometry named as a plasma measurement position** (2 groups, 6
  bindings): groups 14 and 16. An aperture centre and two sensor coils are named
  as though a measurement were made where they sit.
- **A topological critical point named as the magnetic axis** (1 group, 4
  bindings): group 4, settled by the batch manifest's own X-point migration map.
- **Two loci of one structure sharing one name** (2 groups, 4 bindings): the
  inner and outer strike points, on both axes — groups 42 and 51.
- **An averaging operator folded onto the field it averages** (1 group, 2
  bindings): group 25, against a precedent the catalog already sets twice.
- **Two different physical descriptors of one beam** (1 group, 2 bindings):
  group 27, ellipticity against polarization state.
- **Aggregation level conflated** (1 group, 2 bindings): group 28, one launcher
  against the machine total.
- **Launched against coupled power** (1 group, 2 bindings): group 34.
- **An instrument position against a plasma feature's position** (1 group, 2
  bindings): group 36, a channel's sight position against an emissivity peak.

Counting the two axes of the strike-point pair as one defect and the two axes
of the measurement-position collision as one defect, these ten groups are
**eight distinct naming defects**. Two of the eight — the measurement-position
collision and the strike-point pair — were already pointed at by the
every-fourth sample; the remaining **six are new to this record**, as is the
resolution of the aperture container the sample left open.

### Against the prior record

| prior closing finding | this record | outcome |
| --- | --- | --- |
| `radial_coordinate_of_measurement_position` across three loci | group 14 | **confirms**, same three loci, same two proposed spellings |
| `camera_x_rays/aperture/centre` φ against its R and Z siblings | container section | **confirms and resolves** — the aperture spelling wins |
| `vertical_coordinate_of_strike_point` across both legs | group 51 | **confirms**, plus a description that is false for the outer leg |

Nothing in the prior record is contradicted. Six defects are added that the
every-fourth sample could not reach: the contour-tree node named as the magnetic
axis (group 4), and the five quantity-level conflations of groups 25, 27, 28, 34
and 36. Group 16 is not counted among the six — it is the same defect as group
14 on a second axis — but it widens that defect by two bindings and by one
container, `magnetics/b_field_pol_probe/position`, which the sample never
names.

### Narrower failures, reported as notes on ONE-QUANTITY verdicts

- **Description names fewer loci than the identity is bound to** — 1 case:
  group 3, `toroidal_coordinate_of_line_of_sight`, whose description names "the
  first reference point" while 8 of its 14 bindings are second or third points.
  Confirms the prior record's sample row 46. A second instance, group 51, sits
  on a MUST-SPLIT verdict rather than a ONE-QUANTITY one and is the more serious
  form, because the description is false for the bindings it omits rather than
  merely silent about them.
- **Unit disagreement between `sn_unit` and `dd_unit`** — **0 cases** across all
  164 bindings, with the comparator shown above to find 3 in the cohort
  remainder, so the zero is a measurement and not a blind instrument.

