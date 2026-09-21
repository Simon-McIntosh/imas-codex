# WEST batch shared identities — one quantity, or several?

provisional: true — verdicts are appended as each group is judged; the closing
pass rewrites this line and adds the result table.

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

