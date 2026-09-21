# The WEST repair worklist under the no-ordinal convention

provisional: false — all 61 repairs carry a class and the result table is
closed.

**The project lead's convention: ordinal and index dimensions are not carried in
standard names.** Inner and outer, main and secondary, primary, first and
second, upper and lower — *where they enumerate members of an array rather than
name a physical distinction* — are indices in the data structure, not name
segments.

This file re-reads the 61 repairs in
[the deduplicated repair worklist](west-name-repair-worklist) under that
convention and classifies each one **STANDS**, **WITHDRAWN** or
**RECLASSIFIED**. It judges no name afresh against the data dictionary and adds
no defect the records do not contain: it applies one convention to verdicts
that already exist and reports what the convention changes.

**The worklist and the six records it aggregates are not edited.** A peer
session has adopted the worklist as an independent test fixture for a review
instrument it is building, and amending it would destroy that independence.
This file is the revision; it links to the worklist rather than replacing it.

## What the convention decides, and in which direction

Three consequences, and the third is the one that is easy to lose.

1. **A split whose two halves differ only by such a qualifier is invalid.** The
   shared identity binding both members was the convention working rather than
   an ambiguity, so the repair is **WITHDRAWN**.
2. **A rename that ADDS such a qualifier is invalid for the same reason**, and
   is **WITHDRAWN**.
3. **Where the generic spelling is correct and a SIBLING binding carries the
   ordinal, the defect is real but inverted.** The repair is not to add the
   qualifier to the generic name; it is to *strip* it from the sibling and bind
   that sibling to the generic identity — which frequently already exists and is
   accepted. Such a row is **RECLASSIFIED**, and it names the surviving generic
   identity and states, from a graph lookup rather than from assumption, whether
   that identity exists.

A fourth thing survives a withdrawal and sits on a different axis. Where the
ruling makes a shared identity correct, its **description** may still name one
ordinal member as though the identity meant only that one. That is a
description repair, never a rename, and every WITHDRAWN row is checked for it
below.

## How the 61 were scanned, and why the scan is trustworthy

The rejected and proposed spellings of all 61 repairs were parsed out of the
worklist's two tables mechanically — 51 rename rows from part one and 10 split
sections from part two, each with its per-binding resolutions — and every
segment of every proposed spelling was tested against the token set `inner`,
`outer`, `main`, `secondary`, `primary`, `first`, `second`, `third`, `upper`,
`lower`, `innermost`, `outermost`.

The scan reports **six proposals carrying an ordinal token**, and the six are
listed here in full because the classification of the other 55 rests on the
claim that they carry none:

| proposal | token | verdict on the token |
| --- | --- | --- |
| `radial_coordinate_of_inner_strike_point` | inner | an array member — the convention applies |
| `radial_coordinate_of_outer_strike_point` | outer | an array member — the convention applies |
| `vertical_coordinate_of_inner_strike_point` | inner | an array member — the convention applies |
| `vertical_coordinate_of_outer_strike_point` | outer | an array member — the convention applies |
| `radial_coordinate_of_primary_x_point` | primary | an array member — the convention applies |
| `radiative_temperature_at_innermost_ece_channel` | innermost | a physical criterion, not an index — see its row |
| `radial_separation_of_inner_and_outer_separatrices_at_outboard_midplane` | inner, outer | names both, selects neither — see its row |
| `upper_bound_photon_energy` | upper | a band boundary, not an index — see its row |
| `total_launched_power_of_lower_hybrid_antennas` | lower | the wave mode, not an ordinal |
| `launched_power_of_lower_hybrid_antenna` | lower | the wave mode, not an ordinal |

The scan is not a blind instrument: it returns hits, it returns them on the
rows that were expected to carry them, and it separately finds the token set in
**one accepted cohort name** that no repair row proposes —
`vertical_coordinate_of_primary_x_point` — which is the sibling that makes the
RECLASSIFIED row below possible. An instrument that found nothing anywhere
would be reporting its own failure.

## Verdicts

### WITHDRAWN — the two strike-point splits

These are split rows 9 and 10 of the worklist's part two. Their two halves
differ by nothing but `inner` against `outer`, and `summary/boundary` holds
`strike_point_inner_r` and `strike_point_outer_r` as two members of one
conceptual pair. Under the convention the shared identity was correct.

#### 1. `radial_coordinate_of_strike_point` — **WITHDRAWN**

- worklist row: part two, shared group 42; rejected by audit row 72
  (`summary/boundary/strike_point_outer_r/value`) and by the
  spectrometer-visible→wall record row 212
  (`summary/boundary/strike_point_inner_r/value`)
- proposed split: `radial_coordinate_of_inner_strike_point` /
  `radial_coordinate_of_outer_strike_point`
- why withdrawn: the two proposed halves are identical but for `inner` and
  `outer`, which is the signature the convention names. Both records argue the
  split from the physics of the two legs — different heat flux, different
  geometry, different control significance — and that argument is not wrong
  about the divertor; it is the wrong conclusion about the *name*. The legs are
  two members of the strike-point array, and a catalog that carries the leg in
  the name carries the index in the name.
- **description check: clean.** The identity's description reads "Major-radius
  location of **an individual** magnetic strike point where **a** separatrix leg
  intersects **a** divertor target, expressed in the right-handed cylindrical
  (R, φ, Z) frame." It covers both legs without selecting either. This is the
  description the convention requires, and it is the model against which its own
  vertical twin fails.

#### 2. `vertical_coordinate_of_strike_point` — **WITHDRAWN**, and it carries the surviving defect

- worklist row: part two, shared group 51; rejected by the
  spectrometer-visible→wall record rows 213 and 214, both halves inside one
  range
- proposed split: `vertical_coordinate_of_inner_strike_point` /
  `vertical_coordinate_of_outer_strike_point`
- why withdrawn: the Z axis of the same pair, withdrawn for the same reason.
- **description check: FAILS, and this is the model for the class.** The
  identity's description reads:

  > "Signed vertical (Z) coordinate of the **inner** divertor strike point,
  > where the **inner** separatrix leg intersects the divertor target in the
  > right-handed cylindrical (R, φ, Z) frame."

  The identity is bound to `strike_point_inner_z` and `strike_point_outer_z`.
  So for the outer binding the description is not merely incomplete — it is
  **affirmatively false**: a reader who resolves the published entry for the
  outer strike point is told it is the inner leg. The shared-identities record
  calls this "the only group of the 53 where the description asserts the wrong
  member rather than an incomplete set."
- **the remedy is a description repair, not a rename.** The wall record's row
  214 concludes the opposite — "The defect is in the name, not the description:
  one identity cannot carry a description that is true of only one of the two
  paths it is bound to, so the split is the only remedy." Under the convention
  that inference runs backwards. The identity *may* cover both paths; what it
  may not do is describe one of them. Its own radial twin, one row above, shows
  the repair already written: replace the two occurrences of "inner" with the
  indefinite article, and the description covers both legs without selecting
  one, exactly as `radial_coordinate_of_strike_point` does today.

### RECLASSIFIED — the one row where the convention inverts the repair

#### 3. `radial_coordinate_of_x_point` — **RECLASSIFIED**

- worklist row: part one, rejected by audit row 73 on
  `summary/boundary/x_point_main/r`
- the worklist's proposal: `radial_coordinate_of_primary_x_point`
- **the proposal is invalid**, because it adds `primary` to a name that does
  not carry it.
- **but the defect is real, and it points the other way.** The audit's own
  reasoning is what makes this visible: *"The two coordinates of one DD
  container disagree with each other: `summary/boundary/x_point_main/r` is
  `radial_coordinate_of_x_point` while `summary/boundary/x_point_main/z` is
  `vertical_coordinate_of_primary_x_point`. One container, one locus, two
  spellings — so at most one is right."* That much survives the convention
  intact. What the convention changes is **which** of the two is right. The
  audit resolved the disagreement towards the DD's own `_main` qualifier; under
  the convention the DD's qualifier is an index into the X-point array and does
  not belong in either name, so the generic spelling wins.
- **revised repair: rename the SIBLING, leave this row's identity alone.**
  `summary/boundary/x_point_main/z` moves off
  `vertical_coordinate_of_primary_x_point` and onto the generic
  `vertical_coordinate_of_x_point`. `radial_coordinate_of_x_point` is correct
  as it stands and is not renamed.
- **surviving generic identity: `vertical_coordinate_of_x_point` — it already
  exists.** Verified rather than assumed, by one indexed single-row read:
  `MATCH (sn:StandardName {id: 'vertical_coordinate_of_x_point'}) RETURN sn.id,
  sn.name_stage` returns `{'id': 'vertical_coordinate_of_x_point', 'stage':
  'accepted'}`. So the repair binds a path to an identity the catalog already
  publishes and mints nothing. Both of the other two names in this row's
  argument were read the same way and are also present and accepted:
  `radial_coordinate_of_x_point` and `vertical_coordinate_of_primary_x_point`.
- the audit's second argument — that an unqualified `x_point` is wrong in a
  double-null-capable machine because secondary X-points exist — is exactly the
  reasoning the convention rules out. A secondary X-point is another member of
  the same array; a catalog that spells the member into the name has published
  the index.

### STANDS — the remaining 58

Every row below survives the convention untouched: its proposed spelling
carries no ordinal or index segment, so the convention has nothing to say about
it and the record's own verdict is undisturbed. The rejected and proposed
spellings are the worklist's, unaltered.

`target exists` is one indexed single-row read per proposed spelling —
`MATCH (sn:StandardName {id: $name}) RETURN sn.id, sn.name_stage` — and nothing
else. No scan, no traversal, no unbounded query was issued. Both controls pass:
a name known to exist returns present (`electron_temperature`, `accepted`), and
a fabricated name returns an empty result, so `absent` is a measurement rather
than a silent failure.

#### The 50 renames that stand

| # | rejected | proposed | target exists |
| --- | --- | --- | --- |
| 4 | `accumulated_total_gas_count` | `accumulated_total_neutral_particle_count_due_to_gas_injection` | absent |
| 5 | `area_of_diagnostic_aperture` | `area_of_diagnostic_detector` | absent |
| 6 | `area_of_poloidal_magnetic_field_probe` | `turn_area_of_poloidal_magnetic_field_probe` | absent |
| 7 | `area_of_toroidal_magnetic_field_probe` | `turn_area_of_toroidal_magnetic_field_probe` | absent |
| 8 | `capacitance_of_ion_cyclotron_heating_antenna` | `capacitance_of_impedance_matching_element` | absent |
| 9 | `coolant_transit_time_of_plant_component_port` | `coolant_transit_time_of_calorimetry_component` | absent |
| 10 | `effective_turn_count_of_passive_loop` | `effective_turn_count_of_passive_loop_element` | absent |
| 11 | `energy_confinement_enhancement_factor` | `ipb98y2_confinement_enhancement_factor` | present (`superseded`) |
| 12 | `faraday_angle` | `faraday_rotation_angle` | absent |
| 13 | `gap_at_outboard_midplane` | `radial_separation_of_inner_and_outer_separatrices_at_outboard_midplane` | absent |
| 14 | `hard_xray_brightness` | `hard_xray_photon_radiance` | absent |
| 15 | `hard_xray_emissivity` | `hard_xray_photon_emissivity` | absent |
| 16 | `height_of_poloidal_field_coil` | `height_of_conductor_cross_section` | absent |
| 17 | `hot_neutral_temperature_at_plasma_boundary` | `hot_neutral_temperature` | present (`superseded`) |
| 18 | `line_integrated_electron_number_density` | `line_integrated_electron_density` | present (`superseded`) |
| 19 | `mhd_energy` | `total_plasma_stored_energy` | absent |
| 20 | `minimum_safety_factor` | `minimum_absolute_safety_factor` | absent |
| 21 | `opacity_at_ece_channel_emission_position` | `optical_depth_at_ece_channel_emission_position` | absent |
| 22 | `poloidal_angle_of_flux_surface` | `poloidal_orientation_angle_of_poloidal_magnetic_field_probe` | absent |
| 23 | `power_of_soft_xray_detector` | `incident_power_of_soft_xray_detector` | absent |
| 24 | `pressure_of_ion_cyclotron_heating_antenna` | `pressure_amplitude_of_ion_cyclotron_heating_antenna` | absent |
| 25 | `pulse_duration` | `confined_plasma_duration` | absent |
| 26 | `radial_coordinate_at_inboard_midplane` | `radial_coordinate_of_flux_surface_at_inboard_midplane` | absent |
| 27 | `radial_coordinate_at_outboard_midplane` | `radial_coordinate_of_flux_surface_at_outboard_midplane` | absent |
| 28 | `radial_derivative_of_poloidal_magnetic_flux` | `derivative_of_poloidal_magnetic_flux_with_respect_to_toroidal_flux_coordinate` | absent |
| 29 | `radial_outline_of_wall` | `radial_outline_of_plasma_facing_component` | present (`superseded`) |
| 30 | `radiated_power_over_core_region` | `radiated_power_inside_plasma_boundary` | absent |
| 31 | `radiative_temperature_at_magnetic_axis` | `radiative_temperature_at_innermost_ece_channel` | absent |
| 32 | `ratio_of_neutral_density_of_isotope_to_difference_of_total_neutral_density_and_neutral_density_of_isotope` | `ratio_of_neutral_density_of_isotope_to_neutral_density_of_other_isotopes` | absent |
| 33 | `spectral_calibration_factor_at_line_of_sight` | `phase_to_line_integrated_electron_density_conversion_factor` | absent |
| 34 | `spectral_rate_of_spectrometer_channel` | `photoelectron_rate_of_spectrometer_channel` | absent |
| 35 | `spectral_wavelength_of_optical_element` | `spectral_wavelength_of_spectrometer_channel` | absent |
| 36 | `surface_temperature` | `apparent_surface_temperature` | absent |
| 37 | `temperature_of_soft_xray_detector` | `temperature_of_x_ray_detector` | absent |
| 38 | `thermal_electron_pressure_at_post_sawtooth_crash` | `thermal_electron_pressure` | present (`superseded`) |
| 39 | `toroidal_angle_of_antenna_strap` | `toroidal_angle_of_antenna_strap_outline` | absent |
| 40 | `toroidal_angle_of_poloidal_magnetic_field_probe` | `toroidal_orientation_angle_of_poloidal_magnetic_field_probe` | absent |
| 41 | `toroidal_angular_width_of_limiter_tile` | `toroidal_angular_centre_and_full_width_of_limiter_tile` | absent |
| 42 | `toroidal_coordinate_at_detector_pixel` | `toroidal_coordinate_of_detector_pixel` | present (`superseded`) |
| 43 | `toroidal_vacuum_magnetic_field` | `toroidal_vacuum_magnetic_field_at_reference_major_radius` | absent |
| 44 | `total_power_due_to_ion_cyclotron_heating` | `total_coupled_power_due_to_ion_cyclotron_heating` | absent |
| 45 | `upper_photon_energy` | `upper_bound_photon_energy` | absent |
| 46 | `vertical_coordinate_of_ece_channel` | `vertical_coordinate_of_measurement_position` | present (`accepted`) |
| 47 | `vertical_coordinate_of_measurement_position` | `vertical_coordinate_of_aperture` | present (`accepted`) |
| 48 | `voltage_of_mass_spectrometer_channel` | `photomultiplier_voltage_of_mass_spectrometer_channel` | absent |
| 49 | `volume_of_flux_surface` | `volume_of_plasma_boundary` | present (`accepted`) |
| 50 | `wave_current_amplitude_of_antenna_strap` | `wave_current_amplitude_of_ion_cyclotron_heating_antenna` | absent |
| 51 | `wave_phase_of_ion_cyclotron_heating_antenna` | `voltage_phase_of_ion_cyclotron_heating_antenna` | absent |
| 52 | `wave_phase_of_wave_beam` | `fringe_jump_corrected_phase_of_interferometer_beam` | absent |
| 53 | `width_of_poloidal_field_coil` | `width_of_conductor_cross_section` | absent |

#### The 8 splits that stand

Each row keeps the worklist's per-binding resolution; only the spellings the
split must mint are looked up, since a `keeps` spelling is the identity that
is already there.

| # | identity | bindings | spellings the split must mint, and whether each exists |
| --- | --- | --- | --- |
| 54 | `radial_coordinate_of_magnetic_axis` | 4 | `radial_coordinate_of_flux_contour_critical_point` — absent |
| 55 | `radial_coordinate_of_measurement_position` | 3 | `radial_coordinate_of_aperture` — present (`superseded`); `radial_coordinate_of_toroidal_magnetic_field_probe` — present (`superseded`) |
| 56 | `toroidal_angle_of_measurement_position` | 3 | `toroidal_angle_of_toroidal_magnetic_field_probe` — present (`accepted`); `toroidal_angle_of_poloidal_magnetic_field_probe` — present (`accepted`) |
| 57 | `effective_charge` | 2 | `volume_averaged_effective_charge` — present (`accepted`) |
| 58 | `initial_polarization_ellipticity_of_polarimeter_beam` | 2 | `initial_polarization_of_polarimeter_beam` — absent |
| 59 | `launched_power_of_lower_hybrid_antenna` | 2 | `total_launched_power_of_lower_hybrid_antennas` — absent |
| 60 | `net_power_due_to_ion_cyclotron_heating` | 2 | `coupled_power_of_ion_cyclotron_antenna` — absent |
| 61 | `normalized_toroidal_flux_coordinate_at_measurement_position` | 2 | `normalized_toroidal_flux_coordinate_of_emissivity_peak` — absent |

### The four STANDS rows whose proposal contains an ordinal word anyway

The token scan flagged four proposals that stand despite carrying one of the
words. Each is recorded so the judgement is reviewable rather than implicit —
if the lead reads any of these the other way, the row moves to WITHDRAWN and
the counts shift by one.

- **`upper_photon_energy` → `upper_bound_photon_energy`** (row 45 above).
  `upper` here is the **upper bound of an energy band**, a boundary of a
  continuous interval, not a position in an array. Its partner
  `lower_bound_photon_energy` names the other end of the same interval, and the
  two together describe one band rather than enumerating two members of
  anything. Stands.
- **`gap_at_outboard_midplane` →
  `radial_separation_of_inner_and_outer_separatrices_at_outboard_midplane`**
  (row 13). This names **both** members of a pair simultaneously, as the two
  endpoints of a separation, and selects neither. A name that carried one of
  them would fall to the convention; a name for the distance between them
  cannot avoid naming both and is not an index. `outboard` is the low-field
  side of the torus, a geometric fact about the machine rather than an array
  position. Stands.
- **`radiative_temperature_at_magnetic_axis` →
  `radiative_temperature_at_innermost_ece_channel`** (row 31). **This is the
  closest call of the 61.** `innermost` is superlative over the channel array,
  which reads like an index — but the data-dictionary text defines the quantity
  by a *physical* criterion, "Radiation temperature from the closest channel to
  the magnetic axis", not by a slot. Which channel satisfies it changes with the
  equilibrium, so the name selects a measurement geometry rather than an array
  member. Stands, and it is flagged here because it is the one row where a
  reasonable reader could rule the other way.
- **`launched_power_of_lower_hybrid_antenna` and
  `total_launched_power_of_lower_hybrid_antennas`** (split row 59). `lower` is
  part of **lower hybrid**, the name of a wave resonance. Not an ordinal in any
  reading. Stands.

## What the lookups found beyond existence

**15 of the 61 proposed target spellings already exist; 46 are absent and would
mint a new identity.** Minting is the point of most of these rows — the current
spelling is wrong and the catalog needs a right one — so an absent target is
not by itself a finding. Two patterns in the 15 are.

**Eight of the 15 exist at `name_stage = 'superseded'`, not `accepted`.** These
renames do not mint a fresh identity; they move a binding onto a spelling the
catalog has already retired:

| proposed target | stage |
| --- | --- |
| `ipb98y2_confinement_enhancement_factor` | superseded |
| `hot_neutral_temperature` | superseded |
| `line_integrated_electron_density` | superseded |
| `radial_outline_of_plasma_facing_component` | superseded |
| `thermal_electron_pressure` | superseded |
| `toroidal_coordinate_of_detector_pixel` | superseded |
| `radial_coordinate_of_aperture` | superseded |
| `radial_coordinate_of_toroidal_magnetic_field_probe` | superseded |

Whether a retired identity can be revived, or whether these must mint fresh
spellings, is a lifecycle question this node does not decide — but it is not
the "mint a new name" operation the worklist's own arithmetic assumed, and it
touches eight of the 61. It is reported rather than resolved.

**One split target is itself a rejected spelling, which makes a second
order-dependent pair.** Split row 56, `toroidal_angle_of_measurement_position`,
resolves `magnetics/b_field_pol_probe/position/phi` to
`toroidal_angle_of_poloidal_magnetic_field_probe` — and the lookup returns that
name `present`, `accepted`, because it is the current identity of a *different*
binding, `magnetics/b_field_pol_probe/toroidal_angle`, which rename row 40
rejects in favour of `toroidal_orientation_angle_of_poloidal_magnetic_field_probe`.
So the split feeds a binding into an identity that a rename is simultaneously
emptying. Both repairs are correct as written and the order matters: the rename
must run first, or the split's product lands in an identity still carrying the
wrong one. The worklist records one such pair already (the
`vertical_coordinate_of_measurement_position` chain); this is a second, and it
is visible only because the target was looked up rather than assumed new.

## The description defect that survives a withdrawal

**One of the 61 carries it: `vertical_coordinate_of_strike_point`.** Both
WITHDRAWN rows were checked and only the vertical one fails; its radial twin
carries a description that covers both legs without selecting either, and is
the model for the repair. The offending sentence and the remedy are quoted in
that row above.

The shape is not unique to the repair rows, though it is unique among them. The
shared-identities record reports a second instance on a ONE-QUANTITY verdict —
`toroidal_coordinate_of_line_of_sight`, whose description names "the first
reference point" while 8 of its 14 bindings are second or third points. That
identity is not among the 61 because no record rejects its *name*, and under
this convention none should: first, second and third points are members of a
line-of-sight array, so the shared identity is correct and only its description
narrows it. It is named here because a sweep for this defect should not stop at
the repair list — the convention makes shared identities the normal case, and
the descriptions written while they looked like ambiguities are where the
damage now sits.

## Result

| | count |
| --- | --- |
| repairs in the worklist | **61** |
| **STANDS** | **58** |
| **WITHDRAWN** | **2** |
| **RECLASSIFIED** | **1** |
| the three classes, summed | **61** |
| **revised repair total** | **59** |
| of which splits | **8** (10 in the worklist, less the two strike-point pairs) |
| of which renames | **51** (50 that stand, plus the reclassified row's inverted rename) |
| repairs a reader cannot compensate for | **8**, down from 10 |
| proposals carrying an ordinal token | 10 occurrences over 8 spellings; 5 fall to the convention, 5 stand |
| target spellings looked up | **61**, one indexed single-row read each |
| targets already present | **15** — 7 `accepted`, 8 `superseded` |
| targets absent, minting a new identity | **46** |
| of the 61, carrying the ordinal-member description defect | **1** |
| order-dependent repair pairs | **2** — the worklist's one, plus one the lookups surfaced |

**The convention removes two repairs and inverts a third; it does not soften
the worklist.** Both withdrawals are in the class the worklist called the one
that blocks publication, so the count of reader-uncompensable defects falls
from 10 to 8 — but the withdrawal of `vertical_coordinate_of_strike_point`
leaves a defect standing rather than clearing one, and moves it from a rename
that a release can apply mechanically to a description repair that someone has
to write. The RECLASSIFIED row is the one that would have been silently lost by
treating the convention as a filter: read as a filter it deletes a row, and
read as a rule it relocates the repair onto a sibling and lands it on an
identity the catalog already publishes.

All 61 repairs carry a class, the three classes sum to 61, and every target
spelling a STANDS or RECLASSIFIED row proposes has been looked up; the
`provisional` line at the head of this file has been rewritten to `false`.
