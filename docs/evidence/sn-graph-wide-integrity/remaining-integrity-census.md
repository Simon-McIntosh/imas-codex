# Remaining Standard Name integrity census

## Outcome

This is a read-only census of the live `codex` graph at source commit
`2a48233cd7ad155ca847d7673d903bca89fce75a`, using plan version 173 as the
semantic authority. The graph read ran from
`2026-08-19T05:35:52.071718+00:00` to
`2026-08-19T05:35:58.638450+00:00`.

| Integrity class | Measured total | Partition check | Headline result |
|---|---:|---:|---|
| Dual-bound semantic sources | 226 | 226 | 216 touch a catalog edit; 9 are exact lineage-fold candidates; 1 is a stale detach candidate |
| Live unsourced Standard Names | 85 | 85 | 69 accepted, 4 reviewed, 4 drafted, 8 pending |
| Semantic-source invariant violations | 371 | 371 | 225 multiple-live-target, 144 scalar mismatch, 2 projection mismatch |
| Owner/geometry authority cohort | 49 | 49 | The old measurement-position target remains live on all 49 rows; 10 also have a competing owner/geometry target |

Every partition sums exactly to its independently measured total. The complete,
row-level machine receipt is
[`remaining-integrity-census.json`](remaining-integrity-census.json), SHA-256
`6d05b06021f3653bebc0b50846b10b966f0445c2cff0627fe51a1ca923e4d632`.

## Read-only proof

The census sampled both protected node counts immediately before and after all
queries. Their decimal byte strings and integer values were identical:

| Node label | Before | After | Delta | Byte-identical |
|---|---:|---:|---:|---|
| `StandardNameChange` | 7,451 | 7,451 | 0 | yes |
| `LLMCost` | 27,467 | 27,467 | 0 | yes |

The run made no provider call and executed no graph write. The raw graph reply
is retained at
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T053057370828-remaining-integrity-census/census-raw.json`,
SHA-256
`cd12a3a01779293d4ab2e65069940ffb78b661d484c84b62fbfab9957d94252e`.

## Dual-bound semantic sources

A dual-bound source is a composed or attached semantic source with more than
one live `PRODUCED` target; `superseded` and `exhausted` targets do not count as
live. Each source is assigned once using this precedence: stale source with no
upstream entity; exact two-target directed `REFINED_FROM` lineage; catalog-edit
involvement; scalar-selected deduplication; otherwise genuinely ambiguous.

| Disposition | Count | Fraction |
|---|---:|---:|
| Catalog-edit involved | 216 | 95.6% |
| Exact ancestor-fold candidate | 9 | 4.0% |
| Stale detach candidate | 1 | 0.4% |
| Scalar-selected deduplication | 0 | 0.0% |
| Genuinely ambiguous | 0 | 0.0% |
| **Sum** | **226** | **100.0%** |

Representative bindings show why this is not one homogeneous repair class:

- `dd:amns_data/a` binds both `atomic_mass` (derived, accepted) and
  `neutral_species_atomic_mass` (catalog edit, accepted, name score 0.875); its
  scalar projection selects `atomic_mass`. This is one of 55 occurrences of
  that target pair.
- `dd:edge_profiles/ggd/neutral/velocity/phi` binds
  `toroidal_neutral_momentum_convection_velocity` (catalog edit, accepted,
  score 0.88125) and its accepted ancestor `toroidal_neutral_velocity` (score
  0.98125); the scalar selects the ancestor. This is an exact lineage-fold
  candidate, not a hand-acceptance decision.
- `dd:equilibrium/time_slice/boundary_secondary_separatrix/outline/z` is stale,
  has no current upstream DD entity, and still binds
  `vertical_coordinate_of_geometric_axis` plus `vertical_outline`. It is the
  sole stale-detach candidate.

The largest recurring pair families are:

| Live target pair | Sources |
|---|---:|
| `atomic_mass` + `neutral_species_atomic_mass` | 55 |
| `mode_number` + `toroidal_mhd_mode_number` | 9 |
| `energy_density` + `ion_kinetic_energy_density` | 5 |
| `toroidal_angle_of_coil_conductor_element` + `toroidal_angle_of_measurement_position` | 5 |
| `mass_density` + `total_plasma_mass_density` | 4 |
| `toroidal_angle_of_measurement_position` + `toroidal_coordinate_of_aperture` | 4 |

The 226 dual-bound count is one larger than the 225 `multiple_live_targets`
invariant count because the stale source is deliberately outside the production
invariant's composed/attached scope.

## Live unsourced Standard Names

A live unsourced orphan is a Standard Name whose `name_stage` is not
`superseded` or `exhausted` and which has no incoming `PRODUCED` edge from any
`StandardNameSource`. This definition deliberately excludes terminal history;
it does not claim that every unsourced structural or catalog name is invalid.

| Name stage | Count |
|---|---:|
| Accepted | 69 |
| Reviewed | 4 |
| Drafted | 4 |
| Pending | 8 |
| **Sum** | **85** |

By origin, the same 85 names partition as 43 derived, 20 pipeline, 3
catalog-edit, and 19 with no recorded origin. Representative identities and
descriptions are:

- Accepted: `capacitance_of_ion_cyclotron_heating_antenna` (catalog edit,
  score 0.96875, unit F) is the non-negative charge-to-voltage proportionality
  of a specified antenna matching element; `co_passing_fast_current_density`
  is a derived structural name for signed conventional charge flux carried by
  co-passing fast particles; and
  `co_passing_fast_ion_charge_state_torque_density_due_to_collisions` is a
  pipeline name with score 0.95.
- Reviewed: `minimum_of_safety_factor` has score 0.725;
  `neutral_state_power_density` is a catalog edit with score 0.83125; and
  `poloidal_neutral_internal_state_momentum_convected_velocity` has score
  0.56875.
- Drafted: all four are quarantined, including
  `line_integrated_electron_density`,
  `magnetic_field_at_pedestal_top_low_field_side_magnitude`, and
  `poloidal_straight_field_line_angle`.
- Pending: examples include the derived
  `coefficient_of_spectrometer_channel`, the quarantined
  `cross_section_of_flux_surface`, and the derived
  `energy_due_to_ohmic_dissipation`.

## Semantic-source invariant violations

This class uses the production invariant implementation and its precedence:
no live target; multiple live targets; scalar mismatch; projection mismatch.
The source scope is composed or attached. For DD sources, `mapped_ids` is the
projection; for signal sources, `standard_name_ids` is the projection.

| Violation class | Count | Representative source |
|---|---:|---|
| Multiple live targets | 225 | `dd:amns_data/a` → `atomic_mass`, `neutral_species_atomic_mass` |
| Scalar mismatch | 144 | `dd:balance_of_plant/gain_plant` has live/mapped `net_efficiency_of_plant_system` but a null scalar |
| Projection mismatch | 2 | Both rows have scalar and live target `electron_source_rate`, but an empty `mapped_ids` projection |
| No live target | 0 | — |
| Unclassified | 0 | — |
| **Sum** | **371** | |

The two exact projection-mismatch sources are:

- `dd:core_sources/source/profiles_1d/electrons/particles_decomposed/explicit_part`
- `dd:edge_sources/source/ggd/electrons/particles/values`

The row-level receipt retains all 371 violations, including scalar values,
edge targets, and projection values, so repairs can be generated without
reinterpreting this summary.

## Outstanding owner/geometry mispairing rows

The authority cohort contains 49 DD paths previously identified as owner or
geometry spellings rather than generic measurement positions. The current read
found the old live target `toroidal_angle_of_measurement_position` on every
row. Of the 49, 39 have only the old target and 10 also carry a competing
owner/geometry target. The scalar still selects the old target on 45 rows.
Source status is 43 attached, 4 composed, and 2 stale.

### Old target live only (39)

- `dd:b_field_non_axisymmetric/time_slice/field_map/grid/phi`
- `dd:bolometer/camera/channel/detector/centre/phi`
- `dd:camera_visible/channel/aperture/centre/phi`
- `dd:camera_x_rays/aperture/centre/phi`
- `dd:camera_x_rays/camera/centre/phi`
- `dd:ece/channel/beam_tracing/beam/position/phi`
- `dd:ece/polarizer/centre/phi`
- `dd:hard_x_rays/channel/aperture/centre/phi`
- `dd:hard_x_rays/channel/filter_window/centre/phi`
- `dd:interferometer/channel/n_e/positions/phi`
- `dd:langmuir_probes/reciprocating/plunge/position_average/phi`
- `dd:mse/channel/active_spatial_resolution/centre/phi`
- `dd:mse/channel/aperture/centre/phi`
- `dd:mse/channel/detector/centre/phi`
- `dd:neutron_diagnostic/detector/geometry/centre/phi`
- `dd:neutron_diagnostic/detectors/aperture/centre/phi` (stale)
- `dd:neutron_diagnostic/detectors/detector/centre/phi` (stale)
- `dd:pellets/time_slice/pellet/path_geometry/first_point/phi`
- `dd:reflectometer_fluctuation/channel/antenna_detection_static/centre/phi`
- `dd:reflectometer_fluctuation/channel/antenna_emission_static/centre/phi`
- `dd:reflectometer_profile/channel/antenna_detection/centre/phi`
- `dd:reflectometer_profile/channel/antenna_emission/centre/phi`
- `dd:soft_x_rays/channel/detector/centre/phi`
- `dd:soft_x_rays/channel/filter_window/centre/phi`
- `dd:spectrometer_visible/channel/active_spatial_resolution/centre/phi`
- `dd:spectrometer_visible/channel/detector/centre/phi`
- `dd:spectrometer_visible/channel/optical_element/geometry/centre/phi`
- `dd:spectrometer_visible/channel/polarizer/centre/phi`
- `dd:spectrometer_x_ray_crystal/channel/aperture/centre/phi`
- `dd:spectrometer_x_ray_crystal/channel/camera/centre/phi`
- `dd:spectrometer_x_ray_crystal/channel/crystal/centre/phi`
- `dd:spectrometer_x_ray_crystal/channel/filter_window/centre/phi`
- `dd:spectrometer_x_ray_crystal/channel/reflector/centre/phi`
- `dd:spi/injector/fragment/position/phi`
- `dd:spi/injector/pellet/position/phi`
- `dd:spi/injector/shatter_cone/origin/phi`
- `dd:spi/injector/shattering_position/phi`
- `dd:tf/coil/conductor/elements/end_points/phi`
- `dd:thomson_scattering/laser/end_point/phi`

### Old target live with a competing target (10)

| Source | Competing owner/geometry target | Current scalar |
|---|---|---|
| `dd:bolometer/camera/channel/aperture/centre/phi` | `toroidal_coordinate_of_aperture` | competing target |
| `dd:camera_visible/channel/optical_element/geometry/centre/phi` | `toroidal_coordinate_of_optical_element` | competing target |
| `dd:coils_non_axisymmetric/coil/conductor/elements/end_points/phi` | `toroidal_angle_of_coil_conductor_element` | old target |
| `dd:coils_non_axisymmetric/coil/conductor/elements/start_points/phi` | `toroidal_angle_of_coil_conductor_element` | old target |
| `dd:neutron_diagnostic/detector/aperture/centre/phi` | `toroidal_coordinate_of_aperture` | old target |
| `dd:spectrometer_uv/channel/aperture/centre/phi` | `toroidal_coordinate_of_aperture` | competing target |
| `dd:spectrometer_uv/channel/detector/centre/phi` | `toroidal_coordinate_of_aperture` | competing target |
| `dd:tf/coil/conductor/elements/centres/phi` | `toroidal_angle_of_coil_conductor_element` | old target |
| `dd:tf/coil/conductor/elements/intermediate_points/phi` | `toroidal_angle_of_coil_conductor_element` | old target |
| `dd:tf/coil/conductor/elements/start_points/phi` | `toroidal_angle_of_coil_conductor_element` | old target |

This receipt measures the cohort; it does not authorize a global fold. Each
row still requires its governed owner/geometry disposition, with stale-source
handling kept separate from live-source rewiring.
