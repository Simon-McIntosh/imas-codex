# Owner and geometry authority mapping

## Outcome

All 49 outstanding DD paths have one signed, row-level disposition in
[`owner-geometry-authority-mapping.json`](owner-geometry-authority-mapping.json).
The census partitions remain separate and exact: 39 rows have only the old
`toroidal_angle_of_measurement_position` target, 10 also have a competing live
target, and 39 + 10 = 49.

The authority review assigns 41 grammar-valid owner or geometry identities and
records eight reviewed refusals. Six refusals are closed-vocabulary gaps and
two are stale sources for which current DD rewiring authority is absent. No
nearest available object was substituted for a missing token.

The JSON file SHA-256 is
`438392909410ad85499057247133dca13bb8567581030ce59520e7b079beac03`.
Its signed `rows` payload uses sorted-key compact JSON and has SHA-256
`64c868b668d392572c59704dc0e1891130f1a8bd8dce1bbe948dbab78af134ec`.

| Measure | Result |
|---|---:|
| Census rows | 49 |
| Old-target-only rows | 39 |
| Old-target-with-competitor rows | 10 |
| Replacement dispositions | 41 |
| Reviewed refusals | 8 |
| Replacement rows whose target is already accepted and valid | 21 |
| Replacement rows whose target identity is absent and requires ordinary review | 20 |
| Existing competitors matching this authority | 9 |
| Existing competitors rejected as the wrong owner | 1 |

This is semantic authority, not mutation authority. Existing accepted targets
may be considered by a later exact retarget manifest. Absent targets must enter
the ordinary compose/review lifecycle first. Vocabulary-gap and stale-source
rows remain non-actions.

## Authority method

The live plan at version 174 and its locked geometry-cardinality and
non-ordinality decisions govern the reading:

- A scalar centre, origin, endpoint, or other named point retains the physical
  owner and uses `toroidal_coordinate_of_<owner>`. Words such as `first`,
  `start`, and `end` stay in DD provenance and do not enter identity.
- Per-element FLT_1D geometry arrays are not scalar named points. The six coil
  rows use the already accepted
  `toroidal_angle_of_coil_conductor_element`, matching the DD array owner.
- Immediate DD structure and documentation establish the owner. For example,
  `interferometer/channel/n_e/positions` explicitly says “positions along the
  line of sight,” while the ECE beam-tracing row is a sampled
  `beam_tracing_point` and not a generic measurement position.
- A closed-grammar miss is a refusal. `optical_element` is not used for a
  polarizer, `pellet` is not used for a shattered fragment, and `aperture` is
  not used for an active spatial-resolution zone.
- A stale source is not rewired from path resemblance. Its stale membership
  must be handled by the source-lifecycle or version-migration authority first.

The live DD query resolved all 49 IMAS nodes and their four-level parent
context. The query also read the sibling `r`, `z`, and `phi` bindings for all 49
parents, using them only as corroborating or contradictory evidence, never as
automatic authority. That distinction catches the one competing target that is
itself wrong:
`spectrometer_uv/channel/detector/centre/phi` is currently also bound to
`toroidal_coordinate_of_aperture`, but the DD owner is the detector front face,
so its intended identity is `toroidal_coordinate_of_detector`.

## Grammar validation

Validation used the public ISN parser and composer from
`imas-standard-names 0.8.0rc65.dev2+g7e16c45d4`. Every unique proposed
replacement was parsed and composed back to the byte-identical spelling: 19 of
19 passed. The full machine log is retained at
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T054828007400-owner-geometry-authority-mapping/grammar-validation.json`
(SHA-256
`6ca7153fdddc781ccd7833a0e4fec7d021d606cdfca695f44eb9adc86dc6faf3`).

Existing accepted targets provide additional review evidence:

| Identity | Live state | Name review score | Rows |
|---|---|---:|---:|
| `toroidal_coordinate_of_aperture` | accepted, valid | 0.96875 | 8 |
| `toroidal_coordinate_of_filter_window` | accepted, valid | 1.00000 | 3 |
| `toroidal_coordinate_of_line_of_sight` | accepted, valid | 0.96250 | 1 |
| `toroidal_coordinate_of_optical_element` | accepted, valid | 1.00000 | 2 |
| `toroidal_coordinate_of_reflector` | accepted, valid | 0.97500 | 1 |
| `toroidal_angle_of_coil_conductor_element` | accepted, valid | not recorded | 6 |

The four unique missing locus tokens are absent from the public locus registry,
and the corresponding full identity spellings all fail the closed parser with
`UnknownBaseTokenError`:

| Required locus | Tested owner-qualified identity | Affected rows | Disposition |
|---|---|---:|---|
| `field_map_grid` | `toroidal_coordinate_of_field_map_grid` | 1 | reviewed refusal; retain old target pending vocabulary |
| `polarizer` | `toroidal_coordinate_of_polarizer` | 2 | reviewed refusal; do not substitute optical element |
| `active_spatial_resolution_zone` | `toroidal_coordinate_of_active_spatial_resolution_zone` | 2 | reviewed refusal; do not substitute aperture or generic zone |
| `pellet_fragment` | `toroidal_coordinate_of_pellet_fragment` | 1 | reviewed refusal; do not substitute pellet or pellet-path point |

## Identity families

| Disposition | Rows | Graph state |
|---|---:|---|
| `toroidal_coordinate_of_aperture` | 8 | accepted, valid |
| `toroidal_angle_of_coil_conductor_element` | 6 | accepted, valid |
| `toroidal_coordinate_of_detector` | 4 | absent; ordinary review required |
| `toroidal_coordinate_of_reflectometer_antenna` | 4 | absent; ordinary review required |
| `toroidal_coordinate_of_filter_window` | 3 | accepted, valid |
| `toroidal_coordinate_of_camera` | 2 | absent; ordinary review required |
| `toroidal_coordinate_of_optical_element` | 2 | accepted, valid |
| `toroidal_coordinate_of_beam_tracing_point` | 1 | absent; ordinary review required |
| `toroidal_coordinate_of_bragg_crystal` | 1 | absent; ordinary review required |
| `toroidal_coordinate_of_line_of_sight` | 1 | accepted, valid |
| `toroidal_coordinate_of_neutron_detector` | 1 | absent; ordinary review required |
| `toroidal_coordinate_of_pellet` | 1 | absent; ordinary review required |
| `toroidal_coordinate_of_pellet_path_point` | 1 | absent; ordinary review required |
| `toroidal_coordinate_of_reciprocating_probe` | 1 | absent; ordinary review required |
| `toroidal_coordinate_of_reflector` | 1 | accepted, valid |
| `toroidal_coordinate_of_shatter_cone` | 1 | absent; ordinary review required |
| `toroidal_coordinate_of_shattering_position` | 1 | absent; ordinary review required |
| `toroidal_coordinate_of_soft_xray_detector` | 1 | absent; ordinary review required |
| `toroidal_coordinate_of_thomson_scattering_laser` | 1 | absent; ordinary review required |
| Reviewed vocabulary-gap refusal | 6 | old target retained; no apply authority |
| Reviewed stale-source refusal | 2 | old target retained; source lifecycle first |

## Exact row mapping

`old-only` means that the old measurement-position target is the only live
target. `competing` means that the old target and another live target coexist.

| # | DD path | Census class | Signed disposition |
|---:|---|---|---|
| 1 | `b_field_non_axisymmetric/time_slice/field_map/grid/phi` | old-only | **Refuse:** missing `field_map_grid`; keep old target pending vocabulary |
| 2 | `bolometer/camera/channel/aperture/centre/phi` | competing | `toroidal_coordinate_of_aperture` |
| 3 | `bolometer/camera/channel/detector/centre/phi` | old-only | `toroidal_coordinate_of_detector` |
| 4 | `camera_visible/channel/aperture/centre/phi` | old-only | `toroidal_coordinate_of_aperture` |
| 5 | `camera_visible/channel/optical_element/geometry/centre/phi` | competing | `toroidal_coordinate_of_optical_element` |
| 6 | `camera_x_rays/aperture/centre/phi` | old-only | `toroidal_coordinate_of_aperture` |
| 7 | `camera_x_rays/camera/centre/phi` | old-only | `toroidal_coordinate_of_camera` |
| 8 | `coils_non_axisymmetric/coil/conductor/elements/end_points/phi` | competing | `toroidal_angle_of_coil_conductor_element` |
| 9 | `coils_non_axisymmetric/coil/conductor/elements/start_points/phi` | competing | `toroidal_angle_of_coil_conductor_element` |
| 10 | `ece/channel/beam_tracing/beam/position/phi` | old-only | `toroidal_coordinate_of_beam_tracing_point` |
| 11 | `ece/polarizer/centre/phi` | old-only | **Refuse:** missing `polarizer`; keep old target pending vocabulary |
| 12 | `hard_x_rays/channel/aperture/centre/phi` | old-only | `toroidal_coordinate_of_aperture` |
| 13 | `hard_x_rays/channel/filter_window/centre/phi` | old-only | `toroidal_coordinate_of_filter_window` |
| 14 | `interferometer/channel/n_e/positions/phi` | old-only | `toroidal_coordinate_of_line_of_sight` |
| 15 | `langmuir_probes/reciprocating/plunge/position_average/phi` | old-only | `toroidal_coordinate_of_reciprocating_probe` |
| 16 | `mse/channel/active_spatial_resolution/centre/phi` | old-only | **Refuse:** missing `active_spatial_resolution_zone`; keep old target pending vocabulary |
| 17 | `mse/channel/aperture/centre/phi` | old-only | `toroidal_coordinate_of_aperture` |
| 18 | `mse/channel/detector/centre/phi` | old-only | `toroidal_coordinate_of_detector` |
| 19 | `neutron_diagnostic/detector/aperture/centre/phi` | competing | `toroidal_coordinate_of_aperture` |
| 20 | `neutron_diagnostic/detector/geometry/centre/phi` | old-only | `toroidal_coordinate_of_neutron_detector` |
| 21 | `neutron_diagnostic/detectors/aperture/centre/phi` | old-only | **Refuse:** stale source; keep old target pending detach or versioned migration |
| 22 | `neutron_diagnostic/detectors/detector/centre/phi` | old-only | **Refuse:** stale source; keep old target pending detach or versioned migration |
| 23 | `pellets/time_slice/pellet/path_geometry/first_point/phi` | old-only | `toroidal_coordinate_of_pellet_path_point` |
| 24 | `reflectometer_fluctuation/channel/antenna_detection_static/centre/phi` | old-only | `toroidal_coordinate_of_reflectometer_antenna` |
| 25 | `reflectometer_fluctuation/channel/antenna_emission_static/centre/phi` | old-only | `toroidal_coordinate_of_reflectometer_antenna` |
| 26 | `reflectometer_profile/channel/antenna_detection/centre/phi` | old-only | `toroidal_coordinate_of_reflectometer_antenna` |
| 27 | `reflectometer_profile/channel/antenna_emission/centre/phi` | old-only | `toroidal_coordinate_of_reflectometer_antenna` |
| 28 | `soft_x_rays/channel/detector/centre/phi` | old-only | `toroidal_coordinate_of_soft_xray_detector` |
| 29 | `soft_x_rays/channel/filter_window/centre/phi` | old-only | `toroidal_coordinate_of_filter_window` |
| 30 | `spectrometer_uv/channel/aperture/centre/phi` | competing | `toroidal_coordinate_of_aperture` |
| 31 | `spectrometer_uv/channel/detector/centre/phi` | competing | `toroidal_coordinate_of_detector`; reject competing aperture owner |
| 32 | `spectrometer_visible/channel/active_spatial_resolution/centre/phi` | old-only | **Refuse:** missing `active_spatial_resolution_zone`; keep old target pending vocabulary |
| 33 | `spectrometer_visible/channel/detector/centre/phi` | old-only | `toroidal_coordinate_of_detector` |
| 34 | `spectrometer_visible/channel/optical_element/geometry/centre/phi` | old-only | `toroidal_coordinate_of_optical_element` |
| 35 | `spectrometer_visible/channel/polarizer/centre/phi` | old-only | **Refuse:** missing `polarizer`; keep old target pending vocabulary |
| 36 | `spectrometer_x_ray_crystal/channel/aperture/centre/phi` | old-only | `toroidal_coordinate_of_aperture` |
| 37 | `spectrometer_x_ray_crystal/channel/camera/centre/phi` | old-only | `toroidal_coordinate_of_camera` |
| 38 | `spectrometer_x_ray_crystal/channel/crystal/centre/phi` | old-only | `toroidal_coordinate_of_bragg_crystal` |
| 39 | `spectrometer_x_ray_crystal/channel/filter_window/centre/phi` | old-only | `toroidal_coordinate_of_filter_window` |
| 40 | `spectrometer_x_ray_crystal/channel/reflector/centre/phi` | old-only | `toroidal_coordinate_of_reflector` |
| 41 | `spi/injector/fragment/position/phi` | old-only | **Refuse:** missing `pellet_fragment`; keep old target pending vocabulary |
| 42 | `spi/injector/pellet/position/phi` | old-only | `toroidal_coordinate_of_pellet` |
| 43 | `spi/injector/shatter_cone/origin/phi` | old-only | `toroidal_coordinate_of_shatter_cone` |
| 44 | `spi/injector/shattering_position/phi` | old-only | `toroidal_coordinate_of_shattering_position` |
| 45 | `tf/coil/conductor/elements/centres/phi` | competing | `toroidal_angle_of_coil_conductor_element` |
| 46 | `tf/coil/conductor/elements/end_points/phi` | old-only | `toroidal_angle_of_coil_conductor_element` |
| 47 | `tf/coil/conductor/elements/intermediate_points/phi` | competing | `toroidal_angle_of_coil_conductor_element` |
| 48 | `tf/coil/conductor/elements/start_points/phi` | competing | `toroidal_angle_of_coil_conductor_element` |
| 49 | `thomson_scattering/laser/end_point/phi` | old-only | `toroidal_coordinate_of_thomson_scattering_laser` |

## Read-only proof and evidence inputs

The live DD/graph evidence read sampled protected counters before and after all
queries. Both decimal values were identical:

| Node label | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameChange` | 7,451 | 7,451 | 0 |
| `LLMCost` | 27,467 | 27,467 | 0 |

No graph write or provider call occurred. The retained read-only query artifacts
are:

- DD parent context, current toroidal identities, and protected counters:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T054828007400-owner-geometry-authority-mapping/authority-query.json`,
  SHA-256
  `980d0ad355e713438e53eff39cce4f5b6c4a97471889990190571af75c71ac74`.
- Axis-sibling evidence and protected counters:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T054828007400-owner-geometry-authority-mapping/axis-sibling-query.json`,
  SHA-256
  `3ecf0fb12fbc363782de1efda3d9bbfebe18396b19a3c1e90f80f9396d598a52`.

The next executor must preserve the split between semantic authority and apply
readiness: 21 replacement rows point to accepted/valid live identities, 20
point to grammar-valid but absent identities that require ordinary review, six
are vocabulary refusals, and two are stale-source refusals.
