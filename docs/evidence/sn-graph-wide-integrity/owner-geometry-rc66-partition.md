# Owner and geometry rc66 disposition partition

## Outcome

The refreshed authority cohort is complete and disjoint:
**21 ready + 25 identity-absent + 1 vocabulary-refused + 2 stale-source = 49**.
The machine-readable authority is
[`owner-geometry-rc66-partition.json`](owner-geometry-rc66-partition.json).
Every row records the live StandardNameSource status, scalar selection, complete
live target set, non-old competing target ids, and its owner-qualified authority
target where current source authority permits one.

| Classification | Rows | Meaning |
|---|---:|---|
| ready | 21 | Target exists at `name_stage='accepted'` and `validation_status='valid'`; 9 sources already select it exclusively and 12 still require a signed disposition. |
| identity-absent | 25 | rc66 parses and composes the authority identity exactly, but no StandardName node exists; ordinary compose/review must create and accept it before apply. |
| vocabulary-refused | 1 | `field_map_grid` remains a registry-policy refusal and the tested identity fails the pinned closed grammar. |
| stale-source | 2 | The plural neutron-diagnostic sources remain `status='stale'`; detach or versioned migration must precede semantic rewiring. |
| **Total** | **49** | Exact authority cohort. |

This partition separates semantic authority from mutation authority. “Ready”
means the authority target is accepted and valid; it does not itself authorize
a graph write. The nine already-selected ready rows need no target change, while
the twelve remaining ready rows are candidates for the exact signed apply
instrument after the coordinator rechecks its closure guards.

## rc66 vocabulary change

Pinned `imas-standard-names 0.8.0rc66` admits five of the six
rows previously refused for missing locus vocabulary. All five now parse in
strict mode and compose back to the byte-identical spelling, but their target
identities are absent from the graph, so they move to `identity-absent` rather
than directly to `ready`.

| Source | Newly admitted locus | Grammar-valid authority target |
|---|---|---|
| `dd:ece/polarizer/centre/phi` | `polarizer` | `toroidal_coordinate_of_polarizer` |
| `dd:mse/channel/active_spatial_resolution/centre/phi` | `active_spatial_resolution_zone` | `toroidal_coordinate_of_active_spatial_resolution_zone` |
| `dd:spectrometer_visible/channel/active_spatial_resolution/centre/phi` | `active_spatial_resolution_zone` | `toroidal_coordinate_of_active_spatial_resolution_zone` |
| `dd:spectrometer_visible/channel/polarizer/centre/phi` | `polarizer` | `toroidal_coordinate_of_polarizer` |
| `dd:spi/injector/fragment/position/phi` | `pellet_fragment` | `toroidal_coordinate_of_pellet_fragment` |

The one refusal that remains is
`dd:b_field_non_axisymmetric/time_slice/field_map/grid/phi` →
`toroidal_coordinate_of_field_map_grid`. rc66 deliberately does not
admit `field_map_grid`: it describes a discretization artifact rather
than a physics-meaningful locus. The old measurement-position binding therefore
remains a recorded policy non-action; no nearest-object identity is substituted.

## Accepted-and-valid targets

Every ready row names its accepted-and-valid target explicitly in the JSON.
The 21 ready rows group into these six reviewed identity families:

| Accepted-and-valid target | Rows |
|---|---:|
| `toroidal_coordinate_of_aperture` | 8 |
| `toroidal_angle_of_coil_conductor_element` | 6 |
| `toroidal_coordinate_of_filter_window` | 3 |
| `toroidal_coordinate_of_optical_element` | 2 |
| `toroidal_coordinate_of_line_of_sight` | 1 |
| `toroidal_coordinate_of_reflector` | 1 |

The 25 grammar-valid identities that are still absent group as follows:

| Identity requiring ordinary lifecycle | Rows |
|---|---:|
| `toroidal_coordinate_of_detector` | 4 |
| `toroidal_coordinate_of_reflectometer_antenna` | 4 |
| `toroidal_coordinate_of_active_spatial_resolution_zone` | 2 |
| `toroidal_coordinate_of_camera` | 2 |
| `toroidal_coordinate_of_polarizer` | 2 |
| `toroidal_coordinate_of_beam_tracing_point` | 1 |
| `toroidal_coordinate_of_bragg_crystal` | 1 |
| `toroidal_coordinate_of_neutron_detector` | 1 |
| `toroidal_coordinate_of_pellet` | 1 |
| `toroidal_coordinate_of_pellet_fragment` | 1 |
| `toroidal_coordinate_of_pellet_path_point` | 1 |
| `toroidal_coordinate_of_reciprocating_probe` | 1 |
| `toroidal_coordinate_of_shatter_cone` | 1 |
| `toroidal_coordinate_of_shattering_position` | 1 |
| `toroidal_coordinate_of_soft_xray_detector` | 1 |
| `toroidal_coordinate_of_thomson_scattering_laser` | 1 |

## Exact row partition

“Live non-old target(s)” lists every current `PRODUCED_NAME` target other
than `toroidal_angle_of_measurement_position`. The complete live target set remains in the JSON.
A dash means no non-old target is live.

| # | DD source path | Class | Source status | Current scalar selection | Live non-old target(s) | Authority target | Action state |
|---:|---|---|---|---|---|---|---|
| 1 | `b_field_non_axisymmetric/time_slice/field_map/grid/phi` | **vocabulary-refused** | `composed` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_field_map_grid` | `policy-refusal` |
| 2 | `bolometer/camera/channel/aperture/centre/phi` | **ready** | `attached` | `toroidal_coordinate_of_aperture` | `toroidal_coordinate_of_aperture` | `toroidal_coordinate_of_aperture` | `already-selected` |
| 3 | `bolometer/camera/channel/detector/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_detector` | `await-ordinary-lifecycle` |
| 4 | `camera_visible/channel/aperture/centre/phi` | **ready** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_aperture` | `ready-for-signed-apply` |
| 5 | `camera_visible/channel/optical_element/geometry/centre/phi` | **ready** | `attached` | `toroidal_coordinate_of_optical_element` | `toroidal_coordinate_of_optical_element` | `toroidal_coordinate_of_optical_element` | `already-selected` |
| 6 | `camera_x_rays/aperture/centre/phi` | **ready** | `composed` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_aperture` | `ready-for-signed-apply` |
| 7 | `camera_x_rays/camera/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_camera` | `await-ordinary-lifecycle` |
| 8 | `coils_non_axisymmetric/coil/conductor/elements/end_points/phi` | **ready** | `attached` | `toroidal_angle_of_coil_conductor_element` | `toroidal_angle_of_coil_conductor_element` | `toroidal_angle_of_coil_conductor_element` | `already-selected` |
| 9 | `coils_non_axisymmetric/coil/conductor/elements/start_points/phi` | **ready** | `attached` | `toroidal_angle_of_coil_conductor_element` | `toroidal_angle_of_coil_conductor_element` | `toroidal_angle_of_coil_conductor_element` | `already-selected` |
| 10 | `ece/channel/beam_tracing/beam/position/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_beam_tracing_point` | `await-ordinary-lifecycle` |
| 11 | `ece/polarizer/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_polarizer` | `await-ordinary-lifecycle` |
| 12 | `hard_x_rays/channel/aperture/centre/phi` | **ready** | `composed` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_aperture` | `ready-for-signed-apply` |
| 13 | `hard_x_rays/channel/filter_window/centre/phi` | **ready** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_filter_window` | `ready-for-signed-apply` |
| 14 | `interferometer/channel/n_e/positions/phi` | **ready** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_line_of_sight` | `ready-for-signed-apply` |
| 15 | `langmuir_probes/reciprocating/plunge/position_average/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_reciprocating_probe` | `await-ordinary-lifecycle` |
| 16 | `mse/channel/active_spatial_resolution/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_active_spatial_resolution_zone` | `await-ordinary-lifecycle` |
| 17 | `mse/channel/aperture/centre/phi` | **ready** | `composed` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_aperture` | `ready-for-signed-apply` |
| 18 | `mse/channel/detector/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_detector` | `await-ordinary-lifecycle` |
| 19 | `neutron_diagnostic/detector/aperture/centre/phi` | **ready** | `attached` | `toroidal_coordinate_of_aperture` | `toroidal_coordinate_of_aperture` | `toroidal_coordinate_of_aperture` | `already-selected` |
| 20 | `neutron_diagnostic/detector/geometry/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_neutron_detector` | `await-ordinary-lifecycle` |
| 21 | `neutron_diagnostic/detectors/aperture/centre/phi` | **stale-source** | `stale` | `toroidal_angle_of_measurement_position` | — | — | `await-stale-source-lifecycle` |
| 22 | `neutron_diagnostic/detectors/detector/centre/phi` | **stale-source** | `stale` | `toroidal_angle_of_measurement_position` | — | — | `await-stale-source-lifecycle` |
| 23 | `pellets/time_slice/pellet/path_geometry/first_point/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | `toroidal_angle_of_along_pellet_path` | `toroidal_coordinate_of_pellet_path_point` | `await-ordinary-lifecycle` |
| 24 | `reflectometer_fluctuation/channel/antenna_detection_static/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_reflectometer_antenna` | `await-ordinary-lifecycle` |
| 25 | `reflectometer_fluctuation/channel/antenna_emission_static/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_reflectometer_antenna` | `await-ordinary-lifecycle` |
| 26 | `reflectometer_profile/channel/antenna_detection/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_reflectometer_antenna` | `await-ordinary-lifecycle` |
| 27 | `reflectometer_profile/channel/antenna_emission/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_reflectometer_antenna` | `await-ordinary-lifecycle` |
| 28 | `soft_x_rays/channel/detector/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_soft_xray_detector` | `await-ordinary-lifecycle` |
| 29 | `soft_x_rays/channel/filter_window/centre/phi` | **ready** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_filter_window` | `ready-for-signed-apply` |
| 30 | `spectrometer_uv/channel/aperture/centre/phi` | **ready** | `attached` | `toroidal_coordinate_of_aperture` | `toroidal_coordinate_of_aperture` | `toroidal_coordinate_of_aperture` | `already-selected` |
| 31 | `spectrometer_uv/channel/detector/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_detector` | `await-ordinary-lifecycle` |
| 32 | `spectrometer_visible/channel/active_spatial_resolution/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_active_spatial_resolution_zone` | `await-ordinary-lifecycle` |
| 33 | `spectrometer_visible/channel/detector/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_detector` | `await-ordinary-lifecycle` |
| 34 | `spectrometer_visible/channel/optical_element/geometry/centre/phi` | **ready** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_optical_element` | `ready-for-signed-apply` |
| 35 | `spectrometer_visible/channel/polarizer/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_polarizer` | `await-ordinary-lifecycle` |
| 36 | `spectrometer_x_ray_crystal/channel/aperture/centre/phi` | **ready** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_aperture` | `ready-for-signed-apply` |
| 37 | `spectrometer_x_ray_crystal/channel/camera/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_camera` | `await-ordinary-lifecycle` |
| 38 | `spectrometer_x_ray_crystal/channel/crystal/centre/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_bragg_crystal` | `await-ordinary-lifecycle` |
| 39 | `spectrometer_x_ray_crystal/channel/filter_window/centre/phi` | **ready** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_filter_window` | `ready-for-signed-apply` |
| 40 | `spectrometer_x_ray_crystal/channel/reflector/centre/phi` | **ready** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_reflector` | `ready-for-signed-apply` |
| 41 | `spi/injector/fragment/position/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_pellet_fragment` | `await-ordinary-lifecycle` |
| 42 | `spi/injector/pellet/position/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_pellet` | `await-ordinary-lifecycle` |
| 43 | `spi/injector/shatter_cone/origin/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_shatter_cone` | `await-ordinary-lifecycle` |
| 44 | `spi/injector/shattering_position/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_shattering_position` | `await-ordinary-lifecycle` |
| 45 | `tf/coil/conductor/elements/centres/phi` | **ready** | `attached` | `toroidal_angle_of_coil_conductor_element` | `toroidal_angle_of_coil_conductor_element` | `toroidal_angle_of_coil_conductor_element` | `already-selected` |
| 46 | `tf/coil/conductor/elements/end_points/phi` | **ready** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_angle_of_coil_conductor_element` | `ready-for-signed-apply` |
| 47 | `tf/coil/conductor/elements/intermediate_points/phi` | **ready** | `attached` | `toroidal_angle_of_coil_conductor_element` | `toroidal_angle_of_coil_conductor_element` | `toroidal_angle_of_coil_conductor_element` | `already-selected` |
| 48 | `tf/coil/conductor/elements/start_points/phi` | **ready** | `attached` | `toroidal_angle_of_coil_conductor_element` | `toroidal_angle_of_coil_conductor_element` | `toroidal_angle_of_coil_conductor_element` | `already-selected` |
| 49 | `thomson_scattering/laser/end_point/phi` | **identity-absent** | `attached` | `toroidal_angle_of_measurement_position` | — | `toroidal_coordinate_of_thomson_scattering_laser` | `await-ordinary-lifecycle` |

## Signature and write-free proof

The JSON signs the complete 49-row payload with sorted-key compact JSON:

- canonicalization: `jq -cS '.rows'`
- rows SHA-256: `4de9c2df481180931a47b7a8bcc76cb69253e23d96e2dfa151bd86edcb76c8cd`
- JSON file SHA-256: `dbb37f7be12ba99d7e85bf13b9d63e6c19cb6c20bd35fe687e590f798e2dc85b`

The live graph read measured `StandardNameChange` at **7,492 before**
and **7,492 after** (delta 0). No graph write and no provider call occurred.
This is a read-only disposition partition; graph mutation remains a separate,
serialized, closure-guarded node.

## Authority inputs

- Live plan `sn-graph-wide-integrity` at version 186, section
  `remaining-work`.
- Locked rules `geometry-cardinality-rule` and
  `standard-name-ordinality`: the physical owner survives named-point
  parameterization, and start/end/first/second ordering remains DD provenance
  rather than Standard Name identity.
- Prior signed semantic mapping
  `docs/evidence/sn-graph-wide-integrity/owner-geometry-authority-mapping.json`,
  rows digest `64c868b668d392572c59704dc0e1891130f1a8bd8dce1bbe948dbab78af134ec`.
- Pinned public parser/composer from `imas-standard-names 0.8.0rc66`:
  22 of 23
  unique candidate identities pass exact strict round-trip; the sole refusal is
  the policy-rejected `field_map_grid` spelling.
- One read-only live graph query over all 49 exact `StandardNameSource`
  ids, their complete `PRODUCED_NAME` target sets, scalar
  `produced_sn_id` mirrors, source statuses, and candidate target
  lifecycle fields.

