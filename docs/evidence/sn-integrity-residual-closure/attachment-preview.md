# Live attachment authority preview

## Outcome

**COMPLETE — read-only preview outcome `would_apply`.** One invocation derived
the complete live unsourced-identity cohort from the production graph, joined
each identity to its reviewed reverse-search rank-1 path, re-read the exact DD
path unit and current live owner closure, and partitioned all **12** identities
as **6 `attach` + 5 `adjudicate-collision` + 1 `no-candidate`**.

The governed attachment preview evaluated only the six paths not already held
by another live canonical identity. It admitted **4 of 6** and refused **2 of
6**, so the signed receipt reports `outcome=would_apply` and
`would_change=4`. No provider call, review draw, acceptance, identity fold, or
durable graph mutation occurred.

The signed live participant authority SHA-256 is
`fb3f33ccf35acf9b05b2c6f7bd55f00c7ec8b6c3a7725d4af5428ece9634b50e`.
The resulting preview-manifest SHA-256 is
`0a8c5d023e7dc06554e4afaead0e5ab00e83d77ef18eae6a78e1aa017c67c641`.
It pins DD **4.1.1** and DD-resolution manifest
`sha256:65a7ad8b1f1af0be59891f9dd84e506f292dfaa66930fe29601914882cdf9838`.

## Complete live cohort and collision adjudication

The live cohort predicate selected a `StandardName` only when its name and
catalog lifecycles were live, it had zero incoming `PRODUCED_NAME` edges, and
it had no live `HAS_PARENT` child. The invocation did not trust any carried
count. It returned **12 unique identities**.

For each candidate path, `canonical owner` is the union of current live
`StandardNameSource -> PRODUCED_NAME` ownership and the path's live
`IMASNode -> HAS_STANDARD_NAME` projection. Any non-empty owner set means a new
attachment would bind a second live identity to an already-held path and is
therefore `adjudicate-collision`, never `attach`.

| Live unsourced identity | Rank-1 candidate DD path | DD path unit | Current canonical owner | Disposition |
|---|---|---|---|---|
| `capacitance_of_ion_cyclotron_heating_antenna` | `ic_antennas/antenna/module/matching_element/capacitance` | `F` | — | `attach` |
| `cross_section_of_flux_surface` | `core_profiles/profiles_1d/grid/area` | `m^2` | `poloidal_plane_cross_sectional_area_of_flux_surface` | `adjudicate-collision` |
| `fast_ion_charge_state_power_at_inside_flux_surface` | `waves/coherent_wave/profiles_1d/ion/state/power_inside_fast` | `W` | `ion_charge_state_power_at_inside_flux_surface` | `adjudicate-collision` |
| `neutron_flux_due_to_fusion` | `neutron_diagnostic/neutron_flux_total` | `s^-1` | — | `attach` |
| `parallel_neutral_momentum_diffusion_coefficient` | `plasma_transport/model/ggd/neutral/momentum/d_parallel` | `m^2.s^-1` | — | `attach` |
| `poloidal_neutral_internal_state_momentum_convected_velocity` | `plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol` | `m.s^-1` | — | `attach` |
| `tendency_of_total_thermal_plasma_internal_energy` | `summary/global_quantities/denergy_thermal_dt/value` | `W` | `plasma_internal_energy` | `adjudicate-collision` |
| `toroidal_ion_charge_state_torque_density` | `plasma_sources/source/ggd/ion/state/momentum/phi` | `kg.m^-1.s^-2` | — | `attach` |
| `toroidal_line_averaged_plasma_velocity` | `spectrometer_x_ray_crystal/channel/profiles_line_integrated/velocity_tor` | `m.s^-1` | — | `attach` |
| `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | — | — | — | `no-candidate` |
| `x_direction_unit_vector_of_sensor` | `operational_instrumentation/sensor/direction/x` | `1` | `x_first_measurement_direction_unit_vector_of_strain_gauge` | `adjudicate-collision` |
| `z_direction_unit_vector_of_sensor` | `operational_instrumentation/sensor/direction/z` | `1` | `z_first_measurement_direction_unit_vector_of_strain_gauge` | `adjudicate-collision` |

Eleven identities and their ranks come from the committed reverse-search
record, whose SHA-256 is
`b1b6714f22071edaf048baaf79c3b828c0f041ca777f54ff097ccb16d21946d7`.
`toroidal_line_averaged_plasma_velocity` became live after that search
snapshot; the same invocation recovered its exact reviewed reverse-search path
from the live change steering that created the accepted identity. This prevents
the newer identity from being omitted merely because the earlier artifact had
an older cohort boundary.

Representative semantic distinctions remain explicit in the live evidence:

- `cross_section_of_flux_surface` is pending and quarantined, while its path is
  already held by the accepted poloidal-plane cross-sectional identity; it is
  a collision, not a second spelling to attach.
- `fast_ion_charge_state_power_at_inside_flux_surface` is accepted and valid,
  but its candidate path is already held by
  `ion_charge_state_power_at_inside_flux_surface`; the DD fast-versus-thermal
  recipient contradiction remains visible rather than being encoded by a
  second binding.
- `neutron_flux_due_to_fusion` carries unit `Hz` while the path carries
  `s^-1`; the dimensional adjudicator accepts those as the same count-rate
  dimension, but the source migration guard still refuses the proposed move
  because it would strip the prior target's last producing source.
- `toroidal_line_averaged_plasma_velocity` is accepted and valid at stored name
  score **0.95**. Its live description identifies a line-of-sight average of
  toroidal plasma bulk velocity, and its exact X-ray spectroscopy path is
  presently unowned.
- The trapped-thermal torque identity retains the sound negative: the inspected
  torque paths distinguish trapped source distribution from thermal recipient,
  whereas the identity conflates both roles. It remains `no-candidate` rather
  than receiving a nearest-path substitution.

## Signed attachment subset preview

Only the six `attach` rows entered the signed attachment operator. The operator
re-read target lifecycle, validation, exact source state, DD backing and units,
the ordinary attachment guard, predecessor last-producer closure, and the
complete participant set inside the same rollback-only transaction.

| Candidate identity | Proposed action | Preview result | Exact reason where refused |
|---|---|---|---|
| `capacitance_of_ion_cyclotron_heating_antenna` | attach existing unbound source | admitted | — |
| `neutron_flux_due_to_fusion` | retarget existing source | refused | `target would lose its last producing source` |
| `parallel_neutral_momentum_diffusion_coefficient` | create and attach exact DD source | admitted | — |
| `poloidal_neutral_internal_state_momentum_convected_velocity` | attach exact DD source | refused | `target lifecycle is not accepted: name_stage='reviewed'` |
| `toroidal_ion_charge_state_torque_density` | attach existing unbound source | admitted | — |
| `toroidal_line_averaged_plasma_velocity` | attach existing unbound source | admitted | — |

| Signed preview measure | Value |
|---|---:|
| Outcome | `would_apply` |
| Requested attach rows | **6** |
| Admitted | **4** |
| Refused | **2** |
| Would change | **4** |
| Collision rows excluded before attachment authority | **5** |
| No-candidate rows excluded before attachment authority | **1** |

The one missing `StandardNameSource` required for the
`parallel_neutral_momentum_diffusion_coefficient` row was constructed only
inside the preview transaction so the signed participant closure could be
complete; the transaction was rolled back. The receipt records the exact
rolled-back source id as
`dd:plasma_transport/model/ggd/neutral/momentum/d_parallel`.

## Nonmutation proof

The invocation counted both requested graph measures before opening the
preview transaction and after rollback. It also counted them after building
the preview but before rollback. Neither the authority build nor rollback
changed either counter.

| Counter | Before | Inside transaction after preview build | After rollback | Durable delta |
|---|---:|---:|---:|---:|
| `StandardNameChange` | **7,873** | **7,873** | **7,873** | **0** |
| `PRODUCED_NAME` | **5,774** | **5,774** | **5,774** | **0** |

The complete machine-readable receipt is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T204838424079-n-attachprev/attachment-preview-result.json`
(SHA-256
`90cc368c9bd24e009d9bab57fc5d497c6d27f9d042160ff89e424cf97ff78f44`).
The exact invocation driver and full log are retained beside it as
`build_attachment_preview.py` and `attachment-preview.log`; the log terminates
with `EXIT=0`.
