# Live integrity closing census

## Outcome

The read-only closing census at **2026-08-21 14:33:52 UTC** finds that the
zero classes remain closed for missing live targets, backing-projection
mismatches, signed retirements, explicit-axis generic-parent residue, and
accepted-name grammar failures. Four classes still contain rows:

- semantic integrity has **24** rows: **23** sources with multiple live targets
  and **1** scalar/edge mismatch;
- stale provenance has **3** live bindings;
- the genuine-orphan ratchet has **20** accepted, valid names;
- structural provenance has **2** unsourced names with live children.

The separately tracked 49-row owner/geometry authority has **5** sources still
on `toroidal_angle_of_measurement_position`. All **5** are accounted below.
That target has **28** total live producers: the 5 authority residues plus 23
correctly retained measurement-position sources outside that owner/geometry
repair cohort.

Every nonzero row is accounted for in this artifact. A row either reproduces a
recorded fail-closed guard reason verbatim or names the governed next action and
its owning plan. Counts across classes must not be summed as a unique-row total:
the tables are independent integrity projections and may overlap.

## Measurement authority and nonmutation

- Live plan authority: `imas-codex:sn-graph-wide-integrity`, version **225**.
- Source checkout: `fe767440ea783c9eb370075c78b25ebd5a025c4f`.
- Graph selection and credentials came from the repository's configured
  `GraphClient`; the census used read-only Cypher only.
- `StandardNameChange` stayed **7,754 → 7,754**.
- `LLMCost` stayed **27,619 → 27,619**.
- Active grammar authority is exactly **0.8.0rc66**, with **22 segments** and
  **956 tokens**. Historical snapshots are inactive.

The machine-readable query result is retained at
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T142729794531-closecensus/live-census.json`.

## Complete count table

“As found” is the first committed plan census for the class. “Prior close” is
the last committed closing baseline before this fresh measurement. A dash means
the plan did not record that class independently at the first census; it is not
a fabricated zero.

| Tracked integrity class | As found | Prior close | Current live | Current disposition |
|---|---:|---:|---:|---|
| Source with no live target | 596 | 0 | **0** | Closed; no remaining rows |
| Source scalar differs from live target | 144 | 0 | **1** | 1/1 routed below |
| Backing projection differs from live target | 2 | 0 | **0** | Closed; no remaining rows |
| Source with multiple live targets | 226 | 23 | **23** | 23/23 routed below |
| Stale source with a live binding | 58 | 3 | **3** | 3/3 recorded refusals below |
| Signed retirement cohort outstanding | 16 | 0 | **0** | Closed; all 16 are terminal `superseded`, with zero producers |
| Owner/geometry cohort still on old target | 49 | 27 | **5** | 2 recorded refusals + 3 routed actions below |
| All live producers of `toroidal_angle_of_measurement_position` | 61 | 28 | **28** | 5 owner residues + 23 authority-retained producers, fully partitioned below |
| Genuine accepted/valid orphans | 85 | 36 | **20** | 15 recorded refusals + 5 routed actions below |
| Unsourced structural names with live children | — | 0 | **2** | 2/2 routed below |
| Explicit-axis source also bound to generic parent | 1 | 1 | **0** | Closed; no remaining rows |
| Accepted names failing strict canonical grammar | 2 | 0 | **0** | Closed; 0 failures among 2,532 accepted names |

The current semantic total is therefore **24 = 23 multiple-target + 1
scalar-mismatch**, with no missing-target or projection-mismatch rows.

## Semantic rows: 24 of 24 routed

### Multiple live targets: 23 of 23

All 23 rows are routed to `imas-codex:sn-graph-wide-integrity`: perform
per-source semantic adjudication, then use an exact signed reconcile instrument
to retain or detach each target. The shared next action is deliberately not a
bulk detach: every live target shown here is also present in the source's
backing projection, so cardinality alone cannot choose the authoritative
meaning.

| Source | Current live targets |
|---|---|
| `dd:core_sources/source/profiles_1d/ion/momentum/radial` | `momentum_source`; `radial_ion_momentum_source` |
| `dd:edge_profiles/ggd/mass_density/values` | `mass_density`; `total_plasma_mass_density` |
| `dd:edge_profiles/ggd/neutral/velocity/phi` | `toroidal_neutral_velocity`; `toroidal_neutral_momentum_convection_velocity` |
| `dd:edge_sources/source/ggd/ion/momentum/r` | `radial_ion_momentum`; `radial_ion_momentum_source` |
| `dd:edge_transport/model/ggd/neutral/state/particles/d_pol/values` | `neutral_state_particle_diffusivity`; `poloidal_neutral_state_particle_diffusivity` |
| `dd:equilibrium/time_slice/profiles_1d/mass_density` | `mass_density`; `total_plasma_mass_density` |
| `dd:equilibrium/time_slice/profiles_1d/squareness_upper_outer` | `outer_squareness_of_flux_surface`; `squareness_of_flux_surface`; `upper_outer_squareness_of_flux_surface` |
| `dd:gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_gyrocenter_bessel_1/j_parallel` | `normalized_perturbed_current_density`; `parallel_normalized_perturbed_current_density`; `parallel_normalized_perturbed_current_density_bessel_1`; `perturbed_current_density` |
| `dd:langmuir_probes/embedded/surface_area` | `area_of_langmuir_probe`; `wetted_area_of_langmuir_probe` |
| `dd:langmuir_probes/reciprocating/surface_area` | `area_of_langmuir_probe`; `wetted_area_of_langmuir_probe` |
| `dd:mhd/ggd/mass_density/values` | `mass_density`; `total_plasma_mass_density` |
| `dd:mhd_linear/time_slice/toroidal_mode/plasma/phi_potential_perturbed/imaginary` | `electrostatic_potential_imaginary_part`; `perturbed_electrostatic_potential_imaginary_part` |
| `dd:plasma_profiles/ggd/mass_density/values` | `mass_density`; `total_plasma_mass_density` |
| `dd:plasma_profiles/ggd/neutral/velocity/phi` | `toroidal_neutral_velocity`; `toroidal_neutral_momentum_convection_velocity` |
| `dd:plasma_sources/source/ggd/ion/momentum/radial` | `radial_ion_momentum`; `radial_ion_momentum_source` |
| `dd:plasma_sources/source/profiles_1d/ion/momentum/radial` | `radial_ion_momentum`; `radial_ion_momentum_source` |
| `dd:plasma_transport/model/ggd/momentum/flux/radial` | `radial_momentum`; `radial_momentum_flux` |
| `dd:plasma_transport/model/ggd/neutral/energy/v_parallel/values` | `neutral_species_energy_convection_velocity`; `parallel_neutral_species_energy_convection_velocity` |
| `dd:plasma_transport/model/ggd/neutral/state/momentum/flux/poloidal` | `poloidal_linear_neutral_internal_state_momentum_flux`; `poloidal_neutral_state_momentum_flux` |
| `dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial` | `radial_neutral_internal_state_momentum_flux`; `radial_neutral_state_momentum_flux` |
| `dd:plasma_transport/model/profiles_1d/neutral/state/momentum/flux/poloidal` | `poloidal_linear_neutral_internal_state_momentum_flux`; `poloidal_neutral_state_momentum_flux` |
| `dd:plasma_transport/model/profiles_1d/neutral/state/momentum/flux_limiter/z` | `vertical_coordinate_of_active_limiter_point`; `vertical_neutral_state_momentum_flux_limiter_coefficient` |
| `dd:runaway_electrons/global_quantities/volume_average/current_density` | `parallel_runaway_electron_current_density`; `parallel_volume_averaged_runaway_electron_current_density`; `volume_averaged_runaway_electron_current_density` |

### Scalar/edge mismatch: 1 of 1

| Source | Scalar | Sole live target | Routed next action |
|---|---|---|---|
| `dd:plasma_sources/source/ggd/neutral/state/momentum/phi` | `neutral_internal_state_torque_density` | `toroidal_neutral_internal_state_torque_density` | `imas-codex:sn-graph-wide-integrity`: exact signed scalar-mirror reconcile after asserting the sole edge and backing projection; no raw property write |

## Stale live bindings: 3 of 3 recorded refusals

These are the complete survivors of the signed 58-row stale-source authority.
The reason strings below are reproduced verbatim from the applying transaction.

| Source → target | Recorded fail-closed refusal |
|---|---|
| `dd:ece/channel/t_e_voltage` → `voltage_of_diagnostic_antenna` | `Detach would orphan the target; this stale source remains its final producer` |
| `dd:equilibrium/time_slice/boundary_separatrix/closest_wall_point/distance` → `gap_at_plasma_boundary` | `Signed source closure changed; the row no longer matches its signed binding/projection authority` |
| `dd:equilibrium/time_slice/profiles_1d/b_average` → `flux_surface_average_magnetic_field_magnitude` | `Detach would orphan the target; this stale source remains its final producer` |

The voltage row also has the later exact transition refusal:
`source migration compare-and-set failed: dd:ece/channel/t_e_voltage(exists=True, status='stale', claimed=False, bindings=['voltage_of_diagnostic_antenna'], scalar='voltage_of_diagnostic_antenna')`.
Its source-less identity transition remains owned by
`imas-codex:sn-graph-wide-integrity`; the concurrent voltage-rename evidence
node owns that execution record, so it is not duplicated here.

## Genuine orphans: 20 of 20 accounted

The 20 current rows partition exactly as **15 recorded attachment refusals + 5
governed next actions**. The 15 guard strings are copied verbatim from the
31-row governed attachment transaction.

| Standard name | Disposition |
|---|---|
| `capacitance_of_ion_cyclotron_heating_antenna` | Refused: `target would lose its last producing source` |
| `cross_section_of_flux_surface` | Refused: `target lifecycle is not accepted: name_stage='pending'` |
| `fast_ion_charge_state_power_at_inside_flux_surface` | `imas-codex:sn-graph-wide-integrity`: HOLD until DD authority resolves and corrects the `power_inside_fast` recipient-population contradiction |
| `line_integrated_electron_density` | Refused: `target lifecycle is not accepted: name_stage='drafted'` |
| `magnetic_field_at_pedestal_top_low_field_side_magnitude` | Refused: `target lifecycle is not accepted: name_stage='drafted'` |
| `minimum_of_safety_factor` | Refused: `target lifecycle is not accepted: name_stage='reviewed'` |
| `neutral_state_power_density` | Refused: `target lifecycle is not accepted: name_stage='reviewed'` |
| `neutron_flux_due_to_fusion` | Refused: `target would lose its last producing source` |
| `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | Refused: `target would lose its last producing source` |
| `parallel_neutral_momentum_diffusion_coefficient` | Refused: `state-resolution mismatch: path 'plasma_transport/model/profiles_1d/neutral/state/momentum/d_parallel' is state-resolved but SN 'parallel_neutral_momentum_diffusion_coefficient' is species-level` |
| `poloidal_neutral_internal_state_momentum_convected_velocity` | Refused: `target lifecycle is not accepted: name_stage='reviewed'` |
| `poloidal_straight_field_line_angle` | Refused: `target lifecycle is not accepted: name_stage='drafted'` |
| `tendency_of_total_thermal_plasma_internal_energy` | Refused: `target validation is not valid: validation_status='quarantined'` |
| `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | Refused: `target lifecycle is not accepted: name_stage='reviewed'` |
| `toroidal_ion_charge_state_torque_density` | `imas-codex:sn-graph-wide-integrity`: HOLD until one exact DD source simultaneously carries total, charge-state-resolved, and toroidal semantics |
| `toroidal_line_integrated_impurity_ion_velocity` | Refused: `target lifecycle is not accepted: name_stage='drafted'` |
| `toroidal_neutral_state_momentum_diffusivity` | Refused: `source has multiple live targets: ['toroidal_momentum_diffusivity', 'toroidal_neutral_internal_state_momentum_diffusion_coefficient']` |
| `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | `imas-codex:sn-graph-wide-integrity`: execute the adjudicated exact signed retirement only after rechecking source/recipient closure |
| `x_direction_unit_vector_of_sensor` | `imas-codex:sn-graph-wide-integrity`: HOLD attachment until DD corrects the child to unit `1` or redefines the parent as a metric displacement |
| `z_direction_unit_vector_of_sensor` | `imas-codex:sn-graph-wide-integrity`: same DD unit/parent resolution as the x component before attachment |

## Unsourced structural names: 2 of 2 routed

| Structural name | Live children | Routed next action |
|---|---|---|
| `electron_diffusivity` | `effective_electron_diffusivity`; `parallel_electron_diffusivity`; `poloidal_electron_diffusivity` | Prevention owner `imas-codex:sn-pipeline-recovery`: stop stale derived-source reseeding. Live repair owner `imas-codex:sn-graph-wide-integrity`: exact structural-source reconciliation after the graph settles. |
| `ion_diffusivity` | `effective_ion_diffusivity`; `parallel_ion_diffusivity`; `poloidal_ion_diffusivity` | Prevention owner `imas-codex:sn-pipeline-recovery`: stop stale derived-source reseeding. Live repair owner `imas-codex:sn-graph-wide-integrity`: exact structural-source reconciliation after the graph settles. |

These rows are not genuine orphans because each has live structural children;
they are provenance residue and remain visible rather than being folded into the
orphan count.

## Owner/geometry residue: 5 of 5 accounted

| Source still on old target | Recorded refusal or routed next action |
|---|---|
| `dd:b_field_non_axisymmetric/time_slice/field_map/grid/phi` | Refused by grammar authority: `ParseError: residue 'toroidal_coordinate_of_field_map_grid' does not match any physical_base or geometry_carrier; nearest candidates: ['toroidal_flux_coordinate_gradient', 'toroidal_flux_coordinate']`. `imas-codex:sn-graph-wide-integrity` owns semantic adjudication of this discretization coordinate before any retarget. |
| `dd:spectrometer_visible/channel/active_spatial_resolution/centre/phi` | `imas-codex:sn-graph-wide-integrity`: exact signed migration to accepted, valid `toroidal_coordinate_of_active_spatial_resolution_zone` |
| `dd:spectrometer_visible/channel/detector/centre/phi` | `imas-codex:sn-graph-wide-integrity`: exact signed migration to accepted, valid `toroidal_coordinate_of_detector` |
| `dd:spectrometer_visible/channel/polarizer/centre/phi` | `imas-codex:sn-graph-wide-integrity`: exact signed migration to accepted, valid `toroidal_coordinate_of_polarizer` |
| `dd:spectrometer_x_ray_crystal/channel/reflector/centre/phi` | Refused by the live attachment guard: `distinct-vector conflict`. `imas-codex:sn-graph-wide-integrity` owns separate semantic review of reflector surface center versus sphere center of curvature before a new exact identity is authorized. |

The other **23 of 28** live producers of
`toroidal_angle_of_measurement_position` are genuine measurement-position
coordinates and are retained by current authority; they are not outstanding
owner/geometry defects:

1. `dd:barometry/gauge/position/phi`
2. `dd:ece/channel/position/phi`
3. `dd:equilibrium/time_slice/constraints/j_parallel/position/phi`
4. `dd:equilibrium/time_slice/constraints/j_phi/position/phi`
5. `dd:equilibrium/time_slice/constraints/n_e/position/phi`
6. `dd:equilibrium/time_slice/constraints/pressure/position/phi`
7. `dd:equilibrium/time_slice/constraints/pressure_rotational/position/phi`
8. `dd:equilibrium/time_slice/constraints/q/position/phi`
9. `dd:ic_antennas/antenna/module/current/position/phi`
10. `dd:ic_antennas/antenna/module/pressure/position/phi`
11. `dd:ic_antennas/antenna/module/voltage/position/phi`
12. `dd:langmuir_probes/embedded/position/phi`
13. `dd:langmuir_probes/reciprocating/plunge/collector/position/phi`
14. `dd:magnetics/b_field_phi_probe/position/phi`
15. `dd:magnetics/b_field_pol_probe/position/phi`
16. `dd:magnetics/flux_loop/position/phi`
17. `dd:magnetics/rogowski_coil/position/phi`
18. `dd:reflectometer_fluctuation/channel/doppler/position/phi`
19. `dd:reflectometer_fluctuation/channel/fluctuations_level/position/phi`
20. `dd:reflectometer_profile/position/phi`
21. `dd:spi/injector/optical_pellet_diagnostic/position/phi`
22. `dd:summary/heating_current_drive/nbi/position/phi/value`
23. `dd:thomson_scattering/channel/position/phi`

## Integrity ratchets

The exact focused command was:

```text
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync --env-file /home/ITER/mcintos/Code/imas-codex/.env pytest -p no:cacheprovider -m graph tests/graph/test_sn_integrity_ratchets.py -vv
```

| Ratchet test | Live count | Result |
|---|---:|---|
| Multiple live targets do not exceed baseline | 23 | PASSED |
| Stale sources with live bindings do not exceed baseline | 3 | PASSED |
| Genuine accepted/valid orphans do not exceed baseline | 20 | PASSED |
| Explicit-axis sources do not also bind generic parents | 0 | PASSED |
| **Focused file total** | — | **4 passed, 0 failed** |

The run exited **0** in **4.72 s** with two warnings. Its complete log is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T142729794531-closecensus/integrity-ratchets.log`;
the captured exit status is in `integrity-ratchets.exit` beside it.

## Closure statement

This census closes measurement, not the four nonzero populations. It accounts
for **100%** of the current rows in every tracked class, preserves every
fail-closed refusal, and routes every unrefused remainder to a named owning
plan. The graph remained unchanged throughout the census and test run.
