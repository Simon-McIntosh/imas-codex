# Catalog-edit source-disposition instrument and signed preview

## Outcome

The exact source-disposition instrument in
`imas_codex/standard_names/graph_ops.py` now signs each removed target's global
incoming `PRODUCED_NAME` closure and refuses a disposition when the proposed
batch would remove that target's final live producing source. The regenerated
write-free production preview covered all **216** adjudicated rows and returned
`refused`: **138 admitted + 78 fail-closed refused rows = 216 requested**.

The 78 refused source rows protect **89 distinct targets**, exactly matching the
independent pre-implementation last-binding measurement. The 138 otherwise
admitted rows would change **51** scalars and remove **138** exact
`PRODUCED_NAME` bindings plus **138** exact `HAS_STANDARD_NAME` projections.
Because this instrument is atomic, the presence of any refusal prevents every
row from applying; no partial mutation or provider call occurred.

## Signed authority and global closure

The committed adjudication artifact remains byte-for-byte unchanged at
`docs/evidence/sn-graph-wide-integrity/catalog-edit-dual-binding-adjudication.json`:

- file SHA-256:
  `5ca7761a7b022ac7889387d7bf63a027114a168cc3785ed4fdc8d31c08417b6e`;
- declared canonical payload SHA-256:
  `c227e70ec5cd940577ca778ce5ec63e4df3a63bf68c3e845eba92d0a4b9a0efb`;
- canonical signed row-set SHA-256:
  `0cb907d04aeb33b46e1f8ede9b5927d2c574f3500200750c8bc4734e6f8633bf`;
- regenerated live preview manifest SHA-256:
  `e475dfaad8e736c8d27c262b11eb317a69ee7024dc186f787250853dab4090a1`.

The preview signs **151 removed-target closures containing 676 global incoming
bindings**. Each closure includes the target element identity and properties,
every incoming producing source's element identity and properties, and every
incoming `PRODUCED_NAME` relationship's element identity and properties. Apply
locks all of those source nodes, target nodes, and relationships, re-reads the
complete closure, and requires its manifest digest to remain byte-identical
before deleting anything. Stale source nodes remain signed as part of the
complete closure but do not count as live producers. A post-delete invariant
independently requires every removed target to retain at least one non-stale
incoming producing source before the transaction may commit.

The adjudication's outer payload and per-row signatures continue to use sorted
compact JSON with standard ASCII escaping. The dedicated signature verifier and
its known-good digest regression are unchanged; `_authority_payload_hash` and
its existing callers remain unchanged.

## Refused targets

Every target refused by the regenerated preview is named below. Multiple target
losses can belong to one refused source row, which is why 78 refused rows protect
89 targets.

1. `average_external_magnetic_flux`
2. `beam_area_of_neutral_beam_injector`
3. `bremsstrahlung_count`
4. `bulk_plasma_velocity_due_to_diamagnetic_drift_magnitude`
5. `co_passing_fast_electron_torque_density_due_to_collisions`
6. `co_passing_torque_density`
7. `counter_passing_thermal_ion_torque_density_due_to_collisions`
8. `counter_passing_torque_density`
9. `critical_electric_field`
10. `current_density_due_to_wave_driven_current_drive`
11. `current_due_to_wave_driven_current_drive`
12. `current_weighted_average_external_magnetic_flux`
13. `decay_length_over_scrape_off_layer`
14. `electron_torque_density_due_to_coulomb_collisions_with_electrons`
15. `electrostatic_potential_imaginary_part`
16. `electrostatic_potential_real_part`
17. `ethylene_count`
18. `external_magnetic_flux`
19. `fast_electron_torque_due_to_collisions`
20. `fast_particle_torque_due_to_j_cross_b_force`
21. `fast_torque_due_to_collisions`
22. `fluctuating_ion_current_density`
23. `hydrogen_density`
24. `ion_torque_density`
25. `ion_torque_due_to_collisions`
26. `linear_mhd_mode_reference_phase`
27. `mass_density`
28. `mhd_mode_reference_phase`
29. `mode_reference_phase`
30. `momentum_source`
31. `net_absorbed_power_of_plant_system`
32. `neutral_species_energy_convection_velocity`
33. `neutral_state_particle_diffusivity`
34. `neutral_torque_density`
35. `neutron_source_rate_due_to_thermal_fusion`
36. `normalized_gyrocenter_perturbed_current_density`
37. `normalized_particle_perturbed_current_density`
38. `normalized_particle_perturbed_energy`
39. `normalized_perturbed_current_density`
40. `outer_squareness_of_flux_surface`
41. `parallel_neutral_energy_convection_velocity`
42. `parallel_normalized_perturbed_current_density`
43. `particle_torque_due_to_j_cross_b_force`
44. `per_toroidal_mode_current_density_due_to_wave_driven_current_drive`
45. `per_toroidal_mode_left_hand_circularly_polarized_electric_field`
46. `perpendicular_normalized_perturbed_pressure`
47. `perturbed_electrostatic_potential`
48. `plasma_mass_density_imaginary_part`
49. `plasma_mass_density_real_part`
50. `plasma_momentum`
51. `plasma_momentum_diffusion_coefficient`
52. `plasma_momentum_source`
53. `plasma_pressure_imaginary_part`
54. `plasma_pressure_real_part`
55. `plasma_temperature_imaginary_part`
56. `plasma_temperature_real_part`
57. `poloidal_current_density`
58. `poloidal_particle_diffusivity`
59. `power_due_to_recombination`
60. `pressure_over_scrape_off_layer`
61. `radial_current_density_due_to_diamagnetic_drift`
62. `radial_plasma_momentum_source`
63. `radial_total_thermal_electron_energy_flux`
64. `reference_magnetic_field`
65. `reference_phase`
66. `rotation_frequency`
67. `runaway_electron_convection_velocity`
68. `runaway_electron_diffusivity`
69. `runaway_electron_particle_flux`
70. `source_rate_due_to_thermal_fusion`
71. `surface_roughness_of_optical_element`
72. `thermal_electron_torque_density_due_to_coulomb_collisions_with_electrons`
73. `thermal_electron_torque_due_to_collisions`
74. `thermal_ion_torque_due_to_collisions`
75. `time_derivative_of_bremsstrahlung_count_at_detector_pixel`
76. `torque_density_due_to_coulomb_collisions_with_electrons`
77. `torque_due_to_j_cross_b_force`
78. `torque_due_to_neutral_beam_shinethrough`
79. `total_ion_energy_convection_velocity`
80. `total_momentum_flux`
81. `total_plasma_momentum`
82. `trapped_fast_ion_torque_density_due_to_collisions`
83. `trapped_thermal_electron_torque_density_due_to_collisions`
84. `trapped_thermal_electron_torque_density_due_to_coulomb_collisions_with_electrons`
85. `trapped_torque_density`
86. `vertical_current_density_due_to_diamagnetic_drift`
87. `vertical_neutral_state_momentum_flux_limiter_coefficient`
88. `volume_averaged_runaway_electron_current_density`
89. `wetted_area_of_langmuir_probe`

Representative refused bindings show why row count and target count differ:

- `dd:bremsstrahlung_visible/channel/intensity` would remove the final bindings
  for both `bremsstrahlung_count` and
  `time_derivative_of_bremsstrahlung_count_at_detector_pixel`;
- `dd:core_sources/source/profiles_1d/ion/momentum/radial` would remove the final
  bindings for `momentum_source`, `plasma_momentum_source`, and
  `radial_plasma_momentum_source`;
- `dd:balance_of_plant/power_electric_plant_operation/system/power` would remove
  the final binding for `net_absorbed_power_of_plant_system`;
- the camera material-roughness source rows would collectively remove the final
  binding for `surface_roughness_of_optical_element`.

These are safety refusals, not semantic reversals of the signed adjudication.
The 89 targets require separate lifecycle or replacement-source authority before
their source dispositions can become admissible.

## Out-of-allowlist immutability and counter proof

The preview allowlist contains exactly the 216 signed source IDs. The complete
participant closure for every other live `StandardNameSource` was read before
and after the preview:

| Measure | Before | After | Verdict |
|---|---:|---:|---|
| Out-of-allowlist source rows | 9,282 | 9,282 | identical population |
| Out-of-allowlist closure SHA-256 | `b9864539a6d64c523a1259522f6c7ab1a0adf63c1a9f46a238b66f9c7e819f24` | `b9864539a6d64c523a1259522f6c7ab1a0adf63c1a9f46a238b66f9c7e819f24` | immutable |
| `StandardNameChange` nodes | 7,451 | 7,451 | unchanged |
| `LLMCost` nodes | 27,467 | 27,467 | unchanged |

The unchanged closure digest proves the preview left collateral graph state
outside the allowlist immutable. The instrument's transaction rollback and the
unchanged ledger and cost counts prove it created no source mutation, operation
receipt, or provider spend.

## Regression evidence

The complete focused file executed **8 passed, 0 failed, 0 skipped** against a
disposable Neo4j instance where required. In addition to all previously covered
disposition, replay, claim/scalar drift, incomplete-projection, and signature
cases, the new regressions prove:

1. a disposition whose removed target has exactly one incoming producing source
   is refused without changing graph bytes; and
2. adding an incoming producer after preview changes the signed global closure,
   so apply refuses the stale manifest before mutation.

Durable logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T064817725535-last-binding-closure-guard/disposable-tests-final.log`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T064817725535-last-binding-closure-guard/disposable-neo4j-final.log`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T064817725535-last-binding-closure-guard/live-preview-final.log`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T064817725535-last-binding-closure-guard/last-binding-baseline.log`.

The regenerated preview was verified against live plan version 178, whose
independent HOLD census records the same 151-target closure and 89 last-binding
losses (86 accepted and three reviewed). The adjudication bytes were not
regenerated or re-signed.
