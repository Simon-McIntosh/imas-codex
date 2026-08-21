# Governed orphan source attachment receipt

## Outcome

One applying invocation re-read the pinned 31-row candidate authority, loaded
the graph's configured DD version and resolution manifest, re-read the complete
live source/name/DD closure, and derived the deterministic maximal
refusal-free cohort inside its transaction. It applied **15** exact source
attachments and wrote **15** per-row `StandardNameChange` receipts. The signed
manifest is
`6fb792666dd575470a012d56840bbc001a63dbcf9e137e0fd5bbc720586830de`.

The live genuine-orphan census fell from **35 to 20**, exactly **15**, matching
both `changed=15` and `receipt_rows=15`. The full untouched source closure
contained **9,607 rows**; `untouched_changed=0`, with identical before/after
aggregate SHA-256
`b17d61ce9aeb2947caa68cf4a75ccf59cafdef88d930bf03d9790b1509f2087d`.
No LLM call was made: the `LLMCost` census remained **27,614**.

The same signed authority was then replayed read-only. Its outcome was
`already_applied`, with `changed=0`, `persistent_writes=0`, and the same 15
immutable receipt rows. The replay re-measured the untouched closure at 9,607
rows with the same aggregate digest and `untouched_changed=0`.

Authority was DD **4.1.1**. Unit comparisons used the exact DD path, configured
DD version, and the graph resolution manifest whose digest was
`sha256:65a7ad8b1f1af0be59891f9dd84e506f292dfaa66930fe29601914882cdf9838`.
The pinned candidate artifact SHA-256 was
`6c2fa944e2e5aacd1189f23d78022279fc459461d48bcf2714be68e7eb6a4821`.

## Applied cohort

The applying transaction selected the cohort from live state rather than from
a declared count. It performed seven exact retargets, attached two existing
unbound sources, and created then attached six pinned DD sources.

| Row | Action | Exact DD source | Prior identity | Attached identity |
|---|---|---|---|---|
| 03 | retarget | `dd:ic_antennas/antenna/module/phase_forward` | `wave_phase_of_ion_cyclotron_heating_antenna` | `forward_wave_phase_of_ion_cyclotron_heating_antenna` |
| 04 | retarget | `dd:charge_exchange/channel/spectrum/processed_line/radiance` | `photon_radiance_at_spectral_line` | `impurity_ion_photon_radiance_of_spectral_line_due_to_charge_exchange` |
| 06 | retarget | `dd:equilibrium/time_slice/profiles_1d/b_field_min` | `minimum_magnetic_field` | `minimum_magnetic_field_magnitude` |
| 10 | attach existing unbound source | `dd:core_profiles/profiles_1d/j_ohmic` | — | `parallel_current_density_due_to_ohmic_current_drive` |
| 14 | create and attach pinned source | `dd:plasma_transport/model/ggd/neutral/state/particles/v_pol` | — | `poloidal_neutral_state_particle_convection_velocity` |
| 16 | create and attach pinned source | `dd:edge_transport/model/ggd/electrons/particles/d_radial` | — | `radial_effective_electron_diffusivity` |
| 17 | create and attach pinned source | `dd:plasma_transport/model/ggd/ion/particles/d_radial` | — | `radial_effective_ion_diffusivity` |
| 18 | create and attach pinned source | `dd:edge_transport/model/ggd/neutral/particles/d_radial` | — | `radial_effective_neutral_diffusivity` |
| 19 | create and attach pinned source | `dd:plasma_transport/model/ggd/ion/state/energy/d_radial` | — | `radial_thermal_ion_charge_state_energy_diffusion_coefficient` |
| 22 | retarget | `dd:distributions/distribution/global_quantities/collisions/ion/state/torque_thermal_phi` | `thermal_ion_charge_state_torque_due_to_collisions` | `toroidal_thermal_ion_charge_state_torque_due_to_collisions` |
| 23 | retarget | `dd:distributions/distribution/profiles_2d/collisions/ion/torque_thermal_phi` | `toroidal_thermal_ion_torque_density_due_to_collisions` | `toroidal_thermal_ion_torque_density_due_to_thermalization` |
| 24 | attach existing unbound source | `dd:distributions/distribution/profiles_2d/trapped/collisions/ion/state/torque_fast_tor` | — | `toroidal_trapped_fast_ion_charge_state_torque_density_due_to_collisions` |
| 25 | retarget | `dd:interferometer/channel/path_length_variation` | `length_of_interferometer_beam` | `variation_of_length_of_interferometer_beam` |
| 26 | retarget | `dd:equilibrium/time_slice/profiles_1d/gm7` | `toroidal_flux_coordinate_gradient_magnitude` | `flux_surface_averaged_toroidal_flux_coordinate_gradient_magnitude` |
| 28 | create and attach pinned source | `dd:edge_transport/model/ggd/neutral/state/particles/v_pol` | — | `poloidal_neutral_internal_state_convection_velocity` |

## Refusal analysis

The transaction refused **16 of 31 rows (51.6%)**. That rate is a material
finding, but it does not overturn the candidate artifact's 31 unit agreements:
the path-aware canonical unit and graph-resolution checks accepted all 31 unit
pairs. Instead, nine identities were not yet accepted, one was quarantined,
three moves would have orphaned their predecessor, one source was already
applied, one source had conflicting fan-out, and one candidate failed the live
state-resolution semantic guard. Thus the candidate artifact's unit column was
not optimistic; its `ATTACH` disposition was optimistic for row 12 and, for the
other refused rows, did not constitute lifecycle or mutation authority.

| Refusal class | Rows | Count | Interpretation |
|---|---|---:|---|
| Target lifecycle admission | 02, 05, 07, 08, 13, 15, 20, 21, 29 | 9 | A candidate identity cannot receive authoritative provenance before name review accepts it. |
| Last-producing-source closure | 01, 09, 27 | 3 | Retargeting would turn the predecessor into a new genuine orphan. |
| Already applied | 11 | 1 | The exact signed target already owns the source and its prior migration receipt. |
| Semantic state resolution | 12 | 1 | The DD path is state-resolved while the proposed identity is species-level. |
| Target validation | 30 | 1 | A quarantined identity cannot receive authoritative provenance. |
| Source fan-out | 31 | 1 | The source has two current targets, so an exact single-predecessor retarget is not authorized. |

Every refusal below reproduces the transaction's guard reason verbatim.

| Row | Standard name | Verbatim guard reason |
|---|---|---|
| 01 | `capacitance_of_ion_cyclotron_heating_antenna` | `target would lose its last producing source` |
| 02 | `cross_section_of_flux_surface` | `target lifecycle is not accepted: name_stage='pending'` |
| 05 | `line_integrated_electron_density` | `target lifecycle is not accepted: name_stage='drafted'` |
| 07 | `minimum_of_safety_factor` | `target lifecycle is not accepted: name_stage='reviewed'` |
| 08 | `neutral_state_power_density` | `target lifecycle is not accepted: name_stage='reviewed'` |
| 09 | `neutron_flux_due_to_fusion` | `target would lose its last producing source` |
| 11 | `parallel_mach_number` | `source is already attached to the signed target; prior_receipts=[{'id': 'sn-change:source-migration:7ba4a47a500fb7154fbc9ba0b06353df633a7ccf744231758f3d46c3ab11e2ad', 'operation': 'source_migration_manifest'}]` |
| 12 | `parallel_neutral_momentum_diffusion_coefficient` | `state-resolution mismatch: path 'plasma_transport/model/profiles_1d/neutral/state/momentum/d_parallel' is state-resolved but SN 'parallel_neutral_momentum_diffusion_coefficient' is species-level` |
| 13 | `poloidal_neutral_internal_state_momentum_convected_velocity` | `target lifecycle is not accepted: name_stage='reviewed'` |
| 15 | `poloidal_straight_field_line_angle` | `target lifecycle is not accepted: name_stage='drafted'` |
| 20 | `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | `target lifecycle is not accepted: name_stage='reviewed'` |
| 21 | `toroidal_line_integrated_impurity_ion_velocity` | `target lifecycle is not accepted: name_stage='drafted'` |
| 27 | `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | `target would lose its last producing source` |
| 29 | `magnetic_field_at_pedestal_top_low_field_side_magnitude` | `target lifecycle is not accepted: name_stage='drafted'` |
| 30 | `tendency_of_total_thermal_plasma_internal_energy` | `target validation is not valid: validation_status='quarantined'` |
| 31 | `toroidal_neutral_state_momentum_diffusivity` | `source has multiple live targets: ['toroidal_momentum_diffusivity', 'toroidal_neutral_internal_state_momentum_diffusion_coefficient']` |

## Receipt invariants

| Measure | Apply | Replay |
|---|---:|---:|
| Outcome | `applied` | `already_applied` |
| Changed | 15 | 0 |
| Receipt rows | 15 | 15 |
| Persistent writes | transaction-owned | 0 |
| Genuine orphans | 35 → 20 | 20 → 20 |
| Untouched source rows | 9,607 | 9,607 |
| Untouched changed | 0 | 0 |
| Untouched aggregate SHA-256 | `b17d61ce9aeb2947caa68cf4a75ccf59cafdef88d930bf03d9790b1509f2087d` | `b17d61ce9aeb2947caa68cf4a75ccf59cafdef88d930bf03d9790b1509f2087d` |

The machine-readable apply and replay receipts are retained in the node run
envelope as `apply-receipt.json` and `replay-receipt.json`, alongside the full
transaction log. These files bind the complete per-row post-state and receipt
properties to the same manifest hash.
