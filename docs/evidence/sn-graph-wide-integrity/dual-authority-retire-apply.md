# Atomic dual-authority retirement production receipt

## Outcome

The production transaction completed successfully on 2026-08-20. One operator
invocation derived both mutation authorities from their committed signed
artifacts, reconciled 19 source records, released exactly 20 `PRODUCED_NAME`
bindings and their 20 matching `HAS_STANDARD_NAME` projections, superseded
exactly 16 signed identities, and wrote exactly one `StandardNameChange` ledger
row for each retired identity. No caller supplied a source, binding, or target
list.

An immediate invocation with the same signed artifacts returned
`already_applied`, `changed=0`, and `persistent_writes=0`. A separate read-only
program then opened a fresh graph connection, independently re-derived the
19-source / 20-binding / 16-target authority join from the two committed
artifacts, and re-read every affected graph closure.

## Signed authority and manifest identity

The source-selection authority was
`catalog-edit-dual-binding-adjudication.json`, with file SHA-256
`5ca7761a7b022ac7889387d7bf63a027114a168cc3785ed4fdc8d31c08417b6e`
and signed-payload SHA-256
`c227e70ec5cd940577ca778ce5ec63e4df3a63bf68c3e845eba92d0a4b9a0efb`.
The retirement authority was
`refused-target-orphan-adjudication.json`, with file SHA-256
`2c2d38f3241ec3057d24a5d05c27840f5e4ffe99520063059ab31c1e9d4bca36`
and canonical-content SHA-256
`4bac6110486390e95c1cab9620c4723df96fe6f2190b85e6496464c77fbba873`.

The operator's closure-sensitive manifest SHA-256 is
`28736a30fa250c58a07a30f5ad986703818b1cd8b59174a57ee4501e1a866b17`.
The independently serialized canonical JSON digest is
`17252983b12597b9eff395c5b790dadc7729018294d8c1d0d39d5c878a2d1ab4`.
The preview and the transaction-lock re-read compared equal by canonical JSON
digest; no native object equality was used as mutation authority.

## Exact released closures

The following table is the complete 20-pair release set. Each row had one live
`PRODUCED_NAME` binding and one backing `HAS_STANDARD_NAME` projection before
the transaction, and zero of each afterward. Each source retains exactly one
binding and one projection to the signed survivor shown here; its scalar
`produced_sn_id` also equals that survivor.

| Source record | Superseded identity | Signed survivor |
|---|---|---|
| `dd:balance_of_plant/power_electric_plant_operation/system/power` | `net_absorbed_power_of_plant_system` | `power_of_plant_system` |
| `dd:bremsstrahlung_visible/channel/intensity` | `bremsstrahlung_count` | `bremsstrahlung_count_at_detector_pixel` |
| `dd:bremsstrahlung_visible/channel/intensity` | `time_derivative_of_bremsstrahlung_count_at_detector_pixel` | `bremsstrahlung_count_at_detector_pixel` |
| `dd:camera_ir/optical_element/material_properties/roughness` | `surface_roughness_of_optical_element` | `roughness_of_optical_element` |
| `dd:camera_visible/channel/optical_element/material_properties/roughness` | `surface_roughness_of_optical_element` | `roughness_of_optical_element` |
| `dd:core_sources/source/profiles_1d/ion/momentum/radial` | `radial_plasma_momentum_source` | `radial_ion_momentum_source` |
| `dd:edge_transport/model/ggd/momentum/v/diamagnetic` | `bulk_plasma_velocity_due_to_diamagnetic_drift_magnitude` | `plasma_velocity_due_to_diamagnetic_drift` |
| `dd:edge_transport/model/ggd/neutral/state/particles/d_pol/values` | `poloidal_particle_diffusivity` | `poloidal_neutral_state_particle_diffusivity` |
| `dd:mhd_linear/time_slice/toroidal_mode/plasma/phi_potential_perturbed/imaginary` | `perturbed_electrostatic_potential` | `perturbed_electrostatic_potential_imaginary_part` |
| `dd:nbi/unit/source/surface` | `beam_area_of_neutral_beam_injector` | `area_of_neutral_beam_injector` |
| `dd:ntms/time_slice/mode/dphase_dt` | `rotation_frequency` | `rotation_frequency_of_neoclassical_tearing_mode` |
| `dd:plasma_profiles/ggd/j_diamagnetic/r` | `radial_current_density_due_to_diamagnetic_drift` | `radial_diamagnetic_current_density` |
| `dd:plasma_profiles/ggd/j_diamagnetic/z` | `vertical_current_density_due_to_diamagnetic_drift` | `vertical_diamagnetic_current_density` |
| `dd:plasma_profiles/ggd/j_total/poloidal` | `poloidal_current_density` | `poloidal_total_current_density` |
| `dd:plasma_transport/model/ggd/electrons/energy/flux/values` | `radial_total_thermal_electron_energy_flux` | `energy_flux` |
| `dd:plasma_transport/model/ggd/ion/energy/flux/values` | `radial_total_thermal_electron_energy_flux` | `energy_flux` |
| `dd:plasma_transport/model/ggd/neutral/energy/flux/values` | `radial_total_thermal_electron_energy_flux` | `energy_flux` |
| `dd:plasma_transport/model/ggd/neutral/energy/v_parallel/values` | `parallel_neutral_energy_convection_velocity` | `parallel_neutral_species_energy_convection_velocity` |
| `dd:spectrometer_visible/channel/optical_element/material_properties/roughness` | `surface_roughness_of_optical_element` | `roughness_of_optical_element` |
| `dd:waves/coherent_wave/profiles_1d/e_field_n_phi/plus/phase` | `per_toroidal_mode_left_hand_circularly_polarized_electric_field` | `per_toroidal_mode_left_hand_circularly_polarized_wave_electric_field` |

The 16 unique superseded identities are
`beam_area_of_neutral_beam_injector`, `bremsstrahlung_count`,
`bulk_plasma_velocity_due_to_diamagnetic_drift_magnitude`,
`net_absorbed_power_of_plant_system`,
`parallel_neutral_energy_convection_velocity`,
`per_toroidal_mode_left_hand_circularly_polarized_electric_field`,
`perturbed_electrostatic_potential`, `poloidal_current_density`,
`poloidal_particle_diffusivity`,
`radial_current_density_due_to_diamagnetic_drift`,
`radial_plasma_momentum_source`,
`radial_total_thermal_electron_energy_flux`, `rotation_frequency`,
`surface_roughness_of_optical_element`,
`time_derivative_of_bremsstrahlung_count_at_detector_pixel`, and
`vertical_current_density_due_to_diamagnetic_drift`.

## Counter and collateral proof

| Measure | Before | After apply | After replay | Required relation |
|---|---:|---:|---:|---|
| `StandardNameChange` | 7,501 | 7,517 | 7,517 | exactly +16 declared receipt rows |
| manifest receipt rows | 0 | 16 | 16 | exactly one per retired identity |
| `LLMCost` | 27,477 | 27,477 | 27,477 | unchanged |

The complete out-of-allowlist source closure contained 9,520 source rows. Its
aggregate digest was
`b363d5727b9008e95f84047fd77dc4d9b3b7ab200de2b0996cf024e7b8d99b06`
before apply, after apply, and after replay. All 9,520 individual canonical row
digests were identical and the changed-source list was empty.

## Independent postflight

The independent postflight passed every gate from a fresh graph connection:

- all 16 signed identities have both `name_stage=superseded` and
  `status=superseded`;
- all 16 have zero live producers, zero live `HAS_PARENT` children, and exactly
  one ledger row bearing the transaction manifest digest;
- all 20 signed binding pairs and all 20 matching projections are absent;
- all 19 source records select their signed survivor through the scalar mirror,
  one live binding, and one backing projection;
- graph counters remain `StandardNameChange=7517` and `LLMCost=27477`.

The resulting live graph contains 23 source records with more than one live
`PRODUCED_NAME` target, comprising 51 live bindings. There are 101 live
Standard Names with no live producing source: 63 hold at least one live
`HAS_PARENT` child and 38 hold none. The partition is exhaustive
(`63 + 38 = 101`). The unsourced total is unchanged by this transaction because
the 16 retired targets no longer belong to the live-name population.

Representative semantic outcomes retain the signed DD meaning:

| Retired identity and review score | Source path | Surviving identity and description |
|---|---|---|
| `beam_area_of_neutral_beam_injector` (0.925): geometric area of the ion-source emission surface | `nbi/unit/source/surface` | `area_of_neutral_beam_injector`: source cross-section of the active ion-source emission face before neutralization |
| `bremsstrahlung_count` (0.7875): detector photoelectron rate | `bremsstrahlung_visible/channel/intensity` | `bremsstrahlung_count_at_detector_pixel`: detected count rate at one detector pixel, not local emissivity or a time derivative |
| `rotation_frequency` (0.5375): phase derivative of a neoclassical tearing mode | `ntms/time_slice/mode/dphase_dt` | `rotation_frequency_of_neoclassical_tearing_mode` (0.925): angular advance or regression of that magnetic island's helical phase |
| `time_derivative_of_bremsstrahlung_count_at_detector_pixel` (0.85): description actually states a detected count rate | `bremsstrahlung_visible/channel/intensity` | `bremsstrahlung_count_at_detector_pixel`: the non-derivative detector-pixel quantity encoded by the DD path |

## Durable artifacts

- `production-dual-authority-retirement.log`: complete preview, gated apply,
  replay, counter, closure-digest, and census log; exit 0.
- `dual-authority-retirement-preview.json`: exact closure-sensitive preview.
- `dual-authority-retirement-baseline.json`: pre-transaction counters and graph
  closures.
- `dual-authority-retirement-apply-receipt.json`: atomic apply receipt.
- `dual-authority-retirement-replay-receipt.json`: idempotent replay receipt.
- `dual-authority-retirement-postflight.json`: in-process postflight and
  collateral proof.
- `independent-postflight.log`: fresh-connection read-only verification; exit 0.
- `dual-authority-retirement-independent-postflight.json`: complete independent
  target, source, pair, counter, and census re-read.
