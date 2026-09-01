# Chain-cap source-binding round-trip audit

**Outcome:** the five bindings applied by run
`r-20260901T160441383261-n-sourceattach` regenerate exactly in **2/5 cases
(40.0%)**. A fixed control sample of twelve long-standing, DD-grounded
chain-cap bindings regenerates exactly in **6/12 cases (50.0%)**. The new
cohort is therefore 10 percentage points below the baseline, but the baseline
also demonstrates that a single compose draw is not an identity oracle: exact
matches are strong positive evidence, while divergences require semantic and
lineage inspection before any rename.

The five decisions are: **keep 2, rename 2, unbind 1**. No graph change was
made by this audit.

## Method and execution proof

The population was fixed before any model response:

- New cohort: the five exact StandardName/DD-path pairs recorded by
  `r-20260901T160441383261-n-sourceattach`.
- Control: the first twelve distinct identities in alphabetical order from the
  live current-DD, non-stale, DD-grounded `chain_length >= 3` population,
  excluding the new run's change receipts. Twelve were used rather than the
  required minimum of ten.
- Positive controls: all **17/17** selected pairs have a live
  `IMASNode-[:HAS_STANDARD_NAME]->StandardName` binding; all have
  `chain_length = 3`; all **17/17** DD paths were found by targeted extraction.

Each path was then run as a singleton through the production pool function
`compose_batch`, supplied with the same item shape as
`claim_generate_name_batch`. Targeted DD extraction used
`extract_specific_paths(..., write_side_effects=False)`. The live path retained
the complete production sequence: compose-context construction, grammar
context, domain vocabulary, reviewer themes, scored examples, DD and IDS
enrichment, nearby-name lookup, the full `sn/generate_name_system` and
`sn/generate_name_dd` templates, the configured structured response schema,
grammar round-trip, attachment-consistency checks, and inline audits.

Only the ownership read-back and final mutation boundaries were replaced with
in-memory capture, so the test could observe the exact candidate or attachment
without claiming or changing live sources. This is a no-write execution of the
production compose path, not a reduced prompt or a reimplementation of its
logic.

Execution facts:

- Model: `hosted_vllm/deepseek-v4-flash`
- Endpoint class: `local-free`; configured endpoint
  `http://98dci4-gpu-0003:18800/v1`
- Reasoning effort: `medium`
- Calls: **17**, one per selected binding; errors: **0**
- Prompt evidence: every call carried the same **165,914-character** full
  system prompt, SHA-256
  `f98363e1e37dc3b6d978d4c5289aa908497f67721b3e35418d7e39a4be1cdb1c`;
  per-path user prompts ranged from **44,763 to 53,105 characters** and had
  distinct hashes.
- Tokens reported by the endpoint: **938,250 input**, **37,096 output**.
- Attributable returned-call cost: **USD 0.000000**; every one of the 17 calls
  reported `0.0`, and every per-call budget manager reported `spent = 0.0`.
- Live attributable-cost positive control: `LLMCost {for_run:
  'r-20260901T163658621197-n-roundtripaudit'}` was **0 rows / USD 0.000000**
  before and after.
- Live no-mutation census before = after: **4,675 StandardName**, **9,678
  StandardNameSource**, **8,707 StandardNameChange**, **35,180 LLMCost**.

## Five new bindings: verdict and required disposition

Token differences below use underscore-delimited identity tokens. `insert`,
`delete`, and `replace` are relative to the currently bound identity.

| Bound identity | DD path | Production composer output | Verdict and differing tokens | Disposition |
|---|---|---|---|---|
| `electron_temperature_at_separatrix` | `summary/local/separatrix_average/t_e/value` | `flux_surface_averaged_electron_temperature_at_separatrix` | **DIVERGENT** — insert `flux`, `surface`, `averaged` before `electron` | **Rename to composer output.** The DD node is explicitly `separatrix_average`; the current name drops the aggregation. The output passed grammar and inline audits and is not already present, so this must go through the sanctioned edit/review path rather than a direct graph rename. |
| `ion_species_particle_flux_at_wall_due_to_surface_emission` | `wall/description_ggd/ggd/particle_fluxes/ion/emitted/values` | `ion_particle_flux_at_wall_due_to_surface_emission` | **DIVERGENT** — delete `species` | **Unbind.** The output is not a safe rename target: it already exists as a `superseded` identity, and live lineage is `ion_particle_flux...` -> `total_ion_particle_flux...` -> the currently accepted `ion_species_particle_flux...`. Renaming would reverse two reviewed refinements. Remove this inexact binding and route the path through governed regeneration/adjudication so lineage, per-species meaning, and the GGD values representation are resolved together. |
| `launched_power_of_ion_cyclotron_heating_antenna` | `ic_antennas/antenna/module/power_launched` | `launched_power_of_ion_cyclotron_heating_antenna` | **EXACT** — no differing tokens | **Keep.** The composer selected the existing identity as an attachment and explicitly treated `module` as an aggregation level inside the antenna device, not a distinct observable. |
| `normalized_toroidal_flux_coordinate_at_ece_channel_emission_position` | `ece/t_radiation_central/rho_tor_norm` | `normalized_toroidal_flux_coordinate_at_measurement_position` | **DIVERGENT** — replace `ece`, `channel`, `emission` with `measurement` | **Rename/fold to composer output.** The composer selected the already accepted measurement-position identity; it is already grounded by ECE and other measurement-position paths, including `ece/channel/position/rho_tor_norm` and `ece/t_radiation_central_x/rho_tor_norm`. Use the sanctioned edit/consolidation path so the existing identity collision is handled deliberately. |
| `poloidal_magnetic_field_of_magnetic_field_probe` | `magnetics/b_field_pol_probe/non_linear_response/b_field_non_linear` | `poloidal_magnetic_field_of_magnetic_field_probe` | **EXACT** — no differing tokens | **Keep.** The composer selected the existing identity as an attachment and treated non-linear response as probe calibration applied to the same measured field, not as a distinct observable qualifier. |

New-cohort pass rate: **2 exact / 5 = 40.0%**. The three divergences are not
equivalent: aggregation omission supports a new identity, measurement-position
wording supports a fold onto an accepted identity, and the ion-flux divergence
points backward into superseded lineage and therefore requires unbinding rather
than rollback.

## Long-standing grounded control

| Bound identity | DD path | Production composer output | Verdict and differing tokens |
|---|---|---|---|
| `breakdown_initial_time` | `summary/time_breakdown/value` | `<skipped; no name emitted>` | **DIVERGENT** — delete `breakdown`, `initial`, `time` with no replacement |
| `calibration_polarization_angle_of_optical_element` | `ece/polarizer/polarization_angle` | `polarization_angle_of_polarizer` | **DIVERGENT** — delete `calibration`; replace `optical`, `element` with `polarizer` |
| `difference_of_neutral_beam_doppler_wavelength_and_reference_wavelength_of_spectral_line` | `charge_exchange/channel/bes/doppler_shift` | same as bound identity | **EXACT** — no differing tokens |
| `effective_neutral_energy_diffusion_coefficient` | `plasma_transport/model/ggd/neutral/energy/d/values` | `neutral_energy_diffusion_coefficient` | **DIVERGENT** — delete `effective` |
| `electron_density_at_pellet_path` | `pellets/time_slice/pellet/path_profiles/n_e` | same as bound identity | **EXACT** — no differing tokens |
| `first_local_tangential_width_of_aperture` | `nbi/unit/aperture/radius` | `radius_of_neutral_beam_injector` | **DIVERGENT** — replace `first`, `local`, `tangential`, `width` with `radius`; replace `aperture` with `neutral`, `beam`, `injector` |
| `flux_surface_averaged_inverse_of_major_radius` | `equilibrium/time_slice/profiles_1d/gm9` | same as bound identity | **EXACT** — no differing tokens |
| `flux_surface_averaged_ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_major_radius` | `equilibrium/time_slice/profiles_1d/gm2` | same as bound identity | **EXACT** — no differing tokens |
| `flux_surface_averaged_square_of_toroidal_flux_coordinate_gradient_magnitude` | `equilibrium/time_slice/profiles_1d/gm3` | same as bound identity | **EXACT** — no differing tokens |
| `flux_surface_normal_ion_momentum_flux_due_to_diamagnetic_drift` | `core_transport/model/profiles_1d/ion/momentum/diamagnetic/flux` | `ion_diamagnetic_momentum_flux` | **DIVERGENT** — delete `flux`, `surface`, `normal`; insert `diamagnetic` after `ion`; delete `due`, `to`, `diamagnetic`, `drift` |
| `flux_surface_normal_plasma_momentum_diffusivity` | `edge_transport/model/ggd/momentum/d_radial/values` | `radial_total_momentum_diffusion_coefficient` | **DIVERGENT** — replace `flux`, `surface`, `normal`, `plasma` with `radial`, `total`; replace `diffusivity` with `diffusion`, `coefficient` |
| `initial_spun_twist_phase_of_fiber_optic_current_sensor` | `focs/fibre_properties/spun_initial_azimuth` | same as bound identity | **EXACT** — no differing tokens |

Control pass rate: **6 exact / 12 = 50.0%**. The control prevents the five-row
result from being overread: this production composer draw exactly recovered
only half of accepted, grounded chain-cap identities. It is suitable as the
plan's strict binding challenge, but a divergent string alone does not prove
that the composer output is a safe replacement. Lineage and semantic inspection
remain mandatory, as the ion-flux case demonstrates.

## Data-quality follow-on outside this node's write fence

The exact poloidal-field DD path currently matches two `StandardNameSource`
nodes by `source_id`. The active node is
`dd:magnetics/b_field_pol_probe/non_linear_response/b_field_non_linear`, status
`attached`. A stale node with id
`dd:magnetics/bpol_probe/non_linear_response/b_field_non_linear` carries
`dd_lifecycle_status='removed'`, has the newer path copied into `source_id`, and
remains `extracted`. This did not change the compose result because targeted
extraction resolves the current `IMASNode`, but the stale duplicate can be
claimable and should be reconciled by a separately authorized graph repair.

## Final gate statement

The audit measure is met: **5/5 new bindings and 12/12 fixed controls were run
through the full production compose path on the configured local-free model;
17/17 rows report bound identity, exact DD path, emitted identity or explicit
skip, token-level exact/divergent verdict; the separate rates are 40.0% and
50.0%; attributable spend is USD 0.000000; and every new binding has an explicit
keep, rename, or unbind disposition.**
