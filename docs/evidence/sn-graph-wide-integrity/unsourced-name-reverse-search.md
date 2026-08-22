# Unsourced-name reverse Data Dictionary search

## Outcome

**PASS — the live cohort was derived in-invocation at 17 names; searched = 17 and unsearched = 0.** The DD-only service received **51 distinct semantic queries** (exactly three per identity). The search responses contained **577 returned result groups**—direct path hits and semantic clusters—and every returned group was inspected. Exact-path cluster follow-ups were also inspected for each leading candidate. The result is **16 candidate-found** and **1 no-candidate**.

This is search evidence, not attachment authority. Unit agreement is necessary but not sufficient: several near matches agree dimensionally while differing in population, owner, process, representation, or source/recipient role. No graph mutation was requested or performed.

## Live cohort and method

The production graph query selected every `StandardName` whose `name_stage` is neither `superseded` nor `exhausted`, with zero incoming `(:StandardNameSource)-[:PRODUCED_NAME]->(name)` relationships and no live `HAS_PARENT` child. It returned 17 unique identities. Pre-search counters were `StandardNameChange=7,787` and `PRODUCED_NAME=5,780`.

For each identity, query 1 expressed the public name, query 2 paraphrased its description, and query 3 stressed the discriminating physics and unit. Search ran against active DD major 4 records. The lists below preserve every returned result group, including duplicates when the search returned the same label in distinct scopes.

## Per-name search record

### `capacitance_of_ion_cyclotron_heating_antenna`

Name unit: `F`. Outcome: **candidate-found** — exact path `ic_antennas/antenna/module/matching_element/capacitance`. Exact-path cluster follow-up: 3 cluster(s): RF Antenna Power Measurements; ICRH Antenna Operational Parameters; IC Antenna System Configuration.

1. Query: `capacitance of ion cyclotron heating antenna matching network element`
   Returned groups inspected (10): 1. Ion Cyclotron Antenna Phase; 2. Electron Cyclotron Antenna Beam; 3. ICRH Antenna Operational Parameters; 4. IC Antenna Module Data; 5. IC Antenna Reflected Power; 6. IC Antenna Phase Measurements; 7. Antenna Reflected Power Data; 8. IC Antenna Toroidal Angles; 9. IC Antenna Phase Data; 10. Electron Cyclotron Heating Power.
2. Query: `non-negative charge-to-voltage proportionality of a lumped impedance-matching element in an ICRF antenna circuit`
   Returned groups inspected (10): 1. IC Antenna Current Amplitudes; 2. IC Antenna Phase Measurements; 3. Antenna Reflection Coefficients; 4. Antenna Reflection Coefficient Timing; 5. Antenna Reflected Power Data; 6. RF Antenna Power Measurements; 7. Parallel Conductivity Profile Values; 8. Fast Electron Density Profile; 9. Line Integrated Electron Density; 10. IC Antenna Strap Geometry.
3. Query: `ion cyclotron antenna RF matching-element capacitance measured in farads`
   Returned groups inspected (10): 1. Ion Cyclotron Antenna Phase; 2. IC Antenna Current Amplitudes; 3. Electron Cyclotron Antenna Beam; 4. ICRH Antenna Operational Parameters; 5. Antenna Reflected Power Data; 6. IC Antenna Phase Measurements; 7. RF Antenna Power Measurements; 8. IC Antenna Phase Data; 9. IC Antenna Radial Positions; 10. Antenna Radial Position Data.

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `ic_antennas/antenna/module/matching_element/capacitance` | `F` | `F` | **AGREE** | Exact antenna matching-element capacitance. |
| 2 | `ic_antennas/antenna/module/coupling_resistance` | `ohm` | `F` | **DISAGREE** | Same RF module, but resistance rather than capacitance. |
| 3 | `ic_antennas/antenna/module/matching_element/phase/data` | `rad` | `F` | **DISAGREE** | Matching-element phase data, not capacitance. |

### `cross_section_of_flux_surface`

Name unit: `m^2`. Outcome: **candidate-found** — exact path `core_profiles/profiles_1d/grid/area`. Exact-path cluster follow-up: 2 cluster(s): Flux Surface Area Profiles; Flux Surface Cross-Sectional Area.

1. Query: `poloidal cross sectional area of magnetic flux surface`
   Returned groups inspected (12): 1. core_profiles/profiles_1d/grid/area (score: 0.55); 2. core_sources/source/profiles_1d/grid/area (score: 0.53); 3. Flux Surface Cross-Sectional Area (0.86); 4. Poloidal Magnetic Flux Profiles (0.84); 5. Poloidal Magnetic Flux Profiles (0.84); 6. Magnetic Flux Surface Volume (0.84); 7. Toroidal Flux Surface Areas (0.83); 8. Toroidal Flux Surface Areas (0.83); 9. Plasma Boundary Flux Area (0.83); 10. External Poloidal Magnetic Flux (0.83); 11. Poloidal A-Field Components (0.83); 12. Toroidal Flux Surface Area (0.83).
2. Query: `area enclosed by a magnetic flux surface in the poloidal plane`
   Returned groups inspected (14): 1. core_profiles/profiles_1d/grid/area (score: 0.92); 2. core_instant_changes/change/profiles_1d/grid/area (score: 0.90); 3. core_transport/model/profiles_1d/grid_flux/psi (score: 0.54); 4. equilibrium/time_slice/global_quantities/area (score: 0.54); 5. Magnetic Flux Surface Volume (0.86); 6. Magnetic Flux Coordinate Definitions (0.85); 7. Poloidal Magnetic Flux Profiles (0.85); 8. Poloidal Magnetic Flux Profiles (0.85); 9. Toroidal Flux Surface Areas (0.85); 10. Toroidal Flux Surface Areas (0.85); 11. Toroidal Flux Surface Area (0.85); 12. Magnetic Surface Enclosed Volume (0.84); 13. External Poloidal Magnetic Flux (0.84); 14. Poloidal Magnetic Flux Grid (0.84).
3. Query: `flux surface cross section in square metres distinct from toroidally swept surface area`
   Returned groups inspected (12): 1. equilibrium/time_slice/profiles_1d/darea_drho_tor (score: 0.52); 2. core_sources/source/profiles_1d/grid/area (score: 0.52); 3. Flux Surface Cross-Sectional Area (0.86); 4. Toroidal Flux Surface Area (0.83); 5. Toroidal Flux Surface Areas (0.82); 6. Toroidal Flux Surface Areas (0.82); 7. Flux Surface Area Profiles (0.82); 8. Flux Surface Area Grid (0.82); 9. Plasma Boundary Flux Area (0.82); 10. Plasma Flux Surface Geometry (0.81); 11. Toroidal Angle and Flux Data (0.81); 12. Flux Surface Averaged Quantities (0.81).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `core_profiles/profiles_1d/grid/area` | `m^2` | `m^2` | **AGREE** | Exact poloidal flux-surface cross-sectional area in the name description. |
| 2 | `core_profiles/profiles_1d/grid/surface` | `m^2` | `m^2` | **AGREE** | Toroidally swept surface specification; dimension agrees but geometry does not. |
| 3 | `equilibrium/time_slice/global_quantities/area` | `m^2` | `m^2` | **AGREE** | LCFS poloidal area only, not the full radial family. |

### `fast_ion_charge_state_power_at_inside_flux_surface`

Name unit: `W`. Outcome: **candidate-found** — exact path `waves/coherent_wave/profiles_1d/ion/state/power_inside_fast`. Exact-path cluster follow-up: 3 cluster(s): Wave Power Absorption Profiles; Coherent Wave Power Deposition; Coherent Wave Field Profiles.

1. Query: `cumulative wave power absorbed by a fast ion charge state inside a selected flux surface`
   Returned groups inspected (10): 1. Ion Energy Flux Profiles (0.84); 2. Ion Particle Flux Profiles (0.83); 3. Poloidal Ion Particle Flux (0.82); 4. Ion Energy Flux Directions (0.82); 5. Total Thermal Ion Pressure (0.82); 6. Radiated Power Inside Flux (0.81); 7. Thermal and Fast Ion Density (0.81); 8. Coherent Wave Ion Charge (0.81); 9. Coherent Wave Ion Charge (0.81); 10. Total Ion Density Profiles (0.81).
2. Query: `fast non-thermal ion charge-state deposited power inside flux surface in watts`
   Returned groups inspected (10): 1. Ion Energy Flux Profiles (0.85); 2. Ion Particle Flux Profiles (0.84); 3. Total Thermal Ion Pressure (0.84); 4. Plasma Particle Energy Fluxes (0.83); 5. Edge Ion Temperature Profiles (0.83); 6. Radiated Power Inside Flux (0.83); 7. Ion Energy Flux Directions (0.83); 8. Plasma Radiation Power Density (0.82); 9. Ion Temperature Profile Data (0.82); 10. Ion Temperature Profile Data (0.82).
3. Query: `ion cyclotron wave power inside flux surface by ion charge state fast versus thermal`
   Returned groups inspected (10): 1. Ion Cyclotron Heating Power (0.86); 2. Electron Cyclotron Heating Power (0.84); 3. Ion Energy Flux Profiles (0.82); 4. Ion Cyclotron Antenna Phase (0.82); 5. Ion Cyclotron Heating Schedule (0.82); 6. Edge Ion Temperature Profiles (0.82); 7. Thermal Ion Pressure Profiles (0.82); 8. Thermal Ion Pressure Profiles (0.82); 9. Coherent Wave Ion Charge (0.81); 10. Coherent Wave Ion Charge (0.81).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `waves/coherent_wave/profiles_1d/ion/state/power_inside_fast` | `W` | `W` | **AGREE** | Exact cumulative absorbed wave power for a fast ion charge state. |
| 2 | `waves/coherent_wave/profiles_1d/ion/state/power_inside_thermal` | `W` | `W` | **AGREE** | Thermal recipient sibling; population disagrees. |
| 3 | `radiation/process/profiles_1d/ion/state/power_inside` | `W` | `W` | **AGREE** | Radiated power inside the surface; mechanism disagrees. |

### `line_integrated_electron_density`

Name unit: `m^-2`. Outcome: **candidate-found** — exact path `interferometer/channel/n_e_line`. Exact-path cluster follow-up: 2 cluster(s): Interferometer Density Line Integration; Interferometer Density Measurement Channels.

1. Query: `line integrated electron density`
   Returned groups inspected (7): 1. equilibrium/time_slice/constraints/n_e_line (score: 0.53); 2. core_profiles/profiles_1d/electrons/density_fit/local (score: 0.52); 3. Line Integrated Electron Density (0.90); 4. Electron Density Profile Data (0.84); 5. Electron Density Profile Data (0.84); 6. Volume Averaged Electron Density Values (0.84); 7. Electron Density Diagnostic Data (0.83).
2. Query: `electron number density integrated along an interferometer line of sight`
   Returned groups inspected (10): 1. Interferometer Density Line Integration (0.85); 2. Line Integrated Electron Density (0.84); 3. Electron Density Profile Data (0.82); 4. Electron Density Profile Data (0.82); 5. Spectral Line Intensity Data (0.81); 6. Pedestal Electron Density Parameters (0.81); 7. Interferometer Density Measurement Channels (0.81); 8. Pedestal Electron Density Profile (0.81); 9. Fast Electron Density Profiles (0.81); 10. Fast Electron Density Profile (0.81).
3. Query: `interferometer or refractometer electron column density in inverse square metres`
   Returned groups inspected (10): 1. Reflectometer Electron Density Profile (0.83); 2. Electron Density Diagnostic Data (0.82); 3. Electron Density Diagnostic Data (0.82); 4. Volume Averaged Electron Density Values (0.82); 5. Interferometer Density Line Integration (0.82); 6. Electron Density Profile Data (0.81); 7. Electron Density Profile Data (0.81); 8. Total Ion Density Summary (0.81); 9. Spectral Line Intensity Data (0.81); 10. Interferometer Density Validity Timeseries (0.81).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `interferometer/channel/n_e_line` | `m^-2` | `m^-2` | **AGREE** | Exact full-line interferometry observable. |
| 2 | `refractometer/channel/n_e_line/data` | `m^-2` | `m^-2` | **AGREE** | Exact quantity in a refractometer data leaf. |
| 3 | `interferometer/channel/n_e_line_average` | `m^-3` | `m^-2` | **DISAGREE** | Line average, not line integral. |

### `minimum_of_safety_factor`

Name unit: `1`. Outcome: **candidate-found** — exact path `equilibrium/time_slice/global_quantities/q_min/value`. Exact-path cluster follow-up: 3 cluster(s): Minimum Safety Factor Parameters; q_like COCOS-dependent fields; Safety Factor Profile Metrics.

1. Query: `minimum safety factor q profile`
   Returned groups inspected (14): 1. core_profiles/profiles_1d/q (score: 0.54); 2. equilibrium/time_slice/profiles_1d/q (score: 0.54); 3. edge_profiles/profiles_1d/q (score: 0.53); 4. core_instant_changes/change/profiles_1d/q (score: 0.53); 5. Safety Factor At Q95 (0.83); 6. Plasma Safety Factor Profile (0.82); 7. Safety Factor Summary Values (0.81); 8. Safety Factor Profile Metrics (0.81); 9. safety factor (0.79); 10. Perpendicular Pressure Profile Values (0.79); 11. Total Perpendicular Pressure Profile (0.78); 12. Ion Momentum Profile Components (0.78); 13. Plasma Profile Finite Element Coefficients (0.78); 14. Plasma Profile Field Coefficients (0.78).
2. Query: `global minimum q value on magnetic flux surfaces`
   Returned groups inspected (14): 1. equilibrium/time_slice/global_quantities/psi_magnetic_axis (score: 0.52); 2. equilibrium/time_slice/constraints/diamagnetic_flux/measured (score: 0.52); 3. equilibrium/time_slice/profiles_1d/b_field_min (score: 0.51); 4. core_transport/model/profiles_1d/neutral/state/energy/flux (score: 0.50); 5. Magnetic Flux Boundary Values (0.84); 6. Magnetic Flux Boundary Values (0.84); 7. Plasma Boundary Magnetic Flux (0.83); 8. Magnetic Flux Surface Volume (0.82); 9. Pedestal Magnetic Field Values (0.82); 10. Radial Flux Values (0.82); 11. Plasma Boundary Flux Area (0.82); 12. Plasma Flux Surface Geometry (0.81); 13. Diamagnetic Field Component Values (0.81); 14. Magnetic Field Extremum Profiles (0.81).
3. Query: `minimum of equilibrium safety factor dimensionless`
   Returned groups inspected (15): 1. equilibrium/time_slice/global_quantities/q_axis (score: 0.50); 2. equilibrium/time_slice/global_quantities (score: 0.50); 3. equilibrium/time_slice/profiles_1d/q (score: 0.50); 4. equilibrium/time_slice/global_quantities/q_min/value (score: 0.50); 5. equilibrium/time_slice/constraints/q/position (score: 0.49); 6. Minimum Safety Factor Parameters (0.87); 7. Safety Factor Summary Values (0.83); 8. Safety Factor At Q95 (0.83); 9. safety factor (0.82); 10. Plasma Safety Factor Profile (0.82); 11. Safety Factor Profile Metrics (0.80); 12. Bootstrap Current Stability Parameters (0.79); 13. Equilibrium Constraint Measured Values (0.79); 14. Radial Momentum Flux Limiter Coefficients (0.78); 15. Equilibrium Strike Point Radii (0.78).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `equilibrium/time_slice/global_quantities/q_min/value` | `1` | `1` | **AGREE** | Exact minimum q value. |
| 2 | `equilibrium/time_slice/global_quantities/q_axis` | `1` | `1` | **AGREE** | Axis value, not the profile minimum. |
| 3 | `equilibrium/time_slice/global_quantities/q_95` | `1` | `1` | **AGREE** | Boundary-proximate q95, not the minimum. |

### `neutron_flux_due_to_fusion`

Name unit: `Hz`. Outcome: **candidate-found** — exact path `neutron_diagnostic/neutron_flux_total`. Exact-path cluster follow-up: 1 cluster(s): Neutron Diagnostic Fusion Power.

1. Query: `volume integrated neutron production rate due to fusion`
   Returned groups inspected (13): 1. edge_transport/model/ggd_fast/neutral/particle_flux_integrated/value (0.37); 2. edge_transport/model/ggd_fast/neutral/particle_flux_integrated/grid_subset_index (0.37); 3. edge_transport/model/ggd_fast/neutral/particle_flux_integrated/grid_index (0.37); 4. Fusion Neutron Production Rates (0.85); 5. Fusion Neutron Flux Rates (0.82); 6. Fusion Neutron Flux Rates (0.82); 7. Tritium-Tritium Fusion Neutron Rates (0.82); 8. Thermal Neutron Fusion Rates (0.82); 9. Thermal Neutron Flux (0.81); 10. Total Neutron Reaction Rates (0.80); 11. Neutron Emissivity Fusion Profiles (0.80); 12. Inboard Fast Neutron Flux (0.80); 13. DD Beam Neutron Rates (0.80).
2. Query: `total fusion neutron flux from all reactions in hertz`
   Returned groups inspected (10): 1. Fusion Neutron Flux Rates (0.85); 2. Fusion Neutron Flux Rates (0.85); 3. Thermal Neutron Flux (0.83); 4. Fusion Neutron Production Rates (0.82); 5. Neutron Emissivity Fusion Profiles (0.82); 6. Neutron Energy Spectrum Data (0.82); 7. Neutron Flux Event Types (0.82); 8. Plasma Particle Energy Fluxes (0.82); 9. Neutron Diagnostic Frequency Data (0.81); 10. Neutron Diagnostic Spectrum Data (0.81).
3. Query: `neutron emission rate including thermonuclear beam-thermal and beam-beam fusion`
   Returned groups inspected (10): 1. Thermal Neutron Flux (0.83); 2. Tritium-Tritium Fusion Neutron Rates (0.83); 3. Thermal Neutron Fusion Rates (0.82); 4. Neutron Emissivity Fusion Profiles (0.81); 5. DD Beam Neutron Rates (0.81); 6. Neutron Flux Beam Reactions (0.81); 7. Neutron Emissivity Reconstruction Accuracy (0.81); 8. Neutron Energy Spectrum Data (0.80); 9. Fusion Neutron Flux Rates (0.80); 10. Fusion Neutron Flux Rates (0.80).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `neutron_diagnostic/neutron_flux_total` | `s^-1` | `Hz` | **AGREE** | Exact reconstructed total neutron emission rate; s^-1 is Hz. |
| 2 | `summary/fusion/neutron_fluxes/total` | `Hz` | `Hz` | **AGREE** | Exact total across fusion channels, but lifecycle is removed in the active DD. |
| 3 | `neutron_diagnostic/reconstructed_emissivity/emissivity_dd` | `m^-3.s^-1` | `Hz` | **DISAGREE** | Spatial DD emissivity, not volume-integrated total. |

### `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift`

Name unit: `m.s^-1`. Outcome: **candidate-found** — exact path `edge_profiles/ggd/neutral/state/velocity_diamagnetic/parallel`. Exact-path cluster follow-up: 2 cluster(s): Diamagnetic Parallel Velocity Profiles; Diamagnetic Velocity Profile Components.

1. Query: `parallel effective neutral internal state velocity due to diamagnetic drift`
   Returned groups inspected (12): 1. edge_profiles/ggd/neutral/state/velocity_diamagnetic (0.57); 2. edge_profiles/ggd/neutral/state/velocity_diamagnetic/parallel (0.56); 3. Diamagnetic Drift Velocity Profiles (0.87); 4. Plasma Diamagnetic Velocity Components (0.87); 5. Diamagnetic Velocity Vertical Component (0.87); 6. Diamagnetic Velocity Radial Component (0.87); 7. Diamagnetic Parallel Velocity Profiles (0.86); 8. Diamagnetic Velocity Profile Components (0.86); 9. Diamagnetic Velocity Profile Components (0.86); 10. Diamagnetic Velocity Profile Components (0.86); 11. Diamagnetic Velocity Components (0.86); 12. Diamagnetic Velocity Components (0.86).
2. Query: `field aligned effective velocity of neutral particles in a specified internal state from diamagnetic drift`
   Returned groups inspected (12): 1. edge_profiles/ggd/neutral/state/velocity_diamagnetic (0.56); 2. edge_profiles/ggd/ion/velocity_over_b_field/diamagnetic (0.54); 3. Diamagnetic Velocity Vertical Component (0.86); 4. Plasma Diamagnetic Velocity Components (0.86); 5. Diamagnetic Velocity Radial Component (0.86); 6. Diamagnetic Velocity Components (0.85); 7. Diamagnetic Velocity Components (0.85); 8. Diamagnetic Velocity Components (0.85); 9. Diamagnetic ExB Velocity Components (0.85); 10. Diamagnetic Velocity Profile Components (0.85); 11. Diamagnetic Velocity Profile Components (0.85); 12. Diamagnetic Velocity Profile Components (0.85).
3. Query: `parallel neutral state diamagnetic drift velocity in metres per second`
   Returned groups inspected (13): 1. edge_profiles/ggd/neutral/state/velocity_diamagnetic (0.56); 2. edge_profiles/ggd/neutral/state/velocity_diamagnetic/parallel (0.55); 3. edge_transport/model/ggd/neutral/state/momentum/v_parallel (0.55); 4. Diamagnetic Drift Velocity Profiles (0.86); 5. Diamagnetic Velocity Vertical Component (0.86); 6. Plasma Diamagnetic Velocity Components (0.86); 7. Diamagnetic Velocity Profile Components (0.85); 8. Diamagnetic Velocity Profile Components (0.85); 9. Diamagnetic Velocity Profile Components (0.85); 10. Diamagnetic Velocity Radial Component (0.85); 11. Diamagnetic Parallel Velocity Profiles (0.85); 12. Diamagnetic Velocity Components (0.85); 13. Diamagnetic Velocity Components (0.85).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `edge_profiles/ggd/neutral/state/velocity_diamagnetic/parallel` | `m.s^-1` | `m.s^-1` | **AGREE** | Exact neutral-state, diamagnetic, parallel component. |
| 2 | `plasma_profiles/ggd/neutral/state/velocity_diamagnetic/parallel` | `m.s^-1` | `m.s^-1` | **AGREE** | Exact sibling in plasma_profiles. |
| 3 | `edge_profiles/ggd/ion/state/velocity_diamagnetic/parallel` | `m.s^-1` | `m.s^-1` | **AGREE** | Ion-state rather than neutral-state. |

### `parallel_neutral_momentum_diffusion_coefficient`

Name unit: `m^2.s^-1`. Outcome: **candidate-found** — exact path `plasma_transport/model/ggd/neutral/momentum/d_parallel`. Exact-path cluster follow-up: 0 cluster(s): none.

1. Query: `parallel neutral momentum diffusion coefficient`
   Returned groups inspected (11): 1. edge_transport/model/ggd/neutral/state/momentum/d_parallel (0.55); 2. Parallel Energy Particle Diffusivity (0.82); 3. Plasma Transport Effective Diffusivity (0.82); 4. Plasma Transport Diffusion Values (0.82); 5. Radial Diffusion Coefficient Coefficients (0.82); 6. Diamagnetic Momentum Diffusion Coefficients (0.81); 7. Parallel Momentum Diffusion Coefficients (0.81); 8. Radial Transport Diffusion Coefficients (0.81); 9. Radial Transport Diffusion Coefficients (0.81); 10. Plasma Transport Energy Coefficients (0.81); 11. Parallel Momentum Flux Coefficients (0.81).
2. Query: `effective parallel diffusivity for transport of neutral momentum`
   Returned groups inspected (11): 1. edge_transport/model/ggd/neutral/state/momentum/d_parallel (0.57); 2. Plasma Transport Effective Diffusivity (0.86); 3. Parallel Energy Particle Diffusivity (0.84); 4. Poloidal Effective Diffusivity Profiles (0.83); 5. Poloidal Effective Diffusivity Profiles (0.83); 6. Poloidal Effective Diffusivity Coefficients (0.83); 7. Plasma Transport Diffusion Values (0.83); 8. Effective Transport Diffusivity Coefficients (0.82); 9. Radial Momentum Diffusivity Profiles (0.82); 10. Plasma Transport Energy Coefficients (0.82); 11. Parallel Momentum Transport Components (0.81).
3. Query: `field aligned neutral internal-state momentum d_parallel in square metres per second`
   Returned groups inspected (13): 1. edge_transport/model/ggd/neutral/state/momentum/d_parallel (0.53); 2. edge_transport/model/ggd/neutral/state/momentum/flux_parallel (0.53); 3. edge_transport/model/ggd/neutral/state/momentum/flux (0.52); 4. Diamagnetic Momentum Source Profiles (0.82); 5. Plasma Source Diamagnetic Momentum (0.82); 6. Neutral Diamagnetic Momentum Flux (0.81); 7. Momentum Diamagnetic Interpolation Coefficients (0.81); 8. Diamagnetic Momentum Flux Components 2 (0.81); 9. Neutron Detector Direction Vectors (0.81); 10. Diamagnetic Momentum Interpolation Coefficients (0.81); 11. Diamagnetic Momentum Interpolation Coefficients (0.81); 12. Diamagnetic Momentum Interpolation Coefficients (0.81); 13. Diamagnetic Velocity Vertical Component (0.81).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `plasma_transport/model/ggd/neutral/momentum/d_parallel` | `m^2.s^-1` | `m^2.s^-1` | **AGREE** | Exact species-level neutral momentum parallel diffusivity. |
| 2 | `edge_transport/model/ggd/neutral/state/momentum/d_parallel` | `m^2.s^-1` | `m^2.s^-1` | **AGREE** | Charge/internal-state-resolved, more specific than the name. |
| 3 | `plasma_transport/model/profiles_1d/neutral/state/momentum/flux_parallel` | `kg.m^-1.s^-2` | `m^2.s^-1` | **DISAGREE** | Momentum flux rather than diffusivity. |

### `poloidal_neutral_internal_state_momentum_convected_velocity`

Name unit: `m.s^-1`. Outcome: **candidate-found** — exact path `plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol`. Exact-path cluster follow-up: 0 cluster(s): none.

1. Query: `poloidal neutral internal state momentum convected velocity`
   Returned groups inspected (11): 1. edge_transport/model/ggd/neutral/state/momentum/v_pol (0.56); 2. Poloidal Convection Velocity Components (0.87); 3. Poloidal Velocity Transport Components (0.86); 4. Poloidal Mass Center Velocity (0.85); 5. Poloidal Convection Velocity Values (0.85); 6. Effective Poloidal Convection Velocity (0.85); 7. Poloidal Convective Velocity Profiles (0.85); 8. Poloidal Momentum Velocity Components (0.85); 9. Plasma Poloidal Velocity Values (0.85); 10. Poloidal Momentum Flux Profiles (0.84); 11. Poloidal Momentum Flux Profiles (0.84).
2. Query: `poloidal convective velocity for neutral state momentum transport`
   Returned groups inspected (11): 1. edge_transport/model/ggd/neutral/state/momentum/v_pol (0.58); 2. Poloidal Convection Velocity Components (0.87); 3. Poloidal Convection Velocity Values (0.87); 4. Effective Poloidal Convection Velocity (0.87); 5. Poloidal Velocity Transport Components (0.86); 6. Poloidal Convective Velocity Profiles (0.86); 7. Plasma Poloidal Velocity Values (0.85); 8. Poloidal Mass Center Velocity (0.85); 9. Poloidal Momentum Transport Components (0.84); 10. Radial Effective Convection Velocity (0.84); 11. Poloidal Momentum Velocity Components (0.84).
3. Query: `neutral internal-state momentum effective convection in poloidal direction metres per second`
   Returned groups inspected (12): 1. edge_transport/model/ggd/neutral/momentum/v_pol (0.56); 2. edge_transport/model/ggd/neutral/state/momentum/v_pol (0.56); 3. Poloidal Convection Velocity Components (0.87); 4. Effective Poloidal Convection Velocity (0.86); 5. Poloidal Convection Velocity Values (0.86); 6. Radial Effective Convection Velocity (0.86); 7. Plasma Poloidal Velocity Values (0.85); 8. Poloidal Mass Center Velocity (0.85); 9. Ion Poloidal Momentum Flux (0.85); 10. Ion Momentum Radial Flux Components (0.84); 11. Poloidal Convective Velocity Profiles (0.84); 12. Diamagnetic Velocity Poloidal Coefficients (0.84).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol` | `m.s^-1` | `m.s^-1` | **AGREE** | Exact state-resolved poloidal momentum convection velocity. |
| 2 | `edge_transport/model/ggd/neutral/state/momentum/v_pol` | `m.s^-1` | `m.s^-1` | **AGREE** | Exact edge-transport sibling. |
| 3 | `edge_transport/model/ggd/neutral/momentum/v_pol` | `m.s^-1` | `m.s^-1` | **AGREE** | Species-level rather than state-resolved. |

### `poloidal_straight_field_line_angle`

Name unit: `rad`. Outcome: **candidate-found** — exact path `distributions/distribution/profiles_2d/grid/theta_straight`. Exact-path cluster follow-up: 1 cluster(s): Distribution Function Radial Grids.

1. Query: `poloidal straight field line angle`
   Returned groups inspected (12): 1. equilibrium/time_slice/profiles_2d/b_field_r (0.51); 2. equilibrium/time_slice/ggd/b_field_r (0.51); 3. Magnetic Field Angle Data (0.83); 4. Poloidal A-Field Components (0.83); 5. Edge Poloidal Field Components (0.82); 6. Bolometer Line Toroidal Angles (0.82); 7. Pedestal Poloidal Field Values (0.82); 8. Poloidal Electric Field Components (0.82); 9. Poloidal Field Component Profiles (0.82); 10. Poloidal Magnetic Field Components (0.82); 11. Plasma Poloidal Velocity Values (0.82); 12. MSE Polarization Angle Data (0.82).
2. Query: `straight field line poloidal angular coordinate`
   Returned groups inspected (11): 1. equilibrium/time_slice/profiles_2d/b_field_r (0.52); 2. Poloidal A-Field Components (0.84); 3. Edge Poloidal Field Components (0.83); 4. Radial Coordinate Flux Definitions (0.82); 5. Edge Radial Field Components (0.82); 6. Poloidal Magnetic Field Components (0.82); 7. Radial Magnetic Field Components (0.82); 8. Poloidal Field Component Profiles (0.82); 9. Magnetic Field Angle Data (0.82); 10. Magnetic Flux Coordinate Definitions (0.82); 11. Magnetic Field Radial Coefficients (0.82).
3. Query: `theta straight coordinate on wave propagation grid in radians`
   Returned groups inspected (12): 1. equilibrium/time_slice/ggd/theta/grid_index (0.91); 2. equilibrium/time_slice/profiles_2d/theta (0.88); 3. Wave Field Grid Indices (0.85); 4. Wave Field Grid Indices (0.85); 5. Wave Field Grid Indices (0.85); 6. Radial Grid Coordinate Definition (0.84); 7. Radiation Radial Grid Coordinates (0.84); 8. Full Wave Field Grid Indices (0.84); 9. Poloidal Angle Grid Values (0.83); 10. Wave Grid Geometry Indices (0.83); 11. Reflectometer Toroidal Angle Coordinates (0.82); 12. MHD Grid Coordinate Indices (0.82).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `distributions/distribution/profiles_2d/grid/theta_straight` | `rad` | `rad` | **AGREE** | Exact straight-field-line poloidal angle. |
| 2 | `waves/coherent_wave/profiles_2d/grid/theta_straight` | `rad` | `rad` | **AGREE** | Exact wave-grid sibling. |
| 3 | `equilibrium/time_slice/profiles_2d/theta` | `rad` | `rad` | **AGREE** | Geometric poloidal angle, not straight-field-line angle. |

### `tendency_of_total_thermal_plasma_internal_energy`

Name unit: `W`. Outcome: **candidate-found** — exact path `summary/global_quantities/denergy_thermal_dt/value`. Exact-path cluster follow-up: 1 cluster(s): Disruption Decay Time Analysis.

1. Query: `tendency of total thermal plasma internal energy`
   Returned groups inspected (14): 1. core_instant_changes/change/profiles_1d/pressure_ion_total (0.88); 2. edge_profiles/ggd_fast/energy_thermal/value (0.53); 3. edge_profiles/ggd_fast/energy_thermal (0.53); 4. equilibrium/time_slice/global_quantities/energy_mhd (0.53); 5. Plasma Kinetic Energy Density (0.85); 6. Thermal Electron Energy Content (0.84); 7. Plasma Energy Content Metrics (0.84); 8. Total Thermal Ion Pressure (0.84); 9. Plasma Stored Energy Metrics (0.83); 10. Plasma Transport Energy Coefficients (0.83); 11. Ion Temperature Profile Data (0.82); 12. Ion Temperature Profile Data (0.82); 13. Ion Temperature Profile Averages (0.82); 14. Plasma Source Energy Coefficients (0.82).
2. Query: `time derivative of total volume integrated thermal plasma energy`
   Returned groups inspected (13): 1. equilibrium/time_slice/global_quantities/energy_mhd (0.53); 2. core_profiles/global_quantities/ion/t_i_volume_average (0.52); 3. core_sources/source/profiles_1d/ion/power_inside (0.52); 4. Plasma Kinetic Energy Density (0.82); 5. Ion Pressure Time Derivatives (0.82); 6. Plasma Energy Content Metrics (0.82); 7. Average Ion Temperature Derivative (0.82); 8. Total Thermal Ion Pressure (0.82); 9. Plasma Stored Energy Metrics (0.81); 10. Volume Averaged Plasma Parameters (0.81); 11. Volume Averaged Plasma Parameters (0.81); 12. Plasma Transport Energy Coefficients (0.81); 13. Volume Averaged Plasma Quantities (0.81).
3. Query: `global thermal plasma internal energy change rate or power in watts`
   Returned groups inspected (13): 1. equilibrium/time_slice/global_quantities/energy_mhd (0.54); 2. core_sources/source/global_quantities/total_ion_power (0.54); 3. wall/global_quantities/power_to_cooling (0.53); 4. Plasma Energy Content Metrics (0.85); 5. Calorimetry Power Energy Data (0.84); 6. Plasma Stored Energy Metrics (0.84); 7. Plasma Kinetic Energy Density (0.84); 8. Total Thermal Ion Pressure (0.84); 9. Plasma Source Energy Coefficients (0.84); 10. Thermal Electron Energy Content (0.83); 11. Plasma Radiation Power Density (0.83); 12. Plasma Particle Energy Fluxes (0.83); 13. Nuclear Heating Power Density (0.82).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `summary/global_quantities/denergy_thermal_dt/value` | `W` | `W` | **AGREE** | Exact time derivative of thermal stored energy. |
| 2 | `summary/global_quantities/denergy_thermal_dt` | `W` | `W` | **AGREE** | Exact structure containing the measured value. |
| 3 | `equilibrium/time_slice/global_quantities/energy_mhd` | `J` | `W` | **DISAGREE** | Stored MHD energy, not its thermal time derivative. |

### `toroidal_ion_charge_state_torque_density`

Name unit: `kg.m^-1.s^-2`. Outcome: **candidate-found** — exact path `plasma_sources/source/ggd/ion/state/momentum/phi`. Exact-path cluster follow-up: 2 cluster(s): Toroidal Momentum Source Profiles; Toroidal Momentum Source Components.

1. Query: `toroidal ion charge state torque density`
   Returned groups inspected (10): 1. Collisional Toroidal Torque Density (0.85); 2. Toroidal Ion Rotation Frequency (0.84); 3. Ion Toroidal Rotation Velocity (0.83); 4. Toroidal Torque Radial Currents (0.83); 5. Total Toroidal Torque Sources (0.83); 6. Toroidal Electric Field Components (0.83); 7. Induced Toroidal Current Density (0.83); 8. Charge Exchange Toroidal Angle (0.83); 9. Plasma Toroidal Momentum Values (0.83); 10. Plasma Toroidal Momentum Values (0.83).
2. Query: `toroidal angular momentum source density acting on a specified ion charge state`
   Returned groups inspected (11): 1. core_sources/source/profiles_1d/momentum_phi (0.53); 2. Toroidal Momentum Source Components (0.84); 3. Toroidal Momentum Source Components (0.84); 4. Toroidal Momentum Source Components (0.84); 5. Plasma Toroidal Momentum Sources (0.84); 6. Toroidal Momentum Source Profiles (0.84); 7. Toroidal Momentum Source Profiles (0.84); 8. Plasma Toroidal Momentum Summary (0.83); 9. Plasma Toroidal Momentum Summary (0.83); 10. Plasma Toroidal Momentum Profiles (0.83); 11. Plasma Toroidal Momentum Values (0.83).
3. Query: `charge-state-resolved net toroidal torque density summed over mechanisms`
   Returned groups inspected (10): 1. Total Toroidal Torque Sources (0.84); 2. Collisional Toroidal Torque Density (0.84); 3. Plasma Toroidal Momentum Summary (0.83); 4. Plasma Toroidal Momentum Summary (0.83); 5. Toroidal Momentum Torque Profiles (0.83); 6. Plasma Toroidal Momentum Values (0.83); 7. Plasma Toroidal Momentum Values (0.83); 8. Plasma Toroidal Momentum Sources (0.83); 9. Toroidal Torque Radial Currents (0.82); 10. Toroidal Momentum Source Components (0.82).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `plasma_sources/source/ggd/ion/state/momentum/phi` | `kg.m^-1.s^-2` | `kg.m^-1.s^-2` | **AGREE** | Exact charge-state-resolved net toroidal momentum source density by hierarchy. |
| 2 | `plasma_sources/source/ggd/ion/momentum/phi` | `kg.m^-1.s^-2` | `kg.m^-1.s^-2` | **AGREE** | Ion-species resolved only; charge-state ownership absent. |
| 3 | `distributions/distribution/profiles_1d/collisions/ion/state/torque_thermal_phi` | `N.m^-2` | `kg.m^-1.s^-2` | **AGREE** | Algebraically equal unit, but collision-only and thermal-recipient-specific. |

### `toroidal_line_integrated_impurity_ion_velocity`

Name unit: `m.s^-1`. Outcome: **candidate-found** — exact path `spectrometer_x_ray_crystal/channel/profiles_line_integrated/velocity_tor`. Exact-path cluster follow-up: 1 cluster(s): X-Ray Spectrometer Toroidal Velocity.

1. Query: `toroidal line integrated impurity ion velocity`
   Returned groups inspected (15): 1. core_profiles/profiles_2d/ion/velocity/toroidal (0.38); 2. core_instant_changes/change/profiles_1d/ion/velocity/toroidal (0.38); 3. core_profiles/profiles_1d/ion/velocity/toroidal (0.38); 4. core_sources/source/profiles_1d/ion/momentum/toroidal_decomposed/implicit_part (0.37); 5. edge_transport/model/ggd_fast/ion/particle_flux_integrated/grid_subset_index (0.37); 6. Toroidal Ion Velocity Profiles (0.83); 7. Impurity Toroidal Rotation Velocity (0.82); 8. Volume Averaged Ion Temperature (0.82); 9. Ion Toroidal Rotation Velocity (0.82); 10. Iron Impurity Toroidal Velocity (0.81); 11. Ion Toroidal Velocity Profiles (0.81); 12. Xenon Toroidal Velocity Values (0.81); 13. Plasma Toroidal Momentum Values (0.81); 14. Plasma Toroidal Momentum Values (0.81); 15. Diamagnetic Velocity Toroidal Components (0.81).
2. Query: `toroidal impurity-ion velocity inferred from a line-integrated spectral signal`
   Returned groups inspected (14): 1. core_profiles/profiles_2d/ion/velocity/toroidal (0.37); 2. core_instant_changes/change/profiles_1d/ion/velocity/toroidal (0.37); 3. core_profiles/profiles_1d/ion/velocity/toroidal (0.37); 4. core_profiles/profiles_1d/ion/velocity/diamagnetic (0.37); 5. Impurity Toroidal Rotation Velocity (0.81); 6. Toroidal Ion Velocity Profiles (0.81); 7. X-Ray Spectrometer Toroidal Velocity (0.81); 8. Impurity Velocity Summary (0.80); 9. Volume Averaged Ion Temperature (0.80); 10. Tritium Toroidal Velocity Summary (0.80); 11. Iron Impurity Toroidal Velocity (0.80); 12. Ion Toroidal Rotation Velocity (0.80); 13. Helium Isotope Toroidal Velocity (0.79); 14. Plasma Toroidal Momentum Values (0.79).
3. Query: `x-ray crystal spectrometer line-integrated toroidal velocity in metres per second`
   Returned groups inspected (10): 1. X-Ray Spectrometer Toroidal Velocity (0.87); 2. X-Ray Crystal Toroidal Velocity (0.85); 3. X-Ray Spectrometer Toroidal Angles (0.83); 4. Xenon Toroidal Velocity Values (0.83); 5. Xenon Toroidal Velocity Summary (0.83); 6. Xenon Toroidal Velocity Summary (0.83); 7. Ion Toroidal Rotation Velocity (0.82); 8. Ion Toroidal Velocity Profiles (0.81); 9. Plasma Toroidal Momentum Values (0.81); 10. Plasma Toroidal Momentum Values (0.81).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `spectrometer_x_ray_crystal/channel/profiles_line_integrated/velocity_tor` | `m.s^-1` | `m.s^-1` | **AGREE** | Exact line-integrated toroidal spectroscopic velocity structure. |
| 2 | `spectrometer_x_ray_crystal/channel/profiles_line_integrated/velocity_tor/data` | `m.s^-1` | `m.s^-1` | **AGREE** | Exact numeric data leaf. |
| 3 | `core_profiles/profiles_1d/ion/velocity/toroidal` | `m.s^-1` | `m.s^-1` | **AGREE** | Toroidal ion velocity but not diagnostic line-integrated. |

### `toroidal_neutral_state_momentum_diffusivity`

Name unit: `m^2.s^-1`. Outcome: **candidate-found** — exact path `plasma_transport/model/ggd/neutral/state/momentum/d/phi`. Exact-path cluster follow-up: 2 cluster(s): Toroidal Momentum Flux Components; Toroidal Momentum Transport Components.

1. Query: `toroidal neutral state momentum diffusivity`
   Returned groups inspected (11): 1. edge_transport/model/ggd/neutral/state/momentum/d_parallel (0.53); 2. Plasma Toroidal Momentum Values (0.83); 3. Plasma Toroidal Momentum Values (0.83); 4. Plasma Toroidal Momentum Summary (0.83); 5. Plasma Toroidal Momentum Summary (0.83); 6. Toroidal Diamagnetic Velocity Components (0.83); 7. Plasma Transport Toroidal Momentum (0.82); 8. Toroidal Plasma Momentum Measurements (0.82); 9. Plasma Toroidal Momentum Profiles (0.82); 10. Neon Toroidal Velocity Summary (0.82); 11. Diamagnetic Velocity Toroidal Components (0.82).
2. Query: `state-resolved neutral momentum diffusivity in the toroidal direction`
   Returned groups inspected (12): 1. core_transport/model/profiles_1d/ion/state/momentum/toroidal/flux (0.52); 2. edge_transport/model/ggd/neutral/state/momentum/d_parallel (0.52); 3. Plasma Toroidal Momentum Values (0.83); 4. Plasma Toroidal Momentum Values (0.83); 5. Plasma Toroidal Momentum Summary (0.83); 6. Plasma Toroidal Momentum Summary (0.83); 7. Plasma Toroidal Momentum Profiles (0.83); 8. Plasma Transport Toroidal Momentum (0.82); 9. Toroidal Diamagnetic Velocity Components (0.82); 10. Neon Toroidal Velocity Summary (0.82); 11. Radial Momentum Diffusivity Profiles (0.82); 12. Toroidal Plasma Momentum Measurements (0.82).
3. Query: `toroidal component of neutral internal-state momentum diffusion coefficient in square metres per second`
   Returned groups inspected (12): 1. core_transport/model/profiles_1d/ion/state/momentum/toroidal/flux (0.52); 2. edge_transport/model/ggd/neutral/state/momentum/d_parallel (0.52); 3. Diamagnetic Momentum Diffusion Coefficients (0.83); 4. Toroidal Diamagnetic Velocity Coefficients (0.83); 5. Diamagnetic Velocity Toroidal Coefficients (0.83); 6. Diamagnetic Velocity Toroidal Components (0.83); 7. Toroidal Momentum Transport Coefficients (0.82); 8. Toroidal Momentum Transport Coefficients (0.82); 9. Toroidal Diamagnetic Velocity Components (0.82); 10. Toroidal Momentum Profile Components (0.82); 11. Plasma Viscosity Toroidal Coefficients (0.82); 12. Toroidal Momentum Source Coefficients (0.82).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `plasma_transport/model/ggd/neutral/state/momentum/d/phi` | `m^2.s^-1` | `m^2.s^-1` | **AGREE** | Exact neutral-state toroidal momentum diffusivity. |
| 2 | `plasma_transport/model/ggd/neutral/momentum/d/phi` | `m^2.s^-1` | `m^2.s^-1` | **AGREE** | Neutral species-level, not state-resolved. |
| 3 | `plasma_transport/model/ggd/ion/state/momentum/d/phi` | `m^2.s^-1` | `m^2.s^-1` | **AGREE** | Ion state rather than neutral state. |

### `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions`

Name unit: `N.m^-2`. Outcome: **no-candidate**. Exact-path cluster follow-up: 1 cluster(s): Collisional Toroidal Torque Density.

1. Query: `toroidal trapped thermal ion charge state torque density due to collisions`
   Returned groups inspected (10): 1. Collisional Toroidal Torque Density (0.85); 2. Tritium Toroidal Velocity Summary (0.83); 3. Charge Exchange Toroidal Angle (0.82); 4. Toroidal Ion Rotation Frequency (0.82); 5. Toroidal Plasma Momentum Measurements (0.82); 6. Ion Toroidal Rotation Velocity (0.82); 7. Total Thermal Ion Pressure (0.82); 8. Toroidal Ion Velocity Profiles (0.82); 9. Plasma Toroidal Momentum Values (0.82); 10. Plasma Toroidal Momentum Values (0.82).
2. Query: `collisional toroidal torque density delivered to trapped thermal ions in a specified charge state`
   Returned groups inspected (10): 1. Collisional Toroidal Torque Density (0.84); 2. Total Thermal Ion Pressure (0.82); 3. Plasma Toroidal Momentum Values (0.82); 4. Plasma Toroidal Momentum Values (0.82); 5. Toroidal Plasma Momentum Measurements (0.82); 6. Tritium Toroidal Velocity Summary (0.82); 7. Toroidal Ion Velocity Profiles (0.82); 8. Charge Exchange Toroidal Angle (0.82); 9. Toroidal Ion Rotation Frequency (0.81); 10. Plasma Toroidal Momentum Summary (0.81).
3. Query: `toroidal collision torque on trapped thermal ion population resolved by charge state`
   Returned groups inspected (10): 1. Collisional Toroidal Torque Density (0.83); 2. Tritium Toroidal Velocity Summary (0.83); 3. Charge Exchange Toroidal Angle (0.82); 4. Toroidal Plasma Momentum Measurements (0.82); 5. Toroidal Ion Rotation Frequency (0.81); 6. Plasma Toroidal Momentum Values (0.81); 7. Plasma Toroidal Momentum Values (0.81); 8. Plasma Toroidal Momentum Summary (0.81); 9. Plasma Toroidal Momentum Summary (0.81); 10. Toroidal Ion Velocity Profiles (0.81).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `distributions/distribution/profiles_2d/trapped/collisions/ion/state/torque_thermal_phi` | `N.m^-2` | `N.m^-2` | **AGREE** | Torque from the trapped source distribution to a thermal ion state; modifiers occupy different roles. |
| 2 | `distributions/distribution/profiles_2d/trapped/collisions/ion/state/torque_fast_phi` | `N.m^-2` | `N.m^-2` | **AGREE** | Fast recipient rather than thermal recipient. |
| 3 | `distributions/distribution/profiles_1d/trapped/collisions/ion/state/torque_thermal_phi` | `N.m^-2` | `N.m^-2` | **AGREE** | Same source-versus-recipient mismatch in 1D. |

**Exhausted negative:** all three query result sets and the exact-path `Collisional Toroidal Torque Density` cluster were inspected. Every torque-density candidate either makes `trapped` describe the source distribution and `thermal` the recipient, or changes the recipient to fast. None measures torque delivered to a population that is simultaneously trapped and thermal in the identity’s sense. The negative is semantic; all three leading units agree.

### `x_direction_unit_vector_of_sensor`

Name unit: `1`. Outcome: **candidate-found** — exact path `operational_instrumentation/sensor/direction/x`. Exact-path cluster follow-up: 3 cluster(s): Operational Sensor Spatial Coordinates; Instrument Direction Vector Components; Operational Instrumentation Sensor Orientation.

1. Query: `x direction unit vector of sensor`
   Returned groups inspected (10): 1. Camera X-Axis Unit Vectors (0.86); 2. Diagnostic Y-Axis Unit Vectors (0.84); 3. Diagnostic Y-Direction Vectors (0.84); 4. X-Ray Detector Unit Vectors (0.84); 5. Diagnostic Sensor Direction Z (0.84); 6. Antenna Y-Axis Orientation Vectors (0.84); 7. X-Ray Camera Unit Vectors (0.83); 8. X-Ray Camera Unit Vectors (0.83); 9. X-Ray Camera Unit Vectors (0.83); 10. X-Ray Camera Unit Vectors (0.83).
2. Query: `dimensionless x component direction cosine of sensor orientation unit vector`
   Returned groups inspected (10): 1. Instrument Direction Vector Components (0.84); 2. Camera X-Axis Unit Vectors (0.84); 3. Diagnostic Antenna Orientation Vector (0.83); 4. Antenna Y-Axis Orientation Vectors (0.83); 5. Diagnostic Toroidal Angle Coordinates (0.83); 6. Diagnostic Toroidal Angle Coordinates (0.83); 7. Diagnostic Toroidal Angle Coordinates (0.83); 8. Antenna Orientation Unit Vectors (0.83); 9. Antenna Orientation Unit Vectors (0.83); 10. Diagnostic X-Axis Vector Components (0.83).
3. Query: `sensor orientation x direction cosine with unit one`
   Returned groups inspected (10): 1. Operational Instrumentation Sensor Orientation (0.84); 2. Diagnostic Antenna Orientation Vector (0.83); 3. Camera X-Axis Unit Vectors (0.83); 4. Antenna Y-Axis Orientation Vectors (0.83); 5. ECE Measurement Flux Coordinates (0.82); 6. Diagnostic Toroidal Angle Coordinates (0.82); 7. Diagnostic Toroidal Angle Coordinates (0.82); 8. Diagnostic Toroidal Angle Coordinates (0.82); 9. Soft X-Ray Detector Orientation (0.82); 10. Soft X-Ray Detector Orientation (0.82).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `operational_instrumentation/sensor/direction/x` | `1` | `1` | **AGREE** | Exact sensor direction cosine in the current DD. |
| 2 | `camera_visible/channel/aperture/x1_unit_vector/x` | `m` | `1` | **DISAGREE** | Aperture geometry vector component with metre-valued DD convention. |
| 3 | `operational_instrumentation/sensor/direction_second/x` | `1` | `1` | **AGREE** | Second direction vector, not the primary sensor direction. |

### `z_direction_unit_vector_of_sensor`

Name unit: `1`. Outcome: **candidate-found** — exact path `operational_instrumentation/sensor/direction/z`. Exact-path cluster follow-up: 3 cluster(s): Operational Sensor Spatial Coordinates; Diagnostic Sensor Direction Z; Operational Instrumentation Sensor Orientation.

1. Query: `z direction unit vector of sensor`
   Returned groups inspected (10): 1. Diagnostic Sensor Direction Z (0.87); 2. Diagnostic Y-Direction Vectors (0.85); 3. Diagnostic Component Z-Axis Vectors (0.85); 4. Diagnostic Antenna Orientation Vector (0.85); 5. Instrument Direction Vector Components (0.85); 6. Diagnostic Component Z Unit Vector (0.85); 7. Soft X-Ray Z-Axis Vectors (0.84); 8. Antenna Y-Axis Orientation Vectors (0.84); 9. Diagnostic Optical Z-Vector Components (0.84); 10. Diagnostic Antenna Orientation Vectors (0.84).
2. Query: `dimensionless z component direction cosine of sensor orientation unit vector`
   Returned groups inspected (10): 1. Instrument Direction Vector Components (0.85); 2. Diagnostic Component Z Orientation (0.84); 3. Diagnostic Antenna Orientation Vector (0.83); 4. Diagnostic Component Z-Axis Vectors (0.83); 5. Diagnostic Sensor Direction Z (0.83); 6. Diagnostic Optical Z-Vector Components (0.83); 7. Diagnostic Component Z Unit Vector (0.83); 8. Antenna Orientation Unit Vectors (0.83); 9. Antenna Orientation Unit Vectors (0.83); 10. Antenna Y-Axis Orientation Vectors (0.83).
3. Query: `sensor orientation vertical direction cosine with unit one`
   Returned groups inspected (10): 1. Operational Instrumentation Sensor Orientation (0.85); 2. Infrared Camera Orientation Vectors (0.84); 3. Diagnostic Antenna Orientation Vector (0.84); 4. Instrument Direction Vector Components (0.84); 5. Camera X-Axis Unit Vectors (0.83); 6. Diagnostic Antenna Orientation Vectors (0.83); 7. Diagnostic Sensor Direction Z (0.83); 8. Bolometer Detector Orientation Vectors (0.83); 9. Antenna Y-Axis Orientation Vectors (0.83); 10. Magnetic Axis Vertical Position (0.83).

| Rank | Candidate DD path | DD unit | Name unit | Unit verdict | Semantic disposition |
|---:|---|---|---|---|---|
| 1 | `operational_instrumentation/sensor/direction/z` | `1` | `1` | **AGREE** | Exact sensor vertical direction cosine in the current DD. |
| 2 | `spectrometer_uv/channel/grating/x3_unit_vector/z` | `m` | `1` | **DISAGREE** | Grating geometry vector component with metre-valued DD convention. |
| 3 | `operational_instrumentation/sensor/direction_second/z` | `1` | `1` | **AGREE** | Second direction vector, not the primary sensor direction. |

## Names previously dispositioned “attached” while still holding zero producers

The closing census used `attached` as a disposition meaning an exact measuring route or canonical coverage existed; it explicitly did not claim that the legacy identity gained a producer. The live-derived zero-producer cohort still contains these nine previously dispositioned rows:

- `capacitance_of_ion_cyclotron_heating_antenna`
- `cross_section_of_flux_surface`
- `line_integrated_electron_density`
- `neutron_flux_due_to_fusion`
- `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift`
- `parallel_neutral_momentum_diffusion_coefficient`
- `poloidal_straight_field_line_angle`
- `tendency_of_total_thermal_plasma_internal_energy`
- `toroidal_neutral_state_momentum_diffusivity`

This reverse search independently confirms a named measuring path for all nine while preserving the distinction between search evidence, canonical coverage, and a live `PRODUCED_NAME` producer.

## Nonmutation proof

The search node uses receipt run id
`r-20260822T060246617278-sourcesearch`. SHA-256
`32d29fce8ba4547ac5f73ee27639e3e5db5b57ff5ac5f8940a79b48d85b825b4`
binds the canonical 17-row search manifest: for each identity, the manifest
contains its name unit, terminal outcome, and exact selected path where one
exists. The exact run-id-plus-manifest query returned **0**
`StandardNameChange` receipts; the run-id-wide query across every manifest
digest also returned **0**.

A fresh bounded read-only verification invocation measured the counters before
the cohort and receipt queries and immediately after them:

| Graph write measure | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameChange` nodes | **7,836** | **7,836** | **0** |
| `PRODUCED_NAME` relationships | **5,780** | **5,780** | **0** |

An earlier wide window overlapped a concurrent authorized writer and observed
`StandardNameChange` move from 7,787 to 7,836 while `PRODUCED_NAME`
remained 5,780. It is not used as this node's nonmutation proof. The zero exact
receipts, zero run-wide receipts, and stable bounded verification window
together distinguish this read-only search from unrelated concurrent ledger
activity.

## Authority boundary

Live semantic authority: `imas-codex:sn-graph-wide-integrity` §3c, plan version 246, read 2026-08-22. Source checkout commit: `3293becfe9fe6afc956496bce3d472c9a46f7533`. Search service: IMAS DD-only tools, DD major 4, active lifecycle filter.
