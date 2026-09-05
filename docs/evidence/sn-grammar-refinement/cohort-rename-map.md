# Live-cohort canonical renderer rename map

Observed: 2026-09-05T07:32:34.217254+00:00

Consumer revision: `044070eea264ff14bae1ef0fe9b57b48dea93fa7`  
Grammar revision: `59754b7d628351984af430f83883626b242701c3` (runtime `0.8.1.dev32+g12b557363`)  
Population rule: every `StandardName` whose `name_stage` is not `superseded`; proposals were checked against every live and retired identity.

## No-write gate

| Measure | Count |
|---|---:|
| All stored identities | 4937 |
| Live cohort parsed leniently | 2972 |
| Retired identities included in collision check | 1965 |
| Total changed | 149 |
| Locus-tail prefix rule | 141 |
| Locus-tail postfix rule | 7 |
| Indexed-operator rule | 1 |
| Proposals colliding with any stored identity | 0 |
| Many-to-one merge targets | 0 |
| Strict reparses with byte-identical canonical IR | 149 / 149 |
| Identities that fail lenient parsing | 7 |

The three staging prerequisites hold: collisions = 0, many-to-one merges = 0, and every proposed row strict-reparses to the exact canonical JSON byte serialization of the original lenient IR. The live population is 2 below the earlier 2,974-name census because two additional stored identities are now superseded; the measured 149-change delta and seven parse failures are unchanged.

LLM ledger before any migration command: 36914 rows, $1777.493812999981.

## Rename rows

| # | Rule | Old spelling | Canonical spelling | Strict reparse | IR byte-identical | Collision | IR SHA-256 |
|---:|---|---|---|---|---|---|---|
| 1 | locus-tail prefix | `accumulated_carbon_count_due_to_gas_injection` | `carbon_count_accumulated_due_to_gas_injection` | yes | yes | none | `888328604cc72d49fd10f563a340ca6e0210af8ee0f9e8b4d7d39d398358ad3f` |
| 2 | locus-tail prefix | `accumulated_coolant_absorbed_energy_of_plasma_facing_component` | `coolant_absorbed_energy_accumulated_of_plasma_facing_component` | yes | yes | none | `5d33bb2a392c1eba0a771c26eea08b226bbb8c1c92565dc6f1edf05eb24372bb` |
| 3 | locus-tail prefix | `accumulated_deposited_energy_of_plasma_facing_component` | `deposited_energy_accumulated_of_plasma_facing_component` | yes | yes | none | `080fb6823764b5c8326341e1883d0c60c7c3ad071fc6c53ab4ac22fbf98e4131` |
| 4 | locus-tail prefix | `accumulated_deuterated_methane_count_due_to_gas_injection` | `deuterated_methane_count_accumulated_due_to_gas_injection` | yes | yes | none | `46fe7fe79462dd3103a7d1be3d4fca058de0e08f8a93dc5541309010a18bb6de` |
| 5 | locus-tail prefix | `accumulated_deuterium_count_due_to_gas_injection` | `deuterium_count_accumulated_due_to_gas_injection` | yes | yes | none | `1339331773f4ff390b92af42561efae6defe78291b387cee65ca4c4c103f3277` |
| 6 | locus-tail prefix | `accumulated_helium_3_count_due_to_gas_injection` | `helium_3_count_accumulated_due_to_gas_injection` | yes | yes | none | `bfc3cd16746523de7a4426416aa08cb218eed985c8e3c9d35a64870a9dfd4442` |
| 7 | locus-tail prefix | `accumulated_helium_3_prefill_count_due_to_gas_injection` | `helium_3_prefill_count_accumulated_due_to_gas_injection` | yes | yes | none | `b70b1877a13b8b90f9658bf0a48063d4491fd32ea097c2512c7ad06d4a7f5b52` |
| 8 | locus-tail prefix | `accumulated_helium_4_count_due_to_gas_injection` | `helium_4_count_accumulated_due_to_gas_injection` | yes | yes | none | `0d678a8fe3cbf685ad6aa6614502fb5b07f386be92b4c04a3f260bb07700d607` |
| 9 | locus-tail prefix | `accumulated_hydrogen_count_due_to_gas_injection` | `hydrogen_count_accumulated_due_to_gas_injection` | yes | yes | none | `4e8018e191e7e3eec2d5580b4cb109b04bfea0effd872ce986b4c7907a1d8b63` |
| 10 | locus-tail prefix | `accumulated_lithium_count_due_to_gas_injection` | `lithium_count_accumulated_due_to_gas_injection` | yes | yes | none | `84d3af5b3222ab74de201ac81adb0cf1593873514b4e5ffd028492f30b45ea9a` |
| 11 | locus-tail prefix | `accumulated_lithium_prefill_count_due_to_gas_injection` | `lithium_prefill_count_accumulated_due_to_gas_injection` | yes | yes | none | `d643de418a7b132097ab0ed9dfc98913e0bf89870108b61c9a55e938986d55ec` |
| 12 | locus-tail prefix | `accumulated_methane_carbon_13_count_due_to_gas_injection` | `methane_carbon_13_count_accumulated_due_to_gas_injection` | yes | yes | none | `9f41d483f7ea181171cbe1639ac9e94377b62d3618b512ac5734b059d780c6ef` |
| 13 | locus-tail prefix | `accumulated_neutral_count_at_wall` | `neutral_count_accumulated_at_wall` | yes | yes | none | `bf3192c2d1eb5aaf5114f20f935e77607ef88e9c8c47427346628e5fd2c634a4` |
| 14 | locus-tail prefix | `accumulated_nitrogen_count_due_to_gas_injection` | `nitrogen_count_accumulated_due_to_gas_injection` | yes | yes | none | `0c764ce8101c09fed93494c2b396e89bc3608ce02209e1f22c174c33eade0902` |
| 15 | locus-tail prefix | `accumulated_oxygen_count_due_to_gas_injection` | `oxygen_count_accumulated_due_to_gas_injection` | yes | yes | none | `f0e8720a3cf080a16b6717a1df6459458344bbb6adc9625ab23aff677edd7422` |
| 16 | locus-tail prefix | `accumulated_oxygen_prefill_count_due_to_gas_injection` | `oxygen_prefill_count_accumulated_due_to_gas_injection` | yes | yes | none | `0dc9a51caebe5a2f9359c6782d6120d01e49cadb8a6d14aafe1e72ab30a74a54` |
| 17 | locus-tail prefix | `accumulated_particle_count_at_pellet_path_due_to_pellet_injection` | `particle_count_accumulated_at_pellet_path_due_to_pellet_injection` | yes | yes | none | `ed7b681f54f1c47707b624635c688f39e614338192a45c0463e7e9ccaa7a2593` |
| 18 | locus-tail prefix | `accumulated_propane_count_due_to_gas_injection` | `propane_count_accumulated_due_to_gas_injection` | yes | yes | none | `5003689b8d987aa8551d6ac35ccd7847f5677ba6491249d72254085b64636bf4` |
| 19 | locus-tail prefix | `accumulated_radiated_energy_due_to_impurity_radiation` | `radiated_energy_accumulated_due_to_impurity_radiation` | yes | yes | none | `e499a1af8e9935edb7fb88a779a29bd1fb9e7e1d2ce20cafee3a6494c2904d80` |
| 20 | locus-tail prefix | `accumulated_silane_count_due_to_gas_injection` | `silane_count_accumulated_due_to_gas_injection` | yes | yes | none | `a7fdd46fee9999d16a3fa489c2384222bbf38f145e4c0cc58226350255fb377f` |
| 21 | locus-tail prefix | `accumulated_total_gas_count_at_midplane_due_to_gas_injection` | `total_gas_count_accumulated_at_midplane_due_to_gas_injection` | yes | yes | none | `50a46fe133d5948e6f852bd2e5fb75dbc843a8166fc0fb49f3f30b72591abf43` |
| 22 | locus-tail prefix | `accumulated_total_particle_count_due_to_gas_injection` | `total_particle_count_accumulated_due_to_gas_injection` | yes | yes | none | `c602bbda0f4d815f346e6b137b623da3f0af5aa911f94d93e01ed4f3d68fd4ff` |
| 23 | locus-tail prefix | `accumulated_tritium_count_due_to_gas_injection` | `tritium_count_accumulated_due_to_gas_injection` | yes | yes | none | `f5534d78021a85e2d5fad34ba55b42542b7c87afdd73e7d0deec4480acaac882` |
| 24 | locus-tail prefix | `accumulated_xenon_count_due_to_gas_injection` | `xenon_count_accumulated_due_to_gas_injection` | yes | yes | none | `ce3f8aa1e86e3b5084df8b4cad42cbf426174ef83067369c01c04eb7766156a8` |
| 25 | locus-tail prefix | `cumulative_ethylene_count_due_to_gas_injection` | `ethylene_count_cumulative_due_to_gas_injection` | yes | yes | none | `bd5019a45b0aa4ea6351a19e1c6ba75f04625c99ff4233911d0355a14bccdeff` |
| 26 | locus-tail prefix | `derivative_with_respect_to_normalized_minor_radius_of_logarithm_of_density` | `derivative_of_logarithm_of_density_with_respect_to_normalized_minor_radius` | yes | yes | none | `e5e0342bcd79d73fa2bdcb9f384613986c183a71f0dd629dd492891f4ed0a096` |
| 27 | locus-tail prefix | `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface` | `derivative_of_area_of_flux_surface_with_respect_to_normalized_poloidal_flux_coordinate` | yes | yes | none | `31cebc4334201f65bb7bff530155d004c76f5dabb632e13c106020a1321c210c` |
| 28 | locus-tail prefix | `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_electron_density_at_pedestal_maximum` | `derivative_of_electron_density_at_pedestal_maximum_with_respect_to_normalized_poloidal_flux_coordinate` | yes | yes | none | `b354c6468bf4926e562a3081fc9ea998297c78864269633d80a2ce2c1e6f26e2` |
| 29 | locus-tail prefix | `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_electron_pressure_at_pedestal_top` | `derivative_of_electron_pressure_at_pedestal_top_with_respect_to_normalized_poloidal_flux_coordinate` | yes | yes | none | `13af22ae34ba3be0e45725f3dd8aa53fb7fa60368f700cdec28bb81884ae5572` |
| 30 | locus-tail prefix | `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_electron_temperature_at_pedestal_top` | `derivative_of_electron_temperature_at_pedestal_top_with_respect_to_normalized_poloidal_flux_coordinate` | yes | yes | none | `827a4e05c6ff6e61a489cc2f2436ec8be8f54906f8d238106246c4332726ec48` |
| 31 | locus-tail prefix | `derivative_with_respect_to_poloidal_angle_of_normalized_effective_particle_energy` | `derivative_of_normalized_effective_particle_energy_with_respect_to_poloidal_angle` | yes | yes | none | `3e0ee4878010d6acd312e1497374b0cb17179babfe20c91e26a5c55825e02127` |
| 32 | locus-tail prefix | `derivative_with_respect_to_poloidal_magnetic_flux_coordinate_of_volume_of_flux_surface` | `derivative_of_volume_of_flux_surface_with_respect_to_poloidal_magnetic_flux_coordinate` | yes | yes | none | `c42c93c8c4241a352903e05554ac542bacf89dc526108b2786a15bbad653581c` |
| 33 | locus-tail prefix | `derivative_with_respect_to_toroidal_flux_coordinate_of_area_of_flux_surface` | `derivative_of_area_of_flux_surface_with_respect_to_toroidal_flux_coordinate` | yes | yes | none | `ea34d14b41b9bc10ddfd669aa6dbf08aeb1d074913147ff7b2c7d842939c19e9` |
| 34 | locus-tail prefix | `derivative_with_respect_to_toroidal_flux_coordinate_of_volume_of_flux_surface` | `derivative_of_volume_of_flux_surface_with_respect_to_toroidal_flux_coordinate` | yes | yes | none | `3d06a5af301e8c61c191ceb191b966429dc190a2626f3b30ef806b2a3417bd49` |
| 35 | locus-tail prefix | `flux_surface_averaged_argon_density_at_plasma_boundary` | `argon_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `d51f00bfc55a23dbadc3a1e484fa41efaeb6793c7f9556793af24e454132153c` |
| 36 | locus-tail prefix | `flux_surface_averaged_beryllium_density_at_plasma_boundary` | `beryllium_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `ee33c74e2ca2f1b0d5d08f6097110ef001f484976e5f4dede13bd5a9d6fd9683` |
| 37 | locus-tail prefix | `flux_surface_averaged_boron_density_at_plasma_boundary` | `boron_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `7e725d62a75f99295bfdaade6d85e098710ba4de429654f4a88e3f5febe8fdc7` |
| 38 | locus-tail prefix | `flux_surface_averaged_bulk_electron_temperature_at_last_closed_flux_surface` | `bulk_electron_temperature_flux_surface_averaged_at_last_closed_flux_surface` | yes | yes | none | `c4850c0fbc8d0725c3942ce1cb68f94866c9a603091050be2b1adf5c6f17f85e` |
| 39 | locus-tail prefix | `flux_surface_averaged_carbon_density_at_plasma_boundary` | `carbon_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `3c19eca3e9e589f3215da4196f388ee09879d07a1d315addc009ce0da84a84f8` |
| 40 | locus-tail prefix | `flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | `current_density_flux_surface_averaged_due_to_wave_driven_current_drive` | yes | yes | none | `963df8080183c5ae4115bd5deec93f2649187b7c669619312299ebe9cd1372c1` |
| 41 | locus-tail prefix | `flux_surface_averaged_deuterium_density_at_plasma_boundary` | `deuterium_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `cc36e0afbd74b892e450cc3c427d7f7d7991b77df950e214cec0d100599b668f` |
| 42 | locus-tail prefix | `flux_surface_averaged_deuterium_tritium_density_at_plasma_boundary` | `deuterium_tritium_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `c3329be1902847690bc7cc58ce78c575e12125d693db160bae6fb9d64311b768` |
| 43 | locus-tail prefix | `flux_surface_averaged_effective_charge_at_plasma_boundary` | `effective_charge_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `ab6b93df9257cf0dd59a935de5172890250a4b7badfcdab1541139d9b2520966` |
| 44 | locus-tail prefix | `flux_surface_averaged_electron_density_at_plasma_boundary` | `electron_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `4989da64e60045e92f5964f17ad938e38fb69e2fd34cd5e83aa63c43a365d6e0` |
| 45 | locus-tail prefix | `flux_surface_averaged_helium_3_density_at_plasma_boundary` | `helium_3_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `2bf9d2610e11aa2fd11506fb204775aeca63ae03d99401685780bd0f48eb7e64` |
| 46 | locus-tail prefix | `flux_surface_averaged_helium_4_density_at_plasma_boundary` | `helium_4_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `bb51d68f567a963543c4ce7829a522a6f7de71a6266c41bee3a6cffb6824e179` |
| 47 | locus-tail prefix | `flux_surface_averaged_hydrogen_density_at_plasma_boundary` | `hydrogen_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `5abeac17bdf0616c3e2513620f8f99e5ebf32e2cbcfa5e14406d678ac791e3fe` |
| 48 | locus-tail prefix | `flux_surface_averaged_ion_density_at_plasma_boundary` | `ion_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `ec3ec7463c549e2ceb0d131dc929f408d20395c156a637e95120fb9eb51943f0` |
| 49 | locus-tail prefix | `flux_surface_averaged_iron_density_at_plasma_boundary` | `iron_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `567da8c7176a018d2a70e5dbfb55aa92b72363ffe10da4ceb7fa9421a76d4be0` |
| 50 | locus-tail prefix | `flux_surface_averaged_krypton_density_at_plasma_boundary` | `krypton_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `0a3cf8fb977befe0ba4f3e77c37454a81887701460223a2ad1caad242b9548b9` |
| 51 | locus-tail prefix | `flux_surface_averaged_lithium_density_at_plasma_boundary` | `lithium_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `a025ce794ac7bc76ef5cc39c910a74141cbe051426c330e5c2cc20dc0791c1b4` |
| 52 | locus-tail prefix | `flux_surface_averaged_neon_density_at_plasma_boundary` | `neon_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `f77ca62a35538fa83e180331256a367142e4775e91eb6f231f837a22da0d623c` |
| 53 | locus-tail prefix | `flux_surface_averaged_nitrogen_density_at_plasma_boundary` | `nitrogen_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `b105ea0db30a99e98e22fd6d2a202052cd532849b898ad7daf889b96e734963c` |
| 54 | locus-tail prefix | `flux_surface_averaged_oxygen_density_at_plasma_boundary` | `oxygen_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `e602ae569517d29a1d2810ca9e1a40ac48b22cd9a70928b095da538138a385ff` |
| 55 | locus-tail prefix | `flux_surface_averaged_total_ion_density_at_plasma_boundary` | `total_ion_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `1bb1854fc392480d2e60c21e70a786b21f8f7371433d36be66caaa460c217f5f` |
| 56 | locus-tail prefix | `flux_surface_averaged_tritium_density_at_plasma_boundary` | `tritium_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `93e8f3d301e653d1bcd8c78c107a2e972e239e62b58688b211460df99c374af1` |
| 57 | locus-tail prefix | `flux_surface_averaged_tungsten_density_at_plasma_boundary` | `tungsten_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `110a8f88bb09e61fdbb3a77e2b017eebcb5576a8b892762179d6c07142e8fb4d` |
| 58 | locus-tail prefix | `flux_surface_averaged_xenon_density_at_plasma_boundary` | `xenon_density_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `05a0ac810480fc9dab128d4eac6e7017fffcf4ff996982f5e51ae5d9554d1cfa` |
| 59 | locus-tail postfix | `flux_surface_normal_non_axisymmetric_vacuum_magnetic_field_at_control_surface_fourier_coefficient` | `flux_surface_normal_non_axisymmetric_vacuum_magnetic_field_fourier_coefficient_at_control_surface` | yes | yes | none | `7cae1b2b6defce6bea3180b83e67acddacfc517e376d771a023b83d8c06ebd68` |
| 60 | locus-tail prefix | `gradient_of_normalized_pressure_at_flux_surface` | `pressure_normalized_gradient_at_flux_surface` | yes | yes | none | `83c9255ba8942ea97f1bb98aeed8e515b5403221f328ed4d31b522d8016d9121` |
| 61 | locus-tail prefix | `inverse_of_curvature_of_arc_of_circle_center` | `curvature_inverse_of_arc_of_circle_center` | yes | yes | none | `5ef4d70f9d915f8d0083bf90b0a7d99204571d48de77cb9ddae1414f09a820ca` |
| 62 | locus-tail prefix | `inverse_of_tangential_curvature_of_optical_element` | `tangential_curvature_inverse_of_optical_element` | yes | yes | none | `15035737f2ede08e7ebf6c72d0c5e1d833f39fc6015e98205ae015695809c3c0` |
| 63 | locus-tail prefix | `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position` | `spectral_wave_opacity_line_integrated_at_ece_channel_emission_position` | yes | yes | none | `dd9c4cf1c983b641d6e57167e6d3ea66cd866d59ddecb23a77997fc8faebc9aa` |
| 64 | locus-tail prefix | `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel` | `spectral_signal_to_noise_ratio_logarithm_of_spectrometer_channel` | yes | yes | none | `b3425c6708a9ce20b82a8ade2371d442df362459899952ebb07328b0e593a7c5` |
| 65 | locus-tail postfix | `magnetic_field_at_pedestal_top_high_field_side_magnitude` | `magnetic_field_magnitude_at_pedestal_top_high_field_side` | yes | yes | none | `dbab757d7714ec44ff45ae4d9a50edc6fcf4b22a162ed6e1b8ae0925e9464c49` |
| 66 | locus-tail postfix | `magnetic_field_at_pedestal_top_low_field_side_magnitude` | `magnetic_field_magnitude_at_pedestal_top_low_field_side` | yes | yes | none | `2e6dcc6c00053e834176cca5ce97925a1dc3efa715d64043ec5f33aa2449a32e` |
| 67 | locus-tail prefix | `maximum_of_energy_flux_at_divertor_target` | `energy_flux_maximum_at_divertor_target` | yes | yes | none | `cd7f28e93f0f9e1aad91c17a856a797b5281418fb58ffe576a906b5d80c50257` |
| 68 | locus-tail prefix | `maximum_of_energy_flux_at_first_wall` | `energy_flux_maximum_at_first_wall` | yes | yes | none | `2c666c20bc025c7c9fe295bf22e0d8a1050c03f8d38b89610d2184d1b2993230` |
| 69 | locus-tail prefix | `maximum_of_energy_flux_at_limiter` | `energy_flux_maximum_at_limiter` | yes | yes | none | `2ed96233a9a58aa2932d1418931f73a1feaa6256a3e12201c3ff6aaba68ac4df` |
| 70 | locus-tail postfix | `neutral_velocity_due_to_diamagnetic_drift_magnitude` | `neutral_velocity_magnitude_due_to_diamagnetic_drift` | yes | yes | none | `e1d6e4458d69f5a57e225c6011033c73096d78275c2fe587ef7f0c494a9d49f8` |
| 71 | locus-tail prefix | `normalized_count_at_detector_pixel` | `count_normalized_at_detector_pixel` | yes | yes | none | `5932a3092423277f70c7e6c0e3d45244abc9babf1c90fcc0253604ed2777e978` |
| 72 | locus-tail prefix | `normalized_electron_collisionality_at_pedestal_top` | `electron_collisionality_normalized_at_pedestal_top` | yes | yes | none | `e575567281464a69634a6c05ceaace1e49d64fad5daa9acbed812197612cc241` |
| 73 | locus-tail prefix | `normalized_electron_larmor_radius_at_pedestal_top_high_field_side` | `electron_larmor_radius_normalized_at_pedestal_top_high_field_side` | yes | yes | none | `f45fd296702d84a805c54968396ccd4e97c7085c6b3a567783818fa217f3efce` |
| 74 | locus-tail prefix | `normalized_electron_larmor_radius_at_pedestal_top_low_field_side` | `electron_larmor_radius_normalized_at_pedestal_top_low_field_side` | yes | yes | none | `eff7da7b9e88e33f706f4bf1668e486d3a1381400af420f37beaf9f1583233f7` |
| 75 | locus-tail prefix | `normalized_energy_flux_due_to_e_cross_b_drift` | `energy_flux_normalized_due_to_e_cross_b_drift` | yes | yes | none | `87fed5342c840bb0350cf3ae55c27e1be732da5522506a8bfc097d57895bf369` |
| 76 | locus-tail prefix | `normalized_energy_flux_due_to_perturbed_parallel_magnetic_field` | `energy_flux_normalized_due_to_perturbed_parallel_magnetic_field` | yes | yes | none | `14af7162aec8f0e23708e052dff966ac5b168ea3f5f561e278df4d28af84e2ff` |
| 77 | locus-tail prefix | `normalized_energy_flux_due_to_perturbed_parallel_vector_potential` | `energy_flux_normalized_due_to_perturbed_parallel_vector_potential` | yes | yes | none | `3b939466e0dc6d14e0998020469bc3b0010cf1cd05865c18995a6a8cc7ff7ff1` |
| 78 | locus-tail prefix | `normalized_frequency_of_gyrokinetic_eigenmode` | `frequency_normalized_of_gyrokinetic_eigenmode` | yes | yes | none | `edf4bd246fdfa1f51a4707c873076ab24a1134a673550cf8914d31ee1cf23559` |
| 79 | locus-tail prefix | `normalized_linear_growth_rate_of_gyrokinetic_eigenmode` | `linear_growth_rate_normalized_of_gyrokinetic_eigenmode` | yes | yes | none | `fbceb8be6cbedfce1d8971de58673339e1153dd24039e4f77fca6ba9758d45b7` |
| 80 | locus-tail prefix | `normalized_momentum_flux_due_to_e_cross_b_drift` | `momentum_flux_normalized_due_to_e_cross_b_drift` | yes | yes | none | `35fd076d52b2c9f8280d55fe0809ecd3b867651087f583eb02060537c21f78f5` |
| 81 | locus-tail prefix | `normalized_momentum_flux_due_to_perturbed_parallel_magnetic_field` | `momentum_flux_normalized_due_to_perturbed_parallel_magnetic_field` | yes | yes | none | `67edf324581f8efaee6675b612cf76788d68815d765179e7ec3c342b4fb8a43f` |
| 82 | locus-tail prefix | `normalized_particle_flux_due_to_e_cross_b_drift` | `particle_flux_normalized_due_to_e_cross_b_drift` | yes | yes | none | `6826d17d97bf0e2c76469f04beacdad3b691ccc2b88209afbb461c45f763e7cb` |
| 83 | locus-tail prefix | `normalized_particle_flux_due_to_perturbed_parallel_magnetic_field` | `particle_flux_normalized_due_to_perturbed_parallel_magnetic_field` | yes | yes | none | `d725787961013ed3af73ccbf2d797cf3cc34765ede96ca1a8b375b3bc877930b` |
| 84 | locus-tail prefix | `normalized_particle_flux_due_to_perturbed_parallel_vector_potential` | `particle_flux_normalized_due_to_perturbed_parallel_vector_potential` | yes | yes | none | `c771fa55bf8f9b454d7bfd86e764ec497f05194e23a7ddb7879cc89c6fdb9f53` |
| 85 | locus-tail prefix | `normalized_saturated_permeability_of_ferritic_element` | `saturated_permeability_normalized_of_ferritic_element` | yes | yes | none | `051b0c3b37cad3d69e734b1b74e9a810c63c7cba622928165c16fac5197c758d` |
| 86 | locus-tail prefix | `normalized_shearing_rate_due_to_e_cross_b_drift` | `shearing_rate_normalized_due_to_e_cross_b_drift` | yes | yes | none | `05e672905fcb547c05af2557021e3e3a0a7a5131392b414e67f9e6a47cad48b7` |
| 87 | locus-tail prefix | `normalized_total_momentum_flux_due_to_perturbed_parallel_vector_potential` | `total_momentum_flux_normalized_due_to_perturbed_parallel_vector_potential` | yes | yes | none | `acbd8ed571c53b94e0fd0ef1ce1d2ce3b4daea5b2c95a00573ab536ee354ece4` |
| 88 | locus-tail prefix | `normalized_total_particle_perturbed_pressure_of_gyrokinetic_eigenmode` | `total_particle_perturbed_pressure_normalized_of_gyrokinetic_eigenmode` | yes | yes | none | `a13dc042ea3f5d102523f8f6e10a1e25bc53f7d00d13252cdbfc45f55e5f3005` |
| 89 | locus-tail prefix | `parallel_flux_surface_averaged_current_density_at_constraint_position` | `parallel_current_density_flux_surface_averaged_at_constraint_position` | yes | yes | none | `a820af8a42c2f141cfb7d2df05780ed227e5589761ef6a4387f7cc3a68ec9709` |
| 90 | locus-tail prefix | `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | `parallel_current_density_flux_surface_averaged_due_to_wave_driven_current_drive` | yes | yes | none | `af8bfa611e87e3d79dd6af21115f55f5ee88ddccd9ca9c97ddf51f8d9ec2fd7a` |
| 91 | locus-tail prefix | `parallel_flux_surface_averaged_electric_field_at_separatrix` | `parallel_electric_field_flux_surface_averaged_at_separatrix` | yes | yes | none | `39682eee89aeddaa3d894d6e497adde5ef133d2ff5b508d11ede9f58e6f6fc1d` |
| 92 | locus-tail prefix | `parallel_normalized_gyrocenter_momentum_flux_of_gyrokinetic_eigenmode_due_to_e_cross_b_drift` | `parallel_gyrocenter_momentum_flux_normalized_of_gyrokinetic_eigenmode_due_to_e_cross_b_drift` | yes | yes | none | `90a5a76d104671c09b8ff6c1796b4979433570ae8c431ad52cbc9d9663d2382b` |
| 93 | locus-tail prefix | `parallel_normalized_gyrocenter_momentum_flux_of_gyrokinetic_eigenmode_due_to_perturbed_parallel_magnetic_field` | `parallel_gyrocenter_momentum_flux_normalized_of_gyrokinetic_eigenmode_due_to_perturbed_parallel_magnetic_field` | yes | yes | none | `58414e621c216b4687571befb93f86f17c561d88f2848ba340f869afa6c91da7` |
| 94 | locus-tail prefix | `parallel_normalized_gyrocenter_momentum_flux_of_gyrokinetic_eigenmode_due_to_perturbed_parallel_vector_potential` | `parallel_gyrocenter_momentum_flux_normalized_of_gyrokinetic_eigenmode_due_to_perturbed_parallel_vector_potential` | yes | yes | none | `4f5b96a1c7e179a72e6b96037532d3a7c0a11c76b1b7baef4ba52fefc5814de0` |
| 95 | locus-tail prefix | `parallel_normalized_momentum_flux_due_to_perturbed_parallel_magnetic_field` | `parallel_momentum_flux_normalized_due_to_perturbed_parallel_magnetic_field` | yes | yes | none | `f530eb7c098b41fdfa6e6ddb28b1f465db3e9e66e006796cb797d9aa7094edf1` |
| 96 | locus-tail prefix | `parallel_normalized_momentum_flux_due_to_perturbed_parallel_vector_potential` | `parallel_momentum_flux_normalized_due_to_perturbed_parallel_vector_potential` | yes | yes | none | `3ae3ba028466f0127a58671e73616fe832992bfb3f4350d9769c0d93c2ff23f7` |
| 97 | locus-tail prefix | `parallel_normalized_power_of_beam_tracing_beam` | `parallel_power_normalized_of_beam_tracing_beam` | yes | yes | none | `20d708b4eeb9e35551e4f9016d63e4dffd4cd8046d63e32909c8becf65c8a8b7` |
| 98 | locus-tail prefix | `parallel_per_toroidal_mode_current_density_due_to_wave_driven_current_drive` | `parallel_current_density_per_toroidal_mode_due_to_wave_driven_current_drive` | yes | yes | none | `209ad8fc7a324910945ffd06f51cbcbeb03863995815f1c8f8f575d680e030ec` |
| 99 | locus-tail postfix | `peak_wave_current_of_antenna_strap_amplitude` | `peak_wave_current_amplitude_of_antenna_strap` | yes | yes | none | `075e7ef58b792d7dccf4ef98751708e402c341f17c13e907a3bddfd89eb267b9` |
| 100 | locus-tail prefix | `per_toroidal_and_poloidal_mode_number_launched_power_of_lower_hybrid_antenna` | `launched_power_per_toroidal_and_poloidal_mode_number_of_lower_hybrid_antenna` | yes | yes | none | `add83c8ff9adc7b7ef1ff525f16669c4345be41d85b6b256e4ec46868a6f24ff` |
| 101 | locus-tail prefix | `per_toroidal_and_poloidal_mode_number_surface_current_of_ion_cyclotron_heating_antenna` | `surface_current_per_toroidal_and_poloidal_mode_number_of_ion_cyclotron_heating_antenna` | yes | yes | none | `74353380b3abc8d6da7052bca4237502652353bee3f5fa400ff11cd84138a166` |
| 102 | locus-tail prefix | `per_toroidal_mode_current_due_to_wave_driven_current_drive` | `current_per_toroidal_mode_due_to_wave_driven_current_drive` | yes | yes | none | `399551acdb575a91965eb320af409270a9f942c08d2d101d234879bfbbe41fc6` |
| 103 | locus-tail prefix | `per_toroidal_mode_launched_power_of_lower_hybrid_antenna` | `launched_power_per_toroidal_mode_of_lower_hybrid_antenna` | yes | yes | none | `e2bbaa4d985b8e81d778ac5228338577c8014c200ad098dc6aa7f5d7c3adb538` |
| 104 | locus-tail prefix | `perpendicular_normalized_gyrocenter_heat_perturbed_flux_of_gyrokinetic_eigenmode` | `perpendicular_gyrocenter_heat_perturbed_flux_normalized_of_gyrokinetic_eigenmode` | yes | yes | none | `3f9d2214f2e1c4544610a8e984ec0faf7c3c0215d7322af764bad84c716e795a` |
| 105 | locus-tail prefix | `perpendicular_normalized_momentum_flux_due_to_e_cross_b_drift` | `perpendicular_momentum_flux_normalized_due_to_e_cross_b_drift` | yes | yes | none | `4a43e4f3718fc78b12cfe3e82ab8fc9733026903b708a9ce1bbe5e0a6c227821` |
| 106 | locus-tail prefix | `perpendicular_normalized_momentum_flux_due_to_perturbed_parallel_magnetic_field` | `perpendicular_momentum_flux_normalized_due_to_perturbed_parallel_magnetic_field` | yes | yes | none | `3acc330354f7bc221d307de3ffa6410c01c4a6cc00a1ab6ee607564bca584636` |
| 107 | locus-tail prefix | `perpendicular_normalized_momentum_flux_due_to_perturbed_parallel_vector_potential` | `perpendicular_momentum_flux_normalized_due_to_perturbed_parallel_vector_potential` | yes | yes | none | `ec37336b1f28c4d00c82bc2efa8f82a4297c63c10fd68531fcd18424c27565f9` |
| 108 | locus-tail prefix | `perpendicular_normalized_power_of_beam_tracing_beam` | `perpendicular_power_normalized_of_beam_tracing_beam` | yes | yes | none | `8309efd06f8937ea2c5b449c9e918e1cd6a65de45531c98e6da8ff1bd6f2c011` |
| 109 | locus-tail prefix | `poloidal_flux_surface_averaged_electron_beta_at_pedestal_top` | `poloidal_electron_beta_flux_surface_averaged_at_pedestal_top` | yes | yes | none | `fbd310de14525f3470b3708bc46138bf8c647bc2bdb4cbe3a1f4beca840311b2` |
| 110 | locus-tail prefix | `poloidal_perturbed_magnetic_flux_at_measurement_position_due_to_wave_particle_interaction` | `poloidal_magnetic_flux_perturbed_at_measurement_position_due_to_wave_particle_interaction` | yes | yes | none | `63fe0d7682051dca8afe995692ce8e82214267dd85c63103f9ae3e3fa74ef5a9` |
| 111 | locus-tail prefix | `poloidal_perturbed_suprathermal_electron_angle_at_measurement_position` | `poloidal_suprathermal_electron_angle_perturbed_at_measurement_position` | yes | yes | none | `b4bc0a4dc9785e18379ffce738b3d91157c794d57a277f9ec6ce6701d2ec0417` |
| 112 | indexed operator | `product_of_poloidal_current_function_and_derivative_with_respect_to_poloidal_magnetic_flux_coordinate_of_poloidal_current_function` | `product_of_poloidal_current_function_and_derivative_of_poloidal_current_function_with_respect_to_poloidal_magnetic_flux_coordinate` | yes | yes | none | `eb3c743ca7bd4562fdd5d6275f072b9f1504b3dffabe56a1aea5eb7d0def9559` |
| 113 | locus-tail prefix | `radial_derivative_of_elongation_of_flux_surface` | `elongation_radial_derivative_of_flux_surface` | yes | yes | none | `f33a3c2b0f0906a47db8102b16b24fd3c7c766581fafe3e429891897fa9a702d` |
| 114 | locus-tail prefix | `radial_normalized_wave_vector_of_beam_tracing_beam` | `radial_wave_vector_normalized_of_beam_tracing_beam` | yes | yes | none | `16e245e274d8bd21013791099f5bfd0ead75270c8307868541e0b1b7c29caefa` |
| 115 | locus-tail prefix | `root_mean_square_of_spectral_width_of_spectrometer_channel` | `spectral_width_root_mean_square_of_spectrometer_channel` | yes | yes | none | `2df3d1652f2f095fb32090ab694d909f76020c2d44cab1d46c5fb5467f649cc7` |
| 116 | locus-tail prefix | `root_mean_square_of_wave_current_of_antenna_strap` | `wave_current_root_mean_square_of_antenna_strap` | yes | yes | none | `018873a37a93d8b41132bac881ff5107186ba9d049e7a7bdfd3ea8c90ea9572d` |
| 117 | locus-tail prefix | `time_derivative_of_derivative_with_respect_to_toroidal_flux_coordinate_of_volume_of_flux_surface` | `time_derivative_of_derivative_of_volume_of_flux_surface_with_respect_to_toroidal_flux_coordinate` | yes | yes | none | `059fefa60610be8affb4e96875fbc24fe89effde7a10cc9f6d5f60be06356ef5` |
| 118 | locus-tail prefix | `time_derivative_of_radial_width_of_neoclassical_tearing_mode` | `radial_width_time_derivative_of_neoclassical_tearing_mode` | yes | yes | none | `3ce117f2520414e004c18121967c771f032c2cd209b3389d986ee81829c50152` |
| 119 | locus-tail prefix | `time_derivative_of_rotation_frequency_of_neoclassical_tearing_mode` | `rotation_frequency_time_derivative_of_neoclassical_tearing_mode` | yes | yes | none | `2572c3052eb940d5eef4ad5d830fca3e2775fdf41668e6ee434ede3fd79da989` |
| 120 | locus-tail prefix | `toroidal_cumulative_inside_flux_surface_total_plasma_momentum_at_separatrix` | `toroidal_total_plasma_momentum_cumulative_inside_flux_surface_at_separatrix` | yes | yes | none | `9a803a633bb4d557a0d46942795d57f8dd5666a340643595ca34725be9a09bb9` |
| 121 | locus-tail prefix | `toroidal_flux_surface_averaged_argon_velocity_at_plasma_boundary` | `toroidal_argon_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `517b574ee24c24cea59a8cffef059341b07c707aad5d383185e3091ea92c36d9` |
| 122 | locus-tail prefix | `toroidal_flux_surface_averaged_beryllium_velocity_at_plasma_boundary` | `toroidal_beryllium_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `ca46a5b1f9c193d52fc40db3cabcffc9be0cbf0c4b0b6fdd4d21539998b28438` |
| 123 | locus-tail prefix | `toroidal_flux_surface_averaged_carbon_velocity_at_plasma_boundary` | `toroidal_carbon_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `3fbae3494234aa29ad7415f3532af3c6e1cf0a61466d7b3295d7488a4905f70d` |
| 124 | locus-tail prefix | `toroidal_flux_surface_averaged_current_density_at_constraint_position` | `toroidal_current_density_flux_surface_averaged_at_constraint_position` | yes | yes | none | `b31709c159911666cadc29ae4741aade6ecfc458c738da663211fe07dd61e016` |
| 125 | locus-tail prefix | `toroidal_flux_surface_averaged_deuterium_tritium_velocity_at_plasma_boundary` | `toroidal_deuterium_tritium_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `28c0c24259882b746bf4906c2c8cb7703d04cdea3eae312b91d5dc7d4d922d2a` |
| 126 | locus-tail prefix | `toroidal_flux_surface_averaged_deuterium_velocity_at_plasma_boundary` | `toroidal_deuterium_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `afe96e9ceea24c32ae4015f603e13a03e1f05792cd52e12b1e50a2206db0709f` |
| 127 | locus-tail prefix | `toroidal_flux_surface_averaged_helium_3_velocity_at_plasma_boundary` | `toroidal_helium_3_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `4a96bd08f8ba0b2a4b86e4b80bb98fa8a580073e3000da18c3aa16bf5b12126d` |
| 128 | locus-tail prefix | `toroidal_flux_surface_averaged_helium_4_velocity_at_plasma_boundary` | `toroidal_helium_4_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `487c834b8df6eb48bd9b8b6335c373d8916c0bb7836412cfbd65df1803db66c9` |
| 129 | locus-tail prefix | `toroidal_flux_surface_averaged_hydrogen_velocity_at_plasma_boundary` | `toroidal_hydrogen_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `dc7eb9f0344c241ae1e3f56a8ddb1ab617f6636dc0b10e6eab48edebdc787929` |
| 130 | locus-tail prefix | `toroidal_flux_surface_averaged_ion_velocity_at_plasma_boundary` | `toroidal_ion_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `b12f984f377eea26f36207413154f76fa782b49f75362272e650724190babfbe` |
| 131 | locus-tail prefix | `toroidal_flux_surface_averaged_iron_velocity_at_plasma_boundary` | `toroidal_iron_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `18415cd6e2efe49e27cd32c58a56e824588910cc9f565f794330b4c96a64e759` |
| 132 | locus-tail prefix | `toroidal_flux_surface_averaged_krypton_velocity_at_plasma_boundary` | `toroidal_krypton_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `5d2c4f24ac5ec9c24c13ff6cefa02f2507589915f8aec61b893022c17155a029` |
| 133 | locus-tail prefix | `toroidal_flux_surface_averaged_lithium_velocity_at_plasma_boundary` | `toroidal_lithium_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `9c2e2d08a6ef97ad98a8f76a477f621258b33767588c448ca66e17b633650dad` |
| 134 | locus-tail prefix | `toroidal_flux_surface_averaged_neon_velocity_at_plasma_boundary` | `toroidal_neon_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `0258e75e09cb01cab9b10c50793a594f9750434a714d535aaf274d90c7de1c5c` |
| 135 | locus-tail prefix | `toroidal_flux_surface_averaged_nitrogen_velocity_at_plasma_boundary` | `toroidal_nitrogen_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `76716c77c607f537b4076ed69b46d8c0bb3d02103e1aa95348b7a20b7e5e9908` |
| 136 | locus-tail prefix | `toroidal_flux_surface_averaged_oxygen_velocity_at_plasma_boundary` | `toroidal_oxygen_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `3162e532a244940f92e1e99346706c988236089e53d18bfd08edc4beb1dd0dd5` |
| 137 | locus-tail prefix | `toroidal_flux_surface_averaged_total_plasma_momentum_at_plasma_boundary` | `toroidal_total_plasma_momentum_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `e5c3c660cab770feb68f01f65a5ab8b1b1445292f00380ccbad28c404fb1184e` |
| 138 | locus-tail prefix | `toroidal_flux_surface_averaged_tritium_velocity_at_plasma_boundary` | `toroidal_tritium_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `03d8f90fb1697ce7285a4e635a6c55ce8d4e027013ce8ffff62b1a4eda8486d6` |
| 139 | locus-tail prefix | `toroidal_flux_surface_averaged_tungsten_velocity_at_plasma_boundary` | `toroidal_tungsten_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `1fa55cb1082b5bc1cb4cdc215f85b5a5ce722f75b3f7cce8a313c010f46f9113` |
| 140 | locus-tail prefix | `toroidal_flux_surface_averaged_xenon_velocity_at_plasma_boundary` | `toroidal_xenon_velocity_flux_surface_averaged_at_plasma_boundary` | yes | yes | none | `94259b08a245621ccaac2d9acbbdf72cb4195d365dc9ee8089073cc5cb50e62b` |
| 141 | locus-tail prefix | `toroidal_normalized_wave_vector_of_beam_tracing_beam` | `toroidal_wave_vector_normalized_of_beam_tracing_beam` | yes | yes | none | `617523969a3af9bd5350b67d7265302bebb27bf1b0678a1b0ace0f6da68e1ca9` |
| 142 | locus-tail prefix | `toroidal_perturbed_suprathermal_electron_angle_at_measurement_position` | `toroidal_suprathermal_electron_angle_perturbed_at_measurement_position` | yes | yes | none | `626e1352a31e44f14aca7118e462f165b83051bf09c910ae66799c55103c1023` |
| 143 | locus-tail prefix | `toroidal_volume_integrated_fast_electron_torque_density_due_to_collisions` | `toroidal_fast_electron_torque_density_volume_integrated_due_to_collisions` | yes | yes | none | `047080fbfe98bb2ffbf55499e2503798a7815131e186eb893309516f837c43c8` |
| 144 | locus-tail prefix | `variation_of_length_of_interferometer_beam` | `length_variation_of_interferometer_beam` | yes | yes | none | `2a424cd73d919a77c7185d056ac3f7514e18f1f526d2659c7791ef4cbeb56664` |
| 145 | locus-tail prefix | `vertical_normalized_wave_vector_of_beam_tracing_beam` | `vertical_wave_vector_normalized_of_beam_tracing_beam` | yes | yes | none | `5edf60835ab9583b0e53c9a057ef2a80ddc14bb7f337ce5c9d9d2c15feb29d53` |
| 146 | locus-tail postfix | `voltage_of_ion_cyclotron_heating_antenna_amplitude` | `voltage_amplitude_of_ion_cyclotron_heating_antenna` | yes | yes | none | `3e7a3f92f7258b532e9d0553186376e57f22fcabbc744db5a18236ec47ee8fdf` |
| 147 | locus-tail prefix | `volume_averaged_linear_thermal_electron_decay_time_due_to_disruption` | `linear_thermal_electron_decay_time_volume_averaged_due_to_disruption` | yes | yes | none | `7f59a560717d29a6dbcc749ae78f099b9c7640f931aae2a45c2d5ea0228f5e66` |
| 148 | locus-tail prefix | `volume_averaged_runaway_electron_critical_momentum_due_to_avalanche` | `runaway_electron_critical_momentum_volume_averaged_due_to_avalanche` | yes | yes | none | `4ccd3d233ea4836e861ebb80d684a7082dace084ef1826edd96cfe9b832f7363` |
| 149 | locus-tail postfix | `wave_current_of_antenna_strap_amplitude` | `wave_current_amplitude_of_antenna_strap` | yes | yes | none | `981553279eec570f38d080460c45eb8eadc5a8701d4342d01a5bfd9564b1c2ac` |

## Lenient parse failures

These seven pre-existing spellings are outside the deterministic rename set and are not staged.

| Identity | Error type | Error |
|---|---|---|
| `flux_surface_average_magnetic_field_magnitude` | `ParseError` | residue 'flux_surface_average_magnetic_field' does not match any physical_base or geometry_carrier; nearest candidates: ['flux_surface_averaged_metric'] |
| `inertial_current_density_due_to_diamagnetic_drift` | `ParseError` | residue 'inertial_current_density' does not match any physical_base or geometry_carrier; nearest candidates: ['current_density'] |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | `ParseError` | residue 'inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width' does not match any physical_base or geometry_carrier; nearest candidates: ['normalized_toroidal_flux_coordinate'] |
| `normalized_perpendicular_gyroaveraged_perturbed_energy` | `ParseError` | residue 'normalized_perpendicular_gyroaveraged_perturbed_energy' does not match any physical_base or geometry_carrier; nearest candidates: (none) |
| `per_toroidal_mode_flux_surface_average_total_absorbed_power_density` | `ParseError` | residue 'per_toroidal_mode_flux_surface_average_total_absorbed_power_density' does not match any physical_base or geometry_carrier; nearest candidates: (none) |
| `tendency_of_runaway_electron_density` | `ParseError` | residue 'tendency_of_runaway_electron_density' does not match any physical_base or geometry_carrier; nearest candidates: (none) |
| `toroidal_angle_of_along_pellet_path` | `ParseError` | residue 'toroidal_angle_of' does not match any physical_base or geometry_carrier; nearest candidates: ['toroidal_angle', 'poloidal_angle', 'tilt_angle'] |

## Collision and merge detail

No proposal collides with a live or retired identity. No two old identities compose to the same proposal.

## Migration status

Blocked before staging. All 149 rows were exercised through the exact no-write route:

`env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync imas-codex sn edit OLD --rename NEW --reason 'canonical renderer migration; semantic IR unchanged' --scope self --stage-only --dry-run`

Result: 148 passed and 1 was refused. Staged renames: **0**. A partial 148-name migration was not taken because the plan makes the whole live cohort the semantic scope, and bypassing the rename unit-authority guard is not an authorised route.

Postflight after all dry-runs:

| Measure | Result |
|---|---:|
| Old identities still live | 149 / 149 |
| Old identities superseded | 0 |
| Proposed identities created | 0 |
| Descriptions changed | 0 |
| Documentation changed | 0 |
| LLM ledger rows before / after | 36,914 / 36,914 |
| LLM ledger total before / after | $1,777.493812999981 / $1,777.493812999981 |

### Refused route

Row 67 is semantically safe by the renderer gates—strict reparse succeeds, its IR bytes are identical, and its proposal does not collide—but `sn edit` refuses before mutation because the four authoritative DD sources do not agree on a unit:

- Stored spelling: `maximum_of_energy_flux_at_divertor_target`
- Proposed spelling: `energy_flux_maximum_at_divertor_target`
- Description: “Maximum local energy-deposition rate per unit area on a divertor target, representing the peak combined load from incident plasma and radiative energy carriers.”
- Refusal: `rename unit derivation refused: DD source cohort disagrees on unit authority: ['W', 'W.m^-2']`

| Source-path binding | Source status | DD scalar unit | HAS_UNIT authority |
|---|---|---|---|
| `divertors/divertor/target/power_flux_peak` | attached | `W.m^-2` | `W.m^-2` |
| `summary/local/divertor_target/power_flux_peak/value` | composed | `W.m^-2` | `W.m^-2` |
| `wall/global_quantities/power_density_inner_target_max` | attached | `W` | `W` |
| `wall/global_quantities/power_density_outer_target_max` | attached | `W` | `W` |

The immediate follow-on is to adjudicate and repair the two `wall/global_quantities/power_density_*_target_max` bindings or their DD unit authority, then rerun row 67's dry-run and the complete staged migration. This report does not choose between a unit repair and a source-binding repair because that physics decision is outside the assigned report-only write scope.
