# Deterministic validation restored the observed verdicts

## Result

The deterministic admission gate re-established 525 of the 538 initially
unstamped `valid` verdicts. It did not backfill a timestamp: every changed
row passed through `drain_validation_for_ids`, which writes its verdict,
issues, and `validated_at` together.

The fourteen names in the review cut were processed first. Their exact result
was `validated=14`, `quarantined=0`, `requarantined_ids=[]`, and
`cleared_ids=[radial_outline_of_antenna_strap,
toroidal_coordinate_of_camera, vertical_outline_of_plasma_facing_component,
poloidal_length_of_flux_surface, lower_bound_photon_energy,
minimum_over_flux_surface_magnetic_field_magnitude,
toroidal_angle_of_poloidal_magnetic_field_probe,
incident_soft_xray_radiance, total_power_due_to_ohmic_dissipation,
ipb98y2_confinement_time, radiative_temperature_at_ece_channel,
intensity_at_spectral_line, total_electron_count,
vertical_coordinate_of_ece_channel]`. No review-cut identity re-quarantined.

The remaining cohort returned `validated=511`, `quarantined=37`,
`requarantined_ids` containing 37 identities, and 474 `cleared_ids`. The exact
per-call arrays, including every cleared identity, are preserved in the
durable drain logs named in the worker manifest. The 37 re-quarantines are the
finding the prior scalar could not supply; they were not washed back to valid.

| Population | Before | After |
|---|---:|---:|
| `valid` with null `validated_at` | 538 | 13 |
| Review-cut identities in that population | 14 | 0 |
| Valid identities re-established by the gate | 0 | 488 |
| Re-quarantined by the gate | 0 | 37 |

## Residual blocker

The remaining 13 rows still make the invariant false. Each has
`validation_status='valid'`, a null `validated_at`, no active claim, and a
null description. The gate is intentionally unable to admit them because
`claim_ids_for_validation` requires `sn.description IS NOT NULL` before taking
its claim. They therefore cannot receive an honest new observation through
this route, and writing `validated_at` directly would create the false record
this work avoids.

| Name stage | Identities |
|---|---|
| `superseded` | `breakdown_time`, `calibration_spectral_coefficient_of_line_of_sight`, `coolant_delay`, `radial_gap`, `ratio_of_ion_velocity_to_magnetic_field_strength` |
| `pending` | `line_averaged_neon_density`, `poloidal_ion_state_momentum_diffusion_coefficient`, `radial_coordinate_of_reflector`, `radius_of_soft_xray_detector`, `ratio_of_diamagnetic_vorticity_to_major_radius`, `toroidal_coordinate_of_spectrometer`, `toroidal_tritium_velocity`, `vertical_coordinate_of_reflector` |

The next repair must decide how a no-description identity is classified before
it may hold any current validation verdict. It belongs in the validation claim
or lifecycle owner, not in this evidence-only node. The export fence continues
to withhold every undated `valid` row; none of the residual rows is currently
accepted or approved.

## Measurement

The before census found 5,048 identities, 538 `valid` rows without an
observation, 261 otherwise publishable unstamped rows, and 2,058 equivalent
stamped valid rows. Its two bounded graph queries took 0.038 and 0.015
seconds. The final residual census took 0.043 seconds and found 13 unstamped
valid rows, 4,378 stamped valid rows, and 42 stamped quarantined rows.

Each graph operation ran on the login node because the authenticated graph
route is local to that host. Priority-drain elapsed time was 7.484 seconds;
the eleven remaining-drain calls each used at most 50 named identities and
completed in 7.126 to 10.353 seconds.
