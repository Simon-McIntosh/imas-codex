# Accepted-name documentation remainder drain

## Outcome

The fresh global census of accepted Standard Names without accepted
documentation fell from **54 exact `StandardName.id` identities to 33**. The
paid work spent **USD 10.574555 of the hard USD 60.00 node ceiling**, leaving
USD 49.425445 unused. The documentation work charged to this release-readiness
section now totals **USD 107.654472 of the authorized USD 200.00**:

- USD 39.980907 for the first 106-identity accepted-documentation campaign;
- USD 57.099010 for the global accepted-name documentation drain;
- USD 10.574555 for this remainder drain.

The authorized campaign balance is therefore **USD 92.345528**. This is a
qualified result, not a release pass: all 13 newly pending documents cleared,
and all 34 terminal quorum shortfalls received a fresh full-quorum outcome,
but 33 identities still do not have accepted documentation.

| Measure | Before | After |
|---|---:|---:|
| `StandardName` population | 4,656 | 4,656 |
| Accepted names | 2,295 | 2,295 |
| Accepted names without `docs_stage = 'accepted'` | **54** | **33** |
| Distinct backlog `StandardName.id` values | **54** | **33** |
| `pending` | 13 | 0 |
| `reviewed` | 35 | 27 |
| `exhausted` | 6 | 6 |

Both censuses proved the filtered properties before trusting the counts:
4,656 of 4,656 `StandardName` rows carried `id` and `name_stage`, 4,652 carried
`docs_stage`, and all 2,295 accepted rows carried `id`. Row count and
`count(DISTINCT s.id)` agreed exactly at both endpoints. All 54 starting
identities remained present and accepted, 21 gained accepted documentation,
33 remained in the backlog, and no new backlog identity appeared during the
window. No starting or ending backlog identity had a live claim, and no other
`SNRun` was active at either boundary.

## Bar and sanctioned recovery route

The bar remained the live production documentation quorum at **0.85**. No
threshold, reviewer chain, acceptance rule, or refinement cap was weakened.
The 34 reviewed identities carrying
`docs_review_quorum_shortfall = "blind seats disagreed and the escalator seat
did not resolve them"` were staged with the sanctioned
`stage_docs_for_rescore()` compare-and-set primitive. It preserved their
documentation text, description, refinement history, review nodes, and graph
bindings while moving only the aggregate docs decision back to `drafted` for
one fresh quorum draw.

All 34 dry runs and all 34 applies succeeded. The binding census was unchanged
across the transition: 73 incoming source bindings and 34 unit bindings before
and after, with all 34 rows drafted, exact-scope stamped, claim-free, and clear
of the prior shortfall marker before the paid run began. No score, resolution
method, or accepted stage was written by the staging operation.

The paid run used the exact frozen set of those 34 rows plus the 13 pending
rows:

```text
imas-codex sn run --docs-only --name <47 exact ids> \
  --min-score 0.85 --cost-limit 60 --time 25 \
  --skip-global-maintenance
```

Its durable `SNRun` is `8a370837-c1f0-493c-97b0-d0995bf3d7ed`. It ran from
2026-08-25 04:17:48Z to 04:31:32Z. Once the scoped pending counters reached
zero with zero in-flight work, it received a graceful interrupt; the durable
status is therefore honestly `interrupted`, not relabeled complete. The run's
cost mirror is exact.

| Pool | Events | USD |
|---|---:|---:|
| Documentation generation | 13 | 0.086215 |
| Documentation review | 125 | 9.978425 |
| Documentation refinement | 5 | 0.509915 |
| **Total** | **143** | **10.574555** |

The run generated all 13 pending documents, completed 52 documentation review
actions and 5 refinement actions, and left no scoped claim. Spend was a review
cost again: review consumed **94.36%** of this run's total.

## Every terminal quorum-shortfall outcome

Every one of the 34 terminal shortfalls obtained a new winning review group.
The latest groups split exactly **17 `quorum_consensus` and 17
`authoritative_escalation`**. Eight identities cleared the 0.85 bar and became
docs-accepted. Twenty-six received a real quorate score below the bar and now
remain reviewed. None retained `docs_review_quorum_shortfall`, so no row below
is being reported as a quality failure while its reviewers still disagree.

| `StandardName.id` | Prior score | Fresh score | Outcome and disposition |
|---|---:|---:|---|
| `binormal_wave_electric_field` | 0.85625 | 0.83750 | Quorate below bar; HOLD for governed content refinement. |
| `counter_passing_current_density` | 0.40000 | 0.35625 | Quorate below bar; HOLD for governed content refinement. |
| `counter_passing_thermal_ion_torque_density_due_to_collisions` | 0.83750 | 0.88750 | **Accepted on fresh quorum.** |
| `current_density_due_to_fast_ion` | 0.72500 | 0.78750 | Quorate below bar; HOLD for governed content refinement. |
| `current_of_antenna_strap` | 0.61250 | 0.82500 | Quorate below bar; HOLD for governed content refinement. |
| `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface` | 0.59375 | 0.57500 | Quorate below bar; HOLD for governed content refinement. |
| `energy_flux_at_limiter` | 0.37500 | 0.32500 | Quorate below bar; HOLD for governed content refinement. |
| `fluctuating_ion_current_density` | 0.35625 | 0.30000 | Quorate below bar; HOLD for governed content refinement. |
| `flux_due_to_fusion` | 0.68125 | 0.60000 | Quorate below bar; HOLD for governed content refinement. |
| `ion_absorbed_power_of_beam_tracing_beam` | 0.88750 | 0.89375 | **Accepted on fresh quorum.** |
| `ion_average_temperature` | 0.84375 | 0.87500 | **Accepted on fresh quorum.** |
| `ion_charge_state_energy_velocity_due_to_convection` | 0.63750 | 0.82500 | Quorate below bar; HOLD for governed content refinement. |
| `ion_velocity_due_to_diamagnetic_drift` | 0.78750 | 0.83750 | Quorate below bar; HOLD for governed content refinement. |
| `major_length_of_antenna_strap` | 0.66250 | 0.56250 | Quorate below bar; HOLD for governed content refinement. |
| `mhd_mode_reference_phase` | 0.90625 | 0.92500 | **Accepted on fresh quorum.** |
| `neutral_particle_convection_velocity` | 0.55625 | 0.77500 | Quorate below bar; HOLD for governed content refinement. |
| `neutral_species_center_of_mass_velocity` | 0.48750 | 0.52500 | Quorate below bar; HOLD for governed content refinement. |
| `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | 0.76875 | 0.76875 | Quorate below bar; HOLD for governed content refinement. |
| `particle_torque_due_to_j_cross_b_force` | 0.90000 | 0.92500 | **Accepted on fresh quorum.** |
| `perpendicular_normalized_momentum_flux_due_to_perturbed_parallel_vector_potential` | 0.83750 | 0.65000 | Quorate below bar; HOLD for governed content refinement. |
| `poloidal_ion_momentum_flux` | 0.78750 | 0.83750 | Quorate below bar; HOLD for governed content refinement. |
| `poloidal_ion_velocity_at_measurement_position` | 0.81875 | 0.88750 | **Accepted on fresh quorum.** |
| `radial_coordinate_of_conductor_cross_section` | 0.83125 | 0.78750 | Quorate below bar; HOLD for governed content refinement. |
| `radius_of_coil_conductor_element` | 0.83750 | 0.81875 | Quorate below bar; HOLD for governed content refinement. |
| `ratio_of_particle_count_to_particle_simulated_count` | 0.72500 | 0.55000 | Quorate below bar; HOLD for governed content refinement. |
| `second_local_tangential_width_of_electron_cyclotron_launcher_mirror` | 0.83750 | 0.84375 | Quorate below bar; HOLD for governed content refinement. |
| `spectral_wavelength_of_optical_element` | 0.35625 | 0.58750 | Quorate below bar; HOLD for governed content refinement. |
| `square_of_magnetic_field_magnitude` | 0.83125 | 0.81250 | Quorate below bar; HOLD for governed content refinement. |
| `thermal_electron_torque_due_to_collisions` | 0.75625 | 0.82500 | Quorate below bar; HOLD for governed content refinement. |
| `toroidal_coordinate_of_electron_cyclotron_launcher_mirror` | 0.86875 | 0.85625 | **Accepted on fresh quorum.** |
| `toroidal_flux_limiter_coefficient` | 0.51250 | 0.51250 | Quorate below bar; HOLD for governed content refinement. |
| `toroidal_helium_3_velocity_at_plasma_boundary` | 0.91875 | 0.88750 | **Accepted on fresh quorum.** |
| `toroidal_neutral_state_momentum_convection_velocity` | 0.79375 | 0.73125 | Quorate below bar; HOLD for governed content refinement. |
| `volume_averaged_runaway_electron_current_density` | 0.49375 | 0.40000 | Quorate below bar; HOLD for governed content refinement. |

The draw was selective rather than merely optimistic: six of the eight accepted
rows moved upward across the bar, while two rows that were previously above the
bar remained above it on a resolved draw; the remaining 26 landed as low as
0.30000 and as close as 0.84375 without being promoted.

## Pending-document outcome

All 13 identities that entered the census at `docs_stage = pending` completed
generation and the ordinary review/refine loop and became docs-accepted:

- `bulk_center_of_mass_velocity`
- `explicit_ion_torque`
- `fast_particle_torque_density_due_to_coulomb_collisions_with_electrons`
- `ion_particle_convection_velocity`
- `kinetic_energy_density`
- `neutral_species_energy_flux`
- `neutron_source_rate_due_to_beam_beam_fusion`
- `parallel_neutral_momentum_diffusion_coefficient`
- `particle_torque_density_due_to_coulomb_collisions_with_electrons`
- `source_rate_due_to_beam_beam_fusion`
- `toroidal_vacuum_magnetic_field`
- `wave_voltage`
- `wavelength_of_visible_camera`

This clean 13-of-13 result is separate from the 8-of-34 fresh-draw recovery;
together they account exactly for the 21 identities removed from the backlog.

## Why 26 fresh winners cannot enter automatic refinement

A second exact-scope invocation attempted to continue the 26 fresh,
below-threshold outcomes through ordinary `REFINE_DOCS`:

```text
imas-codex sn run --docs-only \
  --scope-run-id f196fa3a-da6d-4013-981b-b7d193bb2d13 \
  --min-score 0.85 --cost-limit 49 --time 15 \
  --skip-global-maintenance
```

Its durable `SNRun`, `29061de0-1ca7-40d3-afc9-fe3cc63d016a`, completed with
`stop_reason = no_eligible_work`, zero events, and **USD 0** spent. This is a
mechanical refusal, not a budget stop. The shared docs winner predicate rejects
an identity when *any* attached historical docs-review group records a
non-winning method. All 34 rows necessarily retain such a historical group—the
terminal shortfall this node was dispatched to redraw—even though every latest
group is now a winner and the scalar shortfall marker is clear. Consequently,
the eight fresh scores above 0.85 can promote, while the 26 below 0.85 cannot
enter refinement.

That predicate repair requires source and test edits outside this node's
exclusive write fence. The named disposition of all 26 is therefore
**release HOLD pending a governed refinement-admission repair**, followed by
ordinary content refinement—not another redraw, a lower threshold, or a hand
accept. Repeated rescoring would mine quorum variance rather than improve the
documentation.

## Other retained holds

Seven starting identities were deliberately excluded from the fresh-draw
cohort. Six had already exhausted the documentation refinement cap and one was
a quorate below-bar row without a shortfall. They remain named release holds:

| `StandardName.id` | Score | Disposition |
|---|---:|---|
| `neutron_flux_due_to_fusion` | 0.52500 | Exhausted HOLD; steer a content correction and re-review. |
| `parallel_normalized_gyrocenter_momentum_flux_of_gyrokinetic_eigenmode_due_to_perturbed_parallel_vector_potential` | 0.45000 | Exhausted HOLD; steer a content correction and re-review. |
| `perturbed_gyrocenter_pressure` | 0.75000 | Exhausted HOLD; steer a content correction and re-review. |
| `poloidal_perturbed_magnetic_flux_at_measurement_position_due_to_wave_particle_interaction` | 0.61875 | Exhausted HOLD; steer a content correction and re-review. |
| `total_neutral_particle_flux_at_wall_due_to_surface_emission` | 0.50000 | Exhausted HOLD; steer a content correction and re-review. |
| `total_thermal_plasma_internal_energy` | 0.81875 | Exhausted HOLD; steer a content correction and re-review. |
| `wetted_area_of_divertor` | 0.60000 | Reviewed HOLD; steer a content correction and re-review. |

The final 33-row backlog is therefore completely dispositioned: **26 fresh
quorate below-bar rows awaiting the refinement-admission repair, 6 exhausted
content holds, and 1 reviewed content hold**. There are no pending or drafted
rows and no unexplained residual.

## Durable evidence

Machine-readable evidence and full logs are stored under the worker run:

- `before-census.json`: property coverage, exact 54-identity census, digest,
  stage split, the 34 shortfall identities, 13 pending identities, and 7
  pre-existing holds;
- `stage-receipt.json`: 34-of-34 dry-run/apply receipt and binding conservation;
- `campaign.log`: full paid run log, including the final zero-pending,
  zero-in-flight pool health and exact spend summary;
- `after-census.json`: exact 33-identity census, cohort reconciliation,
  per-identity fresh-draw outcomes, run accounting, and pool spend;
- `review-authority.json`: latest winning-group methods, proof that all 34
  retain historical non-winning groups, and both durable run records;
- `refine-continuation.log`: the zero-cost `no_eligible_work` continuation.

