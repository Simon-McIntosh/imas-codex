# The unobserved remainder, by instrument

Rows carrying a `validation_status` verdict with no observation
(`validated_at` null), and the disposition each subgroup needs. Measured at
worktree base `d085addaacd29226f52fac45cfb7f2fb40d8cf4a`, 2026-09-10 (UTC).
All graph reads were bounded indexed queries against the login-node-local
tunnel; the only graph writes are the sanctioned pipeline action recorded in
the subgroup-1 receipt.

## Confirmed counts at current HEAD

Population defined as live nonterminal identities (not `superseded` on either
axis, not `name_stage='exhausted'`) carrying a non-null `validation_status`
with a null `validated_at`. Known-present control: the live StandardName
universe is 2,920.

| subgroup | confirmed at HEAD | recorded earlier | drift |
|---|---|---|---|
| valid, no description | **8** | 8 | none |
| quarantined (legacy) | **38** | 38 | none |
| pending | **10** | 18 | 18 → 10 (8 pending rows left the population) |
| **total** | **56** | 64 | the recorded figures were from an earlier pass; pending is still moving |

The 256 `name_stage='exhausted'` quarantined rows are terminal and are not in
this population; excluding them at the population boundary is what reconciles
the recorded 38.

## Subgroup 1 — 8 valid rows with no description (instrument exercised)

These rows carry `validation_status='valid'`, `validated_at` null,
`name_stage='pending'`, `docs_stage='pending'`, and a null/empty description.
A valid verdict with no observation and nothing to audit: the deterministic
drain cannot claim a row with no description (`claim_ids_for_validation`
requires `sn.description IS NOT NULL`), so prose had to be materialised first.

| identity | dd source(s) | source state after the attempt |
|---|---|---|
| line_averaged_neon_density | summary/line_average/n_i/neon/value | failed (attempts 5) |
| poloidal_ion_state_momentum_diffusion_coefficient | plasma_transport/model/ggd/ion/state/momentum/d_pol/values | composed → produced **poloidal_ion_charge_state_momentum_diffusion_coefficient** |
| radial_coordinate_of_reflector | spectrometer_x_ray_crystal/channel/reflector/sphere_centre/r | failed (attempts 5) |
| radius_of_soft_xray_detector | soft_x_rays/channel/detector/radius | failed (attempts 5) |
| ratio_of_diamagnetic_vorticity_to_major_radius | edge_profiles/ggd/vorticity_over_r/diamagnetic, plasma_profiles/ggd/vorticity_over_r/diamagnetic | failed (attempts 5) each |
| toroidal_coordinate_of_spectrometer | spectrometer_visible/channel/geometry_matrix/interpolated/phi | failed (attempts 5) |
| toroidal_tritium_velocity | summary/local/pedestal/velocity_phi/tritium/value, summary/local/separatrix/velocity_phi/tritium/value | failed (attempts 5) each |
| vertical_coordinate_of_reflector | spectrometer_x_ray_crystal/channel/reflector/sphere_centre/z | failed (attempts 5) |

### Materialisation receipt — attempted, failed with a genuine pipeline guard

Sanctioned focused local name-generation compose over exactly the 10 sources:
`sn run --focus <paths> --names-only --only compose --skip-global-maintenance
--cost-limit 0.25` (local seat `deepseek-v4-flash`, probe healthy). Result:
`names_composed=1, spend=$0.00, stop_reason=failed` — the run **failed with 5
pool errors**, all the same refusal:

```
RuntimeError: source migration compare-and-set failed: dd:spectrometer_visible/
channel/geometry_matrix/interpolated/phi(exists=True, status='composed',
claimed=False, bindings=['toroidal_angle_of_spectrometer_channel'],
scalar='toroidal_angle_of_spectrometer_channel')
```

raised from `supersede_prior_source_names → retarget_standard_name_sources`
(`imas_codex/standard_names/provenance_lifecycle.py:554`). Log:
`~/.local/share/imas-codex/logs/sn_sn-compose.log` (run id `489b9c16…`).

**None of the 8 descriptions were materialised.** The receipts:

- The 8 identities are untouched: still `pending`, no description,
  `valid` unobserved.
- The only source that composed (`…/ion/state/momentum/d_pol/values`) re-linked
  to `poloidal_ion_charge_state_momentum_diffusion_coefficient` — an
  **already-accepted, valid, description-bearing identity**. The pending
  `poloidal_ion_state_momentum_diffusion_coefficient` (no `charge_state`) is a
  stale spelling of the same path.
- The remaining 9 sources ended `status='failed'`, `attempt_count=5` (the
  compose attempt cap): the candidates collided with the pending lifecycle or
  a live binding and the persist supersede was guard-refused. This pass
  consumed those sources' compose attempt budget — a side effect recorded
  under residue, recoverable by a governed attempt-budget reset.

**Verdict: the 8 are not a description-materialisation problem; they are a
stale-identity problem.** Where the instrument produced anything, the path's
real current identity is a different spelling that already exists and already
carries prose. Backfilling prose onto the pending stubs would assert a meaning
their paths no longer have. The correct instrument is per-path **identity
reconciliation** (is the pending spelling the current identity, or a stale
predecessor of another?) — which is a decision, listed under residue — and the
compose supersede CAS refusal is its own pipeline finding.

### Drain receipt — not run, deliberately

`drain_validation_for_ids` on the 8 was **not run**: the claim predicate
requires a non-null description and none of the 8 has one, so it would have
been vacuous. Running it and then having it observe nothing would assert an
observation that never happened — the exact un-backed-verdict class this
population is named for. The drain is the correct follow-on once each identity
is reconciled and prose exists.

## Subgroup 2 — 38 legacy quarantined (report, do not clear)

Real findings, not damage — reported with their reasons, not cleared (never
clear a quarantine to make a count move). Twenty-one carry a named per-row
grammar/audit reason in `validation_issues`; seventeen carry an empty issues
list with an all-clear `validation_layer_summary` and no persisted reason — a
quarantine whose evidence was not retained (the same un-backed-verdict class),
left untouched here.

### The 21 with a named per-row reason

| identity | stage | reason (`validation_issues`) |
|---|---|---|
| count_at_detector_pixel | accepted | description implies a rate/time-derivative but name lacks a `tendency_of_`/`change_in_`/`rate_of_change_of_` marker |
| cumulative_inside_flux_surface_torque | accepted | `cumulative_` prefix — DD `_inside`-style quantities use the `_inside_flux_surface` suffix form |
| fast_ion_photon_radiance_of_spectral_line_due_to_charge_exchange | drafted | canonical locus: field-evaluation structure uses `_of_`; rewrite `…_at_spectral_line_…` |
| field_aligned_surface_tilt_angle_of_langmuir_probe | refining | bare `_field` after start; qualify as `magnetic_field`/`electric_field` |
| flux_surface_average_magnetic_field_magnitude | pending | grammar round-trip failure |
| flux_surface_normal_ion_momentum_flux_due_to_diamagnetic_drift | drafted | token `flux` as coordinate prefix missing from coordinate-axes vocabulary; repeated `flux` token tautology |
| flux_surface_normal_non_axisymmetric_vacuum_magnetic_field_fourier_coefficient_at_control_surface | drafted | token `flux` coordinate prefix missing; repeated `surface` token tautology |
| phase_of_electron_cyclotron_beam | drafted | `phase_of_<X>` prefix — canonical noun-suffix form is `electron_cyclotron_beam_phase` |
| poloidal_magnetic_flux_perturbed_at_ece_channel_emission_position_due_to_wave_particle_interaction | drafted | coordinate-prefix token missing; unit `W` but description lacks heating/power/radiated/radiation |
| poloidal_momentum_neutral_internal_state_flux_limiter_coefficient | reviewed | grammar round-trip failure |
| power_of_beam_tracing_beam | accepted | repeated `beam` token tautology |
| rotation_frequency_due_to_e_cross_b_drift | accepted | symbol `$(R, \phi, Z)$` lacks a definition sentence |
| spectral_signal_to_noise_ratio_logarithm_of_spectrometer_channel | drafted | logarithm unary prefix applied to logarithmic unit `dB` |
| thermal_radiative_power_of_divertor_target | refining | canonical locus: `…_of_divertor_target` → `…_at_divertor_target` |
| toroidal_angle_of_active_limiter_point | accepted | name contains `angle` but unit is dimensionless (`1`) |
| toroidal_coordinate_of_launching_position | drafted | canonical locus → `_at_launching_position` |
| toroidal_coordinate_of_pellet_path | drafted | canonical locus → `_at_pellet_path` |
| toroidal_cumulative_inside_flux_surface_torque | drafted | `cumulative_` prefix vs suffix form |
| total_plasma_momentum_field_aligned_convection_velocity | drafted | bare `_field` after `momentum` |
| vertical_coordinate_of_outlet_due_to_gas_injection | refining | canonical locus → `_at_outlet_due_to_gas_injection` |
| volume_integrated_runaway_electron_density | accepted | `integrated_` prefix vs `_inside_flux_surface` suffix form |

### The 17 with no persisted per-row reason

diamagnetic_current_density, energy_density, ion_pressure,
length_of_antenna_strap, magnetic_field_magnitude, major_radius,
normalized_perturbed_pressure, particle_count,
perpendicular_normalized_perturbed_pressure, power_of_lower_hybrid_antenna,
radius_of_plasma_filament, radius_of_poloidal_field_coil,
source_rate_due_to_injection, time_derivative_of_electron_density,
toroidal_current_density, total_current_density, tungsten_density.

All are `accepted` with a present description and an all-clear
`validation_layer_summary`; their `validation_issues` is null or `[]` and
`validation_diagnostics_json` is `[]`. Reported, not cleared; the id-scoped
deterministic audit is the instrument that would re-derive each reason (see
residue).

## Subgroup 3 — 10 pending (left untouched)

A pending validation_status has no verdict to observe; these reach ordinary
validation on their own:

connection_length, ion_state_vibrational_level,
poloidal_straight_field_line_angle,
radial_coordinate_of_plasma_boundary_gap_reference_point,
radial_coordinate_of_reflectometer_antenna, radial_distance_at_midplane,
thickness_of_cryostat,
tritium_tritium_neutron_source_rate_due_to_thermal_fusion,
vertical_coordinate_of_dr_dz_zero_point, vertical_outline_of_vacuum_vessel.

## Residue for a later decision

1. **The 8 valid-no-description identities are stale spellings needing identity
   reconciliation**, not description materialisation: per path, decide whether
   the pending spelling is current (then generate prose through a non-wedging
   route) or a stale predecessor of another identity (then govern its
   retirement/supersede). None was deleted (deletion is forbidden; spend
   attribution protects undeletable identities) and no signed-manifest apply
   was attempted.
2. **Pipeline finding (follow-on, out of scope here):** the focused compose
   `supersede_prior_source_names → retarget_standard_name_sources` CAS refusal
   (`provenance_lifecycle.py:554`) — a compose candidate that would supersede a
   source bound to a different live identity fails the release path with a
   hard error rather than a guided message; the run correctly reported failure
   (exit 1) but the guard's interaction with `--names-only --only compose`
   lifecycle collisions needs its own node.
3. **Side effect of this pass:** 8 of the 10 sources are at
   `status='failed'`, `attempt_count=5` (compose attempt cap); a governed
   attempt-budget reset is needed before any future compose retry on them.
4. **The 17 reason-less legacy quarantines:** run the id-scoped deterministic
   audit on them as an authorised evidence-producing pass and record whether
   each re-quarantines with a reason or re-validates — not to clear a count.
5. **The 21 named-reason quarantines:** each names its own governed repair
   (locus rewrite, prefix-to-suffix form, unit dimension, symbol definition)
   via the sanctioned edit route.

## Evidence that could have failed

- The confirmed counts quote a direct measurement whose live-universe control
  (2,920) is shown, so an empty result would have been visible as vacuous.
- The 38-vs-256 quarantine split is what reconciles the recorded 38 with the
  raw 294: the exhausted rows are terminal and excluded at the population
  boundary, mirroring the plan's "live nonterminal" definition.
- Each of the 21 reasons is quoted from the row's own `validation_issues`, not
  from the plan's prose; the 17 reason-less rows are reported as reason-less,
  not given reasons they do not carry.
- The materialisation instrument was actually exercised and its failure quoted
  (5 pool errors, `names_composed=1`, `$0.00`), and the post-state was re-read
  rather than assumed: 0/8 descriptions, 8/10 sources at attempt cap, one
  source re-linked to an already-accepted different identity.
- The drain was deliberately not run because the claim predicate requires a
  non-null description; a no-op drain followed by a stamped observation would
  have manufactured exactly the unbacked verdict this population is about.

## Artifacts

- Measurement JSON: `/tmp/n-swcr-remainder-nonterminal.json` (population split 8/38/10, per-row data)
- 38-row dump: `/tmp/n-swcr-38-dump.txt` (per-row issues/diag/summary/stages)
- 8-row source paths: `/tmp/n-swcr-10-paths.txt`
- Compose run log: `~/.local/share/imas-codex/logs/sn_sn-compose.log` (run `489b9c16…`, failed, $0.00)
- Post-state query: `/tmp/n-swcr-poststate.py` / `/tmp/n-swcr-prod-char.py`
