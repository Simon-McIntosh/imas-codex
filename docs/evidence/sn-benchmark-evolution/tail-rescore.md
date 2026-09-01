NEEDS-HELP: The bounded rescore wave did not converge: the ordinary runner spent USD 4.982846, then repeatedly deferred one quorum seat without terminating, leaving nonterminal names and one unintended refined successor.

tried: Preflight proved that all 32 frozen identities existed exactly once and were `name_stage=exhausted`. A serial `sn rescore` invocation was stopped after its per-name global-maintenance cost made the 32-name wave exceed the 30-minute fence. The same `stage_name_for_rescore` plus scoped `run_sn_pools` backend was then used once for the cohort. Deterministic validation refused 3 names; the remaining scoped run produced 12 acceptances, 10 exhausted outcomes with fresh scores, 1 reviewed outcome, 3 still refining, 2 still drafted, and 1 superseded identity before the USD 5 cohort limit was reached. The runner then repeated the same budget-refused quorum deferral until terminated at the fence.

options: (1) Fix the scoped runner so an unaffordable mandatory reviewer seat terminates with `budget_exhausted`, then restore or explicitly adjudicate the nonterminal rows and rescore only the unchanged identities. (2) Increase the scoped budget and rerun after first repairing the rescore path so it cannot enter `REFINE_NAME` or create a successor. (3) Treat this wave as a negative pipeline result, repair the validation/quorum/refine defects first, and rebuild a fresh candidate census before another paid draw.

leaning: Option 3. The operation falsified two premises required by the node: 3 frozen candidates now fail deterministic validation, and the current `sn rescore` backend can reword a low-scoring identity through `REFINE_NAME`. More spend cannot make those semantics safe.

cost-if-wrong: Choosing option 1 or 2 without repairing the rescore semantics risks another roughly USD 5 partial run, more nonterminal claims, and additional successor identities that must be audited and recovered individually. Choosing option 3 delays the remaining draw but preserves the acceptance boundary and prevents further mutation.

# Tail rescore outcome

## Quantitative verdict

- Frozen input census: **32/32** unique identities from `docs/evidence/sn-benchmark-evolution/exhausted-tail-triage.md`; preflight found **32/32 exhausted**, **0 missing**, and **0 duplicate graph matches**.
- Fresh quorum results observed before termination: **12/32 accepted**.
- Returned to the parked tail with a fresh score: **10/32 exhausted**.
- Other outcomes: **1 reviewed**, **3 refining**, **2 drafted**, **3 deterministically re-quarantined and restored to exhausted without a new score**, and **1 superseded after the rescore path created a successor**.
- Exact attributable spend: **USD 4.982846**, the sum of **115 `LLMCost` rows** for paid `SNRun` `dc5d226a-c493-4fe0-a103-365dd67141e4` (scope `sn-rescore-20260901T135204Z`). Two interrupted audit rows, `a7283321-18b6-4969-aa08-1c90f874ab8e` and `50da5f87-70c8-4934-a284-9fc6cdf13289`, contain **0 `LLMCost` rows / USD 0.000000**.
- Identity-retention gate: **FAIL**. All 32 original graph nodes still have their exact requested `id` strings, but `particle_convection_velocity` was changed to `superseded` and gained successor `particle_velocity_due_to_convection`. Therefore the claim that no name was reworded is false for this run.
- Pipeline-signal verdict: **YES — defect, not an unlucky draw**. Ten unchanged identities exhausted again with fresh scores; three frozen candidates now fail deterministic admission; one blind-seat disagreement stopped at `reviewed`; the budget refusal did not terminate; and one rescore entered refinement and created a successor. These jointly implicate admission currency, prompt/quorum composition, threshold handling, and the rescore/refine boundary.

## Per-identity result

The “retained identity” column compares the post-run `StandardName.id` to the exact frozen input string. `yes*` means the original node still carries the string but the operation also created a reworded successor, so the no-reword gate fails.

| Rank | Frozen identity | Post-run outcome | New score | Retained identity | Evidence note |
|---:|---|---|---:|:---:|---|
| 1 | `fast_particle_pressure` | accepted | 0.9625 | yes | Fresh quorum accepted unchanged identity. |
| 2 | `ion_diamagnetic_momentum_convection_velocity` | drafted | — | yes | Second serial invocation was interrupted before its paid quorum run. |
| 3 | `magnetic_field_magnitude` | exhausted | 0.75625 | yes | Refine proposal failed strict grammar validation; unchanged identity remained parked. |
| 4 | `toroidal_momentum_convection_velocity` | refining | 0.8000 | yes | Nonterminal when the budget-deferral loop was stopped. |
| 5 | `toroidal_neutral_state_momentum_source` | accepted | 1.0000 | yes | Fresh quorum accepted unchanged identity. |
| 6 | `trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | accepted | 0.9875 | yes | Fresh quorum accepted unchanged identity. |
| 7 | `electron_temperature_at_separatrix` | accepted | 1.0000 | yes | Fresh quorum accepted unchanged identity. |
| 8 | `launched_power_of_ion_cyclotron_heating_antenna` | accepted | 0.9750 | yes | Fresh quorum accepted unchanged identity. |
| 9 | `deuterium_tritium_flux` | refining | 0.5375 | yes | Successor persistence repeatedly failed because its producing DD source is stale; nonterminal at stop. |
| 10 | `particle_convection_velocity` | superseded | 0.7125 | yes* | **Reworded:** successor `particle_velocity_due_to_convection` was created. |
| 11 | `normalized_toroidal_flux_coordinate_at_ece_channel_emission_position` | accepted | 1.0000 | yes | Fresh quorum accepted unchanged identity. |
| 12 | `wave_curvature_of_wave_beam` | refining | 0.8250 | yes | Nonterminal at stop; audit flags duplicated `wave` content. |
| 13 | `diamagnetic_current_density` | exhausted | 0.5750 | yes | Fresh draw returned to tail. |
| 14 | `radial_coordinate_of_launching_position` | accepted | 0.98125 | yes | Fresh quorum accepted unchanged identity. |
| 15 | `ion_charge_state_torque_density` | accepted | 0.98125 | yes | Fresh quorum accepted unchanged identity. |
| 16 | `parallel_momentum_flux` | exhausted | 0.7375 | yes | Fresh draw returned to tail; a proposed occupied successor was refused. |
| 17 | `radial_coordinate_of_flux_surface` | accepted | 0.9500 | yes | Fresh quorum accepted unchanged identity. |
| 18 | `toroidal_current_density_due_to_distribution_function_driven` | accepted | 0.9000 | yes | Fresh quorum accepted unchanged identity. |
| 19 | `total_ion_energy_diffusion_coefficient` | reviewed | 0.8750 | yes | Acceptance refused because blind seats disagreed and escalation did not resolve them. |
| 20 | `flux_surface_normal_neutral_energy_diffusion_coefficient` | exhausted / quarantined | — | yes | Deterministic validation rejected coordinate prefix token `flux`; restored to parked state without paid review. |
| 21 | `gap_at_plasma_boundary` | exhausted | 0.8250 | yes | Fresh draw returned to tail; unit-description audit issue retained. |
| 22 | `toroidal_ion_momentum_diffusion_coefficient` | exhausted | 0.7625 | yes | Fresh draw returned to tail. |
| 23 | `ion_species_particle_flux_at_wall_due_to_surface_emission` | accepted | 0.9875 | yes | Fresh quorum accepted unchanged identity. |
| 24 | `flux_surface_normal_momentum_convection_velocity` | exhausted / quarantined | — | yes | Deterministic validation rejected coordinate prefix token `flux`; restored to parked state without paid review. |
| 25 | `ion_state_average_charge_number` | exhausted | 0.8375 | yes | Fresh draw returned to tail; proposed occupied successor was refused. |
| 26 | `gradient_of_normalized_pressure_at_flux_surface` | exhausted | 0.7000 | yes | Fresh draw returned to tail. |
| 27 | `plasma_power_at_wall` | exhausted | 0.73125 | yes | Fresh draw returned to tail; proposed `total_incident_power_at_wall` collided and was refused. |
| 28 | `wave_curvature_of_beam_tracing_beam` | exhausted | 0.66875 | yes | Fresh draw returned to tail; proposed `wave_curvature_of_wave_beam` collided and was refused. |
| 29 | `fast_ion_charge_state_torque_due_to_collisions` | drafted | — | yes | Mandatory reviewer seat exceeded the remaining budget; runner repeatedly deferred instead of terminating. |
| 30 | `toroidal_angle_of_secondary_x_point` | exhausted / quarantined | — | yes | Deterministic validation found `angle` with unit `m`; restored to parked state without paid review. |
| 31 | `first_local_tangential_back_surface_radius_of_optical_element` | accepted | 0.9625 | yes | Fresh quorum accepted unchanged identity. |
| 32 | `total_launched_power_due_to_ion_cyclotron_heating` | exhausted | 0.7000 | yes | Fresh draw returned to tail. |

## Unmet completion conditions

The requested evidence fence is not met. The run did not return a terminal accepted-or-tail-with-new-score outcome for all 32: 3 were deterministically refused before review, 6 remain nonterminal or were superseded, and one identity was reworded. The graph also retains started `SNRun` audit rows because the bounded worker had to terminate the nonconverging budget-deferral loop. These states require a repair node with graph-mutation authority and a corrected rescore contract; they must not be papered over by hand acceptance or direct text edits.
