# ISN rc67 graph snapshot flush drain

Recorded on 2026-08-22 against the live `codex` graph from the detached
`flushrun` worktree. This was one unscoped, non-dry invocation:

```text
imas-codex sn run --flush --cost-limit 75 --time 25
```

No focus, name, domain, batch, drain-batch, scope-run-id, campaign, family,
edits-only, or maintenance-suppression scope was present. The persisted
`invocation_scope` is exactly
`{"attach_only":false,"edits_only":false,"reconcile_only":false,"skip_global_maintenance":false}`.
No second pipeline invocation was issued.

## Outcome

The graph grammar snapshot advanced successfully from the sole active
`ISNGrammarVersion` **0.8.0rc66** to the sole active version
**0.8.0rc67**. Both snapshots contain **22 GrammarSegment nodes and 956
GrammarToken nodes**. The installed package was already `0.8.0rc67` before the
run; the graph, not the dependency pin, was the stale side of the boundary.

The backlog drain made substantial progress but did **not** converge. The
persisted run `8691041b-dea1-47eb-a651-0c76e956edba` ended after
**1,268.292 seconds** with `status=degraded` and
`stop_reason=stalled`. It recorded **0 names composed, 87 enriched, 69
reviewed, and 20 regenerated**. The terminal refusal was a deterministic
`DDResolutionVersionMismatch` for
`equilibrium/time_slice/constraints/n_e/reconstructed` unit authority: the
row requested DD `4.1.0`, while the reviewed resolution exists only for DD
`4.1.1`. The review pool repeatedly reclaimed and refused that row after all
provider calls had completed. This is therefore a qualified drain outcome,
not a claim that the global backlog reached `no_eligible_work`.

The command exited 1 because the persisted run is stalled. It was not manually
signalled, retried, or followed by another flush. The grammar advancement and
all graph deltas below were independently re-read after process termination.

## Exact accounting

`LLMCost.llm_cost` is the per-call source of truth. The run's
`cost_is_exact=true`, its `events_total=273`, and its run-attributable cost
query exactly matches the global ledger delta.

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| `LLMCost` rows | 27,631 | 27,904 | **+273** |
| Total `LLMCost.llm_cost` | USD 1,366.843569 | USD 1,379.181972 | **USD 12.338403** |
| `StandardNameChange` rows | 7,841 | 7,865 | **+24** |
| `PRODUCED_NAME` relationships | 5,780 | 5,777 | **-3** |
| `StandardName` nodes | 4,393 | 4,395 | **+2** |

The USD 12.338403 invocation delta consumed **16.451204%** of the nominal USD
75 cap and left **USD 62.661597** unused. Against the separately authorized
USD 200 ceiling, it consumed **6.169202%** and left **USD 187.661597** unused.
Neither ceiling was reached, so no budget-driven stop was required. The stop
was the recorded pipeline stall above.

## Lifecycle transitions

The transition census compares the full union of Standard Name identities in
the before and after snapshots. `<absent>` therefore records a node creation or
removal rather than pretending that absence is a lifecycle stage.

### `name_stage`

| Transition | Names | Representative identities |
|---|---:|---|
| `<absent>` -> `accepted` | 3 | `line_averaged_plasma_velocity`, `toroidal_coordinate_of_reflectometer_antenna`, `toroidal_line_averaged_plasma_velocity` |
| `<absent>` -> `exhausted` | 1 | `ratio_of_charge_of_conductor_to_voltage_of_conductor` |
| `<absent>` -> `superseded` | 1 | `capacitance_of_conductor` |
| `accepted` -> `<absent>` | 3 | `impurity_ion_velocity`, `line_integrated_impurity_ion_velocity`, `straight_field_line_angle` |
| `drafted` -> `exhausted` | 2 | `ion_charge_state_torque_density`, `trapped_thermal_ion_charge_state_torque_density_due_to_collisions` |
| `drafted` -> `superseded` | 2 | `toroidal_coordinate_of_diagnostic_antenna`, `toroidal_line_integrated_impurity_ion_velocity` |
| `reviewed` -> `accepted` | 1 | `thermal_ion_charge_state_torque_due_to_collisions` |
| `reviewed` -> `exhausted` | 1 | `radial_ion_momentum` |
| `reviewed` -> `superseded` | 1 | `capacitance` |

The full union comparison therefore contains **15 `name_stage` identity
transitions**, including five creations and three removals.

### `docs_stage`

| Transition | Names | Representative identities |
|---|---:|---|
| `<absent>` -> `drafted` | 2 | `toroidal_coordinate_of_reflectometer_antenna`, `toroidal_line_averaged_plasma_velocity` |
| `<absent>` -> `pending` | 3 | `capacitance_of_conductor`, `line_averaged_plasma_velocity`, `ratio_of_charge_of_conductor_to_voltage_of_conductor` |
| `accepted` -> `<absent>` | 2 | `impurity_ion_velocity`, `line_integrated_impurity_ion_velocity` |
| `drafted` -> `accepted` | 2 | `minimum_magnetic_field`, `safety_factor` |
| `drafted` -> `reviewed` | 1 | `effective_electron_diffusivity` |
| `pending` -> `<absent>` | 1 | `straight_field_line_angle` |
| `pending` -> `accepted` | 35 | `accumulated_total_particle_count_due_to_gas_injection`, `gap_at_outboard_midplane`, `magnetic_field_at_pedestal_top_low_field_side_magnitude` |
| `pending` -> `drafted` | 47 | `accumulated_lithium_prefill_count_due_to_gas_injection`, `area_of_diagnostic_aperture`, `density_at_pedestal_maximum` |
| `pending` -> `reviewed` | 1 | `source_due_to_diamagnetic_drift` |
| `reviewed` -> `accepted` | 3 | `length_of_interferometer_beam`, `neutral_internal_state_momentum_convected_velocity`, `neutral_state_particle_convection_velocity` |
| `reviewed` -> `drafted` | 1 | `radial_momentum_flux` |

The full union comparison therefore contains **98 `docs_stage` identity
transitions**, including five creations and three removals.

## Runtime evidence

Durable evidence is under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T121040183395-flushrun/`:

- `before.json` and `before.log`: package version, active grammar, complete
  Standard Name lifecycle snapshot, existing run ids, and ledger baselines;
- `flush.log` and `flush.exit`: the only pipeline invocation, its grammar-sync
  message, complete runtime log, terminal run table, and exit code;
- `after.json` and `after.log`: independent postflight graph read, new SNRun,
  active grammar, lifecycle snapshot, and counters;
- `verification.json` and `verification.log`: machine comparison, exact
  transitions, budget percentages, and six passing done-when checks.

The machine checks all pass: exactly one new run; installed ISN rc67; rc66
before; rc67 after; the unscoped flush invocation with USD 75 cap; and the USD
200 authorized ceiling not reached. The operational qualification remains the
persisted `stalled` stop reason and the exact DD-version mismatch above.
