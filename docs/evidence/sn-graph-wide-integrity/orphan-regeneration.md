# Orphan regeneration run receipt

## Outcome

**BLOCKED.** The 42-path scoped regeneration started under the single active
`0.8.0rc66` grammar and stayed within both spend limits, but it could not drain.
The name-review pool repeatedly failed closed because
`equilibrium/time_slice/constraints/n_e/reconstructed` is a DD `4.1.0` source
while its active `unit` resolution was reviewed only for DD `4.1.1`. The run was
interrupted after the same deterministic authority mismatch was observed across
review replicas; no retry, rescore, resolution change, or manual lifecycle edit
was attempted.

The requested integrity measure is therefore not met:

- Genuine unsourced names with no live structural child: **36 before, 36
  after**. The lifecycle distribution was byte-for-byte identical at the count
  level: 27 accepted, 4 reviewed, 4 drafted, and 1 pending.
- Previously genuine-orphan names gaining a producing source: **0 of 36**.
- Accepted parents reset by the scoped operation: **23**. Final stages sum to
  **18 accepted + 3 drafted + 2 reviewed = 23**.
- Fresh-quorum acceptance: **1 of the 18** returned accepted parents
  (`safety_factor`, final score 0.9625). The other **17 of 18** were accepted by
  the ordinary derived-parent `structural-inheritance` path with a null
  `reviewer_score_name` and no fresh name-review group. Consequently this
  receipt cannot confirm that every acceptance was earned by a fresh quorum;
  that condition **failed** even though no direct acceptance was performed.
- Parents not returned to accepted: **5**. They remain staged exactly where the
  interrupted ordinary pipeline left them and were not rescored.

## Invocation and immutable inputs

The focus file contains 42 distinct non-empty DD paths (the final record has no
newline, so a raw line-count utility reports 41). Its SHA-256 is
`22a704b855955aa69086c8ccffa68e1142efd27bc9272e70768c59c72966acfc`.
The active grammar census immediately before execution was one active grammar,
version `0.8.0rc66`.

The initial literal `--reset-to extracted` invocation was a documented gap-only
no-op because all 42 paths already had live accepted targets. It returned zero
provider work and no lifecycle reset, explicitly requiring `--reseed`. The
ordinary scoped run that performed the reset was:

```bash
mapfile -t orphan_focus < <(awk 'NF' docs/evidence/sn-graph-wide-integrity/orphan-regeneration-focus-paths.txt)
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
  PYTHONPATH="$PWD" \
  uv run --no-sync imas-codex sn run \
    --reset-to extracted --reseed --include-accepted --cost-limit 25 \
    "${orphan_focus[@]}"
```

The run id was `f1168b3b-f3d9-47b3-809d-99b5dbca7106`; its focus scope id was
`25c083b5-d385-4a0e-bfd0-b0da1e1d1e67`. The run ended with
`stop_reason=interrupted` after 276.125485 seconds, reporting 0 names composed,
17 enriched, 13 reviewed, and 0 regenerated. The durable CLI log is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T085507035497-sgwi-orphan-regeneration/sn-run.log`
(SHA-256 `e6690226454c1d24cdbe6da4cb201db3ae4d72ed08337b7244ca8dcc75c13b7c`).

## Orphan census

The census definition was applied identically before and after: a live
`StandardName` whose `name_stage` is neither `superseded` nor `exhausted`, with
no incoming `PRODUCED_NAME` relationship from a `StandardNameSource`, and no
live incoming structural child over `HAS_PARENT`.

| name_stage | Before | After | Delta |
|---|---:|---:|---:|
| accepted | 27 | 27 | 0 |
| reviewed | 4 | 4 | 0 |
| drafted | 4 | 4 | 0 |
| pending | 1 | 1 | 0 |
| **Total** | **36** | **36** | **0** |

All 36 pre-run identities were re-read after the run. **Zero** had acquired an
incoming producing source. The 42 focus sources instead remained attached to
the reset parent identities, so the family-topology repair did not retarget a
producer to any genuine orphan.

## Parent lifecycle movement

Every row below was accepted immediately before reset and therefore moved out
of acceptance during the reset. A dash is a null score. “Structural” means the
ordinary derived-parent acceptance path, not a quorum review.

| Parent identity | Before stage | Before score | After stage | After score | Return evidence |
|---|---|---:|---|---:|---|
| `capacitance` | accepted | — | reviewed | 0.575 | fresh 3-seat review, below threshold |
| `co_passing_thermal_electron_torque_density_due_to_collisions` | accepted | — | accepted | — | structural; no fresh quorum |
| `current_density_due_to_ohmic_current_drive` | accepted | — | accepted | — | structural; no fresh quorum |
| `effective_electron_diffusivity` | accepted | — | accepted | — | structural; no fresh quorum |
| `effective_ion_diffusivity` | accepted | — | accepted | — | structural; no fresh quorum |
| `effective_neutral_diffusivity` | accepted | — | accepted | — | structural; no fresh quorum |
| `effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | accepted | — | accepted | — | structural; no fresh quorum |
| `electron_density` | accepted | — | drafted | — | no completed fresh review |
| `ion_charge_state_power_at_inside_flux_surface` | accepted | — | accepted | — | structural; no fresh quorum |
| `ion_charge_state_torque_density` | accepted | 0.95 | drafted | — | no completed fresh review |
| `length_of_interferometer_beam` | accepted | — | accepted | — | structural; no fresh quorum |
| `magnetic_field_at_pedestal_top_low_field_side` | accepted | — | accepted | — | structural; no fresh quorum |
| `minimum_magnetic_field` | accepted | — | accepted | — | structural; no fresh quorum |
| `neutral_internal_state_convection_velocity` | accepted | — | accepted | — | structural; no fresh quorum |
| `neutral_internal_state_momentum_convected_velocity` | accepted | — | accepted | — | structural; no fresh quorum |
| `neutral_state_particle_convection_velocity` | accepted | — | accepted | — | structural; no fresh quorum |
| `safety_factor` | accepted | — | accepted | 0.9625 | fresh 2-seat quorum group |
| `straight_field_line_angle` | accepted | — | accepted | — | structural; no fresh quorum |
| `thermal_ion_charge_state_energy_diffusion_coefficient` | accepted | — | accepted | — | structural; no fresh quorum |
| `thermal_ion_charge_state_torque_due_to_collisions` | accepted | 0.91875 | reviewed | 0.6875 | fresh 3-seat review, below threshold |
| `toroidal_flux_coordinate_gradient_magnitude` | accepted | — | accepted | — | structural; no fresh quorum |
| `trapped_fast_ion_charge_state_torque_density_due_to_collisions` | accepted | — | accepted | — | structural; no fresh quorum |
| `trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | accepted | 1.0 | drafted | — | no completed fresh review |

The five parents that did not return to accepted are therefore
`capacitance`, `electron_density`, `ion_charge_state_torque_density`,
`thermal_ion_charge_state_torque_due_to_collisions`, and
`trapped_thermal_ion_charge_state_torque_density_due_to_collisions`. No
`sn rescore` or equivalent recovery was invoked for any of them.

## Spend and graph attribution

Attributable cost was measured from the 53 `LLMCost` nodes whose `for_run`
equals the run id. The per-pool rows sum exactly to **$1.602678**.

| Pool | Model | Calls | USD |
|---|---|---:|---:|
| generate_docs | `openrouter/openai/gpt-5.6-luna` | 17 | 0.094946 |
| review | `openrouter/anthropic/claude-sonnet-5` | 13 | 0.647652 |
| review | `openrouter/x-ai/grok-4.5` | 11 | 0.332164 |
| review_name | `openrouter/anthropic/claude-sonnet-5` | 2 | 0.186667 |
| review_name | `openrouter/openai/gpt-5.6-luna` | 5 | 0.028459 |
| review_name | `openrouter/x-ai/grok-4.5` | 5 | 0.312790 |
| **Total** |  | **53** | **1.602678** |

- Node spend: **$1.602678 / $25.00** (6.410712% used; $23.397322
  unspent).
- Running session spend: prior measured **$1.869714** + this node
  **$1.602678** = **$3.472392 / $150.00** (2.314928% used; $146.527608
  remains authorized).
- Live `LLMCost` census: 27,538 nodes / $1,363.819553 before; 27,591 nodes /
  $1,365.422231 after. Deltas are exactly 53 nodes / $1.602678.
- Live `StandardNameChange` census: 7,687 before; 7,703 after; delta 16,
  attributable to the sanctioned CLI lifecycle path rather than direct graph
  editing.
- Post-stop claim census: 0 claimed sources and 0 claimed names. The interrupted
  focus scope remains durably tagged on 42 sources and 24 names, so recovery can
  identify the exact cohort without reconstructing it.

## Safety and blocker

All bespoke graph inspection in this run was read-only (`MATCH`, `OPTIONAL
MATCH`, `UNWIND`, aggregation, and `RETURN`). **No Cypher `SET`, no direct graph
text edit, no hand acceptance, and no `sn edit` were used.** The only mutations
were performed by the ordinary scoped `sn run` state machine.

The run nevertheless disproves the required acceptance invariant: 17 derived
parents reached accepted through the pipeline's sanctioned structural path,
not through fresh quorum scores. The blocked DD source also remains version
inconsistent with its reviewed resolution: source DD version `4.1.0`, published
unit `1`; resolution DD version `4.1.1`, effective unit `m^-3`. Advancing this
work requires an authority decision outside this node: either supply a reviewed
resolution applicable to `4.1.0`, move the focus source to the authoritative DD
snapshot, or revise the requested reset/review contract to account explicitly
for derived-parent structural acceptance. Re-running unchanged would only
repeat the fail-closed error and spend provider budget.
