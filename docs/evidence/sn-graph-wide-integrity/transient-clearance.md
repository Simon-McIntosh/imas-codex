# Derived no-live-target settlement

Date: 2026-08-25

## Outcome

**CLEARED — 36 of 36 derived rows cleared and 0 persisted.** The cohort was a
real transient liveness state. It is not genuine residue and must not be routed
to the signed legacy-DD release program.

The production liveness settlement returned all 36 exact
`derived:<parent>` sources from `composed` to `extracted`, reset their attempt
budget, cleared their scalar and terminal binding, and left no row in the
composed/attached no-live-target invariant. An immediate replay of the same
exact invocation returned zero for every mutation counter, proving the result
was settled rather than sampled mid-run.

| Measure | Before | After settlement and replay |
|---|---:|---:|
| Composed/attached no-live-target rows | **36** | **0** |
| Derived partition | **36** | **0** |
| DD partition | **0** | **0** |
| Other partition | **0** | **0** |
| Exact cohort at `status=extracted` | 0 | **36** |
| Rows cleared | — | **36** |
| Rows persisted | — | **0** |
| Authored `PRODUCED_NAME` edges | 5,351 | **5,315** |
| `StandardName` nodes | 4,656 | **4,656** |
| `StandardNameSource` nodes | 9,668 | **9,668** |

The exact edge delta is **−36**, one terminal binding for each settled source.
No Standard Name or source node was deleted.

## Exact production invocation

The applying process used the canonical liveness reconciler directly, bounded
to the enumerated 36-source cohort:

```python
reconcile_source_status_liveness(
    gc=production_graph,
    source_ids=[<the 36 exact derived source ids listed below>],
)
```

This was deliberately narrower than `imas-codex sn run --only reconcile`.
Maintenance-only CLI mode also runs graph-wide grammar, structural, unit,
attachment and parent maintenance, while this node was authorized to settle
one measured class and prove collateral immutability. The direct call is the
same `reconcile_source_status_liveness` route that `sn run --only reconcile`
invokes at startup, but its public `source_ids` bound makes every write
predicate exact.

The same call was then made a second time against the same 36 IDs. The settled
replay result was:

```text
live_realigned=0
orphaned_reset=0
terminal_edges_dropped=0
terminal_projections_dropped=0
terminal_source_paths_dropped=0
projection_ghosts_reset=0
ghost_projections_dropped=0
ghost_source_paths_dropped=0
```

The production apply ran in the command window ending at
2026-08-25T08:54:08Z. The independent read-only recovery and replay ran from
2026-08-25T08:57:13.547Z through 2026-08-25T08:57:15.339Z.

## Before census and schema sanity

The immediately preceding production receipt at
2026-08-25T08:41:58.632Z froze the complete 36-row cohort at digest:

```text
de427254697917b0b16f9e56d70c754ac846efd6d9d5a0fce2c368f8374acd5c
```

The applying harness re-read production immediately before mutation and
refused unless all of these conditions held:

- the invariant partition was exactly `36 = 36 derived + 0 DD + 0 other`;
- its ordered derived IDs exactly matched the 36-row baseline below;
- every source resolved exactly once, was `composed` or `attached`, and had
  zero live target;
- every `StandardName.id`, `StandardName.name_stage`,
  `StandardNameSource.id`, `StandardNameSource.status`, and
  `StandardNameSource.source_type` sanity equality held;
- every authored `PRODUCED_NAME` edge pointed from a source to a Standard Name
  carrying both schema keys, with zero reverse-direction edges; and
- the three released DD rows were each still `extracted`, unclaimed, at attempt
  zero, without a target, and backed by exactly one DD node.

The harness reached its post-settlement collateral assertion, so every one of
those before gates passed. The baseline machine record supplies the numeric
schema counts: **4,656/4,656** names with `id` and `name_stage`,
**9,668/9,668** sources with `id`, `status`, and `source_type`, and
**5,351/5,351** authored edges whose targets carry both keys, with **0** reverse
edges. Thus the 36 is not a missing-property or reversed-edge zero.

## After census and real-zero proof

The recovered postflight reports:

| Schema sanity probe | Candidates | With queried property |
|---|---:|---:|
| `StandardName.id` | 4,656 | 4,656 |
| `StandardName.name_stage` | 4,656 | 4,656 |
| `StandardNameSource.id` | 9,668 | 9,668 |
| `StandardNameSource.status` | 9,668 | 9,668 |
| `StandardNameSource.source_type` | 9,668 | 9,668 |
| Authored source → name edges / targets with both keys | 5,315 | 5,315 |
| Reversed name → source edges | 0 | 0 |

The invariant implementation returned an empty list both before and after the
settled replay. Each of the 36 exact sources independently resolved once with:

- `status=extracted`;
- `attempt_count=0`;
- null `claimed_at`, `claim_token`, `produced_sn_id`, and `composed_at`;
- zero `PRODUCED_NAME` bindings; and
- absence from the composed/attached no-live-target invariant.

This is a lifecycle settlement, not a disappearance hidden by a query change:
all 9,668 source nodes and all 4,656 name nodes remain present.

## Per-row verdict

| Source | Verdict |
|---|---|
| `derived:angle_of_antenna_strap` | cleared |
| `derived:cumulative_ethylene_count_due_to_gas_injection` | cleared |
| `derived:deposited_power` | cleared |
| `derived:deposited_power_at_divertor_target` | cleared |
| `derived:diamagnetic_current_density` | cleared |
| `derived:electron_energy_diffusivity` | cleared |
| `derived:electron_particle_diffusivity` | cleared |
| `derived:energy_diffusion_coefficient_due_to_diffusion` | cleared |
| `derived:energy_flux_at_control_surface` | cleared |
| `derived:energy_flux_at_wall_due_to_eddy_current` | cleared |
| `derived:front_surface_area_of_langmuir_probe` | cleared |
| `derived:ion_atomic_number` | cleared |
| `derived:launched_power_of_ion_cyclotron_heating_antenna` | cleared |
| `derived:major_radius` | cleared |
| `derived:momentum_convection_velocity` | cleared |
| `derived:net_plasma_current_due_to_ohmic_current_drive` | cleared |
| `derived:net_plasma_power_density` | cleared |
| `derived:neutral_diffusivity` | cleared |
| `derived:parallel_bulk_ion_velocity` | cleared |
| `derived:perturbed_linear_mhd_mode_number` | cleared |
| `derived:plasma_velocity_due_to_diamagnetic_drift` | cleared |
| `derived:power_of_beam_tracing_beam` | cleared |
| `derived:power_of_neutral_beam_injector` | cleared |
| `derived:radius_of_iron_core_segment` | cleared |
| `derived:radius_of_plasma_filament` | cleared |
| `derived:radius_of_poloidal_field_coil` | cleared |
| `derived:root_mean_square_of_spectral_width_of_spectrometer_channel` | cleared |
| `derived:surface_temperature_of_plasma_facing_component` | cleared |
| `derived:surface_thickness_of_breeder_blanket_module` | cleared |
| `derived:total_ion_density` | cleared |
| `derived:total_ion_momentum_diffusivity` | cleared |
| `derived:total_particle_flux` | cleared |
| `derived:total_suprathermal_electron_power_density_due_to_collisions` | cleared |
| `derived:tungsten_density` | cleared |
| `derived:wave_magnetic_field_amplitude` | cleared |
| `derived:wavelength` | cleared |

Count check: **36 cleared + 0 persisted = 36**. Because none persisted, no row
is reclassified as genuine residue.

## Unchanged collateral and concurrency qualification

The three already-released DD rows were snapshotted immediately before and
after the settled replay with the identical normalized digest:

```text
3cd2010251d6ed3512d3fa47668b22dcb9d733a0a87d3396fae65190209ba3ea
```

All **9,632 non-cohort `StandardNameSource` rows** — full source properties,
all authored target identities/stages/relationship properties, and exact DD
backings — likewise retained one identical digest across the settled replay:

```text
7efc59839b6c40b4d167cb600a9c9e4997e0a229eba3ef1c827f22da79c67896
```

The first harness also computed a broader digest including every name's
`docs_stage`. That digest changed and was correctly rejected, because the
concurrently authorized WEST family node was running a scoped `--docs-only
--skip-global-maintenance` production drain during this exact window. It could
change documentation-axis state but could not invoke global source-liveness
maintenance. That over-broad digest is discarded rather than presented as
collateral evidence. The source-scoped digest above measures the class this
node owns and excludes neither another source row nor any source relationship.

This qualification matters: the evidence proves no other semantic-source
class moved during the settled replay, while it does not claim that an
independent docs-axis writer was idle.

## Durable records

- Baseline before census, row IDs, schema sanity and cohort digest:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T083346749641-n-ddresidueapply/logs/apply-replay-postflight.json`
- Applying harness and its exact source list:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T084727718944-n-transientclearance/logs/run_transient_clearance.py`
- Applying command failure record showing the over-broad collateral assertion
  was the first failed gate:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T084727718944-n-transientclearance/logs/transient-clearance.stderr`
- Independent postflight, all 36 row states, schema sanity, exact replay, DD
  digest and non-cohort source digest:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T084727718944-n-transientclearance/logs/transient-clearance-recovery.json`
- Recovery command diagnostics and exit marker:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T084727718944-n-transientclearance/logs/transient-clearance-recovery.stderr`

No provider call, direct acceptance, raw Cypher mutation, plan-state edit, or
DD-source reattachment was performed.
