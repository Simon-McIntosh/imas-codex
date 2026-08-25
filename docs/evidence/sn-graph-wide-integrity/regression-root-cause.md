# Semantic-source regression root-cause diagnosis

## Receipt corroboration result

The receipt-level probe used the inferred structural-source creation window, **2026-08-25T02:44:15.000Z inclusive through 02:44:16.000Z exclusive**, without joining individual targets. It found **0 `StandardNameChange` rows**, hence **0 `run_id` groups**, **0 run/manifest receipt keys**, and no materializer receipt that can be matched by its own `run_id` and manifest digest. This is a schema-proven zero: the graph contained 8,514 `StandardNameChange` candidates, 8,504 with `changed_at`, 3,334 with `run_id`, and 800 with `manifest_sha256` at **2026-08-25T07:55:29.849986Z**. The receipt mechanism exists elsewhere in the ledger, but did not record this batch.

The exact structural batch writer is therefore not recoverable from a durable receipt. The 36-row attribution remains an explicitly labelled timestamp-and-code-path inference; the accepted row census, transient verdicts, and repair boundary do not depend on choosing between the current structural batch writer variants. The measured follow-on is to stamp durable writer-operation receipts, including `run_id` and manifest digest, on structural `StandardNameSource` materialization.

## Quantitative result

The read-only live measurement at **2026-08-25T07:38:54.592Z** found the same requested cohorts: **39 composed sources with no live target** and **3 sources whose scalar mirror disagreed with their sole live target**. The diagnosis partitions the 42 rows as follows.

| Cohort | Derived | DD | Expected transient reconcile state | Genuine regression/residue |
|---|---:|---:|---:|---:|
| No live target | 36 | 3 | **36** | **3** |
| Scalar mirror | 1 | 2 | **0** | **3** |
| Total | 37 | 5 | **36** | **6** |

The 36 derived no-live rows are one structural-materialization cohort awaiting source-liveness settlement after their targets became terminal. The three DD no-live rows are legacy derived-parent provenance written under DD identities and have no live or terminal target to settle against. The three scalar rows are durable mirror defects: two are long-standing dual-binding dispositions whose former scalar-selected target is now terminal, and the derived row was retargeted by atomic refinement but again carries the predecessor scalar and edge.

No graph mutation was performed.

## Zero sanity and relationship direction

The census did not trust an empty result until the queried keys and relationship direction were proven present.

| Sanity probe | Candidates | With queried property |
|---|---:|---:|
| `StandardName.id` | 4,656 | 4,656 |
| `StandardName.name_stage` | 4,656 | 4,656 |
| `StandardNameSource.id` | 9,668 | 9,668 |
| `StandardNameSource.status` | 9,668 | 9,668 |
| `StandardNameSource.source_type` | 9,668 | 9,668 |
| `StandardNameSource.produced_sn_id` | 9,668 | 5,235 |

There were **5,351** authored `(StandardNameSource)-[:PRODUCED_NAME]->(StandardName)` edges; all 5,351 targets had both `id` and `name_stage`. The reverse-direction probe found **0** edges. Thus each row reported below as having zero live targets is a real zero over an existing property and the authored relationship direction, not a misspelled-property or reversed-edge zero.

The machine-readable capture is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T073510172408-n-regressiondiagnosis/logs/live-regression-detail.json`.

## Write-path attribution

The following path labels are used in the row tables.

- **Structural batch, then terminalization**: this is a timestamp-and-code-path inference to the structural batch family. `_materialize_derived_parent_rows_batched` (`STRUCTURAL_CLOSURE_BATCH_MATERIALIZE`) and `reconcile_orphan_parent_sources_batched` (`STRUCTURAL_CLOSURE_BATCH_SEED_SOURCES`) can both create or reuse `derived:<parent>`, stamp it `composed`, set `produced_sn_id=parent.id`, and merge the binding. Both write a `StandardNameChange` only when supplied an optional `event_by_parent`; neither synthesizes a recoverable run/manifest receipt for the batch. The 36 rows were created from `02:44:15.080Z` through `02:44:15.265Z`, all with `batch_key=derived_parent`, and all have live structural children. Their targets subsequently became `exhausted` or `superseded`. `reconcile_source_status_liveness` is the existing settlement path: it drops terminal edges, clears the scalar, and returns a composed/attached source with no live target to `extracted`. This is expected intermediate state, not a new attachment decision.
- **Legacy parent materializer using a DD identity**: the three DD rows were created on 2026-07-31 with `batch_key=derived_parent` but ids `dd:<container path>`, each with one exact `FROM_DD_PATH` backing and no `PRODUCED_NAME` edge. Current `_derived_parent_source_metadata` explicitly forbids that identity collapse and emits `derived:<parent>` without a DD realization edge. These are genuine historical residue, not structural transient state.
- **Dual-binding lifecycle transition**: the mass-density and radial-momentum sources were named standing dual-binding refusals before this measurement. Their scalar-selected target is now `exhausted` while the other edge targets an accepted name. The target lifecycle changed without atomically updating the source scalar, leaving a durable sole-live mismatch.
- **Atomic refine followed by structural regrowth**: `persist_refined_name` recorded `conductivity` to `plasma_electrical_conductivity` through `source_migration_manifest` at 2026-08-23T15:56:28.552Z. The current source once again has both predecessor and successor edges and selects the superseded predecessor. The only current structural recovery writer that can reuse `derived:conductivity`, preserve `status=composed`, reset the scalar to the parent, and merge the predecessor edge is `reconcile_orphan_parent_sources`. This is genuine regrowth, not an unfinished atomic refine.

## Every no-live-target row

All rows have a measured live-target count of zero.

| Source id | Status | Partition | Current terminal binding and live children | Producing write path | Verdict and route |
|---|---|---|---|---|---|
| `dd:ntms/time_slice/mode` | composed | DD | no binding; 0 children | Legacy parent materializer using a DD identity | **Genuine residue**; signed no-live lifecycle release, then separately adjudicated reattachment if any |
| `dd:summary/pedestal_fits` | composed | DD | no binding; 0 children | Legacy parent materializer using a DD identity | **Genuine residue**; signed no-live lifecycle release, then separately adjudicated reattachment if any |
| `dd:waves/coherent_wave` | composed | DD | no binding; 0 children | Legacy parent materializer using a DD identity | **Genuine residue**; signed no-live lifecycle release, then separately adjudicated reattachment if any |
| `derived:angle_of_antenna_strap` | composed | derived | `angle_of_antenna_strap` exhausted; 1 child | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:cumulative_ethylene_count_due_to_gas_injection` | composed | derived | `cumulative_ethylene_count_due_to_gas_injection` exhausted; 1 child | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:deposited_power` | composed | derived | `deposited_power` superseded; 1 child | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:deposited_power_at_divertor_target` | composed | derived | `deposited_power_at_divertor_target` exhausted; 2 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:diamagnetic_current_density` | composed | derived | `diamagnetic_current_density` exhausted; 3 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:electron_energy_diffusivity` | composed | derived | `electron_energy_diffusivity` superseded; 2 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:electron_particle_diffusivity` | composed | derived | `electron_particle_diffusivity` exhausted; 3 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:energy_diffusion_coefficient_due_to_diffusion` | composed | derived | `energy_diffusion_coefficient_due_to_diffusion` exhausted; 2 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:energy_flux_at_control_surface` | composed | derived | `energy_flux_at_control_surface` exhausted; 5 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:energy_flux_at_wall_due_to_eddy_current` | composed | derived | `energy_flux_at_wall_due_to_eddy_current` exhausted; 1 child | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:front_surface_area_of_langmuir_probe` | composed | derived | `front_surface_area_of_langmuir_probe` exhausted; 2 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:ion_atomic_number` | composed | derived | `ion_atomic_number` superseded; 1 child | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:launched_power_of_ion_cyclotron_heating_antenna` | composed | derived | `launched_power_of_ion_cyclotron_heating_antenna` exhausted; 2 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:major_radius` | composed | derived | `major_radius` exhausted; 9 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:momentum_convection_velocity` | composed | derived | `momentum_convection_velocity` superseded; 8 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:net_plasma_current_due_to_ohmic_current_drive` | composed | derived | `net_plasma_current_due_to_ohmic_current_drive` exhausted; 1 child | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:net_plasma_power_density` | composed | derived | `net_plasma_power_density` exhausted; 8 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:neutral_diffusivity` | composed | derived | `neutral_diffusivity` exhausted; 1 child | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:parallel_bulk_ion_velocity` | composed | derived | `parallel_bulk_ion_velocity` exhausted; 1 child | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:perturbed_linear_mhd_mode_number` | composed | derived | `perturbed_linear_mhd_mode_number` exhausted; 4 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:plasma_velocity_due_to_diamagnetic_drift` | composed | derived | `plasma_velocity_due_to_diamagnetic_drift` exhausted; 1 child | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:power_of_beam_tracing_beam` | composed | derived | `power_of_beam_tracing_beam` exhausted; 2 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:power_of_neutral_beam_injector` | composed | derived | `power_of_neutral_beam_injector` exhausted; 1 child | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:radius_of_iron_core_segment` | composed | derived | `radius_of_iron_core_segment` superseded; 2 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:radius_of_plasma_filament` | composed | derived | `radius_of_plasma_filament` exhausted; 2 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:radius_of_poloidal_field_coil` | composed | derived | `radius_of_poloidal_field_coil` exhausted; 2 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:root_mean_square_of_spectral_width_of_spectrometer_channel` | composed | derived | same-id target exhausted; 1 child | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:surface_temperature_of_plasma_facing_component` | composed | derived | same-id target exhausted; 1 child | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:surface_thickness_of_breeder_blanket_module` | composed | derived | same-id target exhausted; 2 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:total_ion_density` | composed | derived | `total_ion_density` superseded; 3 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:total_ion_momentum_diffusivity` | composed | derived | `total_ion_momentum_diffusivity` exhausted; 2 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:total_particle_flux` | composed | derived | `total_particle_flux` exhausted; 6 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:total_suprathermal_electron_power_density_due_to_collisions` | composed | derived | same-id target exhausted; 2 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:tungsten_density` | composed | derived | `tungsten_density` exhausted; 1 child | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:wave_magnetic_field_amplitude` | composed | derived | `wave_magnetic_field_amplitude` exhausted; 2 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |
| `derived:wavelength` | composed | derived | `wavelength` exhausted; 6 children | Structural batch, then terminalization | **Expected transient**; existing liveness settlement |

Count check: **3 DD genuine + 36 derived transient = 39**.

## Every scalar-mirror row

| Source id | Status | Partition | Scalar; sole live target; other binding | Producing write path | Verdict and route |
|---|---|---|---|---|---|
| `dd:plasma_profiles/ggd/mass_density/values` | composed | DD | `mass_density`; `total_plasma_mass_density`; `mass_density` exhausted | Dual-binding lifecycle transition | **Genuine regression**; reuse `repair_scalar_projection_mismatches` |
| `dd:plasma_sources/source/profiles_1d/ion/momentum/radial` | attached | DD | `radial_ion_momentum`; `radial_ion_momentum_source`; `radial_ion_momentum` exhausted | Dual-binding lifecycle transition | **Genuine regression**; reuse `repair_scalar_projection_mismatches` |
| `derived:conductivity` | composed | derived | `conductivity`; `plasma_electrical_conductivity`; `conductivity` superseded | Atomic refine followed by structural regrowth | **Genuine regression**; reuse `repair_scalar_projection_mismatches` and prevent terminal-parent regrowth |

Count check: **2 DD genuine + 1 derived genuine = 3**.

## Reconciliation against the repair reuse map

The reuse map is correct for the scalar cohort: all three rows have exactly one live target, so the narrow signed `repair_scalar_projection_mismatches` operator remains the right repair. Its fresh preview must still require `requested=3`, `admitted=3`, `refused=0`; no old digest is reusable.

The reuse map's single 39-row no-live mutation cohort is too broad after root-cause partitioning:

- The **36 derived rows** already have a deterministic existing route through `reconcile_source_status_liveness`; they are intermediate structural state and should be measured after that pass settles. Building a new signed release program solely to reproduce that existing settlement would turn transient pipeline state into production repair authority.
- The **3 DD rows** remain genuine no-live residue. They are the only members of this measurement that need the map's proposed signed lifecycle-release mechanics. Release does not authorize a replacement target; any later DD attachment remains a separate, freshly adjudicated action.
- The `derived:conductivity` scalar row additionally exposes the prevention defect: terminal/non-derived parents can be selected by `reconcile_orphan_parent_sources`, reusing a migrated structural source and regrowing the predecessor binding. Repairing the scalar once is safe but not sufficient unless the recovery selector stops recreating that state.

The corrected execution boundary is therefore **3 scalar repairs + 3 DD no-live releases**, with the 36 derived rows required to settle through ordinary liveness reconciliation before a postflight invariant census.

## Evidence and receipt observability

Evidence inputs used:

- Live census and zero sanity: `logs/live-regression-detail.json` in this node's durable run directory.
- Scalar target and change detail: `logs/scalar-lineage.json` in the same directory.
- Receipt-window result and its schema sanity: `logs/materializer-receipt-window.json` in the same directory.
- Baseline measurement: `docs/evidence/archive/integrity-and-operator-closure-verification.md`.
- Repair comparison: `docs/evidence/sn-graph-wide-integrity/regression-repair-reuse-map.md`.
- Earlier named dual-binding state: `docs/evidence/sn-graph-wide-integrity/ratchet-regrowth-triage.md`.

The exact-window receipt query completed successfully and established the negative finding: no change row exists to group by `run_id`, and therefore no receipt-owned run/manifest key exists to resolve. The 36-row structural-batch attribution is consequently inferred from the 185 ms creation burst, structural source shape, and current writer semantics. It is not represented as receipt-proven. This measured observability gap does not alter any row verdict or authorize a mutation.
