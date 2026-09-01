# Strand-cause investigation

Measured read-only against the live `codex` graph and inspected at worktree
HEAD `812e4e35` on 2026-09-01T18:33:46+02:00. The target is the
`StandardName` population at `chain_length >= 3` with no inbound
`StandardNameSource-[:PRODUCED_NAME]` edge. Scalar mirrors were checked in
parallel so a missing edge could not masquerade as a genuinely unbound row.

## Verdict

**CAUSE-LIVE — `persist_refined_name` still commits a new successor when the
predecessor's authoritative source cohort is empty, by passing
`_allow_empty_noop=True` to `retarget_standard_name_sources`.** A plain fresh
`sn run` has no currently eligible ungrounded refine row (0 live rows at the
default 0.85 score bound and three-attempt cap), and its generate pool is
source-gated, so a default run over today's exact state will not add one.
Nevertheless the mechanism is live: the normal refine worker and the live
`sn edit --rename` operator both call `persist_refined_name`; the existing
regression test explicitly demonstrates that a source-less rename commits.
Any newly staged source-less predecessor can therefore mint another source-less
identity until the empty-cohort exemption is removed or converted to a refusal.

The present chain-cap population is **73**, down from the plan's earlier 78
because five bindings landed before this measurement. It is not a homogeneous
never-sourced population: **42/73 are receipt-proven source-lost**, **30/73 are
receipt-proven empty-cohort successor mints**, and **1/73 is a legacy
pre-ledger write with no transaction-era migration receipt**. Treating all 73
as never-grounded from current topology would erase the history carried by the
receipts.

## Positive control and aimed census

The schema was checked first: `StandardName.id`, `StandardName.chain_length`,
`StandardNameSource.id`, `StandardNameSource.produced_sn_id`, and
`StandardNameChange.{id,from_name,to_name,operation}` are declared keys. The
same live query then proved the instrument fires:

| Instrument | Result |
|---|---:|
| `StandardName` candidates / with `id` | 4,675 / 4,675 |
| `StandardName` with `chain_length` | 3,104 |
| `StandardNameSource` candidates / with `id` | 9,678 / 9,678 |
| sources with `produced_sn_id` | 5,189 |
| `StandardNameChange` candidates / with `id` / with `operation` | 8,707 / 8,707 / 8,707 |
| changes with `from_name` / `to_name` | 8,697 / 8,686 |
| chain-cap identities | 126 |
| chain-cap identities with a current source edge | 53 |
| chain-cap identities with no edge and no scalar binding | **73** |
| edge-missing but scalar-only identities | **0** |

The population query was:

```cypher
MATCH (sn:StandardName)
WHERE coalesce(sn.chain_length, 0) >= 3
WITH sn,
     COUNT { (:StandardNameSource)-[:PRODUCED_NAME]->(sn) } AS edge_sources,
     COUNT { (src:StandardNameSource)
             WHERE src.produced_sn_id = sn.id } AS scalar_sources
RETURN count(sn) AS chain_cap_total,
       sum(CASE WHEN edge_sources = 0 AND scalar_sources = 0
                THEN 1 ELSE 0 END) AS fully_ungrounded,
       sum(CASE WHEN edge_sources = 0 AND scalar_sources > 0
                THEN 1 ELSE 0 END) AS scalar_only;
```

This is explicitly aimed at the `StandardNameSource -> StandardName`
`PRODUCED_NAME` relationship and its declared scalar mirror, not the older
direct `IMASNode -> StandardName` projection.

## 1. `write_standard_names` is live, permissive in isolation, and fenced by its only production caller

`write_standard_names` is **not dead code**. There is exactly one production
call site outside tests:

```text
imas_codex/cli/sn.py::sn_run
  -> imas_codex/standard_names/loop.py::run_sn_pools
  -> imas_codex/standard_names/workers.py::process_generate_name_batch
  -> workers.compose_batch
  -> workers._persist_description_checked_candidates
  -> graph_ops.persist_generated_name_batch
  -> graph_ops.persist_generated_name_winners
  -> graph_ops.write_standard_names
```

The concrete call is `graph_ops.py:6852-6856`; the live CLI installs the
generate pool at `loop.py:627-642`, and `sn_run` enters `run_sn_pools` at
`cli/sn.py:852`.

The low-level writer **can persist a `StandardName` with no authoritative
source** when called directly. Its name write is an unconditional
`MERGE (sn:StandardName {id: b.id})` (`graph_ops.py:5199-5204`) and it does not
require a matched `StandardNameSource` or a `PRODUCED_NAME` edge. The existing
fixture at `test_pool_orchestration_contracts.py:704-717` supplies
`source_id=None`; `test_sweep_returns_count` calls the writer and asserts the
result is 1 (`:756-775`). That test passed in this run.

The reachable production call is materially safer than the bare function:

- `persist_generated_name_winners` drops entries without both `source_id` and
  `source_types` (`graph_ops.py:6770-6800`).
- It requires complete claim fences, locks the exact `StandardNameSource`,
  creates a provisional `PRODUCED_NAME` reservation, performs the rich name
  write through the same transaction, and finalizes only exact winners
  (`graph_ops.py:6802-6869`). A failure rolls back the transaction.
- `compose_batch` accepts as written only the exact source identities returned
  by that persistence boundary (`workers.py:6260-6285`).

Therefore the bare writer remains a footgun, but **the live generate entry
point cannot currently exercise it without a claimed source**.

## 2. Receipt join: 42 source-lost, 30 empty-cohort mints, 1 legacy pre-ledger row

The historical distinction was made from durable events, not current edge
absence. For each of the 73 identities I joined:

1. `StandardNameChange(operation='source_migration_manifest')` linked from the
   migration target, with matching `from_name`/`to_name`;
2. explicit detach changes:
   `detach_inconsistent_attachment`, `recover_terminal_source_binding`, and
   `retire_signed_dual_authority_target`;
3. `StandardNameSourceRetry.terminal_sn_id` receipts;
4. `repair_semantic_source_binding` receipts; and
5. the semantic `refine`/`human_edit` change paired with the migration receipt
   created by the same `persist_refined_name` transaction.

The exclusive result is:

| Provenance class | Identities | Evidence |
|---|---:|---|
| Source held, later lost | **42** | Union of non-empty migration-in/out receipts, explicit detach/terminal-retry receipts, and binding-repair receipts |
| Minted from empty authoritative cohort | **30** | A target-linked `refine`/`human_edit` change exists, but the same transaction has no `source_migration_manifest`; current code emits the latter only after a non-empty cohort survives exact postflight |
| Legacy pre-ledger, never authoritatively bound | **1** | `toroidal_diamagnetic_magnetic_flux_at_flux_surface`: generated 2026-06-19 by the old writer representation, no `REFINED_FROM`, only a later `supersede_exhausted_orphan` receipt, and no migration/detach receipt |
| Total | **73** | Residual **0** |

The source-lost evidence kinds overlap, so their raw receipt counts are not an
exclusive sum: **37 migration-in receipts**, **1 migration-out receipt**,
**9 explicit detach receipts** (including **7 terminal retry receipts**), and
**2 binding-repair receipts**. The 42-identity union is the authoritative
count.

Representative source-lost identities show distinct mechanisms:

- `parallel_neutral_state_convection_velocity` carries a
  `detach_inconsistent_attachment` receipt naming the exact DD path
  `plasma_transport/model/ggd/neutral/particles/v_parallel/values` and the
  state-resolution mismatch.
- `ratio_of_neutral_species_gas_count_to_total_gas_count` carries terminal
  source-recovery receipts; seven source releases are recorded against this
  one identity.
- `radial_plasma_momentum_source` carries a signed dual-authority retirement
  receipt.
- `poloidal_cross_sectional_area_of_plasma_boundary` carries a binding-repair
  receipt and a later source-migration manifest to
  `poloidal_plane_cross_sectional_area_of_plasma_boundary`; the successor
  currently has one source edge.
- `flux_surface_normal_neutral_energy_diffusion_coefficient` and
  `flux_surface_normal_momentum_convection_velocity` carry migration-in
  manifests but are currently unbound, proving that current topology alone
  misclassifies their history.

The 42 receipt-proven source-lost identities are:

```text
poloidal_cross_sectional_area_of_plasma_boundary
radial_plasma_momentum_source
parallel_neutral_state_convection_velocity
inverse_of_spectral_surface_curvature_of_optical_element
ratio_of_neutral_species_gas_count_to_total_gas_count
poloidal_parity_of_gyrokinetic_eigenmode
net_coefficient_due_to_neoclassical_tearing_mode
root_mean_square_of_fluctuating_floating_electrostatic_potential
particle_probability
toroidal_net_plasma_torque_of_neoclassical_tearing_mode
total_launched_wave_power_of_electron_cyclotron_launcher
poloidal_accumulated_magnetic_flux_due_to_resistive_dissipation
flux_surface_normal_neutral_energy_diffusion_coefficient
surface_thickness_of_cryostat
first_local_tangential_back_surface_radius_of_optical_element
ion_temperature_at_outboard_midplane_separatrix
normal_distance_of_antenna_strap
root_mean_square_of_spectral_width_of_spectrometer_channel
vertical_coordinate_of_plasma_filament
net_plasma_power_density
normal_width_of_plasma_filament
radial_offset_of_lower_hybrid_antenna
inverse_of_tangential_curvature_of_optical_element
plasma_electrostatic_potential_at_wall
plasma_electrostatic_potential_at_outboard_midplane
net_forward_power_of_wave_beam
total_launched_power_due_to_ion_cyclotron_heating
flux_surface_normal_momentum_convection_velocity
neutral_species_kinetic_energy_flux_at_wall_due_to_surface_emission
deposited_power_at_divertor_target
absorbed_coolant_power_of_plant_component_port
total_particle_flux_at_divertor_target_due_to_recycling
molecular_gas_count_due_to_pellet_injection
tendency_of_runaway_electron_density
non_axisymmetric_current_of_conductor
front_surface_area_of_langmuir_probe
energy_flux_at_control_surface
wave_critical_ordinary_mode_frequency
wave_magnetic_field_amplitude
total_incident_thermal_power
power_over_scrape_off_layer_due_to_radiation
lithium_volume_of_breeder_blanket
```

This corrects the prior report's use of “never-grounded”: that earlier query
meant “no source on the current node or any currently linked ancestor.” It did
not establish that the identity never held a source historically. The receipt
join shows that **57.5% (42/73)** did.

## 3. `retarget_standard_name_sources` cannot strand a moved cohort halfway

For a non-empty cohort, the retarget primitive is atomic and fail-closed. It
cannot detach a binding from the predecessor without the target taking it up:

```cypher
MATCH (old:StandardName {id: $old_name}),
      (new:StandardName {id: $new_name})
UNWIND $source_ids AS source_id
MATCH (source:StandardNameSource {id: source_id})
...
MATCH (source)-[prior:PRODUCED_NAME]->(old)
DELETE prior
MERGE (source)-[:PRODUCED_NAME]->(new)
SET source.produced_sn_id = new.id
...
WHERE size(moved_source_ids) = size($source_ids)
...
MATCH (moved:StandardNameSource {id: expected_source_id})
      -[:PRODUCED_NAME]->(new)
WHERE moved.produced_sn_id = new.id
  AND COUNT { (moved)-[:PRODUCED_NAME]->(:StandardName) } = 1
...
WHERE postflight_count = size($source_ids)
```

That is the deciding code at
`provenance_lifecycle.py:486-524`. It requires both names before deletion,
deletes and merges in one Cypher statement, verifies edge and scalar
postconditions for every exact source, and then writes the deterministic
`source_migration_manifest` receipt (`:525-560`). The Python boundary rejects
any moved-count mismatch (`:570-575`). When called with a `GraphClient`, the
complete operation is wrapped in one explicit transaction and any exception
rolls back (`:372-400`). `persist_refined_name` runs it inside the successor
creation transaction and also rolls back any failure.

The empty exception is different: with `_allow_empty_noop=True`, an empty
cohort skips the preflight and moves **nothing**. It does not detach a source;
it merely lets the already-created successor commit with no source. Thus the
retarget implementation is not the half-move strand mechanism. The caller's
permission to treat absence as success is.

## 4. Can a fresh `sn run` produce a new ungrounded identity today?

**From today's exact queue, a plain run: no. From the live callable surface:
yes.** The distinction is measurable:

- Current generated-name persistence is exact-source gated, as shown above.
- The live graph has **0** ungrounded identities satisfying the exact default
  `REFINE_NAME_ELIGIBILITY_WHERE`: `name_stage='reviewed'`, non-null score below
  0.85, attempts below 3, non-derived origin, no quorum shortfall, and no spent
  pinned-rename resubmission budget.
- The 73 chain-cap rows are terminal with respect to ordinary refinement:
  **42 superseded pipeline**, **29 exhausted pipeline**, **1 accepted pipeline
  at three attempts**, and **1 superseded catalog edit**. All have effective
  attempt count 3.
- Across all chain lengths there are five accepted ungrounded identities, but
  none is an open edit and none is currently in the refine queue.

However, both live callers of `persist_refined_name` remain capable of opening
the leak:

1. `workers.process_refine_name_batch` calls it at
   `workers.py:6989-7021`; any future reviewed, below-bound source-less row under
   the cap will produce a source-less successor.
2. `edit._apply_rename` calls it directly at `edit.py:2544-2568`; accepted is
   an eligible rename stage, and there is no non-empty source precondition.
   The regression `test_source_less_rename_still_records_human_edit`
   explicitly passes an empty cohort, asserts `source_ids == []`, and asserts
   the transaction committed (`test_atomic_refined_name_persistence.py:247-256`).

Inside `persist_refined_name`, the preflight collects zero or more source IDs
without requiring one (`graph_ops.py:16609-16614`), then calls retarget with
`_allow_empty_noop=True` and accepts `moved == len([]) == 0` before recording
the semantic change and committing (`graph_ops.py:16920-16950`). This is the
exact remaining mechanism.

## Verification

Command:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" \
uv run --no-sync pytest -p no:cacheprovider \
  tests/standard_names/test_pool_orchestration_contracts.py::TestSkeletonSweepNoUseAfterClose::test_sweep_returns_count \
  tests/standard_names/test_atomic_refined_name_persistence.py::test_source_less_rename_still_records_human_edit \
  tests/standard_names/test_provenance_lifecycle.py::test_retarget_query_repairs_exact_source_mirrors_and_both_caches \
  tests/standard_names/test_provenance_lifecycle.py::test_retarget_rejects_a_partially_admitted_explicit_cohort
```

Result: **4 passed, 0 failed, 1 environment-config warning, 5.95 s**.

No graph mutation, LLM call, source edit, code edit, or spend occurred.

**One-line verdict: CAUSE-LIVE — source retarget is atomic and today's default
queue has zero trigger rows, but `persist_refined_name` still treats an empty
authoritative source cohort as success (`_allow_empty_noop=True`), so the live
refine/edit paths can mint a new ungrounded successor.**
