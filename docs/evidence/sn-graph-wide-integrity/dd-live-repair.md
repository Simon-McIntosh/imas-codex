# DD no-live-target production repair

## Result

The five production DD semantic sources diagnosed as genuine liveness residue
were repaired through the existing exact-scoped
`reconcile_source_status_liveness` program on **2026-08-25 at 16:39:24Z**.
The applying process independently re-derived the current cohort immediately
before mutation. Its preview admitted **5 of 5 requested rows**, refused **0**,
and retained an empty verbatim refusal list. The apply changed **5** source
rows. The immediate exact replay reported **changed=0** and
**persistent_writes=0**.

After the apply and again after the settled replay, the complete semantic-source
no-live-target class was **0**: partition **DD 0 / derived 0 / other 0**. All
five rows cleared and none persisted. The one dual-bound source and all four
accepted unsourced holds retained the identical protected-state SHA-256 before
and after:

```text
27f1bd679d0aa02dd82926a83de367342c9ff0c8bda1e4a302d4b18dfd94c666
```

No `StandardNameChange` or `LLMCost` row was written. The operation removed
exactly five authored terminal `PRODUCED_NAME` edges, five DD projections and
five target `source_paths` entries. It did not accept, reattach, rename or
retire any Standard Name.

## Preview and authority derivation

The production driver was:

```text
/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T163434923393-n-ddliverepair/logs/run_dd_live_repair.py
```

Within that single applying process, it:

1. called `find_semantic_source_invariant_violations` against production;
2. selected the rows with zero live targets;
3. required that the current global cohort equal the closed five-ID DD
   allowlist, with every source resolving exactly once, carrying
   `source_type=dd`, having `status` `composed` or `attached`, having exactly
   one DD backing, and having no live target;
4. emitted a per-row preview and stopped before mutation on any refusal;
5. re-ran the current no-live-target query at the mutation boundary; and
6. passed those freshly returned IDs, rather than an earlier census result, to:

```python
reconcile_source_status_liveness(
    gc=production_graph,
    source_ids=boundary_no_live_ids,
)
```

The preview and the boundary read were identical:

| Preview measure | Count |
|---|---:|
| Requested | 5 |
| Admitted | 5 |
| Refused | 0 |
| Verbatim refusal list | `[]` |
| Boundary no-live-target rows | 5 |
| Boundary partition | DD 5 / derived 0 / other 0 |

This is an exact-ID invocation of the existing liveness settlement, not a raw
Cypher repair or a broad graph sweep. Had the cohort gained, lost or changed a
row, every requested row would have carried the verbatim refusal reason
`global no-live-target cohort differs from the closed five-id allowlist`, and
the driver would have exited before applying.

## Apply and settled replay

The apply returned:

| Program counter | Apply | Replay |
|---|---:|---:|
| `live_realigned` | 0 | 0 |
| `orphaned_reset` | 5 | 0 |
| `terminal_edges_dropped` | 5 | 0 |
| `terminal_projections_dropped` | 5 | 0 |
| `terminal_source_paths_dropped` | 5 | 0 |
| `projection_ghosts_reset` | 0 | 0 |
| `ghost_projections_dropped` | 0 | 0 |
| `ghost_source_paths_dropped` | 0 | 0 |
| Source rows changed | **5** | **0** |
| Sum of persistent-write counters | **20** | **0** |

The apply completed at **16:39:24.605Z**. The exact replay started only after
that synchronous transaction had committed and completed at
**16:39:24.713Z**. Its eight mutation counters were all zero; therefore replay
was both source-idempotent (`changed=0`) and write-free
(`persistent_writes=0`).

## Per-row before and after

Every retained target was terminal (`name_stage=exhausted`) before settlement.
Every source independently resolved after replay as `status=extracted`,
`attempt_count=0`, null `claimed_at`, `claim_token`, `produced_sn_id`, and
`composed_at`, with exactly one DD backing and zero `PRODUCED_NAME` bindings.

| Exact source ID | Before status and retained terminal target | After settled replay | Verdict |
|---|---|---|---|
| `dd:gyrokinetics_local/linear/wavevector/eigenmode/poloidal_turns` | `composed`; `poloidal_turn_count` | `extracted`; zero bindings | **cleared** |
| `dd:iron_core/segment/geometry/arcs_of_circle/r` | `composed`; `radial_coordinate_of_arc_of_circle_center` | `extracted`; zero bindings | **cleared** |
| `dd:pellets/time_slice/pellet/path_profiles/position/r` | `composed`; `radial_coordinate_of_pellet_path` | `extracted`; zero bindings | **cleared** |
| `dd:summary/line_average/dn_e_dt/value` | `attached`; `time_derivative_of_electron_density` | `extracted`; zero bindings | **cleared** |
| `dd:summary/local/pedestal/q/value` | `composed`; `safety_factor_at_pedestal` | `extracted`; zero bindings | **cleared** |

Count check: **5 cleared + 0 persisted = 5**. These rows were genuine DD
lifecycle residue, not the already-settled derived transient class. Clearing
them returns their DD paths to ordinary composition; it does not endorse their
former exhausted identities.

## Real-zero proof and schema sanity

The invariant moved **5 → 0** on apply and remained **0** after replay. The zero
is backed by complete schema-key coverage and an authored-direction positive
control:

| Schema sanity probe | Before | After replay |
|---|---:|---:|
| `StandardName.id` | 4,658 / 4,658 | 4,658 / 4,658 |
| `StandardName.name_stage` | 4,658 / 4,658 | 4,658 / 4,658 |
| `StandardNameSource.id` | 9,668 / 9,668 | 9,668 / 9,668 |
| `StandardNameSource.status` | 9,668 / 9,668 | 9,668 / 9,668 |
| `StandardNameSource.source_type` | 9,668 / 9,668 | 9,668 / 9,668 |
| Authored source → name edges with complete endpoint keys | 5,315 / 5,315 | 5,310 / 5,310 |
| Reversed name → source edges | 0 | 0 |

The named positive control is the **5,315 authored
`StandardNameSource -> StandardName` edges before settlement**, all with source
ID, target ID and target lifecycle populated. It proves the instrument used the
schema-authored direction and could see the edges whose absence it reports.
The aimed partition control is full `StandardNameSource.source_type` coverage
of **9,668/9,668**, proving that DD/derived/other zeros are not missing-property
zeros. Node counts remained exactly **4,658 Standard Names** and **9,668
sources**; the class cleared by lifecycle transition, not by hiding or deleting
its source nodes.

## Protected stability rows and ledger immutability

The protected digest covered only lifecycle and topology fields owned by this
repair, deliberately excluding documentation and embedding axes that concurrent
authorized nodes may update.

Before and after, the dual-bound source
`dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial` remained
`attached`, scalar-selected to
`radial_neutral_internal_state_momentum_flux`, and bound to exactly these two
accepted targets:

- `radial_neutral_internal_state_momentum_flux` (`origin=pipeline`); and
- `radial_neutral_state_momentum_flux` (`origin=catalog_edit`).

Before and after, each accepted-unsourced hold resolved exactly once as
`name_stage=accepted`, `origin=pipeline`, `validation_status=valid`, with zero
producing sources:

- `fast_ion_charge_state_power_at_inside_flux_surface`;
- `neutron_flux_due_to_fusion`;
- `tendency_of_total_thermal_plasma_internal_energy`; and
- `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions`.

The combined protected digest remained
`27f1bd679d0aa02dd82926a83de367342c9ff0c8bda1e4a302d4b18dfd94c666`.
Ledger counters likewise remained **8,596 StandardNameChange** and **34,914
LLMCost** rows. Authored `PRODUCED_NAME` edges changed exactly
**5,315 → 5,310**, matching only the five enumerated terminal bindings.

## Durable machine evidence

- Applying driver:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T163434923393-n-ddliverepair/logs/run_dd_live_repair.py`
  — SHA-256 `743842bff56517cb5b6a3109a52a9b1a19023f557e51747feae25f228b001637`.
- Complete preview/apply/replay record:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T163434923393-n-ddliverepair/logs/dd-live-repair.json`
  — SHA-256 `2ea97aefe20a8419555c14426869789c0fd542cb51ead1f6efcb232ff1f33445`.
- Diagnostic stream:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T163434923393-n-ddliverepair/logs/dd-live-repair.stderr`
  — SHA-256 `c91ed578d52763adf99dd773a651217940cadeddf926079523fd6cbd86d88d86`.

The command exited successfully. No provider call, direct acceptance, raw
Cypher mutation, graph-wide reconcile, plan-state edit, or workaround for a
refusal occurred.
