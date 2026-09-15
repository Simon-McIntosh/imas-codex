# Accepted names without producing sources

Measured read-only on 2026-09-15 against worktree revision
`1f246a61406f0557a77ee6aa84f203f86268f795` and the live `codex` graph. No
graph mutation was performed. The live count is **9 accepted `StandardName`
nodes with no incoming `PRODUCED_NAME` edge**, exactly the bound of 9 observed
at `2026-09-15T18:20Z`; the difference is **0**.

The zero-row controls are positive. The same query saw **5,130 StandardName
nodes, all 5,130 carrying the schema-owned `id`**, and partitioned the 2,528
accepted names into **2,519 with a producing source plus 9 without one**. The
instrument therefore saw both the key and the relationship it was meant to
measure.

## The nine identities

The cohort is not one writer's output. Five rows are historical edit-driven
renames, one is a historical automated refine, one lost its last attachment to
the attachment-consistency writer, one lost a stale synthetic producer through
the governed stale-source writer, and one is a structural parent whose canonical
derived source now produces a different live identity.

| Accepted identity | Description and name-review score | Surviving provenance | Trace | Export dry-run disposition |
| --- | --- | --- | --- | --- |
| `beta` | Global beta from total perpendicular plasma pressure and combined magnetic pressure; 0.49375 | no path scalar; `derived:beta` now produces `normalized_toroidal_beta`; 3 live children; 1 structural authority | structural acceptance can admit a source-free childful parent | candidate before the global emission gate |
| `coolant_absorbed_energy_accumulated_of_plasma_facing_component` | Cumulative coolant energy transferred from a plasma-facing component; 0.96875 | none | `human_edit` from `accumulated_coolant_absorbed_energy_of_plasma_facing_component` on 2026-09-05 | excluded: `grammar_parse_failure` |
| `equilibrium_weight_of_interferometer_beam` | Weight of an interferometer beam's line-density residual in an equilibrium objective; 1.0 | historical scalar `equilibrium/time_slice/constraints/n_e_line/weight`; no edge or `source_paths` entry | explicit `detach_inconsistent_attachment` receipt on 2026-09-01 | candidate before the global emission gate |
| `flux_surface_averaged_parallel_electric_field_at_separatrix` | Flux-surface average of parallel electric field at the separatrix; 1.0 | none | `human_edit` from `parallel_electric_field_flux_surface_averaged_at_separatrix` on 2026-09-07 | candidate before the global emission gate |
| `ion_pressure` | Isotropic ion pressure over charge states and thermal/fast populations; 0.79375 | `derived:ion_pressure` is stale and unbound; 2 live children; 1 structural authority | explicit `detach_stale_source_binding` receipt on 2026-08-20 | excluded: `invalid_validation_status` (`quarantined`) |
| `neutron_flux_due_to_fusion` | Volume-integrated neutron production rate from fusion; 0.8625 | none | automated `refine` from `total_neutron_flux_due_to_fusion_reactions` on 2026-08-11 | excluded: `release_hold_documentation_not_accepted` |
| `parallel_current_density_per_toroidal_mode_due_to_wave_driven_current_drive` | Field-aligned current-density contribution from one toroidal Fourier harmonic; 0.98125 | none | `human_edit` from `parallel_per_toroidal_mode_current_density_due_to_wave_driven_current_drive` on 2026-09-05 | excluded: `never_reviewed` on the docs axis |
| `root_mean_square_spectral_width_of_spectrometer_channel` | RMS instrumental wavelength spread of a spectrometer channel; 0.975 | none | `human_edit` from `spectral_width_root_mean_square_of_spectrometer_channel` on 2026-09-07 | candidate before the global emission gate |
| `toroidal_wave_vector_normalized_of_beam_tracing_beam` | Toroidal direction cosine of a beam-tracing wave vector; 0.96875 | none | `human_edit` from `toroidal_normalized_wave_vector_of_beam_tracing_beam` on 2026-09-05 | excluded: `never_reviewed` on the docs axis |

The live plan's spelling
`parallel_current_density_per_toroidal_mode_due_to_wave_driven_current` is not a
graph identity. The measured identity ends in `_current_drive`, as shown in the
table.

## The writer path, and why source-status liveness is not it

The historical rename/refine writer is
`imas_codex/standard_names/graph_ops.py:18095`,
`persist_refined_name`. Its current empty-cohort refusal is at lines
18483-18494, with the exact predicate:

```python
if (
    authoritative_cohort_observed
    and not candidate_source_ids
    and not edit_mode
):
    raise RefinedNamePersistenceRefusal(...)
```

That guard was added on 2026-09-01. The automated `neutron_flux_due_to_fusion`
refine predates it. The predicate deliberately exempts edit-driven renames, so
the writer's source migration still permits an empty no-op for `edit_mode` at
`graph_ops.py:18716-18733`. The caller had no independent producer-count
postcondition until 2026-09-08. Five cohort rows carry `human_edit` receipts
dated 2026-09-05 or 2026-09-07, before that postcondition landed.

The current caller is guarded. `imas_codex/standard_names/edit.py:3298-3307`
invokes `_carry_rename_sources`, and `edit.py:1931-1938` now refuses the exact
bad outcome:

```python
missing = [source_id for source_id in expected_source_ids if source_id not in held]
if missing:
    raise RuntimeError(
        f"rename left {len(missing)} of {len(expected_source_ids)} producing "
        ...
    )
```

So the five edit rows are historical residue, not a currently open writer bug.
The other current path is intentional and narrower:
`structural_accept_derived_parents` at
`graph_ops.py:26243`. Its source-free route is selected by the exact predicate
at lines 26233-26238:

```python
if live_child_count < 1:
    return None
if origin == "derived":
    return "derived"
if origin is None and producer_count == 0:
    return "source_free"
```

It writes `name_stage = "accepted"` at lines 26321-26342 before the startup
sequence calls `reconcile_orphan_parent_sources`. That repair refuses to steal a
canonical derived source already bound to another live target and refuses to
revive a stale structural source. Those two constraints explain `beta` and
`ion_pressure` rather than making either an accidental liveness deletion.

`reconcile_source_status_liveness` cannot have removed an edge to an accepted
target. `_TERMINAL_BINDING_NAME_STAGES` at `graph_ops.py:62-67` is exactly
`superseded`, `exhausted`, and `contested`; `accepted` is absent. Its cleanup
query at lines 12531-12538 selects a source only when no target exists outside
that terminal set, and supplies it at `graph_ops.py:12608` before deleting the
selected edges. An accepted target is therefore a live target for this writer
and keeps the source outside the deletion predicate.

Two remaining rows have their own positive provenance receipts rather than an
inference: `equilibrium_weight_of_interferometer_beam` records
`detach_inconsistent_attachment`, and `ion_pressure` records
`detach_stale_source_binding`. They are reported as distinct mechanisms, not
folded into the rename/refine attribution.

## Export dry run

The executed command was:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
PYTHONPATH="$PWD" uv run --no-sync imas-codex sn release \
  --export-only --dry-run \
  --isnc /home/ITER/mcintos/Code/imas-standard-names-catalog
```

The durable `.export_report.json` receipt records these counts; the first five
are the CLI's `Export Summary` fields:

```text
total candidates: 2497
exported: 0
excluded (below score): 1
excluded (unreviewed): 31
excluded (domain): 0
accounted exclusions: 180
accounting residue: 2317
```

This dry run is a **failed cut, not a successful export**. The independent
`identity_token_collision` gate rejected `halo_current` because that published
identity equals a process token. The command therefore selected its explicit
exit-1 path before setting the emitted identity list. Its receipt correctly
keeps `exported = 0` and the 2,317 remaining candidates as accounting residue.

Within this nine-name cohort, five identities have explicit exclusion records
shown in the table. Four — `beta`,
`equilibrium_weight_of_interferometer_beam`,
`flux_surface_averaged_parallel_electric_field_at_separatrix`, and
`root_mean_square_spectral_width_of_spectrometer_channel` — passed the
cohort-level eligibility and score filters and reached the candidate set before
the global emission gate. **None was emitted**, because the global gate blocked
the whole cut. The evidence therefore establishes that missing producer
provenance is not itself an export exclusion, while preserving the stronger
negative fact that this invocation cut zero identities.

That behavior is visible in the current exporter. The population predicate at
`imas_codex/standard_names/export.py:607` is exactly
`sn.name_stage IN ['accepted', 'approved']`; the eligibility chain at lines
798-836 checks domain, validation, name stage, quorum, and documentation but no
producer edge. Source topology is consulted only to derive reviewer-facing
roles at `export.py:1588-1600`. Thus an accepted, otherwise eligible source-free
identity can reach the candidate set.

## Evidence files

- Live graph census, including the positive controls and all nine rows:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260915T190723000709-n-usb-accepted-names-without-a-producing-source-are-traced/live-graph-census.log`
- Relationship and change-ledger traces:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260915T190723000709-n-usb-accepted-names-without-a-producing-source-are-traced/orphan-relationship-trace.log`
  and
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260915T190723000709-n-usb-accepted-names-without-a-producing-source-are-traced/orphan-change-trace.log`
- Descriptions, domains, scores, and path scalars:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260915T190723000709-n-usb-accepted-names-without-a-producing-source-are-traced/orphan-cohort-descriptions.log`
- Preserved export receipt and compact extraction:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260915T190723000709-n-usb-accepted-names-without-a-producing-source-are-traced/export-report.json`
  and
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260915T190723000709-n-usb-accepted-names-without-a-producing-source-are-traced/export-dry-run-summary.log`

The separate `halo_current` export-gate failure is outside this node's write
scope and is carried as a follow-on, not repaired here. The accepted source-free
cohort likewise remains unchanged pending the lead's provenance disposition.
