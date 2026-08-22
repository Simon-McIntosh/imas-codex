# Flux-surface parent release — signed production apply

## Outcome

**Applied once, then replayed with no write.** The signed structural-release
program removed the sole `HAS_PARENT` edge
`surface_area_of_flux_surface` → `area_of_flux_surface` without creating a
replacement parent. The released child remains independently live and retains
its complete producing-source authority.

| Measure | Result |
|---|---:|
| Authority rows | 1 |
| Preview admitted / refused | 1 / 0 |
| Apply outcome | `applied` |
| Apply changed rows | 1 |
| Apply mutations / receipt rows | 1 / 1 |
| Apply persistent writes | 2 |
| Replay outcome | `already_applied` |
| Replay changed rows / persistent writes | 0 / 0 |

The two apply writes are the one exact edge deletion and its one immutable
`StandardNameChange` receipt. No other graph mutation is included in this
authority.

## Signed authority and manifest

The canonical repair-authority builder emitted an exact one-row authority from
the live relationship element, its complete properties, the child and parent
participants, and the closed `recompute_projection` mutation with
`new_end_id=null`.

| Digest | SHA-256 |
|---|---|
| Authority file | `f897885c73090379ef35dda443d634468d8c1c3ac91d8bba3cef9a4c36a1f6d5` |
| Signed authority payload | `c06d6097c7f12ca6708917fe52213934035022da83a5dfdd1299e839704ad1ec` |
| Preview/apply/replay manifest | `8ceb6bfb025ba1d729f8cdbe82422e224cadf71c7c4b1ea3643eed6bf2711d1e` |

Preview returned `outcome=would_apply`, `would_change=1`, and
`counts={authority_rows: 1, admitted: 1, refused: 0}`. Apply re-derived and
locked that same closure, returned `outcome=applied`, `changed=1`,
`mutations=1`, `receipt_rows=1`, and `persistent_writes=2`. Immediate replay
of the same `manifest_sha256` returned `outcome=already_applied`, `changed=0`,
`receipt_rows=1`, and `persistent_writes=0`.

The immutable receipt is:

- id:
  `sn-change:signed-manifest:8ceb6bfb025ba1d729f8cdbe82422e224cadf71c7c4b1ea3643eed6bf2711d1e:dd371c80c4786cf1a2a8c86b`;
- operation: `release_structural_standard_name_children`;
- row: `surface_area_of_flux_surface`;
- mutation kind: `recompute_projection`;
- run: `r-20260822T205336176820-n-release`;
- receipt manifest: `8ceb6bfb025ba1d729f8cdbe82422e224cadf71c7c4b1ea3643eed6bf2711d1e`.

## Persistent post-apply re-read

The apply committed before a fresh graph read. The replay then performed a
second postcondition read before returning `already_applied`.

| Closure measure | Before | After apply and replay |
|---|---:|---:|
| Child `HAS_PARENT` edges | 1 (`area_of_flux_surface`) | **0** |
| `area_of_flux_surface` live children | 1 (`surface_area_of_flux_surface`) | **0** |
| Child producing sources | 20 | **20** |
| Child `name_stage` | `accepted` | **`accepted`** |
| Child `docs_stage` | `accepted` | **`accepted`** |
| Child vocabulary `status` | `draft` | **`draft`** |
| Child `validation_status` | `valid` | **`valid`** |
| Child `origin` | `catalog_edit` | **`catalog_edit`** |
| Full child + producer authority SHA-256 | `ad9382c70336b1b409a7f3354ea6f035025e28dba8ff110f7e3117237b8e1758` | **same** |

The full authority digest covers every child property plus every producer-node
property and `PRODUCED_NAME` relationship property. Its exact equality proves
that lifecycle, validation, documentation state, source mirrors, producer
nodes, and producer bindings survived unchanged; only the signed parent edge
was removed.

All 20 retained source-path bindings are:

| Producing source | Source status |
|---|---|
| `dd:core_profiles/profiles_1d/grid/surface` | `attached` |
| `dd:core_sources/source/profiles_1d/grid/surface` | `attached` |
| `dd:core_transport/model/profiles_1d/grid_d/surface` | `attached` |
| `dd:core_transport/model/profiles_1d/grid_flux/surface` | `attached` |
| `dd:core_transport/model/profiles_1d/grid_v/surface` | `attached` |
| `dd:disruption/profiles_1d/grid/surface` | `attached` |
| `dd:distribution_sources/source/profiles_1d/grid/surface` | `attached` |
| `dd:distributions/distribution/profiles_1d/grid/surface` | `attached` |
| `dd:edge_profiles/profiles_1d/grid/surface` | `composed` |
| `dd:equilibrium/time_slice/global_quantities/surface` | `attached` |
| `dd:equilibrium/time_slice/profiles_1d/surface` | `composed` |
| `dd:plasma_profiles/profiles_1d/grid/surface` | `attached` |
| `dd:plasma_sources/source/profiles_1d/grid/surface` | `composed` |
| `dd:plasma_transport/model/profiles_1d/grid_d/surface` | `attached` |
| `dd:plasma_transport/model/profiles_1d/grid_flux/surface` | `attached` |
| `dd:plasma_transport/model/profiles_1d/grid_v/surface` | `attached` |
| `dd:runaway_electrons/profiles_1d/grid/surface` | `attached` |
| `dd:sawteeth/profiles_1d/grid/surface` | `composed` |
| `dd:transport_solver_numerics/solver_1d/grid/surface` | `attached` |
| `dd:waves/coherent_wave/profiles_1d/grid/surface` | `attached` |

The post-read was against graph `codex` at the recorded production Bolt endpoint
`bolt://98dci4-clu-3062:7687`.

## Reproducible artifacts

The complete authority, receipts, state reads, and command logs are retained
under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T205336176820-n-release/`:

- `parent-release-authority.json` and
  `parent-release-authority-digests.json` — emitted signed authority bytes and
  independent file/payload digests;
- `parent-release-preflight.json` — exact live child, parent, edge, and producer
  closure before apply;
- `parent-release-preview.json` and `parent-release-preview.log` — signed
  preview plus `EXIT_MARKER=0`;
- `parent-release-applied.json` and `parent-release-apply.log` — committed apply
  plus `EXIT_MARKER=0`;
- `parent-release-replay.json` — exact-hash write-free replay;
- `parent-release-post-apply.json` — receipt and persistent post-apply closure.
- `parent-release-evidence-check.log` — machine assertion of every quantitative
  condition above, with `EXIT_MARKER=0`.

The operational check used the repository's shared environment with
`PYTHONPATH` set to this exact worktree and `uv run --no-sync`; both the preview
and the apply-plus-replay invocations exited 0.
