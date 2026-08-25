# Recurring graph-producer diagnosis

## Outcome

The governed deletion did not hold because the ordinary Standard Names loop has
a deterministic writer for the same relationship. The complete recurring path
is:

1. `derive_edges()` parses
   `spectral_signal_to_noise_ratio_of_spectrometer_channel`, treats the leading
   `spectral` token as a qualifier, and emits
   `spectral_signal_to_noise_ratio_of_spectrometer_channel
   -[:HAS_PARENT {operator: 'spectral', operator_kind: 'qualifier'}]->
   logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel`
   (`imas_codex/standard_names/derivation.py:297-316`).
2. `_write_standard_name_edges()` reconciles each live child's structural
   closure, then `MERGE`s that exact edge and property map
   (`imas_codex/standard_names/graph_ops.py:3034-3079`).
3. `rederive_structural_edges()` drives that writer for every live name
   (`graph_ops.py:3251-3303`). `run_sn_pools()` invokes it at ordinary-run
   startup (`loop.py:1730-1753`) and again post-drain (`loop.py:2250-2268`).

Therefore deleting the edge alone is temporary: the next startup or post-drain
structural pass derives and merges it again. There is no durable per-edge writer
receipt, so the exact runtime invocation that most recently performed the
`MERGE` cannot be recovered by run id. The code path capable of producing the
live property-complete edge is nevertheless exact, and its twice-per-run
schedule makes recurrence deterministic while the parse result is unchanged.

The edge itself is semantically wrong. Peeling a `spectral` qualifier cannot
introduce a new `logarithm_of_` prefix. The intended surviving relation is the
opposite direction:

`logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel
-[:HAS_PARENT {operator: 'logarithm', operator_kind: 'unary_prefix'}]->
spectral_signal_to_noise_ratio_of_spectrometer_channel`.

That direction removes the explicit logarithm operator and yields its argument.
The rejected qualifier direction instead points the argument back to its own
logarithmic form and closes a two-node cycle.

## Proof that the governed repair was reversed

The 2026-08-23 reversibility receipt
`docs/evidence/sn-release-readiness/ordering-cycle-repair.md` records an exact,
fail-closed deletion of one relationship:

| Source | Relationship properties | Target |
|---|---|---|
| `spectral_signal_to_noise_ratio_of_spectrometer_channel` | `operator='spectral'`, `operator_kind='qualifier'` | `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel` |

That apply moved `HAS_PARENT` from 61,921 to 61,920, left both endpoints and the
legitimate reverse unary-prefix edge unchanged, and reduced bidirectional
`HAS_PARENT` pairs from one to zero. The live read at
**2026-08-25T07:58:05.045Z** resolved each endpoint by its declared
`StandardName.id` exactly once and found the deleted relationship live again
exactly once with the same complete property map. This is recurrence, not a
different edge that happens to join the same names.

## Can orphan-parent reconciliation regrow a terminal predecessor?

**Yes.** `reconcile_orphan_parent_sources()` can regrow a terminal predecessor
through a migrated, non-stale derived source.

The selector at `graph_ops.py:25560-25594` requires only that the parent has no
incoming source, that `derived:<parent>` is absent or not stale, and that all of
the parent's structural children have a non-null stage. It does not require the
parent itself to be live. It also does not reject an existing structural source
because that source is already bound to a different target. The writer at
`graph_ops.py:25707-25724` then reuses the source, overwrites
`produced_sn_id=parent.id`, and `MERGE`s
`(source)-[:PRODUCED_NAME]->(parent)` without deleting another target binding.
The reconciler runs after structural derivation at startup and post-drain
(`loop.py:1790-1805`, `loop.py:2250-2270`).

`derived:conductivity` is the live counterexample:

- `persist_refined_name()` recorded the atomic
  `conductivity -> plasma_electrical_conductivity` source migration at
  `2026-08-23T15:56:28.552Z`, followed one millisecond later by the refine
  change. The source should therefore select and bind only the accepted
  successor.
- The source is currently `status='composed'`, but again has both
  `conductivity` (`superseded`) and `plasma_electrical_conductivity`
  (`accepted`) bindings, while its scalar again says `conductivity`.
- The predecessor is `origin='catalog_edit'`, is terminal, and has five live
  projection children: `radial_conductivity`, `poloidal_conductivity`,
  `toroidal_conductivity`, `vertical_conductivity`, and
  `parallel_conductivity`. Because it is not `origin='derived'`, candidate
  classification bypasses the derived-parent admission check. Every remaining
  selector condition is satisfied once the migrated predecessor edge is
  absent.

Thus the structural maintenance sequence explains more than one invariant
class: `rederive_structural_edges()` recreates the defective `HAS_PARENT` edge,
and `reconcile_orphan_parent_sources()` can recreate a terminal
`PRODUCED_NAME` edge and stale scalar for a migrated structural source. It does
**not** explain the three DD residues or the two DD scalar mismatches below.

## Three scalar-mirror rows

The live invariant census found exactly three scalar mismatches. Each source
has two authored target relationships, exactly one live target, and a scalar
that still names the terminal target.

| Source | Scalar | Sole live target | Other target | Last producing path and diagnosis |
|---|---|---|---|---|
| `dd:plasma_profiles/ggd/mass_density/values` | `mass_density` | `total_plasma_mass_density` (`accepted`) | `mass_density` (`exhausted`) | The signed dual-binding disposition ran through `apply_adjudicated_source_dispositions()` and deliberately retained this exact two-edge row because deleting the scalar-selected edge would remove that target's last producer. The later name-review lifecycle made `mass_density` terminal without atomically moving the source scalar. No relationship was recreated; a liveness transition converted a governed refusal into a sole-live mirror defect. |
| `dd:plasma_sources/source/profiles_1d/ion/momentum/radial` | `radial_ion_momentum` | `radial_ion_momentum_source` (`accepted`) | `radial_ion_momentum` (`exhausted`) | The same signed dual-binding path retained the row under the same last-producer refusal. The historical `backfill_refine` from `radial_ion_momentum` to `radial_ion_momentum_source` exists, but the old binding and scalar remained; terminalization exposed the mismatch. Again, this is a lifecycle/mirror transition, not edge regrowth. |
| `derived:conductivity` | `conductivity` | `plasma_electrical_conductivity` (`accepted`) | `conductivity` (`superseded`) | `persist_refined_name()` atomically migrated the source to the successor; the later writer capable of restoring the observed predecessor scalar plus edge is `reconcile_orphan_parent_sources()`. This is genuine recurring relationship production. |

The first two rows should use the existing signed sole-live scalar repair, but
their prevention boundary is lifecycle settlement: a target transition to a
terminal stage must not leave a source scalar selecting it. The third row needs
the same scalar repair **and** a selector guard preventing structural recovery
from reusing a migrated source to bind a terminal predecessor.

## Three DD no-live-target rows

These are not current orphan-reconciler products. Each source has
`status='composed'`, no scalar, no `PRODUCED_NAME` edge, exactly one authored
`FROM_DD_PATH` edge to the identically named `IMASNode`, and no DD
`HAS_STANDARD_NAME` projection.

| Source | Created and composed | Last producing path | Diagnosis |
|---|---|---|---|
| `dd:ntms/time_slice/mode` | `2026-07-31T10:20:30.611Z` | Legacy `seed_parent_sources()` / `_materialize_derived_parent_rows()` using `_derived_parent_source_metadata(parent_dd_path=...)` | Historical derived-parent provenance was collapsed onto a DD container identity. It is genuine residue, not a current recurring write. |
| `dd:summary/pedestal_fits` | `2026-07-31T10:20:30.566Z` | Same legacy parent materializer | Same identity-collapse residue. |
| `dd:waves/coherent_wave` | `2026-07-31T10:20:30.653Z` | Same legacy parent materializer | Same identity-collapse residue. Its retained source description concerns a coherent-wave Fourier coefficient, but no live Standard Name target remains. |

The historical implementation selected `dd:<parent_dd_path>` when a common
child DD path existed, created a `source_type='dd'` source with
`batch_key='derived_parent'`, merged its `PRODUCED_NAME` edge, and added
`FROM_DD_PATH`. Commit `3b74745e` replaced that behavior on 2026-07-31: current
`_derived_parent_source_metadata()` always emits `derived:<parent>` and
explicitly forbids treating a common child DD path as realization authority
(`graph_ops.py:3632-3647`). Consequently there is no current producer expected
to recreate these three DD rows after a governed signed release. Any later
attachment needs independent present target authority.

## Schema and zero sanity

The source census was measured live at **2026-08-25T07:38:54.592Z**. The
queried keys were present before any zero was interpreted:

| Probe | Candidates | With queried property |
|---|---:|---:|
| `StandardName.id` | 4,656 | 4,656 |
| `StandardName.name_stage` | 4,656 | 4,656 |
| `StandardNameSource.id` | 9,668 | 9,668 |
| `StandardNameSource.status` | 9,668 | 9,668 |
| `StandardNameSource.source_type` | 9,668 | 9,668 |
| `StandardNameSource.produced_sn_id` | 9,668 | 5,235 |

The graph contained **5,351** authored
`(StandardNameSource)-[:PRODUCED_NAME]->(StandardName)` edges; all 5,351
targets had both `id` and `name_stage`. The reverse-direction probe found zero
edges. Against that proven direction and schema, the census measured **39**
composed/attached sources with zero live target and **3** scalar mismatches.
The three named DD sources each resolved once by `StandardNameSource.id`, each
had one exact `FROM_DD_PATH` target, and each had zero `PRODUCED_NAME` targets;
their zeros are therefore relationship absence, not a guessed key or reversed
direction. The exact qualifier-edge read separately resolved both endpoint ids
once and found one matching `HAS_PARENT` relationship with
`operator_kind='qualifier'`.

## Producer-class boundary

There is no single producer for every reported row:

- **One recurring maintenance chain explains two relationship classes:**
  structural re-derivation recreates the reversed qualifier `HAS_PARENT`; orphan
  source reconciliation can recreate a migrated source's terminal predecessor
  `PRODUCED_NAME` edge and scalar.
- **Two DD scalar rows are lifecycle defects:** their retained governed
  dual-bindings became sole-live mismatches when the scalar-selected targets
  became terminal. No deleted edge was recreated.
- **Three DD no-target rows are legacy identity residue:** the pre-`3b74745e`
  parent materializer wrote them. Current structural materialization no longer
  uses that DD identity shape.

This diagnosis is read-only. It made no graph mutation and supplies no
authority to delete or recreate any relationship.
