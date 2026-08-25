NEEDS-HELP: The exact 12-row signed repair is prepared, but the sanctioned operator's graph-wide collateral snapshot exceeded the five-minute hang boundary twice; both transactions rolled back and no graph row changed.

tried: Re-measured the live graph, generated a canonical signed repair authority for the exact drafted and pending cohort, previewed it, and attempted the authorized apply. The first invocation spent 120 seconds in worktree model regeneration and was stopped at five minutes before preview completed. After the models were current, the corrected retry completed preview and entered apply, but blocked in `_collateral_snapshot` until the same five-minute boundary. An independent query after each stop found the original 14/8/4 split and zero `clear_false_embedding_stamp` receipts, proving neither attempt committed.

options: (1) Run this already-signed authority in an operational lane whose allowed command duration accommodates both graph-wide collateral snapshots. (2) Optimize or bound the sanctioned operator's collateral snapshot, then preview and apply the same exact authority again. (3) Add a dedicated governed stale-embedding-stamp operator with equivalent exact-cohort, compare-and-set, receipt, and collateral guarantees. Raw Cypher is not an option under the governing constraint.

leaning: Option 1. The existing signed operator already expresses the intended exact mutation, participant locks, replay protection, receipt cardinality, and collateral immutability; the observed failure is execution duration, not missing authority or a semantic ambiguity.

cost-if-wrong: If the operational lane still cannot complete the collateral proof, no mutation commits and only execution time is lost. Replacing the operator or weakening its proof would require new implementation and review and could invalidate the programme's sanctioned-write guarantee.

# Stale embedding-stamp repair evidence

## Live population before either attempt

The schema sanity query was aimed explicitly at `StandardName` and returned:

| Measure | Count |
|---|---:|
| StandardName candidates | 4,658 |
| candidates with `id` | 4,658 |
| candidates with `name_stage` | 4,658 |
| candidates with `description` | 4,632 |
| candidates with `embedding` | 4,631 |
| candidates with `embedded_at` | 4,657 |
| candidates with `docs_stage` | 4,654 |

The target predicate was `embedded_at IS NOT NULL AND description IS NULL AND embedding IS NULL`. It returned 26 real rows, split exactly as follows:

| `name_stage` | Rows | Disposition |
|---|---:|---|
| `superseded` | 14 | Excluded; historical lineage must remain untouched |
| `drafted` | 8 | Exact repair target |
| `pending` | 4 | Exact repair target |

The measured live cohort therefore contains exactly 12 rows. All eight drafted rows have `docs_stage='pending'`; the four pending rows have null `docs_stage`. Every target has null description and null embedding and shares the false timestamp `2026-08-21T19:55:30.556Z`.

The 12 measured identities are:

- drafted: `line_averaged_neon_density`, `poloidal_ion_state_momentum_diffusion_coefficient`, `radial_coordinate_of_reflector`, `radius_of_soft_xray_detector`, `ratio_of_diamagnetic_vorticity_to_major_radius`, `toroidal_coordinate_of_spectrometer`, `toroidal_tritium_velocity`, `vertical_coordinate_of_reflector`
- pending: `coolant_mass`, `flux_at_wall_due_to_recombination`, `outline`, `particle_temperature`

Baseline control counts were 8,597 `StandardNameChange` nodes and 5,309 `PRODUCED_NAME` relationships. Identity and field baselines were 4,658 StandardName identities, 4,632 descriptions, 4,631 embeddings, 4,658 populated `name_stage` values, and 4,654 populated `docs_stage` values.

## Sanctioned authority

The repository's generic signed-repair operator is the sanctioned mutation path. The emitted authority selects the complete exact 12-row caller set, applies only `set_properties {embedded_at: null}`, locks participants, compares full participant fingerprints, proves out-of-allowlist immutability, and requires one `StandardNameChange` receipt per target. It does not generate descriptions or embeddings.

- authority: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T185317161763-n-stalestamp/artifacts/false-embedding-stamp-authority.json`
- file SHA-256: `bbc5827bf7f62bb0f40fbb0ef45d00ac80b929fc28cd13b2311782bb99f2a537`
- signed payload SHA-256: `afa7af2646251e54bebb8d0735957119913a56c43defc1fd969d392c953a47d9`

The first invocation was stopped at the five-minute hang boundary after its generated-model freshness check timed out at 120 seconds. The corrected retry reached the apply transaction, then remained in `signed_manifest._collateral_snapshot` until the same boundary. Its interrupt traceback also exposed a Neo4j driver rollback-buffer `BufferError`, so graph state was verified independently rather than inferred from process exit.

## Verified rollback state

After both attempts, the false-stamp population remains exactly 14 superseded / 8 drafted / 4 pending, and the receipt query returns zero `StandardNameChange` nodes with operation `clear_false_embedding_stamp`. Thus:

- exact mutated count: **0**, not the required 12;
- live-row count: **12**;
- superseded count: **14 before and 14 after**;
- `StandardNameChange`: **8,597 before and 8,597 after**;
- `PRODUCED_NAME`: **5,309 before and 5,309 after**;
- StandardName identities, descriptions, embeddings, `name_stage`, and `docs_stage`: unchanged;
- no description was generated and no embedding was computed.

The node is therefore blocked, not complete. Claiming completion would conflate a valid authority and passing preview with a committed graph mutation.

## Named graph assertion

Command:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -m graph tests/graph/test_data_quality.py::TestDescriptionEmbeddingCoverage::test_description_embedding_coverage -p no:cacheprovider
```

Verbatim summary:

```text
8 passed, 5 skipped, 1 warning in 73.80s (0:01:13)
```

The assertion still passes because `StandardName` coverage is intentionally scoped to accepted identities; the 26 false stamps are all non-accepted and remain outside that release gate. This pass demonstrates no regression but does not satisfy the 12-row repair done-when.

Logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T185317161763-n-stalestamp/logs/stale-stamp-repair.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T185317161763-n-stalestamp/logs/stale-stamp-repair-retry.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T185317161763-n-stalestamp/logs/scoped-coverage-assertion.log`
