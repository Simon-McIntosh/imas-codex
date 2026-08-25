# Stale embedding-stamp repair evidence

The signed repair completed through the repository's sanctioned authority path. It cleared `embedded_at` on exactly the 12 freshly measured live false-stamp rows, retained all 14 superseded rows, wrote 12 internal change receipts, and changed no identity, description, embedding, lifecycle stage, documentation stage, or producing-source relationship.

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

The applying invocation re-measured the live cohort and rebuilt the authority immediately before preview and apply. The fresh authority digest matched the earlier prepared value by derivation, not reuse. Preview admitted all 12 rows with zero refusals and produced manifest SHA-256 `778974c30f9d15050af52b608c14e4728ed4bbdfd9117b2732e68feb2bc1bf86`.

The operator's quiet multi-minute intervals were its two full collateral snapshots. A 30-second log heartbeat proved process liveness while those snapshots completed; no operator code or proof scope was changed.

## Applied state and precision proof

The signed receipt records `outcome='applied'`, 12 admitted rows, 12 mutations, 12 receipt rows, and 24 total persistent writes. Before/after state is:

- exact mutated count: **12**, equal to the **12** freshly measured live rows;
- live-row count: **12**;
- superseded count: **14 before and 14 after**;
- remaining false-stamp split: **14 superseded / 0 drafted / 0 pending**;
- populated `embedded_at`: **4,657 before and 4,645 after**, the exact 12-row decrease;
- `StandardNameChange`: **8,597 before and 8,609 after**, exactly 12 sanctioned receipts;
- `clear_false_embedding_stamp` receipts: **0 before and 12 after**;
- `PRODUCED_NAME`: **5,309 before and 5,309 after**;
- StandardName identities: **4,658 before and 4,658 after**;
- populated descriptions: **4,632 before and 4,632 after**;
- populated embeddings: **4,631 before and 4,631 after**;
- populated `name_stage`: **4,658 before and 4,658 after**;
- populated `docs_stage`: **4,654 before and 4,654 after**;
- all 12 selected identity values, descriptions, embeddings, `name_stage`, and `docs_stage` compare equal before and after;
- no description was generated and no embedding was computed.

The superseded historical cohort was not included in the authority and remained untouched. The generic operator's participant fingerprints and out-of-allowlist collateral comparison additionally proved that the committed transaction changed only its admitted targets and receipts.

## Named graph assertion

Command:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -m graph tests/graph/test_data_quality.py::TestDescriptionEmbeddingCoverage::test_description_embedding_coverage -p no:cacheprovider
```

Verbatim summary:

```text
8 passed, 5 skipped, 1 warning in 76.90s (0:01:16)
```

The assertion still passes because `StandardName` coverage is intentionally scoped to accepted identities. Clearing false timestamps from non-accepted work changes no accepted embedding coverage and therefore correctly leaves the assertion's behavior unchanged.

Logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T185317161763-n-stalestamp/logs/stale-stamp-repair.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T185317161763-n-stalestamp/logs/stale-stamp-repair-retry.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T185317161763-n-stalestamp/logs/scoped-coverage-assertion.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T185317161763-n-stalestamp/logs/stale-stamp-repair-authorized.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T185317161763-n-stalestamp/logs/scoped-coverage-assertion-after.log`
