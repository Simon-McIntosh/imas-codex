# Pre-import archive score recovery

## Verdict

The closest available pre-import graph archive does **not** show that the
2026-07-04 catalog import deleted `StandardNameReview` history. The archive has
no `StandardNameReview` label and no `HAS_REVIEW` relationship at all. Only
**10 of the 1,181** live catalog-import identities existed in that checkpoint;
none of those ten carried a name-axis review record, and only two carried a
non-null `reviewer_score_name` scalar.

The recovery disposition is therefore:

- **2 recoverable-by-projection** from an identical archived identity and its
  non-null name-score scalar;
- **1,179 without a recoverable archived score**, comprising **8** identities
  that existed but were name-score-null and **1,171** identities absent from
  the checkpoint.

For operational purposes the latter 1,179 form the **never-scored** recovery
branch: this archive supplies no score to project, so they require ordinary
review unless a nearer pre-import checkpoint is found. This is deliberately
not a claim that all 1,171 absent identities were never reviewed at any moment:
the archive precedes the import by 71 days, 5:21:24.226986, so it cannot observe
work created and lost inside that interval.

The two recoverable scalars also do not prove that the catalog import erased
them. The historical importer at
`imas_codex/standard_names/catalog_import.py@4b949931^:_write_import_entries`
used `MERGE` and omitted `reviewer_score_name`, reviewer-model fields and review
relationships from its catalog-owned `SET`. The snapshot comparison proves
that two scalar scores existed before the import and are null now; it does not
identify the intervening writer that cleared them.

## Archive receipt

| Field | Value |
|---|---|
| Archive | `/home/ITER/mcintos/.local/share/imas-codex/exports/imas-codex-graph-dev-c40162c.tar.gz` |
| Archive size | 2,427,644,023 bytes |
| Archive SHA-256 | `d38487d37f495e49448bac25bf25072f9992698192e72a9e8714d1b646520a12` |
| Manifest timestamp | `2026-04-24T15:59:14.405014+00:00` |
| Manifest version | `5.3.0rc5.dev308+g26b1d2787.d20260424` |
| Manifest commit | `c40162c60868b825addae15f0af1372fa81b6b31` |
| Catalog import timestamp | `2026-07-04T21:20:38.632Z` |
| Precedes import by | 71 days, 5:21:24.226986 |

The manifest timestamp, archive modification date of 2026-04-24, and embedded
commit all predate `2026-07-04T21:20:38.632Z`. The scratch graph contained 269
`StandardName` nodes, zero `StandardNameReview` nodes, and no recognized
`HAS_REVIEW` relationship type.

## Complete identity overlap

The live cohort definition is the exact 1,181-row catalog-import population:
`name_stage='accepted'`, `validation_status='valid'`,
`docs_stage='accepted'`, `origin='catalog_edit'`, null
`reviewer_score_name`, and null `reviewer_model_name`. All 1,181 live rows have
`created_at=2026-07-04T21:20:38.632Z`.

| Identity | In archive by exact `id` | Name-axis reviews | Archived `reviewer_score_name` | Disposition |
|---|---:|---:|---:|---|
| `effective_charge` | yes | 0 | null | never-scored at checkpoint |
| `electron_temperature` | yes | 0 | null | never-scored at checkpoint |
| `fast_electron_density` | yes | 0 | null | never-scored at checkpoint |
| `ion_pressure` | yes | 0 | null | never-scored at checkpoint |
| `lower_triangularity_of_plasma_boundary` | yes | 0 | null | never-scored at checkpoint |
| `normalized_collisionality` | yes | 0 | **0.4375** | **recoverable-by-projection** |
| `normalized_toroidal_flux_coordinate_at_sawtooth_inversion_radius` | yes | 0 | null | never-scored at checkpoint |
| `thermal_ion_density` | yes | 0 | **0.99375** | **recoverable-by-projection** |
| `toroidal_current_density` | yes | 0 | null | never-scored at checkpoint |
| `trapped_fast_particle_pressure` | yes | 0 | null | never-scored at checkpoint |
| Other live cohort identities | **no: 1,171** | 0 | unavailable | no archived score; ordinary review required |

The arithmetic closes in both useful views:

- identity coverage: **10 present + 1,171 absent = 1,181**;
- recovery disposition: **2 recoverable + 8 present-but-never-scored + 1,171
  absent = 1,181**, so **2 recoverable + 1,179 requiring review = 1,181**.

Representative live bindings show that the overlap is not a set of detached
labels:

- `normalized_collisionality` means the dimensionless ratio of a specified
  interspecies collision frequency to its characteristic streaming, transit or
  bounce frequency. It is accepted and bound to
  `dd:gyrokinetics_local/collisions/collisionality_norm`. Its archived score is
  **0.4375**, below the 0.85 acceptance bar; recovering it restores historical
  evidence but does not authorize acceptance.
- `thermal_ion_density` means the local density of thermal ions of a specified
  species, summed over ionic charge states. It is accepted and bound to paths
  including `dd:core_profiles/profiles_1d/ion/density_thermal` and
  `dd:plasma_profiles/profiles_1d/ion/density_thermal`. Its archived score is
  **0.99375**.
- `effective_charge` is accepted and bound to paths including
  `dd:core_profiles/profiles_1d/zeff` and
  `dd:plasma_profiles/profiles_1d/zeff`; it existed in the archive but had
  neither a name review nor a name-score scalar there.

## Exact projection rule

Recovery is an exact-id, same-axis projection, never an acceptance shortcut:

1. Join the current `StandardName.id` to the archived `StandardName.id`
   exactly. A similar spelling or semantic neighbor is not eligible.
2. For attached archived reviews, first filter
   `StandardNameReview.review_axis = 'names'` and require a non-null review
   score. Documentation-axis reviews are never name-score evidence.
3. Within that same-axis set, prefer `is_canonical=true`; if more than one
   canonical record survives, choose the newest `reviewed_at`, then the stable
   review `id` as a deterministic tie-break. If there is no canonical record,
   use the newest scored same-axis record and record that fallback explicitly.
4. If no same-axis review record exists but the exact archived identity has a
   non-null `reviewer_score_name`, project that scalar as legacy evidence. This
   is the only branch available in the April archive, and it yields exactly the
   two rows above.
5. Project the evidence through the governed recovery path and let the ordinary
   lifecycle consume the score. Do not direct-accept a row, manufacture a
   reviewer model, or copy documentation review metadata onto the name axis.

Taking `max(review.score)` across axes is wrong. It can select a high docs score
for a name with no name review, discards the canonical-review preference, and
inflates recoverability. Axis filtering must happen before ranking or
aggregation.

## Read-only and scratch-isolation proof

The production graph was queried only with `MATCH`, `OPTIONAL MATCH`, `UNWIND`
and `RETURN`. No production load, graph switch, write query or catalog command
was run.

| Production counter | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameChange` nodes | 7,900 | 7,900 | **0** |
| `PRODUCED_NAME` relationships | 5,779 | 5,779 | **0** |

The archive was loaded as database `neo4j` in a separate SLURM scratch job on
alternate loopback-only ports 17687/17474 with authentication disabled only in
that isolated configuration. The successful job completed with exit code 0,
logged `SCRATCH_READY=1`, `ARCHIVE_QUERY_COMPLETE=1`, and
`SCRATCH_REMOVED=1`. The scratch directory was independently checked absent
after the job, and no scratch Neo4j process or allocation remains.

## Evidence files

- `live-before.json` — exact 1,181 live identities and initial counters;
  SHA-256 `0dad5ef40117389e6a6060951e45d5d91fbfe3a591f5c1c8444a82eb6496fe8c`.
- `archive-graph.json` — all 269 archived Standard Names and archived scalar /
  review fields; SHA-256
  `2e082d85b3740bce04200f31569a4ac1b8897dbdab9f8b6cf9a8720ceb80517c`.
- `comparison.json` — exact-id overlap and row disposition; SHA-256
  `7c1ac62523da1e2f7fec300a67740ae2f839d9738bafa60aa224f971b90f2e68`.
- `live-after.json` — final production counters and representative live source
  bindings; SHA-256
  `c8afd080f27688b1add48bbd6c549163c64ff1b92bb70c840ff48ee83269ffcb`.
- `scratch-job-attempt2.log` — successful load, query and cleanup receipt;
  SHA-256
  `44f5046612d7a008b57fbdddb5b7a891b01b4141686fceadb6edc7ee2bfb2e0b`.

All runtime evidence files are under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T084542747458-n-archiverecovery/`.
