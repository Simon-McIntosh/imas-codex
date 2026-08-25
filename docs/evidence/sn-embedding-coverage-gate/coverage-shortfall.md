# Description-embedding coverage shortfall

## Outcome

The WEST release survivors do **not** carry an accepted-name embedding
shortfall. The exact 369-identity export survivor set contains 363 identities
whose name lifecycle is accepted, and **0/363 lack an embedding**. The broader
current WEST review cohort likewise has **0/385 accepted identities without an
embedding**. Both measurements resolve identity through `StandardName.id`; the
sanity counts are 363/363 and 385/385 respectively.

The whole current accepted-or-approved corpus has **1/2,302 identities without
an embedding** (coverage **2,301/2,302 = 99.9566%**). That one row has a real
description and no vector, so it is an **embedding gap** that deterministic
re-embedding can repair without provider cost:

- `ratio_of_toroidal_ion_velocity_to_magnetic_field_magnitude` — accepted;
  description begins “Signed ratio of the charge-state-averaged toroidal ion
  bulk velocity…”; `embedded_at` and `embed_failed_at` are both null.

There are **0 accepted description gaps**. The two cost/ownership classes are
therefore reported separately and are not summed:

| Population | Accepted or approved identities | `id` sanity | Missing vector on real description | Missing description and vector |
|---|---:|---:|---:|---:|
| WEST export survivors | 363 | 363/363 | **0** | **0** |
| Current WEST review cohort | 385 | 385/385 | **0** | **0** |
| Whole accepted corpus | 2,302 | 2,302/2,302 | **1** | **0** |

The current schema/property preflight is positive rather than a guessed-key
zero: the graph contains **4,658 `StandardName` nodes, 4,658/4,658 with `id`,
4,632 with `description`, 4,630 with `embedding`, 4,656 with `embedded_at`, and
3,054 with `links`**. The instrument is aimed at `StandardName.embedding`, the
LinkML-declared vector of `StandardName.description`, not at
`description_embedding` or an undeclared `name` key.

The export-time cohort and the current cohort should not be conflated. The
closing WEST export census at 2026-08-25 07:20:29 UTC contained 416 identities
and 379 accepted or approved names; subsequent ordinary pipeline work raised
the current accepted-or-approved count within those same 416 ids to 385. The
export survivor identity list remains the frozen 369-row artifact used above.

## What Gate A actually asserted

The WEST run artifact records:

```text
Gate A: graph test failure(s) (advisory for RC release)
```

At source commit `bbd9b6718e7343506dc76874b4e4ae99b3306077`, Gate A invoked
the existing graph/corpus-health pytest suite. The relevant named check is
`TestDescriptionEmbeddingCoverage::test_description_embedding_coverage[StandardName]`.
It does **not** assert a percentage coverage threshold. It selects rows whose
`embedded_at` is non-null while `embedding` is null and asserts the corrupted
count is exactly **zero**.

The named check was run once against the live graph for this measurement and
reproduced the failure:

```text
FAILED ...test_description_embedding_coverage[StandardName]
AssertionError: StandardName has 26 nodes with embedded_at set but no embedding vector.
```

Thus the gate's actual assertion is **processed-vector retention = 100%, or 0
processed rows missing a vector**; the live result is **26 against a threshold
of 0**. RC logic changed that failed suite result to an advisory. It did not
change the check's threshold.

Those 26 are a different population from the accepted-coverage table above.
All 26 are non-accepted: 14 superseded, 8 drafted, and 4 pending. More
importantly, the required ownership split is **0 embedding gaps and 26
description gaps**: every one lacks a real description as well as a vector.
Consequently, the named Gate A failure cannot be repaired by deterministic
re-embedding alone. These rows need content generation and review or an
explicit lifecycle-based exclusion from this retention check. Conversely, the
single accepted-corpus embedding gap is not selected by the current Gate A
predicate because its `embedded_at` is null.

This distinction changes the disposition discussion:

- **Withhold affected WEST identities** has no work to do for embedding
  coverage: affected accepted identities inside the 369-row WEST release set
  are **zero**.
- **Fix and enforce** requires two mechanisms, not one aggregate repair:
  deterministic re-embedding for the one accepted real-description gap, and
  generation/review or lifecycle scoping for the 26 non-accepted description
  gaps.
- **Relax deliberately** would be a policy choice, not a response compelled by
  WEST release coverage. The release subset already clears accepted-name
  embedding coverage at 100%; the failed zero-threshold check is graph-wide and
  measures processed-vector retention over dead and unfinished lifecycle rows.

No threshold was changed and no graph state was mutated by this node.

## `radial_coordinate_of_reflector`

`radial_coordinate_of_reflector` is currently `name_stage=drafted`,
`docs_stage=pending`, and `validation_status=valid`. It has neither the short
description nor an embedding, although `embedded_at` is
2026-08-21T19:55:30.556Z. It has one live producer and one DD projection, both
for
`spectrometer_x_ray_crystal/channel/reflector/sphere_centre/r`; the source and
DD node both carry the usable source description “Major radius (R) coordinate
of sphere centre.” The name node was generated on 2026-07-28 and is therefore
not currently a bare relationship-created endpoint, but generation persisted
no `description`, leaving review without the content required to proceed.

Its immediate cause **does explain the named-check class**: all 26 rows with
`embedded_at` but no vector also lack a real description, so the reflector is
one of 26 description gaps rather than an isolated embedding-worker miss. It
does **not** explain the accepted-corpus shortfall: that sole accepted row has a
real description and simply lacks an embedding. The reflector is withheld from
the WEST export under `documentation_not_accepted` and is not one of the 369
release survivors.

The plain-language semantic issue remains visible: the DD path is the radial
coordinate of the reflector sphere centre. That is distinct from a generic
reflector surface coordinate or from the centre of curvature unless the
reflector geometry establishes those identities as the same. This measurement
does not adjudicate that identity.

## Dangling links at graph source

The same WEST export run provides two deliberately different counts:

- Gate B inspected the candidate nodes' graph-backed `links` properties and
  found **295 dangling link occurrences** before final catalog validation.
- After two catalog-model rejections fixed the final published identity set,
  the pre-write pruning pass removed **296 dangling internal link occurrences**.

The 296 are graph-source references, not defects discovered in an emitted
catalog: `run_export` fetched each candidate's `StandardName.links`, built the
final published-name set, then pruned before writing domain YAML. The one-count
increase over Gate B is consistent with the later, smaller published set
creating one additional dangling target after catalog validation. The export
then failed during hierarchy ordering, so no complete `catalog.yml` or export
report existed from which to obtain this count.

A current read-only re-census, aimed at the exact frozen 369 survivor ids,
finds **292 dangling occurrences from 211 source identities to 212 absent
targets**. Its schema sanity is **369/369 ids**, **367 nodes with a `links`
property**, and **885 total link occurrences**. The positive control fires on
`accumulated_deposited_energy_of_plasma_facing_component ->
name:power_due_to_convection`, one of the export's recorded examples. The
current 292 versus export-time 296 is real post-run graph drift, not grounds to
rewrite the export result. It is why the export-time 296 and the current-source
292 are both retained with their observation boundary.

## Evidence inputs

- WEST production export log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070724966717-n-westrcstatus/real-export.log`
- Frozen 369/47 export identity partition and 295 Gate B source advisories:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070724966717-n-westrcstatus/export-filter-census.json`
- Closing 416-identity WEST cohort:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070724966717-n-westrcstatus/final-census.json`
- This node's read-only graph measurement:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T141822953561-n-coveragemeasure/coverage-measurement.json`
- Named graph-check log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T141822953561-n-coveragemeasure/named-check.log`
- Named-check lifecycle and ownership split:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T141822953561-n-coveragemeasure/named-check-split.json`
