# WEST batch accepted names — physical-correctness audit

provisional: true

Are the accepted standard names bound to WEST batch sources physically correct
and self-descriptive? Each row below is judged against the data-dictionary
documentation of the source path it is bound to, its unit, and the name's own
description. Rows are appended as they are judged.

## How the cohort was drawn

The WEST batch membership predicate is the 342 data-dictionary source paths in
`imas_codex/standard_names/manifests/west_production_dd_paths.yaml`. Drawn
through `GraphClient()` against the live graph:

```cypher
MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
WHERE src.source_id IN $paths AND sn.name_stage = 'accepted'
OPTIONAL MATCH (dd:IMASNode {id: src.source_id})
RETURN sn.id AS name, src.source_id AS path, sn.unit, sn.description,
       dd.unit, dd.documentation
```

**341 accepted names** are bound to those 342 paths. The audited sample is
**every fourth row of the path-ordered cohort — 86 names**, which spreads the
sample over 20 IDSs (equilibrium 19, summary 15, magnetics 6, ic_antennas 5,
spectrometer_visible 5, camera_x_rays 4, core_profiles 4, hard_x_rays 4,
interferometer 4, ece 3, polarimeter 3, soft_x_rays 3, bremsstrahlung_visible 2,
calorimetry 2, wall 2, barometry 1, camera_ir 1, pf_active 1, pf_passive 1,
spectrometer_mass 1). Sampling is deterministic and independent of the verdict,
so it cannot select for agreement.

### The instrument was controlled before any absence was reported

The first two query forms returned **0 rows** and **null names**, and both were
instrument faults rather than findings: `StandardNameSource` carries
`source_id`, not `path`; the accepted flag is `name_stage`, not `status`
(`sn.status` is the catalog lifecycle field and holds only `draft` (2934) and
`superseded` (2196), so a `status = 'accepted'` predicate matches nothing); and
the name string is `sn.id`. `IMASNode` likewise keys on `id` and `unit`, not
`path`/`units` — the first join returned `dd_doc` on 0 of 341 rows. The
corrected join returns documentation on **341 of 341**, and a positive control
counts 5130 `StandardName` nodes. For a `/value` leaf whose own documentation is
the literal string `Value`, the parent container's documentation is used
(present for 332 of 341).

## Verdicts

