# Inherited residue release disposition

## Outcome

**Complete: 7 inherited residue rows have an explicit first-release
disposition. Five are dispositioned by this plan and two remain holds owned by
named external authorities. No row is left to source-count, source-binding, or
name-stage filtering by accident.**

This record applies to the first full catalog release. It does not mutate the
graph, invent a Standard Name, weaken a last-producer guard, or grant catalog
acceptance. A source-less identity is neither admitted nor rejected merely
because its producing-source count is zero. Instead, each row below states the
release result explicitly. Where the result is a hold, it overrides the current
production classifier until the named lifting condition is met.

The live graph was measured at `2026-08-24T16:07:54.534000000+00:00` against
4,666 `StandardName` nodes and 9,639 `StandardNameSource` nodes. The canonical
full-release population/classifier read 2,335 accepted-or-approved identities
and 2,037 currently eligible identities.

## Exact disposition matrix

| Inherited residue identity or source | Live Cypher result establishing current state | Explicit first-release disposition | Authority or external owner |
|---|---|---|---|
| `fast_ion_charge_state_power_at_inside_flux_surface` | **Identity-state query:** exists; `accepted / valid / docs accepted`; name/docs scores `0.8875 / 0.9125`; **0 total and 0 live producers**. The production classifier currently calls it eligible. | **HOLD — do not publish.** Its accepted lifecycle and scores do not resolve whether the DD leaf describes fast- or thermal-ion deposition. The hold is deliberate even though the current generic classifier would admit it. | **External:** IMAS Data Dictionary maintainers, issue 294 and PR 295. Lift only after the active DD resolves the recipient contradiction and the graph is rebuilt and re-read. |
| `dd:b_field_non_axisymmetric/time_slice/field_map/grid/phi` / intended `toroidal_coordinate_of_field_map_grid` | **Source-binding query:** source exists, `composed`, scalar and sole live target both `toroidal_angle_of_measurement_position`. **Identity-state query:** `toroidal_coordinate_of_field_map_grid` does not exist; `StandardName.id` coverage is 4,666/4,666. | **HOLD — emit no field-map-grid identity.** Retain the incumbent graph binding as a named refusal; it is not semantic authority to publish the grid coordinate as a measurement position. Do not substitute a flux-coordinate identity. | **External:** `imas-standard-names`, which owns the closed locus vocabulary and the governed `field_map_grid` token decision. |
| `neutron_flux_due_to_fusion` | **Identity-state query:** exists; `accepted / valid / docs exhausted`; name/docs scores `0.8625 / 0.8000`; **0 total and 0 live producers**. **Production-classifier query:** normal full-release classification excludes it as `documentation_not_accepted` with `docs_stage='exhausted'`. | **HOLD — do not publish in the first release.** The reason is the unaccepted documentation, not the lack of a producer. If ordinary docs review later earns acceptance, source-less status alone does not bar publication; source attachment remains separately guarded until the exhausted power predecessor no longer depends on the candidate path as its last producer. | **This plan:** the locked drain-first decision plus the ordinary documentation gate supplies the release authority. |
| `tendency_of_total_thermal_plasma_internal_energy` | **Identity-state query:** exists; `accepted / valid / docs accepted`; name/docs scores `0.89375 / 0.85000`; **0 total and 0 live producers**. **Production-classifier query:** the production classifier calls it eligible. | **INCLUDE.** Its source-less state is explicitly authorized for catalog publication. It is a physically meaningful time derivative and must not be folded into `plasma_internal_energy`, whose spelling omits tendency, total, and thermal. A later producer attachment requires a distinct reviewed path, but that graph repair is not a release prerequisite. | **This plan:** section 8 carries the explicit source-less release authority. |
| `dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial` with live targets `radial_neutral_internal_state_momentum_flux` and `radial_neutral_state_momentum_flux` | **Dual-bound query:** the source is `attached`; scalar selects `radial_neutral_internal_state_momentum_flux`; exactly two live targets, both `accepted / valid / docs accepted`, with name scores `0.95625` and `0.85625`. **Production-classifier query:** the production classifier currently calls both eligible. | **PUBLISH ONLY `radial_neutral_state_momentum_flux`; HOLD `radial_neutral_internal_state_momentum_flux`.** The earlier semantic adjudication selected the shorter identity as the canonical family member because the DD adds no distinction beyond neutral state, momentum flux, and radial component. Keep both graph edges unchanged until a separately signed fold can satisfy the last-producer guard; graph mutation authority and release membership are distinct. | **This plan:** it carries the release selection. The semantic basis is the predecessor's exact dual-bound adjudication; no external decision is pending. |
| `voltage_of_diagnostic_antenna` / intended successor `voltage_of_ece_channel` | **Identity-state and source-binding queries:** predecessor exists at `exhausted / valid / docs accepted`, score `0.625`, stop reason `grammar_invalid`; its only source `dd:ece/channel/t_e_voltage` is `stale`, so it has **0 live producers**. The intended successor does not exist and no explicit or refined successor is present. The predecessor is absent from the normal full-release population. | **HOLD BOTH — publish neither the exhausted predecessor nor a fabricated successor.** Preserve the stale binding as ledger history. A future attempt must create a reviewable source-less successor through a sanctioned transition that explicitly excludes the stale producer from migration and then earns ordinary review acceptance. | **This plan:** it carries the first-release hold. No provider, budget, or external-owner wait substitutes for the missing sanctioned transition and review result. |
| `dd:spectrometer_x_ray_crystal/channel/reflector/centre/phi` / missing reflector-surface-center identity | **Source-binding query:** source exists, `attached`, scalar and sole live target both `toroidal_angle_of_measurement_position`. The distinct sibling `reflector/sphere_centre/phi` is solely bound to accepted, valid `toroidal_coordinate_of_reflector`. **Production-classifier query:** both incumbent identities are independently classifier-eligible. | **HOLD THE MISSING SURFACE-CENTER IDENTITY; publish no entry on its behalf.** Do not treat `toroidal_angle_of_measurement_position` as the reflector-center catalog representation, and do not retarget to `toroidal_coordinate_of_reflector`, which names the sphere center of curvature. The two incumbent identities may publish for their independently reviewed meanings; the residue remains a named graph refusal until an ordinary-reviewed surface-center identity exists. | **This plan:** it carries the release hold and preserves the reviewed identity-split requirement. |

Accounting is exact: **7 rows = 5 dispositioned here + 2 externally owned; 0
undispositioned.** Across the eight target outcomes represented by those rows,
there are **2 include decisions + 6 hold decisions**. The dual-bound row contains
one include and one hold because one inherited source carries two live identity
claims; it is counted once in the row accounting and both target identities are
named explicitly.

## Live-query evidence

All graph reads were read-only. Before trusting any zero, the property-coverage
query established the relevant keys: `StandardName.id=4666/4666`,
`name_stage=4666/4666`, `StandardNameSource.id=9639/9639`, and
`status=9639/9639`. Four Standard Names lack validation/docs lifecycle fields
graph-wide, but every existing identity in this matrix returned the fields used
above; absence claims use the fully covered `id` key.

### Property coverage

```cypher
MATCH (n:StandardName)
WITH count(n) AS sn_candidates,
     count(n.id) AS sn_with_id,
     count(n.name_stage) AS sn_with_name_stage,
     count(n.validation_status) AS sn_with_validation_status,
     count(n.docs_stage) AS sn_with_docs_stage
MATCH (s:StandardNameSource)
RETURN sn_candidates, sn_with_id, sn_with_name_stage,
       sn_with_validation_status, sn_with_docs_stage,
       count(s) AS source_candidates,
       count(s.id) AS source_with_id,
       count(s.status) AS source_with_status,
       count(s.produced_sn_id) AS source_with_scalar,
       count(s.dd_path) AS source_with_dd_path
```

Result: `StandardName 4666 / id 4666 / name_stage 4666 /
validation_status 4662 / docs_stage 4662`; `StandardNameSource 9639 / id 9639 /
status 9639 / scalar 5644 / dd_path 8741`.

### Named identities, producer closure, and successors

```cypher
UNWIND $ids AS requested_id
OPTIONAL MATCH (n:StandardName {id: requested_id})
OPTIONAL MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n)
WITH requested_id, n, collect(DISTINCT s) AS sources
OPTIONAL MATCH (n)-[:HAS_SUCCESSOR]->(successor:StandardName)
WITH requested_id, n, sources, collect(DISTINCT successor.id) AS successors
OPTIONAL MATCH (refined:StandardName)-[:REFINED_FROM]->(n)
RETURN requested_id, n IS NOT NULL AS exists,
       n.name_stage AS name_stage,
       n.validation_status AS validation_status,
       n.docs_stage AS docs_stage,
       n.reviewer_score_name AS reviewer_score_name,
       n.reviewer_score_docs AS reviewer_score_docs,
       n.refine_stop_reason AS refine_stop_reason,
       size([s IN sources WHERE s IS NOT NULL]) AS total_sources,
       size([s IN sources
             WHERE s IS NOT NULL AND s.status IN ['composed', 'attached']])
         AS live_sources,
       successors,
       [x IN collect(DISTINCT refined.id) WHERE x IS NOT NULL]
         AS refined_successors
ORDER BY requested_id
```

The parameter set contained every Standard Name named in the matrix plus the
two intended but absent identities. The per-row results are reported verbatim
in the matrix. Both absent-identity results returned `exists=false`; the
antenna predecessor returned empty explicit and refined successor lists.

### Complete current dual-bound cohort

```cypher
MATCH (source:StandardNameSource)
WHERE source.status IN ['composed', 'attached']
OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(target:StandardName)
WITH source, collect(DISTINCT target) AS targets
WITH source,
     [target IN targets
      WHERE NOT (target.name_stage IN ['superseded', 'exhausted'])]
       AS live_targets
WHERE size(live_targets) > 1
RETURN source.id AS source_id,
       source.dd_path AS dd_path,
       source.status AS source_status,
       source.produced_sn_id AS produced_sn_id,
       [target IN live_targets | {
         id: target.id,
         stage: target.name_stage,
         validation: target.validation_status,
         docs_stage: target.docs_stage,
         name_score: target.reviewer_score_name,
         docs_score: target.reviewer_score_docs
       }] AS live_targets
ORDER BY source.id
```

Result: exactly **1** row,
`dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial`, with the
two targets and lifecycle values stated above.

### Exact DD source bindings

```cypher
UNWIND $source_ids AS requested_id
OPTIONAL MATCH (s:StandardNameSource {id: requested_id})
OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(target:StandardName)
WITH requested_id, s, collect(DISTINCT target) AS targets
RETURN requested_id, s IS NOT NULL AS exists,
       s.status AS source_status,
       s.dd_path AS dd_path,
       s.produced_sn_id AS produced_sn_id,
       [target IN targets WHERE target IS NOT NULL | {
         id: target.id,
         stage: target.name_stage,
         validation: target.validation_status
       }] AS all_targets,
       [target IN targets
        WHERE target IS NOT NULL
          AND NOT (target.name_stage IN ['superseded', 'exhausted'])
        | target.id] AS live_targets
ORDER BY requested_id
```

The parameter set was the field-map, antenna, dual-bound, reflector-surface,
and reflector-sphere source ids. Every source existed. Their exact states and
targets are reported in the matrix.

### Canonical production release classification

The repository's canonical full-release population query and shared
documentation-review traversal were run without a batch or domain filter, then
the matrix identities were projected from the result. This is the production
Cypher in `_fetch_export_population()` followed by
`_classify_export_population()`, not a hand-written approximation.

Result: population `2335`, currently eligible `2037`. Among matrix identities,
`fast_ion_charge_state_power_at_inside_flux_surface`, both radial-neutral
momentum-flux identities,
`tendency_of_total_thermal_plasma_internal_energy`,
`toroidal_angle_of_measurement_position`, and
`toroidal_coordinate_of_reflector` were eligible;
`neutron_flux_due_to_fusion` was excluded as
`documentation_not_accepted`; and `voltage_of_diagnostic_antenna` was absent
from the accepted-or-approved population.

This classifier result is evidence of current mechanics, not the disposition
authority. In particular, the externally held fast-ion row and the losing
dual-bound identity must not be published merely because the generic classifier
currently admits them.

## Release handoff

The release owner must carry the six hold outcomes into the first-release
exclusion record by identity or intended identity. A release artifact that contains either externally held
identity, both dual-bound target identities, the exhausted antenna predecessor,
or an invented reflector-center identity contradicts this disposition. A
release artifact that omits
`tendency_of_total_thermal_plasma_internal_energy` solely because it has no
producer also contradicts it. Re-read the seven rows immediately before release
because the evidence above is a live-state snapshot, while the dispositions and
their lifting conditions are durable authority.
