# Field-map-grid source refusal disposition

## Outcome

The three coordinate sources under
`b_field_non_axisymmetric/time_slice/field_map/grid` are now durable, settled
vocabulary refusals. No `field_map_grid` token was added to ISN, no dependency
pin moved, and the separately settled `radial_coordinate`,
`vertical_coordinate`, and `toroidal_coordinate` base spellings were not
changed.

The terminal source state is `vocab_gap`. The source schema declares:

> `extracted → composed | attached | vocab_gap | failed | stale | skipped | not_physical_quantity`
>
> `vocab_gap: Composition blocked by missing grammar vocabulary`

This is the exact condition here: the coordinate meanings are real, but their
required owner is a discretization structure that the closed grammar does not
admit as an identity locus. `skipped` is not the right state because its schema
description is limited to sources that cannot be resolved because of upstream
defects such as prose units, unresolvable templates, or context-dependent
inheritance.

The grammar-facing gap is recorded as
`vocab_gap:object:field_map_grid`. ISN's human-facing locus registry supplies
both `object` and `position` grammar segments according to locus type;
`field_map_grid` was proposed as an entity owner, so `object` is the applicable
closed segment. The live classifier established that `object` is valid, has
105 registered tokens, and classifies `field_map_grid` as `absent`. The literal
segment `locus` is not a compositional grammar segment and correctly classified
as `invalid_segment`.

## Before and after

The pre-write source census addressed exact declared `StandardNameSource.id`
values. All three requested identities returned exactly one row
(`exact_source_count=1`, `exact_id_count=1` for each), while the schema sanity
scan returned 9,668 candidates, 9,668 populated `id` values, and 9,668 populated
`status` values. The source figures therefore come from declared, populated
keys rather than missing-property zeros.

| DD source | Before status | Before produced name | After status | After produced name |
|---|---|---|---|---|
| `b_field_non_axisymmetric/time_slice/field_map/grid/r` | `extracted` | none | `vocab_gap` | none |
| `b_field_non_axisymmetric/time_slice/field_map/grid/z` | `extracted` | none | `vocab_gap` | none |
| `b_field_non_axisymmetric/time_slice/field_map/grid/phi` | `composed` | `toroidal_angle_of_measurement_position` | `vocab_gap` | none |

The post-write exact controls again returned one source, one populated `id`,
and one populated `status` for each requested row. Every source now carries:

- `skip_reason=settled_vocabulary_refusal`;
- the same policy-grounded refusal detail;
- one `HAS_STANDARD_NAME_VOCAB_GAP` link to
  `vocab_gap:object:field_map_grid`;
- no `produced_sn_id` scalar and no `PRODUCED_NAME` relationship.

The normalized gap has editorial disposition `reject`, no canonical target,
actor `Simon McIntosh`, and grammar version `0.8.0rc67`. It has three source
links, three DD-node links, and three independent `VocabGapEvidence`
observations. The recorded reason is:

> Settled vocabulary refusal: field_map_grid is a DD discretization structure,
> not a physics-meaningful locus; the closed ISN vocabulary deliberately
> excludes grid, mesh, GGD, and field-map owners. The radial_coordinate,
> vertical_coordinate, and toroidal_coordinate base semantics remain in DD
> provenance.

## Released incorrect binding and name-integrity proof

The sanctioned semantic-detach path removed phi's
`toroidal_angle_of_measurement_position` realization from both source and DD
projections, rewound the source, and wrote the required audit ledger before the
source was parked at `vocab_gap`. The ledger event is
`sn-change:7a62057e-b16b-4368-93b6-04593fc92ee3`, operation
`detach_inconsistent_attachment`; its reason records that the DD carries a
toroidal coordinate of a field-map discretization grid, not a measurement
position.

| Integrity counter | Before | After | Delta | Interpretation |
|---|---:|---:|---:|---|
| `StandardNameChange` nodes | 8,596 | 8,597 | +1 | Exactly the governed detach audit event; not a create, rename, or acceptance event |
| `PRODUCED_NAME` relationships | 5,310 | 5,309 | -1 | Exactly phi's incorrect binding was released |

Exact post-write checks found zero DD projections from any of the three paths
to either the rejected measurement-position identity or any intended
field-map-grid identity. They also found zero `StandardName` nodes for all
three would-be spellings:

- `radial_coordinate_of_field_map_grid`;
- `vertical_coordinate_of_field_map_grid`;
- `toroidal_coordinate_of_field_map_grid`.

Thus no `StandardName` was created, renamed, or accepted. The only name-change
ledger delta is the sanctioned removal of the one semantically wrong source
binding.

## Sanctioned write paths

No hand-written Cypher mutation was used. The graph changes went through the
repository's governed operations:

1. `detach_one_attachment` for the physics-judgement detach and audit ledger;
2. `write_vocab_gaps` for normalized gap and per-source observations;
3. `write_skipped_sources` with the schema-declared `vocab_gap` status for the
   three exact source identities;
4. `apply_vocab_gap_adjudications` with typed disposition `reject` for the
   settled editorial refusal.

The operation wrote one normalized rejected gap, three source dispositions,
three source links, three DD links, three evidence observations, and released
one incorrect binding. It wrote zero vocabulary tokens and zero Standard Names.
