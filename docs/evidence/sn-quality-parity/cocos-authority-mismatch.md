# COCOS authority mismatch

Live evidence snapshot: `2026-08-25T08:05:25Z` against the default `codex`
graph. Source checkout: `ac21fcb22ed9`.

## Outcome

The eighteen residual sign-gate failures do **not** share one cause.

| Partition | Count | Stored authority | Recomputed authority | Cause | Repair route |
|---|---:|---|---|---|---|
| Stored agrees with recomputed | 17 | `one_like` | `one_like` | Documentation contradicts valid invariant metadata by carrying one final sign-convention paragraph. | Governed documentation edit: remove only that paragraph, retain the COCOS metadata, and pass the edited document through the ordinary docs review path. |
| Stored disagrees with recomputed | 1 | `one_like` | `b0_like` | `magnetic_field` carries a stale persisted projection that is no longer synchronized with its structural children. | Recompute followed by a governed data repair: change only the transformation class to `b0_like`, retain COCOS 17 and its `HAS_COCOS` edge, then postflight the structural authority and sign gate. Its current documentation needs no edit. |
| **Total** | **18** |  |  |  | **17 + 1 = 18** |

The live repair simulation passed both routes: removal of the sign paragraph
made **17/17** agreement-partition documents pass the sign gate, while scoring
the unchanged `magnetic_field` document against recomputed `b0_like` returned
**PASS**, “COCOS-sensitive quantity has canonical sign-convention prose.”

## Measurement and authority rule

The live measurement selected accepted names with accepted, non-null
documentation and ran the production `score_documentation()` sign gate against
each stored `cocos_transformation_type`. It evaluated 2,262 documentation
bodies and reproduced exactly 18 sign failures.

For each failure, Cypher read both supported DD bindings:

- `(IMASNode)-[:HAS_STANDARD_NAME]->(StandardName)`;
- `(StandardNameSource)-[:PRODUCED_NAME]->(StandardName)` together with
  `(StandardNameSource)-[:FROM_DD_PATH]->(IMASNode)`.

When a name had DD bindings, DD was authoritative: one distinct non-null DD
class was used directly, and a bound cohort with no non-null class recomputed
to `one_like` under the established documentation-holdout contract. Only names
without DD bindings used structural recomputation. Structural children in a
terminal state were removed, binary operands were excluded, and normalized
children were excluded from a non-normalized parent; one remaining distinct
non-null child class became the recomputed class, otherwise authority was
unavailable.

The property-presence preflight returned nonzero candidates and nonzero
`cocos_transformation_type` coverage before the result was trusted. This
avoids treating a misspelled Cypher property as a real empty result.

## Live `magnetic_field` result

The authority query returned the following current facts:

| Name | Description | Stored | Recomputed | Basis | DD paths | Structural children | Eligible children | Stored gate | Recomputed gate |
|---|---|---|---|---|---:|---:|---:|---|---|
| `magnetic_field` | Local magnetic induction vector before any source, state, spatial projection, magnitude, or locus is selected. | `one_like` | `b0_like` | structural | 0 | 13 | 12 | FAIL | PASS |

The eligible children have exactly one non-null transformation class:
`vacuum_magnetic_field` is `b0_like`. Eleven other eligible children have no
class. `ratio_of_ion_velocity_to_magnetic_field` is `one_like`, but its
`operator_kind='binary'` makes it an operand relation rather than inheritance
authority, so it is correctly excluded. The recomputation is therefore
unambiguous: `b0_like`.

The node also stores `cocos=17` and has one `HAS_COCOS` edge to the `COCOS`
node whose id is 17. It has no `HAS_STRUCTURAL_AUTHORITY` receipt. Thus the
structural class is presently a recomputed fact, whereas the contradictory
scalar is a durable property with no durable dependency or invalidation link.

Representative review evidence is consistent with a metadata-only repair:
`magnetic_field` has name score 0.8875 and docs score 0.9125. The existing final
sign paragraph is canonical for `b0_like`; changing only the scoring context
from stored `one_like` to recomputed `b0_like` passes.

## Why the stored value diverged

The last timestamped write path that set the surviving scalar is
`persist_generated_name_batch()` → `write_standard_names()` at
`2026-07-11T06:06:19.749Z`. The node records that time in `generated_at`, the
writer model as `openrouter/deepseek/deepseek-v4-flash`, and the writer source
revision as `d33a09efa`. At that revision, `persist_generated_name_batch()`
stamped `generated_at` on the candidate and called `write_standard_names()`;
the same `UNWIND` write assigned
`sn.cocos_transformation_type = coalesce(b.cocos_transformation_type,
sn.cocos_transformation_type)` and persisted the timestamp. The incoming or
preserved value was `one_like`.

An earlier structural materializer invocation is visible through
`derived:magnetic_field`, whose `created_at` and `composed_at` are both
`2026-07-09T08:25:24.906Z`. That historical materializer also assigned the
parent transformation scalar from then-visible children. The graph does not
retain an edge-creation time or the candidate payload, so it cannot prove
whether `vacuum_magnetic_field`'s `HAS_PARENT` edge was already visible at that
earlier instant, nor whether the July 11 batch changed or merely reasserted the
value. This does not weaken identification of the last timestamped scalar
writer, but it prevents a stronger claim about which invocation first made the
value wrong.

The durable cause is an authority synchronization defect:

1. transformation class was copied onto the parent as a mutable scalar;
2. no field-level `updated_at`, write receipt, child-set signature, or
   `StructuralNameAuthority` node records what child topology justified it;
3. later topology and metadata writes can evolve independently of that scalar;
4. `magnetic_field` is now `origin='catalog_edit'`, while both current
   structural materializer candidate queries admit only `origin IS NULL` or
   `origin='derived'`, so the sanctioned recompute path cannot self-correct it;
5. the docs edit requested at `2026-08-25T02:41:00.038542Z` and accepted at
   `2026-08-25T02:46:13.425Z` changed documentation, not authority metadata.

The current materializer itself has the right overwrite semantics—it assigns
the recomputed scalar directly and clears stale COCOS metadata when authority
is unavailable. The remaining mismatch is routing and governance: a protected
catalog-edited parent is outside that automatic writer even when its structural
projection is stale.

## The other seventeen failures

All seventeen names are DD-backed. Each DD cohort contains no non-null DD
transformation class, so each recomputes to the same `one_like` value already
stored on the Standard Name. Every document has exactly one sign-convention
paragraph and that paragraph is last. This is a documentation cohort, not an
authority-mismatch cohort.

| Standard Name | Bound DD paths | Stored | Recomputed |
|---|---:|---|---|
| `change_in_ion_state_mean_ionisation_potential` | 1 | `one_like` | `one_like` |
| `effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | 7 | `one_like` | `one_like` |
| `ion_average_temperature` | 4 | `one_like` | `one_like` |
| `parallel_current_density_due_to_ohmic_current_drive` | 1 | `one_like` | `one_like` |
| `poloidal_ion_velocity_at_measurement_position` | 1 | `one_like` | `one_like` |
| `toroidal_helium_3_velocity_at_plasma_boundary` | 2 | `one_like` | `one_like` |
| `x_direction_unit_vector_of_electron_cyclotron_launcher_mirror` | 1 | `one_like` | `one_like` |
| `x_minor_axis_unit_vector_of_shatter_cone` | 1 | `one_like` | `one_like` |
| `x_unit_vector_of_pellet_injector` | 1 | `one_like` | `one_like` |
| `y_direction_unit_vector_of_electron_cyclotron_launcher_mirror` | 1 | `one_like` | `one_like` |
| `y_direction_unit_vector_of_shatter_cone` | 1 | `one_like` | `one_like` |
| `y_minor_axis_unit_vector_of_shatter_cone` | 1 | `one_like` | `one_like` |
| `z_direction_unit_vector_of_camera` | 1 | `one_like` | `one_like` |
| `z_direction_unit_vector_of_electron_cyclotron_launcher_mirror` | 1 | `one_like` | `one_like` |
| `z_direction_unit_vector_of_pellet_injector` | 1 | `one_like` | `one_like` |
| `z_major_axis_unit_vector_of_shatter_cone` | 1 | `one_like` | `one_like` |
| `z_minor_axis_unit_vector_of_shatter_cone` | 1 | `one_like` | `one_like` |

Representative source bindings make the distinction concrete:

- `ion_average_temperature` means “Density-weighted mean thermal energy of all
  ion species and charge states...” and binds four DD paths, including
  `core_profiles/profiles_1d/t_i_average`; its name score is 0.93125 and docs
  score is 0.875.
- `effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` means
  the vector contribution to an internal-state neutral fluid velocity from
  diamagnetic-drift transport and binds seven DD paths, including
  `edge_profiles/ggd/neutral/state/velocity_exb/diamagnetic`; its docs score is
  0.9875.

For this partition, governed documentation edits should preserve every name,
description, source binding, review history, `one_like` scalar, COCOS 17 scalar,
and COCOS edge. The signed edit manifest should bind the exact pre-edit
documentation hash and remove only the final sign paragraph. The live in-memory
simulation confirms that this exact change passes **17/17** sign gates.

## Repair ordering and postflight

1. Run a fresh authority recomputation for `magnetic_field` and sign an exact
   one-row metadata manifest. Because the name is catalog-edited, use a governed
   compare-and-set repair rather than widening the automatic derived-parent
   materializer. Change `one_like` to `b0_like`; preserve COCOS 17 and verify the
   existing `HAS_COCOS` target remains 17.
2. Postflight the child closure and require the recomputed class to remain
   `b0_like`. Score the unchanged document and require sign-gate PASS.
3. Independently sign the 17-row documentation cohort against exact current
   document hashes. Remove only each final sign paragraph, send the resulting
   documents through the ordinary docs review route, and require 17/17
   `one_like` sign-gate PASS with unchanged authority and source bindings.
4. Re-run the whole accepted-document sign gate and require 0 residual failures.

This investigation performed no graph write, provider call, documentation
mutation, or acceptance transition.
