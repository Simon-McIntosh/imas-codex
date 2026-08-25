# Field-map-grid locus adjudication

## Recommendation

**Disposition: warranted.** Add `field_map_grid` to the closed ISN locus
registry as an entity admitting the `of` relation, then follow the separately
authorized vocabulary-rotation path. The measured cohort contains **3 DD
coordinate paths**, including **2 beyond**
`dd:b_field_non_axisymmetric/time_slice/field_map/grid/phi`. The current
runtime grammar contains **181 locus tokens**, but no token containing `grid`
or `map` and no existing token with the required semantics.

This adjudication is read-only. It makes no ISN change, dependency-pin change,
or graph mutation.

## Is the field-map grid a locus?

It is not a material point, diagnostic, or measurement position. It is a
computational sampling structure. Nevertheless, the locus framing is correct
for ISN because the locus registry also represents entity owners: the DD
declares `field_map` as the “Description of the error field map”, its `grid`
child as a “3D grid”, and `r`, `z`, and `phi` as coordinate leaves owned by
that grid. The corresponding magnetic-field quantities are three-dimensional
arrays under the same `field_map` structure. The grid is therefore the
discrete spatial support on which the map is represented, and the identities
name coordinates **of that support**, not measurements at a physical site.

The intended identities are:

| DD path | DD description | Intended identity |
|---|---|---|
| `b_field_non_axisymmetric/time_slice/field_map/grid/r` | Major radius | `major_radius_of_field_map_grid` |
| `b_field_non_axisymmetric/time_slice/field_map/grid/z` | Height | `vertical_coordinate_of_field_map_grid` |
| `b_field_non_axisymmetric/time_slice/field_map/grid/phi` | Toroidal angle, counter-clockwise when viewed from above | `toroidal_coordinate_of_field_map_grid` |

`field_map_grid` is preferable to bare `grid`, which would discard what the
support belongs to, and to `magnetic_field_map_grid`, which would over-specialize
the reusable representational concept beyond the DD owner spelling. The
registry meaning should be equivalent to “the discrete spatial grid on which
a field map is represented.” Classifying it as an `entity` does not claim it
is a material device; it lets `of_field_map_grid` express the DD ownership
relationship without misclassifying the coordinates as measurement
positions.

## How many DD paths require it?

The live graph measurement selected coordinate children of every DD parent
whose path ends in `/field_map/grid`:

```cypher
MATCH (n:IMASNode)-[:HAS_PARENT]->(p:IMASNode)
WHERE p.id ENDS WITH '/field_map/grid'
  AND n.node_category = 'coordinate'
RETURN count(n) AS family_count,
       count(n.id) AS with_id,
       collect(n.id) AS paths,
       sum(CASE WHEN n.id = $target THEN 0 ELSE 1 END) AS beyond_target
```

With `$target = 'b_field_non_axisymmetric/time_slice/field_map/grid/phi'`, the
result was **family_count = 3**, **with_id = 3**, and **beyond_target = 2**.
The three paths are exactly the `r`, `z`, and `phi` rows listed above. The
broader parent check found two matching grid structures:

| Parent | Immediate children | Coordinate children |
|---|---:|---:|
| `b_field_non_axisymmetric/time_slice/field_map/grid` | 12 | **3** |
| `tf/field_map/grid` | 4 | **0** |

The `tf` structure carries a general grid description (`identifier`, `path`,
`space`, and `grid_subset`), not additional coordinate leaves needing these
identities. Thus the token resolves a real three-coordinate family without
inflating the count with structural, metadata, identifier, representation, or
error rows.

The schema sanity query was:

```cypher
MATCH (n:IMASNode)
RETURN count(n) AS candidates,
       count(n.id) AS with_id,
       count(n.node_category) AS with_node_category
```

It returned **61,366 candidates, 61,366 with `id`, and 61,366 with
`node_category`**. An exact positive control for the motivating `phi` path
returned **1 row with both fields populated**. The count is therefore aimed
at the declared `IMASNode.id` and `IMASNode.node_category` properties, not a
plausible zero produced by a missing key.

The related live source rows are also a closed set of three: `r` and `z` are
`extracted` with no produced name, while `phi` is `composed` to
`toroidal_angle_of_measurement_position`. That current binding is not a
near-equivalent: it asserts a measurement position where the DD declares a
coordinate axis of a stored three-dimensional field-map representation.

## Existing-locus near misses

The candidate sweep came directly from
`get_grammar_context()["grammar"]["vocabularies"]["locus_registry"]` in the
installed `imas-standard-names` **0.8.0rc67** package. It selected every token
containing any of `grid`, `map`, `field`, `coordinate`, or `measurement`. The
runtime registry had **181 tokens**; exact `field_map_grid` was absent, as were
all tokens containing `grid` or `map`. The resulting ten near misses all fail:

| Runtime locus token | Why it is not equivalent |
|---|---|
| `coordinate_system` | Defines the reference frame and basis used to express quantities; it is not the discrete support carrying the field samples. |
| `measurement_position` | Names an observed spatial location; these DD rows are computational coordinate axes and do not describe a measurement event or diagnostic site. |
| `magnetic_field_probe` | Names a diagnostic sensor, not a representation grid. |
| `poloidal_magnetic_field_probe` | Names the poloidal-component probe, not sampled support. |
| `radial_magnetic_field_probe` | Names the radial-component probe, not sampled support. |
| `toroidal_magnetic_field_probe` | Names the toroidal-component probe, not sampled support. |
| `poloidal_field_coil` | Names a physical field coil, not a computational map grid. |
| `toroidal_field_coil` | Names a physical field coil, not a computational map grid. |
| `pedestal_top_high_field_side` | Names a plasma pedestal position on the high-field side; neither a grid nor the owner of these coordinates. |
| `pedestal_top_low_field_side` | Names a plasma pedestal position on the low-field side; neither a grid nor the owner of these coordinates. |

The public strict parser independently rejected all three intended identities:

| Identity | `parse(..., strict=True)` |
|---|---|
| `major_radius_of_field_map_grid` | Rejected; no matching physical base or geometry carrier and no nearest candidate. |
| `vertical_coordinate_of_field_map_grid` | Rejected; no matching physical base or geometry carrier and no nearest candidate. |
| `toroidal_coordinate_of_field_map_grid` | Rejected; nearest candidates were `toroidal_flux_coordinate_gradient` and `toroidal_flux_coordinate`. |

The toroidal-flux candidates locate a magnetic-flux coordinate or its
gradient. They do not own the independent `r`, `z`, and `phi` axes of an error
field map, so substituting either would change the physics.

## Read-only proof and evidence

Immediately before and after the graph inspection, the live counters remained
**8,524 `StandardNameChange` nodes** and **5,315 `PRODUCED_NAME` edges**: both
deltas were zero. The graph reported current DD version **4.1.1** using numeric
`major`, `minor`, and `patch` fields; no release-candidate version was ordered
as a string.

Raw measurements:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T141822953635-n-locusadjudicate/logs/live-adjudication.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T141822953635-n-locusadjudicate/logs/field-map-family-count.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T141822953635-n-locusadjudicate/logs/field-map-coordinate-bindings.json`

The evidence supports the plan disposition **warranted**, not `not-a-locus`
and not `not-warranted`: the grid is a representational entity rather than a
measurement place, it owns a measured family of three DD coordinate paths,
and every runtime near miss denotes a different object or location.
