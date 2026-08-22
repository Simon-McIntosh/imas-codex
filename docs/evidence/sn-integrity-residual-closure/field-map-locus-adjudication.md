# Field-map grid locus adjudication

## Verdict

**Recommendation: `propose-token`.** Propose `field_map_grid` as a closed ISN
locus token. The measured impact is **3 current Data Dictionary coordinate
paths and 3 live `StandardNameSource` rows**. All three require the missing
locus to express their correct grid-owned identities.

The physics distinction is representational, not merely lexical. The Data
Dictionary declares one three-dimensional grid as the discretization support
of an error-field map. Its `r`, `z`, and `phi` arrays locate samples on that
grid. They are not measurement positions, and the registered
`coordinate_system` locus is not a substitute: a coordinate system defines a
reference frame and basis, whereas the field-map grid is the particular
discrete support on which the field is represented. No existing locus token
expresses that support.

This is proposal authority only. This invocation did not change the ISN
registry, compose or attach a name, or mutate the graph.

## Measured blocked cohort

The cohort is the complete set of current DD nodes whose path is an immediate
coordinate leaf below
`b_field_non_axisymmetric/time_slice/field_map/grid`. The parent documentation
identifies `field_map` as “Description of the error field map” and `grid` as
“3D grid”. A live source is a corresponding `StandardNameSource` row that is
present and not `stale`.

| # | Blocked DD path | DD meaning | Required identity | Live source state | Current binding |
|---:|---|---|---|---|---|
| 1 | `b_field_non_axisymmetric/time_slice/field_map/grid/r` | Major radius | `major_radius_of_field_map_grid` | `extracted` | none |
| 2 | `b_field_non_axisymmetric/time_slice/field_map/grid/z` | Height | `vertical_coordinate_of_field_map_grid` | `extracted` | none |
| 3 | `b_field_non_axisymmetric/time_slice/field_map/grid/phi` | Toroidal angle, counter-clockwise viewed from above | `toroidal_coordinate_of_field_map_grid` | `composed` | only `toroidal_angle_of_measurement_position` |

The partition closes exactly:

- **Blocked current DD paths: 3.** Every path is named in the table.
- **Blocked live `StandardNameSource` rows: 3.** Their status distribution is
  **2 `extracted` + 1 `composed`**.
- **Direct `HAS_STANDARD_NAME_VOCAB_GAP` source edges: 0.** The absence of a
  gap edge does not make the identities expressible. The two extracted rows
  have not produced a name, while the composed `phi` row retains a semantically
  wrong legacy target.
- **Correct bindings available now: 0 of 3.** The missing locus blocks the
  correct composition of all three rows. For `phi`, “blocked” means blocked
  from the correct grid-owned identity, not that the row lacks any historical
  composition.

The live graph's current DD version was `4.1.1`; the three source snapshots are
pinned to `4.1.0`. The current DD parent and leaf descriptions preserve the
same grid ownership used for this adjudication.

## Runtime grammar comparison

The comparison used the public `get_grammar_context()` result from the
installed and graph-active `imas-standard-names 0.8.0rc67` package. Its runtime
`locus_registry` contained **181 tokens**.

| Runtime comparison | Result | Consequence |
|---|---:|---|
| Exact `field_map_grid` token | absent | The candidate identities cannot pass the closed parser. |
| Registered locus tokens containing `grid` | 0 | There is no grid-bearing spelling to reuse. |
| Registered locus tokens containing `map` | 0 | There is no map-bearing spelling to reuse. |
| Registered `coordinate_system` | present | It denotes the reference frame and coordinate basis, not the discrete support of a field map. |
| Registered tokens containing `field` | 8 | They denote probes, coils, or high/low-field-side positions; none denotes a field-map representation. |

The eight runtime field-bearing loci were
`magnetic_field_probe`, `pedestal_top_high_field_side`,
`pedestal_top_low_field_side`, `poloidal_field_coil`,
`poloidal_magnetic_field_probe`, `radial_magnetic_field_probe`,
`toroidal_field_coil`, and `toroidal_magnetic_field_probe`. None is a semantic
near-substitute for a grid.

Strict public parsing recorded the expected rejection as the measurement:

| Candidate identity | Runtime result |
|---|---|
| `major_radius_of_field_map_grid` | Rejected: residue does not match a registered physical base or geometry carrier; no nearest candidate. |
| `vertical_coordinate_of_field_map_grid` | Rejected: residue does not match a registered physical base or geometry carrier; no nearest candidate. |
| `toroidal_coordinate_of_field_map_grid` | Rejected: residue does not match a registered physical base or geometry carrier; nearest candidates were `toroidal_flux_coordinate_gradient` and `toroidal_flux_coordinate`. |

Those flux-coordinate spellings are not substitutes: they locate magnetic flux
surfaces or gradients, while these arrays define the independent spatial
coordinates of a three-dimensional error-field-map discretization.

### Candidate spelling

`field_map_grid` is the preferred proposal spelling:

- it preserves both semantic parts of the DD ownership chain,
  `field_map/grid`;
- `grid` is the head noun and `field_map` identifies what the grid supports;
- bare `grid` would be underspecified;
- `coordinate_system` would conflate the sampled support with its reference
  frame;
- `magnetic_field_map_grid` would over-specialize a representation concept
  that can be reused for other field maps.

The proposed registry semantics are an `entity` locus with relation `of` and a
definition equivalent to: “The discrete spatial grid on which a field map is
represented.” Treating the representational grid as an entity follows the
existing `coordinate_system` precedent without claiming that the grid is a
material device or a physical measurement location.

## Read-only counter proof

The successful graph invocation read both counters immediately before and
after the cohort inspection. Both deltas are zero.

| Protected graph counter | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameChange` | 7,873 | 7,873 | **0** |
| `PRODUCED_NAME` | 5,774 | 5,774 | **0** |

No graph write or provider call occurred. The evidence inputs are retained in:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T205539351358-n-fieldmapvocab/logs/live-graph-exploration.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T205539351358-n-fieldmapvocab/logs/runtime-locus-comparison.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T205539351358-n-fieldmapvocab/logs/field-map-keys-and-dd-version.json`

The expected parser rejection is part of the evidence. It is not a failed
measurement and does not authorize a nearest-token substitution.
