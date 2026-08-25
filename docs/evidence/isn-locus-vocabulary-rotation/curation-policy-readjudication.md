# Field-map-grid locus curation-policy re-adjudication

## Recommendation

Choose the second of the two live options: **make the three DD coordinate paths
a settled permanent refusal**. The discretization structure belongs in DD
provenance, not in a Standard Name identity. Do not admit `field_map_grid` to
the locus registry.

This recommendation changes only the owner/locus disposition. The already
settled bases remain:

| DD path | Identity that would have been spelled if the owner were admissible |
|---|---|
| `b_field_non_axisymmetric/time_slice/field_map/grid/r` | `radial_coordinate_of_field_map_grid` |
| `b_field_non_axisymmetric/time_slice/field_map/grid/z` | `vertical_coordinate_of_field_map_grid` |
| `b_field_non_axisymmetric/time_slice/field_map/grid/phi` | `toroidal_coordinate_of_field_map_grid` |

The refusal is not a claim that the DD leaves lack meaning. They are the R, Z,
and phi coordinate arrays of a three-dimensional field-map grid. It is a claim
about the Standard Name boundary: the grid is the storage and discretization
framework that owns those arrays in the DD, not a physics locus that should
enter the identity.

This work was read-only with respect to both systems. It made no ISN vocabulary
change, dependency-pin change, graph mutation, or new `StandardName`.

## Governing policy, applied verbatim

The source registry contains four curation bullets that are absent from the
runtime `get_grammar_context()` payload. Applied to `field_map_grid`, they give
one satisfied bullet and three violated bullets:

| Registry policy bullet | Result for `field_map_grid` | Reason |
|---|---|---|
| “Tokens must describe physics-meaningful loci, not DD structural artifacts” | **Violates** | The DD calls this owner a three-dimensional grid. It is the discrete support of stored field-map arrays, not a material object, diagnostic, plant object, or physical evaluation location. |
| “Ordinal point labels (first_point, second_point) belong in DD metadata” | **Satisfies** | `field_map_grid` contains no ordinal point label and does not try to promote sample ordering into identity. |
| “Discretization framework tokens (ggd_*) are not physics positions” | **Violates in mechanism** | Although the proposed spelling does not start with `ggd_`, it denotes the same excluded kind of thing: a discretization framework. Changing the framework's spelling from GGD to a field-map grid does not turn it into a physics position. |
| “Prefer atomic tokens composable via qualifiers over long compounds” | **Violates** | `field_map_grid` is a three-part compound that bakes the represented quantity, representation, and discretization container into one locus token. It is not an atomic physics location that qualifiers specialize. |

The first adjudication correctly established that no existing locus is a
semantic substitute. It nevertheless treated “entity owner” as sufficient for
admission without reading these curation constraints. Under the registry's own
policy, ownership in the DD is necessary evidence for provenance but is not
sufficient evidence for a Standard Name locus.

## Entity precedent measured from the registry

The current registry contains **181 total tokens**, of which **105 are typed
`entity`**. Reading every entity definition, only **2 of 105** explicitly denote
a non-material abstraction rather than hardware, a plant object, a diagnostic
object, a material object, or a physical location:

| Entity token | Registry definition | What kind of abstraction it is |
|---|---|---|
| `coordinate_system` | “The reference frame and coordinate basis used to express geometrical or vector quantities.” | A reference frame and basis. It is representational, but it defines how physical quantities are expressed rather than how sampled values are discretized or stored. |
| `ion_state` | “A governed ion configuration distinguished by its charge state and, where specified, internal quantum state.” | A governed physical-state configuration. It classifies the state of an ion rather than a numerical sampling framework. |

The census uses the definitions as the classification authority. In particular,
tokens with generic definitions that explicitly call their referent a distinct
physical, plant, or diagnostic object were not silently reclassified from their
names alone. That rule keeps apparently abstract-looking names such as
`beam_tracing_ray`, `gyrokinetic_eigenmode`, and
`neoclassical_tearing_mode` out of the abstraction count because their registry
definitions explicitly place them in the physical-object class.

This is a small abstraction class, not a single precedent, but neither member
is a discretization precedent. `coordinate_system` is the strongest analogy and
still stops at the reference frame; `ion_state` is a governed physical state.
The class therefore supports the general proposition that an `entity` need not
be hardware, but does not support admitting a grid or mesh against the explicit
discretization rule.

The live census corrects the **160 total / 89 entity** figures recorded in the
reopened decision rationale. The installed package and the ISN checkout both
currently contain **181 total / 105 entity** tokens and have identical registry
SHA-256 content. The recommendation does not depend on the stale smaller
denominator: the expanded registry still contains no discretization token.

## Grid, mesh, GGD, and field-map precedent

An exact, case-normalized scan over all 181 **token keys** found:

| Key concept | Matching locus tokens |
|---|---:|
| `grid` | **0** |
| `mesh` | **0** |
| `ggd` | **0** |
| `field_map` | **0** |
| Union of all four scans | **0** |

The instrument was aimed at token keys, not arbitrary YAML text. Two positive
controls distinguish that question from a broken text search: the file itself
contains `ggd_*` in the curation-policy comment, and the `camera` definition
contains the word “maps”; neither is a locus token key and neither appears in
the key result. The known token `coordinate_system` was also retrieved with its
`entity` type and `of` relation in the same parse.

The complete absence matters. Across a registry broad enough to contain 181
loci and 105 entity owners, there is no grid, mesh, GGD structure, or field-map
owner. Read together with the explicit rule excluding discretization framework
tokens, zero is evidence of a deliberate ontology boundary, not merely an
accidental missing synonym.

## The two live options

1. **Admit `field_map_grid` despite the policy.** This would make all three
   coordinate identities strictly spellable and preserve the DD ownership
   relation directly in each name. It would require treating a discrete support
   as a special entity-locus exception.
2. **Make the three coordinate paths a settled permanent refusal.** Their
   radial, vertical, and toroidal coordinate semantics remain in DD provenance,
   while the Standard Name layer declines to turn the owning discretization
   container into identity.

Recommend option 2. The strongest argument against it is that the three paths
form a coherent family, have no valid substitute locus, and lose a concise
cross-DD identity that could distinguish their coordinates from other radial,
vertical, and toroidal coordinates. The registry already permits at least two
non-material entity owners, so an entity is plainly not restricted to hardware.

That argument loses because it proves usefulness, not admissibility. Neither
abstract entity is a grid-like storage structure, the curation policy expressly
places discretization outside physics positions, and the zero-token census
shows that this boundary has been maintained across the entire registry. A
one-off exception would weaken the closed vocabulary by making DD containment a
sufficient identity criterion. Keeping the three rows as explicit refusals
preserves their exact DD provenance without asserting that a computational grid
is a physics locus.

## Reproducible evidence

The named registry census completed successfully against
`imas-standard-names` package version `0.8.0rc67`. It verified **181 total**,
**105 entity**, **2 explicit non-material abstractions**, and **0 token-key
matches** across `grid`, `mesh`, `ggd`, and `field_map`. The installed registry
and `/home/ITER/mcintos/Code/imas-standard-names/imas_standard_names/grammar/vocabularies/locus_registry.yml`
had the same SHA-256 digest:
`d54a8d8ac38d59de1f85b64dc8d2e55465cd113e3070e90b5ed34964ee6d303b`.

Raw result:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T173738947501-n-locusreadj/logs/registry-curation-census.json`
