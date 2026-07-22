---
name: sn/generate_name_dd
description: Dynamic user prompt for SN composition — per-batch DD paths with enriched context
used_by: imas_codex.standard_names.workers.compose_worker
task: composition
dynamic: true
schema_needs: []
---

Name the **physical or geometric quantities** represented by the following
IMAS Data Dictionary paths. Each standard name describes the underlying
physics — NOT the DD path, IDS section, or measurement instrument.
When multiple DD paths represent the same quantity (e.g. across IDSs), they
share ONE standard name.

**Subject–base separation rule:** Species and entity qualifiers (electron,
ion, neutral, fast_ion) go in the `subject` segment, NOT fused into
`physical_base`. Example: subject=electron + physical_base=temperature →
`electron_temperature` — but `temperature` is the base, not
`electron_temperature` as a monolithic token.

{% if retry_reason %}
## ⚠️ Retry Context

{{ retry_reason }}
{% endif %}

{% if domain_vocabulary %}
## PREFERRED VOCABULARY FOR THIS DOMAIN — reuse unless concept is genuinely different

The following standard names already exist in this physics domain and have been
validated. **Reuse** these terms and naming patterns unless the concept you are
naming is genuinely different. Synonymous proliferation within a domain is the
single most common quality failure.

{{ domain_vocabulary }}
{% endif %}

{% if reviewer_themes %}
## RECENT REVIEWER FEEDBACK FOR THIS DOMAIN — address these

Expert reviewers have flagged these recurring issues in this domain's standard names.
Pay special attention to avoiding these patterns:

{% for theme in reviewer_themes %}
- {{ theme }}
{% endfor %}
{% endif %}

{% include "sn/_compose_scored_examples.md" %}

## Unit Policy

The `unit` field for each path is pre-populated from the IMAS Data Dictionary
(`HAS_UNIT` relationship). It is **authoritative and final**:
- Do NOT include unit in your output — it will be injected from the DD at persistence time
- Use the provided unit to inform your naming (e.g., "eV" tells you this is an energy/temperature quantity)
- "dimensionless" means the quantity is genuinely unitless (e.g., safety factor, elongation)

## Common Anti-Patterns (AVOID these)

| ❌ Wrong | ✅ Correct | Why |
|----------|-----------|-----|
| `electron_temp` | `electron_temperature` | No abbreviations in `physical_base` |
| `langmuir_probe_electron_temperature` | `electron_temperature` | Method independence — measurement device is metadata, not name |
| `filtered_electron_density` | `electron_density` | Processing is metadata — `filtered_` is not a name segment |
| `core_electron_temperature` | `electron_temperature_core` | Position goes after physical_base per grammar |
| `Te` | `electron_temperature` | No symbol abbreviations |
| `electron_temperature_in_eV` | `electron_temperature` | Unit is never part of the name |
| `safety_factor_q` | `safety_factor` | No symbol suffixes |
| `plasma_current_IP` | `plasma_current` | No symbol suffixes |
| `current_from_passive_loop` | `passive_loop_current` | `_from_` implies causation — use device prefix for signals |
| `poloidal_flux` | `poloidal_magnetic_flux` | Use controlled vocabulary term; no synonymous short forms |
| `reconstructed_faraday_rotation_angle` | `faraday_rotation_angle` | Processing method is metadata, not part of the name |
| `geometric_minor_radius` | `minor_radius` | DD section prefix leaking into standard name |
| `flux_surface_averaged_elongation` | `elongation` | Elongation is a geometric property of a contour, not a flux-surface average |
| `energy_flux_at_wall_surface` | `energy_flux_at_wall` | Position token is `wall`, not `wall_surface` — the `_surface` suffix is redundant |
| `energy_due_to_recombination_at_ion_state` | `energy_due_to_recombination` | Process tokens are bare vocabulary entries — never append `_at_X` / `_in_X` / `_on_X` qualifiers |
| `energy_due_to_impurity_radiation_in_halo_region` | `halo_region_radiated_energy_due_to_impurity_radiation` | Region qualifiers go in the subject prefix, not after `due_to_<process>` |
| `vertical_coordinate_of_outline_point` | `vertical_outline` | **Enumeration is a coordinate, not a name** — outline vertices are an ordinal array of ONE geometry; collapse to `radial_outline` / `vertical_outline` (`base_token=outline`, axis as `coordinate` projection). The vertex index lives in the DD path; emit every vertex path in `dd_paths`. Never encode `outline_point` |
| `x_ray_crystal_spectrometer_pixel_photon_energy_lower_bound` | `lower_bound_photon_energy` | **Instrument-prefix carry-over** — drop the instrument when the leaf is a generic physics observable. Keep as `_of_<instrument>` ONLY when the quantity is intrinsic to the hardware (e.g. `cross_sectional_area_of_rogowski_coil`) |
| `x1_coordinate_of_neutron_detector_geometry_outline` | `first_local_tangential_coordinate_of_neutron_detector` | **Local tangent axes are DISTINCT semantic directions, not storage labels** — map source-local X1/X2 to the REGISTERED carriers `first_local_tangential_coordinate` / `second_local_tangential_coordinate` (`base_kind=geometry`), retain the intrinsic owning object, and never emit `x1_coordinate` / `x2_coordinate` |
| `vertical_front_surface_radius_of_optical_element` for reflector X2 curvature | `second_local_tangential_radius_of_reflector` | X2 is an object-local surface tangent, not machine vertical. Keep `reflector`, drop the redundant `front_surface` wording, omit diagnostic/channel provenance, and do not claim a principal-curvature direction without source evidence |
| `radial_wave_electric_field` for `e_field/normal` | `flux_surface_normal_wave_electric_field` | The DD parent explicitly defines a projection normal to the magnetic flux surface. `radial` is reserved for cylindrical $R$; `perpendicular` would incorrectly imply the magnetic-field-relative plane |
| `halo_region_parallel_energy_due_to_heat_flux` | `parallel_halo_energy` | **Suffix-form for component** — component / transformation / reducer tokens come BEFORE the base as a leading qualifier prefix. Compare ★0.95 `parallel_fast_electron_pressure` |
| `z_coordinate_of_sensor_direction_unit_vector` | `z_direction_unit_vector_of_camera` | **Compound hardware identifiers** — a component of a device's orientation/direction unit vector is a locus-qualified coordinate. Drop stacked intermediate hardware tokens but KEEP the owning device as the `_of_<device>` locus. A device direction unit vector is a machine-frame Cartesian vector (triple `x`, `y`, `z`), so a `z` leaf is the `z` axis. `vertical` is the cylindrical Z (`radial`, `toroidal`, `vertical`) — never mix frames. Locus-less `z_direction_unit_vector` collapses every device's orientation to one name |
| both `camera/direction/z` and `camera/up/z` → `z_direction_unit_vector` | distinct bases per vector | **Distinct vectors of one device get DISTINCT bases** — `direction` is the line-of-sight, `up` is the image-up vector: different physical vectors. Never name a non-pointing vector (image-up, ellipse axis) bare `direction`; name what the vector IS |

> **UNIT-VECTOR LOCUS IS MANDATORY — this OVERRIDES any semantic-cluster suggestion.** A `*_unit_vector` base (direction, image-up, minor/major axis, …) ALWAYS requires a per-device locus `_of_<device>`; a locus-less unit-vector name (e.g. `z_direction_unit_vector`) is **rejected by the grammar at error severity** and collapses every device's orientation onto one name. If a cross-IDS semantic cluster groups this leaf with other devices (EC launcher mirror, camera, sensor, SPI shatter cone, …) and its description suggests "one generic name across device types without a device-specific locus", **IGNORE that suggestion for unit vectors** — emit the name for THIS source's own device using its `DEV:`/path device as the `_of_<device>` locus (`x_direction_unit_vector_of_electron_cyclotron_launcher_mirror`, `x_direction_unit_vector_of_camera`, `x_direction_unit_vector_of_sensor`, `x_direction_unit_vector_of_shatter_cone`). Each device gets its OWN distinct name. Only a displacement/position vector carrying a length unit (`m`) is NOT a unit vector — name it with the appropriate coordinate/position base, still per-device. |

## Hardware & Diagnostic Geometry — Specificity Required

Many DD paths describe hardware geometry (coil cross-sections, detector outlines,
aperture positions). These CAN yield valid standard names, but ONLY when the name
is **tokamak-universal** — meaningful across different fusion devices.

**The rule:** geometry names must include enough context to identify WHAT hardware
component is being described. Generic geometric primitives alone are useless.

| ❌ Too generic (SKIP these) | ✅ Specific enough | Why |
|-----------------------------|-------------------|-----|
| `radius_of_annulus` | `inner_radius_of_poloidal_field_coil_cross_section` | Names which annular geometry |
| `alpha_of_oblique` | `oblique_angle_of_poloidal_field_coil_element` | Names the engineering context |
| `radius_of_circle` | SKIP — no tokamak-universal meaning | Pure geometric primitive |
| `height_of_rectangle` | `height_of_poloidal_field_coil_cross_section` | Rectangle alone is meaningless |
| `outline_r` | `radial_outline` | Outline vertices are an ordinal array of ONE geometry → collapse (`base_token=outline`, radial `coordinate` projection); the vertex index lives in the DD path |
| `first_point_r` | `radial_coordinate_of_line_of_sight` | **Enumeration is a coordinate, not a name** — `first_point`/`second_point`/`third_point` are endpoints of ONE line-of-sight; collapse ALL to `radial_coordinate_of_line_of_sight` (`base_token=coordinate`, radial `coordinate` projection, `locus_token=line_of_sight`, `locus_relation=of`). `line_of_sight` is a LOCUS, never a base. List every endpoint path in `dd_paths`. NEVER encode the ordinal (`first_point`) in the name |

**When to SKIP geometry paths entirely:**

- The DD path describes a generic geometric primitive with no physics or engineering context
  (e.g., `*/geometry/arcs_of_circle/radius` — "radius of arc of circle" is a math concept, not a tokamak quantity)
- The quantity is purely local to one specific machine design and has no cross-device meaning
- The path describes coordinate bookkeeping (index arrays, grid connectivity)

**When to NAME geometry paths:**

- The quantity describes a recognizable hardware component that exists across tokamaks
  (coil centroids, detector positions, antenna dimensions, wall coordinates)
- The name includes the hardware context: `_of_poloidal_field_coil`, `_of_detector_aperture`,
  `_of_antenna_strap`, `_of_first_wall`
- Multiple tokamaks would use the same term for the same concept

### Enumerated geometry points collapse to ONE name

**Enumeration is a coordinate, not a name.** When a DD leaf's **parent is an
ordinal point** — `first_point` / `second_point` / `third_point` / `point` /
`<entity>_point`, or an `outline` vertex array — the ordinal is an array index,
NOT part of the quantity's identity. Name by the **grandparent geometry
carrier** with the axis projection, and emit `dd_paths` covering **all** the
ordinal siblings so the multiple DD paths attach to the ONE name:

| DD leaves (ordinal siblings) | ✅ ONE name | Segments |
|---|---|---|
| `.../line_of_sight/first_point/r` + `.../second_point/r` + `.../third_point/r` | `radial_coordinate_of_line_of_sight` | `base_token=coordinate`, `base_kind=geometry`, `projection_axis=radial`, `projection_shape=coordinate`, `locus_token=line_of_sight`, `locus_relation=of`, `locus_type=geometry` |
| `.../line_of_sight/*_point/z` | `vertical_coordinate_of_line_of_sight` | …`projection_axis=vertical` |
| `.../line_of_sight/*_point/phi` | `toroidal_coordinate_of_line_of_sight` | …`projection_axis=toroidal` |
| `<entity>/outline/r` (vertex array) | `radial_outline` | `base_token=outline`, `base_kind=geometry`, `projection_axis=radial`, `projection_shape=coordinate` |
| `<entity>/outline/z` (vertex array) | `vertical_outline` | …`projection_axis=vertical` |

`dd_paths` for `radial_coordinate_of_line_of_sight` MUST list every
`.../*_point/r` endpoint — the collapse is realised by attaching every endpoint
path to the single name.

**Distinguish points only by physical ENTITY, never by ordinal.** A point earns
a distinct name ONLY when it is a distinct physical entity (an aperture vs a
wall) — name it by that entity:

| Entity-distinguished point | Segments |
|---|---|
| `radial_position_of_aperture` | `base_token=position`, `base_kind=geometry`, `projection_axis=radial`, `projection_shape=coordinate`, `locus_token=aperture`, `locus_relation=of`, `locus_type=entity` |
| `radial_position_of_first_wall` | …`locus_token=first_wall`, `locus_relation=of`, `locus_type=position` |

Never name such a point by its ordinal (no `first_point`/`second_point` in the name).

**Object-local X1/X2 axes are tangent directions, not ordinal points or machine
axes.** Map them to the REGISTERED semantic carriers
`first_local_tangential_coordinate` / `second_local_tangential_coordinate`
(`base_kind=geometry`) and keep each direction as its OWN name. Retain the
intrinsic owning object where it changes the quantity. Do NOT emit
`x1_coordinate` / `x2_coordinate`, call X2 `vertical`, or infer a principal
curvature direction from an opaque local-axis label.

## Non-Nameable Paths — route to `skipped` (do NOT compose)

Some DD paths are **coordinate or infrastructure bookkeeping**, not physics
observables. Composing a bare name for one of these fails the semantic gate and
then burns every refine rotation to exhaustion. Add the `source_id` to the
`skipped` list instead of emitting a candidate.

| ❌ Do NOT compose | DD path example | Why it is non-nameable |
|-------------------|-----------------|------------------------|
| `time` | `real_time_data/topic/time_stamp` | Time is the independent coordinate of a signal, not a named quantity |
| `initial_time_of_simulation` | `summary/simulation/time_begin` | Simulation start/stop timestamps are run metadata |
| `delay` | `bremsstrahlung_visible/latency` | Signal-chain latency is data-pipeline infrastructure, not plasma physics |
| `acquisition_period`, `dead_time` | diagnostic timing fields | Acquisition timing describes the instrument chain |
| `channel_index`, `element_count` | `*/index`, `*/count` | Counters and ordinals are array bookkeeping |
| version / comment / status strings | metadata leaves | Pure metadata, not a measurable quantity |

**Rule of thumb:** unit `s` + a description naming a timestamp / latency /
acquisition interval → `skipped`. A genuine physics time constant (confinement
time, decay time) is the rare nameable exception.

## Segment Routing — Common Confusions

The following tokens are frequently misrouted to the wrong grammar segment.
Study this table before composing — it eliminates the most common vocab-gap rejections.

| Token/Concept | ❌ Wrong Segment | ✅ Correct Segment | Correct Usage |
|---|---|---|---|
| `langmuir_probe`, `bolometer`, `interferometer` | position | device or object | Use as `_of_<device>` suffix |
| `r`, `major_radius_direction` | component | — | Use `radial` (closed component vocab) |
| `perpendicular`, `vertical`, `poloidal` | physical_base | component | These are direction tokens — use component segment |
| `unit_vector_*_component` | geometric_base | — | Decompose: component=`x`/`y`/`z` + geometric_base=`unit_vector` |
| `perturbed_*_field`, `electrostatic_potential` | process | physical_base | These are quantities, not mechanisms |
| `separatrix_average`, `flux_surface_average` (a `local/separatrix_average/...` path) | position + **transformation** | — | CONDITIONAL rule. Position=`plasma_boundary` ALWAYS (the boundary is one of a continuum of flux surfaces — never drop the locus). Transformation=`flux_surface_averaged` IFF the base is NOT constant on a flux surface: for surface-varying bases (densities, temperatures, velocities, momentum, parallel electric field) the prefix is REQUIRED — e.g. `flux_surface_averaged_<species>_density_at_plasma_boundary`, `toroidal_flux_surface_averaged_<species>_velocity_at_plasma_boundary` — because the bare `<species>_density_at_plasma_boundary` is the LOCAL (`local/separatrix/...`) value and would COLLIDE. For flux-function bases (safety factor, magnetic shear, flux labels psi/rho_tor, pressure — flagged `constant_on_flux_surface` in the grammar vocabulary) FSA is a no-op (FSA of an FSA) and the prefix is BANNED by the grammar gate: the local and averaged leaves share ONE name, `<q>_at_plasma_boundary` (two DD sources, one StandardName). `separatrix` is a deprecated synonym; use `plasma_boundary`. |
| `separatrix` (bare, in a `local/separatrix/...` path — NO `_average`) | position | — | Use position=`plasma_boundary`, NO transformation (this is the local LCFS value). The bare separatrix IS the plasma boundary / LCFS; reserve `secondary_separatrix` etc. for genuinely distinct qualified surfaces. |
| `measurement_position` | position (as token) | — | Already exists in position vocab — use it correctly as a locus |
| `derivative_with_respect_to_*` | operators | transformation | Use transformation segment for derivatives |
| `diffusion_coefficient`, `convection_velocity` | process | physical_base | Transport coefficients are quantities, not processes |
| `parallel_viscosity`, `heat_viscosity` | process | physical_base | Viscosity is a quantity — process would be `viscous_diffusion` |

### Process vs Physical_base Decision Rule

The `process` segment is for mechanisms that MODIFY a quantity — they appear via `_due_to_<process>`.

- **Process (via `due_to_`):** conduction, convection, diffusion, neoclassical, turbulent, ohmic, radiation, recombination
- **Physical_base (the quantity itself):** temperature, pressure, flux, field, potential, coefficient, viscosity, diffusivity

**Test:** Can you say "X due_to Y"? If Y is a mechanism causing X, then Y is a process.
If Y is itself measurable, it's a physical_base.

### Loci come from the injected registry — a locus is not a zone

The registered locus tokens, the relations each allows (`_at_`/`_of_`/`_over_`/
along), and any finer VARIANTS of one feature are defined by the injected
grammar vocabulary — do not invent locus tokens or enumerate them from memory.
When the registry offers more than one locus for the same feature — a locus for
a field's *value* there vs one for its steepest-gradient point or the
coordinate/flux that *locates* it, or a sampled point vs a distribution/peak
over a surface vs a distinct contact/tangency point — pick the variant matching
what the source measures, using the relation that variant allows. Never fold a
locus into a zone prefix or force a bare feature token where the registry
defines a more specific locus.

✅ `energy_due_to_recombination` — recombination is a mechanism → process
✅ `current_due_to_ohmic` — ohmic heating is a mechanism → process  
❌ `energy_due_to_diffusion_coefficient` — a coefficient is not a mechanism
❌ `temperature_due_to_magnetic_field` — magnetic field is a quantity, not a mechanism

## Description Quality Rules

- **No storage-shape tags** — NEVER write "1D", "2D", "3D", "scalar", "array",
  "profile", "time-dependent" in descriptions. The description defines the
  *physics*, not the data layout. Say "radial distribution of electron temperature"
  NOT "1D profile of electron temperature".
- **American English only** — use "center" not "centre", "meter" not "metre",
  "ionized" not "ionised", "behavior" not "behaviour", "polarization" not
  "polarisation". ISN catalog follows American spelling exclusively.
- **Physics-first framing** — describe what the quantity IS physically, not what
  the DD path stores or how it is measured.

## Batch Consistency Check

Before finalizing your output, verify:
1. **No synonymous names** — if you used `magnetic_flux` in one entry, don't use just `flux` in another
2. **Consistent suffixes** — all boundary quantities use `_of_plasma_boundary`, not a mix of patterns
3. **No DD leakage** — none of your names start with an IDS or DD section name
4. **No storage-shape tags** — none of your descriptions mention "1D", "2D", "3D", "profile", "array"
5. **American spelling** — check for "centre", "metre", "behaviour" and correct to American

## IDS Context

{% if ids_contexts %}
{% for ids_ctx in ids_contexts %}
### {{ ids_ctx.ids_name }}
{{ ids_ctx.ids_description }}
{% if ids_ctx.ids_documentation %}*{{ ids_ctx.ids_documentation }}*{% endif %}
{% if ids_ctx.top_sections %}
**Top-level sections:**
{% for sec in ids_ctx.top_sections %}
- `{{ sec.name }}` ({{ sec.data_type }}): {{ sec.description or 'no description' }}
{% endfor %}
{% endif %}
{% endfor %}
{% else %}
**Batch:** {{ ids_name }}
{% endif %}
{% if cluster_context %}
{{ cluster_context }}
{% endif %}

{% if existing_names %}
## Existing Standard Names (reuse when applicable)

These names already exist. **Reuse** them when the DD path measures the same
quantity — do not create a duplicate with different wording.

{% for name in existing_names %}
- {{ name }}
{% endfor %}
{% endif %}

{% if nearby_existing_names %}
## Nearby Existing Standard Names

These names already exist in the catalog. Reuse them if they match your source, or avoid creating duplicates:
{% for name in nearby_existing_names %}
- **{{ name.id }}**: {{ name.description | default('', true) }} ({{ name.kind | default('scalar', true) }}, {{ name.unit | default('dimensionless', true) }})
{% endfor %}
{% endif %}

## DD Paths to Name

{% for item in items %}
### {{ item.path }}
{% if item.rate_hint %}
> ⚠️ **HARD CONSTRAINT — RATE QUANTITY:** The DD documentation for this path
> indicates a rate / time-derivative quantity (phrases like "instantaneous
> change", "signed change", "rate of change", "time derivative",
> "per unit time"). Your name MUST begin with `tendency_of_` (preferred),
> `change_in_`, or `rate_of_change_of_`. NEVER use `instant_change_*` or
> `instantaneous_change_*` as a prefix. The description MUST be consistent
> with the rate-marker prefix (e.g. if the name is `tendency_of_X`, the
> description should read "Instantaneous signed change in X" or
> "Time derivative of X"). Do NOT produce a base-quantity name (e.g.
> `electron_density`) and then describe it as a rate — that is a critical
> drift error that quarantines the entry.
>
> **CRITICAL — rate + component ordering:** If the quantity is a rate of a
> vector component, the orientation token (`parallel`, `perpendicular`,
> `poloidal`, `toroidal`, `radial`, `diamagnetic`) MUST be placed OUTSIDE
> the rate marker, wrapping the rate phrase:
>   ✅ `parallel_change_in_fast_electron_pressure`
>   ✅ `poloidal_tendency_of_electron_velocity`
>   ❌ `change_in_parallel_fast_electron_pressure` (grammar rejects)
>   ❌ `change_in_poloidal_electron_velocity` (grammar rejects)
> The grammar parses `{orientation}_X` as a unit — the rate
> marker must modify the base quantity X, not intrude between orientation
> and the base.
{% endif %}
{% if item.value_provenance %}
> ⚠️ **HARD CONSTRAINT — {{ item.value_provenance | upper }} ESTIMATOR:** this
> path is the `{{ item.value_provenance }}` estimate of the quantity at
> `{{ item.provenance_base_path }}` (the description above is that base
> quantity). Name the **underlying physical quantity ONLY** — do NOT encode
> `{{ item.value_provenance }}`, `measured`, `reconstructed`, `reference`,
> `target`, `constraint`, or `fit` in the name. The measured / reconstructed /
> reference estimates of one quantity share ONE standard name; the estimator is
> recorded as link metadata, never in the name.
> **Also drop the FIT-CONSTRAINT framing:** the grounding mentions a "position"
> / "constraint position" / "at various positions" — that is the fit's sampling
> locus, NOT a physical locus. Do NOT add `_at_constraint_position`,
> `_at_measurement_position`, `_at_sensor_attachment_point`, or any
> position/sensor locus from the constraint substructure. (A surface/flux-average
> that the raw doc states as part of the quantity — e.g. flux-surface-averaged
> current density — IS kept; the sampling position is not.)
>   ✅ `plasma_current`   ❌ `measured_plasma_current`   ❌ `plasma_current_constraint`
>   ✅ `poloidal_magnetic_field`   ❌ `poloidal_magnetic_field_at_constraint_position`
{% endif %}
{% if item.species_context %}- **⚠️ Species context:** `{{ item.species_context }}` — this quantity is specific to **{{ item.species_context }}** species. The standard name MUST include the species in the `subject` segment (e.g., `{{ item.species_context }}_temperature`, not just `temperature`).
{% endif %}- **Description:** {{ item.description }}
{% if item.documentation and item.documentation != item.description %}- **Source documentation:** {{ item.documentation }}{% endif %}
- **Unit:** {{ item.unit or 'dimensionless' }} *(authoritative from DD — use for naming context only, do NOT output)*
- **Data type:** {{ item.data_type or 'unspecified' }}
{% if item.node_type %}- **Node type:** {{ item.node_type }} *(dynamic=time-varying quantity; static=machine-fixed parameter, e.g. wall geometry; constant=single scalar value; none=unclassified — use other context)*{% endif %}
{% if item.physics_domain %}- **Physics domain:** {{ item.physics_domain }}{% endif %}
{% if item.ndim is not none %}- **Dimensions:** {{ item.ndim }}D{% endif %}
{% if item.lifecycle_status %}- **Lifecycle:** {{ item.lifecycle_status }} ⚠️{% endif %}
{% if item.keywords %}- **Keywords:** {{ item.keywords | join(', ') if item.keywords is iterable and item.keywords is not string else item.keywords }}{% endif %}
{% if item.coordinate_paths %}- **Coordinates:** {{ item.coordinate_paths | join(', ') }}{% endif %}
{% if item.timebase %}- **Timebase:** {{ item.timebase }}{% endif %}
{% if item.cocos_label %}- **COCOS transformation type:** `{{ item.cocos_label }}`{% if item.cocos_expression %} — expression: `{{ item.cocos_expression }}`{% endif %}
{% if cocos_version is defined and cocos_version %}- **COCOS convention:** {{ cocos_version }}{% if dd_version %} (DD {{ dd_version }}){% endif %}{% endif %}
{% if item.cocos_guidance %}- **Sign convention guidance:** {{ item.cocos_guidance }}{% endif %}
  ⚠️ This quantity is COCOS-dependent. You MUST include a sign convention paragraph
  in the documentation section of the form:
  `Sign convention: Positive when <concrete physical condition consistent with COCOS {{ cocos_version | default('') }}>.`
  Write a CONCRETE plain-English condition (e.g. "the current flows counter-clockwise
  viewed from above"). If you cannot supply a concrete condition, omit that paragraph
  entirely and write `This quantity has no sign ambiguity.` instead.{% endif %}
{% if item.identifier_schema %}- **Identifier schema:** {{ item.identifier_schema }}{% if item.identifier_schema_doc %} — {{ item.identifier_schema_doc }}{% endif %}{% endif %}
{% if item.identifier_values %}
- **Identifier enum values:**
{% for iv in item.identifier_values %}  - `{{ iv.name }}` ({{ iv.index }}): {{ iv.description | default('', true) }}
{% endfor %}{% endif %}
{% if item.coord_path %}- **Coordinate:** {{ item.coord_path }}{% if item.coord_unit %} ({{ item.coord_unit }}){% endif %}{% endif %}
{% if item.parent_path %}- **Parent structure:** {{ item.parent_path }} ({{ item.parent_type or 'STRUCTURE' }}){% endif %}
{% if item.parent_description %}- **Parent description:** {{ item.parent_description }}{% endif %}
{% if item.ancestor_context %}- **DD path lineage** (ancestor nodes — the quantity's meaning and its evaluation locus live here; use them to resolve the correct locus, e.g. `pedestal_top` vs the bare path segment `pedestal`):
{% for anc in item.ancestor_context %}  - `{{ anc.path }}`: {{ anc.text }}
{% endfor %}{% endif %}
{% if item.clusters %}
- **Semantic clusters:**
{% for cl in item.clusters %}  - **{{ cl.label }}** ({{ cl.scope }}): {{ cl.description }}
    Members: {{ cl.members | join(', ') }}
{% endfor %}{% endif %}
{% if item.cross_ids_paths %}
- **Cross-IDS equivalents:** These paths in other IDSs represent the same quantity:
{% for xp in item.cross_ids_paths %}  - `{{ xp }}`
{% endfor %}  → Generate ONE name that covers all cross-IDS instances.
{% endif %}
{% if item.dd_paths_docs %}
- **Member DD documentation** (the sibling leaves this name must cover — ground every qualifier so the name fits all of them, not just the primary path above):
{% for mpath, mdoc in item.dd_paths_docs.items() %}  - `{{ mpath }}`: {{ mdoc }}
{% endfor %}{% endif %}
{% if item.hybrid_neighbours %}
- **Hybrid-search neighbours** (physics-concept + structural cousins):
{% for n in item.hybrid_neighbours %}  - `{{ n.tag }}` [{{ n.unit }}, {{ n.physics_domain }}]: {{ n.doc_short }}{% if n.cocos_label %} (COCOS {{ n.cocos_label }}){% endif %}
{% endfor %}  → Reuse a `name:` entry above when your source measures the same quantity.
{% endif %}
{% if item.related_neighbours %}
- **Graph-relationship neighbours** (explicit cross-IDS peers):
{% for r in item.related_neighbours %}  - `{{ r.path }}` ({{ r.ids }}) — {{ r.relationship_type }}{% if r.via %} via {{ r.via }}{% endif %}
{% endfor %}  → These paths share structural relationships (cluster, coordinate, unit, identifier, COCOS) with this path.
{% endif %}
{% if item.error_fields %}
- **DD error companions:**
{% for ef in item.error_fields %}  - `{{ ef }}`
{% endfor %}  → Error/uncertainty companions are minted deterministically — do NOT produce `*_uncertainty` variants. Skip this path entirely if it IS an error field (`_error_upper`, `_error_lower`, `_error_index`).
{% endif %}
{% if item.version_history %}
- **Version history:**
{% for vh in item.version_history %}  - {{ vh.version }}: {{ vh.change_type }}{% if vh.description %} — {{ vh.description }}{% endif %}
{% endfor %}{% endif %}
{% if item.sibling_fields %}
- **Sibling fields** (same parent structure — use for documentation cross-references):
{% for sib in item.sibling_fields %}  - `{{ sib.path }}`: {{ sib.description or 'no description' }} ({{ sib.data_type or '?' }})
{% endfor %}{% endif %}
{% if item.previous_name %}
- **⟳ Previous generation:** `{{ item.previous_name.name }}` ({{ item.previous_name.name_stage or 'drafted' }}{% if item.previous_name.reviewer_score %}, score={{ item.previous_name.reviewer_score | round(2) }}{% endif %}{% if item.previous_name.review_tier %}, {{ item.previous_name.review_tier }}{% endif %})
{% if item.previous_name.description %}- **Prior description:** {{ item.previous_name.description }}{% endif %}
{% if item.previous_name.documentation %}- **Prior documentation:** {{ item.previous_name.documentation }}{% endif %}
{% if item.previous_name.links %}- **Prior links:** {{ item.previous_name.links | join(', ') if item.previous_name.links is iterable and item.previous_name.links is not string else item.previous_name.links }}{% endif %}
{% if item.previous_name.validation_issues %}- **⚠️ Validation issues from prior run:** {{ item.previous_name.validation_issues | join('; ') if item.previous_name.validation_issues is iterable and item.previous_name.validation_issues is not string else item.previous_name.validation_issues }}{% endif %}
{% if item.previous_name.linked_dd_paths %}- **Other DD paths sharing this name:** These paths were also mapped to `{{ item.previous_name.name }}` — your generated name should be appropriate for all of them:
{% for ldp in item.previous_name.linked_dd_paths %}  - `{{ ldp }}`
{% endfor %}{% endif %}
{% if item.previous_name.name_stage == 'accepted' %}- **⚠️ This name was human-accepted** — only replace with a clearly better alternative.{% endif %}
{% endif %}
{% if item.review_feedback %}
- **📝 Prior reviewer feedback — you MUST address the issues below in your new name:**
  - **Previous name:** `{{ item.review_feedback.previous_name }}`{% if item.review_feedback.reviewer_score is not none %} (score={{ item.review_feedback.reviewer_score | round(2) }}{% if item.review_feedback.review_tier %}, tier={{ item.review_feedback.review_tier }}{% endif %}){% endif %}
{% if item.review_feedback.previous_description %}  - **Prior description:** {{ item.review_feedback.previous_description }}
{% endif %}{% if item.review_feedback.previous_documentation %}  - **Prior documentation:** {{ item.review_feedback.previous_documentation | replace('\n', '\n    ') }}
{% endif %}{% if item.review_feedback.reviewer_scores %}  - **Rubric scores (out of 20 each):**
{% for dim, dim_score in item.review_feedback.reviewer_scores.items() %}{% if dim not in ('score', 'tier') and dim_score is number %}    - `{{ dim }}`: {{ dim_score }}
{% endif %}{% endfor %}{% endif %}{% if item.review_feedback.reviewer_comments %}  - **Reviewer critique:**
    {{ item.review_feedback.reviewer_comments | replace('\n', '\n    ') }}
{% endif %}{% if item.review_feedback.reviewer_suggested_name %}  - **Reviewer's suggested replacement:** `{{ item.review_feedback.reviewer_suggested_name }}`{% if item.review_feedback.reviewer_suggestion_justification %}
    - *Justification:* {{ item.review_feedback.reviewer_suggestion_justification | replace('\n', ' ') }}{% endif %}
    - Use this suggestion as your **starting point**. You may refine it (e.g. fix grammar slot order, swap synonyms) but do not regress to the previous name.
{% endif %}  - **Instruction:** Produce a name that directly fixes every concrete issue raised above. Do NOT re-emit the previous name unchanged. If the reviewer flagged excessive length, redundant qualifiers, or convention violations, your new name must be shorter / cleaner / more idiomatic. If the reviewer was satisfied with a dimension (score ≥ 15), preserve that aspect.
{% endif %}{% if item.review_feedback and item.review_feedback.name_hint and item.review_feedback.edit_reason %}
- **🧭 Expert steering ({{ item.review_feedback.edit_origin or "human" }}):** A domain expert has proposed this naming direction: "{{ item.review_feedback.name_hint }}" — for this reason: {{ item.review_feedback.edit_reason }}
  - This proposal is subordinate to the grammar and composition rules above — realize the intent within the rules; if the rules forbid the literal proposal, compose the nearest rule-compliant name. Do not treat it as pre-approved.
{% endif %}
{% if item.reviewer_history %}
- **📜 Full reviewer history — HARD RULE: if a prior reviewer raised an issue, your new name MUST address it. Do not regenerate with the same flaw.**
  - **Latest review** (model={{ item.reviewer_history.latest.model }}, score={{ "%.2f"|format(item.reviewer_history.latest.score) if item.reviewer_history.latest.score is not none else 'N/A' }}):
    {{ item.reviewer_history.latest.comment | replace('\n', '\n    ') }}
{% if item.reviewer_history.prior_themes %}
  - **Recurring concerns across earlier reviews:**
{% for theme in item.reviewer_history.prior_themes %}    - ({{ theme.count }}×) {{ theme.theme }} — e.g. "{{ theme.example }}"
{% endfor %}{% endif %}
{% endif %}
{% if item.cluster_siblings %}- **Cross-IDS siblings:**
{% for sib in item.cluster_siblings[:5] %}  - {{ sib.path }} ({{ sib.unit or '?' }})
{% endfor %}
- **Concept identity:** These {{ item.cluster_siblings|length + 1 }} cross-IDS paths represent the SAME physics concept. Generate ONE name that covers all of them.
{% endif %}
{% if item.family_type %}
- **🔗 Family context ({{ item.family_type }}):**
{% if item.family_type == "physical_vector" %}  - This path is the **{{ item.family_axis }}** component of a vector quantity.
  - **Sibling components:** {% for sib in item.family_siblings %}`{{ sib }}`{% if not loop.last %}, {% endif %}{% endfor %}

  - **ISN naming convention:** Each component should be named `{axis}_{parent}` where `{parent}` is the shared vector name (e.g., `toroidal_current_density`). The `_component_of_` connector is REJECTED by the grammar — use the short leading-qualifier form.
  - All siblings MUST share the same `physical_base`, the same locus, and the same physics_domain — only the `component` (axis) segment differs.
  - **A machine-frame vector uses exactly ONE frame's triple:** Cartesian is `x`, `y`, `z`; cylindrical is `radial`, `toroidal`, `vertical`. A `z` leaf of a Cartesian vector (e.g. a device direction unit vector) is the `z` axis; a `z` leaf of a cylindrical vector is `vertical`. Never mix `z` with `radial`/`toroidal`, nor `vertical` into an `x`, `y`, `z` triple.
  - **Device orientation / direction unit-vector components KEEP the owning-object locus** (`_of_<device>`). ✓ `z_direction_unit_vector_of_camera`; ✗ locus-less `z_direction_unit_vector` (collapses every device's orientation to one name). Use the registered device token as the locus, and make sure the locus matches the DD path's device.
  - **Distinct vectors of one device get DISTINCT base carriers.** A device node may expose several vectors (`camera/direction` = line-of-sight, `camera/up` = image-up). Never fold a non-pointing vector into `direction_unit_vector` — name what the vector IS.
{% elif item.family_type == "geometric_coordinate" %}  - This path is the **{{ item.family_axis }}** coordinate of a geometric position.
  - **Sibling coordinates:** {% for sib in item.family_siblings %}`{{ sib }}`{% if not loop.last %}, {% endif %}{% endfor %}

{% if item.family_parent_name %}  - **Geometric base:** `{{ item.family_parent_name }}`{% endif %}

  - **ISN naming convention:** Geometric coordinates use `{axis}_{geometric_base}` form (e.g., `radial_position`, `vertical_position`, `toroidal_angle`). Do NOT use `component_of` or `coordinate_of` connectors for coordinates.
  - **Enumeration is a coordinate, not a name.** If the geometric position is an **ordinal point** (`first_point`/`second_point`/`third_point`/`point`/`<entity>_point`) or an `outline` vertex, the ordinal is an array index — COLLAPSE to the grandparent geometry carrier + axis: line-of-sight endpoints → `radial_coordinate_of_line_of_sight` / `vertical_coordinate_of_line_of_sight` (`base_token=coordinate`, `locus_token=line_of_sight`, `locus_relation=of` — `line_of_sight` is a LOCUS, not a base); outline vertices → `radial_outline` / `vertical_outline` (`base_token=outline`). Emit `dd_paths` covering EVERY ordinal sibling so they share the ONE name. Never put `first_point`/`second_point`/`outline_point` in the name; distinguish a point only by physical entity (`radial_position_of_aperture`, `radial_position_of_first_wall`), never by ordinal.
  - **Object-local X1/X2 axes are DISTINCT tangent directions, not storage-shaped name tokens** — use `first_local_tangential_coordinate` / `second_local_tangential_coordinate`, retain the intrinsic owning object, and never emit `x1_coordinate` / `x2_coordinate` or reinterpret X2 as machine `vertical`.
  - Note: these siblings may have DIFFERENT units (e.g., metres vs radians) — this is expected for geometric coordinates.
{% elif item.family_type == "derivative" %}  - This path is a **derivative** quantity.
  - **Sibling derivatives:** {% for sib in item.family_siblings %}`{{ sib }}`{% if not loop.last %}, {% endif %}{% endfor %}

  - **ISN naming convention:** a derivative is an OPERATOR applied to the numerator quantity X — never a coined compound base. Common cases use a dedicated operator with `operator_kind="unary_prefix"` and `base_token`=X: time derivative → `operator_token="time_derivative"`; radial derivative → `operator_token="radial_derivative"`; spatial gradient → `operator_token="gradient"`. For a derivative with respect to a flux coordinate Y (e.g. dV/dψ, dp/dψ), if no operator yet expresses it, emit a `vocab_gap` (segment `operator`) — do NOT coin `derivative_of_X_with_respect_to_Y` as a `base_token`.
  - All siblings sharing the same denominator should use the consistent operator + base form.
{% endif %}{% endif %}

{% endfor %}

## Description — MANDATORY

For **every** candidate, include a `description` field: a single-line
plain-English summary of the physical quantity (≤120 characters). This
description is used for embedding, search, and human review. Examples:

- `"Electron temperature measured in the plasma core"`
- `"Toroidal component of the magnetic field"`
- `"Safety factor at the 95% poloidal flux surface"`

Do NOT repeat the standard name verbatim — add context that helps a reader
understand what the quantity represents physically.

## IR Segment Fields — MANDATORY

**You do NOT output a `standard_name` string.** You fill individual IR segment
fields. Code assembles the canonical name via ISN's `compose()` function.

For **every** candidate you emit, populate the IR segment fields inside a `segments` object.
This is not optional — it is how downstream tooling assembles and validates the name.

**Required fields inside `segments`:**
- `base_token`: the irreducible base quantity from the registry (e.g., `"temperature"`, `"magnetic_field"`)
- `base_kind`: `"quantity"` or `"geometry"`

**Optional segment fields** (omit or set null when not applicable):
- `projection_axis` + `projection_shape`: for vector/coordinate projections
- `qualifiers`: list of species/population/modifier tokens
- `locus_token` + `locus_relation` + `locus_type`: for postfix locus
- `process_token`: for `_due_to_` mechanism
- `operator_token` + `operator_kind`: for mathematical operators

**Examples:**

- `electron_temperature` →
  `segments: {base_token: "temperature", base_kind: "quantity", qualifiers: ["electron"]}`
- `radial_magnetic_field` →
  `segments: {base_token: "magnetic_field", base_kind: "quantity", projection_axis: "radial", projection_shape: "component"}`
- `minor_radius_of_plasma_boundary` →
  `segments: {base_token: "minor_radius", base_kind: "geometry", locus_token: "plasma_boundary", locus_relation: "of", locus_type: "geometry"}`
- `time_derivative_of_electron_density` →
  `segments: {base_token: "density", base_kind: "quantity", qualifiers: ["electron"], operator_token: "time_derivative", operator_kind: "unary_prefix"}`

If you cannot decompose into valid IR segments, the concept is wrong — revise
rather than emit empty segments.

## Vocabulary Gaps

If a path requires a token that does **not** exist in a closed grammar segment
(e.g., a new `subject` species, a new `position`), do NOT invent an invalid name.
Instead, add the path to the `vocab_gaps` list in your response with:
- `source_id`: the DD path
- `segment`: which grammar segment is missing a token
- `token`: the token value you would need
- `reason`: why this token is needed

**⚠️ CRITICAL: Most vocab gaps are false positives.** Before emitting:
1. Search the token in ALL segment registries — it may exist in another segment
2. For compound tokens, check if each part exists as a registered token — decompose instead
3. Verify no existing token already covers the concept

**But a genuine missing base IS a real gap — emit it, do not guess.** When the
irreducible quantity has no registered `physical_base`/`geometric_base`, emit a
`vocab_gap` for that segment rather than substituting a near-synonym base or
fusing the concept into another token:
- A device **angle** (shatter angle, beam tilt angle) when no angle base is
  registered → `vocab_gap` (`segment: geometric_base`); never invent `tilt`.
- A **phase shift** of a probing wave (`refractometer/.../phase`) when
  `phase`/`phase_shift` is unregistered → `vocab_gap`, not a bare `wave_phase`.
- A **mode/perturbation phase** with an unregistered qualifier/base →
  `vocab_gap`, not a guessed compound.
- A characteristic **length/extent** when no length base is registered → reuse
  an accepted sibling (e.g. `extent_of_pellet`) via `attachments`, else
  `vocab_gap`.

A clean compose-time `vocab_gap` is cheap; a guessed near-base churns through
review and every refine rotation to exhaustion.
