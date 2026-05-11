## Standard Name Grammar Reference

### Core Axioms

1. **One concept, one name** — every physical concept maps to exactly one canonical string.
2. **Liberal parser, strict generator** — the parser accepts legacy and colloquial forms with diagnostics; the canonical rendering is unique.
3. **Postfix locus** — spatial qualifiers (`_of_`, `_at_`, `_over_`) and mechanism (`_due_to_`) always follow the base quantity.
4. **Prefix projection** — axis projections (`radial_component_of_`) precede the base.
5. **Explicit operator scope** — prefix operators carry `_of_` as a scope marker (`gradient_of_X`); postfix operators concatenate directly (`X_magnitude`).
6. **Fixed vocabularies** — every segment has a defined token list (see registry below). Use only listed tokens; if none fits, emit `vocab_gap`.

### ⚠️ DECOMPOSITION RULE — compound physical_base is the #1 error

**Before emitting a compound `physical_base`, check if any leading tokens are
registered qualifiers or subjects. If yes, SPLIT them into their segment slots.**

The most common systematic error is cramming multiple grammar tokens into
`physical_base` as a single compound token. The `physical_base` slot must
contain ONLY an irreducible dimensional quantity from the closed registry.

| ❌ WRONG (compound physical_base) | ✅ CORRECT (decomposed) | Why |
|-------------------------------------|--------------------------|-----|
| `momentum_diffusivity` | qualifier=`momentum` + base=`diffusivity` | `momentum` is a registered qualifier |
| `convection_velocity` | qualifier=`convection` + base=`velocity` | `convection` is a registered qualifier |
| `energy_convection_velocity` | qualifier=`energy` + qualifier=`convection` + base=`velocity` | both `energy` and `convection` are qualifiers |
| `thermal_pressure` | subject=`thermal` + base=`pressure` | `thermal` is a registered subject |
| `state_torque_density` | qualifier=`state` + base=`torque_density` | `state` is a registered qualifier; `torque_density` is a registered base |

**Rule:** Walk the candidate `physical_base` token left-to-right. For each
underscore-delimited prefix, check the qualifier and subject registries.
If the prefix is a registered qualifier → move it to the `qualifier` slot.
If the prefix is a registered subject → move it to the `subject` slot.
The remainder (which must be a registered base) stays in `physical_base`.

### 5-Group Internal Representation

Every standard name decomposes into five groups:

| Group | Role | Example tokens |
|-------|------|----------------|
| **operators** | Math ops, applied outer→inner | prefix: `time_derivative`, `gradient`, `normalized`, `per_toroidal_mode`, `per_poloidal_mode`, `cumulative_inside_flux_surface`; postfix: `magnitude`, `real_part`, `fourier_coefficient` |
| **projection** | Axis decomposition of vector/tensor | `radial_component_of_`, `toroidal_component_of_`, `parallel_component_of_` |
| **qualifiers** | Species or population prefix | `electron`, `ion`, `deuterium`, `fast_ion`, `thermal_electron` |
| **base** | Irreducible physical quantity (80 registered tokens) | `temperature`, `pressure`, `density`, `magnetic_field`, `safety_factor` |
| **locus + mechanism** | Where (postfix) + process (postfix) | `_of_plasma_boundary`, `_at_magnetic_axis`, `_over_core_region`, `_due_to_bootstrap` |

### Canonical Rendering

```
[operators] [projection_component_of_] [qualifiers] base [_of/_at/_over locus] [_due_to process]
```

### Segment Composition Order

Names are assembled from segments in this fixed order:

```
[operator_of_] [component_component_of_] [qualifier]* [subject_] physical_base [_at_position | _of_object | _over_region] [_due_to_process]
```

| Segment | Required? | Cardinality | Source |
|---------|-----------|-------------|--------|
| `operator` | optional | 0..N (nested) | operators registry |
| `component` | optional | 0..1 | component/coordinate registry |
| `qualifier` | optional | 0..N (ordered) | qualifier registry (108 tokens) |
| `subject` | optional | 0..1 | subject registry |
| `physical_base` | **REQUIRED** | exactly 1 | base registry (80 tokens) |
| `geometric_base` | alternative to physical_base | exactly 1 | geometric_base registry |
| `position` | optional | 0..1 via `_at_` | geometry/position registry |
| `object` | optional | 0..1 via `_of_` | device/object registry |
| `region` | optional | 0..1 via `_over_` | region registry |
| `process` | optional | 0..1 via `_due_to_` | process registry |

**Qualifier stacking:** Multiple qualifiers combine left-to-right before the base:
`trapped_fast_particle_collisional_power_density` = qualifiers=[trapped, fast_particle, collisional] + base=power_density

**Operator templates:**
- **Unary prefix**: `op_of_` + inner → `time_derivative_of_electron_temperature`
- **Unary postfix**: inner + `_op` → `electron_temperature_magnitude`, `perturbed_magnetic_field_real_part`
- **Binary**: `op_of_` + A + `_and_`/`_to_` + B → `ratio_of_electron_to_ion_temperature`

### `_of_` Disambiguation

In the ISN grammar, `_of_` appears in exactly three structural roles:

| Role | Template | Disambiguator |
|------|----------|---------------|
| Prefix operator scope | `op_of_` + inner | Longest-match against operator registry |
| Binary operator | `op_of_` A `_and_`/`_to_` B | `_and_`/`_to_` keyword present |
| Locus (entity/geometry) | base + `_of_` + locus | Always the **last** `_of_` in the name |

### Key Rules

- **Use only registered tokens** — every segment has a fixed token list. When no registered token fits, report as `vocab_gap`. Never invent tokens.
- **Generic bases require qualification** — tokens like `temperature`, `current`, `pressure`, `density` must have at least one qualifier (species, component, or locus).
- **Process attribution** uses `_due_to_` + a Process vocabulary noun: `plasma_current_due_to_bootstrap`, never `bootstrap_current`.
- **DD path independence** — names describe physics, not DD location. Never include IDS names or DD section prefixes.
- **No processing verbs** — `reconstructed_`, `measured_`, `fitted_` are provenance, not physics. Drop them.
- **Preposition discipline** — use `_of_` for properties of named entities, `_at_` for field values at points, `_over_` for region integrals. Never use `_from_`.
- **Spectral decomposition** — `per_toroidal_mode` and `per_poloidal_mode` are registered **unary prefix** operators. Canonical form: `per_toroidal_mode_of_X`. They indicate the quantity is resolved per Fourier toroidal/poloidal mode component. Do not penalise these as unknown operators.

### Rejected rc20 Forms

| ❌ rc20 form (now invalid) | ✅ ISN canonical | Reason |
|----------------------------|--------------------|----|
| `real_part_of_X` | `X_real_part` | Postfix operator, not prefix |
| `amplitude_of_X` | `X_amplitude` | Postfix operator, not prefix |
| `imaginary_part_of_X` | `X_imaginary_part` | Postfix operator, not prefix |
| Registered token absorbed into base | `<segment>_token_<rest>` decomposition | Place every registered token in its segment |
| `volume_averaged_X` (bare concat) | `volume_averaged_of_X` | Operator scope requires `_of_` |
| `electron_thermal_pressure` | `thermal_electron_pressure` | Population precedes species |
| `ion_rotation_frequency_toroidal` | `toroidal_component_of_ion_rotation_frequency` | No trailing component |
| `diamagnetic_component_of_X` | `X_due_to_diamagnetic_drift` | Diamagnetic is a drift, not an axis |

{% if closed_vocab_full %}
### Token Registry

Use only tokens from these lists. If no token fits, emit a `vocab_gap`.
Each token belongs in its declared segment — decompose compound candidates
so every recognised token occupies its segment slot.

Examples:

- `toroidal_torque` → `toroidal` is in `component`; decompose to
  `component=toroidal, physical_base=torque` → `toroidal_component_of_torque`.
- `volume_averaged_electron_temperature` → `volume_averaged` is a
  `transformation`; `electron` is a `subject` → `volume_averaged_of_electron_temperature`.
- `parallel_viscosity_current_density` → `parallel` is a `component` →
  `parallel_component_of_viscosity_current_density`.

{% for vs in closed_vocab_full %}
#### `{{ vs.segment }}`{% if vs.aliases %} (alias{{ "es" if vs.aliases|length > 1 else "" }}: {{ vs.aliases | join(', ') }}){% endif %} — {{ vs.tokens | length }} tokens

```
{{ vs.tokens | join(', ') }}
```

{% endfor %}

### Decomposition Checklist — apply BEFORE you commit a name

For every candidate name, run these checks IN ORDER. If any fires, restructure
before emitting:

1. **Tokenise the candidate on `_`.**  Walk the resulting tokens left-to-right.
2. **For each token (and 2-token / 3-token compound), look it up in the
   registry above.**  If the token appears in any segment, it MUST occupy
   that segment slot in `grammar_fields`, not be absorbed into `physical_base`.
3. **Whitelist genuine atomic compounds** that share a prefix with a
   registered token but are NOT decomposable: `poloidal_flux`, `minor_radius`,
   `cross_sectional_area`, `safety_factor`, `polarization_angle`,
   `ellipticity_angle`, `loop_voltage`, `internal_inductance`. These are
   single, lexicalised physics terms.
4. **If a registered token is present but no atomic compound rule exempts
   it**, restructure:
   - `<component>_<base>` → `<component>_component_of_<base>` (or place
     `<component>` in the `component` slot of `grammar_fields`).
   - `<subject>_<base>` → keep `<subject>` in the `subject` slot; never let
     it leak into `physical_base`.
   - `<transformation>_<base>` → `<transformation>_of_<base>` with the
     transformation in its own slot.
   - `<base>_<process>` → `<base>_due_to_<process>` with the process in
     its own slot.
   - `<base>_<region>` → `<base>_over_<region>` with the region in its own slot.
5. **Re-render the name** from the corrected `grammar_fields` and confirm the
   `physical_base` slot contains ONLY one of the registered base tokens.
   If the intended base is not in the registry, emit a `vocab_gap` for `physical_base`.
6. **If a needed token is missing from ANY registry**, emit a `vocab_gap`
   against that segment — DO NOT invent tokens.

This checklist addresses the dominant failure mode: registered tokens
(toroidal, parallel, thermal, e_cross_b_drift, normalized, fast_ion, …)
absorbed into `physical_base` instead of placed in their correct grammar slot.

### Top Absorption Failures — Concrete Examples

These are the MOST FREQUENTLY absorbed tokens (observed in 73% of reviewed
names). Study each example — if you see any of these patterns in your
output, restructure immediately:

| ❌ Wrong (absorbed into physical_base) | ✅ Correct (decomposed) | Absorbed Token → Correct Segment |
|-----------------------------------------|-------------------------|----------------------------------|
| `toroidal_angle_of_position` | `toroidal_angle` (coord=toroidal, geom_base=angle) | `toroidal` → coordinate |
| `parallel_current_density` | `parallel_component_of_current_density` | `parallel` → component |
| `toroidal_torque` | `toroidal_component_of_torque` | `toroidal` → component |
| `radial_electric_field` | `radial_component_of_electric_field` | `radial` → component |
| `beam_position_variation` | report as `vocab_gap` (variation is not a registered token) | context-dependent |
| `thermal_electron_energy` | subject=`thermal_electron`, base=`energy` | `thermal_electron` → subject |
| `normalized_poloidal_flux` | `normalized_of_poloidal_magnetic_flux` | `normalized` → transformation |
{% endif %}
