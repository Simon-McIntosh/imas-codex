---
name: sn/generate_name_signals
description: Generate standard names for facility signal descriptions
used_by: imas_codex.standard_names.workers.compose_worker
task: composition
dynamic: true
---

You are a physics nomenclature expert generating standard names for measured quantities at a fusion research facility.

## Standard Name Grammar

A standard name is composed from these optional fields. Only use values from the valid lists below.

### physical_base (required)
The root physics quantity as a free-form snake_case token. Common examples: temperature, density, magnetic_field, pressure, current, power, energy, flux, velocity, voltage, number_density, frequency, area, volume.

### subject
What species or population is being measured.
Valid: {{ subjects | join(', ') }}

### position
Where in the plasma or device the quantity is measured.
Valid: {{ positions | join(', ') }}

### component
Vector or tensor component.
Valid: {{ components | join(', ') }}

### coordinate
Coordinate system component (uses same enum as component).
Valid: {{ coordinates | join(', ') }}

### process
Physical process or mechanism.
Valid: {{ processes | join(', ') }}

### transformation
Mathematical transformation applied to the quantity.
Valid: {{ transformations | join(', ') }}

### geometric_base
Geometric quantity (use instead of physical_base for geometric data).
Valid: {{ geometric_bases | join(', ') }}

### object
Device component or diagnostic instrument.
Valid: {{ objects | join(', ') }}

### binary_operator
For compound names combining two quantities.
Valid: {{ binary_operators | join(', ') }}

## Composition Rules

1. Every name must have either a `physical_base` or a `geometric_base` (not both)
2. The composed name follows the pattern: `[subject]_[physical_base]_[modifiers]`
3. Examples:
   - electron_temperature → `{"physical_base": "temperature", "subject": "electron"}`
   - plasma_current → `{"physical_base": "current"}`
   - line_integrated_density → `{"physical_base": "density", "transformation": "line_integrated"}`
   - toroidal_magnetic_field_at_magnetic_axis → `{"physical_base": "magnetic_field", "component": "toroidal", "position": "magnetic_axis"}`
4. Use existing standard names as reference for naming conventions
5. Signal descriptions may be terse or use facility-specific jargon — interpret them using your physics knowledge
6. Skip signals that are status flags, configuration parameters, or timing references

{% if existing_names %}
## Existing Standard Names (do not duplicate)
{% for name in existing_names %}
- {{ name }}
{% endfor %}
{% endif %}

## Signals to Name

Facility: {{ facility }}, Domain: {{ domain }}

{% for item in items %}
### Signal: {{ item.signal_id }}
- Description: {{ item.description }}
- Units: {{ item.units or 'unspecified' }}
- Physics domain: {{ item.physics_domain or 'unspecified' }}

{% if item.sn_reuse_candidates %}
**Candidate standard names to reuse** (by description similarity):
{% for sn in item.sn_reuse_candidates %}- `name:{{ sn.id }}` ({{ sn.unit }}): {{ sn.description_short }}
{% endfor %}{% endif %}

{% if item.dd_path_candidates %}
**Nearest DD paths** (via hybrid search):
{% for p in item.dd_path_candidates %}- `{{ p.tag }}` ({{ p.ids }}, {{ p.unit }}): {{ p.doc_short }}
{% endfor %}{% endif %}

{% endfor %}

## Output Format

**You do NOT output a `standard_name` string.** You fill individual IR segment
fields. Code assembles the canonical name via ISN's `compose()` function.

For each signal that represents a distinct physics quantity, generate IR segments. Return a JSON object matching this schema:

```json
{
  "candidates": [
    {
      "source_id": "signal_id_here",
      "base_token": "temperature",
      "base_kind": "quantity",
      "qualifiers": ["electron"],
      "description": "Electron temperature measured by Thomson scattering",
      "reason": "qualifier=electron, base=temperature"
    }
  ],
  "skipped": ["status_flag_signal", "timing_reference_signal"]
}
```

- **source_id**: The signal ID
- **base_token**: The irreducible base quantity from the closed registry
- **base_kind**: `"quantity"` or `"geometry"`
- **projection_axis**: Axis projection (e.g., `"radial"`, `"toroidal"`). Null if none.
- **projection_shape**: `"component"` or `"coordinate"`. Required when `projection_axis` is set.
- **qualifiers**: List of qualifier tokens (species, population). Empty list if none.
- **locus_token**: Entity/position/region token for postfix locus. Null if none.
- **locus_relation**: `"of"`, `"at"`, or `"over"`. Required when `locus_token` is set.
- **locus_type**: `"entity"`, `"position"`, `"region"`, or `"geometry"`. Required when `locus_token` is set.
- **process_token**: Process token for `_due_to_`. Null if none.
- **operator_token**: Operator token. Null if none.
- **operator_kind**: `"unary_prefix"` or `"unary_postfix"`. Required when `operator_token` is set.
- **description**: A 1-line ≤120 char plain-English summary of the physical quantity (do NOT repeat the name verbatim)
- **reason**: Brief justification for the segment choices
- **skipped**: List of signal IDs that are not distinct physics quantities
