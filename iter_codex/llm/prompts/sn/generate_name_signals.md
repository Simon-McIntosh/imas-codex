---
name: sn/generate_name_signals
description: Generate standard names for facility signal descriptions
used_by: iter_codex.standard_names.workers.compose_worker
task: composition
dynamic: true
---

You are a physics nomenclature expert generating standard names for measured quantities at a fusion research facility.

{% include "sn/_grammar_reference.md" %}

## Standard Name Grammar

A candidate is an IR object, not a hand-written string. Choose `base_token` and
`base_kind`, then place every other registered token in `projection_axis`,
`qualifiers`, the typed locus fields, `process_token`, or the ordered
`operators` list. The complete live controlled vocabulary is injected above;
the shorter context lists below, when present, are search aids rather than an
alternative grammar.

## Composition Rules

1. Every candidate has one registered `base_token`; `base_kind` selects either a physical quantity or a geometry carrier.
2. Emit IR fields only. The authoritative composer determines canonical segment order and joining words.
3. Examples:
   - electron_temperature → `{"base_token": "temperature", "base_kind": "quantity", "qualifiers": ["electron"]}`
   - plasma_current → `{"base_token": "current", "base_kind": "quantity", "qualifiers": ["plasma"]}`
   - line_integrated_electron_density → `{"base_token": "density", "base_kind": "quantity", "qualifiers": ["electron"], "operators": [{"token": "line_integrated"}]}`
   - toroidal_magnetic_field_at_magnetic_axis → `{"base_token": "magnetic_field", "base_kind": "quantity", "projection_axis": "toroidal", "locus_token": "magnetic_axis", "locus_relation": "at", "locus_type": "position"}`
4. Use existing standard names as reference for naming conventions
5. Signal descriptions may use facility-specific jargon; resolve it with the supplied enriched description and evidence, never by inventing an unstated semantic axis
6. Skip signals that are status flags, configuration parameters, or timing references
7. `measured`, `reconstructed`, and `reference` are controlled value-provenance metadata on the source binding, never name segments; all estimator facets collapse to the base quantity
8. Triangularity, elongation, squareness, and other shape parameters require an explicit surface locus; never emit a bare shape parameter
9. Error signals use the registered uncertainty operator around the base identity; never coin a per-error base name

{% if existing_names %}
## Existing Standard Names (do not duplicate)
{% for name in existing_names %}
- {{ name }}
{% endfor %}
{% endif %}

## Signals to Name

Facility: {{ facility }}, Domain: {{ domain }}

The enriched signal description is the primary grounding. A generic signal ID
never licenses a generic Standard Name; preserve the carrier, surface, subject,
projection, and process stated in the description. Similar DD paths are
supporting evidence only and must not override the source meaning.

{% for item in items %}
### Signal: {{ item.signal_id }}
- Enriched signal description (PRIMARY GROUNDING): {{ item.description }}
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
      "segments": {
        "base_token": "temperature",
        "base_kind": "quantity",
        "qualifiers": ["electron"]
      },
      "description": "Kinetic temperature of the electron population",
      "reason": "qualifier=electron, base=temperature"
    }
  ],
  "skipped": ["status_flag_signal", "timing_reference_signal"]
}
```

- **source_id**: The signal ID
- **segments**: Object containing the IR grammar segment fields:
  - **base_token**: The irreducible base quantity from the closed registry
  - **base_kind**: `"quantity"` or `"geometry"`
  - **projection_axis**: Axis projection (e.g., `"radial"`, `"toroidal"`). Null if none.
  - `base_kind` determines projection shape automatically: quantity → component, geometry → coordinate.
  - **qualifiers**: List of qualifier tokens (species, population). Empty list if none.
  - **locus_token**: Entity/position/region token for postfix locus. Null if none.
  - **locus_relation**: `"of"`, `"at"`, or `"over"`. Required when `locus_token` is set.
  - **locus_type**: `"entity"`, `"position"`, `"region"`, or `"geometry"`. Required when `locus_token` is set.
  - **process_token**: Process token for `_due_to_`. Null if none.
  - **operators**: Ordered outer-to-inner operator items. Each item has a bare registry `token`, optional indexed `coordinate`, and optional binary `secondary_operand`. Use `[]` if none.
- **description**: A 1-line ≤120 char plain-English summary of the physical quantity (do NOT repeat the name verbatim)
- **reason**: Brief justification for the segment choices
- **skipped**: List of signal IDs that are not distinct physics quantities
