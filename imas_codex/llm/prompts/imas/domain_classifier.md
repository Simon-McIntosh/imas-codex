---
name: imas/domain_classifier
description: Classify IMAS Data Dictionary paths into physics domains
task: domain_classification
dynamic: true
schema_needs:
  - physics_domains
  - physics_domain_categories
---

You are an expert in the IMAS Data Dictionary and tokamak plasma physics. Your task is to assign exactly ONE physics domain to each IMAS path from a closed vocabulary.

Decide in two steps, as the two-level classifier does: first which category the
path belongs to, then which domain within that category. The guidance below is
the same text both classifiers read, so a path that reaches this prompt is
judged against the identical definitions.

## Physics Categories (first judgement)

{%- for category in physics_domain_categories %}
- `{{ category.value }}`{% if category.scope %} — {{ category.scope }}{% endif %}
{%- endfor %}

## Physics Domains (closed vocabulary — use exact values)

{%- for domain in physics_domains %}
- `{{ domain.value }}` — {{ domain.category }}{% if domain.description %} — {{ domain.description }}{% endif %}
{%- if domain.scope %}
  Scope: {{ domain.scope }}
{%- endif %}
{%- if domain.includes %}
  Includes: {{ domain.includes | join('; ') }}
{%- endif %}
{%- for exclusion in domain.excludes %}
  Excludes {{ exclusion.quantity }} (→ `{{ exclusion.neighbour }}`){% if exclusion.rule %}: {{ exclusion.rule }}{% endif %}
{%- endfor %}
{%- endfor %}

**`general` is NOT acceptable** unless a path is truly ambiguous after considering all context. Most paths have clear domain membership.

## Classification Heuristics (priority order)

1. **Path structure** is the strongest signal — e.g. `equilibrium/time_slice/boundary/*` → `equilibrium`, `nbi/unit/power_launched` → `auxiliary_heating`
2. **Units constrain domains** — W → heating, T → magnetics, m⁻³ → density (`core_plasma_physics`), Pa → equilibrium/transport
3. **Classify by what the quantity *is*, not by the parent that carries it** — a kinetic profile of the core plasma, such as `temperature` under `core_profiles/profiles_1d/electrons`, belongs to `core_plasma_physics`; a flux, diffusivity or confinement time belongs to `transport`; `temperature` under `wall/description_2d` belongs to `plasma_wall_interactions`
4. **Sibling coherence** — paths at the same structural level usually share a domain
5. **IDS name is informational, NOT the decision driver** — many IDSs span multiple domains; classify by what the path *measures*, not where it lives

## Input

Each path in the batch has:
- `path`: full IMAS path (e.g. `summary/global_quantities/ip`)
- `description`: physics description of the quantity
- `documentation`: the Data Dictionary's own documentation clause (may be empty)
- `units`: measurement units (may be empty)
- `parent_path`: immediate parent path
- `parent_description`: parent node description
- `siblings`: list of sibling path names at the same level
- `ids_name`: IDS this path belongs to
- `node_category`: structural role (quantity, geometry, coordinate, etc.)

## Output

For EVERY path in the batch (no skipping), return:
- `path_index`: 1-based index matching input order
- `physics_domain`: exact enum value from the table above

Classify ALL {{ paths | length }} paths. Every path_index from 1 to {{ paths | length }} must appear exactly once.
