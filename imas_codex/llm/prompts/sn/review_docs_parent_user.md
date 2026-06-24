---
name: sn/review_docs_parent_user
description: Per-item DERIVED-PARENT docs review — candidate + its children (the generalization reference)
used_by: imas_codex.standard_names.workers.process_review_docs_batch
task: review
dynamic: true
schema_needs: []
---

Apply the **parent rubric** (provided in the system prompt) to the derived
parent below. Score it as an abstraction over its children — reward correct
generalization and positioning; do NOT dock for missing child-level specifics.

## Candidate (derived parent)

- **Standard name**: {{ item.id }}
- **Unit**: {{ item.unit | default('N/A', true) }}
- **Kind**: {{ item.kind | default('N/A', true) }}
{% if item.physics_domain %}- **Physics domain**: {{ item.physics_domain }}
{% endif %}
- **Description**: {{ item.description | default('(missing)', true) }}
- **Documentation**:

{{ item.documentation | default('(missing)', true) }}

## Children — the concrete instances this parent generalizes (PRIMARY reference)

The parent must correctly capture the **common quantity** these children share,
and be positioned as their abstraction (not a restatement of any single one):

{% if item.derived_children %}
{% for c in item.derived_children %}- `{{ c.name }}`{% if c.unit %} [{{ c.unit }}]{% endif %}{% if c.physics_domain %} ({{ c.physics_domain }}){% endif %}{% if c.description %} — {{ c.description }}{% endif %}
{% endfor %}
{% else %}
_(no live children currently linked — score generalization on the name + the
parent's own grammar; flag in positioning if the parent appears orphaned)_
{% endif %}

## Sibling-Comparison Context (style/terminology consistency only)

{% if vector_neighbours %}
### Nearest by description (vector similarity)
{% for n in vector_neighbours %}
- **`{{ n.id }}`** ({{ n.kind | default('scalar', true) }}, {{ n.unit | default('dimensionless', true) }}) — {{ n.description | default('', true) }}
{% endfor %}
{% endif %}

{% if same_base_neighbours %}
### Same `physical_base`
{% for n in same_base_neighbours %}
- **`{{ n.id }}`** ({{ n.kind | default('scalar', true) }}, {{ n.unit | default('dimensionless', true) }}) — {{ n.description | default('', true) }}
{% endfor %}
{% endif %}

{% if not vector_neighbours and not same_base_neighbours %}
*No accepted siblings found — score on generalization + physics correctness alone.*
{% endif %}

{% include "sn/_review_scored_examples.md" %}

Return JSON matching the parent schema defined in the system prompt; set
`source_id` and `standard_name` to `{{ item.id }}`.
