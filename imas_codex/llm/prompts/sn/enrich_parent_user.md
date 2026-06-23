---
name: sn/enrich_parent_user
description: Per-item user prompt for enrich_parents — synthesizes one derived parent's description from its children
used_by: imas_codex.standard_names.workers.process_enrich_parents_batch
task: enrich_parents
dynamic: true
schema_needs: []
---

# Describe the derived parent: {{ item.name }}

This is a structural **derived parent** standard name. Write a concise
`description` of the common physical quantity GENERALISED over its children
listed below. Do NOT change the name, unit, or kind.

**Parent standard name:** `{{ item.name }}`
**Unit:** {{ item.unit or "—" }}
**Kind:** {{ item.kind or "scalar" }}
**Physics domain:** {{ item.physics_domain or "—" }}

## Children (the concrete instances this parent abstracts over)

{% if children and children | length > 0 %}
{% for c in children %}
- `{{ c.name }}`{% if c.unit %} [{{ c.unit }}]{% endif %}{% if c.physics_domain %} ({{ c.physics_domain }}){% endif %}{% if c.description %} — {{ c.description }}{% endif %}
{% endfor %}
{% else %}
_(no child descriptions available — ground on the parent name's grammar)_
{% endif %}

## Instructions

Synthesize the **shared** physical meaning of these children into a single
1–2 sentence description of `{{ item.name }}`. Capture what the children have
in common; do not describe any one child's specific component, axis, surface,
or region. Ground strictly on the children above — do not invent physics they
do not attest. American spelling, no LaTeX, no markdown links, ≤ 500 characters.
