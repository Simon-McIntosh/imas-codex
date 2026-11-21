---
name: sn/review_name_definition_user
description: Dynamic user-message portion of the name-versus-definition review seat (companion to review_name_definition_system)
used_by: iter_codex.standard_names.review.name_definition_seat
task: review
dynamic: true
schema_needs: []
---

Classify each item against the class sheet and worked examples in the system
prompt.

## Items to review

{% for item in items %}
### Item {{ loop.index }}
- **Standard name**: {{ item.standard_name }}
- **Description (definition)**: {{ item.description | default('(none)', true) }}
{% if item.documentation %}- **Documentation**: {{ item.documentation }}
{% endif %}
- **Unit**: {{ item.unit | default('N/A', true) }}
{% if item.derived_children %}- **Child names** (the name is a parent of these): {{ item.derived_children | join(', ') }}
{% endif %}

{% endfor %}

Return the JSON object described at the end of the system prompt: a `verdicts`
array, one entry per item in the order given, each with `name`, `verdict`,
`reader_class`, `reasoning` and `proposed_name`.