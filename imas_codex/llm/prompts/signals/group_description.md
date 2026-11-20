---
name: signals/group_description
description: One grounded description of an archive component or field pattern
task: enrichment
dynamic: true
schema_needs: []
---
You describe one group of archive variables using only the context supplied for
that group. The group is {% if kind == "component" %}one physical component identified by a control path and a component code{% else %}one field pattern shared by several components of the same function category{% endif %}.

## Description contract

- Every factual statement must be supported by at least one supplied context
  item, cited by its exact bracketed identifier.
- Say less when the context says less. Do not fill gaps from general physics
  knowledge, from the component or field code, or from the shape of an
  identifier.
- An identifier names a record; it does not establish an acquisition rate,
  operating mode, spatial location, circuit connection, measurement technique
  or purpose.
- Use at most {{ sentence_limit }} short sentence(s).

{% if kind == "component" %}
## Component description

Describe what the component is and what its channels record or control, only as
the supplied function-category meaning, plant structure and source rows state
it. A source row speaks only about the field it names. Leave `fields` empty.
{% else %}
## Field-pattern description

Describe what the pattern's field codes mean, field by field, only where a
supplied source row names that field and states a meaning. Put the supported
field meanings in `fields`, keyed by the exact field code, and keep `statements`
to a short summary of the shared fields. Omit a field whose meaning no supplied
row states.
{% endif %}

Return JSON only using this output schema:

{{ output_schema | tojson }}