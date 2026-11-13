---
name: sn/review_name_definition_user
description: Dynamic user-message portion of the name-versus-definition review seat (companion to review_name_definition_system)
used_by: imas_codex.standard_names.review.name_definition_seat
task: review
dynamic: true
schema_needs: []
---

Apply the comparison rules in the system prompt to each item below. Decide
whether the definition states a qualifier the name does not, and name the kind.
The unit is supplied so you can check that the quantity the definition describes
has the dimension of the quantity the name states.

## Items to Review

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

## Output Format

Return a JSON object with a `verdicts` array, one verdict per item, in the order
given: `name`, `verdict` (`flag` or `pass`), `qualifier` (`normalisation`,
`counting-convention`, `weighting`, `integration`, `restriction-to-one-object`,
`family-of-quantities`, or `agrees`), and `reasoning` naming the wording in the
definition that states the extra qualifier, or the one quantity both texts share,
and `proposed_name`: when the flag is a `restriction-to-one-object`, a spelling of
the name that states the restriction (for example a name that says which face,
coil, module, reaction branch, instrument class or discharge phase it covers);
empty otherwise.
Follow the schema given in the system prompt.