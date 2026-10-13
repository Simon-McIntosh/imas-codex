{% if conventions %}
## Selected documentation conventions for this name

Use these entries from the conventions registry for this item.
{% for concept in conventions.concepts %}
### Concept: {{ concept.phrase }} (`{{ concept.token }}`)
- Definition: {{ concept.definition }}
- Canonical equation: `{{ concept.equation }}`
- Equation placement: {% if concept.restates_equation %}write this equation in the documentation{% else %}name the concept and link to it; an equation you do state for it is the canonical equation above, written exactly that way with only its symbols replaced by the quantity's own{% endif %}
{% for symbol in concept.symbols %}- `{{ symbol.symbol }}`: {{ symbol.meaning }}
{% endfor %}{% if concept.forbidden_variants %}- Forbidden variants: {{ concept.forbidden_variants | join('; ') }}
{% endif %}
{% endfor %}{% for notation in conventions.notation %}
### Notation: `{{ notation.token }}`
- Canonical symbol: `{{ notation.symbol }}`
{% for parameter in notation.parameters %}- Parameter `{{ parameter[0] }}`: {{ parameter[1] }}
{% endfor %}{% if notation.forbidden_spellings %}- Forbidden spellings: {{ notation.forbidden_spellings | join('; ') }}
{% endif %}
{% endfor %}{% for template in conventions.templates %}
### Documentation opening: `{{ template.base }}`{% if template.segments %} with {{ template.segments | join(', ') }}{% endif %}
- Description opening: {{ template.description }}
- Documentation opening: {{ template.documentation }}
{% endfor %}
{% endif %}
