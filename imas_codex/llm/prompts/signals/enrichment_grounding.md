---
name: signals/enrichment_grounding
description: Optional context grounding for signal descriptions
task: enrichment
dynamic: true
schema_needs: []
---
{{ baseline }}

## Description contract
{% if grounding %}
Every factual statement in the description must be supported by the supplied
context for this particular signal. Say less when that context says less.
Do not fill gaps from general physics knowledge or a plausible interpretation
of a signal name. Names and accessors identify a record; they do not establish
an acquisition rate, operating mode, spatial location, circuit connection,
measurement technique, or purpose. Preserve ambiguous source tags without
expanding them. Shared wiki and code describe this signal only where the context
explicitly binds them to it. A statement about a neighbouring system or a
different channel is not evidence about this signal. Do not describe a setting,
configuration payload, reference or model output as a physical measurement.
{% endif %}
{% if claim_evidence %}
Return the description as atomic `statements`, each with `text` and a nonempty
`context_items` list of the exact bracketed context identifiers supporting it.
The description delivered to users is assembled ONLY from these statements;
the ordinary `description` field does not bypass this requirement. Cite the
supplied context, not these instructions or the output example. No supporting
item means no statement. An existing citation is necessary but its content
must also support the entire statement.
{% endif %}
{% if sentence_limit %}
Use at most {{ sentence_limit }} short sentence(s) in the description. Prefer
the source's explicit quantity and role over additional background detail.
{% endif %}

{% if claim_evidence %}
Return JSON only using this output schema, which supersedes the schema above:
{{ output_schema | tojson }}
{% endif %}
