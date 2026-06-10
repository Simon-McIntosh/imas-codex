---
name: sn/review_description_user
description: Dynamic user-message portion of compose-time description review (companion to review_description_system)
used_by: imas_codex.standard_names.benchmark.score_descriptions
task: review
dynamic: true
schema_needs: []
---

Apply the description rubric (provided in the system prompt) to the candidate(s) below. Score **only** the compose-time `description`; the standard name is shown so you can check name↔description consistency.

## Candidates to Review

{% for item in items %}
### Candidate {{ loop.index }}
- **Standard name**: {{ item.standard_name or item.source_id }}
- **Source ID**: {{ item.source_id }}
- **Description (under review)**: {{ item.description | default('(empty)', true) }}
- **Unit**: {{ item.unit | default('N/A', true) }}
- **Kind**: {{ item.kind | default('N/A', true) }}
{% if item.physics_domain %}- **Physics domain**: {{ item.physics_domain }}
{% endif %}
{% if item.source_paths %}- **Source DD paths** (provenance context): {{ item.source_paths | join(', ') }}
{% endif %}

{% endfor %}

## Output Format

Return a JSON object with a `reviews` array. One review per candidate, each with `source_id`, `standard_name`, four 0–20 `scores` (`physics_accuracy`, `specificity`, `consistency`, `concision`), a one-line `reasoning`, and an `issues` list (empty when none). Follow the schema given in the system prompt.
