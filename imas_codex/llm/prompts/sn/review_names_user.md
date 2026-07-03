---
name: sn/review_names_user
description: Dynamic user-message portion of name-axis review (companion to review_names_system)
used_by: imas_codex.standard_names.workers.process_review_name_batch
task: review
dynamic: true
schema_needs: []
---

Apply the rubric (provided in the system prompt) to the candidate(s) below.

{% if batch_context %}
## Source Context (same as composer received)

{{ batch_context }}
{% endif %}

## Sibling-Comparison Context

Use these accepted, in-catalog names as your **third-party reference set**. They are NOT to be reviewed. Score the candidate(s) below against the **patterns** these siblings establish (decomposition style, segment usage, naming consistency). Cite specific sibling `id`s when you dock points.

{% if vector_neighbours %}
### Nearest by description (vector similarity)
{% for n in vector_neighbours %}
- **`{{ n.id }}`** ({{ n.kind | default('scalar', true) }}, {{ n.unit | default('dimensionless', true) }}) — {{ n.description | default('', true) }}{% if n.score is defined %} [sim={{ '%.2f' | format(n.score) }}]{% endif %}
{% endfor %}
{% endif %}

{% if same_base_neighbours %}
### Same `physical_base` (sibling decomposition pattern)
{% for n in same_base_neighbours %}
- **`{{ n.id }}`** ({{ n.kind | default('scalar', true) }}, {{ n.unit | default('dimensionless', true) }}) — {{ n.description | default('', true) }}
{% endfor %}
{% endif %}

{% if same_path_neighbours %}
### Same physics domain family
{% for n in same_path_neighbours %}
- **`{{ n.id }}`** ({{ n.kind | default('scalar', true) }}, {{ n.unit | default('dimensionless', true) }}) — {{ n.description | default('', true) }}
{% endfor %}
{% endif %}

{% if not vector_neighbours and not same_base_neighbours and not same_path_neighbours %}
*No accepted siblings found — score on grammar + physics correctness alone.*
{% endif %}

{% if nearby_existing_names %}
## Nearby Existing Standard Names

These names already exist in the catalog. Flag candidates that duplicate them:
{% for name in nearby_existing_names %}
- **{{ name.id }}**: {{ name.description | default('', true) }} ({{ name.kind | default('scalar', true) }}, {{ name.unit | default('dimensionless', true) }})
{% endfor %}
{% endif %}

## Candidates to Review

{% for item in items %}
### Candidate {{ loop.index }}
- **Standard name**: {{ item.standard_name or item.id }}
- **Source ID**: {{ item.source_id }}
- **Unit**: {{ item.unit | default('N/A', true) }}
- **Kind**: {{ item.kind | default('N/A', true) }}
- **Grammar Fields**: {% if item.physical_base %}physical_base={{ item.physical_base }}{% endif %}{% if item.subject %}, subject={{ item.subject }}{% endif %}{% if item.component %}, component={{ item.component }}{% endif %}{% if item.coordinate %}, coordinate={{ item.coordinate }}{% endif %}{% if item.position %}, position={{ item.position }}{% endif %}{% if item.process %}, process={{ item.process }}{% endif %}
{% if item.dd_documentation %}- **DD ground truth** (authoritative source definition — verify physics_accuracy against THIS): {{ item.dd_documentation }}
{% endif %}{% if item.dd_description %}- **DD enriched description**: {{ item.dd_description }}
{% endif %}{% if item.physics_domain %}- **Physics domain**: {{ item.physics_domain }}
{% endif %}{% if item.dd_keywords %}- **DD keywords**: {{ item.dd_keywords | join(', ') if item.dd_keywords is iterable and item.dd_keywords is not string else item.dd_keywords }}
{% endif %}{% if item.source_paths %}- **Source paths** (provenance context): {{ item.source_paths | join(', ') }}
{% endif %}
{% if item.validation_issues %}
**ISN Validation Issues:**
{% for issue in item.validation_issues %}
- {{ issue }}
{% endfor %}
{% endif %}
{% if item.semantic_warning %}

{{ item.semantic_warning }}
{% endif %}
{% if item.dd_clusters %}
- **Semantic clusters:**
{% for cl in item.dd_clusters %}  - **{{ cl.label }}** ({{ cl.scope }}): {{ cl.description }}
{% endfor %}{% endif %}
{% if item.dd_version_history %}
- **DD version history:**
{% for vh in item.dd_version_history %}  - {{ vh.change_type }} (v{{ vh.version }})
{% endfor %}{% endif %}
{% if item.edit_reason %}
- **Deliberate expert steering**: a domain expert ({{ item.edit_origin or "human" }}) has deliberately steered this candidate for the following reason: {{ item.edit_reason }}. Judge the candidate on its physical and grammatical merits given this intent; do NOT penalize it merely for differing from a prior or established variant.
{% endif %}
{% endfor %}

{% include "sn/_review_scored_examples.md" %}

## Output Format

Return a JSON object with a `reviews` array. Each review MUST include:

```json
{
  "reviews": [
    {
      "source_id": "path/to/quantity",
      "standard_name": "electron_temperature",
      "scores": {
        "grammar": 20,
        "semantic": 18,
        "convention": 19,
        "completeness": 18
      },
      "reasoning": "Brief specific justification covering each dimension",
      "revised_name": null,
      "revised_fields": null,
      "issues": []
    }
  ]
}
```
