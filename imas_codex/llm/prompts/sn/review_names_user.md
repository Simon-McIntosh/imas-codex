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
{% endif %}{% if prior_reviews %}
## Authoritative Escalation Context

The independent reviewers below disagreed enough to trigger this escalation.
Resolve their concrete critiques against the authoritative source and grammar
evidence in each candidate. Do not merely average their scores.

{% for review in prior_reviews %}
### {{ review.role | title }} review — {{ review.model }}
- **Overall score**: {{ review.score }}
- **Dimension scores**: {{ review.scores_json }}
- **Critique**: {{ review.reasoning | default('No prose critique supplied.', true) }}
{% if review.comments_per_dim_json and review.comments_per_dim_json != '{}' %}- **Per-dimension critique**: {{ review.comments_per_dim_json }}
{% endif %}
{% endfor %}
{% endif %}

## Candidates to Review

Score each exact candidate against its authoritative bound sources and any
structurally proven representation/owner family shown below. Semantic-cluster,
vector, and graph-neighbor context is comparison-only: it can reveal a naming
pattern or collision, but it never expands the candidate's identity obligation.

{% for item in items %}
### Candidate {{ loop.index }}
- **Standard name**: {{ item.standard_name or item.id }}
- **Source ID**: {{ item.source_id | default('N/A', true) }}
- **Unit**: {{ item.unit | default('N/A', true) }}
- **Kind**: {{ item.kind | default('N/A', true) }}
{% if item.grammar_projection %}- **Strict public-ISN round trip**: `{{ item.grammar_round_trip }}` (ISN {{ item.grammar_parse_version }})
- **Complete grammar projection**: {% for entry in item.grammar_projection %}{% if not loop.first %}, {% endif %}{{ entry.field }}={{ entry.value }}{% endfor %}
- **Recursive semantic IR**: `{{ item.semantic_ir }}`
{% else %}- **Grammar Fields**: {% if item.physical_base %}physical_base={{ item.physical_base }}{% endif %}{% if item.subject %}, subject={{ item.subject }}{% endif %}{% if item.component %}, component={{ item.component }}{% endif %}{% if item.coordinate %}, coordinate={{ item.coordinate }}{% endif %}{% if item.position %}, position={{ item.position }}{% endif %}{% if item.process %}, process={{ item.process }}{% endif %}
{% endif %}{% if item.dd_source_docs %}
**Authoritative source and structural-family obligations**:

**Exact DD source definitions**:
{% if item.unpinned_source_count %}
⚠️ **Provenance incomplete:** {{ item.unpinned_source_count }} exact DD source binding(s) lack a pinned extraction snapshot. Unpinned text below is mutable context, not authoritative evidence; flag the grounding deficiency in the review.
{% endif %}
{% for source in item.dd_source_docs %}- {% if source.ids %}{% for source_id in source.ids %}{% if not loop.first %}, {% endif %}`{{ source_id }}`{% endfor %}{% else %}`{{ source.id }}`{% endif %}{% if source.dd_version %} (DD {{ source.dd_version }}){% endif %} [{{ source.unit | default('N/A', true) }}]
{% if source.snapshot_pinned %}{% if source.documentation %}  - **Pinned immutable DD definition (authoritative)**: {{ source.documentation }}
{% else %}  - **Pinned snapshot has no DD definition text.**
{% endif %}{% else %}  - ⚠️ **Unpinned DD definition (non-authoritative)**: {{ source.documentation | default('No DD definition supplied.', true) }}
{% endif %}{% if source.description and source.description != source.documentation %}  - **Non-authoritative enriched description**: {{ source.description }}
{% endif %}{% endfor %}
{% endif %}{% if item.dd_parent_contexts %}
**Exact DD parent-array context**:
{% for parent in item.dd_parent_contexts %}- {% if parent.paths %}{% for path in parent.paths %}{% if not loop.first %}, {% endif %}`{{ path }}`{% endfor %}{% else %}`{{ parent.path | default('parent structure', true) }}`{% endif %}{% if parent.dd_version %} (DD {{ parent.dd_version }}){% endif %}
{% if parent.snapshot_pinned %}  - **Pinned immutable parent definition (authoritative)**: {{ parent.documentation | default('No parent definition supplied.', true) }}
{% else %}  - ⚠️ **Unpinned parent definition (non-authoritative)**: {{ parent.documentation | default('No parent definition supplied.', true) }}
{% endif %}
{% endfor %}
When this parent describes a set or array, a generic entity locus in the strict
parse denotes the member selected by the array index. It is an intentional
parameter, not a missing concrete species or instance. Require a concrete
member only when the DD leaf itself fixes one.
{% endif %}{% if item.source_hints %}
**Exact-source composition steering** (intent context; never overrides pinned DD or grammar):
{% for source_hint in item.source_hints %}- `{{ source_hint.source_id }}`: {{ source_hint.hint }}{% if source_hint.reason %} — {{ source_hint.reason }}{% endif %}
{% endfor %}
{% endif %}{% if item.dd_documentation and not item.unpinned_source_count %}- **DD ground truth** (authoritative source definition — verify physics_accuracy against THIS): {{ item.dd_documentation }}
{% elif item.dd_documentation %}- **Unpinned DD context** (non-authoritative): {{ item.dd_documentation }}
{% endif %}{% if item.dd_description %}- **DD enriched description**: {{ item.dd_description }}
{% endif %}{% if item.physics_domain %}- **Physics domain**: {{ item.physics_domain }}
{% endif %}{% if item.dd_keywords %}- **DD keywords**: {{ item.dd_keywords | join(', ') if item.dd_keywords is iterable and item.dd_keywords is not string else item.dd_keywords }}
{% endif %}{% if item.source_paths %}- **Source paths** (authoritative bound-source cohort): {{ item.source_paths | join(', ') }}
{% endif %}{% if item.source_context_omitted %}- **Bounded source context**: {{ item.source_context_omitted }} additional exact source binding(s) omitted after deterministic ordering.
{% endif %}
{% if item.validation_issues %}
**ISN Validation Issues:**
{% for issue in item.validation_issues %}
- {{ issue }}
{% endfor %}
{% endif %}
{% if item.semantic_warning %}

{{ item.semantic_warning }}
{% endif %}{% if item.value_provenance or item.data_type or item.node_type or item.coordinate_paths or item.timebase or item.cocos_label or item.lifecycle_status or item.parent_path or item.parent_description or item.ancestor_context or item.identifier_schema or item.identifier_values or item.semantic_comparators or item.dd_paths_docs or item.hybrid_neighbours or item.related_neighbours or item.error_fields or item.sibling_fields %}
{% if item.value_provenance %}- **Value provenance**: {{ item.value_provenance }}; review the underlying quantity at `{{ item.provenance_base_path }}` rather than encoding the estimator in the name.
{% endif %}{% if item.data_type %}- **Data type**: {{ item.data_type }}
{% endif %}{% if item.node_type %}- **Node type**: {{ item.node_type }}
{% endif %}{% if item.coordinate_paths %}- **Coordinates**: {{ item.coordinate_paths | join(', ') }}
{% endif %}{% if item.timebase %}- **Timebase**: {{ item.timebase }}
{% endif %}{% if item.cocos_label %}- **COCOS transformation type**: `{{ item.cocos_label }}`{% if item.cocos_expression %} — expression: `{{ item.cocos_expression }}`{% endif %}
{% endif %}{% if item.lifecycle_status %}- **Lifecycle**: {{ item.lifecycle_status }}
{% endif %}
{% if item.parent_path or item.parent_description %}
- **Current DD parent context** (supplementary to the pinned snapshot above): {% if item.parent_path %}`{{ item.parent_path }}`{% else %}parent structure{% endif %}{% if item.parent_description %} — {{ item.parent_description }}{% endif %}
{% endif %}
{% if item.ancestor_context %}
- **DD path lineage** (nearest ancestor first; use it to resolve physics meaning and evaluation locus):
{% for ancestor in item.ancestor_context %}  - `{{ ancestor.path }}`: {{ ancestor.text }}
{% endfor %}{% endif %}
{% if item.identifier_schema %}- **Identifier schema**: {{ item.identifier_schema }}{% if item.identifier_schema_doc %} — {{ item.identifier_schema_doc }}{% endif %}
{% endif %}
{% if item.identifier_values %}
- **Identifier enum values**:
{% for value in item.identifier_values %}  - `{{ value.name }}` ({{ value.index }}): {{ value.description | default('', true) }}
{% endfor %}{% endif %}
{% if item.semantic_comparators %}
- **Non-binding semantic comparators** (optional comparison context; these never expand the candidate's identity obligation):
{% for comparator in item.semantic_comparators %}  - `{{ comparator.path }}`{% if comparator.basis %} — {{ comparator.basis | replace('_', ' ') }}{% endif %}
{% endfor %}{% endif %}
{% if item.dd_paths_docs %}
- **Current-graph member DD context** (supplementary and non-authoritative; pinned source snapshots above are the sole definition authority):
{% for path, documentation in item.dd_paths_docs | dictsort %}  - `{{ path }}`: {{ documentation }}
{% endfor %}{% endif %}
{% if item.hybrid_neighbours %}
- **Hybrid-search neighbours** (semantic and structural comparison only):
{% for neighbour in item.hybrid_neighbours %}  - `{{ neighbour.tag }}` [{{ neighbour.unit }}, {{ neighbour.physics_domain }}]: {{ neighbour.doc_short }}{% if neighbour.cocos_label %} (COCOS {{ neighbour.cocos_label }}){% endif %}
{% endfor %}{% endif %}
{% if item.related_neighbours %}
- **Graph-relationship neighbours**:
{% for neighbour in item.related_neighbours %}  - `{{ neighbour.path }}` ({{ neighbour.ids }}) — {{ neighbour.relationship_type }}{% if neighbour.via %} via {{ neighbour.via }}{% endif %}
{% endfor %}{% endif %}
{% if item.error_fields %}
- **DD error companions** (do not encode these deterministic companions in the base name):
{% for path in item.error_fields %}  - `{{ path }}`
{% endfor %}{% endif %}
{% if item.sibling_fields %}
- **Sibling fields** (same parent structure):
{% for sibling in item.sibling_fields %}  - `{{ sibling.path }}`: {{ sibling.description or 'no description' }} ({{ sibling.data_type or '?' }})
{% endfor %}{% endif %}
{% endif %}{% if item.clusters or item.dd_clusters %}
- **Semantic clusters** (comparison-only; membership is not identity authority):
{% for cluster in item.clusters or item.dd_clusters %}  - **{{ cluster.label }}** ({{ cluster.scope }}): {{ cluster.description }}{% if cluster.members %}
    Members: {{ cluster.members | join(', ') }}{% endif %}
{% endfor %}{% endif %}
{% if item.version_history or item.dd_version_history %}
- **DD version history:**
{% for change in item.version_history or item.dd_version_history %}  - {{ change.change_type }} (v{{ change.version }}){% if change.description %} — {{ change.description }}{% endif %}
{% endfor %}{% endif %}

{% if item.edit_reason %}
- **Deliberate expert steering**: a domain expert ({{ item.edit_origin or "human" }}) has deliberately steered this candidate for the following reason: {{ item.edit_reason }}. Judge the candidate on its physical and grammatical merits given this intent; do NOT penalize it merely for differing from a prior or established variant.
{% if item.physical_base %}
- **Deterministic grammar check**: this candidate PARSES under the registered ISN grammar (verified decomposition: physical_base=`{{ item.physical_base }}`{% if item.geometry %}, locus=`{{ item.geometry }}`{% endif %}{% if item.grammar_parse_version %}; grammar v{{ item.grammar_parse_version }}{% endif %}). Score the grammar dimension from this verified decomposition; do not assume any of its tokens are unregistered.
{% endif %}
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
      "issues": [],
      "dd_gaps": []
    }
  ]
}
```

Leave `"dd_gaps": []` unless the system prompt's flag-only contract is met.
When reporting evidence, every entry requires `"path"`, `"kind"`, and a
substantive `"reason"`.
