{% if review_scored_examples %}
## REVIEWER CALIBRATION EXAMPLES

Previously reviewed standard names spanning the full score range. Each
example shows the per-dimension score you must produce and the reasoning
tied to each dimension. Grammar decomposition and semantic similarity
scores are included to calibrate your grammar and self-descriptiveness checks.

{% for ex in review_scored_examples %}
### Aggregate score {{ "%.2f"|format(ex.reviewer_score) }}

**`{{ ex.id }}`** [{{ ex.unit or 'dimensionless' }}, kind={{ ex.kind }}]
{% if ex.grammar_fields is defined and ex.grammar_fields %}IR segments: {% for seg, val in ex.grammar_fields.items() %}`{{ seg }}`=`{{ val }}`{% if not loop.last %}, {% endif %}{% endfor %}{% endif %}
{% if ex.semantic_sim is defined and ex.semantic_sim is not none %}Semantic similarity (name↔description): {{ "%.3f"|format(ex.semantic_sim) }}{% if ex.semantic_sim >= 0.85 %} ✅{% elif ex.semantic_sim >= 0.70 %} ⚠️{% else %} ❌{% endif %}{% endif %}
Description: {{ ex.description }}
{% if ex.documentation %}Documentation: {{ ex.documentation }}{% endif %}

Per-dimension scores and reasoning:
{% for dim, score in ex.scores.items() %}
- **{{ dim }}: {{ score }}/20** — {{ ex.dimension_comments.get(dim, '(no per-dimension comment recorded)') }}
{% endfor %}

{% if ex.reviewer_comments %}Reviewer summary: *{{ ex.reviewer_comments }}*{% endif %}
{% if ex.physics_domain %}Physics domain: {{ ex.physics_domain }}{% endif %}

{% endfor %}
{% endif %}
