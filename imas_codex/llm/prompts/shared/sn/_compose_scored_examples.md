{% if compose_scored_examples %}
## SCORED EXAMPLES — calibrate your output quality

Previously reviewed standard names spanning the quality spectrum.
**Emulate** Outstanding/Good examples; **avoid** the patterns in Threshold/Inadequate ones.
Grammar decomposition shows how each name is structured — use the same segment discipline.

{% for ex in compose_scored_examples %}
### Score {{ "%.2f"|format(ex.reviewer_score) }} example (band {{ "%.2f"|format(ex.target_score) }}){% if ex.target_score >= 0.80 %} ✅ EMULATE{% else %} ⚠️ AVOID{% endif %}

**`{{ ex.id }}`** [{{ ex.unit or 'dimensionless' }}, kind={{ ex.kind }}]
{% if ex.grammar_fields is defined and ex.grammar_fields %}Grammar: {% for seg, val in ex.grammar_fields.items() %}`{{ seg }}`=`{{ val }}`{% if not loop.last %}, {% endif %}{% endfor %}{% endif %}
{% if ex.semantic_sim is defined and ex.semantic_sim is not none %}Semantic similarity (name↔description): {{ "%.3f"|format(ex.semantic_sim) }}{% if ex.semantic_sim >= 0.85 %} ✅{% elif ex.semantic_sim >= 0.70 %} ⚠️{% else %} ❌{% endif %}{% endif %}
Description: {{ ex.description }}

Per-dimension scores:
{% for dim, score in ex.scores.items() %}
- **{{ dim }}: {{ score }}/20**{% if ex.dimension_comments.get(dim) %} — {{ ex.dimension_comments[dim] }}{% endif %}

{% endfor %}
{% if ex.reviewer_comments %}Reviewer summary: *{{ ex.reviewer_comments }}*{% endif %}
{% if ex.physics_domain %}Physics domain: {{ ex.physics_domain }}{% endif %}

{% endfor %}
{% endif %}
