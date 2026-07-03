---
name: sn/refine_docs_user
description: User prompt for refine_docs — renders DOCS_REVISION_OF chain history so the LLM learns from prior reviewer feedback on documentation
used_by: imas_codex.standard_names.workers.refine_docs_worker
task: enrichment
dynamic: true
schema_needs: []
---
You are refining the documentation for an existing standard name. A reviewer
scored a previous documentation attempt below the acceptance threshold. Study
the revision history below and produce improved documentation that materially
addresses the reviewer's concerns.

---

## Standard name being documented

- **Name:** `{{ sn_name }}`
- **Unit:** {{ unit or "—" }}
- **Kind:** {{ kind or "scalar" }}
- **Physics domain:** {{ physics_domain or "—" }}
{% if description %}
- **One-line description:** {{ description }}
{% endif %}

### Linked DD paths
{% if dd_paths %}
{% for path in dd_paths %}
- `{{ path.path }}`{% if path.ids %} ({{ path.ids }}{% if path.unit %}, unit: {{ path.unit }}{% endif %}){% elif path.unit %} (unit: {{ path.unit }}){% endif %}: {{ path.description or path.documentation or "(no documentation)" }}
{% endfor %}
{% else %}
_(no linked DD paths)_
{% endif %}

{% if derived_children %}
### THIS IS A DERIVED PARENT — generalize over its children when you rewrite

`{{ sn_name }}` is a structural **parent**: an abstraction over the more
specific child names below (its concrete physics + only DD grounding):

{% for c in derived_children %}- `{{ c.name }}`{% if c.unit %} [{{ c.unit }}]{% endif %}{% if c.physics_domain %} ({{ c.physics_domain }}){% endif %}{% if c.description %} — {{ c.description }}{% endif %}
{% endfor %}

The reviewer feedback below most likely flags **over-specialization** — docs
that re-describe a single child instead of the general quantity. Your rewrite
**must describe the general quantity these children share** and must **NOT**
narrow to any one child's species, component, axis, projection, qualifier, or
normalization (e.g. for parent `pressure_at_post_sawtooth_crash` write the
generic pressure at that state, not its `electron_pressure_*` child). Cross-
reference a representative child or two with `[label](name:bare_id)`. Ground
strictly on what the children attest.
{% endif %}

{% if sibling_family %}
### Sibling family — converge on the family template (shared parent `{{ sibling_family.parent.name }}`)

`{{ sn_name }}` is one member of a matched sibling set. Your rewrite must be
**structurally parallel** to the family: same opening noun-phrase template and
section structure as the anchor / accepted siblings, varying ONLY the
axis/species/zone-specific token, member-specific symbols, and genuinely
member-specific physics. Never copy a sibling's physics claim that does not
hold for this member — harmonize the structure, not the physics.

{% if sibling_family.anchor %}Template anchor **`{{ sibling_family.anchor.name }}`**{% if sibling_family.anchor.is_parent %} (the family parent){% endif %}:
- Anchor description: {{ sibling_family.anchor.description }}
{% if sibling_family.anchor.documentation %}- Anchor documentation:

{{ sibling_family.anchor.documentation }}
{% endif %}
{% endif %}
Members:
{% for s in sibling_family.siblings %}
- `{{ s.name }}`{% if s.axis %} (axis: {{ s.axis }}){% endif %}{% if s.docs_stage %} — docs {{ s.docs_stage }}{% endif %}{% if s.description %} — {{ s.description }}{% endif %}{% if s.documentation_opening %}
  - documentation opens: "{{ s.documentation_opening }}"{% endif %}
{% endfor %}
{% endif %}

---

## Docs revision history (oldest first; docs chain length so far: {{ docs_chain_length }})

{% if docs_chain_history %}
{% for h in docs_chain_history %}
### Revision {{ loop.index }} (model: {{ h.model }})

- **Reviewer score:** {{ "%.2f"|format(h.reviewer_score) }}
- **Per-dimension comments:**
{% if h.reviewer_comments_per_dim %}
{% for dim, comment in h.reviewer_comments_per_dim.items() %}
  - **{{ dim }}**: {{ comment }}
{% endfor %}
{% else %}
  _(no per-dimension comments recorded)_
{% endif %}

**Prior documentation:**

{{ h.documentation or "(empty)" }}

---
{% endfor %}
{% else %}
_(no prior revision history — this is the first docs refine attempt)_
{% endif %}
{% set _cur_score_docs = reviewer_score_docs | default(none, true) %}
{% set _cur_comments_docs = reviewer_comments_per_dim_docs | default(none, true) %}
{% if _cur_score_docs is not none or _cur_comments_docs %}
### Current node docs review

- **Reviewer score:** {{ "%.2f"|format(_cur_score_docs) if _cur_score_docs is not none else "—" }}
- **Per-dimension comments:**
{% set _per_dim_docs = (_cur_comments_docs | fromjson) if _cur_comments_docs else {} %}
{% if _per_dim_docs %}
{% for dim, comment in _per_dim_docs.items() %}
  - **{{ dim }}**: {{ comment }}
{% endfor %}
{% else %}
  _(no per-dimension comments recorded)_
{% endif %}

{% endif %}
{% if docs_hint and edit_reason %}
### Expert steering ({{ edit_origin or "human" }})

A domain expert has proposed this documentation direction: "{{ docs_hint }}"
— for this reason: {{ edit_reason }}

This proposal is subordinate to the grammar and composition rules above —
realize the intent within the rules; if the rules forbid the literal
proposal, compose the nearest rule-compliant documentation. Do not treat it
as pre-approved.
{% endif %}
---

## Your task

Produce updated documentation for `{{ sn_name }}` that materially addresses
the **lowest-scoring dimensions** identified in the revision history above.

Rules:
- `description`: 1–3 sentences, ≤ 500 chars, no LaTeX, **American English**
  ("center", "meter", "ionization"), **no storage-shape tags** ("1D", "2D", "3D",
  "profile", "array" — describe the physics, not data layout).
- `documentation`: rich text with LaTeX math notation where appropriate,
  typical value ranges, physical intuition, measurement context.
  Do **not** regress on dimensions that already scored well.
  **American English** throughout.
- `links`: list of related standard names in `name:xxx` format.
  Only include genuine conceptual relationships.
- Sign conventions (if COCOS-dependent): use exactly
  `Sign convention: Positive when <condition>.` as a standalone paragraph
  (blank line before/after, plain text — no headings, no bold).
- The `name` field must equal `{{ sn_name }}` exactly — do not alter it.

Return a JSON object matching the output schema with fields:
``description``, ``documentation``, ``links``.
