---
name: sn/refine_name_user
description: User prompt for refine_name — renders REFINED_FROM chain history so the LLM learns from prior reviewer feedback
used_by: imas_codex.standard_names.workers.refine_name_worker
task: composition
dynamic: true
schema_needs: []
---
You are refining an existing draft standard name. A reviewer scored a previous
attempt below the acceptance threshold. Study the refinement history below and
produce an improved name that materially addresses the reviewer's concerns.

---

## Path being named

- **Path:** `{{ item.path }}`
- **IDS:** {{ item.ids_name or "—" }}
- **Description:** {{ item.description or "—" }}
- **Unit:** {{ item.unit or "—" }}
- **Data type:** {{ item.data_type or "—" }}
- **Physics domain:** {{ item.physics_domain or "—" }}
{% if item.parent_path %}
- **Parent path:** `{{ item.parent_path }}` — {{ item.parent_description or "" }}
{% endif %}

---

## Hybrid neighbours (semantically related DD paths)

{% if hybrid_neighbours %}
{% for n in hybrid_neighbours %}
- `{{ n.path }}` — {{ n.description or "(no description)" }}
{% endfor %}
{% else %}
_(none available)_
{% endif %}

{% if item.dd_clusters %}
## Semantic Clusters

{% for cl in item.dd_clusters %}- **{{ cl.label }}** ({{ cl.scope }}): {{ cl.description }}
{% endfor %}
{% endif %}

{% if item.dd_version_history %}
## DD Version History

{% for vh in item.dd_version_history %}- {{ vh.change_type }} (v{{ vh.version }})
{% endfor %}
{% endif %}

{% if item.dd_keywords %}
## Keywords

{{ item.dd_keywords | join(', ') }}
{% endif %}

---

## Refinement history (oldest first; chain length so far: {{ chain_length }})

{% if chain_history %}
{% for h in chain_history %}
### Attempt {{ loop.index }} (model: {{ h.model }})

- **Name:** `{{ h.name }}`
- **Reviewer score:** {{ "%.2f"|format(h.reviewer_score) }}
- **Per-dimension comments:**
{% if h.reviewer_comments_per_dim %}
{% for dim, comment in h.reviewer_comments_per_dim.items() %}
  - **{{ dim }}**: {{ comment }}
{% endfor %}
{% else %}
  _(no per-dimension comments recorded)_
{% endif %}

{% endfor %}
{% else %}
_(no prior refinement history — this is the first refine attempt)_
{% endif %}
{% set _cur_score = item.reviewer_score_name | default(none, true) %}
{% set _cur_comments = item.reviewer_comments_per_dim_name | default(none, true) %}
{% if _cur_score is not none or _cur_comments %}
### Current node review (name: `{{ item.id }}`)

- **Reviewer score:** {{ "%.2f"|format(_cur_score) if _cur_score is not none else "—" }}
- **Per-dimension comments:**
{% set _per_dim = (_cur_comments | fromjson) if _cur_comments else {} %}
{% if _per_dim %}
{% for dim, comment in _per_dim.items() %}
  - **{{ dim }}**: {{ comment }}
{% endfor %}
{% else %}
  _(no per-dimension comments recorded)_
{% endif %}

{% endif %}
{% if item.name_hint and item.edit_reason %}
### Expert steering ({{ item.edit_origin or "human" }})

A domain expert has proposed this naming direction: "{{ item.name_hint }}" —
for this reason: {{ item.edit_reason }}

This proposal is subordinate to the grammar and composition rules above —
realize the intent within the rules; if the rules forbid the literal
proposal, compose the nearest rule-compliant name. Do not treat it as
pre-approved.
{% endif %}{% if vocab_gap_detail %}

---

## ⚠️ Vocabulary Gap — Previous attempt rejected

The previous name was rejected because it used a token not in the registered vocabulary:

- **Segment:** {{ vocab_gap_detail.segment }}
- **Needed token:** `{{ vocab_gap_detail.token }}`
- **Reason:** {{ vocab_gap_detail.reason }}

**Fix:** Route this concept to the correct grammar segment. Check the segment routing table in the system prompt. If no registered token fits, emit a `vocab_gap`.
{% endif %}
{% if validation_issues %}

---

## Validation Issues

{% for issue in validation_issues %}
- {{ issue }}
{% endfor %}
{% endif %}
{% if fanout_evidence %}

---

{{ fanout_evidence }}
{% endif %}

---

{% include "sn/_compose_scored_examples.md" %}

## Your task

Propose a new name that materially addresses the **lowest-scoring dimensions**
identified in the history above.

Rules:
- Do **not** repeat any name that appears in the refinement history.
- Do **not** include unit or physics_domain — those are injected post-LLM.
- Follow the standard name grammar: fill IR segment fields inside the `segments`
  object (`base_token`, `base_kind`, `projection_axis`, `qualifiers`, etc.),
  no abbreviations, no instrument prefixes for generic observables.
- **Locus prepositions:** Named geometric entities (separatrix, magnetic_axis,
  plasma_boundary, x_point) and hardware objects (probe, coil, antenna) use
  `locus_relation="of"`. Abstract spatial regions from the position vocabulary
  (core, pedestal_top, lcfs) use `locus_relation="at"`.
- Provide a short `description` (≤ 120 chars, one sentence, no LaTeX).
- Set `kind` to exactly one of `"scalar"`, `"vector"`, `"tensor"`,
  `"complex"`, `"metadata"` — the structural classification of the quantity.
  Almost always `"scalar"` (every projected component or reduction is a
  scalar); `"vector"` only for an unprojected vector field; never invent
  other values.
- **No storage-shape tags** — NEVER write "1D", "2D", "3D", "profile", "array"
  in descriptions. Describe the *physics*, not data layout.
- **American English only** — "center" not "centre", "meter" not "metre".
- Provide all applicable IR segment fields inside `segments` — `base_token`,
  `base_kind`, `projection_axis`, `projection_shape`, `qualifiers`,
  `locus_token`, `locus_relation`, `locus_type`, `process_token`,
  `operator_token`, `operator_kind`.
- Provide a brief `reason` explaining how this attempt addresses the reviewer's
  specific concerns from the history above.

Return a JSON object matching the output schema.
