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
- **Unit:** {{ item.dd_units or item.unit or "—" }}
- **Data type:** {{ item.data_type or "—" }}
- **Physics domain:** {{ item.physics_domain or "—" }}

### Source meaning (authoritative — the quantity this name must denote)

{% if item.dd_description %}
{{ item.dd_description }}
{% elif item.dd_documentation %}
{{ item.dd_documentation }}
{% else %}
_(no source description available — ground on the path, the unit and the
neighbours below; do not invent physics)_
{% endif %}
{% if item.dd_description and item.dd_documentation
      and item.dd_documentation != item.dd_description %}
Terse data-dictionary string for the same path (secondary — the enriched
description above is the grounding; use this only to check a detail):
{{ item.dd_documentation }}
{% endif %}
{% if item.dd_parent_description %}
Parent structure this path sits under — {{ item.dd_parent_description }}
{% endif %}
{% if item.description
      and "description pending LLM enrichment" not in item.description %}
- **Current one-line description of the name under refinement:**
  {{ item.description }}
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
{% if _cur_score is not none or _cur_comments or failed_checks is defined %}
### Current node review (name: `{{ item.id }}`)

- **Reviewer score:** {{ "%.2f"|format(_cur_score) if _cur_score is not none else "—" }}
{% if failed_checks is defined %}
- **Failed checks:**
{% for check in failed_checks %}
  - **{{ check.id }}** — {{ check.criterion }}
    - Evidence: {{ check.evidence }}
    - Segment or sentence: {{ check.segment or check.sentence }}
{% endfor %}
{% else %}
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

{% endif %}
{% if item.name_hint and item.edit_reason %}
### Expert steering ({{ item.edit_origin or "human" }})

A domain expert has proposed this naming direction: "{{ item.name_hint }}" —
for this reason: {{ item.edit_reason }}

This proposal is subordinate to the grammar and composition rules above —
realize the intent within the rules; if the rules forbid the literal
proposal, compose the nearest rule-compliant name. Do not treat it as
pre-approved.
{% endif %}{% if item.refine_stop_reason %}

---

## ⚠️ How the previous attempt stopped: `{{ item.refine_stop_reason }}`
{% if item.refine_collision_name %}

The previous attempt proposed **`{{ item.refine_collision_name }}`**, which is
already an occupied StandardName identity. Refinement cannot take an occupied
identity, so that proposal was refused and no successor was created.

**Do not propose `{{ item.refine_collision_name }}` again**, and do not reach
for a cosmetic variant of it that means the same thing — the same refusal
follows. Either propose a genuinely distinct identity for this quantity, or, if
`{{ item.refine_collision_name }}` really is the only correct name for it, say
exactly that in `reason`: that is a fold decision for an operator, not a
rewrite you can perform here.
{% else %}

Take this into account: repeating whatever the previous attempt did will
reproduce the same stop and spend another attempt for nothing.
{% endif %}
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

Propose a name that materially addresses the **lowest-scoring dimensions**
identified in the history above, for the **same physical quantity** the source
section describes. The unit and the source path are fixed; you are repairing
how the quantity is labelled, not what it is.

Rules:
- Do **not** repeat any name that appears in the refinement history, and do not
  propose an identity that already exists as a StandardName — refinement
  cannot take an occupied name. The one exception: if the reviewer's objection
  identifies no defect in the current name, return the current name and say so
  in `reason`. Returning a correct name deliberately is a valid outcome;
  rotating to a synonym to appear responsive is not.
- Do **not** include unit or physics_domain — those are injected post-LLM.
- Follow the standard name grammar: fill IR segment fields inside the `segments`
  object (`base_token`, `base_kind`, `projection_axis`, `qualifiers`, etc.),
  no abbreviations, no instrument prefixes for generic observables.
- **Locus prepositions:** Follow the installed locus relation matrix. Set
  `locus_type` to the locus's registered type and `locus_relation` to one of
  the relations that type admits:
{% if grammar and grammar.locus_relation_matrix %}
{% for ltype, relations in grammar.locus_relation_matrix | dictsort %}
  - `locus_type="{{ ltype }}"` admits {% for r in relations %}`{{ r }}`{% if not loop.last %}, {% endif %}{% endfor %}
{% endfor %}
{% endif %}
  Where a type admits more than one relation the quantity semantics decide:
  `of` for an intrinsic property of, or association with, a named entity;
  `at` for a field evaluated at a position; `along` for a quantity following a
  path; `over` for a reduction across a region. Never invent a preposition and
  never use `_from_`.
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
  `base_kind`, `projection_axis`, `qualifiers`,
  `locus_token`, `locus_relation`, `locus_type`, `process_token`,
  and the outer-to-inner `operators` list.
- Provide a brief `reason` explaining how this attempt addresses the reviewer's
  specific concerns from the history above.

Return a JSON object matching the output schema.
