---
name: sn/self_refine_user
description: User prompt for the free local self-refine pass — presents the composed name, description, source DD documentation, and deterministic grammar diagnostics for an improve-or-no-op review
used_by: imas_codex.standard_names.workers._self_refine_candidate
task: composition
dynamic: true
schema_needs: []
---
You just composed the following standard name. Review it once and either
improve it or return it unchanged.

## Composed candidate

- **Name:** `{{ name }}`
- **Description:** {{ description or "—" }}

## Grammar segments (as composed)

{% if segments %}
{% for seg, val in segments.items() %}
- **{{ seg }}**: {{ val }}
{% endfor %}
{% else %}
_(segments not available)_
{% endif %}

## Source data-dictionary context (authoritative — do not contradict)

{% if dd_context.path %}- **Path:** `{{ dd_context.path }}`
{% endif %}{% if dd_context.ids_name %}- **IDS:** {{ dd_context.ids_name }}
{% endif %}{% if dd_context.unit %}- **Unit:** {{ dd_context.unit }}
{% endif %}{% if dd_context.physics_domain %}- **Physics domain:** {{ dd_context.physics_domain }}
{% endif %}{% if dd_context.documentation %}- **Documentation:** {{ dd_context.documentation }}
{% elif dd_context.description %}- **Documentation:** {{ dd_context.description }}
{% endif %}

## Deterministic grammar diagnostics

{% if diagnostics %}
{% for d in diagnostics %}
- {{ d }}
{% endfor %}
{% else %}
- Round-trip: OK (name parses and re-composes to the canonical form).
{% endif %}

## Your task

Decide whether you can make this name *clearly* more self-describing, or its
description crisper / more consistent, **without** changing the physical
quantity, the unit, or inventing un-registered tokens.

- If yes: set `changed: true` and return the improved `name` (filling the IR
  `segments` fields) plus an improved `description` (≤ 120 chars, one sentence,
  no LaTeX, no storage-shape words, American English).
- If no: set `changed: false` and echo the original `name` and `description`.

Do **not** include `unit`, `physics_domain`, or `cocos` — those are injected
post-LLM. Return a JSON object matching the output schema.
