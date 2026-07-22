---
name: sn/release_notes_user
description: Per-release user prompt for release-notes synthesis — carries the release message, batch record, and catalog diff evidence
used_by: imas_codex.standard_names.release_notes.build_pr_notes
task: release_notes
dynamic: true
schema_needs: []
---

# Write the PR title and body for this catalog review batch

## Release message (the maintainer's intent — one line)

{{ message or "(none given)" }}

## Batch record (the frozen review-batch artifact)

- RC version: `{{ rc_version }}`
- Batch size: {{ batch_size }} standard name(s)
- Minted from: `{{ minted_from }}`
{% if unmatched_count %}- Source paths with no linked name (reported, not in batch): {{ unmatched_count }}
{% endif %}

## Catalog diff (per physics domain, computed against the base branch)

{% if domains %}
{% for d in domains %}
- **{{ d.domain }}**: {{ d.added | length }} added{% if d.changed %}, {{ d.changed | length }} changed{% endif %}{% if d.removed %}, {{ d.removed | length }} REMOVED{% endif %}

{% if d.added %}  added: {% for n in d.added[:12] %}`{{ n }}`{% if not loop.last %}, {% endif %}{% endfor %}{% if d.added | length > 12 %} … (+{{ d.added | length - 12 }} more){% endif %}
{% endif %}
{% if d.changed %}  changed: {% for n in d.changed[:12] %}`{{ n }}`{% if not loop.last %}, {% endif %}{% endfor %}{% if d.changed | length > 12 %} … (+{{ d.changed | length - 12 }} more){% endif %}
{% endif %}
{% if d.removed %}  removed: {% for n in d.removed[:12] %}`{{ n }}`{% if not loop.last %}, {% endif %}{% endfor %}{% if d.removed | length > 12 %} … (+{{ d.removed | length - 12 }} more){% endif %}
{% endif %}
{% endfor %}
{% else %}
(no per-domain diff evidence available — describe only the batch record)
{% endif %}

Write the `title` and markdown `body` now, grounded strictly on the evidence
above.
