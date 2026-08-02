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

## Data Dictionary caveats (read-only lifecycle evidence)

- Total linked facts: {{ dd_gaps.total }}
- Awaiting triage: {{ dd_gaps.open_count }}
- Human-triaged or governed dispositions still unresolved: {{ dd_gaps.triaged_count }}
- Unresolved total: {{ dd_gaps.unresolved_count }}
- Retired: {{ dd_gaps.retired_count }}
- Retired facts whose enforcement registry entry is now stale: {{ dd_gaps.stale_registry_count }}
- Release-blocking: no (warning-only evidence)

{% if dd_gaps.facts %}
{% for gap in dd_gaps.facts %}
- **{{ gap.kind }} / {{ gap.status }}**{% if gap.upstream_url %} — upstream: {{ gap.upstream_url }}{% endif %}
  - Fact path or pattern: `{{ gap.path }}`
  - Exact linked path(s): {% if gap.exact_paths %}{% for path in gap.exact_paths %}`{{ path }}`{% if not loop.last %}, {% endif %}{% endfor %}{% else %}(none linked){% endif %}
  {% if gap.registry_backend %}- Registry backend: `{{ gap.registry_backend }}`{% endif %}
  {% if gap.resolved_dd_version %}- Corrected in published DD: `{{ gap.resolved_dd_version }}`{% endif %}
{% endfor %}
{% else %}
(no linked DD defects were reported)
{% endif %}

Write the `title` and markdown `body` now, grounded strictly on the evidence
above.
