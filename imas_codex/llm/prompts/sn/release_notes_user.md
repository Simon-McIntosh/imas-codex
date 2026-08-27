---
name: sn/release_notes_user
description: Per-release user prompt for release-notes synthesis — carries the release message, batch record, and catalog diff evidence
used_by: imas_codex.standard_names.release_notes.build_pr_notes
task: release_notes
dynamic: true
schema_needs: []
---

# Write the PR title and body for this catalog review batch

The title must be exactly: `{{ required_title }}`

The maintainer described the release as “{{ message or "review batch" }}”. The
batch contains {{ batch_size }} standard names{% if facility %} for {{ facility }}{% endif %}{% if dominant_domain %} in the {{ dominant_domain }} physics scope{% endif %}
and was assembled from `{{ minted_from }}`. Its release identifier is supplied
only as provenance and must not appear in the title: `{{ rc_version }}`.

The catalog diff has {{ change_counts.added }} additions,
{{ change_counts.changed }} changes, and {{ change_counts.removed }} removals.
{% if unmatched_count %}There are also {{ unmatched_count }} source paths without
a linked name; they are outside the batch.{% endif %}

{% if not dd_gaps.available %}
Linked Data Dictionary caveat evidence could not be read, so do not claim the
caveat count is zero.
{% else %}
Linked Data Dictionary evidence has {{ dd_gaps.unresolved_count }} unresolved
and {{ dd_gaps.retired_count }} retired caveats. These counts are warning-only.
{% endif %}

Return the exact required `title` and one short prose-paragraph `body` now.
