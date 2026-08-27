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

Return the exact required `title` and one short prose-paragraph `body` now.
