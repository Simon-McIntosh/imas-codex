---
name: sn/merge_notes_user
description: Per-merge user prompt for the fold-back tag summary — carries the PR description, conversation, commit messages, and review-delta diff
used_by: imas_codex.standard_names.release_notes.build_merge_notes
task: merge_notes
dynamic: true
schema_needs: []
---

# Summarise what this review round did to the batch

Write the `summary` for the fold-back tag, grounded strictly on the evidence
below.

## PR description (the batch as it was proposed)

{{ pr_description or "(none)" }}

## Review conversation (comments and reviews)

{% if conversation %}
{% for c in conversation %}
- **{{ c.author or "reviewer" }}** ({{ c.kind }}): {{ c.body }}
{% endfor %}
{% else %}
(no review comments or reviews recorded)
{% endif %}

## Commit messages (every commit in the PR)

{% if commit_messages %}
{% for m in commit_messages %}
- {{ m }}
{% endfor %}
{% else %}
(no commit messages recorded)
{% endif %}

## Review delta (diff of `standard_names/` — the batch as pushed vs the merged state)

{% if review_delta %}
```diff
{{ review_delta }}
```
{% else %}
(no diff — reviewers merged the batch unchanged)
{% endif %}

Write the grounded `summary` now — what review changed, citing the affected
standard name(s), with the why from the conversation where it is stated.
