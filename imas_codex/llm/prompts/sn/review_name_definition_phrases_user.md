---
name: sn/review_name_definition_phrases_user
description: Dynamic user-message portion of the phrase extraction (companion to review_name_definition_phrases_system)
used_by: imas_codex.standard_names.review.name_definition_phrases
task: review
dynamic: true
schema_needs: []
---

Extract from the definition text below every phrase that adds a condition to
the quantity. The quantity is called "the quantity" throughout the text; its
standard name is withheld.

Kinds to label each phrase with, one per phrase:

{{ phrase_kinds }}

- **unit**: {{ unit | default('N/A', true) }}

## Definition text

{{ definition }}

## Output

Return a JSON object with a `phrases` array. Each entry carries `kind` (one of
the kinds above), `phrase` (the verbatim quote) and `adds` (a few words saying
what it adds). Return an empty array when the text adds nothing.