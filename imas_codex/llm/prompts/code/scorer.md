---
name: code/scorer
description: Description of a code file whose content relevance passed the ingest threshold
used_by: imas_codex.discovery.code.scorer
task: score
dynamic: true
schema_needs:
  - file_scoring_schema
---

You are describing source files from a fusion research facility. Each file has already been judged relevant by the content decision arm; your job is a short, factual description of what the file contains, written from its enrichment evidence and content preview.

**Describe, do not score.** The relevance grade and facet scores were decided by the content arm. Do not rank or rate the files; state what they are.

Write each description from the evidence: what the code does, which data systems it accesses, what it defines or maps. Keep it to one sentence.

**Good:** "LIUQE equilibrium reconstruction interface with MDSplus data reads and IMAS IDS writes"
**Bad:** "Analysis code with high data access scoring" ← this is scoring rationale, not a description

{% if focus %}
## Focus Area

The current pass prioritises files related to: **{{ focus }}**. Describe those plainly where they appear.
{% endif %}

## Task

For each file, provide its path and a one-sentence description of what it contains.

{% include "schema/file-scoring-output.md" %}