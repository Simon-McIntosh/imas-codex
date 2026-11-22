# A skill for finding ITER records and documents with Lucy's direct tools

## Goal

Write a project skill that tells an agent how to find ITER documents and
records through Lucy's direct record tools instead of its chat tool.

## Why

On 2026-11-22 the lead said "the lucy mcp has other tools, investigate them,
we should not always use the chat interface, in fact we should avoid it", and
asked for a skill because agents, the coordinator included, had used the chat
tool by default. A direct search of kind `Document` with a `title contains`
predicate found about 160 signal lists in two pages in seconds; the chat run
was still searching minutes later.

## What the skill must carry

- When to use which tool: `icp_describe_record_schema` first (the kind
  `Document` and its queryable fields and operators), `icp_search_records` for
  typed predicates, `icp_get_records` for known identities,
  `icp_get_document_versions` for versions, `icp_get_record_relationships` and
  the folder tools for containment; `get_document_file` only to inspect one
  file in the conversation; the chat tool only for a question that needs
  documents read and an answer written.
- The traps the tools themselves document: only queryable fields filter;
  projections must use declared property names (`current_version`, not
  `version`); pages have no ordering and are not a census; withheld rows mean
  rows exist that the user may not see; at most two calls in parallel;
  continuation tokens need identical arguments; request a narrow projection
  because a page of a hundred documents with PBS references is large.
- Worked search recipes for this project: signal lists, self-description
  documents, documents naming one PBS node, versions of one document. Use
  invented identities in the examples.
- How this relates to the project's own read-only client: a pipeline fetches
  bytes through `iter_codex/discovery/icp`, read-only, never through the
  conversation tools; the access rule is GET and HEAD only.

## Done when

- The skill file exists under `.claude/skills/` with a frontmatter `name` and a
  `description` that triggers on finding ITER documents, signal lists,
  self-description data, PBS records or IDM identities.
- Every tool name, field name and operator it cites is checked against the
  server's own descriptions (the tool list a session sees); the evidence lists
  each with where it was confirmed.
- It holds no real document identity, title, person name or host name.
- The project's skill index, if there is one, lists it.

Negative control: one deliberately wrong field name in a draft recipe is caught
by the check above and corrected before commit, and the evidence shows it.

## Rules

- Documentation only: no source or test change.
- Commit locally with a conventional subject and a body, and give `commits` as
  a JSON list of full shas. Do not push or merge.

## Scope

Write paths: `.claude/skills/lucy-records`, and
`docs/evidence/fragments/iter-plant-signal-ontology/lucy-record-tools-skill.html`.
