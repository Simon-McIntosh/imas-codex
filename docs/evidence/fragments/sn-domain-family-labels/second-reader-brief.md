# Brief: label a blind sheet of standard names with one physics domain each

You are an independent reader. Your labels become the reference a domain
decision is measured against, so they are worth nothing if you have seen what
is stored. Read this brief and the sheet. Read nothing else about these names.

## Input
- `docs/evidence/fragments/sn-domain-family-labels/blind-sheet.json`
  - `domains`: every physics domain the catalog files names under, with its description.
  - `items`: 113 items. A `family` item gives a name pattern (`<X>` stands for a
    species or a vector component), its member count and three example members.
    A `single` item gives one name. Each has a description and a unit.

## What you may not open
- `labels.json` and `labels-notes.md` in this directory, which hold another reader's answers.
- The standard-names catalog repository, the knowledge graph, any MCP tool that
  returns standard names, `docs/evidence/` outside this directory, `docs/plans/`,
  and anything under `~/.cache/`. Each of these shows the stored domain.
- You may read the Data Dictionary documentation and general physics references
  if a name's meaning is unclear to you.

## What to write
`docs/evidence/fragments/sn-domain-family-labels/labels-second-reader.json`: a list with one
entry per item, in sheet order:

```json
{"id": "item-001", "domain": "<one key of domains>", "confidence": "high|medium|low",
 "second_choice": "<a key of domains, or null>", "reason": "<one sentence>"}
```

- Give the domain a physicist looking for the quantity would open first. Judge
  what the quantity is, not which instrument or code might produce it. A family
  gets one domain for all its members.
- Use `general` only when no other domain fits, and say why.
- Write the file incrementally, at least every twenty items.
- Also write `labels-second-reader-notes.md` in the same directory: which domains you found
  hard to tell apart and the rule you applied, in under 300 words.

## Done when
- `labels-second-reader.json` has exactly one entry for each of the 113 ids, every `domain` is
  a key of `domains`, and the file parses.
- Your manifest lists every file you opened and states that you opened none of
  the forbidden sources.

## Standing rules
- This is light work: no test run, no graph, no job submission.
- Commit locally with a conventional subject and a body; stage only the two
  files you wrote. Manifest status is `complete`, `blocked` or `failed`, written
  before any wait.
