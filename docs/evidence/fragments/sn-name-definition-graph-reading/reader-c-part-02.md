# Brief: a blind reading of 63 standard names against their definitions

Read only the sheet assigned below and make your own judgments.

## Input
`docs/evidence/fragments/sn-name-definition-graph-reading/sheet-part-02.json`
- `classes`: the five classes and what each means.
- `items`: the names of this part, each with its unit, one-line description and full documentation.

## A ruling that changes one class
The catalog's rule is that a bare species name counts ions: `neon_density` is
the density of neon ions summed over all charge states, with neutral atoms
excluded. So a definition that says "ions only" under a bare species name
AGREES with its name. A definition that includes neutral atoms under a bare
species name does not. Apply this wherever the class text says otherwise.

## What you may not open
`docs/evidence/fragments/sn-definitions-by-construction/`,
`docs/evidence/fragments/sn-name-definition-second-reading/`,
`docs/evidence/fragments/sn-name-definition-fresh-labels/`,
every file in `docs/evidence/fragments/sn-name-definition-catalog-reading/`, every file in `docs/evidence/fragments/sn-name-definition-graph-reading/` except your own `sheet-part-02.json`, `docs/plans/`,
`~/.cache/`, any other worktree, and any tool that returns review results for
standard names. You may read physics references and the Data Dictionary
documentation. Your output file is for writing your judgments, not an evidence source.

## What to write
`docs/evidence/fragments/sn-name-definition-graph-reading/labels-c-part-02.json`: one
entry per item, in sheet order:

```json
{"id": "item-0001", "class": "<one key of classes>", "confidence": "high|medium|low",
 "evidence": "<the phrase in the definition that decides it, quoted>",
 "reason": "<one sentence>"}
```

Read the name, description, documentation and unit together. Ask: would a
physicist who read only the name be surprised by what the definition says it
is? Write the file incrementally, at least every fifteen items.

## Done when
A one-line python check prints one entry per sheet item, as many distinct ids matching the sheet
and 0 classes outside the sheet's class keys, and your manifest lists every
file you opened and states that none of the forbidden sources was opened.

## Standing rules
Light work: no test run, no graph, no job submission. Commit locally with a
conventional subject and a body, staging only the file you wrote. Manifest
status is `complete`, `blocked` or `failed`, written before any wait.
