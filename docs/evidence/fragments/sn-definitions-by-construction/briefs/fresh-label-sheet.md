# Brief: a fresh blind sheet of published names for two independent readers

The name-against-definition check has been scored three times on the same held-out names, so a successor check needs names nobody has scored. You draw them and prepare the readers' material. You label nothing.

## Inputs
- Published names: the catalog checkout at `/home/ITER/mcintos/Code/imas-standard-names-catalog/standard_names/*.yml` (read-only; one file per physics domain). Use each entry's name, unit, one-line description and full documentation as published.
- Names already labelled, to exclude: every `name` in `docs/evidence/fragments/sn-name-definition-second-reading/two-reader-labels.json` (381 rows).
- The class keys and their meanings: `classes` in `docs/evidence/fragments/sn-name-definition-second-reading/blind-sheet.json`. Copy them unchanged.
- The reader brief to adapt: `docs/evidence/fragments/sn-name-definition-second-reading/reader-brief-part-1.md`.

## Deliver, under `docs/evidence/fragments/sn-name-definition-fresh-labels/`
1. `sheet-part-1.json`, `sheet-part-2.json`, `sheet-part-3.json`: 120 names each, 360 in all, none among the 381, drawn at random with a recorded seed and spread over the domain files in proportion to their size, with every domain file that has at least five eligible names contributing at least one. Same shape as the earlier blind sheet (`classes`, `items` with `id`, name, unit, description, documentation). Item ids are unique across the three parts.
2. `sheet-draw.json`: the seed, the catalog revision (`git -C <catalog> rev-parse HEAD`), eligible and drawn counts per domain, and the count excluded as already labelled.
3. Six reader briefs, `reader-a-part-<n>.md` and `reader-b-part-<n>.md` for n = 1, 2, 3, adapted from the earlier reader brief: each names its own sheet part and its own output file `labels-<a|b>-part-<n>.json`, keeps the species ruling, the output format and the done-when, and forbids opening `docs/evidence/fragments/sn-definitions-by-construction/`, `docs/evidence/fragments/sn-name-definition-second-reading/`, every other file in the fresh-labels directory except its own sheet part, `docs/plans/`, `~/.cache/`, any other worktree, and any tool that returns review results for standard names. A reader brief does not say there is another reader's file to find.
4. The draw script, `sheet-draw.py`, in the same directory, so the draw can be repeated from the seed.

## Done when
A committed check (its output in your manifest) prints: 360 items, 360 distinct names, 0 names shared with the 381, three parts of 120, class keys equal to the earlier sheet's, and six reader briefs each naming a distinct output file.

## Standing rules
Light work: no graph, no model calls, no job submission, no test run. Export `UV_NO_SYNC=1`; plain `python3` suffices for the draw. Commit locally with a conventional subject and a body, staging explicit paths only; no version, plan or ticket labels in file names or comments. Manifest status is `complete`, `blocked` or `failed`, written before any wait.
